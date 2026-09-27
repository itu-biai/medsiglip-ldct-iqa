import json
import os
import re
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoModel

from data import GATED_MODEL_HINT, build_text_inputs


# =========================================
# Loss
# =========================================
class RankNetMSELoss(nn.Module):
    def __init__(self, mse_weight=0.5, rank_weight=0.5, temperature=1.0):
        super().__init__()
        self.mse = nn.MSELoss()
        self.mse_w = mse_weight
        self.rank_w = rank_weight
        self.tau = temperature

    def forward(self, preds, targets):
        # --- MSE Loss ---
        mse = self.mse(preds, targets)
        total = self.mse_w * mse

        # --- pairwise logistic ranking loss ---
        if preds.size(0) > 1 and self.rank_w > 0:
            pdiff = preds.unsqueeze(1) - preds.unsqueeze(0)  # ŷ_i - ŷ_j
            tdiff = targets.unsqueeze(1) - targets.unsqueeze(0)  # y_i - y_j
            s_ij = torch.sign(tdiff)  # {-1,0,1}

            # w_ij = 0 for ties, 1 otherwise
            w_ij = (s_ij != 0).float()

            # logistic loss
            rank_loss = (
                w_ij * torch.log(1 + torch.exp(-s_ij * pdiff / self.tau))
            ).mean()
            total = total + self.rank_w * rank_loss

        return total


# =========================================
# Model
# =========================================
class PromptedMultiScaleMedSigLIPIQA(nn.Module):
    # NOTE: attribute names (backbone, text_to_film, g_head, l_head, t_head, fuse)
    # are the state_dict keys of the released checkpoints - do not rename them.
    def __init__(
        self,
        model_name: str,
        text_inputs: Optional[Dict[str, torch.Tensor]] = None,
        film_strength: float = 1.0,
        sigmoid_tau: float = 2.0,
        bound_output: bool = True,
        text_freeze: bool = True,
        freeze_backbone: bool = False,
    ):
        super().__init__()

        # Load backbone
        try:
            self.backbone = AutoModel.from_pretrained(model_name)
        except Exception as e:
            raise RuntimeError(
                f"Could not load backbone '{model_name}'. {GATED_MODEL_HINT}"
            ) from e
        print(f"Model loaded: {model_name}")

        # Get hidden size
        if hasattr(self.backbone.config, "hidden_size"):
            h = self.backbone.config.hidden_size
        elif hasattr(self.backbone.config, "vision_config") and hasattr(
            self.backbone.config.vision_config, "hidden_size"
        ):
            h = self.backbone.config.vision_config.hidden_size
        else:
            h = 1024
        self.hidden_size = h

        # Check if model has text capabilities
        self.has_text = hasattr(self.backbone, "text_model") or hasattr(
            self.backbone, "get_text_features"
        )

        self.film_strength = film_strength
        self.bound_output = bound_output
        self.tau = float(sigmoid_tau)

        # Text to FiLM parameters
        self.text_to_film = nn.Sequential(
            nn.Linear(h, 2 * h), nn.ReLU(), nn.Linear(2 * h, 2 * h)
        )

        # Pooling heads
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.local_pool = nn.AdaptiveAvgPool1d(4)
        self.texture_pool = nn.AdaptiveMaxPool1d(2)

        # Quality heads
        self.g_head = nn.Sequential(
            nn.Linear(h, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 1),
        )
        self.l_head = nn.Sequential(
            nn.Linear(h * 4, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 1),
        )
        self.t_head = nn.Sequential(
            nn.Linear(h * 2, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1),
        )
        self.fuse = nn.Sequential(
            nn.Linear(3, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

        # Prompt embedding buffer (not saved in checkpoints; recomputed from the prompt)
        self.register_buffer("prompt_embed", torch.zeros(1, h), persistent=False)

        # Initialize prompt embedding if text inputs provided
        if text_inputs is not None and self.has_text:
            self._init_prompt_embed(text_inputs)

        # Freeze text tower if configured
        if text_freeze and hasattr(self.backbone, "text_model"):
            for p in self.backbone.text_model.parameters():
                p.requires_grad = False

        # Freeze vision tower if configured (default: fine-tuned end-to-end)
        if freeze_backbone:
            vision = getattr(self.backbone, "vision_model", self.backbone)
            for p in vision.parameters():
                p.requires_grad = False

    @torch.no_grad()
    def _encode_text_once(self, text_inputs):
        dev = next(self.backbone.parameters()).device
        tinp = {k: v.to(dev) for k, v in text_inputs.items()}
        if hasattr(self.backbone, "get_text_features"):
            out = self.backbone.get_text_features(**tinp)
            # Some transformers versions return a model output instead of a tensor
            if not isinstance(out, torch.Tensor):
                out = (
                    out.pooler_output
                    if getattr(out, "pooler_output", None) is not None
                    else out.last_hidden_state[:, 0]
                )
        else:
            tout = self.backbone.text_model(**tinp)
            out = (
                tout.pooler_output
                if getattr(tout, "pooler_output", None) is not None
                else tout.last_hidden_state[:, 0]
            )
        return out

    def _init_prompt_embed(self, text_inputs):
        self.eval()
        pe = self._encode_text_once(text_inputs)
        self.prompt_embed = F.normalize(pe, dim=-1).detach()

    def forward(self, inputs):
        x = inputs["pixel_values"].to(next(self.backbone.parameters()).device)

        # Vision forward
        out = (
            self.backbone.vision_model(pixel_values=x)
            if hasattr(self.backbone, "vision_model")
            else self.backbone(pixel_values=x)
        )
        # SigLIP has no CLS token, so [:, 1:] drops the first patch token
        # (1024 -> 1023 tokens at 448px). Kept as-is to match the reported runs.
        patches = (
            out.last_hidden_state[:, 1:]
            if hasattr(out, "last_hidden_state")
            else out[0][:, 1:]
        )

        # FiLM (Feature-wise Linear Modulation)
        if self.film_strength > 0 and self.prompt_embed.numel() > 0:
            B, P, H = patches.shape
            pe = self.prompt_embed
            if pe.size(0) == 1:
                pe = pe.expand(B, -1)
            film = self.text_to_film(pe)  # [B, 2H]
            gamma, beta = film.chunk(2, dim=-1)
            # Apply affine transformation: x = x * (1 + gamma) + beta
            patches = patches * (
                1 + self.film_strength * torch.tanh(gamma).unsqueeze(1)
            ) + self.film_strength * beta.unsqueeze(1)

        # Pooling + heads
        patches = patches.transpose(1, 2)  # [B, H, P]
        g = self.global_pool(patches).squeeze(-1)
        l = self.local_pool(patches).flatten(1)
        t = self.texture_pool(patches).flatten(1)

        gq = self.g_head(g)
        lq = self.l_head(l)
        tq = self.t_head(t)

        cat = torch.cat([gq, lq, tq], dim=1)
        logit = self.fuse(cat).squeeze(-1)

        if self.bound_output:
            return 4.0 * torch.sigmoid(logit / self.tau)
        return logit


# =========================================
# Build / load helpers
# =========================================
def build_model(cfg: Dict[str, Any], processor) -> PromptedMultiScaleMedSigLIPIQA:
    prompt = cfg["text_prompt"] if cfg.get("use_text_prompt", False) else None
    return PromptedMultiScaleMedSigLIPIQA(
        cfg["model_name"],
        text_inputs=build_text_inputs(processor, prompt),
        film_strength=cfg["film_strength"],
        sigmoid_tau=cfg["sigmoid_tau"],
        bound_output=cfg["bound_output_0_4"],
        text_freeze=cfg.get("text_freeze", True),
        freeze_backbone=cfg.get("freeze_backbone", False),
    )


def load_checkpoint(model: nn.Module, ckpt_path: str):
    """Load a state_dict checkpoint and fail if any non-backbone weight is missing."""
    state = torch.load(ckpt_path, map_location="cpu", weights_only=True, mmap=True)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    result = model.load_state_dict(state, strict=False)
    head_missing = [k for k in result.missing_keys if not k.startswith("backbone.")]
    if head_missing:
        raise RuntimeError(
            f"{ckpt_path} is missing head weights (wrong checkpoint?): {head_missing[:10]}"
        )
    if result.missing_keys or result.unexpected_keys:
        print(
            f"Warning: {os.path.basename(ckpt_path)}: {len(result.missing_keys)} missing / "
            f"{len(result.unexpected_keys)} unexpected backbone keys "
            f"(e.g. {(result.missing_keys + result.unexpected_keys)[:3]})"
        )
    return result


CKPT_NAME_RE = re.compile(r"^(?P<exp>.+)_(?:fold(?P<fold>\d+)|best)\.pth$")


def load_run_config(ckpt_dir: str, exp_name: Optional[str]) -> Optional[Dict[str, Any]]:
    """Read {exp_name}_config.json written by train.py next to the checkpoints."""
    if not exp_name:
        return None
    path = os.path.join(ckpt_dir, f"{exp_name}_config.json")
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# Settings that must match between training and inference
RUN_CONFIG_KEYS = (
    "model_name",
    "image_size",
    "use_text_prompt",
    "text_prompt",
    "film_strength",
    "sigmoid_tau",
    "bound_output_0_4",
)


def resolve_inference_config(cfg, ckpt_dir, exp_name, prompt=None, film_strength=None):
    """CLI flags > {exp_name}_config.json next to the checkpoints > config.py."""
    cfg = dict(cfg)
    source = "config.py"
    run_cfg = load_run_config(ckpt_dir, exp_name)
    if run_cfg is not None:
        cfg.update({k: run_cfg[k] for k in RUN_CONFIG_KEYS if k in run_cfg})
        source = f"{exp_name}_config.json"
    if prompt is not None:
        cfg["text_prompt"] = prompt
        source = "--prompt"
    if film_strength is not None:
        cfg["film_strength"] = film_strength
    print(f"Prompt ({source}): {cfg['text_prompt'] if cfg['use_text_prompt'] else '<none>'}")
    return cfg
