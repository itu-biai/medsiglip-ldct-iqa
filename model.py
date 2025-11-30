import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoModel, SiglipVisionModel

from config import CONFIG


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
            pdiff = preds.unsqueeze(1) - preds.unsqueeze(0)    # ŷ_i - ŷ_j
            tdiff = targets.unsqueeze(1) - targets.unsqueeze(0)  # y_i - y_j
            s_ij = torch.sign(tdiff)                            # {-1,0,1}

            # w_ij = 0 for ties, 1 otherwise
            w_ij = (s_ij != 0).float()

            # logistic loss
            rank_loss = (w_ij * torch.log(1 + torch.exp(-s_ij * pdiff / self.tau))).mean()
            total = total + self.rank_w * rank_loss

        return total


# =========================================
# Model
# =========================================
class PromptedMultiScaleMedSigLIPIQA(nn.Module):
    def __init__(self, model_name, text_inputs=None, film_strength=None, bound=None):
        super().__init__()
        
        # Load backbone
        try:
            self.backbone = AutoModel.from_pretrained(model_name)
            print(f"✅ Model loaded: {model_name}")
        except Exception as e:
            print(f"❌ {model_name} could not be loaded ({e}) — Falling back to SigLIP Vision.")
            self.backbone = SiglipVisionModel.from_pretrained("google/siglip-base-patch16-224")

        # Get hidden size
        if hasattr(self.backbone.config, "hidden_size"):
            h = self.backbone.config.hidden_size
        elif hasattr(self.backbone.config, "vision_config") and hasattr(self.backbone.config.vision_config, "hidden_size"):
            h = self.backbone.config.vision_config.hidden_size
        else:
            h = 1024
        self.hidden_size = h

        # Check if model has text capabilities
        self.has_text = hasattr(self.backbone, "text_model") or hasattr(self.backbone, "get_text_features")
        
        # FiLM strength (use provided value or from config)
        self.film_strength = film_strength if film_strength is not None else CONFIG.get("film_strength", 1.0)
        
        # Text to FiLM parameters
        self.text_to_film = nn.Sequential(
            nn.Linear(h, 2*h), nn.ReLU(),
            nn.Linear(2*h, 2*h)
        )

        # Pooling heads
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.local_pool = nn.AdaptiveAvgPool1d(4)
        self.texture_pool = nn.AdaptiveMaxPool1d(2)

        # Quality heads
        self.g_head = nn.Sequential(
            nn.Linear(h, 256), nn.BatchNorm1d(256), nn.ReLU(), 
            nn.Dropout(0.2), nn.Linear(256, 1)
        )
        self.l_head = nn.Sequential(
            nn.Linear(h*4, 256), nn.BatchNorm1d(256), nn.ReLU(), 
            nn.Dropout(0.2), nn.Linear(256, 1)
        )
        self.t_head = nn.Sequential(
            nn.Linear(h*2, 128), nn.BatchNorm1d(128), nn.ReLU(), 
            nn.Dropout(0.1), nn.Linear(128, 1)
        )
        self.fuse = nn.Sequential(
            nn.Linear(3, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1)
        )

        # Prompt embedding buffer
        self.register_buffer("prompt_embed", torch.zeros(1, h), persistent=False)
        
        # Output bounding
        self.bound_output = bound if bound is not None else CONFIG.get("bound_output_0_4", True)
        self.tau = float(CONFIG.get("sigmoid_tau", 1.0))

        # Initialize prompt embedding if text inputs provided
        if CONFIG.get("use_text_prompt", False) and text_inputs is not None and self.has_text:
            self._init_prompt_embed(text_inputs)

        # Freeze text tower if configured
        if CONFIG.get("text_freeze", True) and self.has_text:
            try:
                for p in self.backbone.text_model.parameters():
                    p.requires_grad = False
            except Exception:
                pass

    @torch.no_grad()
    def _encode_text_once(self, text_inputs):
        dev = next(self.backbone.parameters()).device
        tinp = {k: v.to(dev) for k, v in text_inputs.items()}
        out = None
        try:
            if hasattr(self.backbone, "get_text_features"):
                out = self.backbone.get_text_features(**tinp)
            elif hasattr(self.backbone, "text_model"):
                tout = self.backbone.text_model(**tinp)
                out = tout.pooler_output if hasattr(tout, "pooler_output") and tout.pooler_output is not None \
                    else tout.last_hidden_state[:, 0]
            else:
                tout = self.backbone(**tinp)
                out = tout.last_hidden_state[:, 0]
        except Exception as e:
            print(f"Text encode error, continuing with image-only: {e}")
            out = None
        return out

    def _init_prompt_embed(self, text_inputs):
        self.eval()
        pe = self._encode_text_once(text_inputs)
        if pe is not None:
            pe = F.normalize(pe, dim=-1)
            self.prompt_embed = pe.detach()

    def forward(self, inputs):
        x = inputs["pixel_values"].to(next(self.backbone.parameters()).device)
        
        # Vision forward
        try:
            out = self.backbone.vision_model(pixel_values=x) if hasattr(self.backbone, "vision_model") \
                else self.backbone(pixel_values=x)
            patches = out.last_hidden_state[:, 1:] if hasattr(out, "last_hidden_state") else out[0][:, 1:]
        except Exception as e:
            print(f"Vision forward error: {e}")
            patches = torch.zeros(x.size(0), 196, self.hidden_size, device=x.device)

        # FiLM (Feature-wise Linear Modulation)
        if self.film_strength > 0 and self.prompt_embed is not None and self.prompt_embed.numel() > 0:
            B, P, H = patches.shape
            pe = self.prompt_embed
            if pe.size(0) == 1:
                pe = pe.expand(B, -1)
            film = self.text_to_film(pe)  # [B, 2H]
            gamma, beta = film.chunk(2, dim=-1)
            # Apply affine transformation: x = x * (1 + gamma) + beta
            patches = patches * (1 + self.film_strength * torch.tanh(gamma).unsqueeze(1)) \
                      + self.film_strength * beta.unsqueeze(1)

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
