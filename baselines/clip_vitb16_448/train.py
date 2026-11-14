"""CLIP ViT-B/16 baseline at 448x448: 5-fold cross-validation training.

General-domain counterpart of the BiomedCLIP baseline of the paper (Table 2, row "CLIP / CLIP / 448").
Image tower: OpenAI CLIP ViT-B/16 (timm port), position embeddings resampled to the 28x28 grid; patch embedding,
position embeddings and blocks 1-6 frozen. The frozen CLIP text encoder embeds the prompt, which conditions the
patch tokens through FiLM; three pooling heads and a fusion MLP give the score. Loss: 0.7 RankNet + 0.3 MSE.

Paths: data/train/train.json and data/train/image by default (override with LDCT_TRAIN_JSON, LDCT_TRAIN_DIR);
checkpoints are written to checkpoints/clip_vitb16_448 (override with CLIP_CKPT_DIR).
Extra dependencies: pip install open_clip_torch timm
"""
import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import json, random, warnings
warnings.filterwarnings("ignore")
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd
from PIL import Image
import imageio.v3 as iio

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import pearsonr
from tqdm import tqdm
import torchvision.transforms as T
import gc
import shutil

# =========================================
# CONFIG
# =========================================
CONFIG: Dict[str, Any] = {
    "learning_rate": 1e-5,
    "batch_size": 4,
    "num_epochs": 40,
    "weight_decay": 1e-4,
    "num_workers": 2,
    "seed": 42,
    "image_size": 448,
    "hidden_size": 768,
    "T_max": 40,
    "eta_min": 1e-7,
    "checkpoint_local_dir": os.environ.get("CLIP_CKPT_DIR", "checkpoints/clip_vitb16_448"),
    "checkpoint_drive_dir": os.environ.get("CLIP_CKPT_DIR", "checkpoints/clip_vitb16_448"),
    "use_amp": True,
    "grad_accum_steps": 2,
    "use_text_prompt": True,
    "text_prompt": (
        "Rate this low-dose CT (MOS 0-4): 0 Nondiagnostic\u2014desired features not shown; "
        "1 Poor\u2014diagnostic interpretation impossible; 2 Fair\u2014limited interpretation; "
        "3 Good\u2014diagnostic; 4 Excellent\u2014anatomy highly visible. Return only one number 0-4."
    ),
    "film_strength": 1.0,
    "sigmoid_tau": 2.0,
}

BIOMEDCLIP_MODEL_ID = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
BIOMED_MEAN = [0.48145466, 0.4578275, 0.40821073]
BIOMED_STD  = [0.26862954, 0.26130258, 0.27577711]

# =========================================
# TRANSFORMS
# =========================================
train_transform = T.Compose([
    T.Resize((CONFIG["image_size"], CONFIG["image_size"])),
    T.RandomHorizontalFlip(p=0.5),
    T.RandomRotation(degrees=10, fill=0),
    T.ColorJitter(brightness=0.1, contrast=0.1),
    T.ToTensor(),
    T.Normalize(mean=BIOMED_MEAN, std=BIOMED_STD),
])

val_transform = T.Compose([
    T.Resize((CONFIG["image_size"], CONFIG["image_size"])),
    T.ToTensor(),
    T.Normalize(mean=BIOMED_MEAN, std=BIOMED_STD),
])

# =========================================
# UTILITIES
# =========================================
def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def safe_float(x):
    try: return float(x)
    except: return np.nan

def load_master_list(json_path: str, data_dir: str) -> List[Tuple[str, float]]:
    with open(json_path, "r") as f:
        data = json.load(f)
    df = pd.DataFrame([{"Filename": k, "MOS": safe_float(v)} for k, v in data.items()])
    df = df.dropna(subset=["MOS"])
    master, missing = [], 0
    for _, row in df.iterrows():
        p = os.path.join(data_dir, row["Filename"])
        if os.path.exists(p):
            master.append((p, row["MOS"]))
        else:
            missing += 1
    print(f"Total images: {len(master)} | Missing: {missing}")
    return master

# =========================================
# DATASET
# =========================================
class BiomedCLIPDataset(Dataset):
    def __init__(self, data_list, image_size=224, transform=None):
        self.data_list = data_list
        self.image_size = image_size
        self.transform = transform if transform is not None else val_transform

    def __len__(self): return len(self.data_list)

    def __getitem__(self, idx):
        image_path, score = self.data_list[idx]
        try:
            img = iio.imread(image_path)
        except Exception as e:
            print(f"Read error {image_path}: {e}")
            img = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)

        if img.ndim == 3 and img.shape[-1] == 1: img = img.squeeze(-1)
        if img.ndim == 2: img = np.stack([img, img, img], axis=-1)
        if img.ndim == 3 and img.shape[-1] > 3: img = img[..., :3]
        img = (img * 255).astype(np.uint8) if img.max() <= 1.0 else img.astype(np.uint8)

        pil = Image.fromarray(img)
        pixel_values = self.transform(pil)
        return {"pixel_values": pixel_values}, torch.tensor(float(score), dtype=torch.float32)

# =========================================
# LOSS
# =========================================
class RankNetMSELoss(nn.Module):
    def __init__(self, mse_weight=0.5, rank_weight=0.5, temperature=1.0):
        super().__init__()
        self.mse = nn.MSELoss()
        self.mse_w = mse_weight
        self.rank_w = rank_weight
        self.tau = temperature

    def forward(self, preds, targets):
        total = self.mse_w * self.mse(preds, targets)
        if preds.size(0) > 1 and self.rank_w > 0:
            pdiff = preds.unsqueeze(1) - preds.unsqueeze(0)
            tdiff = targets.unsqueeze(1) - targets.unsqueeze(0)
            s_ij  = torch.sign(tdiff)
            w_ij  = (s_ij != 0).float()
            rank_loss = (w_ij * torch.log(1 + torch.exp(-s_ij * pdiff / self.tau))).mean()
            total = total + self.rank_w * rank_loss
        return total

# =========================================
# MODEL: CLIP ViT-B/16 + FiLM (class names kept from the BiomedCLIP baseline)
# =========================================
class BiomedCLIPFiLMIQA(nn.Module):
    """CLIP ViT-B/16 vision encoder (blocks 7-12 trainable) + FiLM conditioning + 3-head multi-scale IQA."""

    def __init__(self, film_strength=1.0, sigmoid_tau=2.0):
        super().__init__()
        import open_clip
        clip_model, _, _ = open_clip.create_model_and_transforms("ViT-B-16", pretrained="openai")
        tokenizer = open_clip.get_tokenizer("ViT-B-16")

        # Vision encoder — top 6/12 blocks trainable, lower blocks frozen
        import timm
        self.visual = nn.Module()
        self.visual.trunk = timm.create_model("vit_base_patch16_clip_224.openai", pretrained=True, num_classes=0, img_size=448)
        for p in self.visual.trunk.patch_embed.parameters():
            p.requires_grad = False
        if hasattr(self.visual.trunk, "pos_embed"):
            self.visual.trunk.pos_embed.requires_grad = False
        for i, blk in enumerate(self.visual.trunk.blocks):
            if i < 6:
                for p in blk.parameters():
                    p.requires_grad = False
        h = 768   # BiomedCLIP ViT-B/16 trunk embed_dim
        self.hidden_size = h
        self.film_strength = film_strength
        self.tau = sigmoid_tau

        # Text embed from encode_text is 512-dim (CLIP projection space)
        text_dim = 512
        self.text_to_film = nn.Sequential(
            nn.Linear(text_dim, 2 * h), nn.ReLU(),
            nn.Linear(2 * h, 2 * h)
        )

        # 3-head multi-scale pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.local_pool  = nn.AdaptiveAvgPool1d(4)
        self.text_pool   = nn.AdaptiveMaxPool1d(2)

        self.g_head = nn.Sequential(nn.Linear(h,       256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.2), nn.Linear(256, 1))
        self.l_head = nn.Sequential(nn.Linear(h * 4,   256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.2), nn.Linear(256, 1))
        self.t_head = nn.Sequential(nn.Linear(h * 2,   128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.1), nn.Linear(128, 1))
        self.fuse   = nn.Sequential(nn.Linear(3, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.1),
                                    nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))

        self.prompt_embed = nn.Parameter(torch.zeros(1, text_dim))

        # Encode prompt once; clip_model is a local variable freed after __init__
        self._init_prompt_embed(clip_model, tokenizer, CONFIG["text_prompt"])
        del clip_model, tokenizer

    @torch.no_grad()
    def _init_prompt_embed(self, clip_model, tokenizer, prompt: str):
        dev = next(self.visual.parameters()).device
        tokens = tokenizer([prompt]).to(dev)
        pe = clip_model.encode_text(tokens)
        pe = F.normalize(pe, dim=-1)
        self.prompt_embed.data.copy_(pe.detach())

    def forward(self, inputs):
        x = inputs["pixel_values"].to(next(self.visual.parameters()).device)

        # ViT-B/16 @ 224px: trunk.forward_features -> [B, 197, 768], drop CLS
        all_tokens = self.visual.trunk.forward_features(x)
        patches = all_tokens[:, 1:]   # [B, 196, 768]

        # FiLM conditioning
        B, P, H = patches.shape
        pe = self.prompt_embed.expand(B, -1)
        gamma, beta = self.text_to_film(pe).chunk(2, dim=-1)
        patches = patches * (1 + self.film_strength * torch.tanh(gamma).unsqueeze(1)) \
                + self.film_strength * beta.unsqueeze(1)

        # 3-head pooling
        patches = patches.transpose(1, 2)           # [B, H, P]
        g = self.global_pool(patches).squeeze(-1)   # [B, H]
        l = self.local_pool(patches).flatten(1)     # [B, 4H]
        t = self.text_pool(patches).flatten(1)      # [B, 2H]

        gq = self.g_head(g)
        lq = self.l_head(l)
        tq = self.t_head(t)

        logit = self.fuse(torch.cat([gq, lq, tq], dim=1)).squeeze(-1)
        return 4.0 * torch.sigmoid(logit / self.tau)


# =========================================
# TRAINING / EVALUATION
# =========================================
@torch.no_grad()
def run_val(model, loader, loss_fn, device):
    model.train(False)
    tot, n = 0.0, 0
    preds, targs = [], []
    for inputs, y in loader:
        x = {"pixel_values": inputs["pixel_values"].to(device)}
        y = y.to(device)
        out = model(x)
        loss = loss_fn(out, y)
        tot += loss.item(); n += 1
        preds.extend(out.detach().cpu().numpy().tolist())
        targs.extend(y.detach().cpu().numpy().tolist())
    if len(preds) == 0:
        return float("inf"), float("inf"), float("inf"), 0.0
    mae  = mean_absolute_error(targs, preds)
    rmse = np.sqrt(mean_squared_error(targs, preds))
    corr, _ = pearsonr(targs, preds)
    return tot / max(n, 1), mae, rmse, corr


def train_one_epoch(model, loader, loss_fn, opt, device, scaler, grad_accum=1, use_amp=True):
    model.train()
    tot, n = 0.0, 0
    opt.zero_grad(set_to_none=True)
    for step, (inputs, y) in enumerate(tqdm(loader, desc="Training")):
        try:
            x = inputs["pixel_values"].to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                out  = model({"pixel_values": x})
                loss = loss_fn(out, y) / grad_accum
            scaler.scale(loss).backward()
            last = (step + 1) % grad_accum == 0 or (step + 1) == len(loader)
            if last:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
            tot += loss.item() * grad_accum; n += 1
        except torch.cuda.OutOfMemoryError:
            print("OOM — batch skipped")
            opt.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"Batch error: {e}")
            opt.zero_grad(set_to_none=True)
    return tot / max(n, 1)


# =========================================
# MAIN
# =========================================
def main():
    set_seed(CONFIG["seed"])

    json_path = os.environ.get("LDCT_TRAIN_JSON", "data/train/train.json")
    data_dir  = os.environ.get("LDCT_TRAIN_DIR", "data/train/image")

    master = load_master_list(json_path, data_dir)
    random.shuffle(master)
    train_pool = master
    print(f"Training pool: {len(train_pool)}")
    print(f"CLIP ViT-B/16 @448 + FiLM | h=768 | epochs={CONFIG['num_epochs']}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    kf   = KFold(n_splits=5, shuffle=True, random_state=CONFIG["seed"])
    idxs = np.arange(len(train_pool))
    pin  = (device.type == "cuda")

    best_overall = float("inf")
    best_fold    = None
    fold_summ    = []

    for fold, (tr_idx, va_idx) in enumerate(kf.split(idxs), start=1):
        print(f"\n========== Fold {fold} ==========")
        tr_split = [train_pool[i] for i in tr_idx]
        va_split = [train_pool[i] for i in va_idx]

        tr_ds = BiomedCLIPDataset(tr_split, CONFIG["image_size"], transform=train_transform)
        va_ds = BiomedCLIPDataset(va_split, CONFIG["image_size"], transform=val_transform)
        tr_dl = DataLoader(tr_ds, batch_size=CONFIG["batch_size"], shuffle=True,
                           num_workers=CONFIG["num_workers"], pin_memory=pin,
                           persistent_workers=False, drop_last=False)
        va_dl = DataLoader(va_ds, batch_size=CONFIG["batch_size"], shuffle=False,
                           num_workers=CONFIG["num_workers"], pin_memory=pin,
                           persistent_workers=False, drop_last=False)

        model   = BiomedCLIPFiLMIQA(film_strength=CONFIG["film_strength"],
                                     sigmoid_tau=CONFIG["sigmoid_tau"]).to(device)
        loss_fn = RankNetMSELoss(mse_weight=0.3, rank_weight=0.7, temperature=0.5)
        opt     = torch.optim.AdamW(model.parameters(), lr=CONFIG["learning_rate"],
                                    weight_decay=CONFIG["weight_decay"], eps=1e-8)
        sch     = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=CONFIG["T_max"],
                                                              eta_min=CONFIG["eta_min"])
        scaler  = torch.cuda.amp.GradScaler(enabled=(CONFIG["use_amp"] and device.type == "cuda"))

        best_val, patience, max_patience = float("inf"), 0, 10

        for epoch in range(1, CONFIG["num_epochs"] + 1):
            print(f"\nEpoch {epoch}/{CONFIG['num_epochs']}")
            tr_loss = train_one_epoch(model, tr_dl, loss_fn, opt, device, scaler,
                                      grad_accum=CONFIG["grad_accum_steps"],
                                      use_amp=(CONFIG["use_amp"] and device.type == "cuda"))
            va_loss, va_mae, va_rmse, va_corr = run_val(model, va_dl, loss_fn, device)
            print(f"Train {tr_loss:.4f} | Val {va_loss:.4f} | MAE {va_mae:.4f} | RMSE {va_rmse:.4f} | Corr {va_corr:.4f}")

            if va_loss < best_val:
                best_val = va_loss
                local_dir = CONFIG["checkpoint_local_dir"]
                os.makedirs(local_dir, exist_ok=True)
                local_ckpt = os.path.join(local_dir, f"clip_vitb16_448_fold{fold}.pth")
                torch.save(model.state_dict(), local_ckpt)
                print(f"Local checkpoint saved: {local_ckpt}")
                patience = 0
            else:
                patience += 1

            try: sch.step()
            except Exception as e: print(f"Scheduler error: {e}")

            if patience >= max_patience:
                print(f"Early stopping (epoch {epoch})")
                break

        # Copy best fold checkpoint to Drive once at end of fold
        local_ckpt = os.path.join(CONFIG["checkpoint_local_dir"], f"clip_vitb16_448_fold{fold}.pth")
        if os.path.exists(local_ckpt) and CONFIG["checkpoint_drive_dir"] != CONFIG["checkpoint_local_dir"]:
            try:
                drive_dir = CONFIG["checkpoint_drive_dir"]
                os.makedirs(drive_dir, exist_ok=True)
                shutil.copy2(local_ckpt, os.path.join(drive_dir, f"clip_vitb16_448_fold{fold}.pth"))
                print(f"Fold {fold} best checkpoint copied to Drive: {drive_dir}")
            except Exception as e:
                print(f"Drive copy failed for fold {fold}: {e}")

        fold_summ.append({"fold": fold, "val_loss": best_val,
                          "mae": va_mae, "rmse": va_rmse, "corr": va_corr})
        if best_val < best_overall:
            best_overall = best_val; best_fold = fold

        del tr_dl, va_dl, tr_ds, va_ds, loss_fn, opt, sch, scaler, model
        try: tqdm._instances.clear()
        except Exception: pass
        for _ in range(2):
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache(); torch.cuda.synchronize()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    print("\n=== 5-Fold Summary ===")
    for m in fold_summ:
        print(f"Fold {m['fold']} | Val {m['val_loss']:.4f} | MAE {m['mae']:.4f} | RMSE {m['rmse']:.4f} | Corr {m['corr']:.4f}")
    print(f"\nAverage MAE: {np.mean([m['mae'] for m in fold_summ]):.4f} | "
          f"RMSE: {np.mean([m['rmse'] for m in fold_summ]):.4f} | "
          f"Pearson: {np.mean([m['corr'] for m in fold_summ]):.4f}")
    print(f"Best fold: {best_fold} (Val {best_overall:.4f})")
    if best_fold is not None:
        best_local = os.path.join(CONFIG["checkpoint_local_dir"], f"clip_vitb16_448_fold{best_fold}.pth")
        if os.path.exists(best_local):
            try:
                drive_dir = CONFIG["checkpoint_drive_dir"]
                os.makedirs(drive_dir, exist_ok=True)
                shutil.copy2(best_local, os.path.join(drive_dir, "clip_vitb16_448_best.pth"))
                print(f"Overall best (fold {best_fold}) saved as clip_vitb16_448_best.pth")
            except Exception as e:
                print(f"Best checkpoint Drive copy failed: {e}")
    print("Training complete.")


if __name__ == "__main__":
    main()
