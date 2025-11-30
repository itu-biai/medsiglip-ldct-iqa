import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import json
import random
import warnings
import gc
import shutil
from typing import List, Tuple

import numpy as np
import pandas as pd
import imageio.v3 as iio
from PIL import Image
from scipy.stats import pearsonr
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

from transformers import AutoProcessor, SiglipProcessor

from config import CONFIG
from model import PromptedMultiScaleMedSigLIPIQA, RankNetMSELoss

warnings.filterwarnings("ignore")

# =========================================
# AUGMENTATION
# =========================================
train_transform = T.Compose([
    T.Resize((CONFIG["image_size"], CONFIG["image_size"])),
    T.RandomHorizontalFlip(p=0.5),
    T.RandomRotation(degrees=10, fill=0), 
    T.ColorJitter(brightness=0.1, contrast=0.1),
])

val_test_transform = T.Compose([
    T.Resize((CONFIG["image_size"], CONFIG["image_size"])),
])

# =========================================
# Helpers
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
    master = []
    missing = 0
    for _, row in df.iterrows():
        p = os.path.join(data_dir, row["Filename"])
        if os.path.exists(p):
            master.append((p, row["MOS"]))
        else:
            missing += 1
    print(f"Total images: {len(master)} | Missing: {missing}")
    return master

def build_processor(model_name: str):
    try:
        proc = AutoProcessor.from_pretrained(model_name)
        print(f"✅ Processor loaded: {model_name}")
        return proc
    except Exception as e:
        print(f"❌ Processor could not be loaded ({e}) — Falling back to SigLIP processor.")
        return SiglipProcessor.from_pretrained("google/siglip-base-patch16-224")

# =========================================
# Dataset 
# =========================================
class MedSigLIPIQADataset(Dataset):
    def __init__(self, data_list, processor, image_size=448, transform=None):
        self.data_list = data_list
        self.processor = processor
        self.image_size = image_size
        self.transform = transform  

    def __len__(self): return len(self.data_list)

    def __getitem__(self, idx):
        image_path, score = self.data_list[idx]
        try:
            img = iio.imread(image_path)
        except Exception as e:
            print(f"Read error {image_path}: {e}")
            img = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)

        # channel & range
        if img.ndim == 3 and img.shape[-1] == 1: img = img.squeeze(-1)
        if img.ndim == 2: img = np.stack([img, img, img], axis=-1)
        if img.ndim == 3 and img.shape[-1] > 3: img = img[..., :3]
        # Normalize float [0,1] to uint8 [0,255] if necessary
        img = (img * 255).astype(np.uint8) if img.max() <= 1.0 else img.astype(np.uint8)

        pil = Image.fromarray(img)

        # --- AUGMENTATION / RESIZE ---
        if self.transform is not None:
            pil = self.transform(pil)
        else:
            pil = pil.resize((self.image_size, self.image_size), Image.Resampling.LANCZOS)

        # Tensor conversion and normalization with AutoProcessor 
        try:
            inputs = self.processor(images=pil, return_tensors="pt",
                                    do_resize=False, do_center_crop=False)
        except TypeError:
            inputs = self.processor(images=pil, return_tensors="pt")
        except Exception as e:
            print(f"Processor error: {e} — manual normalization.")
            tr = T.Compose([
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
            ])
            inputs = {"pixel_values": tr(pil).unsqueeze(0)}

        for k in list(inputs.keys()):
            v = inputs[k]
            if hasattr(v, "dim") and v.dim() > 1:
                inputs[k] = v.squeeze(0)

        return inputs, torch.tensor(float(score), dtype=torch.float32)
    



# =========================================
# Training / Evaluation
# =========================================
def create_model(processor, text_prompt: str):
    text_inputs = None
    if CONFIG.get("use_text_prompt", False) and text_prompt:
        try:
            ti = processor(text=text_prompt, return_tensors="pt")
            text_inputs = {k: v for k, v in ti.items() if isinstance(v, torch.Tensor)}
        except Exception:
            text_inputs = None
    return PromptedMultiScaleMedSigLIPIQA(CONFIG["medsiglip_model"], text_inputs=text_inputs)

@torch.no_grad()
def evaluate(model, loader, loss_fn, device):
    model.eval()
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
    mae = mean_absolute_error(targs, preds)
    rmse = np.sqrt(mean_squared_error(targs, preds))
    corr, _ = pearsonr(targs, preds)
    return tot/max(n,1), mae, rmse, corr

def train_one_epoch(model, loader, loss_fn, opt, device, scaler, grad_accum=1, use_amp=True):
    model.train()
    tot, n = 0.0, 0
    opt.zero_grad(set_to_none=True)
    for step, (inputs, y) in enumerate(tqdm(loader, desc="Training")):
        try:
            x = inputs["pixel_values"].to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                out = model({"pixel_values": x})
                loss = loss_fn(out, y) / grad_accum
            scaler.scale(loss).backward()

            # Gradient Accumulation: Update weights only every 'grad_accum' steps
            last = (step + 1) % grad_accum == 0 or (step + 1) == len(loader)
            if last:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)

            tot += loss.item()*grad_accum
            n += 1
        except torch.cuda.OutOfMemoryError:
            print("OOM — batch skipped")
            opt.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"Batch error: {e}")
            opt.zero_grad(set_to_none=True)
    return tot/max(n,1)

# =========================================
# Inference helpers
# =========================================
def preprocess_for_infer(image_path: str, processor, image_size=448):
    img = iio.imread(image_path)
    if img.ndim == 3 and img.shape[-1] == 1: img = img.squeeze(-1)
    if img.ndim == 2: img = np.stack([img, img, img], axis=-1)
    if img.ndim == 3 and img.shape[-1] > 3: img = img[..., :3]
    img = (img * 255).astype(np.uint8) if img.max() <= 1.0 else img.astype(np.uint8)
    pil = Image.fromarray(img).resize((image_size, image_size), Image.Resampling.LANCZOS)
    try:
        inputs = processor(images=pil, return_tensors="pt", do_resize=False, do_center_crop=False)
    except TypeError:
        inputs = processor(images=pil, return_tensors="pt")
    return {"pixel_values": inputs["pixel_values"]}

@torch.no_grad()
def predict(image_paths: List[str], ckpt_path: str,
            model_name="google/medsiglip-448", prompt: str=None, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    processor = build_processor(model_name)
    
    # model with prompt
    text_inputs = None
    if prompt:
        try:
            ti = processor(text=prompt, return_tensors="pt")
            text_inputs = {k: v for k, v in ti.items() if isinstance(v, torch.Tensor)}
        except: pass
    model = PromptedMultiScaleMedSigLIPIQA(model_name, text_inputs=text_inputs).to(device)

    sd = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(sd, strict=False)
    model.eval()

    out = {}
    for p in image_paths:
        inp = preprocess_for_infer(p, processor, image_size=CONFIG["image_size"])
        inp["pixel_values"] = inp["pixel_values"].to(device)
        y = model(inp).item()
        out[p] = y
    return out




def get_film_strength(m):
    target = m.module if hasattr(m, "module") else m
    return getattr(target, "film_strength", "-")


# =========================================
# MAIN: Training
# =========================================
def main():
    set_seed(CONFIG["seed"])

    json_path = CONFIG["json_path"]
    data_dir = CONFIG["data_dir"]

    master = load_master_list(json_path, data_dir)
    random.shuffle(master)
    train_pool = master 
    print(f"Training pool: {len(train_pool)}")

    processor = build_processor(CONFIG["medsiglip_model"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | Model: {CONFIG['medsiglip_model']}")

    kf = KFold(n_splits=5, shuffle=True, random_state=CONFIG["seed"])
    idxs = np.arange(len(train_pool))

    best_overall_mae = float("inf")
    best_fold = None
    fold_summ = []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pin = (device.type == "cuda")


    print("🚀 Starting 5-Fold Cross Validation")
    for fold, (tr_idx, va_idx) in enumerate(kf.split(idxs), start=1):
        print(f"\n========== Fold {fold} ==========")
        tr_split = [train_pool[i] for i in tr_idx]
        va_split = [train_pool[i] for i in va_idx]

        
        tr_ds = MedSigLIPIQADataset(tr_split, processor, CONFIG["image_size"], transform=train_transform)
        va_ds = MedSigLIPIQADataset(va_split, processor, CONFIG["image_size"], transform=val_test_transform)

        tr_dl = DataLoader(
            tr_ds,
            batch_size=CONFIG["batch_size"],
            shuffle=True,
            num_workers=CONFIG["num_workers"],
            pin_memory=pin,
            persistent_workers=False,  
            drop_last=False
        )
        va_dl = DataLoader(
            va_ds,
            batch_size=CONFIG["batch_size"],
            shuffle=False,
            num_workers=CONFIG["num_workers"],
            pin_memory=pin,
            persistent_workers=False,   
            drop_last=False
        )

        model = create_model(processor, CONFIG.get("text_prompt", "")).to(device)

        loss_fn = RankNetMSELoss(mse_weight=0.0, rank_weight=1.0, temperature=0.5)

        opt = torch.optim.AdamW(model.parameters(), lr=CONFIG["learning_rate"],
                                weight_decay=CONFIG["weight_decay"], eps=1e-8)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=CONFIG["T_max"], eta_min=CONFIG["eta_min"])
        scaler = torch.cuda.amp.GradScaler(enabled=(CONFIG["use_amp"] and device.type == "cuda"))

        best_val = float("inf")
        best_mae = float("inf")
        best_rmse = float("inf")
        best_corr = 0.0

        for epoch in range(1, CONFIG["num_epochs"]+1):
            
            print(f"\nEpoch {epoch}/{CONFIG['num_epochs']} (tau={loss_fn.tau:.3f}, film={get_film_strength(model)})")

            tr_loss = train_one_epoch(model, tr_dl, loss_fn, opt, device, scaler,
                                      grad_accum=CONFIG["grad_accum_steps"],
                                      use_amp=(CONFIG["use_amp"] and device.type=='cuda'))
            va_loss, va_mae, va_rmse, va_corr = evaluate(model, va_dl, loss_fn, device)
            print(f"Train {tr_loss:.4f} | Val {va_loss:.4f} | MAE {va_mae:.4f} | RMSE {va_rmse:.4f} | Corr {va_corr:.4f}")

            if va_loss < best_val:
                best_val = va_loss
                best_mae = va_mae
                best_rmse = va_rmse
                best_corr = va_corr
                ckpt = f"{CONFIG['checkpoint_path']}_fold{fold}.pth"
                torch.save(model.state_dict(), ckpt)
                print(f"✅ Best fold model saved: {ckpt}")

            try:
                sch.step()
            except Exception as e:
                print(f"Scheduler error: {e}")


        fold_summ.append({"fold": fold, "val_loss": best_val,
                          "mae": best_mae, "rmse": best_rmse, "corr": best_corr})
        if best_mae < best_overall_mae:
            best_overall_mae = best_mae
            best_fold = fold

                # --- END OF FOLD CLEANUP ---
        del tr_dl, va_dl, tr_ds, va_ds
        del loss_fn, opt, sch, scaler, model

        try: tqdm._instances.clear()
        except Exception: pass

        for _ in range(2):
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()


    print("\n=== 5-Fold Summary ===")
    for m in fold_summ:
        print(f"Fold {m['fold']} | Val {m['val_loss']:.4f} | MAE {m['mae']:.4f} | RMSE {m['rmse']:.4f} | Corr {m['corr']:.4f}")
    avg_mae  = np.mean([m["mae"] for m in fold_summ])
    avg_rmse = np.mean([m["rmse"] for m in fold_summ])
    avg_corr = np.mean([m["corr"] for m in fold_summ])
    print(f"\nAverage MAE: {avg_mae:.4f} | RMSE: {avg_rmse:.4f} | Pearson: {avg_corr:.4f}")
    print(f"Best fold: {best_fold} (MAE {best_overall_mae:.4f})")
    
    # save best model
    best_ckpt = f"{CONFIG['checkpoint_path']}_fold{best_fold}.pth"
    final_ckpt = f"{CONFIG['checkpoint_path']}_best.pth"
    shutil.copy(best_ckpt, final_ckpt)
    print(f"🏆 Best model saved: {final_ckpt}")
    
    print("🏥 Training completed.")

if __name__ == "__main__":
    main()