import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse
import gc
import json
import random
import shutil
import warnings

import numpy as np
from scipy.stats import pearsonr
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from config import CONFIG
from data import IQADataset, build_processor, build_transforms, load_train_list, set_seed
from model import RankNetMSELoss, build_model

warnings.filterwarnings("ignore")


# =========================================
# CLI
# =========================================
def parse_args():
    p = argparse.ArgumentParser(description="5-fold CV training of the MedSigLIP IQA model.")
    p.add_argument("--json", dest="json_path", help="training annotations (filename -> MOS)")
    p.add_argument("--data-dir", help="training image directory")
    p.add_argument("--checkpoint-dir", help="where checkpoints and summaries are written")
    p.add_argument("--exp-name", help="checkpoint prefix ({exp_name}_fold{k}.pth)")
    p.add_argument("--prompt", dest="text_prompt", help="text prompt for FiLM conditioning")
    p.add_argument("--film-strength", type=float, help="FiLM strength (0 disables FiLM)")
    p.add_argument("--model-name", help="Hugging Face backbone id")
    p.add_argument("--epochs", dest="num_epochs", type=int)
    p.add_argument("--batch-size", type=int)
    p.add_argument("--num-workers", type=int)
    p.add_argument("--seed", type=int)
    p.add_argument("--folds", type=int, nargs="+", help="run only these folds (1-based), e.g. --folds 1 3")
    p.add_argument("--freeze-backbone", action="store_true", default=None, help="freeze the vision encoder")
    return p.parse_args()


def apply_overrides(cfg, args):
    cfg = dict(cfg)
    for k, v in vars(args).items():
        if k != "folds" and v is not None:
            cfg[k] = v
    return cfg


# =========================================
# Training / Evaluation
# =========================================
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
        tot += loss.item()
        n += 1
        preds.extend(out.detach().cpu().numpy().tolist())
        targs.extend(y.detach().cpu().numpy().tolist())
    if len(preds) == 0:
        return float("inf"), float("inf"), float("inf"), 0.0
    mae = mean_absolute_error(targs, preds)
    rmse = np.sqrt(mean_squared_error(targs, preds))
    corr, _ = pearsonr(targs, preds)
    return tot / max(n, 1), mae, rmse, corr


def train_one_epoch(
    model, loader, loss_fn, opt, device, scaler, grad_accum=1, use_amp=True
):
    model.train()
    tot, n = 0.0, 0
    opt.zero_grad(set_to_none=True)
    for step, (inputs, y) in enumerate(tqdm(loader, desc="Training")):
        try:
            x = inputs["pixel_values"].to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
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

            tot += loss.item() * grad_accum
            n += 1
        except torch.cuda.OutOfMemoryError:
            print("OOM — batch skipped")
            opt.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
    return tot / max(n, 1)


# =========================================
# MAIN: Training
# =========================================
def main():
    args = parse_args()
    cfg = apply_overrides(CONFIG, args)
    exp = cfg["exp_name"]
    ckpt_dir = cfg["checkpoint_dir"]
    os.makedirs(ckpt_dir, exist_ok=True)

    set_seed(cfg["seed"])

    train_pool = load_train_list(cfg["json_path"], cfg["data_dir"])
    random.shuffle(train_pool)
    print(f"Training pool: {len(train_pool)}")

    processor = build_processor(cfg["model_name"])
    train_transform, val_transform = build_transforms(cfg["image_size"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pin = device.type == "cuda"
    use_amp = cfg["use_amp"] and device.type == "cuda"
    print(f"Device: {device} | Model: {cfg['model_name']} | Experiment: {exp}")
    print(f"Prompt: {cfg['text_prompt'] if cfg['use_text_prompt'] else '<none>'}")

    # Record which prompt/hyperparameters produced these checkpoints;
    # test.py and predict.py read it back to rebuild the same model.
    with open(os.path.join(ckpt_dir, f"{exp}_config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)

    kf = KFold(n_splits=cfg["num_folds"], shuffle=True, random_state=cfg["seed"])
    idxs = np.arange(len(train_pool))
    patience_limit = cfg.get("early_stop_patience")
    fold_summ = []

    print(f"Starting {cfg['num_folds']}-Fold Cross Validation")
    for fold, (tr_idx, va_idx) in enumerate(kf.split(idxs), start=1):
        if args.folds and fold not in args.folds:
            continue
        print(f"\n========== Fold {fold} ==========")
        tr_split = [train_pool[i] for i in tr_idx]
        va_split = [train_pool[i] for i in va_idx]

        tr_ds = IQADataset(tr_split, processor, cfg["image_size"], transform=train_transform)
        va_ds = IQADataset(va_split, processor, cfg["image_size"], transform=val_transform)

        tr_dl = DataLoader(
            tr_ds,
            batch_size=cfg["batch_size"],
            shuffle=True,
            num_workers=cfg["num_workers"],
            pin_memory=pin,
            persistent_workers=False,
            # BatchNorm cannot train on a single-sample batch
            drop_last=len(tr_ds) % cfg["batch_size"] == 1,
        )
        va_dl = DataLoader(
            va_ds,
            batch_size=cfg["batch_size"],
            shuffle=False,
            num_workers=cfg["num_workers"],
            pin_memory=pin,
            persistent_workers=False,
            drop_last=False,
        )

        model = build_model(cfg, processor).to(device)

        loss_fn = RankNetMSELoss(
            mse_weight=cfg["mse_weight"],
            rank_weight=cfg["rank_weight"],
            temperature=cfg["rank_tau"],
        )
        opt = torch.optim.AdamW(
            model.parameters(),
            lr=cfg["learning_rate"],
            weight_decay=cfg["weight_decay"],
            eps=1e-8,
        )
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=cfg["T_max"], eta_min=cfg["eta_min"]
        )
        scaler = torch.amp.GradScaler(device.type, enabled=use_amp)

        ckpt = os.path.join(ckpt_dir, f"{exp}_fold{fold}.pth")
        best = {"val_loss": float("inf"), "mae": float("inf"), "rmse": float("inf"), "corr": 0.0, "epoch": 0}
        patience = 0

        for epoch in range(1, cfg["num_epochs"] + 1):
            print(f"\nEpoch {epoch}/{cfg['num_epochs']}")

            tr_loss = train_one_epoch(
                model,
                tr_dl,
                loss_fn,
                opt,
                device,
                scaler,
                grad_accum=cfg["grad_accum_steps"],
                use_amp=use_amp,
            )
            va_loss, va_mae, va_rmse, va_corr = evaluate(model, va_dl, loss_fn, device)
            print(
                f"Train {tr_loss:.4f} | Val {va_loss:.4f} | MAE {va_mae:.4f} | RMSE {va_rmse:.4f} | Corr {va_corr:.4f}"
            )

            # Checkpoint = epoch with the lowest validation loss
            if va_loss < best["val_loss"]:
                best = {
                    "val_loss": float(va_loss),
                    "mae": float(va_mae),
                    "rmse": float(va_rmse),
                    "corr": float(va_corr),
                    "epoch": epoch,
                }
                torch.save(model.state_dict(), ckpt)
                print(f"Best fold model saved: {ckpt}")
                patience = 0
            else:
                patience += 1

            sch.step()

            if patience_limit is not None and patience >= patience_limit:
                print(f"Early stopping (epoch {epoch})")
                break

        fold_summ.append({"fold": fold, **best})

        # --- END OF FOLD CLEANUP ---
        del tr_dl, va_dl, tr_ds, va_ds
        del loss_fn, opt, sch, scaler, model

        try:
            tqdm._instances.clear()
        except Exception:
            pass

        for _ in range(2):
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    if not fold_summ:
        raise SystemExit("No folds were run. Check --folds.")

    print(f"\n=== {len(fold_summ)}-Fold Summary (best epoch per fold) ===")
    for m in fold_summ:
        print(
            f"Fold {m['fold']} | Epoch {m['epoch']} | Val {m['val_loss']:.4f} | MAE {m['mae']:.4f} | RMSE {m['rmse']:.4f} | Corr {m['corr']:.4f}"
        )
    mean = {k: float(np.mean([m[k] for m in fold_summ])) for k in ("val_loss", "mae", "rmse", "corr")}
    print(
        f"\nAverage MAE: {mean['mae']:.4f} | RMSE: {mean['rmse']:.4f} | Pearson: {mean['corr']:.4f}"
    )

    # Best fold by cfg["select_metric"] (validation loss by default)
    metric = cfg["select_metric"]
    best_fold = min(fold_summ, key=lambda m: m[metric])["fold"]
    print(f"Best fold: {best_fold} (lowest {metric})")

    best_ckpt = os.path.join(ckpt_dir, f"{exp}_fold{best_fold}.pth")
    final_ckpt = os.path.join(ckpt_dir, f"{exp}_best.pth")
    shutil.copy2(best_ckpt, final_ckpt)
    print(f"Best model saved: {final_ckpt}")

    summary = {
        "exp_name": exp,
        "select_metric": metric,
        "best_fold": best_fold,
        "folds": fold_summ,
        "mean": mean,
    }
    with open(os.path.join(ckpt_dir, f"{exp}_cv_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Training completed.")


if __name__ == "__main__":
    main()
