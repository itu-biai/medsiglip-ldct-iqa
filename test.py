import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse
import csv
import glob
import json
import warnings

import numpy as np
from scipy.stats import pearsonr, spearmanr, kendalltau
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from config import CONFIG
from data import IQADataset, build_processor, build_transforms, load_test_list
from model import CKPT_NAME_RE, build_model, load_checkpoint, resolve_inference_config

warnings.filterwarnings("ignore")


# -------------------- CLI --------------------
def parse_args():
    p = argparse.ArgumentParser(
        description="Evaluate one checkpoint, or every {exp}_fold*.pth in a directory (mean ± std)."
    )
    src = p.add_mutually_exclusive_group()
    src.add_argument("--ckpt", help="a single checkpoint (.pth)")
    src.add_argument("--ckpt-dir", help="directory with {exp_name}_fold{k}.pth (default: config checkpoint_dir)")
    p.add_argument("--exp-name", help="checkpoint prefix (default: config exp_name, or parsed from --ckpt)")
    p.add_argument("--prompt", help="override the text prompt (default: from {exp_name}_config.json, else config.py)")
    p.add_argument("--film-strength", type=float, help="override FiLM strength")
    p.add_argument("--test-json", help="test annotations (filename -> MOS)")
    p.add_argument("--test-dir", help="test image directory")
    p.add_argument("--out", help="results JSON (default in --ckpt-dir mode: <dir>/test_results.json)")
    p.add_argument("--save-preds", help="write per-image predictions to this CSV")
    p.add_argument("--batch-size", type=int, default=CONFIG["batch_size"])
    p.add_argument("--num-workers", type=int, default=CONFIG["num_workers"])
    return p.parse_args()


def find_fold_checkpoints(ckpt_dir, exp_name):
    ckpts = []
    for path in glob.glob(os.path.join(ckpt_dir, f"{glob.escape(exp_name)}_fold*.pth")):
        m = CKPT_NAME_RE.match(os.path.basename(path))
        if m and m.group("exp") == exp_name and m.group("fold"):
            ckpts.append((int(m.group("fold")), path))
    return sorted(ckpts)


# -------------------- METRICS --------------------
def compute_metrics(preds, tgts):
    preds = np.asarray(preds, dtype=np.float64)
    tgts = np.asarray(tgts, dtype=np.float64)
    res = {
        "plcc": float(abs(pearsonr(preds, tgts)[0])),
        "srocc": float(abs(spearmanr(preds, tgts)[0])),
        "krocc": float(abs(kendalltau(preds, tgts)[0])),
    }
    res["overall"] = res["plcc"] + res["srocc"] + res["krocc"]
    return res


@torch.no_grad()
def predict_loader(model, loader, device):
    model.eval()
    preds, tgts, paths = [], [], []
    for inputs, y, p in tqdm(loader, desc="Test", leave=False):
        pv = inputs["pixel_values"].to(device, non_blocking=True)
        yhat = model({"pixel_values": pv})
        preds.extend(yhat.detach().cpu().numpy().tolist())
        tgts.extend(y.numpy().tolist())
        paths.extend(p)
    return preds, tgts, paths


# -------------------- MAIN --------------------
def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device used: {device}")

    # Which checkpoints?
    if args.ckpt:
        m = CKPT_NAME_RE.match(os.path.basename(args.ckpt))
        exp_name = args.exp_name or (m.group("exp") if m else None)
        ckpt_dir = os.path.dirname(os.path.abspath(args.ckpt))
        label = f"fold{m.group('fold')}" if m and m.group("fold") else os.path.basename(args.ckpt)
        ckpts = [(label, args.ckpt)]
    else:
        exp_name = args.exp_name or CONFIG["exp_name"]
        ckpt_dir = args.ckpt_dir or CONFIG["checkpoint_dir"]
        ckpts = [(f"fold{k}", p) for k, p in find_fold_checkpoints(ckpt_dir, exp_name)]
        if not ckpts:
            raise SystemExit(f"No {exp_name}_fold*.pth found in {ckpt_dir}")
    print(f"Checkpoints: {[os.path.basename(p) for _, p in ckpts]}")

    cfg = resolve_inference_config(CONFIG, ckpt_dir, exp_name, args.prompt, args.film_strength)

    # Test data
    items = load_test_list(
        args.test_json or cfg["test_json_path"], args.test_dir or cfg["test_data_dir"]
    )
    if not items:
        raise SystemExit("Test list is empty. Check paths and test.json content.")

    processor = build_processor(cfg["model_name"])
    _, eval_transform = build_transforms(cfg["image_size"])
    test_ds = IQADataset(
        items, processor, image_size=cfg["image_size"], transform=eval_transform, return_path=True
    )
    test_dl = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    # The backbone is built once; each checkpoint is loaded into it in turn.
    model = build_model(cfg, processor).to(device)

    results, pred_cols, paths, tgts = [], {}, None, None
    for label, path in ckpts:
        load_checkpoint(model, path)
        preds, tgts, paths = predict_loader(model, test_dl, device)
        res = {"checkpoint": os.path.basename(path), **compute_metrics(preds, tgts)}
        results.append(res)
        pred_cols[label] = preds

    # Report
    keys = ("plcc", "srocc", "krocc", "overall")
    print(f"\nTEST RESULTS ({exp_name or 'checkpoint'})")
    print(f"{'Model':<24} {'PLCC':>8} {'SROCC':>8} {'KROCC':>8} {'Overall':>10}")
    print("-" * 62)
    for r in results:
        print(f"{r['checkpoint']:<24} {r['plcc']:>8.4f} {r['srocc']:>8.4f} {r['krocc']:>8.4f} {r['overall']:>10.4f}")

    summary = {"prompt": cfg["text_prompt"] if cfg["use_text_prompt"] else None, "folds": results}
    if len(results) > 1:
        mean = {k: float(np.mean([r[k] for r in results])) for k in keys}
        std = {k: float(np.std([r[k] for r in results])) for k in keys}
        print("-" * 62)
        print(f"{'Mean':<24} {mean['plcc']:>8.4f} {mean['srocc']:>8.4f} {mean['krocc']:>8.4f} {mean['overall']:>10.4f}")
        print(f"{'Std':<24} {std['plcc']:>8.4f} {std['srocc']:>8.4f} {std['krocc']:>8.4f} {std['overall']:>10.4f}")
        summary.update({"mean_metrics": mean, "std_metrics": std})

    out = args.out or (None if args.ckpt else os.path.join(ckpt_dir, "test_results.json"))
    if out:
        os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=4, ensure_ascii=False)
        print(f"\nSaved test results to {out}")

    if args.save_preds:
        with open(args.save_preds, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["path", "mos"] + [f"pred_{c}" for c in pred_cols])
            for i, p in enumerate(paths):
                w.writerow([p, tgts[i]] + [pred_cols[c][i] for c in pred_cols])
        print(f"Saved per-image predictions to {args.save_preds}")


if __name__ == "__main__":
    main()
