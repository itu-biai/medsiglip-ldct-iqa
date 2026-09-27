import argparse
import csv
import os
import warnings

import torch
from torch.utils.data import DataLoader

from config import CONFIG
from data import IQADataset, build_processor, build_transforms
from model import CKPT_NAME_RE, build_model, load_checkpoint, resolve_inference_config

warnings.filterwarnings("ignore")


def parse_args():
    p = argparse.ArgumentParser(description="Predict MOS (0-4) for individual CT images.")
    p.add_argument("images", nargs="+", help="image files (.tif/.png/.jpg/.dcm ...)")
    p.add_argument("--ckpt", required=True, help="checkpoint (.pth)")
    p.add_argument("--exp-name", help="read {exp_name}_config.json next to the checkpoint (default: parsed from file name)")
    p.add_argument("--prompt", help="override the text prompt")
    p.add_argument("--out", help="optional CSV output")
    p.add_argument("--batch-size", type=int, default=CONFIG["batch_size"])
    return p.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    m = CKPT_NAME_RE.match(os.path.basename(args.ckpt))
    exp_name = args.exp_name or (m.group("exp") if m else None)
    ckpt_dir = os.path.dirname(os.path.abspath(args.ckpt))
    cfg = resolve_inference_config(CONFIG, ckpt_dir, exp_name, args.prompt)

    processor = build_processor(cfg["model_name"])
    _, eval_transform = build_transforms(cfg["image_size"])
    model = build_model(cfg, processor)
    load_checkpoint(model, args.ckpt)
    model.to(device).eval()

    # Scores are unused for prediction; NaN placeholders keep the dataset format.
    ds = IQADataset(
        [(p, float("nan")) for p in args.images],
        processor,
        image_size=cfg["image_size"],
        transform=eval_transform,
        return_path=True,
    )
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False)

    rows = []
    for inputs, _, paths in dl:
        preds = model({"pixel_values": inputs["pixel_values"].to(device)})
        rows.extend(zip(paths, preds.cpu().tolist()))

    for path, score in rows:
        print(f"{score:.4f}\t{path}")

    if args.out:
        with open(args.out, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["path", "pred_mos"])
            w.writerows(rows)
        print(f"Saved predictions to {args.out}")


if __name__ == "__main__":
    main()
