"""Evaluation script for the Flood baseline.

Computes IoU and F1 for the Flood class (class 1) on a validation/test split,
ignoring no-data pixels (label == -1).

Run from the repository root:
    python src/evaluate.py --checkpoint checkpoints/best_model.pth
"""

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split

from dataset import FloodDataset
from models.model import FloodModel
from metrics import compute_iou, compute_f1

REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate the flood baseline.")
    parser.add_argument("--sar", type=Path, default=REPO_ROOT / "data" / "sar_ready.npy")
    parser.add_argument("--optical", type=Path, default=REPO_ROOT / "data" / "optical_ready.npy")
    parser.add_argument("--mask", type=Path, default=REPO_ROOT / "data" / "mask_ready.npy")
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help="Path to the trained model weights (.pth).")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--val-split", type=float, default=0.2,
                        help="Fraction of the dataset used for validation.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


@torch.no_grad()
def evaluate(model, loader, device):
    """Run the model over the loader and return mean IoU and F1 for flood class."""
    model.eval()
    ious, f1s = [], []

    for batch in loader:
        sar = batch["sar"].to(device, non_blocking=True)
        optical = batch["optical"].to(device, non_blocking=True)
        target = batch["mask"].to(device, non_blocking=True)

        logits = model(sar, optical)

        iou = compute_iou(logits, target)
        f1 = compute_f1(logits, target)

        # NaN means no flood in GT and none predicted -> skip this image
        if not torch.isnan(iou):
            ious.append(iou.item())
        if not torch.isnan(f1):
            f1s.append(f1.item())

    mean_iou = sum(ious) / len(ious) if ious else float("nan")
    mean_f1 = sum(f1s) / len(f1s) if f1s else float("nan")
    return mean_iou, mean_f1


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = FloodDataset(args.sar, args.optical, args.mask)

    # ---- Split into train / val (we only need val here) ----
    val_size = max(1, int(len(dataset) * args.val_split))
    train_size = len(dataset) - val_size
    generator = torch.Generator().manual_seed(args.seed)
    _, val_ds = random_split(dataset, [train_size, val_size], generator=generator)

    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    # ---- Load model ----
    model = FloodModel(num_classes=3).to(device)
    state = torch.load(args.checkpoint, map_location=device)
    # support both raw state_dict and {"model": state_dict}
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    model.load_state_dict(state)

    print(f"Device: {device} | Val samples: {len(val_ds)} | Checkpoint: {args.checkpoint}")

    mean_iou, mean_f1 = evaluate(model, val_loader, device)
    print(f"Flood IoU: {mean_iou:.4f}")
    print(f"Flood F1 : {mean_f1:.4f}")


if __name__ == "__main__":
    main()