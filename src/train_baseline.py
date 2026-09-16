"""Minimal end-to-end training baseline for the 100-sample flood subset.

Run from the repository root:
    python src/train_baseline.py --epochs 3 --batch-size 1

The target uses three classes: land=0, flood=1, permanent-water=2.
Pixels labelled -1 are ignored by the loss and therefore do not
contribute to gradients or the reported loss.
"""

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split

from dataset import FloodDataset
from models.model import FloodModel
from losses import DiceBCELoss
from metrics import compute_iou, compute_f1

REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser(description="Train the flood baseline on .npy arrays.")
    parser.add_argument("--sar", type=Path, default=REPO_ROOT / "data" / "sar_ready.npy")
    parser.add_argument("--optical", type=Path, default=REPO_ROOT / "data" / "optical_ready.npy")
    parser.add_argument("--mask", type=Path, default=REPO_ROOT / "data" / "mask_ready.npy")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--val-split", type=float, default=0.2,
                        help="Fraction of the dataset held out for evaluation.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--max-batches",
        type=int,
        default=None,
        help="Optional cap for a quick smoke test; omit to use every batch.",
    )
    return parser.parse_args()


def validate_batch(batch):
    sar, optical, target, valid_mask = (
        batch["sar"], batch["optical"], batch["mask"], batch["valid_mask"],
    )
    if sar.ndim != 4 or sar.shape[1:] != (2, 256, 256):
        raise ValueError(f"Expected SAR (B, 2, 256, 256), got {tuple(sar.shape)}")
    if optical.ndim != 4 or optical.shape[1:] != (12, 256, 256):
        raise ValueError(f"Expected optical (B, 12, 256, 256), got {tuple(optical.shape)}")
    if target.ndim != 3 or target.shape[1:] != (256, 256):
        raise ValueError(f"Expected target (B, 256, 256), got {tuple(target.shape)}")
    if not torch.equal(valid_mask, target.ne(-1)):
        raise ValueError("valid_mask must be True exactly where mask is not -1.")
    allowed_labels = torch.tensor([-1, 0, 1, 2], dtype=target.dtype)
    if not torch.isin(target.cpu(), allowed_labels).all():
        raise ValueError("Mask contains a label outside {-1, 0, 1, 2}.")


def main():
    args = parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        raise ValueError("epochs and batch-size must both be positive.")

    dataset = FloodDataset(args.sar, args.optical, args.mask)
    val_size = max(1, int(len(dataset) * args.val_split))
    train_size = len(dataset) - val_size
    generator = torch.Generator().manual_seed(args.seed)
    train_ds, _ = random_split(dataset, [train_size, val_size], generator=generator)

    loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    if len(loader) == 0:
        raise ValueError("Dataset is empty.")

    first_batch = next(iter(loader))
    validate_batch(first_batch)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Train samples: {len(train_ds)} | Device: {device}")
    print(
        "Real batch shapes: "
        f"SAR={tuple(first_batch['sar'].shape)}, "
        f"Optical={tuple(first_batch['optical'].shape)}, "
        f"Mask={tuple(first_batch['mask'].shape)}"
    )

    model = FloodModel(num_classes=3).to(device)
    criterion = DiceBCELoss(ce_weight=0.5, ignore_index=-1, num_classes=3)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    checkpoint_dir = REPO_ROOT / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    best_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        batches_with_valid_pixels = 0
        total_iou = 0.0
        total_f1 = 0.0
        metrics_count = 0

        for batch_index, batch in enumerate(loader, start=1):
            if args.max_batches is not None and batch_index > args.max_batches:
                break

            sar = batch["sar"].to(device, non_blocking=True)
            optical = batch["optical"].to(device, non_blocking=True)
            target = batch["mask"].to(device, non_blocking=True)

            if not batch["valid_mask"].any():
                continue

            optimizer.zero_grad(set_to_none=True)
            logits = model(sar, optical)
            if logits.shape != (sar.shape[0], 3, 256, 256):
                raise RuntimeError(f"Unexpected model output shape: {tuple(logits.shape)}")
            loss = criterion(logits, target)
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                iou = compute_iou(logits.detach(), target)
                f1 = compute_f1(logits.detach(), target)

            if not torch.isnan(iou):
                total_iou += iou.item()
                total_f1 += f1.item()
                metrics_count += 1

            total_loss += loss.item()
            batches_with_valid_pixels += 1

        if batches_with_valid_pixels == 0:
            raise RuntimeError("No valid pixels were found in this epoch.")
        mean_loss = total_loss / batches_with_valid_pixels
        mean_iou = total_iou / metrics_count if metrics_count else float("nan")
        mean_f1 = total_f1 / metrics_count if metrics_count else float("nan")
        print(
            f"Epoch {epoch}/{args.epochs} | loss: {mean_loss:.6f} "
            f"| flood IoU: {mean_iou:.4f} | flood F1: {mean_f1:.4f}"
        )

        if mean_loss < best_loss:
            best_loss = mean_loss
            torch.save(
                {"epoch": epoch, "model": model.state_dict(),
                 "optimizer": optimizer.state_dict(), "loss": mean_loss},
                checkpoint_dir / "best_model.pth",
            )
            print(f"  ✅ Saved new best model (loss={mean_loss:.6f})")


if __name__ == "__main__":
    main()