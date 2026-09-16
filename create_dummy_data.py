"""Generate small dummy .npy files matching the shapes FloodDataset expects.

This lets you smoke-test train_baseline.py and evaluate.py end-to-end
before the real ImpactMesh data is downloaded and converted.

Run from the repo root (the folder that contains data/ and src/):
    python create_dummy_data.py
"""

import numpy as np
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
DATA_DIR = REPO_ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)

N_SAMPLES = 20
H, W = 256, 256
SAR_CHANNELS = 2
OPTICAL_CHANNELS = 12

rng = np.random.default_rng(42)

# SAR and optical: random float arrays, shape (N, H, W, C).
# dataset.py transposes each sample to (C, H, W) itself, so we store (H, W, C) here.
sar = rng.normal(size=(N_SAMPLES, H, W, SAR_CHANNELS)).astype(np.float32)
optical = rng.normal(size=(N_SAMPLES, H, W, OPTICAL_CHANNELS)).astype(np.float32)

# Mask: mostly land (0), some flood (1), some permanent water (2), a bit of no-data (-1).
mask = rng.choice(
    [0, 1, 2, -1],
    size=(N_SAMPLES, H, W),
    p=[0.70, 0.15, 0.10, 0.05],
).astype(np.int64)

np.save(DATA_DIR / "sar_ready.npy", sar)
np.save(DATA_DIR / "optical_ready.npy", optical)
np.save(DATA_DIR / "mask_ready.npy", mask)

print(f"Wrote {N_SAMPLES} dummy samples to {DATA_DIR}")
print(f"  sar_ready.npy      : {sar.shape}")
print(f"  optical_ready.npy  : {optical.shape}")
print(f"  mask_ready.npy     : {mask.shape}  (values: -1, 0, 1, 2)")
