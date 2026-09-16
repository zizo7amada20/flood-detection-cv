import numpy as np

mask = np.load("data/mask_ready.npy")
valid = mask[mask != -1]
unique, counts = np.unique(valid, return_counts=True)
total = valid.size

for cls, count in zip(unique, counts):
    print(f"Class {cls}: {count:,} pixels ({100*count/total:.3f}%)")