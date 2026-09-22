import numpy as np
import torch
from torch.utils.data import Dataset

class FloodDataset(Dataset):
    def __init__(self, sar_path, optical_path, mask_path, augment=False):
        # إضافة mmap_mode='r' عشان نسحب الصور من الهارد مباشرة من غير ما نملى الرامات
        self.sar_data = np.load(sar_path, mmap_mode='r')
        self.optical_data = np.load(optical_path, mmap_mode='r')
        self.mask_data = np.load(mask_path, mmap_mode='r')
        self.augment = augment

    def __len__(self):
        return len(self.mask_data)

    def __getitem__(self, idx):
        sar = self.sar_data[idx]
        optical = self.optical_data[idx]
        mask = self.mask_data[idx]

        sar = np.transpose(sar, (2, 0, 1)).astype(np.float32)
        optical = np.transpose(optical, (2, 0, 1)).astype(np.float32)

        sar = torch.from_numpy(sar)
        optical = torch.from_numpy(optical)
        mask = torch.from_numpy(mask.astype(np.int64))

        imagenet_mean = torch.tensor([0.485, 0.456, 0.406], dtype=optical.dtype).view(3, 1, 1)
        imagenet_std = torch.tensor([0.229, 0.224, 0.225], dtype=optical.dtype).view(3, 1, 1)
        optical = (optical - imagenet_mean) / imagenet_std

        if self.augment:
            if torch.rand(1).item() < 0.5:
                sar = torch.flip(sar, dims=[2])
                optical = torch.flip(optical, dims=[2])
                mask = torch.flip(mask, dims=[1])
            if torch.rand(1).item() < 0.5:
                sar = torch.flip(sar, dims=[1])
                optical = torch.flip(optical, dims=[1])
                mask = torch.flip(mask, dims=[0])

        valid_mask = (mask != -1)

        return {
            "sar": sar,
            "optical": optical,
            "mask": mask,
            "valid_mask": valid_mask
        }


if __name__ == "__main__":
    dataset = FloodDataset(
        sar_path="cv/sar_ready.npy",
        optical_path="cv/optical_ready.npy",
        mask_path="cv/mask_ready.npy"
    )

    print("عدد العينات:", len(dataset))

    sample = dataset[0]
    print("SAR shape:", sample["sar"].shape)
    print("Optical shape:", sample["optical"].shape)
    print("Mask shape:", sample["mask"].shape)