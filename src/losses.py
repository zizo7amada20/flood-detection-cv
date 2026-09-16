import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceBCELoss(nn.Module):
    """
    Multi-class Dice + CrossEntropy loss.

    Compatible with the baseline pipeline:
        logits : (B, num_classes, H, W)
        target : (B, H, W) with values {-1, 0, 1, 2}
                 -1 = no-data  -> ignored by both CE and Dice
    """

    def __init__(self, ce_weight=0.5, ignore_index=-1, num_classes=3, class_weights=None):
        super().__init__()
        # CrossEntropyLoss already supports ignore_index for the no-data pixels,
        # so we use it as the "BCE-like" term for the multi-class setting.
        if class_weights is not None:
            class_weights = torch.as_tensor(class_weights, dtype=torch.float32)
            if class_weights.numel() != num_classes:
                raise ValueError("class_weights must contain one weight per class.")
        self.register_buffer("class_weights", class_weights)
        self.ce = nn.CrossEntropyLoss(ignore_index=ignore_index, weight=self.class_weights)
        self.ce_weight = ce_weight
        self.ignore_index = ignore_index
        self.num_classes = num_classes

    def forward(self, logits, target, eps=1e-7):
        """
        logits : (B, C, H, W)  raw scores (no softmax applied yet)
        target : (B, H, W)     int labels in {-1, 0, 1, 2}
        """
        # CE term (handles ignore_index internally) 
        ce_loss = self.ce(logits, target)

        # Dice term
        # softmax over the class dimension so each pixel sums to 1
        probs = F.softmax(logits, dim=1)                      # (B, C, H, W)

        # build a valid mask from the target so we can drop no-data pixels
        valid = (target != self.ignore_index)                 # (B, H, W)
        valid = valid.unsqueeze(1).float()                    # (B, 1, H, W)
        probs = probs * valid                                  # zero-out no-data

        # one-hot encode target; clamp -1 -> 0 first to avoid index errors
        target_clamped = target.clone()
        target_clamped[target == self.ignore_index] = 0
        target_onehot = F.one_hot(target_clamped, self.num_classes)   # (B,H,W,C)
        target_onehot = target_onehot.permute(0, 3, 1, 2).float()     # (B,C,H,W)
        target_onehot = target_onehot * valid                         # zero-out no-data

        # intersection / union per class (summed over batch + H + W)
        dims = (0, 2, 3)
        intersection = (probs * target_onehot).sum(dims)              # (C,)
        denom = probs.sum(dims) + target_onehot.sum(dims)             # (C,)

        # keep only classes that actually appear in the valid pixels of this batch
        present = denom > 0
        if present.any():
            dice_per_class = (2 * intersection[present] + eps) / (denom[present] + eps)
            dice_loss = 1 - dice_per_class.mean()
        else:
            dice_loss = torch.tensor(0.0, device=logits.device)

        #combine
        return self.ce_weight * ce_loss + (1 - self.ce_weight) * dice_loss
