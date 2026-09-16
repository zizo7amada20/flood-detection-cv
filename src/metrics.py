import torch


def _binarize(logits, target, flood_class=1, ignore_index=-1):
    """
    Helper: convert multi-class logits + labels into binary masks for the
    flood class, and return a valid mask for the no-data pixels.

    logits : (B, C, H, W)  raw scores
    target : (B, H, W)     labels in {-1, 0, 1, 2}

    returns:
        pred_binary   (B, H, W) bool  -> where the model predicts flood
        target_binary (B, H, W) bool  -> where the GT is actually flood
        valid         (B, H, W) bool  -> where the label is not no-data
    """
    pred_class = logits.argmax(dim=1)                 # (B, H, W) picked class per pixel
    pred_binary = (pred_class == flood_class)         # true where predicted flood
    target_binary = (target == flood_class)           # true where GT is flood
    valid = (target != ignore_index)                  # true where label is usable
    return pred_binary, target_binary, valid


def compute_iou(logits, target, flood_class=1, ignore_index=-1, eps=1e-7):
    """Compute IoU for the flood class, ignoring no-data pixels.

    logits : (B, C, H, W)
    target : (B, H, W) with -1 marking no-data
    """
    pred_binary, target_binary, valid = _binarize(logits, target, flood_class, ignore_index)

    # drop no-data pixels from both prediction and ground truth
    pred_binary = pred_binary & valid
    target_binary = target_binary & valid

    # count pixels predicted and labeled as flood
    intersection = (pred_binary & target_binary).sum().float()

    # step 2: count all predicted and actual flood pixels without double counting
    union = (pred_binary | target_binary).sum().float()

    if union == 0:
        # no flood in GT and no flood predicted -> undefined, return NaN
        # (so the caller can skip this image instead of getting a misleading 1.0)
        return torch.tensor(float('nan'), device=logits.device)

    # last step: IoU = intersection / union
    return (intersection + eps) / (union + eps)


def compute_f1(logits, target, flood_class=1, ignore_index=-1, eps=1e-7):
    """Compute F1 score for the flood class, ignoring no-data pixels.

    logits : (B, C, H, W)
    target : (B, H, W) with -1 marking no-data
    """
    pred_binary, target_binary, valid = _binarize(logits, target, flood_class, ignore_index)

    # drop no-data pixels
    pred_binary = pred_binary & valid
    target_binary = target_binary & valid

    tp = (pred_binary & target_binary).sum().float()   # correctly predicted flood pixels
    fp = (pred_binary & ~target_binary).sum().float()  # incorrectly predicted flood pixels
    fn = (~pred_binary & target_binary).sum().float()  # missed flood pixels

    if (tp + fp + fn) == 0:
        # no flood in GT and none predicted -> undefined
        return torch.tensor(float('nan'), device=logits.device)

    precision = tp / (tp + fp + eps)   # how many predicted flood pixels are actually flood
    recall = tp / (tp + fn + eps)      # how many actual flood pixels were detected
    return (2 * precision * recall) / (precision + recall + eps)  # F1 from precision & recall