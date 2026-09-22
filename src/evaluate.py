import argparse
import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
from dataset import FloodDataset
from models.model import FloodModel

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate and visualize the flood MVP.")
    parser.add_argument("--sar", type=Path, required=True)
    parser.add_argument("--optical", type=Path, required=True)
    parser.add_argument("--mask", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()

@torch.no_grad()
def evaluate_and_visualize(model, loader, device, num_images_to_save=5):
    model.eval()
    saved_count = 0
    output_dir = "/kaggle/working/"
    
    for i, batch in enumerate(loader):
        if saved_count >= num_images_to_save:
            break
            
        sar = batch["sar"].to(device, non_blocking=True)
        optical = batch["optical"].to(device, non_blocking=True)
        target = batch["mask"].to(device, non_blocking=True)

        logits = model(sar, optical)
        preds = torch.argmax(logits, dim=1)

        img_optical = optical[0].permute(1, 2, 0).cpu().numpy().astype(np.float32)
        # تفتيح الصورة شوية عشان تبان في الديمو
        img_optical = np.clip(img_optical * 2.0, 0, 1) 
        
        mask_true = target[0].cpu().numpy()
        mask_pred = preds[0].cpu().numpy()

        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 3, 1)
        plt.title("Satellite Image (Optical)")
        plt.imshow(img_optical)
        plt.axis('off')
        
        plt.subplot(1, 3, 2)
        plt.title("Ground Truth (Actual Flood)")
        plt.imshow(mask_true, cmap='jet')
        plt.axis('off')
        
        plt.subplot(1, 3, 3)
        plt.title("MVP Prediction")
        plt.imshow(mask_pred, cmap='jet')
        plt.axis('off')
        
        save_path = os.path.join(output_dir, f"demo_result_{saved_count+1}.png")
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Saved MVP demo image to: {save_path}")
        saved_count += 1

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = FloodDataset(args.sar, args.optical, args.mask)
    val_size = max(1, int(len(dataset) * args.val_split))
    train_size = len(dataset) - val_size
    
    generator = torch.Generator().manual_seed(args.seed)
    _, val_ds = random_split(dataset, [train_size, val_size], generator=generator)

    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    model = FloodModel(num_classes=3).to(device)
    state = torch.load(args.checkpoint, map_location=device)
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    model.load_state_dict(state)

    print("Generating Demo Visualization...")
    evaluate_and_visualize(model, val_loader, device, num_images_to_save=5)

if __name__ == "__main__":
    main()