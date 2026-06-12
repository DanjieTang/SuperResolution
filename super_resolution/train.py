from dataloader import prepare_dataset
from model import SuperResolution

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import argparse
import yaml

import wandb
import os
from datetime import datetime

def parse_args():
    parser = argparse.ArgumentParser(description="Train a SuperResolution Model")

    # Dataset & Paths
    parser.add_argument("--dataset_path", type=str, default="ImagenetHighResolution")

    # Model Architecture
    parser.add_argument("--embedding_dim", type=int, nargs="+", default=[3, 128, 256])
    parser.add_argument("--input_image_size", type=int, default=64)
    parser.add_argument("--output_image_size", type=int, default=256)

    # Training Hyperparameters
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--min_lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

    # WandB
    parser.add_argument("--project", type=str, default=None)
    parser.add_argument("--entity", type=str, default=None)
    parser.add_argument("--run_name", type=str, default=None)

    return parser.parse_args()

def main():
    args = parse_args()
    torch.manual_seed(0)

    # Use wandb if applicable
    use_wandb = args.project is not None and args.entity is not None

    run_log_dir = os.path.join("train_log", datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(run_log_dir, exist_ok=True)

    with open(os.path.join(run_log_dir, "hyperparameters.yaml"), "w") as f:
        yaml.dump(vars(args), f)

    if use_wandb:
        run = wandb.init(
            entity=args.entity,
            project=args.project,
            name=args.run_name,
            config={
                "embedding_dim": args.embedding_dim,
                "input_image_size": args.input_image_size,
                "output_image_size": args.output_image_size,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "min_lr": args.min_lr,
                "weight_decay": args.weight_decay,
            },
        )

    train_loader, val_loader = prepare_dataset(args.dataset_path, args.batch_size, image_size=args.output_image_size)

    model = SuperResolution(
        embedding_dim=args.embedding_dim,
        input_image_size=args.input_image_size,
    ).to(args.device)
    print(f"Model Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Optimizer & Scheduler
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs * len(train_loader), eta_min=args.min_lr)

    # Tracking metrics
    train_losses, val_losses = [], []

    for epoch in range(args.epochs):
        model.train()
        epoch_train_loss = []

        for step, (images, _) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]")):
            # Downsample for model input, original image is the target
            low_resolution_image = F.interpolate(images, size=(args.input_image_size, args.input_image_size), mode='bicubic', align_corners=False).to(args.device)
            high_resolution_image = images.to(args.device)

            # Forward pass
            optimizer.zero_grad()
            predicted_high_resolution_image = model(low_resolution_image)
            loss = criterion(predicted_high_resolution_image, high_resolution_image)

            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            # Record loss
            epoch_train_loss.append(loss.item())

            if step % 100 == 0:
                # Visualize the first image in the batch
                lr_img = low_resolution_image[0].detach().cpu()
                pr_hr_img = predicted_high_resolution_image[0].detach().cpu()

                # Unnormalize [-1, 1] -> [0, 1]
                lr_img = lr_img * 0.5 + 0.5
                pr_hr_img = pr_hr_img * 0.5 + 0.5

                lr_img = lr_img.clamp(0, 1).permute(1, 2, 0).numpy()
                pr_hr_img = pr_hr_img.clamp(0, 1).permute(1, 2, 0).numpy()

                fig, axes = plt.subplots(1, 2, figsize=(10, 5))
                axes[0].imshow(lr_img)
                axes[0].set_title(f"Input ({args.input_image_size}x{args.input_image_size})")
                axes[0].axis("off")

                axes[1].imshow(pr_hr_img)
                axes[1].set_title(f"Output ({args.output_image_size}x{args.output_image_size})")
                axes[1].axis("off")

                plt.suptitle(f"Epoch {epoch+1}, Iteration {step}")
                
                save_path = os.path.join(run_log_dir, f"epoch_{epoch+1}_iteration_{step}.png")
                plt.savefig(save_path)

                if use_wandb:
                    run.log({"Visualization": wandb.Image(fig)})
                plt.close(fig)

        avg_train_loss = np.mean(epoch_train_loss)
        train_losses.append(avg_train_loss)

        model.eval()
        epoch_val_loss = []
        with torch.no_grad():
            for images, _ in tqdm(val_loader, desc="Validating"):
                low_resolution_image = F.interpolate(images, size=(args.input_image_size, args.input_image_size), mode='bicubic', align_corners=False).to(args.device)
                high_resolution_image = images.to(args.device)

                predicted_high_resolution_image = model(low_resolution_image)
                loss = criterion(predicted_high_resolution_image, high_resolution_image)

                # Record loss
                epoch_val_loss.append(loss.item())

        avg_val_loss = np.mean(epoch_val_loss)
        val_losses.append(avg_val_loss)

        print(f"Epoch {epoch}: Train Loss {avg_train_loss:.4f} | Val Loss {avg_val_loss:.4f}")
        if use_wandb:
            run.log({"Training Loss": train_losses[-1], "Val loss": val_losses[-1]})

    if not use_wandb:
        plt.plot(train_losses, label="Training loss")
        plt.plot(val_losses, label="Validation loss")
        print("Training loss: ", train_losses[-1])
        print("Validation loss: ", val_losses[-1])
        plt.legend()
        plt.show()

    if use_wandb:
        run.finish()

if __name__ == "__main__":
    main()
