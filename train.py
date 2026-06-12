from dataloader import prepare_dataset, sample_known_region_mask
from model import DiT

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
from contextlib import nullcontext
from datetime import datetime

def parse_args():
    parser = argparse.ArgumentParser(description="Train an Outpainting Model (pixel-space flow matching DiT)")

    # Dataset & Paths
    parser.add_argument("--dataset_path", type=str, default="ImagenetHighResolution")

    # Model Architecture
    parser.add_argument("--canvas_size", type=int, default=512)
    parser.add_argument("--patch_size", type=int, default=16)
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--depth", type=int, default=12)
    parser.add_argument("--num_head", type=int, default=8)

    # Training Hyperparameters
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--min_lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-3)
    parser.add_argument("--sample_steps", type=int, default=50)
    parser.add_argument("--visualize_every", type=int, default=500)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    # WandB
    parser.add_argument("--project", type=str, default=None)
    parser.add_argument("--entity", type=str, default=None)
    parser.add_argument("--run_name", type=str, default=None)

    return parser.parse_args()

@torch.no_grad()
def sample(model: DiT, known_image: torch.Tensor, mask: torch.Tensor, steps: int) -> torch.Tensor:
    """
    Generate images by integrating the learned velocity field from pure noise
    (t=1) back to data (t=0) with Euler steps.
    """
    image = torch.randn_like(known_image)
    timesteps = torch.linspace(1.0, 0.0, steps + 1, device=image.device)

    for i in range(steps):
        timestep = timesteps[i].expand(image.shape[0])
        velocity = model(image, timestep, known_image, mask)
        image = image - (timesteps[i] - timesteps[i + 1]) * velocity

    return image

def visualize(model, images, mask, args, run_log_dir, epoch, step, use_wandb, run):
    """Outpaint the first image of the batch and save input/output side by side."""
    model.eval()

    image = images[:1]
    image_mask = mask[:1]
    masked_image = image * image_mask

    generated = sample(model, masked_image, image_mask, args.sample_steps)

    # Keep the original pixels where they are known
    generated = image * image_mask + generated * (1 - image_mask)

    masked_view = (masked_image[0].float().cpu() * 0.5 + 0.5).clamp(0, 1).permute(1, 2, 0).numpy()
    generated_view = (generated[0].float().cpu() * 0.5 + 0.5).clamp(0, 1).permute(1, 2, 0).numpy()

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(masked_view)
    axes[0].set_title("Known region")
    axes[0].axis("off")

    axes[1].imshow(generated_view)
    axes[1].set_title("Outpainted")
    axes[1].axis("off")

    plt.suptitle(f"Epoch {epoch + 1}, Iteration {step}")

    save_path = os.path.join(run_log_dir, f"epoch_{epoch + 1}_iteration_{step}.png")
    plt.savefig(save_path)

    if use_wandb:
        run.log({"Visualization": wandb.Image(fig)})
    plt.close(fig)

    model.train()

def flow_matching_loss(model, images, mask, criterion):
    masked_image = images * mask

    # Flow matching: interpolate between data (t=0) and noise (t=1),
    # the model learns the constant velocity from data to noise
    noise = torch.randn_like(images)
    timestep = torch.rand(images.shape[0], device=images.device)
    t = timestep.view(-1, 1, 1, 1)
    noisy_image = (1 - t) * images + t * noise
    velocity_target = noise - images

    velocity_prediction = model(noisy_image, timestep, masked_image, mask)
    return criterion(velocity_prediction, velocity_target)

def main():
    args = parse_args()
    torch.manual_seed(0)

    # Use wandb if applicable
    use_wandb = args.project is not None and args.entity is not None

    run_log_dir = os.path.join("train_log", datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(run_log_dir, exist_ok=True)

    with open(os.path.join(run_log_dir, "hyperparameters.yaml"), "w") as f:
        yaml.dump(vars(args), f)

    run = None
    if use_wandb:
        run = wandb.init(
            entity=args.entity,
            project=args.project,
            name=args.run_name,
            config=vars(args),
        )

    train_loader, val_loader = prepare_dataset(args.dataset_path, args.batch_size, canvas_size=args.canvas_size)

    model = DiT(
        image_size=args.canvas_size,
        patch_size=args.patch_size,
        hidden_dim=args.hidden_dim,
        depth=args.depth,
        num_head=args.num_head,
    ).to(args.device)
    print(f"Model Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # bf16 autocast on CUDA, full precision elsewhere
    autocast = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if args.device.startswith("cuda") else nullcontext()

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
            images = images.to(args.device)
            mask = sample_known_region_mask(images.shape[0], args.canvas_size, args.device)

            # Forward pass
            optimizer.zero_grad()
            with autocast:
                loss = flow_matching_loss(model, images, mask, criterion)

            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            # Record loss
            epoch_train_loss.append(loss.item())

            if step % args.visualize_every == 0:
                with autocast:
                    visualize(model, images, mask, args, run_log_dir, epoch, step, use_wandb, run)

        avg_train_loss = np.mean(epoch_train_loss)
        train_losses.append(avg_train_loss)

        model.eval()
        epoch_val_loss = []
        with torch.no_grad():
            for images, _ in tqdm(val_loader, desc="Validating"):
                images = images.to(args.device)
                mask = sample_known_region_mask(images.shape[0], args.canvas_size, args.device)
                with autocast:
                    loss = flow_matching_loss(model, images, mask, criterion)

                # Record loss
                epoch_val_loss.append(loss.item())

        avg_val_loss = np.mean(epoch_val_loss)
        val_losses.append(avg_val_loss)

        print(f"Epoch {epoch}: Train Loss {avg_train_loss:.4f} | Val Loss {avg_val_loss:.4f}")
        if use_wandb:
            run.log({"Training Loss": train_losses[-1], "Val loss": val_losses[-1]})

        torch.save(model.state_dict(), os.path.join(run_log_dir, "model.pt"))

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
