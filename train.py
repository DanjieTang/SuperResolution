from dataloader import prepare_dataset, prepare_latent_loaders, sample_known_region_mask
from latents import load_latents, sample_latent
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

# The pretrained VAE downsamples by 8x, so a 512 canvas becomes a 64x64x4 latent
VAE_DOWNSAMPLE_FACTOR = 8

def parse_args():
    parser = argparse.ArgumentParser(description="Train an Outpainting Model (flow matching DiT)")

    # Dataset & Paths
    parser.add_argument("--dataset_path", type=str, default="ImagenetHighResolution")

    # Model Architecture
    parser.add_argument("--canvas_size", type=int, default=512)
    parser.add_argument("--patch_size", type=int, default=None, help="Defaults to 16 in pixel space, 2 in VAE latent space")
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--depth", type=int, default=12)
    parser.add_argument("--num_head", type=int, default=8)

    # Optional latent diffusion: train on frozen pretrained VAE latents
    # instead of raw pixels (downloads the VAE from Hugging Face)
    parser.add_argument("--use_vae", action="store_true")
    parser.add_argument("--vae", type=str, default="stabilityai/sd-vae-ft-ema")
    parser.add_argument("--latent_cache", type=str, default="latent_cache/latents.pt",
                        help="Precomputed latents from precompute_latents.py; read instead of encoding")

    # Training Hyperparameters
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=64)
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

    args = parser.parse_args()
    if args.patch_size is None:
        args.patch_size = 2 if args.use_vae else 16
    return args

@torch.no_grad()
def decode(vae, tensor: torch.Tensor) -> torch.Tensor:
    """Map the model's working space back to pixels."""
    if vae is None:
        return tensor
    return vae.decode(tensor / vae.config.scaling_factor).sample

@torch.no_grad()
def sample(model: DiT, known: torch.Tensor, mask: torch.Tensor, steps: int) -> torch.Tensor:
    """
    Generate by integrating the learned velocity field from pure noise (t=1)
    back to data (t=0) with Euler steps.
    """
    tensor = torch.randn_like(known)
    timesteps = torch.linspace(1.0, 0.0, steps + 1, device=tensor.device)

    for i in range(steps):
        timestep = timesteps[i].expand(tensor.shape[0])
        velocity = model(tensor, timestep, known, mask)
        tensor = tensor - (timesteps[i] - timesteps[i + 1]) * velocity

    return tensor

def visualize(model, vae, working, mask, args, run_log_dir, epoch, step, use_wandb, run):
    """Outpaint the first sample of the batch and save known/output side by side."""
    model.eval()

    work = working[:1]
    work_mask = mask[:1]
    known = work * work_mask

    generated = sample(model, known, work_mask, args.sample_steps)

    # Back to pixels: identity in pixel mode, VAE decode in latent mode
    target_px = decode(vae, work)
    generated_px = decode(vae, generated)
    pixel_mask = F.interpolate(work_mask, size=target_px.shape[-2:], mode="nearest")

    # Keep the original pixels where they are known
    generated_px = target_px * pixel_mask + generated_px * (1 - pixel_mask)
    known_px = target_px * pixel_mask

    masked_view = (known_px[0].float().cpu() * 0.5 + 0.5).clamp(0, 1).permute(1, 2, 0).numpy()
    generated_view = (generated_px[0].float().cpu() * 0.5 + 0.5).clamp(0, 1).permute(1, 2, 0).numpy()

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

def flow_matching_loss(model, working, working_mask, criterion):
    """Flow matching loss in the model's working space (pixels or VAE latents)."""
    target = working
    known = working * working_mask

    # Flow matching: interpolate between data (t=0) and noise (t=1),
    # the model learns the constant velocity from data to noise
    noise = torch.randn_like(target)
    timestep = torch.rand(target.shape[0], device=target.device)
    t = timestep.view(-1, 1, 1, 1)
    noisy_target = (1 - t) * target + t * noise
    velocity_target = noise - target

    velocity_prediction = model(noisy_target, timestep, known, working_mask)
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

    # Frozen pretrained VAE (latent diffusion) or raw pixel space
    vae = None
    scaling_factor = 1.0
    if args.use_vae:
        try:
            from diffusers import AutoencoderKL
        except ImportError:
            raise ImportError("--use_vae requires diffusers: run 'uv sync --extra vae' or 'pip install diffusers'")
        if not os.path.exists(args.latent_cache):
            raise FileNotFoundError(
                f"No latent cache at '{args.latent_cache}'. Precompute it once with:\n"
                f"  python precompute_latents.py --dataset_path {args.dataset_path} "
                f"--canvas_size {args.canvas_size} --vae {args.vae} --output {args.latent_cache}"
            )
        # The VAE is loaded only to decode samples for visualization; the
        # training loop reads latents from disk and never encodes.
        vae = AutoencoderKL.from_pretrained(args.vae).to(args.device)
        vae.requires_grad_(False)
        vae.eval()

        moments, scaling_factor = load_latents(args.latent_cache)
        print(f"Loaded {moments.shape[0]} cached latents from {args.latent_cache}")
        train_loader, val_loader = prepare_latent_loaders(moments, args.batch_size)
    else:
        train_loader, val_loader = prepare_dataset(args.dataset_path, args.batch_size, canvas_size=args.canvas_size)

    working_channels = 4 if args.use_vae else 3
    working_size = args.canvas_size // VAE_DOWNSAMPLE_FACTOR if args.use_vae else args.canvas_size

    model = DiT(
        image_size=working_size,
        patch_size=args.patch_size,
        in_channels=working_channels * 2 + 1,
        out_channels=working_channels,
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

        for step, (x, _) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]")):
            x = x.to(args.device)
            working = sample_latent(x, scaling_factor) if args.use_vae else x
            mask = sample_known_region_mask(working.shape[0], working.shape[-1], args.device)

            # Forward pass
            optimizer.zero_grad()
            with autocast:
                loss = flow_matching_loss(model, working, mask, criterion)

            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            # Record loss
            epoch_train_loss.append(loss.item())

            if step % args.visualize_every == 0:
                with autocast:
                    visualize(model, vae, working, mask, args, run_log_dir, epoch, step, use_wandb, run)

        avg_train_loss = np.mean(epoch_train_loss)
        train_losses.append(avg_train_loss)

        model.eval()
        epoch_val_loss = []
        with torch.no_grad():
            for x, _ in tqdm(val_loader, desc="Validating"):
                x = x.to(args.device)
                working = sample_latent(x, scaling_factor) if args.use_vae else x
                mask = sample_known_region_mask(working.shape[0], working.shape[-1], args.device)
                with autocast:
                    loss = flow_matching_loss(model, working, mask, criterion)

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
