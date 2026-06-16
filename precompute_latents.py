"""
Encode an ImageFolder dataset to frozen-VAE latents once and store them to
disk, so training (train.py --use_vae) never runs the VAE encoder in its loop.

This is the expensive step that previously happened twice per training step;
doing it once offline brings --use_vae back to roughly pixel-mode speed.

Usage:
    python precompute_latents.py \
        --dataset_path ImagenetHighResolution \
        --canvas_size 512 \
        --vae stabilityai/sd-vae-ft-ema \
        --output latent_cache/latents.pt
"""
import argparse
from contextlib import nullcontext

import torch
from tqdm import tqdm

from dataloader import prepare_raw_loader
from latents import save_latents


def parse_args():
    parser = argparse.ArgumentParser(description="Precompute VAE latents for outpainting training")
    parser.add_argument("--dataset_path", type=str, default="ImagenetHighResolution")
    parser.add_argument("--canvas_size", type=int, default=512)
    parser.add_argument("--vae", type=str, default="stabilityai/sd-vae-ft-ema")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--output", type=str, default="latent_cache/latents.pt")
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    return parser.parse_args()


@torch.no_grad()
def main():
    args = parse_args()

    from diffusers import AutoencoderKL
    vae = AutoencoderKL.from_pretrained(args.vae).to(args.device)
    vae.requires_grad_(False)
    vae.eval()

    loader = prepare_raw_loader(args.dataset_path, args.batch_size, canvas_size=args.canvas_size)
    autocast = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if args.device.startswith("cuda") else nullcontext()

    moments = []
    for images, _ in tqdm(loader, desc="Encoding latents"):
        images = images.to(args.device)
        with autocast:
            # .parameters is the raw 8-channel (mean, logvar) the encoder emits.
            params = vae.encode(images).latent_dist.parameters
        moments.append(params.float().to(torch.float16).cpu())

    moments = torch.cat(moments, dim=0)
    save_latents(args.output, moments, vae.config.scaling_factor,
                 meta={"canvas_size": args.canvas_size, "vae": args.vae})
    print(f"Saved {moments.shape[0]} latents of shape {tuple(moments.shape[1:])} to {args.output}")


if __name__ == "__main__":
    main()
