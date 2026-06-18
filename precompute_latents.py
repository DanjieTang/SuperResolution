"""
Encode an ImageFolder dataset to frozen-VAE latents once and store them to
disk, so training (train.py --use_vae) never runs the VAE encoder in its loop.

This is the expensive step that previously happened twice per training step;
doing it once offline brings --use_vae back to roughly pixel-mode speed.

One moments file is written per image into a directory tree that mirrors the
source dataset (n01440764/img.JPEG -> <output>/n01440764/img.pt), plus a single
metadata file at the output root. Each image is encoded and written as it is
read, so memory stays flat regardless of dataset size, and re-running skips
images whose moments file already exists (cheap resume after a crash).

Usage:
    python precompute_latents.py \
        --dataset_path ImagenetHighResolution \
        --canvas_size 512 \
        --vae stabilityai/sd-vae-ft-ema \
        --output latent_cache
"""
import argparse
import os
from contextlib import nullcontext

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from tqdm import tqdm

from dataloader import ImagePathDataset, valid_image_folder
from latents import save_cache_meta, save_latent_moment


def parse_args():
    parser = argparse.ArgumentParser(description="Precompute VAE latents for outpainting training")
    parser.add_argument("--dataset_path", type=str, default="ImagenetHighResolution")
    parser.add_argument("--canvas_size", type=int, default=512)
    parser.add_argument("--vae", type=str, default="stabilityai/sd-vae-ft-ema")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--output", type=str, default="latent_cache",
                        help="Cache directory; mirrors the dataset tree, one .pt per image")
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    return parser.parse_args()


def build_pairs(dataset_path: str, output_dir: str) -> list[tuple[str, str]]:
    """Map every (filtered) source image to its mirrored .pt destination path."""
    base = datasets.ImageFolder(root=dataset_path, is_valid_file=valid_image_folder)
    pairs = []
    for src_path, _ in base.samples:
        rel = os.path.relpath(src_path, dataset_path)
        dst_path = os.path.join(output_dir, os.path.splitext(rel)[0] + ".pt")
        pairs.append((src_path, dst_path))
    return pairs


@torch.no_grad()
def main():
    args = parse_args()

    from diffusers import AutoencoderKL
    vae = AutoencoderKL.from_pretrained(args.vae).to(args.device)
    vae.requires_grad_(False)
    vae.eval()

    pairs = build_pairs(args.dataset_path, args.output)
    todo = [(src, dst) for src, dst in pairs if not os.path.exists(dst)]
    print(f"{len(pairs)} images total, {len(pairs) - len(todo)} already cached, {len(todo)} to encode")

    dataset = ImagePathDataset(todo, canvas_size=args.canvas_size)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=8, pin_memory=True)
    autocast = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if args.device.startswith("cuda") else nullcontext()

    for images, dst_paths in tqdm(loader, desc="Encoding latents"):
        images = images.to(args.device)
        with autocast:
            # .parameters is the raw 8-channel (mean, logvar) the encoder emits.
            params = vae.encode(images).latent_dist.parameters
        params = params.float().to(torch.float16).cpu()
        for moment, dst_path in zip(params, dst_paths):
            save_latent_moment(dst_path, moment)

    # Always (re)write metadata so the cache is usable even on a pure resume.
    save_cache_meta(args.output, vae.config.scaling_factor,
                    meta={"canvas_size": args.canvas_size, "vae": args.vae})
    print(f"Cache ready at {args.output} ({len(pairs)} images)")


if __name__ == "__main__":
    main()
