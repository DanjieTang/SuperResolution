"""Shared helpers for the precomputed-VAE-latent training path.

precompute_latents.py writes one cache file per image (the raw 8-channel VAE
moments, mean/logvar) into a directory tree that mirrors the source dataset,
plus a single metadata file at the cache root. train.py reads moments back
lazily, one image at a time, and samples a latent per step. No image is ever
encoded during training, and the full cache is never held in RAM.
"""
import os
import torch

# Metadata file written once at the cache root (scaling_factor + provenance),
# kept alongside the per-image moment files but excluded when enumerating them.
CACHE_META_NAME = "cache_meta.pt"


def save_latent_moment(path: str, moment: torch.Tensor) -> None:
    """Persist one image's raw 8-channel VAE moments (mean, logvar) as fp16."""
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    torch.save(moment.to(torch.float16).contiguous(), path)


def load_latent_moment(path: str) -> torch.Tensor:
    """Load one image's moments tensor of shape (8, H, W)."""
    return torch.load(path, map_location="cpu")


def save_cache_meta(cache_dir: str, scaling_factor: float, meta: dict) -> None:
    """Write the VAE scaling factor and provenance once at the cache root."""
    os.makedirs(cache_dir, exist_ok=True)
    torch.save({"scaling_factor": float(scaling_factor), **meta},
               os.path.join(cache_dir, CACHE_META_NAME))


def load_cache_meta(cache_dir: str) -> tuple[float, dict]:
    """Load the scaling factor and remaining metadata from the cache root."""
    blob = torch.load(os.path.join(cache_dir, CACHE_META_NAME), map_location="cpu")
    scaling_factor = blob.pop("scaling_factor")
    return scaling_factor, blob


def sample_latent(moments: torch.Tensor, scaling_factor: float) -> torch.Tensor:
    """
    Draw a latent from the stored Gaussian moments, matching what the VAE's
    DiagonalGaussianDistribution.sample() would produce, scaled into the
    model's working space.

    :param moments: Raw encoder output of shape (B, 8, H, W): mean then logvar.
    :param scaling_factor: VAE latent scaling factor.
    :return: Sampled latent of shape (B, 4, H, W).
    """
    mean, logvar = torch.chunk(moments.float(), 2, dim=1)
    logvar = torch.clamp(logvar, -30.0, 20.0)
    std = torch.exp(0.5 * logvar)
    latent = mean + std * torch.randn_like(mean)
    return latent * scaling_factor
