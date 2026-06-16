"""Shared helpers for the precomputed-VAE-latent training path.

precompute_latents.py writes a cache of the raw 8-channel VAE moments
(mean/logvar) for every image; train.py reads it back and samples a latent
per step. No image is ever encoded during training.
"""
import os
import torch


def save_latents(path: str, moments: torch.Tensor, scaling_factor: float, meta: dict) -> None:
    """Persist latent moments plus the metadata needed to use them."""
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    torch.save({"moments": moments, "scaling_factor": float(scaling_factor), **meta}, path)


def load_latents(path: str) -> tuple[torch.Tensor, float]:
    """Load latent moments and the VAE scaling factor from disk."""
    blob = torch.load(path, map_location="cpu")
    return blob["moments"], blob["scaling_factor"]


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
