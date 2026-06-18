import glob
import os
import random
import torch
from PIL import Image
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from latents import CACHE_META_NAME, load_latent_moment


def _canvas_transform(canvas_size: int) -> transforms.Compose:
    """Resize to the canvas and normalize to [-1, 1] (shared by all loaders)."""
    return transforms.Compose([
        transforms.Resize((canvas_size, canvas_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

def valid_image_folder(path: str) -> bool:
    # Check if file starts with '._' or ends with '.DS_Store'
    filename = os.path.basename(path)
    if filename.startswith("._") or filename == ".DS_Store": # Stupid MacOS
        return False

    return True

def prepare_dataset(dataset_path: str, batch_size: int, canvas_size: int = 512, val_split: float = 0.1) -> tuple[DataLoader, DataLoader]:
    """
    Build train and validation dataloaders of full outpainting canvases.

    :param dataset_path: Root folder of an ImageFolder-style dataset.
    :param batch_size: Batch size for both loaders.
    :param canvas_size: Side length of the full canvas the model learns to fill.
    :param val_split: Fraction of the dataset held out for validation.
    :return: (train_loader, val_loader)
    """
    # Define image transformations for preprocessing
    transform = _canvas_transform(canvas_size)

    # Use ImageFolder to automatically label images based on folder names
    dataset = datasets.ImageFolder(root=dataset_path, is_valid_file=valid_image_folder, transform=transform)

    train_size = int((1 - val_split) * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    # Create DataLoaders for training and validation
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=8, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=4, pin_memory=True, persistent_workers=True)

    return train_loader, val_loader

class ImagePathDataset(torch.utils.data.Dataset):
    """
    Yields (transformed_image, destination_path) for the precompute pass.
    Carrying the per-image output path through the loader lets the encoder
    write each moment to its own mirrored cache file, so nothing accumulates
    in RAM (default collate returns the paths as a list of strings).
    """

    def __init__(self, pairs: list[tuple[str, str]], canvas_size: int = 512):
        self.pairs = pairs
        self.transform = _canvas_transform(canvas_size)

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, str]:
        src_path, dst_path = self.pairs[index]
        image = Image.open(src_path).convert("RGB")
        return self.transform(image), dst_path


class LatentFolderDataset(torch.utils.data.Dataset):
    """
    Lazily loads one precomputed moments file per access, so training never
    holds the whole cache in memory. Labels are unused (outpainting is
    unconditional) and returned as a constant to match the (x, _) loop.
    """

    def __init__(self, files: list[str]):
        self.files = files

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        return load_latent_moment(self.files[index]), 0


def list_latent_files(latent_dir: str) -> list[str]:
    """All per-image moment files under the cache, excluding the metadata file."""
    files = glob.glob(os.path.join(latent_dir, "**", "*.pt"), recursive=True)
    return sorted(p for p in files if os.path.basename(p) != CACHE_META_NAME)


def prepare_raw_loader(dataset_path: str, batch_size: int, canvas_size: int = 512) -> DataLoader:
    """
    A single, deterministically-ordered loader over the full dataset, used to
    encode every image to VAE latents exactly once (see precompute_latents).
    """
    transform = _canvas_transform(canvas_size)
    dataset = datasets.ImageFolder(root=dataset_path, is_valid_file=valid_image_folder, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False,
                      num_workers=8, pin_memory=True)

def prepare_latent_loaders(latent_dir: str, batch_size: int, val_split: float = 0.1) -> tuple[DataLoader, DataLoader]:
    """
    Build train/val loaders over a directory of precomputed per-image moment
    files (the raw 8-channel mean/logvar tensors the encoder produced). Files
    are read lazily by the loader workers; no images are read or encoded, and
    the full cache is never resident in memory.
    """
    files = list_latent_files(latent_dir)
    if not files:
        raise FileNotFoundError(f"No latent .pt files found under '{latent_dir}'")
    dataset = LatentFolderDataset(files)

    train_size = int((1 - val_split) * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=8, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=4, pin_memory=True, persistent_workers=True)

    return train_loader, val_loader

def sample_known_region_mask(batch_size: int, canvas_size: int, device: str, min_known_fraction: float = 0.5) -> torch.Tensor:
    """
    Sample a random known region per image: 1 where pixels are given, 0 where
    the model must outpaint.

    Each side of the known rectangle is between min_known_fraction and 1.0 of
    the canvas, so with the default 0.5 the canvas is at most a 2x expansion
    per dimension. Random placement of the rectangle covers every direction
    (left, right, up, down, corners) with one model.

    :param batch_size: Number of masks to sample.
    :param canvas_size: Side length of the canvas.
    :param device: Device to create the mask on.
    :param min_known_fraction: Minimum known-side length as a fraction of the canvas.
    :return: Mask tensor of shape (batch_size, 1, canvas_size, canvas_size).
    """
    mask = torch.zeros(batch_size, 1, canvas_size, canvas_size, device=device)
    minimum_side = int(canvas_size * min_known_fraction)

    for i in range(batch_size):
        known_height = random.randint(minimum_side, canvas_size)
        known_width = random.randint(minimum_side, canvas_size)
        top = random.randint(0, canvas_size - known_height)
        left = random.randint(0, canvas_size - known_width)
        mask[i, :, top:top + known_height, left:left + known_width] = 1.0

    return mask
