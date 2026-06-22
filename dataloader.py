import os
import random
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


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
