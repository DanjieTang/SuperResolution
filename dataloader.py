import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def valid_image_folder(path: str) -> bool:
    # Check if file starts with '._' or ends with '.DS_Store'
    filename = os.path.basename(path)
    if filename.startswith("._") or filename == ".DS_Store": # Stupid MacOS
        return False

    return True

def prepare_dataset(dataset_path: str, batch_size: int, image_size: int = 256, val_split: float = 0.1) -> tuple[DataLoader, DataLoader]:
    """
    Build train and validation dataloaders of high resolution images.

    :param dataset_path: Root folder of an ImageFolder-style dataset.
    :param batch_size: Batch size for both loaders.
    :param image_size: Side length the high resolution images are resized to.
    :param val_split: Fraction of the dataset held out for validation.
    :return: (train_loader, val_loader)
    """
    # Define image transformations for preprocessing
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # Use ImageFolder to automatically label images based on folder names
    dataset = datasets.ImageFolder(root=dataset_path, is_valid_file=valid_image_folder, transform=transform)

    train_size = int((1 - val_split) * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    # Create DataLoaders for training and validation
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader
