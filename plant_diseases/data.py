"""
Dataset loading and batching.

Photos are stored on disk in an "ImageFolder" layout:

    data_dir/
      train/
        Apple___Apple_scab/
          photo1.jpg
          photo2.jpg
        Apple___healthy/
          ...
      valid/
        Apple___Apple_scab/
          ...

Each subfolder name IS the class label. PyTorch reads this structure
and automatically assigns a number (0, 1, 2, ...) to each class.

A DataLoader wraps the dataset and serves photos to the network in
"batches" -- groups of images processed together. Batching is much
faster than feeding one image at a time because GPUs are built to
do many calculations in parallel.
"""

from dataclasses import dataclass
from pathlib import Path

from torch.utils.data import DataLoader
from torchvision import datasets

from .transforms import build_train_transforms, build_val_transforms


@dataclass
class DataBundle:
    """
    Holds everything the trainer needs from the data side:
      - train_loader : serves batches of training photos
      - val_loader   : serves batches of validation photos
      - classes      : list of disease names (e.g. "Apple___Apple_scab")
    """
    train_loader: DataLoader
    val_loader: DataLoader
    classes: list[str]


def build_dataloaders(
    data_dir: Path,
    batch_size: int = 32,
    num_workers: int = 4,
) -> DataBundle:
    """
    Load photos from disk and wrap them in DataLoaders.

    We have two sets of photos:
      TRAIN -- photos the model LEARNS from.
      VALID -- photos used to CHECK how well the model is doing.
               The model never learns from these -- it's like a
               surprise test to see if it truly understands.

    Args:
        data_dir:    Root folder containing train/ and valid/ subfolders.
        batch_size:  How many images per batch (default 32).
        num_workers: Background processes that load images in parallel
                     while the GPU is busy computing.

    Returns:
        A DataBundle with both loaders and the class name list.
    """
    # ImageFolder reads the subfolder names as class labels.
    train_ds = datasets.ImageFolder(
        data_dir / "train", transform=build_train_transforms()
    )
    val_ds = datasets.ImageFolder(
        data_dir / "valid", transform=build_val_transforms()
    )

    # A DataLoader feeds images to the network in batches.
    # shuffle=True randomises the order each epoch so the model can't
    # cheat by memorising the sequence.
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,  # speeds up RAM-to-GPU transfer
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,  # keep validation order fixed for consistency
        num_workers=num_workers,
        pin_memory=True,
    )

    return DataBundle(
        train_loader=train_loader,
        val_loader=val_loader,
        classes=list(train_ds.classes),
    )
