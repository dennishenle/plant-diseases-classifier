"""
Image transformation pipelines.

Before the network can look at a photo, the photo must go through a
pipeline of small changes -- like getting a letter ready to post:
  1. Resize it to the right rough size
  2. Crop it to an exact square
  3. Turn the colours into numbers
  4. Shift those numbers into the range the network understands

We have THREE different pipelines:

  TRAINING   -- adds random changes (flips, zooms, colour tweaks) so
                the network sees each photo slightly differently every
                time and learns the general idea instead of memorising.

  VALIDATION -- no random changes; a fair, consistent test every time.

  INFERENCE  -- same as validation; used when classifying a single new
                photo after training is finished.
"""

from torchvision import transforms

from .config import IMAGENET_MEAN, IMAGENET_STD, INPUT_SIZE, RESIZE_SIZE


def build_train_transforms() -> transforms.Compose:
    """
    Pipeline for TRAINING photos.

    Random changes (called "data augmentation") are added on purpose.
    Imagine showing the same photo from slightly different angles or
    in different lighting -- this teaches the network to handle
    real-world variety instead of memorising exact pictures.
    """
    return transforms.Compose([
        # Randomly zoom in/out and crop to 224x224 pixels.
        transforms.RandomResizedCrop(INPUT_SIZE),

        # Randomly flip the photo left<->right (50 % chance).
        # A diseased leaf looks the same mirrored.
        transforms.RandomHorizontalFlip(),

        # Randomly flip the photo upside down (50 % chance).
        transforms.RandomVerticalFlip(),

        # Randomly tweak brightness, contrast, and saturation.
        # Simulates different lighting or camera settings.
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),

        # Convert the picture into a grid of numbers (a "tensor").
        transforms.ToTensor(),

        # Shift and scale pixel values using ImageNet stats so they
        # are centred around zero. Networks train much better this way.
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def build_val_transforms() -> transforms.Compose:
    """
    Pipeline for VALIDATION and INFERENCE photos.

    No random changes here -- we want a fair, consistent measurement
    every time we check how well the network is doing.
    """
    return transforms.Compose([
        # Resize the shorter side to 256 pixels (keeps aspect ratio).
        transforms.Resize(RESIZE_SIZE),

        # Cut out the central 224x224 square -- avoids edge noise.
        transforms.CenterCrop(INPUT_SIZE),

        # Same tensor conversion and normalisation as training.
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
