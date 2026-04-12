"""
Neural network construction and checkpoint management.

The network has two parts (imagine a person):

  BACKBONE ("eyes")
      Already trained on 1.2 million general photos. It knows how to
      spot edges, textures, shapes, and colours in any image.

  CLASSIFIER HEAD ("brain")
      A small, fresh layer we add on top. It takes the visual features
      the backbone extracted and decides which of the 38 plant disease
      classes the photo belongs to.

Training happens in two phases:
  Phase 1 -- The backbone is FROZEN (locked). Only the head learns.
             This is fast and safe because we keep the backbone's
             existing knowledge intact.
  Phase 2 -- The backbone is UNFROZEN. The whole network learns
             together with a very small learning rate, making gentle
             adjustments so the "eyes" specialise in plant leaves.
"""

from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import models

from .config import DROPOUT_RATE


# ---------------------------------------------------------------------------
# Checkpoint -- a small container that holds everything we save to disk
# ---------------------------------------------------------------------------

@dataclass
class Checkpoint:
    """
    Everything stored inside a saved model file.

    Think of it like a zip file that bundles:
      - epoch       : which round of training produced this
      - model_state : all the numbers the network learned (its "memory")
      - val_acc     : how accurate it was on unseen photos
      - classes     : the list of 38 disease names
    """
    epoch: int
    model_state: dict
    val_acc: float
    classes: list[str]


# ---------------------------------------------------------------------------
# PlantDiseaseModel -- builds, manages, and saves the neural network
# ---------------------------------------------------------------------------

class PlantDiseaseModel:
    """
    Wraps an EfficientNet-B0 network customised for plant disease photos.

    This class handles everything about the network itself:
      - Building it from a pretrained starting point
      - Freezing / unfreezing the backbone
      - Saving the learned weights to a file (checkpoint)
      - Loading weights back from a file

    Usage:
        # During training:
        pdm = PlantDiseaseModel(num_classes=38, freeze_backbone=True)
        pdm.to(device)
        ...
        pdm.unfreeze_backbone()
        ...
        pdm.save_checkpoint(path, epoch=5, val_acc=0.94, classes=[...])

        # During inference:
        pdm = PlantDiseaseModel.from_checkpoint(path, device)
    """

    def __init__(self, num_classes: int, freeze_backbone: bool = True) -> None:
        """
        Build a fresh EfficientNet-B0 with a custom classifier head.

        Args:
            num_classes:     How many disease categories to predict (38).
            freeze_backbone: If True, lock the pretrained "eyes" so only
                             the new "brain" (head) learns at first.
        """
        # Load EfficientNet-B0 with weights Google trained on ImageNet.
        self.net = models.efficientnet_b0(
            weights=models.EfficientNet_B0_Weights.DEFAULT
        )

        if freeze_backbone:
            # "Freezing" = tell PyTorch: do NOT update these weights.
            # We keep the backbone's knowledge safe while the new head
            # learns the basics of plant diseases.
            for param in self.net.parameters():
                param.requires_grad = False

        # How many numbers the backbone outputs for each image.
        in_features = self.net.classifier[1].in_features

        # Replace the original 1000-class head with our own:
        #   Dropout -- randomly turns off 30 % of connections during
        #              training so the network doesn't rely on any
        #              single feature too much (prevents memorising).
        #   Linear  -- maps backbone features to one score per class.
        #              The class with the highest score wins.
        self.net.classifier = nn.Sequential(
            nn.Dropout(p=DROPOUT_RATE, inplace=True),
            nn.Linear(in_features, num_classes),
        )

    # -- Device management ---------------------------------------------------

    def to(self, device: torch.device) -> "PlantDiseaseModel":
        """Move the network to a specific device (GPU or CPU)."""
        self.net.to(device)
        return self

    # -- Freeze / unfreeze ---------------------------------------------------

    def unfreeze_backbone(self) -> None:
        """
        Unlock every weight in the network so the backbone can learn too.

        Call this after the head has had a few epochs to settle in.
        Use a smaller learning rate afterwards so the backbone makes
        only gentle adjustments -- it already knows a lot.
        """
        for param in self.net.parameters():
            param.requires_grad = True

    def trainable_parameters(self):
        """Return only the parameters that are currently allowed to learn."""
        return filter(lambda p: p.requires_grad, self.net.parameters())

    # -- Training / evaluation mode ------------------------------------------

    def train(self) -> None:
        """Switch to training mode (Dropout is active)."""
        self.net.train()

    def eval(self) -> None:
        """Switch to evaluation mode (Dropout is off)."""
        self.net.eval()

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Run a batch of images through the network and return scores."""
        return self.net(x)

    def state_dict(self) -> dict:
        """Return the raw weight dictionary (used by PyTorch internals)."""
        return self.net.state_dict()

    # -- Checkpoint save / load ----------------------------------------------

    def save_checkpoint(
        self, path: Path, epoch: int, val_acc: float, classes: list[str]
    ) -> None:
        """
        Save the network's learned weights and metadata to a file.

        Think of it like saving your game -- if the computer crashes
        you can pick up right where you left off.
        """
        torch.save(
            {
                "epoch": epoch,
                "model_state": self.net.state_dict(),
                "val_acc": val_acc,
                "classes": classes,
            },
            path,
        )

    @classmethod
    def from_checkpoint(
        cls, path: Path, device: torch.device
    ) -> tuple["PlantDiseaseModel", Checkpoint]:
        """
        Rebuild a trained network from a saved checkpoint file.

        This is used after training is done, when you want to classify
        new photos. It:
          1. Reads the saved file
          2. Builds the same network architecture
          3. Pours the saved weights back in
          4. Switches to evaluation mode (no learning, just predicting)

        Returns:
            A tuple of (model, checkpoint_info).
        """
        # Read the checkpoint file from disk.
        raw = torch.load(path, map_location=device, weights_only=False)

        checkpoint = Checkpoint(
            epoch=raw["epoch"],
            model_state=raw["model_state"],
            val_acc=raw["val_acc"],
            classes=raw["classes"],
        )

        # Build a fresh network with random weights (we'll overwrite them).
        model = cls(num_classes=len(checkpoint.classes), freeze_backbone=False)

        # Pour the saved weights into the network.
        model.net.load_state_dict(checkpoint.model_state)

        # Move to the chosen device and switch to prediction mode.
        model.to(device)
        model.eval()

        return model, checkpoint
