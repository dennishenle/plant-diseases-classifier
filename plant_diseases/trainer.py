"""
Training loop orchestrator.

Training a neural network is like teaching a child to recognise
plant diseases by showing them thousands of photos:

  1. SHOW    -- feed a batch of photos through the network
  2. GRADE   -- measure how wrong the guesses were (the "loss")
  3. EXPLAIN -- figure out which weights caused the mistakes
  4. ADJUST  -- nudge the weights to do better next time
  5. REPEAT  -- do this for every batch, then start a new round (epoch)

The Trainer class manages this entire cycle, including:
  - The optimizer (decides HOW to adjust the weights)
  - The scheduler (slowly reduces the step size over time)
  - The scaler (uses lower-precision math on NVIDIA GPUs for speed)
  - The two-phase freeze/unfreeze strategy
"""

import time
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from .model import PlantDiseaseModel


# ---------------------------------------------------------------------------
# EpochResult -- a small bag that holds the numbers from one epoch
# ---------------------------------------------------------------------------

@dataclass
class EpochResult:
    """
    The report card for a single epoch.

    epoch      : which round this was (1, 2, 3, ...)
    train_loss : average "wrongness" on training photos (lower = better)
    train_acc  : fraction of training photos guessed correctly (0.0-1.0)
    val_loss   : average "wrongness" on unseen photos
    val_acc    : fraction of unseen photos guessed correctly
    elapsed    : how many seconds this epoch took
    """
    epoch: int
    train_loss: float
    train_acc: float
    val_loss: float
    val_acc: float
    elapsed: float


# ---------------------------------------------------------------------------
# Trainer -- runs the full training loop
# ---------------------------------------------------------------------------

class Trainer:
    """
    Orchestrates the entire training process.

    It owns the model, the optimizer, the learning rate scheduler, and
    the optional mixed-precision scaler. Call ``run()`` to execute all
    epochs, or call ``train_one_epoch()`` and ``evaluate()`` individually
    for finer control.

    Args:
        model:         The PlantDiseaseModel to train.
        device:        Where to run calculations (GPU or CPU).
        lr:            Learning rate -- how big a step to take when
                       adjusting weights. Too large = unstable.
                       Too small = painfully slow.
        weight_decay:  A small penalty on large weights to keep the model
                       from becoming overly complex (fights overfitting).
        total_epochs:  Total number of training rounds.
        freeze_epochs: How many rounds to train only the head before
                       unlocking the backbone.
        use_amp:       Whether to use Automatic Mixed Precision on CUDA
                       (makes training ~2x faster with almost no quality loss).
    """

    def __init__(
        self,
        model: PlantDiseaseModel,
        device: torch.device,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        total_epochs: int = 20,
        freeze_epochs: int = 5,
        use_amp: bool = True,
    ) -> None:
        self.model = model
        self.device = device
        self.freeze_epochs = freeze_epochs
        self.lr = lr

        # LOSS FUNCTION: measures how wrong the model's guesses are.
        # label_smoothing=0.1 means we aim for 90 % confidence instead
        # of 100 %. This prevents the model from becoming over-confident
        # and usually improves accuracy on new images.
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

        # OPTIMIZER: AdamW decides how to change the weights after each
        # batch. It looks at the gradients (the "direction to improve")
        # and takes a smart, adaptive step.
        # We only pass the currently trainable parameters (the head at
        # first, then everything after unfreezing).
        self.optimizer = optim.AdamW(
            model.trainable_parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )

        # LEARNING RATE SCHEDULER: slowly reduces the learning rate
        # following a cosine curve -- big steps early, tiny steps late.
        # This helps the model "settle" into a good solution near the end.
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=total_epochs
        )

        # GRAD SCALER: works with float16 math on CUDA GPUs to keep
        # training numerically stable while running ~2x faster.
        self.scaler = (
            torch.amp.GradScaler()
            if use_amp and device.type == "cuda"
            else None
        )

    # -- Single epoch: training ----------------------------------------------

    def train_one_epoch(self, loader: DataLoader) -> tuple[float, float]:
        """
        Show every training image to the model once and update weights.

        One "epoch" = the model has seen every training photo once.
        For each batch the cycle is:
          1. Forward pass  -- model looks at images and makes predictions
          2. Loss          -- measure how wrong the predictions were
          3. Backward pass -- calculate how each weight contributed to error
          4. Optimizer     -- nudge weights in the direction that helps

        Returns:
            (average_loss, accuracy) over the entire training set.
        """
        self.model.train()

        running_loss = 0.0  # total loss accumulated over all batches
        correct = 0         # number of correct predictions
        total = 0           # total images seen

        for images, labels in tqdm(loader, desc="  train", leave=False):
            # Move data to the same device as the model.
            images, labels = images.to(self.device), labels.to(self.device)

            # Clear leftover gradients from the previous batch.
            self.optimizer.zero_grad()

            # autocast uses float16 math on CUDA for ~2x speed.
            with torch.autocast(
                device_type=self.device.type,
                enabled=self.scaler is not None,
            ):
                # FORWARD: feed images through the network.
                outputs = self.model(images)

                # LOSS: how wrong were the predictions?
                loss = self.criterion(outputs, labels)

            # BACKWARD + UPDATE: compute gradients and adjust weights.
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()

            # Accumulate stats for the end-of-epoch report.
            running_loss += loss.item() * images.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += images.size(0)

        return running_loss / total, correct / total

    # -- Single epoch: evaluation --------------------------------------------

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> tuple[float, float]:
        """
        Measure how well the model does on photos it has NEVER trained on.

        No weights are updated here -- this is purely a report card.
        If training accuracy is high but validation accuracy is low,
        the model has "overfit" (memorised instead of learned).

        Returns:
            (average_loss, accuracy) over the entire validation set.
        """
        self.model.eval()

        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in tqdm(loader, desc="  valid", leave=False):
            images, labels = images.to(self.device), labels.to(self.device)

            with torch.autocast(
                device_type=self.device.type,
                enabled=self.device.type != "cpu",
            ):
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += images.size(0)

        return running_loss / total, correct / total

    # -- Phase transition: unfreeze ------------------------------------------

    def maybe_unfreeze(self, epoch: int) -> None:
        """
        At the right epoch, unlock the backbone and lower the learning rate.

        After the head has had a few epochs to learn the basics, we let
        the backbone learn too -- but with a 10x smaller learning rate
        so it only makes gentle adjustments to its existing knowledge.
        """
        if epoch != self.freeze_epochs + 1:
            return

        print("Unfreezing backbone -- reducing LR")
        self.model.unfreeze_backbone()

        for group in self.optimizer.param_groups:
            group["lr"] = self.lr * 0.1

    # -- Full training run ---------------------------------------------------

    def run(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
    ):
        """
        Execute the complete training loop, yielding results after each epoch.

        This is a generator -- it produces one EpochResult at a time so
        the caller can save checkpoints and log history as training
        progresses, not just at the very end.

        For each epoch:
          1. Check if it's time to unfreeze the backbone
          2. Train on all training images
          3. Evaluate on all validation images
          4. Step the learning rate scheduler
          5. Yield the results to the caller

        Yields:
            An EpochResult after each epoch completes.
        """
        total_epochs = self.scheduler.T_max

        for epoch in range(1, total_epochs + 1):
            self.maybe_unfreeze(epoch)

            t0 = time.time()
            train_loss, train_acc = self.train_one_epoch(train_loader)
            val_loss, val_acc = self.evaluate(val_loader)
            self.scheduler.step()
            elapsed = time.time() - t0

            result = EpochResult(
                epoch=epoch,
                train_loss=train_loss,
                train_acc=train_acc,
                val_loss=val_loss,
                val_acc=val_acc,
                elapsed=elapsed,
            )

            print(
                f"Epoch {epoch:3d}/{total_epochs} | "
                f"train loss {train_loss:.4f} acc {train_acc:.4f} | "
                f"val loss {val_loss:.4f} acc {val_acc:.4f} | "
                f"{elapsed:.0f}s"
            )

            yield result
