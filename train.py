"""
Plant disease classification using EfficientNet-B0 transfer learning.
Dataset: New Plant Diseases Dataset (Augmented) — 38 classes.

HOW THIS SCRIPT WORKS (big picture)
-------------------------------------
Imagine you want to teach a child to recognise plant diseases from photos.
You would show them thousands of pictures and tell them the correct answer
each time ("this is apple scab", "this is healthy corn", ...).
After many rounds of looking and correcting, the child gets really good at it.

That is exactly what this script does — but instead of a child, we train a
neural network (a program that loosely mimics how the brain works).

The network sees a photo → makes a guess → we tell it how wrong it was →
it adjusts itself to do better next time. Repeat many thousands of times.

We use a trick called "transfer learning": instead of starting from scratch,
we start from a network that already knows how to look at photos in general
(it was trained on 1.2 million images from the internet). We then teach it
the specific plant disease task on top of that knowledge.
"""

# ---------------------------------------------------------------------------
# Imports — loading tools we need from external libraries
# ---------------------------------------------------------------------------

import argparse   # reads command-line flags like --epochs 20
import csv        # writes spreadsheet-style log files
import json       # writes human-readable data files
import time       # measures how long each epoch takes
from pathlib import Path  # handles file and folder paths nicely

import torch                        # the main deep-learning library
import torch.nn as nn               # building blocks for neural networks
import torch.optim as optim         # algorithms that adjust the network weights
from torch.utils.data import DataLoader          # feeds images to the network in batches
from torchvision import datasets, models, transforms  # image tools + pretrained models
from tqdm import tqdm               # draws a progress bar in the terminal


# ---------------------------------------------------------------------------
# Constants — numbers used for image normalisation
# ---------------------------------------------------------------------------

# Every pixel in an image has a colour value between 0 and 255.
# Neural networks learn faster when numbers are small and centred around zero.
# These two lists are the average colour and the spread of colours across
# 1.2 million ImageNet photos. We reuse them because our pretrained model
# was trained with these exact values.
IMAGENET_MEAN = [0.485, 0.456, 0.406]  # average Red, Green, Blue
IMAGENET_STD  = [0.229, 0.224, 0.225]  # spread  Red, Green, Blue


# ---------------------------------------------------------------------------
# Step 1 — Prepare the images (DataLoaders)
# ---------------------------------------------------------------------------

def build_dataloaders(data_dir: Path, batch_size: int, num_workers: int):
    """
    Loads photos from disk and prepares them for the network.

    We have two sets of photos:
      - train/  : photos the model LEARNS from
      - valid/  : photos we use to CHECK how well the model is doing
                  (the model never learns from these — it's like a surprise test)

    Each photo goes through a "pipeline" of small changes before the network
    sees it.  For training photos we add random changes on purpose so the
    model doesn't just memorise the exact images but learns the general idea.
    """

    # --- Transformations applied to TRAINING images ---
    # We randomly alter each photo a little bit every time we show it.
    # This is called "data augmentation" and it's like showing the same
    # photo from slightly different angles or in different lighting so the
    # model learns to handle real-world variety.
    train_transforms = transforms.Compose([
        # Randomly zoom in/out and crop to 224×224 pixels.
        # 224×224 is the size EfficientNet-B0 expects as input.
        transforms.RandomResizedCrop(224),

        # Randomly flip the photo left↔right (50% chance).
        # A diseased leaf looks the same mirrored.
        transforms.RandomHorizontalFlip(),

        # Randomly flip the photo upside down (50% chance).
        transforms.RandomVerticalFlip(),

        # Randomly tweak brightness, contrast, and saturation a little.
        # This simulates different lighting conditions or camera settings.
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),

        # Convert the PIL image object into a PyTorch tensor
        # (a grid of numbers the network can process).
        transforms.ToTensor(),

        # Shift and scale the pixel values using the ImageNet stats above
        # so they are centred around zero. Networks train much better this way.
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    # --- Transformations applied to VALIDATION images ---
    # No random changes here — we want a fair, consistent test every time.
    val_transforms = transforms.Compose([
        # Resize the shorter side to 256 pixels (keeps the aspect ratio).
        transforms.Resize(256),

        # Cut out the central 224×224 square — avoids edge noise.
        transforms.CenterCrop(224),

        # Same conversion to tensor and normalisation as above.
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    # ImageFolder reads a folder where each sub-folder is a class name.
    # It automatically assigns a number (0, 1, 2, …) to each class.
    train_ds = datasets.ImageFolder(data_dir / "train", transform=train_transforms)
    val_ds   = datasets.ImageFolder(data_dir / "valid", transform=val_transforms)

    # A DataLoader takes a dataset and serves it to the network in "batches".
    # Instead of feeding one image at a time (slow), we feed 32 at once.
    # shuffle=True randomises the order each epoch so the model can't learn
    # by memorising the sequence of images.
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,   # how many images per batch
        shuffle=True,            # mix up the order every epoch
        num_workers=num_workers, # helper processes that load images in parallel
        pin_memory=True,         # speeds up transfer from RAM to GPU
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,           # keep validation order fixed for consistency
        num_workers=num_workers,
        pin_memory=True,
    )

    # Return both loaders and the list of class names (e.g. "Apple___Apple_scab")
    return train_loader, val_loader, train_ds.classes


# ---------------------------------------------------------------------------
# Step 2 — Build the neural network (Model)
# ---------------------------------------------------------------------------

def build_model(num_classes: int, freeze_backbone: bool) -> nn.Module:
    """
    Creates the neural network we will train.

    We use EfficientNet-B0, which has already been trained on millions of
    general photos (ImageNet). Think of it as a very experienced eye doctor
    who knows how to look at images — we just need to teach them the
    specific task of spotting plant diseases.

    The network has two parts:
      1. Backbone — the "eyes" that extract visual features from a photo
                    (edges, textures, shapes, colours, …)
      2. Classifier head — the "brain" that takes those features and decides
                           which of the 38 disease classes the photo belongs to

    We replace the original head (designed for 1000 ImageNet classes) with a
    fresh one designed for our 38 plant disease classes.
    """

    # Load EfficientNet-B0 with pretrained weights downloaded automatically.
    # This network was trained by Google on 1.2 million photos.
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)

    if freeze_backbone:
        # "Freezing" means we tell PyTorch: do NOT change these weights during
        # training. We keep the backbone's knowledge intact while we only teach
        # the new head. This is faster and avoids accidentally destroying the
        # pretrained features in the early chaotic stages of training.
        for param in model.parameters():
            param.requires_grad = False  # requires_grad=False → don't update this

    # Find out how many numbers the backbone produces for each image.
    # The head needs to accept exactly that many numbers as input.
    in_features = model.classifier[1].in_features

    # Replace the classifier head with our own:
    #   Dropout  → randomly turns off 30% of connections during training.
    #              This forces the network to not rely too heavily on any
    #              single feature, which reduces overfitting (memorising
    #              training data instead of learning general patterns).
    #   Linear   → a simple layer that maps the features to 38 scores,
    #              one score per disease class. The class with the highest
    #              score is the model's prediction.
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3, inplace=True),
        nn.Linear(in_features, num_classes),
    )

    return model


# ---------------------------------------------------------------------------
# Step 3a — Train for one epoch (one full pass over all training images)
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, criterion, optimizer, device, scaler):
    """
    Shows every training image to the model exactly once and updates the
    model's weights after each batch.

    One "epoch" = the model has seen every training photo once.
    We usually repeat this process many times (20–30 epochs) because a
    single pass is not enough to learn well.

    After each batch the learning cycle is:
      1. Forward pass  — model looks at the images and makes predictions
      2. Loss          — we measure how wrong the predictions were
      3. Backward pass — we calculate how each weight contributed to the error
      4. Optimizer     — we nudge the weights in the direction that reduces error
    """

    # Put the model in "training mode". Some layers (like Dropout) behave
    # differently during training vs evaluation, so we must tell PyTorch
    # which mode we are in.
    model.train()

    running_loss = 0.0  # total loss accumulated over all batches
    correct      = 0    # number of correct predictions so far
    total        = 0    # total number of images seen so far

    # tqdm wraps the loader and draws a live progress bar in the terminal.
    for images, labels in tqdm(loader, desc="  train", leave=False):
        # Move the images and correct answers to the GPU (or CPU if no GPU).
        # The model lives on the same device, so everything must match.
        images, labels = images.to(device), labels.to(device)

        # Clear the gradients left over from the previous batch.
        # If we didn't do this, gradients would keep piling up and the
        # weight updates would be wrong.
        optimizer.zero_grad()

        # autocast uses lower precision math (float16) on CUDA GPUs.
        # This makes calculations about 2× faster with almost no accuracy loss.
        with torch.autocast(device_type=device.type, enabled=scaler is not None):
            # FORWARD PASS: feed images through the network.
            # outputs is a tensor of shape [batch_size, 38] —
            # 38 raw scores (called "logits") for each image.
            outputs = model(images)

            # LOSS: measure how wrong the predictions are.
            # CrossEntropyLoss converts the raw scores to probabilities and
            # then penalises the model for assigning a low probability to
            # the correct class.
            # A loss of 0.0 = perfect. A loss of 3.6 = random guessing on
            # 38 classes (because ln(38) ≈ 3.6).
            loss = criterion(outputs, labels)

        # BACKWARD PASS: calculate how much each weight contributed to the loss.
        # This uses calculus (the chain rule / backpropagation) under the hood.
        # The scaler handles the float16 precision safely on CUDA.
        if scaler is not None:
            scaler.scale(loss).backward()  # compute gradients
            scaler.step(optimizer)         # update weights
            scaler.update()                # adjust the scaler for next step
        else:
            loss.backward()   # compute gradients (CPU / MPS path)
            optimizer.step()  # update weights

        # Accumulate statistics for reporting at the end of the epoch.
        # loss.item() gives us a plain Python float from the tensor.
        # images.size(0) is the number of images in this batch.
        running_loss += loss.item() * images.size(0)

        # argmax(1) picks the class with the highest score for each image.
        # We compare that to the true label and count correct predictions.
        correct += (outputs.argmax(1) == labels).sum().item()
        total   += images.size(0)

    # Return average loss and accuracy (fraction of correct predictions).
    return running_loss / total, correct / total


# ---------------------------------------------------------------------------
# Step 3b — Evaluate on validation images (no learning, just measuring)
# ---------------------------------------------------------------------------

@torch.no_grad()  # disables gradient tracking — saves memory and speeds things up
def evaluate(model, loader, criterion, device):
    """
    Runs the model on the validation set to measure how well it generalises
    to photos it has NEVER been trained on.

    We do NOT update the weights here. This is purely a report card:
    "given everything the model has learned so far, how accurate is it on
    new, unseen images?"

    If training accuracy is high but validation accuracy is low, the model
    has "overfit" — it memorised the training photos instead of learning
    the general concept. Good training / augmentation techniques prevent this.
    """

    # Evaluation mode: turns off Dropout (we want deterministic results).
    model.eval()

    running_loss = 0.0
    correct      = 0
    total        = 0

    for images, labels in tqdm(loader, desc="  valid", leave=False):
        images, labels = images.to(device), labels.to(device)

        # Same forward pass and loss calculation as in training,
        # but no backward pass or optimizer step — we are just observing.
        with torch.autocast(device_type=device.type, enabled=(device.type != "cpu")):
            outputs = model(images)
            loss    = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        correct      += (outputs.argmax(1) == labels).sum().item()
        total        += images.size(0)

    return running_loss / total, correct / total


# ---------------------------------------------------------------------------
# Command-line arguments — let the user customise training without editing code
# ---------------------------------------------------------------------------

def parse_args():
    """
    Reads options passed when running the script from the terminal.
    Example: uv run train.py --epochs 30 --batch-size 64
    If you don't pass an option, the default value is used.
    """
    parser = argparse.ArgumentParser(description="Train plant disease classifier")

    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("New Plant Diseases Dataset(Augmented) copy"),
        help="Root directory containing train/ and valid/ folders",
    )
    # An "epoch" is one full pass over ALL training images.
    parser.add_argument("--epochs", type=int, default=20)

    # Batch size = how many images the model sees at once before updating weights.
    # Larger batches = faster but needs more GPU memory.
    parser.add_argument("--batch-size", type=int, default=32)

    # Learning rate = how big a step to take when adjusting weights.
    # Too large → training is unstable. Too small → training is very slow.
    parser.add_argument("--lr", type=float, default=1e-3)

    # Weight decay = a small penalty added to large weights to keep the model
    # from becoming too complex (another way to fight overfitting).
    parser.add_argument("--weight-decay", type=float, default=1e-4)

    # Number of background processes that load images while the GPU is busy.
    parser.add_argument("--num-workers", type=int, default=4)

    # How many epochs to train ONLY the new head before unfreezing the backbone.
    parser.add_argument(
        "--freeze-epochs", type=int, default=5,
        help="Train only the classifier head for this many epochs before unfreezing",
    )

    parser.add_argument(
        "--output-dir", type=Path, default=Path("output"),
        help="Directory to save model checkpoints and logs",
    )
    parser.add_argument(
        "--no-amp", action="store_true",
        help="Disable automatic mixed precision (AMP)",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main — puts everything together and runs the training loop
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # Create the output folder if it doesn't exist yet.
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Choose the computing device.
    # CUDA = NVIDIA GPU (fastest), MPS = Apple Silicon GPU, CPU = slowest fallback.
    # Training on a GPU can be 10–50× faster than on a CPU.
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Device: {device}")

    # Load the images and get back the two DataLoaders and the class names.
    train_loader, val_loader, classes = build_dataloaders(
        args.data_dir, args.batch_size, args.num_workers
    )
    print(f"Classes: {len(classes)}  |  Train batches: {len(train_loader)}  |  Val batches: {len(val_loader)}")

    # Save a file that maps numbers to disease names, e.g. {0: "Apple___Apple_scab"}.
    # We need this later when we want to show human-readable predictions.
    class_map = {i: name for i, name in enumerate(classes)}
    with open(args.output_dir / "class_map.json", "w") as f:
        json.dump(class_map, f, indent=2)

    # Build the network and move it to the chosen device (GPU or CPU).
    # We start with freeze_backbone=True so only the new head is trained first.
    model = build_model(len(classes), freeze_backbone=True).to(device)

    # LOSS FUNCTION: CrossEntropyLoss measures how wrong the model's guesses are.
    # label_smoothing=0.1 means instead of targeting a perfect 100% confidence,
    # we aim for 90% — this prevents the model from becoming over-confident
    # and usually improves accuracy on new images.
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # OPTIMIZER: AdamW decides how to change the weights after each batch.
    # It looks at the gradients (the "direction" to improve) and takes a
    # smart, adaptive step. filter(...) passes only the unfrozen weights.
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # LEARNING RATE SCHEDULER: slowly reduces the learning rate over time.
    # CosineAnnealingLR follows a cosine curve — starts at lr, smoothly
    # decreases to near zero by the last epoch. Smaller steps near the end
    # help the model "settle" into a good solution rather than jumping around.
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # GradScaler works with autocast (float16) on CUDA to keep the training
    # numerically stable. Set to None when not using CUDA or when AMP is off.
    scaler = torch.amp.GradScaler() if (not args.no_amp and device.type == "cuda") else None

    best_val_acc = 0.0  # track the highest validation accuracy seen so far
    history = []        # list of dicts, one per epoch, for the JSON log

    # Open a CSV file for live logging — one row is written after every epoch.
    # This lets you run plot_history.py while training is still running.
    csv_path = args.output_dir / "history.csv"
    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.DictWriter(
        csv_file, fieldnames=["epoch", "train_loss", "train_acc", "val_loss", "val_acc"]
    )
    csv_writer.writeheader()

    # -----------------------------------------------------------------------
    # THE MAIN TRAINING LOOP — repeat for the requested number of epochs
    # -----------------------------------------------------------------------
    for epoch in range(1, args.epochs + 1):

        # After the freeze phase is over, unlock the backbone so its weights
        # can also be updated. We use a much smaller learning rate (lr × 0.1)
        # to make only tiny adjustments — the backbone is already good, we
        # just want to fine-tune it gently for plant photos.
        if epoch == args.freeze_epochs + 1:
            print("Unfreezing backbone — reducing LR")
            for param in model.parameters():
                param.requires_grad = True          # unlock all weights
            for group in optimizer.param_groups:
                group["lr"] = args.lr * 0.1         # 10× smaller learning rate

        t0 = time.time()  # record start time so we can report seconds per epoch

        # --- Train on all training images ---
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, scaler
        )

        # --- Evaluate on all validation images ---
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        # Advance the learning rate scheduler by one step.
        scheduler.step()

        elapsed = time.time() - t0

        # Print a one-line summary for this epoch.
        print(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"train loss {train_loss:.4f} acc {train_acc:.4f} | "
            f"val loss {val_loss:.4f} acc {val_acc:.4f} | "
            f"{elapsed:.0f}s"
        )

        # Record this epoch's results and flush them to disk immediately
        # so plot_history.py can read them without waiting for training to end.
        row = {
            "epoch": epoch,
            "train_loss": train_loss, "train_acc": train_acc,
            "val_loss": val_loss,     "val_acc": val_acc,
        }
        history.append(row)
        csv_writer.writerow(row)
        csv_file.flush()  # force the OS to write the buffer to disk now

        # If this epoch produced the best validation accuracy so far,
        # save a checkpoint. We save the model WEIGHTS (not the architecture)
        # plus some extra info so we can reload it later.
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),  # all the learned weights
                    "val_acc": val_acc,
                    "classes": classes,
                },
                args.output_dir / "best_model.pt",
            )
            print(f"  -> Saved best model (val_acc={val_acc:.4f})")

    csv_file.close()

    # Save the weights from the very last epoch (may not be the best).
    torch.save(
        {
            "epoch": args.epochs,
            "model_state": model.state_dict(),
            "val_acc": val_acc,
            "classes": classes,
        },
        args.output_dir / "final_model.pt",
    )

    # Save the full history as a JSON file (same data as the CSV, different format).
    with open(args.output_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nDone. Best val accuracy: {best_val_acc:.4f}")
    print(f"Outputs saved to: {args.output_dir}/")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

# Python runs this block only when you execute the file directly
# (e.g. `uv run train.py`), not when another file imports it.
if __name__ == "__main__":
    main()
