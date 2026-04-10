"""
Classify a single plant image using a trained best_model.pt checkpoint.

HOW THIS SCRIPT WORKS (big picture)
-------------------------------------
Imagine you trained a dog to recognise sick plants by showing it thousands
of photos. Now you want to show the dog ONE new photo and ask: "Hey, what
disease does this plant have?"

That is exactly what this script does. We already have a trained neural
network (saved to a file called best_model.pt). We load that network,
show it one photo, and it tells us which of the 38 plant diseases it sees —
and how sure it is.

Usage:
    uv run classify.py path/to/image.jpg
    uv run classify.py path/to/image.jpg --model output/best_model.pt
"""

# ---------------------------------------------------------------------------
# Imports — loading tools we need from external libraries
# ---------------------------------------------------------------------------

import argparse    # reads command-line flags, like the image path the user types in
from pathlib import Path  # handles file and folder paths in a clean way

import torch                   # the main deep-learning library — our "brain" toolkit
import torch.nn as nn          # building blocks for neural networks (layers, etc.)
from PIL import Image          # opens image files (JPEGs, PNGs, etc.)
from torchvision import models, transforms  # pretrained networks + image processing tools


# ---------------------------------------------------------------------------
# Image pre-processing — getting the photo ready for the network
# ---------------------------------------------------------------------------

# Every pixel in a photo is a colour, described by three numbers:
# how much Red, Green, and Blue light it has (each between 0 and 255).
# Neural networks learn faster and more reliably when these numbers are
# small and centred around zero instead of 0–255.
#
# These two lists are the average colour and spread of colours across
# 1.2 million photos that EfficientNet was originally trained on.
# We reuse them here because the network learned to expect these exact values.
IMAGENET_MEAN = [0.485, 0.456, 0.406]  # average Red, Green, Blue across ImageNet
IMAGENET_STD  = [0.229, 0.224, 0.225]  # spread  Red, Green, Blue across ImageNet

# This is a pipeline of steps that prepares a photo before we show it to
# the network. Think of it like getting a photo ready for a very picky reader:
#   1. Make it the right rough size
#   2. Trim it to an exact square
#   3. Turn the colours into numbers
#   4. Shift those numbers into the range the network understands
#
# These are the SAME steps used on validation images during training,
# so the network sees the photo in a format it already knows.
PREPROCESS = transforms.Compose([
    # Resize the shorter side of the photo to 256 pixels.
    # This keeps the photo's shape (a tall photo stays tall, a wide one stays wide)
    # while making sure it is not too small or gigantic.
    transforms.Resize(256),

    # Cut out the centre 224×224 square.
    # 224×224 is the exact size EfficientNet-B0 expects as input.
    # Taking the centre avoids noisy edges and focuses on the main subject.
    transforms.CenterCrop(224),

    # Convert the image from a picture object into a PyTorch tensor —
    # essentially a big grid of numbers the network can do maths on.
    # Pixel values go from 0–255 down to 0.0–1.0 during this step.
    transforms.ToTensor(),

    # Shift and scale the pixel values using the ImageNet averages above
    # so the numbers end up centred around zero.
    # Without this step the network would give wrong answers because it
    # learned on numbers in a different range.
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


# ---------------------------------------------------------------------------
# Step 1 — Load the trained model from disk
# ---------------------------------------------------------------------------

def load_model(checkpoint_path: Path, device: torch.device) -> tuple[nn.Module, list[str]]:
    """
    Reads the saved model file and rebuilds the neural network.

    When training finished, we saved two things to a file:
      - The network's weights (all the numbers it learned)
      - The list of 38 disease class names

    Here we reload those saved weights into a fresh copy of the same
    network architecture, so it is ready to make predictions again.

    Think of it like printing out a recipe (the architecture) and then
    filling in all the exact measurements the chef discovered (the weights).
    """

    # torch.load reads the checkpoint file from disk into memory.
    # map_location=device makes sure the data lands on the right hardware
    # (GPU or CPU), even if the model was originally saved on a different machine.
    # weights_only=False lets us also load the class list, not just the weights.
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Pull out the list of class names, e.g. ["Apple___Apple_scab", "Apple___healthy", ...]
    # These were saved alongside the weights during training so we always know
    # which number maps to which disease name.
    classes = checkpoint["classes"]

    # Rebuild the EXACT same network architecture that was used during training.
    # weights=None means "start with random weights" — we will overwrite them
    # in a moment with the saved ones, so it doesn't matter that they start random.
    model = models.efficientnet_b0(weights=None)

    # Find out how many numbers the backbone (the "eyes" of the network) outputs
    # for each image. The classifier head must accept exactly that many as input.
    in_features = model.classifier[1].in_features

    # Replace the original classifier head (designed for 1000 ImageNet classes)
    # with the same custom head that was used during training:
    #   Dropout  → randomly turns off 30% of connections — only matters during
    #              training, but we must include it here so the architecture
    #              matches exactly and the weights load correctly.
    #   Linear   → maps the features to one score per disease class.
    #              The class with the highest score is the predicted disease.
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3, inplace=True),
        nn.Linear(in_features, len(classes)),
    )

    # Pour the saved weights (all the numbers the network learned) into our
    # freshly built architecture. After this line the network knows everything
    # it learned during training.
    model.load_state_dict(checkpoint["model_state"])

    # Move the network to the chosen device (GPU or CPU).
    model.to(device)

    # Switch to evaluation mode. Some parts of the network (like Dropout)
    # behave differently during training vs. making predictions.
    # eval() tells them: "we are not learning right now, just predicting."
    model.eval()

    return model, classes


# ---------------------------------------------------------------------------
# Step 2 — Classify a single image
# ---------------------------------------------------------------------------

# @torch.no_grad() is a decorator that tells PyTorch:
# "While this function runs, do NOT track gradients."
# Gradients are the notes PyTorch keeps so it can update the weights later.
# Since we are only predicting (not learning), we do not need those notes —
# skipping them makes the script faster and uses less memory.
@torch.no_grad()
def classify(image_path: Path, model: nn.Module, classes: list[str], device: torch.device) -> tuple[str, float]:
    """
    Shows one image to the network and returns its best guess + confidence.

    The network produces one score per disease class (38 scores total).
    The class with the highest score is the prediction.
    We convert the scores to percentages (0–100%) so the confidence is
    easy for humans to understand.
    """

    # Open the image file from disk.
    # .convert("RGB") ensures the image has exactly three colour channels
    # (Red, Green, Blue). Some images are greyscale (1 channel) or have a
    # transparency layer (4 channels) — this line normalises them all to RGB.
    image = Image.open(image_path).convert("RGB")

    # Run the image through the pre-processing pipeline we defined above
    # (resize → crop → convert to numbers → normalise).
    # .unsqueeze(0) adds an extra dimension at the front.
    # The network always expects a BATCH of images, not just one.
    # So a single 3×224×224 image becomes a 1×3×224×224 batch of one image.
    # .to(device) moves it to GPU or CPU, wherever the model lives.
    tensor = PREPROCESS(image).unsqueeze(0).to(device)

    # FORWARD PASS: feed the image through the network.
    # logits is a list of 38 raw scores — one per disease class.
    # Higher score = the network thinks this class is more likely.
    # The scores are not percentages yet; they can be any number, even negative.
    logits = model(tensor)

    # Convert the raw scores into probabilities that add up to 100%.
    # torch.softmax does this using a maths trick: it makes all scores positive
    # and then divides each one by the total, so together they sum to 1.0 (= 100%).
    # [0] unwraps the batch dimension — we only have one image, so we take item 0.
    probs = torch.softmax(logits, dim=1)[0]

    # Find the index (position) of the highest probability.
    # .item() converts the PyTorch tensor to a plain Python integer.
    idx = probs.argmax().item()

    # Look up the disease name for that index and return it together with
    # the confidence (the probability as a plain Python float, 0.0–1.0).
    return classes[idx], probs[idx].item()


# ---------------------------------------------------------------------------
# Step 3 — Read the command-line arguments the user typed in
# ---------------------------------------------------------------------------

def main():
    """
    The entry point of the script — this is where everything comes together.

    It reads what the user typed in the terminal, checks that the files exist,
    picks the right hardware, loads the model, classifies the image, and
    prints the result.
    """

    # Set up the argument reader so the user can pass the image path and
    # optionally a different model file when running the script.
    parser = argparse.ArgumentParser(description="Classify a plant image for disease")

    # The image path is required — the user must always provide it.
    parser.add_argument("image", type=Path, help="Path to the image file")

    # The model path is optional; if the user does not provide one,
    # we fall back to the default location where training saves the best model.
    parser.add_argument(
        "--model", type=Path, default=Path("output/best_model.pt"),
        help="Path to the model checkpoint (default: output/best_model.pt)",
    )
    args = parser.parse_args()

    # ---------------------------------------------------------------------------
    # Safety checks — make sure both files actually exist before we start
    # ---------------------------------------------------------------------------

    # If the image file does not exist, stop immediately with a clear message.
    # There is no point loading the model if we have no image to classify.
    if not args.image.exists():
        raise SystemExit(f"Image not found: {args.image}")

    # If the model file does not exist, stop with a clear message.
    # This usually means training has not finished yet or the path is wrong.
    if not args.model.exists():
        raise SystemExit(f"Model not found: {args.model}")

    # ---------------------------------------------------------------------------
    # Pick the best available hardware
    # ---------------------------------------------------------------------------

    # Choose where to run the maths.
    # CUDA = an NVIDIA graphics card (GPU) — by far the fastest option.
    # MPS  = the GPU inside Apple Silicon chips (M1, M2, …) — also fast.
    # CPU  = the normal processor — always available but much slower.
    # We automatically pick the best one that is present on this computer.
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )

    # ---------------------------------------------------------------------------
    # Load, classify, and print
    # ---------------------------------------------------------------------------

    # Load the trained network and the list of disease names from the checkpoint.
    model, classes = load_model(args.model, device)

    # Show the image to the network and get back the predicted disease name
    # and a confidence score between 0.0 and 1.0.
    label, confidence = classify(args.image, model, classes, device)

    # Print the result.
    # confidence * 100 converts 0.0–1.0 to 0–100%.
    # :.1f means "show one decimal place", e.g. 94.3%.
    print(f"{label}  ({confidence * 100:.1f}%)")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

# Python runs this block only when you execute the file directly
# (e.g. `uv run classify.py leaf.jpg`), not when another file imports it.
if __name__ == "__main__":
    main()
