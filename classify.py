"""
Classify a single plant image using a trained checkpoint.

This is the command-line entry point. It loads a saved model, shows it
one photo, and prints the predicted disease with a confidence score.

Usage:
    uv run classify.py path/to/leaf.jpg
    uv run classify.py path/to/leaf.jpg --model output/best_model.pt
"""

import argparse
from pathlib import Path

import torch
from PIL import Image

from plant_diseases.device import select_device
from plant_diseases.model import PlantDiseaseModel
from plant_diseases.transforms import build_val_transforms


def parse_args():
    """Read the image path and optional model path from the command line."""
    p = argparse.ArgumentParser(description="Classify a plant image for disease")
    p.add_argument("image", type=Path, help="Path to the image file")
    p.add_argument(
        "--model", type=Path, default=Path("output/best_model.pt"),
        help="Path to the model checkpoint",
    )
    return p.parse_args()


@torch.no_grad()
def classify(
    image_path: Path,
    model: PlantDiseaseModel,
    classes: list[str],
    device: torch.device,
) -> tuple[str, float]:
    """
    Show one image to the network and return its best guess + confidence.

    The network produces 38 scores (one per disease class). The class
    with the highest score is the prediction. We convert scores to
    percentages so the confidence is easy to understand.
    """
    # Open the image and ensure it has exactly 3 colour channels (RGB).
    image = Image.open(image_path).convert("RGB")

    # Run through the same preprocessing pipeline used during validation.
    # .unsqueeze(0) adds a batch dimension -- the network always expects
    # a batch, even when there's only one image.
    preprocess = build_val_transforms()
    tensor = preprocess(image).unsqueeze(0).to(device)

    # Forward pass: get one score per disease class.
    logits = model(tensor)

    # Convert raw scores to probabilities that sum to 100 %.
    probs = torch.softmax(logits, dim=1)[0]

    # Pick the class with the highest probability.
    idx = probs.argmax().item()
    return classes[idx], probs[idx].item()


def main():
    args = parse_args()

    if not args.image.exists():
        raise SystemExit(f"Image not found: {args.image}")
    if not args.model.exists():
        raise SystemExit(f"Model not found: {args.model}")

    device = select_device()

    # Load the trained network and class list from the checkpoint.
    model, checkpoint = PlantDiseaseModel.from_checkpoint(args.model, device)

    # Classify the image and print the result.
    label, confidence = classify(args.image, model, checkpoint.classes, device)
    print(f"{label}  ({confidence * 100:.1f}%)")


if __name__ == "__main__":
    main()
