"""
Train a plant disease classifier using EfficientNet-B0 transfer learning.

This is the command-line entry point. All heavy lifting lives in the
plant_diseases package -- this script just reads your settings,
wires everything together, and kicks off training.

Usage:
    uv run train.py                           # train with defaults
    uv run train.py --epochs 30 --batch-size 64
"""

import argparse
import json
from pathlib import Path

from plant_diseases.data import build_dataloaders
from plant_diseases.device import select_device
from plant_diseases.history import TrainingHistory
from plant_diseases.model import PlantDiseaseModel
from plant_diseases.trainer import Trainer


def parse_args():
    """Read training options from the command line."""
    p = argparse.ArgumentParser(description="Train plant disease classifier")

    p.add_argument(
        "--data-dir", type=Path,
        default=Path("New Plant Diseases Dataset(Augmented) copy"),
        help="Root directory containing train/ and valid/ folders",
    )
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument(
        "--freeze-epochs", type=int, default=5,
        help="Train only the classifier head for this many epochs",
    )
    p.add_argument("--output-dir", type=Path, default=Path("output"))
    p.add_argument("--no-amp", action="store_true",
                   help="Disable automatic mixed precision")

    return p.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Pick the fastest available hardware.
    device = select_device()
    print(f"Device: {device}")

    # Load training and validation photos.
    bundle = build_dataloaders(args.data_dir, args.batch_size, args.num_workers)
    print(
        f"Classes: {len(bundle.classes)}  |  "
        f"Train batches: {len(bundle.train_loader)}  |  "
        f"Val batches: {len(bundle.val_loader)}"
    )

    # Save the number-to-name mapping for later use by classify.py.
    class_map = {i: name for i, name in enumerate(bundle.classes)}
    with open(args.output_dir / "class_map.json", "w") as f:
        json.dump(class_map, f, indent=2)

    # Build the model and move it to the chosen device.
    model = PlantDiseaseModel(
        num_classes=len(bundle.classes), freeze_backbone=True
    ).to(device)

    # Set up the trainer with optimizer, scheduler, and scaler.
    trainer = Trainer(
        model=model,
        device=device,
        lr=args.lr,
        weight_decay=args.weight_decay,
        total_epochs=args.epochs,
        freeze_epochs=args.freeze_epochs,
        use_amp=not args.no_amp,
    )

    # Run training and log every epoch to CSV/JSON.
    best_val_acc = 0.0

    with TrainingHistory(args.output_dir) as history:
        for result in trainer.run(bundle.train_loader, bundle.val_loader):
            history.record(result)

            # Save the best checkpoint whenever validation accuracy improves.
            if result.val_acc > best_val_acc:
                best_val_acc = result.val_acc
                model.save_checkpoint(
                    args.output_dir / "best_model.pt",
                    epoch=result.epoch,
                    val_acc=result.val_acc,
                    classes=bundle.classes,
                )
                print(f"  -> Saved best model (val_acc={result.val_acc:.4f})")

    # Always save the final epoch's weights too.
    model.save_checkpoint(
        args.output_dir / "final_model.pt",
        epoch=args.epochs,
        val_acc=best_val_acc,
        classes=bundle.classes,
    )

    print(f"\nDone. Best val accuracy: {best_val_acc:.4f}")
    print(f"Outputs saved to: {args.output_dir}/")


if __name__ == "__main__":
    main()
