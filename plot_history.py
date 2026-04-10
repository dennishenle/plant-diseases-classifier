"""
Plot training history from output/history.csv.

Works mid-training (reads whatever rows are written so far) and after.
Saves a PNG next to the CSV.

Usage:
    uv run plot_history.py                    # reads output/history.csv
    uv run plot_history.py --csv output/history.csv --out output/curves.png
"""

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


def load_csv(path: Path) -> dict[str, list]:
    data: dict[str, list] = {
        "epoch": [], "train_loss": [], "val_loss": [],
        "train_acc": [], "val_acc": [],
    }
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            for key in data:
                data[key].append(float(row[key]))
    return data


def plot(data: dict, out_path: Path, freeze_epochs: int | None):
    epochs = data["epoch"]
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Plant Disease Classifier — Training History", fontsize=13)

    # Loss
    ax_loss.plot(epochs, data["train_loss"], label="Train loss")
    ax_loss.plot(epochs, data["val_loss"], label="Val loss")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Loss")
    ax_loss.set_title("Loss")
    ax_loss.legend()
    ax_loss.grid(True, alpha=0.3)

    # Accuracy
    ax_acc.plot(epochs, [a * 100 for a in data["train_acc"]], label="Train acc")
    ax_acc.plot(epochs, [a * 100 for a in data["val_acc"]], label="Val acc")
    ax_acc.set_xlabel("Epoch")
    ax_acc.set_ylabel("Accuracy (%)")
    ax_acc.set_title("Accuracy")
    ax_acc.legend()
    ax_acc.grid(True, alpha=0.3)

    # Mark best val accuracy
    best_epoch = max(range(len(data["val_acc"])), key=lambda i: data["val_acc"][i])
    best_acc = data["val_acc"][best_epoch] * 100
    ax_acc.axvline(epochs[best_epoch], color="green", linestyle="--", alpha=0.6,
                   label=f"Best: {best_acc:.2f}% @ epoch {epochs[best_epoch]:.0f}")
    ax_acc.legend()

    # Mark freeze/unfreeze boundary
    if freeze_epochs and freeze_epochs < max(epochs):
        for ax in (ax_loss, ax_acc):
            ax.axvline(freeze_epochs + 0.5, color="orange", linestyle=":",
                       alpha=0.8, label="Backbone unfrozen")
        ax_loss.legend()
        ax_acc.legend()

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"Saved plot to {out_path}")

    # Print summary table
    print(f"\n{'Epoch':>6}  {'Train loss':>10}  {'Val loss':>9}  {'Train acc':>10}  {'Val acc':>9}")
    print("-" * 55)
    for i, ep in enumerate(epochs):
        marker = " <-- best" if i == best_epoch else ""
        print(
            f"{ep:6.0f}  {data['train_loss'][i]:10.4f}  {data['val_loss'][i]:9.4f}"
            f"  {data['train_acc'][i]*100:9.2f}%  {data['val_acc'][i]*100:8.2f}%{marker}"
        )


def parse_args():
    parser = argparse.ArgumentParser(description="Plot training history")
    parser.add_argument("--csv", type=Path, default=Path("output/history.csv"))
    parser.add_argument("--out", type=Path, default=None,
                        help="Output PNG path (default: next to CSV)")
    parser.add_argument("--freeze-epochs", type=int, default=5,
                        help="Draw a vertical line at the freeze/unfreeze boundary")
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.csv.exists():
        raise FileNotFoundError(f"History file not found: {args.csv}\nRun train.py first.")

    out_path = args.out or args.csv.with_suffix(".png")
    data = load_csv(args.csv)

    if not data["epoch"]:
        print("No data in CSV yet — wait for the first epoch to finish.")
        return

    plot(data, out_path, args.freeze_epochs)


if __name__ == "__main__":
    main()
