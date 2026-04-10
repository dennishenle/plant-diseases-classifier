"""
Plot training history from output/history.csv.

HOW THIS SCRIPT WORKS (big picture)
-------------------------------------
When the network is training, it keeps a diary. After every round of
learning (called an "epoch") it writes down how well it did:
  - How wrong its guesses were (the "loss") — lower is better
  - How many photos it got right (the "accuracy") — higher is better

It tracks this for BOTH the photos it learned from (train) and the
photos it has never seen before (val = validation). Watching both
numbers together tells you if the network is truly learning or just
memorising.

This script reads that diary (a file called history.csv), draws two
charts — one for loss, one for accuracy — and prints a neat table in
the terminal.

You can run it while training is still going. It will simply draw
whatever data has been written so far, like peeking at a student's
homework before they finish.

Works mid-training (reads whatever rows are written so far) and after.
Saves a PNG next to the CSV.

Usage:
    uv run plot_history.py                    # reads output/history.csv
    uv run plot_history.py --csv output/history.csv --out output/curves.png
"""

# ---------------------------------------------------------------------------
# Imports — loading tools we need from external libraries
# ---------------------------------------------------------------------------

import argparse   # reads command-line flags the user types in
import csv        # reads spreadsheet-style files (rows and columns of numbers)
from pathlib import Path  # handles file and folder paths in a clean way

import matplotlib.pyplot as plt  # the library that draws charts and saves them as images


# ---------------------------------------------------------------------------
# Step 1 — Read the training diary (CSV file)
# ---------------------------------------------------------------------------

def load_csv(path: Path) -> dict[str, list]:
    """
    Opens the CSV file and loads all the numbers into memory.

    A CSV (Comma-Separated Values) file is like a very simple spreadsheet.
    Each row is one epoch, and the columns are:
      epoch, train_loss, val_loss, train_acc, val_acc

    We read every row and collect each column into its own list, so we end
    up with something like:
      {
        "epoch":      [1, 2, 3, ...],
        "train_loss": [1.23, 0.98, 0.75, ...],
        ...
      }
    This makes it easy to hand the lists straight to the chart library.
    """

    # Start with empty lists for each column we care about.
    data: dict[str, list] = {
        "epoch": [], "train_loss": [], "val_loss": [],
        "train_acc": [], "val_acc": [],
    }

    # Open the file and use DictReader, which reads each row as a small
    # dictionary where the column name is the key and the cell value is
    # the value — much easier than counting column positions by hand.
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            # float() converts the text "0.9823" into an actual number 0.9823
            # so we can do maths with it and draw it on a chart.
            for key in data:
                data[key].append(float(row[key]))

    return data


# ---------------------------------------------------------------------------
# Step 2 — Draw the charts and print the summary table
# ---------------------------------------------------------------------------

def plot(data: dict, out_path: Path, freeze_epochs: int | None):
    """
    Draws two side-by-side charts and a summary table.

    Left chart  → Loss over time (how wrong the network was each epoch).
    Right chart → Accuracy over time (what percentage of photos it got right).

    Each chart shows two lines:
      - Train: performance on photos the model LEARNED from
      - Val:   performance on NEW photos it has never seen

    The gap between these two lines is very informative:
      - Lines close together → the model is learning well and generalising
      - Train much better than Val → the model is memorising (overfitting)
    """

    epochs = data["epoch"]  # the list [1, 2, 3, ...] used as the x-axis

    # Create a figure (the whole image) with two side-by-side subplots (axes).
    # figsize=(12, 5) sets the image size in inches: 12 wide, 5 tall.
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(12, 5))

    # Add a big title at the very top of the whole image.
    fig.suptitle("Plant Disease Classifier — Training History", fontsize=13)

    # -----------------------------------------------------------------------
    # Left chart: Loss
    # -----------------------------------------------------------------------
    # Loss measures how wrong the network was on average.
    # A loss of 0 means perfect. Higher means more mistakes.
    # We want to see this line go DOWN over time.

    ax_loss.plot(epochs, data["train_loss"], label="Train loss")
    ax_loss.plot(epochs, data["val_loss"],   label="Val loss")
    ax_loss.set_xlabel("Epoch")       # label for the horizontal axis
    ax_loss.set_ylabel("Loss")        # label for the vertical axis
    ax_loss.set_title("Loss")
    ax_loss.legend()                  # show a small key explaining which line is which
    ax_loss.grid(True, alpha=0.3)     # light grey grid lines to make values easier to read

    # -----------------------------------------------------------------------
    # Right chart: Accuracy
    # -----------------------------------------------------------------------
    # Accuracy is the percentage of photos the network labelled correctly.
    # We want to see this line go UP over time.
    # We multiply by 100 to turn 0.9 into 90% — easier to read.

    ax_acc.plot(epochs, [a * 100 for a in data["train_acc"]], label="Train acc")
    ax_acc.plot(epochs, [a * 100 for a in data["val_acc"]],   label="Val acc")
    ax_acc.set_xlabel("Epoch")
    ax_acc.set_ylabel("Accuracy (%)")
    ax_acc.set_title("Accuracy")
    ax_acc.legend()
    ax_acc.grid(True, alpha=0.3)

    # -----------------------------------------------------------------------
    # Mark the best epoch with a green dashed vertical line
    # -----------------------------------------------------------------------
    # We scan through all validation accuracies and find the epoch where the
    # network did best on unseen photos. That is the epoch whose model weights
    # we saved as best_model.pt.
    # max(..., key=...) finds the index of the largest value in the list.
    best_epoch = max(range(len(data["val_acc"])), key=lambda i: data["val_acc"][i])
    best_acc   = data["val_acc"][best_epoch] * 100

    # axvline draws a vertical line at the given x position (the best epoch number).
    ax_acc.axvline(
        epochs[best_epoch],
        color="green", linestyle="--", alpha=0.6,
        label=f"Best: {best_acc:.2f}% @ epoch {epochs[best_epoch]:.0f}",
    )
    ax_acc.legend()  # refresh the legend so the new green line appears in it

    # -----------------------------------------------------------------------
    # Mark the freeze/unfreeze boundary with an orange dotted vertical line
    # -----------------------------------------------------------------------
    # Training happens in two phases:
    #   Phase 1 (frozen):   Only the new classifier head learns. The backbone
    #                        (the "eyes") is locked so its knowledge is preserved.
    #   Phase 2 (unfrozen): The whole network learns together, making fine
    #                        adjustments to specialise in plant diseases.
    #
    # The orange line shows where the switch from phase 1 to phase 2 happened.
    # You often see a small dip in accuracy right after this line as the network
    # adjusts — that is normal.
    if freeze_epochs and freeze_epochs < max(epochs):
        for ax in (ax_loss, ax_acc):
            # The boundary sits between epoch freeze_epochs and freeze_epochs+1,
            # so we draw it at freeze_epochs + 0.5 (halfway between the two).
            ax.axvline(
                freeze_epochs + 0.5,
                color="orange", linestyle=":", alpha=0.8,
                label="Backbone unfrozen",
            )
        ax_loss.legend()
        ax_acc.legend()

    # tight_layout() automatically adjusts spacing so nothing overlaps.
    plt.tight_layout()

    # Save the finished image to a file. dpi=150 gives a crisp, high-resolution PNG.
    plt.savefig(out_path, dpi=150)
    print(f"Saved plot to {out_path}")

    # -----------------------------------------------------------------------
    # Print a summary table in the terminal
    # -----------------------------------------------------------------------
    # Sometimes you just want to read the numbers instead of looking at a chart.
    # This prints a neat table with one row per epoch.
    # The "<-- best" marker shows which epoch produced the best model.

    # Print the header row with right-aligned column names.
    print(f"\n{'Epoch':>6}  {'Train loss':>10}  {'Val loss':>9}  {'Train acc':>10}  {'Val acc':>9}")
    print("-" * 55)  # a line of dashes to separate the header from the data

    for i, ep in enumerate(epochs):
        # Add a marker to the row where validation accuracy was the highest.
        marker = " <-- best" if i == best_epoch else ""
        print(
            f"{ep:6.0f}  {data['train_loss'][i]:10.4f}  {data['val_loss'][i]:9.4f}"
            f"  {data['train_acc'][i]*100:9.2f}%  {data['val_acc'][i]*100:8.2f}%{marker}"
        )


# ---------------------------------------------------------------------------
# Step 3 — Read the command-line arguments the user typed in
# ---------------------------------------------------------------------------

def parse_args():
    """
    Reads the optional settings the user can pass when running the script.
    If nothing is passed, sensible defaults are used so the script just works.
    """
    parser = argparse.ArgumentParser(description="Plot training history")

    # Which CSV file to read. Defaults to the standard output location.
    parser.add_argument("--csv", type=Path, default=Path("output/history.csv"))

    # Where to save the PNG image. If not given, it is saved next to the CSV.
    parser.add_argument("--out", type=Path, default=None,
                        help="Output PNG path (default: next to CSV)")

    # How many epochs the backbone was frozen. Used to draw the orange line.
    # If you trained with a different --freeze-epochs value, pass it here.
    parser.add_argument("--freeze-epochs", type=int, default=5,
                        help="Draw a vertical line at the freeze/unfreeze boundary")

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main — puts everything together
# ---------------------------------------------------------------------------

def main():
    """
    The entry point of the script — reads arguments, checks the file exists,
    loads the data, and draws the charts.
    """
    args = parse_args()

    # If the CSV file does not exist yet, training has not started (or the path
    # is wrong). Stop with a helpful message instead of a confusing crash.
    if not args.csv.exists():
        raise FileNotFoundError(f"History file not found: {args.csv}\nRun train.py first.")

    # If the user did not provide an output path, save the PNG in the same
    # folder as the CSV, just with a .png extension instead of .csv.
    out_path = args.out or args.csv.with_suffix(".png")

    # Read all the numbers from the CSV file.
    data = load_csv(args.csv)

    # If the epoch list is empty the file exists but training has not written
    # a single row yet. Let the user know and exit cleanly.
    if not data["epoch"]:
        print("No data in CSV yet — wait for the first epoch to finish.")
        return

    # Draw the charts and print the table.
    plot(data, out_path, args.freeze_epochs)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

# Python runs this block only when you execute the file directly
# (e.g. `uv run plot_history.py`), not when another file imports it.
if __name__ == "__main__":
    main()
