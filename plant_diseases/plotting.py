"""
Training history visualisation.

After (or during) training, this module reads the diary CSV and draws
two side-by-side charts:

  LEFT  -- Loss over time (how wrong the network was each round).
           We want this line to go DOWN.

  RIGHT -- Accuracy over time (what % of photos it got right).
           We want this line to go UP.

Each chart shows two lines:
  Train : performance on photos the model learned from.
  Val   : performance on NEW photos it has never seen.

The gap between these two lines is very informative:
  Lines close together  --> the model is learning real patterns.
  Train much better     --> the model is memorising (overfitting).
"""

import csv
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# HistoryData -- the numbers loaded from the CSV
# ---------------------------------------------------------------------------

@dataclass
class HistoryData:
    """
    All columns from history.csv, each stored as a list of floats.

    Example:
        data.epoch      == [1.0, 2.0, 3.0, ...]
        data.train_loss == [1.23, 0.98, 0.75, ...]
    """
    epoch: list[float] = field(default_factory=list)
    train_loss: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    train_acc: list[float] = field(default_factory=list)
    val_acc: list[float] = field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        """True if no epochs have been recorded yet."""
        return len(self.epoch) == 0

    @property
    def best_epoch_index(self) -> int:
        """Index of the epoch with the highest validation accuracy."""
        return max(range(len(self.val_acc)), key=lambda i: self.val_acc[i])


# ---------------------------------------------------------------------------
# HistoryPlotter -- reads CSV and draws charts
# ---------------------------------------------------------------------------

class HistoryPlotter:
    """
    Reads a training history CSV and produces charts + a summary table.

    Usage:
        plotter = HistoryPlotter(csv_path)
        plotter.plot(out_path, freeze_epochs=5)
        plotter.print_table()
    """

    def __init__(self, csv_path: Path) -> None:
        self._csv_path = csv_path
        self.data = self._load_csv()

    # -- CSV loading ---------------------------------------------------------

    def _load_csv(self) -> HistoryData:
        """
        Read the CSV file and collect each column into a list of numbers.

        A CSV (Comma-Separated Values) file is like a simple spreadsheet.
        Each row is one epoch. We convert every cell from text ("0.98")
        into an actual number (0.98) so we can draw charts with it.
        """
        data = HistoryData()

        with open(self._csv_path, newline="") as f:
            for row in csv.DictReader(f):
                data.epoch.append(float(row["epoch"]))
                data.train_loss.append(float(row["train_loss"]))
                data.val_loss.append(float(row["val_loss"]))
                data.train_acc.append(float(row["train_acc"]))
                data.val_acc.append(float(row["val_acc"]))

        return data

    # -- Chart drawing -------------------------------------------------------

    def plot(self, out_path: Path, freeze_epochs: int | None = None) -> None:
        """
        Draw loss and accuracy charts and save as a PNG image.

        Args:
            out_path:      Where to save the PNG file.
            freeze_epochs: If set, draw an orange line showing where the
                           backbone was unfrozen (phase 1 -> phase 2).
        """
        d = self.data
        epochs = d.epoch

        # Create a figure with two side-by-side charts.
        fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle("Plant Disease Classifier -- Training History", fontsize=13)

        # --- Left chart: Loss (lower is better) ---
        ax_loss.plot(epochs, d.train_loss, label="Train loss")
        ax_loss.plot(epochs, d.val_loss, label="Val loss")
        ax_loss.set_xlabel("Epoch")
        ax_loss.set_ylabel("Loss")
        ax_loss.set_title("Loss")
        ax_loss.legend()
        ax_loss.grid(True, alpha=0.3)

        # --- Right chart: Accuracy (higher is better) ---
        ax_acc.plot(epochs, [a * 100 for a in d.train_acc], label="Train acc")
        ax_acc.plot(epochs, [a * 100 for a in d.val_acc], label="Val acc")
        ax_acc.set_xlabel("Epoch")
        ax_acc.set_ylabel("Accuracy (%)")
        ax_acc.set_title("Accuracy")
        ax_acc.legend()
        ax_acc.grid(True, alpha=0.3)

        # --- Green dashed line at the best epoch ---
        best_i = d.best_epoch_index
        best_acc = d.val_acc[best_i] * 100
        ax_acc.axvline(
            epochs[best_i],
            color="green", linestyle="--", alpha=0.6,
            label=f"Best: {best_acc:.2f}% @ epoch {epochs[best_i]:.0f}",
        )
        ax_acc.legend()

        # --- Orange dotted line at the freeze/unfreeze boundary ---
        if freeze_epochs and freeze_epochs < max(epochs):
            for ax in (ax_loss, ax_acc):
                ax.axvline(
                    freeze_epochs + 0.5,
                    color="orange", linestyle=":", alpha=0.8,
                    label="Backbone unfrozen",
                )
            ax_loss.legend()
            ax_acc.legend()

        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        print(f"Saved plot to {out_path}")

    # -- Text table ----------------------------------------------------------

    def print_table(self) -> None:
        """Print a neat per-epoch summary table in the terminal."""
        d = self.data
        best_i = d.best_epoch_index

        print(
            f"\n{'Epoch':>6}  {'Train loss':>10}  {'Val loss':>9}"
            f"  {'Train acc':>10}  {'Val acc':>9}"
        )
        print("-" * 55)

        for i, ep in enumerate(d.epoch):
            marker = " <-- best" if i == best_i else ""
            print(
                f"{ep:6.0f}  {d.train_loss[i]:10.4f}  {d.val_loss[i]:9.4f}"
                f"  {d.train_acc[i]*100:9.2f}%  {d.val_acc[i]*100:8.2f}%{marker}"
            )
