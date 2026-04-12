"""
Training history logger.

While the network is learning, we keep a diary of how well it does
after each round (epoch). This diary has two formats:

  CSV  -- a simple spreadsheet that can be read mid-training
          (plot_history.py reads it while training is still running).

  JSON -- a structured file saved at the very end, handy for scripts.

The TrainingHistory class manages both formats and makes sure every
row is flushed to disk immediately so you never lose progress.
"""

import csv
import json
from pathlib import Path
from typing import TextIO

from .trainer import EpochResult


class TrainingHistory:
    """
    Records training metrics to CSV (live) and JSON (on close).

    Usage:
        history = TrainingHistory(output_dir)
        for epoch_result in trainer.run(...):
            history.record(epoch_result)
        history.close()

    Or as a context manager:
        with TrainingHistory(output_dir) as history:
            ...
    """

    # The columns in our CSV diary.
    _FIELDS = ["epoch", "train_loss", "train_acc", "val_loss", "val_acc"]

    def __init__(self, output_dir: Path) -> None:
        self._output_dir = output_dir
        self._rows: list[dict] = []

        # Open the CSV file for live writing.
        self._csv_path = output_dir / "history.csv"
        self._csv_file: TextIO = open(self._csv_path, "w", newline="")
        self._csv_writer = csv.DictWriter(self._csv_file, fieldnames=self._FIELDS)
        self._csv_writer.writeheader()

    # -- Recording -----------------------------------------------------------

    def record(self, result: EpochResult) -> None:
        """
        Write one epoch's results to CSV (immediately) and keep a copy
        in memory for the JSON file we'll write at the end.
        """
        row = {
            "epoch": result.epoch,
            "train_loss": result.train_loss,
            "train_acc": result.train_acc,
            "val_loss": result.val_loss,
            "val_acc": result.val_acc,
        }
        self._rows.append(row)
        self._csv_writer.writerow(row)

        # Flush forces the OS to actually write the data to disk right
        # now, so plot_history.py can read it without waiting.
        self._csv_file.flush()

    # -- Closing / saving ----------------------------------------------------

    def close(self) -> None:
        """Close the CSV file and save the full history as JSON."""
        self._csv_file.close()

        json_path = self._output_dir / "history.json"
        with open(json_path, "w") as f:
            json.dump(self._rows, f, indent=2)

    # -- Context manager support ---------------------------------------------

    def __enter__(self) -> "TrainingHistory":
        return self

    def __exit__(self, *_exc) -> None:
        self.close()
