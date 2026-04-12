"""
Plot training history from a CSV file.

This is the command-line entry point. It reads the training diary,
draws loss and accuracy charts, and prints a summary table.

Can be run while training is still going -- it reads whatever rows
have been written so far, like peeking at homework before it's done.

Usage:
    uv run plot_history.py
    uv run plot_history.py --csv output/history.csv --out output/curves.png
"""

import argparse
from pathlib import Path

from plant_diseases.plotting import HistoryPlotter


def parse_args():
    """Read optional CSV path and output path from the command line."""
    p = argparse.ArgumentParser(description="Plot training history")
    p.add_argument("--csv", type=Path, default=Path("output/history.csv"))
    p.add_argument("--out", type=Path, default=None,
                   help="Output PNG path (default: next to CSV)")
    p.add_argument("--freeze-epochs", type=int, default=5,
                   help="Draw a vertical line at the freeze/unfreeze boundary")
    return p.parse_args()


def main():
    args = parse_args()

    if not args.csv.exists():
        raise FileNotFoundError(
            f"History file not found: {args.csv}\nRun train.py first."
        )

    out_path = args.out or args.csv.with_suffix(".png")

    plotter = HistoryPlotter(args.csv)

    if plotter.data.is_empty:
        print("No data in CSV yet -- wait for the first epoch to finish.")
        return

    plotter.plot(out_path, args.freeze_epochs)
    plotter.print_table()


if __name__ == "__main__":
    main()
