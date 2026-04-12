# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
uv sync                              # install dependencies into .venv
uv run train.py                      # train with defaults (20 epochs, batch 32, lr 1e-3)
uv run plot_history.py               # print summary table and save output/history.png
uv run classify.py path/to/leaf.jpg  # predict disease in a single image
```

`plot_history.py` can be run while training is in progress — it reads whatever rows are in `output/history.csv` so far.

## Architecture

Three thin CLI scripts delegate to the `plant_diseases/` package:

```
plant_diseases/
├── __init__.py       — package docstring
├── config.py         — shared constants (ImageNet mean/std, input size, dropout rate)
├── device.py         — auto-detects CUDA → MPS → CPU
├── transforms.py     — image preprocessing pipelines (train, val/inference)
├── data.py           — DataBundle: loads ImageFolder datasets into DataLoaders
├── model.py          — PlantDiseaseModel: builds EfficientNet-B0, freeze/unfreeze, checkpoint I/O
├── trainer.py        — Trainer: orchestrates training loop, optimizer, scheduler, AMP scaler
├── history.py        — TrainingHistory: live CSV logging + JSON export
└── plotting.py       — HistoryPlotter: reads CSV, draws loss/accuracy charts, prints table
```

### Key classes

- **`PlantDiseaseModel`** (`model.py`) — wraps EfficientNet-B0 with a custom classifier head. Handles building from pretrained weights, freezing/unfreezing the backbone, saving/loading checkpoints. Class method `from_checkpoint()` rebuilds a trained model from a `.pt` file.
- **`Trainer`** (`trainer.py`) — owns the loss function, optimizer (AdamW), LR scheduler (CosineAnnealingLR), and optional AMP GradScaler. `run()` is a generator yielding `EpochResult` after each epoch, enabling live checkpointing and logging.
- **`TrainingHistory`** (`history.py`) — context-manager that writes `history.csv` (flushed after every epoch) and `history.json` on close.
- **`HistoryPlotter`** (`plotting.py`) — reads CSV, draws loss/accuracy charts with best-epoch and freeze-boundary markers, prints a summary table.
- **`DataBundle`** (`data.py`) — dataclass holding `train_loader`, `val_loader`, and `classes` list.

### CLI entry points

- **`train.py`** — parses args, wires `DataBundle` → `PlantDiseaseModel` → `Trainer` → `TrainingHistory`. Saves `best_model.pt`, `final_model.pt`, `class_map.json`.
- **`classify.py`** — loads checkpoint via `PlantDiseaseModel.from_checkpoint()`, preprocesses one image, prints predicted class + confidence.
- **`plot_history.py`** — instantiates `HistoryPlotter`, draws charts, prints table.

### Training strategy

Two-phase: head-only for `--freeze-epochs` epochs (backbone frozen), then full fine-tune at `lr × 0.1`. AdamW + CosineAnnealingLR + CrossEntropyLoss with label smoothing 0.1.

## Dataset

Expected at `New Plant Diseases Dataset(Augmented) copy/` (override with `--data-dir`). 38 classes, `ImageFolder`-compatible layout: one subdirectory per class under `train/` and `valid/`.

## Checkpoint format

Saved checkpoints are dicts with keys `epoch`, `model_state`, `val_acc`, `classes`. `PlantDiseaseModel.from_checkpoint()` handles reconstruction automatically.

## Device selection

Auto-detects: CUDA → MPS (Apple Silicon) → CPU. Centralised in `plant_diseases/device.py`.
