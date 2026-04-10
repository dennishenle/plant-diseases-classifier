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

Three scripts, no packages:

**`train.py`** — end-to-end training pipeline:
1. `build_dataloaders` — loads `ImageFolder` datasets from `<data-dir>/train/` and `<data-dir>/valid/`, applies augmentation (RandomResizedCrop, flips, ColorJitter) to train and deterministic center-crop to val.
2. `build_model` — loads pretrained EfficientNet-B0, replaces the classifier head with `Dropout(0.3) → Linear(1280, num_classes)`.
3. `train_one_epoch` / `evaluate` — standard PyTorch train/eval loops with optional CUDA AMP (`GradScaler` + `autocast`).
4. Main loop — two-phase training: head-only for `--freeze-epochs` epochs (backbone frozen), then full fine-tune at `lr × 0.1`. Uses AdamW (with `--weight-decay`, default `1e-4`) + CosineAnnealingLR + CrossEntropyLoss with label smoothing 0.1.
5. Writes `output/history.csv` (flushed after every epoch), `output/history.json`, `output/best_model.pt`, `output/final_model.pt`, `output/class_map.json`.

**`plot_history.py`** — reads `output/history.csv`, prints a per-epoch table, saves `output/history.png` with loss and accuracy curves. Optionally marks the freeze/unfreeze boundary with `--freeze-epochs`. Accepts `--csv` and `--out` to override input/output paths.

**`classify.py`** — loads a saved checkpoint (`output/best_model.pt` by default), preprocesses a single image (Resize(256) → CenterCrop(224) → Normalize), runs a forward pass, and prints the top predicted class with softmax confidence. Override the checkpoint with `--model`.

## Dataset

Expected at `New Plant Diseases Dataset(Augmented) copy/` (override with `--data-dir`). 38 classes, `ImageFolder`-compatible layout: one subdirectory per class under `train/` and `valid/`.

## Checkpoint format

Saved checkpoints are dicts with keys `epoch`, `model_state`, `val_acc`, `classes`. To reload, reconstruct the same `nn.Sequential(Dropout, Linear)` head before calling `load_state_dict`.

## Device selection

Auto-detects: CUDA → MPS (Apple Silicon) → CPU.
