"""
Automatic hardware selection.

A neural network does millions of multiplications. Doing them on a
graphics card (GPU) is 10-50x faster than on a normal processor (CPU).

This module checks what hardware is available and picks the fastest:
  1. CUDA  -- an NVIDIA GPU (fastest)
  2. MPS   -- the GPU inside Apple Silicon chips (M1, M2, ...)
  3. CPU   -- the normal processor (always available, but slowest)
"""

import torch


def select_device() -> torch.device:
    """Return the fastest available compute device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
