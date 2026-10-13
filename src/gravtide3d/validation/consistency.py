from __future__ import annotations

import numpy as np


def eigenvector_alignment(a, b):
    """Absolute column-wise dot products, robust to e -> -e sign ambiguity."""
    a = np.asarray(a)
    b = np.asarray(b)
    return np.abs(np.sum(a * b, axis=-2))


def max_relative_error(a, b, *, floor=1e-30) -> float:
    a = np.asarray(a)
    b = np.asarray(b)
    scale = np.maximum(np.maximum(np.abs(a), np.abs(b)), float(floor))
    return float(np.max(np.abs(a - b) / scale))
