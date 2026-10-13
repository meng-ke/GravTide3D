from __future__ import annotations

from pathlib import Path
import numpy as np


def load_npy_mmap(path, mode="r"):
    return np.load(Path(path), mmap_mode=mode)
