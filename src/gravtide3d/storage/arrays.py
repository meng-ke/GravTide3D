from __future__ import annotations

from pathlib import Path
import numpy as np


def open_output_array(path, *, shape, dtype, mode="w+"):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return np.lib.format.open_memmap(path, mode=mode, dtype=dtype, shape=shape)
