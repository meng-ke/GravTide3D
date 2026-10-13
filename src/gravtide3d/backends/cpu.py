from __future__ import annotations

import numpy as np


class CPUBackend:
    name = "cpu"
    xp = np

    def asarray(self, x, dtype=None):
        return np.asarray(x, dtype=dtype)

    def to_numpy(self, x):
        return np.asarray(x)

    def synchronize(self) -> None:
        return None

    def release(self) -> None:
        return None
