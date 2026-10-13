from __future__ import annotations

import numpy as np


def potential_summary(phi) -> dict[str, float]:
    arr = np.asarray(phi)
    return {
        "min": float(np.nanmin(arr)),
        "max": float(np.nanmax(arr)),
        "mean": float(np.nanmean(arr)),
        "std": float(np.nanstd(arr)),
    }
