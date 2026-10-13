from __future__ import annotations

from pathlib import Path
import numpy as np

from ..utils.axes import normalize_voxel_size


def validate_voxel_size(voxel_size):
    return normalize_voxel_size(voxel_size)


def _array_from_source(density):
    if isinstance(density, (str, Path)):
        arr = np.load(Path(density), mmap_mode="r")
        return arr, True
    return density, False


def validate_density(density, *, chunk_z: int = 8, require_nonnegative: bool = True):
    """Validate a 3-D density field without forcing a huge array into RAM."""
    arr, opened = _array_from_source(density)
    try:
        if getattr(arr, "ndim", None) != 3:
            raise ValueError("density must be a 3-D array ordered (z,y,x)")
        shape = tuple(int(v) for v in arr.shape)
        if min(shape) < 1:
            raise ValueError("density axes must all be non-empty")
        for z0 in range(0, shape[0], max(1, int(chunk_z))):
            block = np.asarray(arr[z0:z0 + chunk_z])
            if not np.isfinite(block).all():
                raise ValueError(f"density contains NaN or Inf near z={z0}")
            if require_nonnegative and np.any(block < 0):
                raise ValueError(f"density contains negative values near z={z0}")
        return shape
    finally:
        if opened:
            del arr
