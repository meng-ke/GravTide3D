from __future__ import annotations


def normalize_voxel_size(voxel_size) -> tuple[float, float, float]:
    if voxel_size is None:
        raise ValueError("voxel_size=(dz,dy,dx) is required")
    if len(voxel_size) != 3:
        raise ValueError("voxel_size must contain exactly (dz,dy,dx)")
    dz, dy, dx = (float(v) for v in voxel_size)
    if min(dz, dy, dx) <= 0:
        raise ValueError("all voxel sizes must be positive")
    return dz, dy, dx
