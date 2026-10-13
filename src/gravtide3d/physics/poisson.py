from __future__ import annotations

import math
import numpy as np

G_CGS = 6.67430e-8


def make_k_axes(shape, voxel_size, *, xp=np):
    """Return angular spatial frequencies for arrays ordered (z,y,x)."""
    nz, ny, nx = (int(v) for v in shape)
    dz, dy, dx = (float(v) for v in voxel_size)
    kz = 2.0 * math.pi * xp.fft.fftfreq(nz, d=dz)
    ky = 2.0 * math.pi * xp.fft.fftfreq(ny, d=dy)
    kx = 2.0 * math.pi * xp.fft.rfftfreq(nx, d=dx)
    return kz, ky, kx


def poisson_factor(kz, ky, kx, *, xp=np, G: float = G_CGS):
    """Fourier multiplier mapping rho_hat to phi_hat.

    phi_hat = factor * rho_hat, with the k=0 mode fixed to zero.
    """
    k2 = (
        kz[:, None, None] ** 2
        + ky[None, :, None] ** 2
        + kx[None, None, :] ** 2
    )
    factor = xp.zeros_like(k2)
    mask = k2 != 0
    factor[mask] = -4.0 * math.pi * G / k2[mask]
    return factor
