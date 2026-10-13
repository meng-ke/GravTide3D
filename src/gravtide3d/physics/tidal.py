from __future__ import annotations

import numpy as np

COMPONENTS = ("zz", "yy", "xx", "zy", "zx", "yx")
COMPONENT_AXES = {
    "zz": (0, 0),
    "yy": (1, 1),
    "xx": (2, 2),
    "zy": (0, 1),
    "zx": (0, 2),
    "yx": (1, 2),
}


def _cross_derivative_axis(k, n: int, *, xp=np):
    """Return first-derivative k with even-grid Nyquist set to zero.

    Spectral first derivatives on an even grid conventionally set the
    self-conjugate Nyquist coefficient to zero.  For mixed second derivatives
    this preserves the Hermitian structure required by a real inverse FFT.
    """
    out = k.copy()
    if int(n) % 2 == 0:
        out[int(n) // 2] = 0.0
    return out


def component_multiplier(
    component: str,
    kz,
    ky,
    kx,
    shape,
    *,
    xp=np,
):
    """Return the Fourier multiplier for T_ij = d_i d_j Phi."""
    if component not in COMPONENT_AXES:
        raise ValueError(f"unknown tensor component {component!r}")
    a, b = COMPONENT_AXES[component]
    axes = [kz, ky, kx]
    sizes = [int(shape[0]), int(shape[1]), int(shape[2])]

    if a == b:
        ka = axes[a]
        shapes = [(len(kz), 1, 1), (1, len(ky), 1), (1, 1, len(kx))]
        return -(ka.reshape(shapes[a]) ** 2)

    ka = _cross_derivative_axis(axes[a], sizes[a], xp=xp)
    kb = _cross_derivative_axis(axes[b], sizes[b], xp=xp)
    shapes = [(len(kz), 1, 1), (1, len(ky), 1), (1, 1, len(kx))]
    return -(ka.reshape(shapes[a]) * kb.reshape(shapes[b]))
