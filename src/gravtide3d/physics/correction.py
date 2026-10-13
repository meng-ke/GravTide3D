from __future__ import annotations

import math
from .poisson import G_CGS


def normalize_correction(value: str) -> str:
    value = str(value).strip().lower()
    aliases = {
        "dc": "dc",
        "none": "none",
        "raw": "none",
        "trace": "trace",
        "local": "trace",
    }
    if value not in aliases:
        raise ValueError("correction must be one of: 'dc', 'none', 'trace'")
    return aliases[value]


def dc_offset(density_mean: float, *, G: float = G_CGS) -> float:
    return 4.0 * math.pi * G * float(density_mean) / 3.0


def apply_isotropic_correction(T, density, density_mean, correction, *, xp):
    """Modify a (...,3,3) tensor in place and return delta metadata."""
    mode = normalize_correction(correction)
    if mode == "none":
        return None
    if mode == "dc":
        delta = dc_offset(float(density_mean))
        T[..., 0, 0] += delta
        T[..., 1, 1] += delta
        T[..., 2, 2] += delta
        return float(delta)

    # Diagnostic trace restoration, deliberately not the default.
    target = (4.0 * math.pi * G_CGS) * density
    trace = T[..., 0, 0] + T[..., 1, 1] + T[..., 2, 2]
    delta = (target - trace) / 3.0
    T[..., 0, 0] += delta
    T[..., 1, 1] += delta
    T[..., 2, 2] += delta
    return delta
