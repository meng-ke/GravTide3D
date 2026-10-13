from __future__ import annotations

import math
import numpy as np

from ..physics.poisson import G_CGS


def validate_eigenvalue_order(eigenvalues, *, atol=0.0) -> bool:
    lam = np.asarray(eigenvalues)
    return bool(
        np.all(lam[..., 0] <= lam[..., 1] + atol)
        and np.all(lam[..., 1] <= lam[..., 2] + atol)
    )


def eigenvector_orthogonality_error(eigenvectors) -> float:
    vec = np.asarray(eigenvectors)
    gram = np.swapaxes(vec, -1, -2) @ vec
    eye = np.eye(3, dtype=gram.dtype)
    return float(np.max(np.abs(gram - eye)))


def trace_residual(eigenvalues, density, *, corrected=True, density_mean=None):
    lam = np.asarray(eigenvalues)
    rho = np.asarray(density)
    if corrected:
        target = 4.0 * math.pi * G_CGS * rho
    else:
        if density_mean is None:
            density_mean = float(np.mean(rho, dtype=np.float64))
        target = 4.0 * math.pi * G_CGS * (rho - density_mean)
    return np.sum(lam, axis=-1) - target
