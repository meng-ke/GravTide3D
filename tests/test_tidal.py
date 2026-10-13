import math
import numpy as np

from gravtide3d import tidal_eigensystem
from gravtide3d.physics.poisson import G_CGS
from gravtide3d.validation.tidal import (
    validate_eigenvalue_order,
    eigenvector_orthogonality_error,
)


def periodic_density(shape=(8, 10, 12)):
    z, y, x = np.indices(shape)
    return 2e-20 * (
        1.0
        + 0.10 * np.cos(2 * np.pi * z / shape[0])
        + 0.07 * np.cos(2 * np.pi * y / shape[1])
        + 0.05 * np.cos(2 * np.pi * x / shape[2])
    )


def test_corrected_trace_matches_poisson_source():
    rho = periodic_density()
    result = tidal_eigensystem(rho, (1.2e17, 1.1e17, 0.9e17), memory="in_core")
    target = 4.0 * math.pi * G_CGS * rho
    trace = np.sum(result.eigenvalues, axis=-1)
    scale = np.max(np.abs(target))
    assert np.max(np.abs(trace - target)) / scale < 5e-12
    assert validate_eigenvalue_order(result.eigenvalues)
    assert eigenvector_orthogonality_error(result.eigenvectors) < 5e-12


def test_raw_trace_is_density_contrast():
    rho = periodic_density()
    result = tidal_eigensystem(
        rho,
        (1.2e17, 1.1e17, 0.9e17),
        correction="none",
        memory="in_core",
    )
    target = 4.0 * math.pi * G_CGS * (rho - rho.mean())
    trace = np.sum(result.eigenvalues, axis=-1)
    scale = np.max(np.abs(target))
    assert np.max(np.abs(trace - target)) / scale < 5e-12
