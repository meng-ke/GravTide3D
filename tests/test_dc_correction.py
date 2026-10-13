import math
import numpy as np

from gravtide3d import tidal_eigensystem
from gravtide3d.physics.poisson import G_CGS


def test_uniform_density_dc_eigenvalues():
    rho0 = 4e-20
    rho = np.full((6, 8, 10), rho0, dtype=np.float64)
    result = tidal_eigensystem(rho, (1e17, 1e17, 1e17), memory="in_core")
    expected = 4.0 * math.pi * G_CGS * rho0 / 3.0
    assert np.allclose(result.eigenvalues, expected, rtol=1e-12, atol=0.0)
    assert math.isclose(result.delta_dc, expected, rel_tol=1e-14)
