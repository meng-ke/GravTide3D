import numpy as np

from gravtide3d import gravitational_potential


def test_uniform_density_has_zero_periodic_potential():
    # The periodic FFT solver removes the k=0 potential mode.
    rho = np.full((8, 10, 12), 3e-20, dtype=np.float64)
    result = gravitational_potential(rho, (2e17, 2e17, 2e17), memory="in_core")
    assert np.max(np.abs(result.phi)) < 1e-20
