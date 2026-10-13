import numpy as np

from gravtide3d import tidal_eigensystem


def test_tidal_result_unpacking():
    rho = np.ones((4, 5, 6), dtype=np.float64) * 1e-20
    result = tidal_eigensystem(rho, (1e17, 1e17, 1e17), memory="in_core")
    lam, vec = result
    assert lam is result.eigenvalues
    assert vec is result.eigenvectors
