import numpy as np

from gravtide3d import gravitational_potential, tidal_eigensystem


def test_public_shapes():
    rho = np.ones((6, 8, 10), dtype=np.float64) * 2e-20
    voxel = (1.0e17, 1.0e17, 1.0e17)
    p = gravitational_potential(rho, voxel, memory="in_core")
    t = tidal_eigensystem(rho, voxel, memory="in_core")
    assert p.phi.shape == rho.shape
    assert t.eigenvalues.shape == rho.shape + (3,)
    assert t.eigenvectors.shape == rho.shape + (3, 3)
    assert t.correction == "dc"
