from pathlib import Path
import numpy as np

from gravtide3d import ExecutionConfig, gravitational_potential, tidal_eigensystem
from gravtide3d.validation.consistency import eigenvector_alignment


def test_cpu_outofcore_matches_float32_incore(tmp_path: Path):
    shape = (6, 8, 10)
    z, y, x = np.indices(shape)
    rho = (2e-20 * (
        1.0 + 0.10*np.cos(2*np.pi*z/shape[0])
        + 0.04*np.cos(2*np.pi*y/shape[1])
        + 0.03*np.cos(2*np.pi*x/shape[2])
    )).astype(np.float32)
    density_path = tmp_path / "rho.npy"
    np.save(density_path, rho)
    voxel = (1e17, 1.1e17, 0.9e17)

    inc = tidal_eigensystem(
        density_path, voxel, dtype="float32", memory="in_core", backend="cpu"
    )
    cfg = ExecutionConfig(
        workers=1,
        spectral_tile=(4, 4),
        eig_tile=(4, 5),
        cleanup_scratch=True,
    )
    out = tidal_eigensystem(
        density_path,
        voxel,
        dtype="float32",
        memory="out_of_core",
        backend="cpu",
        output=tmp_path / "ooc",
        scratch=tmp_path / "scratch",
        config=cfg,
    )

    assert np.allclose(out.eigenvalues, inc.eigenvalues, rtol=3e-5, atol=1e-30)
    alignment = eigenvector_alignment(out.eigenvectors, inc.eigenvectors)
    # Degenerate eigenvalues can rotate their eigenspaces; the chosen density
    # normally avoids exact degeneracy, so this is a useful regression check.
    assert np.nanmin(alignment) > 0.995
