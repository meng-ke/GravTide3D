import numpy as np
import pytest

from gravtide3d import tidal_eigensystem
from gravtide3d.validation.consistency import eigenvector_alignment


def _cupy_available():
    try:
        import cupy as cp
        return cp.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


@pytest.mark.skipif(not _cupy_available(), reason="CuPy/CUDA GPU not available")
def test_cpu_gpu_small_cube():
    shape = (8, 10, 12)
    z, y, x = np.indices(shape)
    rho = (2e-20 * (1 + 0.1*np.cos(2*np.pi*z/8) + 0.03*np.cos(2*np.pi*x/12))).astype(np.float32)
    voxel = (1e17, 1e17, 1e17)
    cpu = tidal_eigensystem(rho, voxel, backend="cpu", dtype="float32", memory="in_core")
    gpu = tidal_eigensystem(rho, voxel, backend="gpu", devices=0, dtype="float32", memory="in_core")
    import cupy as cp
    glam = cp.asnumpy(gpu.eigenvalues)
    gvec = cp.asnumpy(gpu.eigenvectors)
    assert np.allclose(glam, cpu.eigenvalues, rtol=2e-5, atol=1e-30)
    assert np.nanmin(eigenvector_alignment(gvec, cpu.eigenvectors)) > 0.995
