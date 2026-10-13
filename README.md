# GravTide3D

**GravTide3D** calculates the gravitational potential and tidal-tensor
**eigenvalues/eigenvectors** of a three-dimensional mass-density field using
Fourier methods. It provides a small Python API, CPU execution by default,
optional CuPy acceleration, and an out-of-core split FFT for large cubes.

**Status: v0.1.0 alpha.** CPU in-core/out-of-core and single-GPU in-core
execution have been exercised; large-scale multi-GPU and GPU out-of-core
regressions are still required before a stable release.

## Installation

Requires Python >= 3.10 and NumPy. From the source checkout:

```bash
python -m pip install -e .
```

**GPU is optional.** Install **one** CuPy wheel appropriate to your CUDA
installation, for example:

```bash
# Choose ONE, not both. The [ctk] extra provides compatible toolkit components.
python -m pip install "cupy-cuda12x[ctk]"  # CUDA 12
# OR
python -m pip install "cupy-cuda13x[ctk]"  # CUDA 13
```

Do **not** install both CuPy CUDA wheels in the same environment. We recommend
a separate Python/conda environment for GPU work. The package does not install
CuPy automatically and does not select your CUDA major version for you.

## Quick start

```python
import numpy as np
from gravtide3d import gravitational_potential, tidal_eigensystem

rho = np.full((16, 20, 24), 2.0e-20)  # g cm^-3; (z, y, x)
voxel_size = (1.0e17, 1.0e17, 1.0e17)  # cm; (dz, dy, dx)

potential = gravitational_potential(rho, voxel_size)
tidal = tidal_eigensystem(rho, voxel_size)

phi = potential.phi                # (Nz, Ny, Nx), cm^2 s^-2
lam = tidal.eigenvalues           # (Nz, Ny, Nx, 3), s^-2
vec = tidal.eigenvectors          # (Nz, Ny, Nx, 3, 3)
lambda_min = lam[..., 0]
```

The uniform-density example above has zero periodic potential and a nonzero
constant tidal eigenvalue after the default DC correction. For a nonuniform
example with figures, see [`examples/example.ipynb`](examples/example.ipynb).

## Scientific conventions and limitations

| Quantity | Convention |
| --- | --- |
| Mass-density input | Three-dimensional `(Nz, Ny, Nx)` array, in `g cm^-3` |
| Physical voxel sizes | `(dz, dy, dx)`, in `cm` |
| Gravitational potential `Phi` | `cm^2 s^-2` |
| Poisson equation | `∇² Phi = 4πG rho`, with `G = 6.67430e-8` in cgs |
| Tidal tensor | `T_ij = +∂_i∂_j Phi`, in `s^-2` |
| Positive tidal eigenvalue | Compressive (negative: stretching) |
| Eigenvalue order | Ascending: `(lambda_min, lambda_mid, lambda_max)` |
| Eigenvectors | Columns of `vec`; component order `(z, y, x)` |

**Boundary conditions matter.** The solver uses a **periodic FFT grid without
zero-padding** and sets the gravitational-potential `k=0` mode to zero. This is
a periodic Poisson solution for density contrast, not an isolated/free-space
gravity solver. For observational cubes, examine edge effects and whether the
periodic assumption is appropriate before interpreting the tidal field.

**DC correction.** The default `correction="dc"` adds the constant
`delta_DC = 4πG * mean(rho) / 3` to each of the three eigenvalues, restoring
the uniform-density isotropic term. Consequently, the corrected trace is
approximately `4πG*rho`. With `correction="none"` (or `"raw"`), the trace
is approximately `4πG*(rho - mean(rho))`. `correction="trace"` is a local
trace-restoration **diagnostic**, not the default physical convention.

Eigenvector signs are arbitrary; eigenvectors can rotate within degenerate or
nearly degenerate eigenspaces. Compare directions using absolute dot products
or subspaces, not raw signed components.

## GPU acceleration

```python
import cupy as cp

result = tidal_eigensystem(
    rho,
    voxel_size,
    backend="gpu",
    devices=0,       # CUDA device index
    memory="in_core",
)

# In-core GPU results are CuPy arrays. Transfer before NumPy/Matplotlib work.
lam_gpu_on_cpu = cp.asnumpy(result.eigenvalues)
```

The API also accepts a list of device IDs or `devices="all"` for split-FFT
multi-GPU execution; that path requires out-of-core storage and remains
**experimental pending server-scale regression**.

## Large cubes and out-of-core execution

For input cubes saved as `.npy` files:

```python
from gravtide3d import tidal_eigensystem, ExecutionConfig

tidal = tidal_eigensystem(
    "density.npy",                     # shape (Nz, Ny, Nx), g cm^-3
    (1.0e17, 1.0e17, 1.0e17),
    backend="cpu",                    # or "gpu" with CuPy installed
    memory="out_of_core",
    output="./tidal_run",              # persistent results and run metadata
    scratch="/path/to/scratch",        # disk or sufficiently large /dev/shm
    config=ExecutionConfig(resume=True),
)

# Results are file-backed NumPy arrays (memmaps).
print(tidal.eigenvalues.shape, tidal.eigenvectors.shape)
```

The split FFT is performed on the **whole periodic cube** via separable FFT
stages; it does not solve gravity independently in spatial chunks.

**Out-of-core limitations:**

- The current split-FFT implementation stores real fields as **float32** and
  spectra as **complex64**, even if `dtype="float64"` is requested.
- `return_tensor=True` is **in-core only**; it raises `NotImplementedError`
  in out-of-core mode.
- In-core CPU returns NumPy arrays; in-core GPU returns CuPy arrays;
  out-of-core outputs are file-backed NumPy arrays.
- A run's `output` directory holds final files and state. `scratch` is treated
  as a **root** under which GravTide3D creates a marked, per-output workspace;
  unrelated files at the root are left alone. Scratch storage can be large.
- Resume is intended for the **same density input and run configuration**.
  The implementation checks the input file's path, size, modification/change
  timestamps, geometry, backend, devices and workspace before reusing results.
  Keep input files immutable while a run executes. These file-stat checks are
  **not** a cryptographic content hash. For distinct calculations, use a new
  `output` (and preferably a new scratch root); even `overwrite=True` will
  refuse to replace a different calculation's workdir.
- For resumable runs, prefer a **persistent `.npy` input file**. In-memory
  input arrays get a fresh snapshot for each out-of-core API call, and thus
  cannot silently reuse a previous run's input.
- To change an existing tidal correction, use a new output directory, or
  intentionally recompute with `ExecutionConfig(overwrite=True)`.

## Examples and testing

- [`examples/example.ipynb`](examples/example.ipynb) walks through OMC-1 H₂
  column density, optional VDM reconstruction, CPU potential, CPU tidal
  eigensystem, and an **optional GPU tidal** demonstration. The notebook needs
  `astropy`, `matplotlib`, and [Volume Density Mapper](https://github.com/gxli/volume-density-mapper)
  in addition to GravTide3D; see [`examples/README.md`](examples/README.md).
- The included `omc1_nh2.fits` is an **OMC-1 H₂ column-density cutout derived
  from the Herschel Gould Belt Survey (HGBS)**, as identified by the dataset
  provider. It is tutorial data, **not** a required package dependency.
  Please acknowledge the HGBS survey via **André et al. (2010)** and the
  [HGBS archive dataset DOI](https://doi.org/10.26131/IRSA72), as requested
  by the [IRSA HGBS data portal](https://irsa.ipac.caltech.edu/data/Herschel/HGBS/overview.html).
  The FITS header does not identify the exact parent mosaic, map-processing
  publication, or redistribution terms for this particular cutout. Verify
  those details and permission to rehost the FITS file before a public release.

```bash
python -m pip install pytest
python -m pytest -q
```

The test suite covers the CPU solvers, analytic Fourier modes, resume safety,
and (when CuPy/CUDA is present) CPU-versus-GPU comparisons. Local CI cannot
substitute for large GPU-server validation.

## Validation status, citation, and license

- **CPU in-core / CPU out-of-core:** tested on small periodic cubes.
- **Single-GPU in-core:** user-tested on an NVIDIA GeForce RTX 5090 with CuPy
  and CUDA 13. On the OMC-1 example, CPU/GPU tidal eigenvalues had maximum
  normalized absolute difference about `7.03e-7`; see
  [`DEVELOPMENT_NOTES.md`](DEVELOPMENT_NOTES.md) for the comparison definition.
- **Multi-GPU / GPU out-of-core / full production-scale:** validation pending.

The software is MIT-licensed; see [`LICENSE`](LICENSE). Formal software
citation details are pending maintainer confirmation. For this notebook's
observational data, cite the **Herschel Gould Belt Survey (HGBS)**:
**André et al. (2010)** and dataset DOI
[10.26131/IRSA72](https://doi.org/10.26131/IRSA72), as directed by IRSA.
Cite the specific map-processing product/paper as well, once its lineage is
confirmed. Cite VDM independently if used.
