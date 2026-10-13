# GravTide3D v0.1 development notes

## Implemented

- Public APIs: `gravitational_potential()` and `tidal_eigensystem()`.
- CPU default; optional CuPy single-GPU in-core support.
- CPU/GPU split-dimensional out-of-core FFT engine, designed for multi-GPU
  processing of a whole periodic cube rather than independent subvolumes.
- Potential and tidal result containers; eigenvalues ascending and
  eigenvectors as columns in component order `(z,y,x)`.
- Default constant `correction="dc"`, `"none"` and diagnostic `"trace"`.
- Periodic/no-padding boundary convention; no isolated-boundary solver.
- In-core optional tensor output. Out-of-core tensor output through the public
  `return_tensor=True` API is *not* implemented.

## Validation completed so far

The CPU test suite covers API shape/result semantics, the potential's removed
zero mode, the DC and raw traces, tensor eigenproperties, CPU in-core versus
out-of-core agreement, and input validation. Additional release-preparation
regressions include:

- Analytic single Fourier-mode potential amplitude and mixed tidal derivative.
- Nyquist handling for an even-grid mixed derivative.
- Scratch manager never deleting user-owned directory roots.
- Refusing to resume completed potential results after a density file changes,
  including when `overwrite=True` is requested.
- Refusing to reuse an earlier snapshot for a different in-memory input.
- Rejecting changed tidal correction for a reused workdir unless an explicit
  recomputation is requested.

**RTX 5090 user validation (2026-10):** A fresh Python 3.12 environment with
CUDA 13/CuPy was used to successfully run the notebook's single-GPU
in-core tidal eigensystem on the OMC-1 reconstructed density cube. The GPU
reduction, FFT, batched eigen-decomposition, and tidal plotting all executed.
The user compared eigenvalues against the CPU version and reported:

```text
max(abs(lambda_GPU - lambda_CPU)) = 1.4130119567339053e-30 s^-2
RMS difference                 = 1.4621009935586023e-32 s^-2
max(abs(diff)) / max(abs(CPU)) = 7.03355594916603e-7
np.allclose(CPU, GPU, rtol=1e-5, atol=1e-7 * max(abs(CPU))) = True
```

This is a **single-GPU, in-core comparison on one example**, not a validation
of every GPU mode or a guarantee of bit-for-bit identical results. The local
test environment used for preparing this release does not have CUDA; its
GPU test is skipped. In the user's original mixed CUDA 12/13 `base` environment,
NVRTC failed during JIT compilation; a clean CUDA 13 environment resolved it.

## Safe out-of-core resume behavior

A distinct run cannot silently replace/resume an existing workdir, including
when the density file changes but keeps the same name and shape. A signature
tracks path, shape, voxel sizes, selected backend/devices, scratch workspace,
and file size/mtime/ctime. This **file-stat check is not a cryptographic hash**:
treat the source file as immutable throughout a run. Use a new `output` and
scratch workspace for distinct input calculations.

The user-specified scratch path is a **root**: GravTide3D writes into a marked
`run-*` subdirectory and does not recursively delete the root. Input arrays
are snapshotted under unique names. Resumable cross-session workflows should
use persistent `.npy` sources. `ScratchManager.remove()` explicitly refuses
to remove a path that was supplied by the caller.

## Server regressions required before a stable release

1. CPU float64 versus A100 float32 reference on matching small cubes.
2. Single-GPU in-core versus single-GPU out-of-core.
3. Single-GPU versus eight-GPU out-of-core.
4. Eigenvector eigenspace comparison accounting for sign and degeneracy.
5. Independent raw/DC-corrected trace residuals.
6. Resume and interruption tests at potential/tensor/eigensystem stages.
7. Peak RAM, `/dev/shm`, GPU memory, scratch cleanup and disk requirements.
8. Full production `(724, 6930, 5400)` density-cube regression against
   independently retained production results, with metadata checks.
9. GPU package install and smoke tests on server/CUDA versions used.

## Documentation/release follow-ups

- `examples/omc1_nh2.fits` is identified by the provider as derived from
  the **Herschel Gould Belt Survey (HGBS)**; add citations for André et al.
  (2010) and HGBS dataset DOI https://doi.org/10.26131/IRSA72.
  **Still pending:** the exact original map/cutout provenance (including any
  reduction or processing paper) and permission to redistribute this file.
  The local FITS header alone does not establish these details.
- Replace generic "GravTide3D contributors" author/copyright attribution with
  maintainer-approved metadata and add canonical project / issue URLs when
  known (do not invent these).
- Add `CITATION.cff` after authors and preferred citation have been confirmed.
- Keep alpha status until the server-scale tests above are complete.
