from __future__ import annotations

from pathlib import Path
import numpy as np

from .config import ExecutionConfig
from .planning import make_execution_plan
from .results import PotentialResult
from .utils.axes import normalize_voxel_size
from .validation.inputs import validate_density
from .execution.incore import compute_potential_incore, compute_tidal_incore
from .execution.outofcore import compute_potential_outofcore, compute_tidal_outofcore


def _shape_of(source):
    if isinstance(source, PotentialResult):
        return source.shape
    if isinstance(source, (str, Path)):
        arr = np.load(Path(source), mmap_mode="r")
        shape = tuple(int(v) for v in arr.shape)
        del arr
        return shape
    return tuple(int(v) for v in np.shape(source))


def gravitational_potential(
    density,
    voxel_size,
    *,
    backend="cpu",
    devices=None,
    dtype=None,
    memory="auto",
    output=None,
    scratch=None,
    keep_spectrum=False,
    config: ExecutionConfig | None = None,
):
    """Compute the 3-D gravitational potential from mass density.

    Parameters
    ----------
    density:
        3-D array-like or path to a ``.npy`` array, ordered ``(z,y,x)`` and in
        ``g cm^-3``.
    voxel_size:
        ``(dz,dy,dx)`` in cm.
    backend:
        ``"cpu"`` (default) or ``"gpu"``.
    devices:
        GPU ID, sequence of IDs, or ``"all"``. Multiple GPUs use the exact
        out-of-core split FFT engine.
    memory:
        ``"auto"``, ``"in_core"``, or ``"out_of_core"``.
    output:
        Optional output directory. Required in out-of-core mode.
    keep_spectrum:
        Retain the Fourier potential for efficient chaining when practical.
    """
    cfg = config or ExecutionConfig()
    voxel_size = normalize_voxel_size(voxel_size)
    if cfg.validate_input:
        validate_density(density)
    shape = _shape_of(density)
    plan = make_execution_plan(
        shape,
        task="potential",
        backend=backend,
        devices=devices,
        dtype=dtype,
        memory=memory,
        memory_fraction=cfg.memory_fraction,
    )
    if plan.mode == "in_core":
        return compute_potential_incore(
            density, voxel_size, plan=plan, output=output, keep_spectrum=keep_spectrum
        )
    return compute_potential_outofcore(
        density,
        voxel_size,
        plan=plan,
        output=output,
        scratch=scratch,
        config=cfg,
        keep_spectrum=keep_spectrum,
    )


def tidal_eigensystem(
    density_or_potential,
    voxel_size=None,
    *,
    backend=None,
    devices=None,
    dtype=None,
    memory="auto",
    correction="dc",
    output=None,
    scratch=None,
    return_tensor=False,
    config: ExecutionConfig | None = None,
):
    """Compute tidal eigenvalues and eigenvectors.

    The default output eigenvalues are after the constant DC correction

        delta_DC = 4*pi*G*mean(rho)/3.

    Eigenvalues are ascending. Eigenvectors are columns, with vector-component
    order ``(z,y,x)``.
    """
    cfg = config or ExecutionConfig()
    source = density_or_potential
    if isinstance(source, PotentialResult):
        if voxel_size is None:
            voxel_size = source.voxel_size
        if backend is None:
            backend = source.backend
        if devices is None and source.backend == "gpu":
            stored_devices = source.metadata.get("devices")
            if stored_devices:
                devices = tuple(int(v) for v in stored_devices)
        if output is None and source.memory_mode == "out_of_core":
            output = source.output_path
    else:
        backend = backend or "cpu"
        if cfg.validate_input:
            validate_density(source)
    voxel_size = normalize_voxel_size(voxel_size)
    backend = backend or "cpu"
    shape = _shape_of(source)
    plan = make_execution_plan(
        shape,
        task="tidal",
        backend=backend,
        devices=devices,
        dtype=dtype,
        memory=memory,
        memory_fraction=cfg.memory_fraction,
    )
    if plan.mode == "in_core":
        return compute_tidal_incore(
            source,
            voxel_size,
            plan=plan,
            correction=correction,
            output=output,
            return_tensor=return_tensor,
        )
    return compute_tidal_outofcore(
        source,
        voxel_size,
        plan=plan,
        correction=correction,
        output=output,
        scratch=scratch,
        config=cfg,
        return_tensor=return_tensor,
    )
