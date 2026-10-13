from __future__ import annotations

import hashlib
import json
import uuid
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from ..config import ExecutionConfig
from ..physics.correction import dc_offset, normalize_correction
from ..results import PotentialResult, TidalResult
from .._internal import production_splitfft as engine


def _ensure_output(output) -> Path:
    if output is None:
        raise ValueError(
            "out-of-core execution requires output=<directory> so final arrays have "
            "a persistent destination"
        )
    path = Path(output).expanduser().resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path


def _ensure_scratch(scratch, output: Path) -> Path:
    """Use a marked, per-output workspace inside the requested scratch root."""
    root = Path(scratch).expanduser().resolve() if scratch is not None else output / ".gravtide3d_scratch"
    key = hashlib.sha256(str(output.resolve()).encode("utf-8")).hexdigest()[:16]
    path = root / f"run-{key}"
    marker = path / ".gravtide3d-owned"
    if path.exists() and not marker.is_file():
        raise ValueError(f"Unmarked scratch workspace {path}; refusing to reuse it")
    path.mkdir(parents=True, exist_ok=True)
    if not marker.exists():
        marker.write_text("GravTide3D scratch workspace\n", encoding="utf-8")
    return path


def _density_path(density, scratch: Path) -> Path:
    if isinstance(density, (str, Path)):
        path = Path(density).expanduser().resolve()
        if path.suffix.lower() != ".npy":
            raise ValueError("out-of-core density input must be a .npy file")
        return path
    # A fresh, immutable snapshot per call prevents different in-memory arrays
    # from ever silently reusing a previous out-of-core input. For resumable
    # calculations across separate calls, use a persistent .npy file instead.
    path = scratch / f"density_input_{uuid.uuid4().hex}.npy"
    temporary = path.with_suffix(".tmp")
    try:
        with temporary.open("wb") as f:
            np.save(f, np.asarray(density, dtype=np.float32))
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def _device_string(plan) -> str:
    return ",".join(str(v) for v in plan.devices) if plan.devices else "0"


def _potential_args(density_path, voxel_size, output, scratch, plan, config):
    return SimpleNamespace(
        density=str(density_path),
        voxel_size=tuple(float(v) for v in voxel_size),
        devices=_device_string(plan),
        backend=plan.backend,
        workers=int(config.workers),
        workdir=str(output),
        scratchdir=str(scratch),
        spectral_tile=tuple(int(v) for v in config.spectral_tile),
        resume=bool(config.resume),
        overwrite=bool(config.overwrite),
        keep_rho_xy=False,
    )


def _cleanup_args(output, *, delete_phi=False, delete_tensor=False, delete_eigen=False):
    return SimpleNamespace(
        workdir=str(output),
        delete_transient=True,
        delete_phi_shards=bool(delete_phi),
        delete_tensor=bool(delete_tensor),
        delete_eigen_shards=bool(delete_eigen),
    )


def compute_potential_outofcore(
    density,
    voxel_size,
    *,
    plan,
    output,
    scratch=None,
    config: ExecutionConfig,
    keep_spectrum=False,
):
    output = _ensure_output(output)
    scratch = _ensure_scratch(scratch, output)
    density_path = _density_path(density, scratch)
    args = _potential_args(density_path, voxel_size, output, scratch, plan, config)
    state = engine.stage_potential(args)

    spectral = Path(state.scratchdir) / "phi_hat_yshards"
    if not keep_spectrum:
        engine.stage_cleanup(_cleanup_args(output, delete_phi=True))
        spectral_path = None
    else:
        spectral_path = spectral

    phi = np.load(output / "phi3d.npy", mmap_mode="r")
    return PotentialResult(
        phi=phi,
        phi_hat=None,
        shape=tuple(state.shape),
        voxel_size=tuple(state.voxel_size_cm),
        density_mean=float(state.density_mean_g_cm3),
        backend=state.backend,
        memory_mode="out_of_core",
        output_path=output,
        spectral_path=spectral_path,
        density_source=Path(state.density_path),
        metadata={
            "dtype": "float32",
            "complex_dtype": "complex64",
            "boundary": state.boundary,
            "devices": list(state.devices),
        },
    )


def _ensure_ooc_potential(source, voxel_size, output, scratch, plan, config):
    if isinstance(source, PotentialResult):
        density_source = source.density_source
        if density_source is None:
            raise ValueError(
                "out-of-core tidal calculation from PotentialResult requires a density_source"
            )
        # Use the existing workdir when possible, because its pipeline_state tracks
        # the spectrum shards. Re-running stage_potential with resume=True rebuilds
        # missing shards if the user previously released them.
        workdir = source.output_path or _ensure_output(output)
        if scratch is not None:
            scratchdir = _ensure_scratch(scratch, workdir)
        elif (workdir / "pipeline_state.json").is_file():
            # A PotentialResult created with a custom scratch root may have
            # spectral_path=None after the spectrum was released. Recover the
            # exact original workspace from the recorded pipeline state.
            scratchdir = Path(engine.PipelineState.load(workdir).scratchdir)
            if not (scratchdir / ".gravtide3d-owned").is_file():
                raise ValueError("Saved scratch workspace is unmarked or missing")
        else:
            scratchdir = _ensure_scratch(None, workdir)
        args = _potential_args(density_source, voxel_size, workdir, scratchdir, plan, config)
        state = engine.stage_potential(args)
        return state, workdir, scratchdir

    workdir = _ensure_output(output)
    scratchdir = _ensure_scratch(scratch, workdir)
    density_path = _density_path(source, scratchdir)
    args = _potential_args(density_path, voxel_size, workdir, scratchdir, plan, config)
    state = engine.stage_potential(args)
    return state, workdir, scratchdir


def compute_tidal_outofcore(
    source,
    voxel_size,
    *,
    plan,
    correction="dc",
    output=None,
    scratch=None,
    config: ExecutionConfig,
    return_tensor=False,
):
    if return_tensor:
        raise NotImplementedError(
            "return_tensor=True is not yet exposed for out-of-core mode; the six "
            "tensor shard sets exist in scratch during the calculation"
        )

    mode = normalize_correction(correction)
    legacy_mode = "local" if mode == "trace" else mode
    state, workdir, scratchdir = _ensure_ooc_potential(
        source, voxel_size, output, scratch, plan, config
    )

    existing_metadata = workdir / "tidal_metadata.json"
    if not existing_metadata.exists() and (workdir / "eigenvalues.npy").exists() and not config.overwrite:
        raise ValueError(
            "Existing eigenvalues have no tidal metadata; cannot safely resume. "
            "Use a new workdir or explicitly set config.overwrite=True."
        )
    if existing_metadata.exists():
        previous = json.loads(existing_metadata.read_text(encoding="utf-8"))
        previous_mode = previous.get("trace_correction")
        if previous_mode == "local":
            previous_mode = "trace"
        if previous_mode is not None and previous_mode != mode and not config.overwrite:
            raise ValueError(
                "Existing tidal output uses a different correction. "
                "Choose a new output directory or set config.overwrite=True."
            )

    common = dict(
        workdir=str(workdir),
        scratchdir=str(scratchdir),
        spectral_tile=tuple(int(v) for v in config.spectral_tile),
        resume=bool(config.resume),
        overwrite=bool(config.overwrite),
    )
    engine.stage_tensor(SimpleNamespace(**common))
    engine.stage_eigensystem(
        SimpleNamespace(
            workdir=str(workdir),
            gpu_tile=tuple(int(v) for v in config.eig_tile),
            trace_correction=legacy_mode,
            resume=bool(config.resume),
            overwrite=bool(config.overwrite),
        )
    )

    meta_path = workdir / "tidal_metadata.json"
    metadata = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    # Present the public diagnostic name even though the production adapter uses
    # the historical internal spelling "local".
    if mode == "trace":
        metadata["trace_correction"] = "trace"
        meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    evals = np.load(workdir / "eigenvalues.npy", mmap_mode="r")
    evecs = np.load(workdir / "eigenvectors.npy", mmap_mode="r")
    delta = dc_offset(state.density_mean_g_cm3) if mode == "dc" else None

    result = TidalResult(
        eigenvalues=evals,
        eigenvectors=evecs,
        tensor=None,
        shape=tuple(state.shape),
        voxel_size=tuple(state.voxel_size_cm),
        density_mean=float(state.density_mean_g_cm3),
        correction=mode,
        delta_dc=delta,
        backend=state.backend,
        memory_mode="out_of_core",
        output_path=workdir,
        metadata=metadata,
    )

    if config.cleanup_scratch:
        engine.stage_cleanup(
            _cleanup_args(
                workdir,
                delete_phi=True,
                delete_tensor=True,
                delete_eigen=True,
            )
        )
        # Keep the marked workspace (and any input snapshot) for safe reuse.
        # The caller can explicitly remove its run-* directory after finishing.
    return result
