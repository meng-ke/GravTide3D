from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from ..backends.cpu import CPUBackend
from ..backends.gpu import GPUBackend
from ..physics.poisson import G_CGS, make_k_axes, poisson_factor
from ..physics.tidal import COMPONENTS, COMPONENT_AXES, component_multiplier
from ..physics.correction import apply_isotropic_correction, dc_offset, normalize_correction
from ..results import PotentialResult, TidalResult


def _backend(name: str, devices: tuple[int, ...]):
    if name == "cpu":
        return CPUBackend()
    if len(devices) != 1:
        raise ValueError("in-core GPU execution requires exactly one device")
    return GPUBackend(devices[0])


def _load_density(source, backend, dtype):
    if isinstance(source, (str, Path)):
        host = np.load(Path(source), mmap_mode="r")
        return backend.asarray(host, dtype=dtype), Path(source).expanduser().resolve()
    return backend.asarray(source, dtype=dtype), None


def _save_metadata(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _mean_as_float(xp, arr) -> float:
    value = xp.mean(arr, dtype=xp.float64)
    if xp is np:
        return float(value)
    return float(value.get())


def _potential_spectrum(rho, voxel_size, *, xp, density_mean=None):
    if density_mean is None:
        density_mean = xp.mean(rho, dtype=xp.float64)
    # Solve the periodic Poisson equation for the density contrast explicitly.
    # This is mathematically equivalent to zeroing k=0, but avoids round-off
    # leakage from a large uniform background into non-zero Fourier modes.
    rho_hat = xp.fft.rfftn(rho - density_mean, axes=(0, 1, 2))
    kz, ky, kx = make_k_axes(rho.shape, voxel_size, xp=xp)
    factor = poisson_factor(kz, ky, kx, xp=xp)
    # Match spectrum precision to the FFT result rather than upcasting it.
    factor = factor.astype(rho_hat.real.dtype, copy=False)
    phi_hat = rho_hat * factor
    return phi_hat, (kz, ky, kx)


def compute_potential_incore(
    density,
    voxel_size,
    *,
    plan,
    output=None,
    keep_spectrum=False,
):
    backend = _backend(plan.backend, plan.devices)
    xp = backend.xp
    dtype = xp.float32 if plan.dtype == np.dtype("float32") else xp.float64
    rho, source_path = _load_density(density, backend, dtype)
    density_mean = _mean_as_float(xp, rho)

    phi_hat, _ = _potential_spectrum(rho, voxel_size, xp=xp, density_mean=density_mean)
    phi = xp.fft.irfftn(phi_hat, s=rho.shape, axes=(0, 1, 2)).real
    phi = phi.astype(dtype, copy=False)

    output_path = None
    spectral_path = None
    phi_result = phi
    phi_hat_result = phi_hat if keep_spectrum else None
    if output is not None:
        output_path = Path(output).expanduser().resolve()
        output_path.mkdir(parents=True, exist_ok=True)
        np.save(output_path / "phi3d.npy", backend.to_numpy(phi))
        phi_result = np.load(output_path / "phi3d.npy", mmap_mode="r")
        if keep_spectrum:
            np.save(output_path / "phi_hat.npy", backend.to_numpy(phi_hat))
            spectral_path = output_path / "phi_hat.npy"
            phi_hat_result = np.load(spectral_path, mmap_mode="r")
        _save_metadata(
            output_path / "potential_metadata.json",
            {
                "shape": list(map(int, rho.shape)),
                "axis_order": ["z", "y", "x"],
                "voxel_size_cm": list(map(float, voxel_size)),
                "density_mean_g_cm3": density_mean,
                "backend": plan.backend,
                "memory_mode": "in_core",
                "dtype": str(plan.dtype),
                "boundary": "periodic FFT grid; no zero-padding",
                "phi_units": "cm^2 s^-2",
            },
        )

    result = PotentialResult(
        phi=phi_result,
        phi_hat=phi_hat_result,
        shape=tuple(int(v) for v in rho.shape),
        voxel_size=tuple(float(v) for v in voxel_size),
        density_mean=density_mean,
        backend=plan.backend,
        memory_mode="in_core",
        output_path=output_path,
        spectral_path=spectral_path,
        density_source=source_path,
        metadata={"dtype": str(plan.dtype), "boundary": "periodic FFT grid; no zero-padding", "devices": list(plan.devices)},
    )
    if output is not None:
        backend.release()
    return result


def _source_to_spectrum(source, voxel_size, backend, dtype):
    xp = backend.xp
    if isinstance(source, PotentialResult):
        if source.phi_hat is None:
            raise ValueError(
                "PotentialResult does not contain phi_hat. Recompute potential with "
                "keep_spectrum=True, or call tidal_eigensystem with the density field."
            )
        if tuple(source.voxel_size) != tuple(voxel_size):
            raise ValueError("PotentialResult voxel_size does not match the requested voxel_size")
        phi_hat = backend.asarray(source.phi_hat)
        shape = source.shape
        density_mean = source.density_mean
        return None, phi_hat, shape, density_mean, source.density_source

    rho, source_path = _load_density(source, backend, dtype)
    density_mean = _mean_as_float(xp, rho)
    phi_hat, _ = _potential_spectrum(rho, voxel_size, xp=xp, density_mean=density_mean)
    return rho, phi_hat, tuple(int(v) for v in rho.shape), density_mean, source_path


def compute_tidal_incore(
    source,
    voxel_size,
    *,
    plan,
    correction="dc",
    output=None,
    return_tensor=False,
):
    backend = _backend(plan.backend, plan.devices)
    xp = backend.xp
    dtype = xp.float32 if plan.dtype == np.dtype("float32") else xp.float64
    mode = normalize_correction(correction)

    rho, phi_hat, shape, density_mean, source_path = _source_to_spectrum(
        source, voxel_size, backend, dtype
    )
    kz, ky, kx = make_k_axes(shape, voxel_size, xp=xp)

    T = xp.empty(shape + (3, 3), dtype=dtype)
    for component in COMPONENTS:
        a, b = COMPONENT_AXES[component]
        mult = component_multiplier(component, kz, ky, kx, shape, xp=xp)
        mult = mult.astype(phi_hat.real.dtype, copy=False)
        field_hat = phi_hat * mult
        field = xp.fft.irfftn(field_hat, s=shape, axes=(0, 1, 2)).real
        field = field.astype(dtype, copy=False)
        T[..., a, b] = field
        T[..., b, a] = field
        del mult, field_hat, field

    if rho is None and mode == "trace":
        if source_path is None:
            raise ValueError(
                "correction='trace' requires the density field; pass density directly "
                "or use a PotentialResult with density_source."
            )
        rho, _ = _load_density(source_path, backend, dtype)

    delta = apply_isotropic_correction(
        T, rho, density_mean, mode, xp=xp
    )
    evals, evecs = xp.linalg.eigh(T)
    evals = evals.astype(dtype, copy=False)
    evecs = evecs.astype(dtype, copy=False)

    delta_dc_value = dc_offset(density_mean) if mode == "dc" else None
    output_path = None
    tensor_result = T if return_tensor else None
    eval_result = evals
    evec_result = evecs

    if output is not None:
        output_path = Path(output).expanduser().resolve()
        output_path.mkdir(parents=True, exist_ok=True)
        np.save(output_path / "eigenvalues.npy", backend.to_numpy(evals))
        np.save(output_path / "eigenvectors.npy", backend.to_numpy(evecs))
        eval_result = np.load(output_path / "eigenvalues.npy", mmap_mode="r")
        evec_result = np.load(output_path / "eigenvectors.npy", mmap_mode="r")
        if return_tensor:
            np.save(output_path / "tidal_tensor.npy", backend.to_numpy(T))
            tensor_result = np.load(output_path / "tidal_tensor.npy", mmap_mode="r")
        _save_metadata(
            output_path / "tidal_metadata.json",
            {
                "shape": list(shape),
                "axis_order": ["z", "y", "x"],
                "voxel_size_cm": list(map(float, voxel_size)),
                "density_mean_g_cm3": density_mean,
                "trace_correction": mode,
                "delta_dc_s-2": delta_dc_value,
                "eigenvalue_order": ["lambda_min", "lambda_mid", "lambda_max"],
                "eigenvectors_are_columns": True,
                "vector_component_order": ["z", "y", "x"],
                "backend": plan.backend,
                "memory_mode": "in_core",
                "dtype": str(plan.dtype),
                "boundary": "periodic FFT grid; no zero-padding",
            },
        )
        backend.release()

    return TidalResult(
        eigenvalues=eval_result,
        eigenvectors=evec_result,
        tensor=tensor_result,
        shape=shape,
        voxel_size=tuple(float(v) for v in voxel_size),
        density_mean=density_mean,
        correction=mode,
        delta_dc=delta_dc_value,
        backend=plan.backend,
        memory_mode="in_core",
        output_path=output_path,
        metadata={"dtype": str(plan.dtype), "boundary": "periodic FFT grid; no zero-padding", "devices": list(plan.devices)},
    )
