from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .utils.memory import available_ram_bytes, visible_gpu_memory_bytes


@dataclass(frozen=True, slots=True)
class ExecutionPlan:
    backend: str
    mode: str
    devices: tuple[int, ...]
    dtype: np.dtype
    estimated_peak_bytes: int
    available_bytes: int | None
    reason: str

    @property
    def multi_gpu(self) -> bool:
        return self.backend == "gpu" and len(self.devices) > 1


def normalize_backend(backend: str) -> str:
    value = str(backend).strip().lower()
    if value not in {"cpu", "gpu"}:
        raise ValueError("backend must be 'cpu' or 'gpu'")
    return value


def normalize_devices(backend: str, devices) -> tuple[int, ...]:
    backend = normalize_backend(backend)
    if backend == "cpu":
        return ()

    if devices is None:
        return (0,)
    if isinstance(devices, int):
        if devices < 0:
            raise ValueError("GPU device IDs must be non-negative")
        return (int(devices),)
    if isinstance(devices, str):
        if devices.strip().lower() != "all":
            raise ValueError("string devices must be 'all'")
        try:
            import cupy as cp
        except ImportError as exc:
            raise ImportError("devices='all' requires CuPy") from exc
        count = int(cp.cuda.runtime.getDeviceCount())
        if count < 1:
            raise RuntimeError("no visible CUDA GPUs")
        return tuple(range(count))

    result = tuple(int(v) for v in devices)
    if not result:
        raise ValueError("devices cannot be empty")
    if min(result) < 0 or len(set(result)) != len(result):
        raise ValueError("GPU device IDs must be unique non-negative integers")
    return result


def default_dtype(backend: str, dtype=None) -> np.dtype:
    if dtype is not None:
        dt = np.dtype(dtype)
        if dt not in {np.dtype("float32"), np.dtype("float64")}:
            raise ValueError("dtype must be float32 or float64")
        return dt
    return np.dtype("float64" if backend == "cpu" else "float32")


def estimate_peak_bytes(shape, dtype, task: str) -> int:
    """Conservative in-core estimate, used only for planning."""
    nz, ny, nx = (int(v) for v in shape)
    n = math.prod((nz, ny, nx))
    real = np.dtype(dtype).itemsize
    complex_bytes = 8 if real == 4 else 16
    nh = nz * ny * (nx // 2 + 1)
    rho = n * real
    spectrum = nh * complex_bytes
    if task == "potential":
        # density + rho_hat + phi_hat + phi + FFT workspace margin
        return int(1.35 * (2 * rho + 2 * spectrum))
    if task == "tidal":
        # density + spectra + full symmetric matrix + eigensystem + workspace
        tensor_matrix = n * real * 9
        evals = n * real * 3
        evecs = n * real * 9
        return int(1.25 * (rho + 2 * spectrum + tensor_matrix + evals + evecs))
    raise ValueError("task must be 'potential' or 'tidal'")


def make_execution_plan(
    shape,
    *,
    task: str,
    backend: str = "cpu",
    devices=None,
    dtype=None,
    memory: str = "auto",
    memory_fraction: float = 0.70,
) -> ExecutionPlan:
    backend = normalize_backend(backend)
    devs = normalize_devices(backend, devices)
    dt = default_dtype(backend, dtype)
    estimate = estimate_peak_bytes(shape, dt, task)

    mode = str(memory).strip().lower().replace("-", "_")
    aliases = {"auto": "auto", "incore": "in_core", "in_core": "in_core",
               "outofcore": "out_of_core", "out_of_core": "out_of_core"}
    if mode not in aliases:
        raise ValueError("memory must be 'auto', 'in_core', or 'out_of_core'")
    mode = aliases[mode]

    if backend == "gpu" and len(devs) > 1:
        if mode == "in_core":
            raise ValueError(
                "multi-GPU execution uses the split out-of-core engine; "
                "use memory='auto' or 'out_of_core'"
            )
        return ExecutionPlan(
            backend, "out_of_core", devs, np.dtype("float32"), estimate, None,
            "multiple GPUs use the exact split-dimensional FFT engine",
        )

    if mode != "auto":
        available = None
        if backend == "cpu":
            available = available_ram_bytes()
        elif devs:
            info = visible_gpu_memory_bytes(devs[0])
            available = info[0] if info else None
        if mode == "out_of_core":
            dt = np.dtype("float32")  # production split FFT storage convention
        return ExecutionPlan(backend, mode, devs, dt, estimate, available, "explicit request")

    if backend == "cpu":
        available = available_ram_bytes()
    else:
        info = visible_gpu_memory_bytes(devs[0]) if devs else None
        available = info[0] if info else None

    if available is None:
        # Unknown capacity: favor predictable in-core behavior for small API calls.
        return ExecutionPlan(backend, "in_core", devs, dt, estimate, None,
                             "available memory could not be queried")

    if estimate <= int(memory_fraction * available):
        return ExecutionPlan(backend, "in_core", devs, dt, estimate, available,
                             "estimated working set fits the configured memory fraction")

    return ExecutionPlan(backend, "out_of_core", devs, np.dtype("float32"), estimate,
                         available, "estimated in-core working set exceeds available memory")
