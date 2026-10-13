#!/usr/bin/env python3
"""
Eight-GPU out-of-core gravitational potential and tidal eigensystem pipeline.

The three-dimensional transform is evaluated exactly as a separable FFT:

    density(z,y,x)
      -> rFFT2 over (y,x), independently for each z plane
      -> FFT over z, independently for (ky,kx) tiles
      -> global half-spectrum phi_hat(kz,ky,kx)

The inverse follows the reverse order.  Independent GPUs process disjoint
z-slabs or ky-slabs; spatial regions are never solved as independent gravity
problems.

Array order: (z, y, x)
Units:
    density       g cm^-3
    voxel size    cm, ordered (dz,dy,dx)
    phi           cm^2 s^-2
    tidal/eigen   s^-2

Tensor convention:
    T_ij = + d_i d_j Phi
Positive eigenvalues are compressive.

This implementation intentionally does not use MPI or CuPy's experimental
multi-GPU FFT.  It uses one spawned Python process per GPU and disk/RAM-backed
shards.  The large cube never has to fit on one GPU.

For the large production run, put --scratchdir on /dev/shm.  The global
half-spectrum and all six tensor-component shards are kept in RAM-backed
scratch so the SATA/block device is used mainly for the input and final
requested .npy outputs.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import multiprocessing as mp
import os
import shutil
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

G_CGS = 6.67430e-8
COMPONENTS = ("00", "11", "22", "01", "02", "12")
COMPONENT_AXES = {
    "00": (0, 0),
    "11": (1, 1),
    "22": (2, 2),
    "01": (0, 1),
    "02": (0, 2),
    "12": (1, 2),
}


# ---------------------------------------------------------------------------
# General helpers
# ---------------------------------------------------------------------------

def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(path)


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def human_bytes(nbytes: int | float) -> str:
    value = float(nbytes)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB", "PiB"):
        if value < 1024.0 or unit == "PiB":
            return f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{value:.2f} PiB"


def split_range(n: int, parts: int, rank: int) -> tuple[int, int]:
    q, r = divmod(int(n), int(parts))
    start = rank * q + min(rank, r)
    stop = start + q + (1 if rank < r else 0)
    return start, stop


def iter_tiles(ny: int, nx: int, tile: tuple[int, int]):
    ty, tx = (int(tile[0]), int(tile[1]))
    if ty <= 0 or tx <= 0:
        raise ValueError("Tile entries must be positive.")
    for y0 in range(0, ny, ty):
        y1 = min(y0 + ty, ny)
        for x0 in range(0, nx, tx):
            x1 = min(x0 + tx, nx)
            yield y0, y1, x0, x1


def parse_voxel_size(values: Sequence[float]) -> tuple[float, float, float]:
    if len(values) != 3:
        raise ValueError("voxel size must contain (dz,dy,dx).")
    result = tuple(float(v) for v in values)
    if min(result) <= 0:
        raise ValueError("Voxel sizes must be positive.")
    return result  # type: ignore[return-value]


def parse_devices(value: str) -> list[int]:
    devices = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not devices:
        raise ValueError("--devices must contain at least one device ID.")
    if len(set(devices)) != len(devices):
        raise ValueError("--devices contains duplicate IDs.")
    return devices


def get_backend(backend: str, device: int):
    if backend == "cpu":
        return np, None
    if backend != "gpu":
        raise ValueError("backend must be gpu or cpu")
    import cupy as cp
    cp.cuda.Device(int(device)).use()
    return cp, cp


def to_numpy(array, cp_module):
    if cp_module is None:
        return np.asarray(array)
    return cp_module.asnumpy(array)


def release_backend(cp_module) -> None:
    if cp_module is None:
        return
    cp_module.cuda.Stream.null.synchronize()
    cp_module.get_default_memory_pool().free_all_blocks()
    cp_module.get_default_pinned_memory_pool().free_all_blocks()


def fft_rfft2(xp, array):
    result = xp.fft.rfft2(array, axes=(-2, -1))
    return result.astype(xp.complex64, copy=False)


def fft_fft_z(xp, array):
    result = xp.fft.fft(array, axis=0)
    return result.astype(xp.complex64, copy=False)


def fft_ifft_z(xp, array):
    result = xp.fft.ifft(array, axis=0)
    return result.astype(xp.complex64, copy=False)


def fft_irfft2(xp, array, shape):
    result = xp.fft.irfft2(array, s=shape, axes=(-2, -1))
    return result.real.astype(xp.float32, copy=False)


def validate_npy(path: Path, shape: tuple[int, ...], dtype) -> bool:
    if not path.exists():
        return False
    try:
        arr = np.load(path, mmap_mode="r")
        okay = arr.shape == shape and arr.dtype == np.dtype(dtype)
        del arr
        return bool(okay)
    except Exception:
        return False


def create_or_open(
    path: Path,
    shape: tuple[int, ...],
    dtype,
    *,
    resume: bool,
    overwrite: bool,
):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if overwrite:
            path.unlink()
        elif resume:
            arr = np.load(path, mmap_mode="r+")
            if arr.shape != shape or arr.dtype != np.dtype(dtype):
                raise ValueError(
                    f"{path} has shape={arr.shape}, dtype={arr.dtype}; "
                    f"expected shape={shape}, dtype={np.dtype(dtype)}"
                )
            return arr
        else:
            raise FileExistsError(f"{path} already exists.")
    return np.lib.format.open_memmap(
        path, mode="w+", dtype=dtype, shape=shape
    )


def run_jobs(worker, jobs: list[dict[str, Any]], workers: int) -> list[dict[str, Any]]:
    """Run CUDA workers with spawn; propagate the first worker exception."""
    if workers == 1:
        return [worker(jobs[0])]
    ctx = mp.get_context("spawn")
    results: list[dict[str, Any]] = []
    with ProcessPoolExecutor(
        max_workers=workers,
        mp_context=ctx,
    ) as pool:
        futures = {pool.submit(worker, job): job for job in jobs}
        for future in as_completed(futures):
            results.append(future.result())
    return results


def progress_line(
    label: str,
    rank: int,
    done: int,
    total: int,
    started: float,
) -> str:
    elapsed = max(time.perf_counter() - started, 1e-9)
    rate = done / elapsed
    remaining = (total - done) / rate if rate > 0 else float("nan")
    return (
        f"[{label} rank={rank}] {done}/{total}; "
        f"elapsed={elapsed/60:.1f} min; ETA={remaining/60:.1f} min"
    )


@dataclass(frozen=True)
class PipelineState:
    density_path: str
    workdir: str
    scratchdir: str
    shape: tuple[int, int, int]
    spectral_shape: tuple[int, int, int]
    voxel_size_cm: tuple[float, float, float]
    devices: tuple[int, ...]
    backend: str
    density_mean_g_cm3: float
    dtype: str = "float32"
    complex_dtype: str = "complex64"
    G_cgs: float = G_CGS
    axis_order: tuple[str, str, str] = ("z", "y", "x")
    boundary: str = "native periodic FFT grid; no zero-padding"

    @property
    def path(self) -> Path:
        return Path(self.workdir) / "pipeline_state.json"

    def save(self) -> None:
        atomic_json(self.path, asdict(self))

    @classmethod
    def load(cls, workdir: str | os.PathLike[str]) -> "PipelineState":
        payload = load_json(Path(workdir) / "pipeline_state.json")
        for key in (
            "shape", "spectral_shape", "voxel_size_cm", "devices", "axis_order"
        ):
            payload[key] = tuple(payload[key])
        return cls(**payload)


# ---------------------------------------------------------------------------
# Paths and signatures
# ---------------------------------------------------------------------------

def z_shard_path(base: Path, rank: int) -> Path:
    return base / f"z_rank_{rank:03d}.npy"


def y_shard_path(base: Path, rank: int) -> Path:
    return base / f"y_rank_{rank:03d}.npy"


def progress_path(base: Path, rank: int) -> Path:
    return base / f"progress_rank_{rank:03d}.json"


def signature(
    density_path: Path,
    shape: tuple[int, int, int],
    voxel_size: tuple[float, float, float],
    devices: list[int],
    backend: str,
    scratchdir: Path | None = None,
) -> dict[str, Any]:
    # Cheap file identity checks for very large .npy files. Updating an input
    # normally changes size, mtime or ctime; do not rely on resume if an input
    # can be modified while a calculation is running.
    stat = density_path.stat()
    return {
        "density_path": str(density_path.resolve()),
        "density_file_stat": {
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "ctime_ns": stat.st_ctime_ns,
        },
        "scratchdir": str(scratchdir.resolve()) if scratchdir is not None else None,
        "shape": list(shape),
        "voxel_size_cm": list(voxel_size),
        "devices": list(devices),
        "backend": backend,
        "dtype": "float32",
        "complex_dtype": "complex64",
        "padding_factor": [1, 1, 1],
        "G_cgs": G_CGS,
    }


def ensure_signature(
    workdir: Path,
    expected: dict[str, Any],
    *,
    overwrite: bool,
) -> None:
    path = workdir / "run_signature.json"
    if path.exists():
        actual = load_json(path)
        if actual != expected:
            # Never recursively delete a caller-provided workdir, even with
            # --overwrite; there may be unrelated results inside it.  Start a
            # different calculation in a new output/scratch workspace.
            raise ValueError(
                "Existing workdir belongs to a different calculation or its "
                "density file changed. Choose a new workdir (and scratch) "
                "instead of overwriting unrelated results."
            )
    if not path.exists():
        # An unfinished/legacy run without a signature cannot be verified.
        # Never stamp a new signature onto unverified existing results.
        if any((workdir / name).exists() for name in (
            "pipeline_state.json", "potential_complete.json", "phi3d.npy",
            "eigenvalues.npy", "eigenvectors.npy", "tidal_metadata.json",
        )):
            raise ValueError(
                "Existing calculation files have no run signature. "
                "Choose a new workdir rather than resuming unverified results."
            )
        workdir.mkdir(parents=True, exist_ok=True)
        atomic_json(path, expected)


# ---------------------------------------------------------------------------
# Stage 1A: per-plane 2-D forward transforms
# ---------------------------------------------------------------------------

def worker_xy_forward(job: dict[str, Any]) -> dict[str, Any]:
    rank = int(job["rank"])
    device = int(job["device"])
    backend = str(job["backend"])
    xp, cp = get_backend(backend, device)

    density = np.load(job["density_path"], mmap_mode="r")
    nz, ny, nx = tuple(job["shape"])
    nxh = nx // 2 + 1
    z0, z1 = tuple(job["z_range"])
    local_nz = z1 - z0
    output = Path(job["output"])
    status_file = Path(job["status"])
    resume = bool(job["resume"])
    overwrite = bool(job["overwrite"])

    shard = create_or_open(
        output,
        (local_nz, ny, nxh),
        np.complex64,
        resume=resume,
        overwrite=overwrite,
    )

    status = load_json(status_file) if resume else {}
    done = int(status.get("done", 0))
    density_sum = float(status.get("density_sum", 0.0))
    if done < 0 or done > local_nz:
        raise ValueError(f"Invalid progress in {status_file}")

    started = time.perf_counter()
    for local_z in range(done, local_nz):
        global_z = z0 + local_z
        plane = np.asarray(density[global_z], dtype=np.float32)
        if not np.isfinite(plane).all():
            raise ValueError(f"Non-finite density in z={global_z}")
        if np.any(plane < 0):
            raise ValueError(f"Negative density in z={global_z}")

        density_sum += float(np.sum(plane, dtype=np.float64))
        plane_gpu = xp.asarray(plane, dtype=xp.float32)
        transformed = fft_rfft2(xp, plane_gpu)
        shard[local_z] = to_numpy(transformed, cp)

        done = local_z + 1
        shard.flush()
        atomic_json(
            status_file,
            {
                "rank": rank,
                "device": device,
                "z0": z0,
                "z1": z1,
                "done": done,
                "density_sum": density_sum,
                "complete": done == local_nz,
            },
        )
        if done == 1 or done % 4 == 0 or done == local_nz:
            print(progress_line("xy-rfft2", rank, done, local_nz, started), flush=True)

        del plane, plane_gpu, transformed
        if cp is not None and done % 8 == 0:
            release_backend(cp)

    shard.flush()
    del shard, density
    release_backend(cp)
    return {
        "rank": rank,
        "density_sum": density_sum,
        "z0": z0,
        "z1": z1,
        "complete": True,
    }


# ---------------------------------------------------------------------------
# Stage 1B: z FFT and Poisson kernel, y-sharded output
# ---------------------------------------------------------------------------

def worker_z_poisson(job: dict[str, Any]) -> dict[str, Any]:
    rank = int(job["rank"])
    device = int(job["device"])
    backend = str(job["backend"])
    xp, cp = get_backend(backend, device)

    nz, ny, nx = tuple(job["shape"])
    nxh = nx // 2 + 1
    y0_global, y1_global = tuple(job["y_range"])
    local_ny = y1_global - y0_global
    ty, tx = tuple(job["spectral_tile"])
    dz, dy, dx = tuple(job["voxel_size"])
    z_sources = [
        (int(item["z0"]), int(item["z1"]), np.load(item["path"], mmap_mode="r"))
        for item in job["z_sources"]
    ]

    output = Path(job["output"])
    status_file = Path(job["status"])
    resume = bool(job["resume"])
    overwrite = bool(job["overwrite"])
    shard = create_or_open(
        output,
        (nz, local_ny, nxh),
        np.complex64,
        resume=resume,
        overwrite=overwrite,
    )

    tiles = list(iter_tiles(local_ny, nxh, (ty, tx)))
    status = load_json(status_file) if resume else {}
    done = int(status.get("done", 0))
    if done < 0 or done > len(tiles):
        raise ValueError(f"Invalid progress in {status_file}")

    kz_np = 2.0 * np.pi * np.fft.fftfreq(nz, d=dz)
    ky_np = 2.0 * np.pi * np.fft.fftfreq(ny, d=dy)
    kx_np = 2.0 * np.pi * np.fft.rfftfreq(nx, d=dx)
    kz = xp.asarray(kz_np, dtype=xp.float64).reshape(nz, 1, 1)

    started = time.perf_counter()
    for index in range(done, len(tiles)):
        ly0, ly1, x0, x1 = tiles[index]
        gy0 = y0_global + ly0
        gy1 = y0_global + ly1
        tile_cpu = np.empty(
            (nz, ly1 - ly0, x1 - x0),
            dtype=np.complex64,
        )
        for zs0, zs1, source in z_sources:
            tile_cpu[zs0:zs1] = source[:, gy0:gy1, x0:x1]

        tile_gpu = xp.asarray(tile_cpu, dtype=xp.complex64)
        rho_hat = fft_fft_z(xp, tile_gpu)

        ky = xp.asarray(ky_np[gy0:gy1], dtype=xp.float64).reshape(
            1, gy1 - gy0, 1
        )
        kx = xp.asarray(kx_np[x0:x1], dtype=xp.float64).reshape(
            1, 1, x1 - x0
        )
        k2 = kz * kz + ky * ky + kx * kx
        if gy0 == 0 and x0 == 0:
            k2[0, 0, 0] = xp.inf
        factor = (-4.0 * math.pi * G_CGS / k2).astype(xp.float32)

        if gy0 == 0 and x0 == 0:
            factor[0, 0, 0] = 0.0

        phi = rho_hat * factor
        shard[:, ly0:ly1, x0:x1] = to_numpy(phi, cp).astype(
            np.complex64, copy=False
        )

        done = index + 1
        if done % 4 == 0 or done == len(tiles):
            shard.flush()
            atomic_json(
                status_file,
                {
                    "rank": rank,
                    "device": device,
                    "y0": y0_global,
                    "y1": y1_global,
                    "done": done,
                    "total": len(tiles),
                    "complete": done == len(tiles),
                },
            )
            print(
                progress_line("z-fft-poisson", rank, done, len(tiles), started),
                flush=True,
            )

        del tile_cpu, tile_gpu, rho_hat, ky, kx, k2, factor, phi
        if cp is not None and done % 16 == 0:
            release_backend(cp)

    shard.flush()
    del shard
    for _, _, source in z_sources:
        del source
    release_backend(cp)
    return {"rank": rank, "complete": True}


# ---------------------------------------------------------------------------
# Inverse field: inverse z on y-shards, then inverse 2-D on z-shards
# ---------------------------------------------------------------------------

def component_multiplier(
    xp,
    component: str | None,
    kz_np: np.ndarray,
    ky_np: np.ndarray,
    kx_np: np.ndarray,
    nz: int,
    ny: int,
    nx: int,
    *,
    ky_offset: int,
    kx_offset: int,
):
    """Return -k_a*k_b for one local spectral tile.

    Cross derivatives use zero first-derivative multipliers at self-conjugate
    Nyquist modes.  The y and x vectors are local tile slices, so global
    Nyquist indices must first be converted to local indices.
    """
    if component is None:
        return None
    a, b = COMPONENT_AXES[component]

    vectors = [
        np.array(kz_np, copy=True),
        np.array(ky_np, copy=True),
        np.array(kx_np, copy=True),
    ]
    if a != b:
        for axis in (a, b):
            n = (nz, ny, nx)[axis]
            if n % 2 != 0:
                continue
            global_nyquist = n // 2
            if axis == 0:
                vectors[0][global_nyquist] = 0.0
            elif axis == 1:
                local_index = global_nyquist - int(ky_offset)
                if 0 <= local_index < len(vectors[1]):
                    vectors[1][local_index] = 0.0
            elif axis == 2:
                local_index = global_nyquist - int(kx_offset)
                if 0 <= local_index < len(vectors[2]):
                    vectors[2][local_index] = 0.0

    ka = xp.asarray(vectors[a], dtype=xp.float64)
    kb = xp.asarray(vectors[b], dtype=xp.float64)

    shapes = [(nz, 1, 1), (1, len(ky_np), 1), (1, 1, len(kx_np))]
    ka = ka.reshape(shapes[a])
    kb = kb.reshape(shapes[b])
    return (-(ka * kb)).astype(xp.float32)


def worker_inverse_z(job: dict[str, Any]) -> dict[str, Any]:
    rank = int(job["rank"])
    device = int(job["device"])
    backend = str(job["backend"])
    xp, cp = get_backend(backend, device)

    nz, ny, nx = tuple(job["shape"])
    nxh = nx // 2 + 1
    y0_global, y1_global = tuple(job["y_range"])
    local_ny = y1_global - y0_global
    ty, tx = tuple(job["spectral_tile"])
    dz, dy, dx = tuple(job["voxel_size"])
    component = job["component"]

    phi = np.load(job["phi_path"], mmap_mode="r")
    output = Path(job["output"])
    status_file = Path(job["status"])
    resume = bool(job["resume"])
    overwrite = bool(job["overwrite"])
    out = create_or_open(
        output,
        (nz, local_ny, nxh),
        np.complex64,
        resume=resume,
        overwrite=overwrite,
    )

    tiles = list(iter_tiles(local_ny, nxh, (ty, tx)))
    status = load_json(status_file) if resume else {}
    done = int(status.get("done", 0))

    kz_np = 2.0 * np.pi * np.fft.fftfreq(nz, d=dz)
    ky_all = 2.0 * np.pi * np.fft.fftfreq(ny, d=dy)
    kx_all = 2.0 * np.pi * np.fft.rfftfreq(nx, d=dx)

    started = time.perf_counter()
    for index in range(done, len(tiles)):
        ly0, ly1, x0, x1 = tiles[index]
        gy0 = y0_global + ly0
        gy1 = y0_global + ly1
        tile_gpu = xp.asarray(
            np.ascontiguousarray(phi[:, ly0:ly1, x0:x1]),
            dtype=xp.complex64,
        )
        multiplier = component_multiplier(
            xp,
            component,
            kz_np,
            ky_all[gy0:gy1],
            kx_all[x0:x1],
            nz,
            ny,
            nx,
            ky_offset=gy0,
            kx_offset=x0,
        )
        if multiplier is not None:
            tile_gpu = tile_gpu * multiplier
        transformed = fft_ifft_z(xp, tile_gpu)
        out[:, ly0:ly1, x0:x1] = to_numpy(transformed, cp)

        done = index + 1
        if done % 4 == 0 or done == len(tiles):
            out.flush()
            atomic_json(
                status_file,
                {
                    "rank": rank,
                    "device": device,
                    "done": done,
                    "total": len(tiles),
                    "complete": done == len(tiles),
                    "component": component,
                },
            )
            label = "phi" if component is None else f"T{component}"
            print(
                progress_line(f"inverse-z-{label}", rank, done, len(tiles), started),
                flush=True,
            )

        del tile_gpu, multiplier, transformed
        if cp is not None and done % 16 == 0:
            release_backend(cp)

    out.flush()
    del phi, out
    release_backend(cp)
    return {"rank": rank, "complete": True}


def worker_inverse_xy(job: dict[str, Any]) -> dict[str, Any]:
    rank = int(job["rank"])
    device = int(job["device"])
    backend = str(job["backend"])
    xp, cp = get_backend(backend, device)

    nz, ny, nx = tuple(job["shape"])
    nxh = nx // 2 + 1
    z0, z1 = tuple(job["z_range"])
    local_nz = z1 - z0
    y_sources = [
        (int(item["y0"]), int(item["y1"]), np.load(item["path"], mmap_mode="r"))
        for item in job["y_sources"]
    ]

    output = Path(job["output"])
    status_file = Path(job["status"])
    resume = bool(job["resume"])
    overwrite = bool(job["overwrite"])
    out = create_or_open(
        output,
        (local_nz, ny, nx),
        np.float32,
        resume=resume,
        overwrite=overwrite,
    )
    status = load_json(status_file) if resume else {}
    done = int(status.get("done", 0))

    plane_spec = np.empty((ny, nxh), dtype=np.complex64)
    started = time.perf_counter()
    for local_z in range(done, local_nz):
        global_z = z0 + local_z
        for ys0, ys1, source in y_sources:
            plane_spec[ys0:ys1] = source[global_z]

        plane_gpu = xp.asarray(plane_spec, dtype=xp.complex64)
        real_gpu = fft_irfft2(xp, plane_gpu, (ny, nx))
        out[local_z] = to_numpy(real_gpu, cp)

        completed = local_z + 1
        out.flush()
        atomic_json(
            status_file,
            {
                "rank": rank,
                "device": device,
                "z0": z0,
                "z1": z1,
                "done": completed,
                "complete": completed == local_nz,
            },
        )
        if completed == 1 or completed % 4 == 0 or completed == local_nz:
            print(
                progress_line("inverse-rfft2", rank, completed, local_nz, started),
                flush=True,
            )

        del plane_gpu, real_gpu
        if cp is not None and completed % 8 == 0:
            release_backend(cp)

    out.flush()
    del out, plane_spec
    for _, _, source in y_sources:
        del source
    release_backend(cp)
    return {"rank": rank, "complete": True}


def merge_z_shards(
    shard_dir: Path,
    output: Path,
    shape: tuple[int, ...],
    nshards: int,
    *,
    overwrite: bool,
    suffix_shape: tuple[int, ...] = (),
) -> Path:
    expected = shape + suffix_shape
    dtype = np.float32
    if validate_npy(output, expected, dtype) and not overwrite:
        return output
    if output.exists():
        output.unlink()
    final = np.lib.format.open_memmap(
        output, mode="w+", dtype=dtype, shape=expected
    )
    nz = shape[0]
    started = time.perf_counter()
    merged = 0
    for rank in range(nshards):
        z0, z1 = split_range(nz, nshards, rank)
        source = np.load(z_shard_path(shard_dir, rank), mmap_mode="r")
        for local_z in range(z1 - z0):
            final[z0 + local_z] = source[local_z]
            merged += 1
            if merged % 4 == 0 or merged == nz:
                final.flush()
                print(
                    progress_line("merge-z", 0, merged, nz, started),
                    flush=True,
                )
        del source
    final.flush()
    del final
    return output


# ---------------------------------------------------------------------------
# Pipeline orchestration
# ---------------------------------------------------------------------------

def build_jobs_common(
    devices: list[int],
    backend: str,
) -> list[tuple[int, int]]:
    if backend == "cpu":
        return [(rank, rank) for rank in range(len(devices))]
    return list(enumerate(devices))


def run_xy_forward(
    density_path: Path,
    shape: tuple[int, int, int],
    scratch: Path,
    devices: list[int],
    backend: str,
    resume: bool,
    overwrite: bool,
) -> tuple[Path, float]:
    base = scratch / "rho_xy_zshards"
    base.mkdir(parents=True, exist_ok=True)
    jobs = []
    for rank, device in build_jobs_common(devices, backend):
        z0, z1 = split_range(shape[0], len(devices), rank)
        jobs.append(
            {
                "rank": rank,
                "device": device,
                "backend": backend,
                "density_path": str(density_path),
                "shape": shape,
                "z_range": (z0, z1),
                "output": str(z_shard_path(base, rank)),
                "status": str(progress_path(base, rank)),
                "resume": resume,
                "overwrite": overwrite,
            }
        )
    results = run_jobs(worker_xy_forward, jobs, len(devices))
    density_sum = sum(float(item["density_sum"]) for item in results)
    return base, density_sum


def run_z_poisson(
    rho_xy_base: Path,
    shape: tuple[int, int, int],
    voxel_size: tuple[float, float, float],
    scratch: Path,
    devices: list[int],
    backend: str,
    spectral_tile: tuple[int, int],
    resume: bool,
    overwrite: bool,
) -> Path:
    base = scratch / "phi_hat_yshards"
    base.mkdir(parents=True, exist_ok=True)
    z_sources = []
    for rank in range(len(devices)):
        z0, z1 = split_range(shape[0], len(devices), rank)
        z_sources.append(
            {
                "z0": z0,
                "z1": z1,
                "path": str(z_shard_path(rho_xy_base, rank)),
            }
        )

    jobs = []
    for rank, device in build_jobs_common(devices, backend):
        y0, y1 = split_range(shape[1], len(devices), rank)
        jobs.append(
            {
                "rank": rank,
                "device": device,
                "backend": backend,
                "shape": shape,
                "voxel_size": voxel_size,
                "y_range": (y0, y1),
                "spectral_tile": spectral_tile,
                "z_sources": z_sources,
                "output": str(y_shard_path(base, rank)),
                "status": str(progress_path(base, rank)),
                "resume": resume,
                "overwrite": overwrite,
            }
        )
    run_jobs(worker_z_poisson, jobs, len(devices))
    return base


def run_inverse_field(
    phi_base: Path,
    shape: tuple[int, int, int],
    voxel_size: tuple[float, float, float],
    scratch: Path,
    output_base: Path,
    devices: list[int],
    backend: str,
    spectral_tile: tuple[int, int],
    component: str | None,
    resume: bool,
    overwrite: bool,
) -> Path:
    label = "phi" if component is None else f"T{component}"
    inverse_base = scratch / f"inverse_z_{label}_yshards"
    if overwrite and inverse_base.exists():
        shutil.rmtree(inverse_base)
    inverse_base.mkdir(parents=True, exist_ok=True)

    inverse_z_jobs = []
    for rank, device in build_jobs_common(devices, backend):
        y0, y1 = split_range(shape[1], len(devices), rank)
        inverse_z_jobs.append(
            {
                "rank": rank,
                "device": device,
                "backend": backend,
                "shape": shape,
                "voxel_size": voxel_size,
                "y_range": (y0, y1),
                "spectral_tile": spectral_tile,
                "component": component,
                "phi_path": str(y_shard_path(phi_base, rank)),
                "output": str(y_shard_path(inverse_base, rank)),
                "status": str(progress_path(inverse_base, rank)),
                "resume": resume,
                "overwrite": overwrite,
            }
        )
    run_jobs(worker_inverse_z, inverse_z_jobs, len(devices))

    if overwrite and output_base.exists():
        shutil.rmtree(output_base)
    output_base.mkdir(parents=True, exist_ok=True)
    y_sources = []
    for rank in range(len(devices)):
        y0, y1 = split_range(shape[1], len(devices), rank)
        y_sources.append(
            {
                "y0": y0,
                "y1": y1,
                "path": str(y_shard_path(inverse_base, rank)),
            }
        )

    inverse_xy_jobs = []
    for rank, device in build_jobs_common(devices, backend):
        z0, z1 = split_range(shape[0], len(devices), rank)
        inverse_xy_jobs.append(
            {
                "rank": rank,
                "device": device,
                "backend": backend,
                "shape": shape,
                "z_range": (z0, z1),
                "y_sources": y_sources,
                "output": str(z_shard_path(output_base, rank)),
                "status": str(progress_path(output_base, rank)),
                "resume": resume,
                "overwrite": overwrite,
            }
        )
    run_jobs(worker_inverse_xy, inverse_xy_jobs, len(devices))

    # The inverse-z workspace is reproducible from phi_hat and is deleted only
    # after all real-space shards are complete.
    shutil.rmtree(inverse_base)
    return output_base


# ---------------------------------------------------------------------------
# Eigensystem
# ---------------------------------------------------------------------------

def worker_eigensystem(job: dict[str, Any]) -> dict[str, Any]:
    rank = int(job["rank"])
    device = int(job["device"])
    backend = str(job["backend"])
    xp, cp = get_backend(backend, device)

    shape = tuple(job["shape"])
    nz, ny, nx = shape
    z0, z1 = tuple(job["z_range"])
    local_nz = z1 - z0
    tile_y, tile_x = tuple(job["gpu_tile"])
    trace_mode = str(job["trace_correction"])
    density_mean = float(job["density_mean"])

    density = np.load(job["density_path"], mmap_mode="r")
    tensors = {
        component: np.load(job["tensor_paths"][component], mmap_mode="r")
        for component in COMPONENTS
    }

    eval_path = Path(job["eval_output"])
    evec_path = Path(job["evec_output"])
    status_file = Path(job["status"])
    resume = bool(job["resume"])
    overwrite = bool(job["overwrite"])

    eval_out = create_or_open(
        eval_path,
        (local_nz, ny, nx, 3),
        np.float32,
        resume=resume,
        overwrite=overwrite,
    )
    evec_out = create_or_open(
        evec_path,
        (local_nz, ny, nx, 3, 3),
        np.float32,
        resume=resume,
        overwrite=overwrite,
    )
    status = load_json(status_file) if resume else {}
    done = int(status.get("done", 0))
    max_residual = float(status.get("max_abs_residual", 0.0))
    max_target = float(status.get("max_abs_target", 0.0))
    dc_delta = np.float32(4.0 * math.pi * G_CGS * density_mean / 3.0)

    started = time.perf_counter()
    for local_z in range(done, local_nz):
        global_z = z0 + local_z
        for y0, y1, x0, x1 in iter_tiles(ny, nx, (tile_y, tile_x)):
            sl = (local_z, slice(y0, y1), slice(x0, x1))
            rho_sl = (global_z, slice(y0, y1), slice(x0, x1))
            tile_shape = (y1 - y0, x1 - x0)
            T = np.empty(tile_shape + (3, 3), dtype=np.float32)

            T00 = np.asarray(tensors["00"][sl], dtype=np.float32)
            T11 = np.asarray(tensors["11"][sl], dtype=np.float32)
            T22 = np.asarray(tensors["22"][sl], dtype=np.float32)
            T01 = np.asarray(tensors["01"][sl], dtype=np.float32)
            T02 = np.asarray(tensors["02"][sl], dtype=np.float32)
            T12 = np.asarray(tensors["12"][sl], dtype=np.float32)

            T[..., 0, 0] = T00
            T[..., 1, 1] = T11
            T[..., 2, 2] = T22
            T[..., 0, 1] = T[..., 1, 0] = T01
            T[..., 0, 2] = T[..., 2, 0] = T02
            T[..., 1, 2] = T[..., 2, 1] = T12

            rho_tile = None
            target = None
            if trace_mode == "dc":
                T[..., 0, 0] += dc_delta
                T[..., 1, 1] += dc_delta
                T[..., 2, 2] += dc_delta
                rho_tile = np.asarray(density[rho_sl], dtype=np.float32)
                target = np.float32(4.0 * math.pi * G_CGS) * rho_tile
            elif trace_mode == "local":
                rho_tile = np.asarray(density[rho_sl], dtype=np.float32)
                target = np.float32(4.0 * math.pi * G_CGS) * rho_tile
                trace_raw = T00 + T11 + T22
                delta = (target - trace_raw) / np.float32(3.0)
                T[..., 0, 0] += delta
                T[..., 1, 1] += delta
                T[..., 2, 2] += delta

            T_gpu = xp.asarray(T, dtype=xp.float32)
            eval_gpu, evec_gpu = xp.linalg.eigh(T_gpu)
            eval_tile = to_numpy(eval_gpu, cp).astype(np.float32, copy=False)
            evec_tile = to_numpy(evec_gpu, cp).astype(np.float32, copy=False)
            eval_out[sl] = eval_tile
            evec_out[sl] = evec_tile

            if target is not None:
                residual = np.sum(eval_tile, axis=-1) - target
                max_residual = max(max_residual, float(np.max(np.abs(residual))))
                max_target = max(max_target, float(np.max(np.abs(target))))

            del (
                T, T00, T11, T22, T01, T02, T12,
                rho_tile, target, T_gpu, eval_gpu, evec_gpu,
                eval_tile, evec_tile,
            )

        completed = local_z + 1
        eval_out.flush()
        evec_out.flush()
        atomic_json(
            status_file,
            {
                "rank": rank,
                "device": device,
                "z0": z0,
                "z1": z1,
                "done": completed,
                "complete": completed == local_nz,
                "max_abs_residual": max_residual,
                "max_abs_target": max_target,
            },
        )
        print(
            progress_line("eigh", rank, completed, local_nz, started),
            flush=True,
        )
        release_backend(cp)

    eval_out.flush()
    evec_out.flush()
    del eval_out, evec_out, density
    for array in tensors.values():
        del array
    release_backend(cp)
    return {
        "rank": rank,
        "max_abs_residual": max_residual,
        "max_abs_target": max_target,
        "complete": True,
    }


def run_eigensystem(
    state: PipelineState,
    density_path: Path,
    workdir: Path,
    devices: list[int],
    backend: str,
    gpu_tile: tuple[int, int],
    trace_correction: str,
    resume: bool,
    overwrite: bool,
) -> None:
    shape = state.shape
    tensor_root = Path(state.scratchdir) / "tensor"
    for component in COMPONENTS:
        for rank in range(len(devices)):
            path = z_shard_path(tensor_root / f"T{component}", rank)
            z0, z1 = split_range(shape[0], len(devices), rank)
            if not validate_npy(
                path, (z1 - z0, shape[1], shape[2]), np.float32
            ):
                raise FileNotFoundError(f"Missing tensor shard: {path}")

    eig_root = workdir / "eigensystem_shards"
    if overwrite and eig_root.exists():
        shutil.rmtree(eig_root)
    eig_root.mkdir(parents=True, exist_ok=True)

    jobs = []
    for rank, device in build_jobs_common(devices, backend):
        z0, z1 = split_range(shape[0], len(devices), rank)
        tensor_paths = {
            component: str(z_shard_path(tensor_root / f"T{component}", rank))
            for component in COMPONENTS
        }
        jobs.append(
            {
                "rank": rank,
                "device": device,
                "backend": backend,
                "shape": shape,
                "z_range": (z0, z1),
                "gpu_tile": gpu_tile,
                "trace_correction": trace_correction,
                "density_mean": state.density_mean_g_cm3,
                "density_path": str(density_path),
                "tensor_paths": tensor_paths,
                "eval_output": str(
                    z_shard_path(eig_root / "eigenvalues", rank)
                ),
                "evec_output": str(
                    z_shard_path(eig_root / "eigenvectors", rank)
                ),
                "status": str(progress_path(eig_root / "status", rank)),
                "resume": resume,
                "overwrite": overwrite,
            }
        )
    results = run_jobs(worker_eigensystem, jobs, len(devices))

    merge_z_shards(
        eig_root / "eigenvalues",
        workdir / "eigenvalues.npy",
        shape,
        len(devices),
        overwrite=overwrite,
        suffix_shape=(3,),
    )
    merge_z_shards(
        eig_root / "eigenvectors",
        workdir / "eigenvectors.npy",
        shape,
        len(devices),
        overwrite=overwrite,
        suffix_shape=(3, 3),
    )

    max_residual = max(float(item["max_abs_residual"]) for item in results)
    max_target = max(float(item["max_abs_target"]) for item in results)
    atomic_json(
        workdir / "tidal_metadata.json",
        {
            "shape": list(shape),
            "axis_order": ["z", "y", "x"],
            "voxel_size_cm": list(state.voxel_size_cm),
            "trace_correction": trace_correction,
            "eigenvalue_order": ["lambda_min", "lambda_med", "lambda_max"],
            "eigenvectors_are_columns": True,
            "vector_component_order": ["z", "y", "x"],
            "max_abs_trace_residual_s-2": max_residual,
            "max_relative_trace_residual": (
                max_residual / max_target if max_target > 0 else None
            ),
            "backend": backend,
            "devices": devices,
            "boundary": state.boundary,
            "outputs": {
                "phi3d": str(workdir / "phi3d.npy"),
                "eigenvalues": str(workdir / "eigenvalues.npy"),
                "eigenvectors": str(workdir / "eigenvectors.npy"),
            },
        },
    )


# ---------------------------------------------------------------------------
# Stages
# ---------------------------------------------------------------------------

def stage_preflight(args) -> None:
    density = np.load(args.density, mmap_mode="r")
    if density.ndim != 3:
        raise ValueError("Density must be a 3-D .npy array.")
    shape = tuple(int(n) for n in density.shape)
    nz, ny, nx = shape
    n = math.prod(shape)
    nh = nz * ny * (nx // 2 + 1)
    sizes = {
        "density input": n * density.dtype.itemsize,
        "one complex half-spectrum": nh * 8,
        "phi3d": n * 4,
        "six tensor components": n * 4 * 6,
        "eigenvalues": n * 4 * 3,
        "eigenvectors": n * 4 * 9,
        "final requested outputs": n * 4 * (1 + 3 + 9),
    }
    print("Density:", args.density)
    print("Shape:", shape, "(z,y,x)")
    print("Input dtype:", density.dtype)
    for name, nbytes in sizes.items():
        print(f"{name:30s}: {human_bytes(nbytes)}")
    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(workdir).free
    print(f"Workdir free space:            {human_bytes(free)}")
    scratch = Path(args.scratchdir or args.workdir)
    scratch.mkdir(parents=True, exist_ok=True)
    scratch_free = shutil.disk_usage(scratch).free
    print(f"Scratch free space:            {human_bytes(scratch_free)}")
    field = n * 4
    half = nh * 8
    potential_peak = 3 * half
    tensor_peak = half + 6 * field + half
    print(f"Estimated scratch peak, potential: {human_bytes(potential_peak)}")
    print(f"Estimated scratch peak, tensor:    {human_bytes(tensor_peak)}")
    if scratch_free < tensor_peak * 1.05:
        print(
            "WARNING: scratch may be too small for the all-in-RAM tensor stage; "
            "free at least ~5% above the estimated peak.",
            flush=True,
        )
    if args.backend == "gpu":
        import cupy as cp
        count = cp.cuda.runtime.getDeviceCount()
        print("Visible GPUs:", count)
        print("Requested devices:", parse_devices(args.devices))


def stage_potential(args) -> PipelineState:
    density_path = Path(args.density).expanduser().resolve()
    density = np.load(density_path, mmap_mode="r")
    if density.ndim != 3:
        raise ValueError("Density must be 3-D.")
    shape = tuple(int(n) for n in density.shape)
    voxel_size = parse_voxel_size(args.voxel_size)
    devices = (
        parse_devices(args.devices)
        if args.backend == "gpu"
        else list(range(int(args.workers)))
    )
    workdir = Path(args.workdir).expanduser().resolve()
    scratch = Path(args.scratchdir or args.workdir).expanduser().resolve()

    # Always validate the signature BEFORE accepting a completed stage. The
    # previous fast path could return stale results for a new density/config.
    ensure_signature(
        workdir,
        signature(density_path, shape, voxel_size, devices, args.backend, scratch),
        overwrite=args.overwrite,
    )
    state_path = workdir / "pipeline_state.json"
    complete_path = workdir / "potential_complete.json"
    if args.resume and not args.overwrite and state_path.exists() and complete_path.exists():
        existing_state = PipelineState.load(workdir)
        phi_ok = validate_npy(workdir / "phi3d.npy", shape, np.float32)
        phi_base = Path(existing_state.scratchdir) / "phi_hat_yshards"
        shards_ok = True
        for rank in range(len(devices)):
            y0, y1 = split_range(shape[1], len(devices), rank)
            shards_ok = shards_ok and validate_npy(
                y_shard_path(phi_base, rank),
                (shape[0], y1 - y0, shape[2] // 2 + 1),
                np.complex64,
            )
        if phi_ok and shards_ok:
            print("Potential stage already complete; skipping.", flush=True)
            return existing_state

    scratch.mkdir(parents=True, exist_ok=True)

    rho_xy_base, density_sum = run_xy_forward(
        density_path,
        shape,
        scratch,
        devices,
        args.backend,
        args.resume,
        args.overwrite,
    )
    phi_base = run_z_poisson(
        rho_xy_base,
        shape,
        voxel_size,
        scratch,
        devices,
        args.backend,
        tuple(args.spectral_tile),
        args.resume,
        args.overwrite,
    )

    # rho_xy is no longer needed once the global phi half-spectrum exists.
    # Delete it before inverse transforms to keep /dev/shm peak usage low.
    if not args.keep_rho_xy and rho_xy_base.exists():
        shutil.rmtree(rho_xy_base)

    state = PipelineState(
        density_path=str(density_path),
        workdir=str(workdir),
        scratchdir=str(scratch),
        shape=shape,
        spectral_shape=(shape[0], shape[1], shape[2] // 2 + 1),
        voxel_size_cm=voxel_size,
        devices=tuple(devices),
        backend=args.backend,
        density_mean_g_cm3=density_sum / float(math.prod(shape)),
    )
    state.save()

    phi_shards = run_inverse_field(
        phi_base,
        shape,
        voxel_size,
        scratch,
        scratch / "phi3d_zshards",
        devices,
        args.backend,
        tuple(args.spectral_tile),
        component=None,
        resume=args.resume,
        overwrite=args.overwrite,
    )
    merge_z_shards(
        phi_shards,
        workdir / "phi3d.npy",
        shape,
        len(devices),
        overwrite=args.overwrite,
    )
    # The real-space phi z-shards are only a staging area for the final npy.
    if phi_shards.exists():
        shutil.rmtree(phi_shards)
    atomic_json(
        workdir / "potential_complete.json",
        {
            "complete": True,
            "shape": list(shape),
            "phi3d": str(workdir / "phi3d.npy"),
            "phi_hat_yshards": str(phi_base),
        },
    )
    print("Potential stage complete.", flush=True)
    return state


def stage_tensor(args) -> PipelineState:
    state = PipelineState.load(args.workdir)
    devices = list(state.devices)
    workdir = Path(state.workdir)
    scratch = Path(args.scratchdir or state.scratchdir)
    phi_base = scratch / "phi_hat_yshards"
    if not phi_base.exists():
        raise FileNotFoundError(phi_base)

    for component in COMPONENTS:
        output_base = scratch / "tensor" / f"T{component}"
        marker = scratch / "tensor_status" / f"T{component}_complete.json"
        shards_ok = all(
            validate_npy(
                z_shard_path(output_base, rank),
                (
                    split_range(state.shape[0], len(devices), rank)[1]
                    - split_range(state.shape[0], len(devices), rank)[0],
                    state.shape[1],
                    state.shape[2],
                ),
                np.float32,
            )
            for rank in range(len(devices))
        )
        if args.resume and not args.overwrite and marker.exists() and shards_ok:
            print(f"T{component} already complete in scratch; skipping.", flush=True)
            continue
        run_inverse_field(
            phi_base,
            state.shape,
            state.voxel_size_cm,
            scratch,
            output_base,
            devices,
            state.backend,
            tuple(args.spectral_tile),
            component=component,
            resume=args.resume,
            overwrite=args.overwrite,
        )
        atomic_json(
            marker,
            {
                "complete": True,
                "component": component,
                "shape": list(state.shape),
                "shards": len(devices),
            },
        )
        print(f"T{component} complete.", flush=True)

    atomic_json(
        scratch / "tensor_complete.json",
        {"complete": True, "components": list(COMPONENTS)},
    )
    print("Tensor stage complete.", flush=True)
    return state


def stage_eigensystem(args) -> PipelineState:
    state = PipelineState.load(args.workdir)
    density_path = Path(state.density_path)
    workdir = Path(state.workdir)
    run_eigensystem(
        state,
        density_path,
        workdir,
        list(state.devices),
        state.backend,
        tuple(args.gpu_tile),
        args.trace_correction,
        args.resume,
        args.overwrite,
    )
    print("Eigensystem stage complete.", flush=True)
    return state


def stage_cleanup(args) -> None:
    state = PipelineState.load(args.workdir)
    workdir = Path(state.workdir)
    scratch = Path(state.scratchdir)
    targets = []
    if args.delete_transient:
        targets.extend(
            [
                scratch / "rho_xy_zshards",
                scratch / "phi3d_zshards",
            ]
        )
        targets.extend(scratch.glob("inverse_z_*_yshards"))
    if args.delete_phi_shards:
        targets.extend(
            [
                scratch / "phi3d_zshards",
                scratch / "phi_hat_yshards",
            ]
        )
    if args.delete_tensor:
        targets.append(scratch / "tensor")
        targets.append(scratch / "tensor_status")
        targets.append(scratch / "tensor_complete.json")
    if args.delete_eigen_shards:
        targets.append(workdir / "eigensystem_shards")
    for target in targets:
        target = Path(target)
        if target.exists():
            print("Deleting", target, flush=True)
            if target.is_dir():
                shutil.rmtree(target)
            else:
                target.unlink()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Out-of-core split-dimensional multi-GPU tidal pipeline."
    )
    parser.add_argument(
        "--stage",
        choices=("preflight", "potential", "tensor", "eigensystem", "all", "cleanup"),
        required=True,
    )
    parser.add_argument("--density", help="Density .npy, shape (z,y,x), g cm^-3")
    parser.add_argument(
        "--voxel-size",
        type=float,
        nargs=3,
        metavar=("DZ_CM", "DY_CM", "DX_CM"),
    )
    parser.add_argument("--workdir", required=True)
    parser.add_argument(
        "--scratchdir",
        default=None,
        help="Transient rho_xy and inverse-z shards; fast NVMe/tmpfs recommended.",
    )
    parser.add_argument("--backend", choices=("gpu", "cpu"), default="gpu")
    parser.add_argument("--devices", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--workers", type=int, default=2, help="CPU test workers.")
    parser.add_argument(
        "--spectral-tile", type=int, nargs=2, default=(128, 256),
        metavar=("KY", "KX"),
    )
    parser.add_argument(
        "--gpu-tile", type=int, nargs=2, default=(1024, 1024),
        metavar=("Y", "X"),
    )
    parser.add_argument(
        "--trace-correction",
        choices=("none", "dc", "local"),
        default="local",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--keep-rho-xy", action="store_true")
    parser.add_argument("--delete-transient", action="store_true")
    parser.add_argument("--delete-phi-shards", action="store_true")
    parser.add_argument("--delete-tensor", action="store_true")
    parser.add_argument("--delete-eigen-shards", action="store_true")
    return parser


def main() -> None:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

    args = build_parser().parse_args()
    if args.stage != "cleanup":
        if not args.density:
            raise ValueError("--density is required.")
        if not args.voxel_size:
            raise ValueError("--voxel-size is required.")

    if args.stage == "preflight":
        stage_preflight(args)
    elif args.stage == "potential":
        stage_potential(args)
    elif args.stage == "tensor":
        stage_tensor(args)
    elif args.stage == "eigensystem":
        stage_eigensystem(args)
    elif args.stage == "all":
        stage_potential(args)
        stage_tensor(args)
        stage_eigensystem(args)
    elif args.stage == "cleanup":
        stage_cleanup(args)


if __name__ == "__main__":
    main()
