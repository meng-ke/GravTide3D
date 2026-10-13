from __future__ import annotations

import os


def available_ram_bytes() -> int | None:
    """Best-effort available physical RAM without adding a dependency."""
    try:
        pages = os.sysconf("SC_AVPHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        return int(pages * page_size)
    except (ValueError, OSError, AttributeError):
        return None


def visible_gpu_memory_bytes(device: int = 0) -> tuple[int, int] | None:
    """Return (free,total) GPU bytes if CuPy is available."""
    try:
        import cupy as cp

        with cp.cuda.Device(int(device)):
            free_b, total_b = cp.cuda.runtime.memGetInfo()
        return int(free_b), int(total_b)
    except Exception:
        return None
