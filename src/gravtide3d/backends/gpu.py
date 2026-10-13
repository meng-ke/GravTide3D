from __future__ import annotations


class GPUBackend:
    name = "gpu"

    def __init__(self, device: int = 0):
        try:
            import cupy as cp
        except ImportError as exc:
            raise ImportError(
                "GPU backend requires CuPy. Install GravTide3D with the GPU extra "
                "or install a CuPy build matching your CUDA runtime."
            ) from exc
        self.cp = cp
        self.device = int(device)
        cp.cuda.Device(self.device).use()
        self.xp = cp

    def asarray(self, x, dtype=None):
        return self.cp.asarray(x, dtype=dtype)

    def to_numpy(self, x):
        return self.cp.asnumpy(x)

    def synchronize(self) -> None:
        self.cp.cuda.Stream.null.synchronize()

    def release(self) -> None:
        self.synchronize()
        self.cp.get_default_memory_pool().free_all_blocks()
        self.cp.get_default_pinned_memory_pool().free_all_blocks()
