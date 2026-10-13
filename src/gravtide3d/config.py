from __future__ import annotations

from dataclasses import dataclass

DEFAULT_BACKEND = "cpu"
DEFAULT_MEMORY = "auto"
DEFAULT_CORRECTION = "dc"
DEFAULT_AXIS_ORDER = ("z", "y", "x")
DEFAULT_PADDING = 1
DEFAULT_CPU_DTYPE = "float64"
DEFAULT_GPU_DTYPE = "float32"


@dataclass(slots=True)
class ExecutionConfig:
    """Advanced execution controls.

    Most users should not need to construct this object.  The public API keeps
    CPU/GPU and in-core/out-of-core choices separate; this class contains the
    lower-level tuning and resumability settings.
    """

    workers: int = 1
    spectral_tile: tuple[int, int] = (32, 256)
    eig_tile: tuple[int, int] = (512, 512)
    resume: bool = True
    overwrite: bool = False
    cleanup_scratch: bool = False
    validate_input: bool = True
    memory_fraction: float = 0.70

    def __post_init__(self) -> None:
        if self.workers < 1:
            raise ValueError("workers must be >= 1")
        if len(self.spectral_tile) != 2 or min(self.spectral_tile) <= 0:
            raise ValueError("spectral_tile must contain two positive integers")
        if len(self.eig_tile) != 2 or min(self.eig_tile) <= 0:
            raise ValueError("eig_tile must contain two positive integers")
        if not (0.05 <= self.memory_fraction <= 0.95):
            raise ValueError("memory_fraction must lie between 0.05 and 0.95")
