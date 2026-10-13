from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class MultiGPUConfig:
    """Description of a multi-GPU execution request.

    Multi-GPU FFT execution is implemented by the out-of-core split-FFT
    executor, not by pretending several GPUs form one CuPy array.
    """

    devices: tuple[int, ...]

    def __post_init__(self):
        if len(self.devices) < 2:
            raise ValueError("MultiGPUConfig requires at least two devices")
        if len(set(self.devices)) != len(self.devices):
            raise ValueError("devices contain duplicates")
