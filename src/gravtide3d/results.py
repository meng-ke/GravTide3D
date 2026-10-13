from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator


@dataclass(slots=True)
class PotentialResult:
    """Result of :func:`gravitational_potential`."""

    phi: Any
    shape: tuple[int, int, int]
    voxel_size: tuple[float, float, float]
    density_mean: float
    backend: str
    memory_mode: str
    phi_hat: Any | None = None
    output_path: Path | None = None
    spectral_path: Path | None = None
    density_source: Path | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def axis_order(self) -> tuple[str, str, str]:
        return ("z", "y", "x")


@dataclass(slots=True)
class TidalResult:
    """DC-corrected tidal eigensystem and associated metadata."""

    eigenvalues: Any
    eigenvectors: Any
    shape: tuple[int, int, int]
    voxel_size: tuple[float, float, float]
    density_mean: float
    correction: str
    delta_dc: float | None
    backend: str
    memory_mode: str
    tensor: Any | None = None
    output_path: Path | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def axis_order(self) -> tuple[str, str, str]:
        return ("z", "y", "x")

    @property
    def eigenvalue_order(self) -> tuple[str, str, str]:
        return ("lambda_min", "lambda_mid", "lambda_max")

    @property
    def vector_component_order(self) -> tuple[str, str, str]:
        return ("z", "y", "x")

    def __iter__(self) -> Iterator[Any]:
        yield self.eigenvalues
        yield self.eigenvectors
