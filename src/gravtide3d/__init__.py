"""Public package interface for GravTide3D."""

from .api import gravitational_potential, tidal_eigensystem
from .config import ExecutionConfig
from .results import PotentialResult, TidalResult

__all__ = [
    "gravitational_potential",
    "tidal_eigensystem",
    "ExecutionConfig",
    "PotentialResult",
    "TidalResult",
]

__version__ = "0.1.0"
