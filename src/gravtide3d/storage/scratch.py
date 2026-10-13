from __future__ import annotations

import shutil
import tempfile
from pathlib import Path


class ScratchManager:
    def __init__(self, path=None, *, cleanup=False):
        self.cleanup = bool(cleanup)
        self._owned = path is None
        self.path = Path(path) if path is not None else Path(tempfile.mkdtemp(prefix="gravtide3d-"))
        self.path.mkdir(parents=True, exist_ok=True)

    def free_bytes(self) -> int:
        return int(shutil.disk_usage(self.path).free)

    def remove(self) -> None:
        """Remove only a temporary directory created by this manager."""
        if not self._owned:
            raise ValueError("Refusing to delete a user-provided scratch directory")
        if self.path.exists():
            shutil.rmtree(self.path)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.cleanup and self._owned:
            self.remove()
