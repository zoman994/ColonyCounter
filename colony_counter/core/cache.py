"""Lazy image cache — stores large images as .npy for speed."""
import os
import tempfile

import numpy as np


class LazyImageCache:
    """Store/load BGR numpy arrays via .npy (fastest possible I/O)."""

    def __init__(self):
        self._tmpdir = tempfile.mkdtemp(prefix="colony_")
        self._n = 0
        self._paths: dict[str, str] = {}

    def store(self, key: str, img: np.ndarray) -> None:
        self._n += 1
        p = os.path.join(self._tmpdir, f"i{self._n}.npy")
        np.save(p, img)
        old = self._paths.pop(key, None)
        if old and os.path.exists(old):
            try:
                os.remove(old)
            except OSError:
                pass
        self._paths[key] = p

    def load(self, key: str) -> np.ndarray | None:
        p = self._paths.get(key)
        if p and os.path.exists(p):
            return np.load(p)
        return None

    def remove(self, key: str) -> None:
        p = self._paths.pop(key, None)
        if p and os.path.exists(p):
            try:
                os.remove(p)
            except OSError:
                pass

    def cleanup(self):
        for p in self._paths.values():
            try:
                os.remove(p)
            except OSError:
                pass
        self._paths.clear()
        try:
            os.rmdir(self._tmpdir)
        except OSError:
            pass
