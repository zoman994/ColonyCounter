"""Lazy image cache — stores large images to temp files instead of RAM."""
import os
import tempfile

import cv2

from colony_counter.core.io_utils import cv_imread, cv_imwrite


class LazyImageCache:
    def __init__(self):
        self._tmpdir = tempfile.mkdtemp(prefix="colony_")
        self._n = 0
        self._paths = {}

    def store(self, key, img):
        self._n += 1
        p = os.path.join(self._tmpdir, f"i{self._n}.png")
        cv_imwrite(p, img, [cv2.IMWRITE_PNG_COMPRESSION, 1])
        old = self._paths.pop(key, None)
        if old and os.path.exists(old):
            try:
                os.remove(old)
            except OSError:
                pass
        self._paths[key] = p

    def load(self, key):
        p = self._paths.get(key)
        return cv_imread(p) if p and os.path.exists(p) else None

    def remove(self, key):
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
