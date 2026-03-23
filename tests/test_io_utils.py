"""Tests for colony_counter.core.io_utils."""
import os
import tempfile

import cv2
import numpy as np
import pytest

from colony_counter.core.io_utils import cv_imread, cv_imwrite, count_tiff_frames


class TestCvImreadWrite:
    def test_roundtrip(self, tmp_path):
        img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        path = str(tmp_path / "test.png")
        cv_imwrite(path, img)
        loaded = cv_imread(path)
        assert loaded is not None
        assert loaded.shape == img.shape

    def test_unicode_path(self, tmp_path):
        img = np.zeros((50, 50, 3), dtype=np.uint8)
        img[:, :, 2] = 255  # red
        path = str(tmp_path / "тест_кириллица.png")
        cv_imwrite(path, img)
        loaded = cv_imread(path)
        assert loaded is not None
        assert loaded[0, 0, 2] == 255


class TestCountTiffFrames:
    def test_single_frame_png(self, tmp_path):
        from PIL import Image
        img = Image.new('RGB', (10, 10), 'red')
        path = str(tmp_path / "test.png")
        img.save(path)
        assert count_tiff_frames(path) == 1

    def test_nonexistent_file(self):
        assert count_tiff_frames("/nonexistent/path.tif") == 1
