"""Unicode-safe image I/O and TIFF frame loading."""
import os
import cv2
import numpy as np
from PIL import Image


def cv_imread(path):
    """cv2.imread that works with unicode/cyrillic paths on Windows."""
    data = np.fromfile(path, dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def cv_imwrite(path, img, params=None):
    """cv2.imwrite that works with unicode/cyrillic paths on Windows."""
    ext = os.path.splitext(path)[1]
    ok, buf = cv2.imencode(ext, img, params or [])
    if ok:
        buf.tofile(path)
    return ok


def load_tiff_frame(virtual_path):
    """Load a specific frame from a multi-frame TIFF.
    virtual_path format: '/path/to/file.tif::frame0'
    Returns BGR numpy array or None.
    """
    if '::frame' not in virtual_path:
        return None
    real_path, frame_part = virtual_path.rsplit('::frame', 1)
    try:
        with Image.open(real_path) as pil_img:
            pil_img.seek(int(frame_part))
            return cv2.cvtColor(np.array(pil_img.convert('RGB')), cv2.COLOR_RGB2BGR)
    except Exception:
        return None


def count_tiff_frames(path):
    """Return number of frames in an image file. 1 for non-TIFF."""
    try:
        with Image.open(path) as pil_img:
            return getattr(pil_img, 'n_frames', 1)
    except Exception:
        return 1
