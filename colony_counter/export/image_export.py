"""Image export — save annotated image to file."""
from colony_counter.core.io_utils import cv_imwrite


def export_image(filepath, cv_img):
    """Save CV2 image to file (PNG/JPG/BMP)."""
    cv_imwrite(filepath, cv_img)
