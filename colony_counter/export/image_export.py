"""Image export — single and batch."""
import os
from pathlib import Path

from colony_counter.core.io_utils import cv_imwrite


def export_image(filepath: str, cv_img) -> None:
    """Save single CV2 image to file (PNG/JPG/BMP)."""
    cv_imwrite(filepath, cv_img)


def export_images_batch(output_dir: str, image_paths: list,
                        get_annotated_fn, display_names: dict,
                        fmt: str = 'png') -> int:
    """Export all processed images to a directory. Returns count exported."""
    os.makedirs(output_dir, exist_ok=True)
    count = 0
    for path in image_paths:
        ann = get_annotated_fn(path)
        if ann is None:
            continue
        name = display_names.get(path, Path(path).stem)
        safe = "".join(c if c.isalnum() or c in '._- ' else '_' for c in name)
        cv_imwrite(os.path.join(output_dir, f"{safe}_result.{fmt}"), ann)
        count += 1
    return count
