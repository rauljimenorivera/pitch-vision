"""Dataset helpers for YOLO-formatted football detection data."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass(frozen=True)
class YoloLabel:
    """One normalized YOLO bounding box annotation."""

    class_id: int
    x_center: float
    y_center: float
    width: float
    height: float

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def aspect_ratio(self) -> float:
        if self.height == 0:
            return 0.0
        return self.width / self.height


def iter_images(raw_root: str | Path, split: str) -> list[Path]:
    """Return image files for a split using the expected YOLO layout."""

    image_dir = Path(raw_root) / "images" / split
    if not image_dir.exists():
        return []
    return sorted(path for path in image_dir.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS)


def label_path_for_image(image_path: str | Path, raw_root: str | Path) -> Path:
    """Map a YOLO image path to its corresponding label path."""

    image_path = Path(image_path)
    raw_root = Path(raw_root)
    relative = image_path.relative_to(raw_root / "images")
    return raw_root / "labels" / relative.with_suffix(".txt")


def read_yolo_labels(label_path: str | Path) -> list[YoloLabel]:
    """Read YOLO labels from a txt file. Missing files return an empty list."""

    label_path = Path(label_path)
    if not label_path.exists():
        return []

    labels: list[YoloLabel] = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        class_id, x_center, y_center, width, height = line.split()
        labels.append(
            YoloLabel(
                class_id=int(class_id),
                x_center=float(x_center),
                y_center=float(y_center),
                width=float(width),
                height=float(height),
            )
        )
    return labels


def imread_unicode(image_path: str | Path) -> np.ndarray | None:
    """Read an image from any path, including those with non-ASCII characters.

    cv2.imread silently returns None on Windows when the path contains
    characters outside the current ANSI code page (e.g. accented letters).
    Reading raw bytes via numpy sidesteps that limitation.
    """
    data = np.fromfile(str(image_path), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def image_shape(image_path: str | Path) -> tuple[int, int]:
    """Return image shape as (height, width) by reading only the file header."""

    with Image.open(image_path) as img:
        width, height = img.size
    return height, width


def collect_annotations(raw_root: str | Path, splits: tuple[str, ...] = ("train", "val")) -> pd.DataFrame:
    """Collect YOLO annotations into a tabular dataframe for EDA and baselines."""

    rows: list[dict[str, object]] = []
    raw_root = Path(raw_root)

    for split in splits:
        for image_path in iter_images(raw_root, split):
            label_path = label_path_for_image(image_path, raw_root)
            labels = read_yolo_labels(label_path)
            height, width = image_shape(image_path)

            if not labels:
                rows.append(
                    {
                        "split": split,
                        "image_path": image_path,
                        "label_path": label_path,
                        "image_width": width,
                        "image_height": height,
                        "class_id": None,
                        "x_center": None,
                        "y_center": None,
                        "bbox_width": None,
                        "bbox_height": None,
                        "bbox_area": None,
                        "aspect_ratio": None,
                    }
                )
                continue

            for label in labels:
                rows.append(
                    {
                        "split": split,
                        "image_path": image_path,
                        "label_path": label_path,
                        "image_width": width,
                        "image_height": height,
                        "class_id": label.class_id,
                        "x_center": label.x_center,
                        "y_center": label.y_center,
                        "bbox_width": label.width,
                        "bbox_height": label.height,
                        "bbox_area": label.area,
                        "aspect_ratio": label.aspect_ratio,
                    }
                )

    return pd.DataFrame(rows)


def validate_layout(raw_root: str | Path) -> pd.DataFrame:
    """Summarize image and label counts by split."""

    raw_root = Path(raw_root)
    rows = []
    for split in ("train", "val"):
        image_count = len(iter_images(raw_root, split))
        label_dir = raw_root / "labels" / split
        label_count = len(list(label_dir.glob("*.txt"))) if label_dir.exists() else 0
        rows.append({"split": split, "images": image_count, "labels": label_count})
    return pd.DataFrame(rows)
