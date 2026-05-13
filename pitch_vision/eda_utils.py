"""Plotting and visualization helpers for dataset exploration."""

from __future__ import annotations

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pitch_vision.dataset import imread_unicode, label_path_for_image, read_yolo_labels


def ensure_parent(path: str | Path) -> Path:
    """Create the parent directory for an output path."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def save_figure(fig: plt.Figure, output_path: str | Path, dpi: int = 150) -> None:
    """Save a Matplotlib figure using a consistent layout."""

    output_path = ensure_parent(output_path)
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")


def plot_class_distribution(
    annotations: pd.DataFrame,
    class_names: dict[int, str],
    output_path: str | Path,
) -> None:
    """Plot annotation counts by class."""

    labeled = annotations.dropna(subset=["class_id"]).copy()
    labeled["class_name"] = labeled["class_id"].astype(int).map(class_names)

    fig, ax = plt.subplots(figsize=(7, 4))
    labeled["class_name"].value_counts().sort_index().plot(kind="bar", ax=ax)
    ax.set_title("Class distribution")
    ax.set_xlabel("Class")
    ax.set_ylabel("Bounding boxes")
    save_figure(fig, output_path)
    plt.close(fig)


def plot_bbox_distributions(annotations: pd.DataFrame, output_dir: str | Path) -> None:
    """Save size and aspect-ratio diagnostic plots for bounding boxes."""

    labeled = annotations.dropna(subset=["bbox_area", "aspect_ratio"])
    output_dir = Path(output_dir)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(labeled["bbox_area"], bins=40)
    ax.set_title("Normalized bounding box area")
    ax.set_xlabel("Area")
    ax.set_ylabel("Count")
    save_figure(fig, output_dir / "bbox_size_distribution.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter(labeled["bbox_width"], labeled["bbox_height"], alpha=0.35)
    ax.set_title("Bounding box width vs height")
    ax.set_xlabel("Normalized width")
    ax.set_ylabel("Normalized height")
    save_figure(fig, output_dir / "bbox_aspect_ratio_scatter.png")
    plt.close(fig)


def plot_rgb_histograms(image_paths: list[Path], output_path: str | Path, max_images: int = 50) -> None:
    """Plot RGB channel histograms from a sample of images."""

    channels = {"red": [], "green": [], "blue": []}
    for image_path in image_paths[:max_images]:
        image = imread_unicode(image_path)
        if image is None:
            continue
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        for idx, name in enumerate(("red", "green", "blue")):
            hist = cv2.calcHist([rgb], [idx], None, [256], [0, 256]).ravel()
            channels[name].append(hist)

    fig, ax = plt.subplots(figsize=(8, 4))
    for name, histograms in channels.items():
        if histograms:
            ax.plot(np.mean(histograms, axis=0), label=name)
    ax.set_title("Average RGB histograms")
    ax.set_xlabel("Pixel intensity")
    ax.set_ylabel("Average frequency")
    ax.legend()
    save_figure(fig, output_path)
    plt.close(fig)


def draw_yolo_boxes(
    image_path: str | Path,
    raw_root: str | Path,
    class_names: dict[int, str],
) -> np.ndarray:
    """Draw YOLO annotations on an image and return RGB pixels."""

    image_path = Path(image_path)
    image = imread_unicode(image_path)
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    height, width = image.shape[:2]
    for label in read_yolo_labels(label_path_for_image(image_path, raw_root)):
        x1 = int((label.x_center - label.width / 2) * width)
        y1 = int((label.y_center - label.height / 2) * height)
        x2 = int((label.x_center + label.width / 2) * width)
        y2 = int((label.y_center + label.height / 2) * height)
        color = (0, 255, 0) if label.class_id == 0 else (0, 165, 255)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            image,
            class_names.get(label.class_id, str(label.class_id)),
            (x1, max(0, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )

    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def save_annotated_grid(
    image_paths: list[Path],
    raw_root: str | Path,
    class_names: dict[int, str],
    output_path: str | Path,
    columns: int = 3,
) -> None:
    """Save a grid of annotated sample images."""

    sample = image_paths[: columns * 2]
    rows = int(np.ceil(len(sample) / columns)) if sample else 1
    fig, axes = plt.subplots(rows, columns, figsize=(4 * columns, 4 * rows))
    axes_array = np.atleast_1d(axes).ravel()

    for ax, image_path in zip(axes_array, sample, strict=False):
        ax.imshow(draw_yolo_boxes(image_path, raw_root, class_names))
        ax.set_title(Path(image_path).name)
        ax.axis("off")

    for ax in axes_array[len(sample) :]:
        ax.axis("off")

    save_figure(fig, output_path)
    plt.close(fig)
