"""Classical machine learning baseline with HOG features and SVM."""

from __future__ import annotations

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from skimage.feature import hog
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from pitch_vision.dataset import imread_unicode
from pitch_vision.eda_utils import save_figure


def crop_from_yolo_row(image: np.ndarray, row: pd.Series) -> np.ndarray | None:
    """Crop one normalized YOLO box from an image."""

    height, width = image.shape[:2]
    box_width = float(row["bbox_width"]) * width
    box_height = float(row["bbox_height"]) * height
    x_center = float(row["x_center"]) * width
    y_center = float(row["y_center"]) * height

    x1 = max(0, int(x_center - box_width / 2))
    y1 = max(0, int(y_center - box_height / 2))
    x2 = min(width, int(x_center + box_width / 2))
    y2 = min(height, int(y_center + box_height / 2))
    if x2 <= x1 or y2 <= y1:
        return None
    return image[y1:y2, x1:x2]


def extract_hog_features(crop: np.ndarray, size: tuple[int, int] = (64, 128)) -> np.ndarray:
    """Extract HOG features from a crop resized to a fixed size."""

    resized = cv2.resize(crop, size)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    return hog(
        gray,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        feature_vector=True,
    )


def build_crop_feature_table(
    annotations: pd.DataFrame,
    max_per_class: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Build an X/y dataset from annotated object crops."""

    labeled = annotations.dropna(subset=["class_id"]).copy()
    if max_per_class is not None:
        # groupby.apply drops the grouping column in pandas >=3.0; iterate instead.
        parts = [
            group.sample(min(len(group), max_per_class), random_state=42)
            for _, group in labeled.groupby("class_id", group_keys=False)
        ]
        labeled = pd.concat(parts).reset_index(drop=True) if parts else labeled.iloc[0:0]

    features: list[np.ndarray] = []
    targets: list[int] = []
    image_cache: dict[Path, np.ndarray] = {}

    for _, row in labeled.iterrows():
        image_path = Path(row["image_path"])
        if image_path not in image_cache:
            image = imread_unicode(image_path)
            if image is None:
                continue
            image_cache[image_path] = image

        crop = crop_from_yolo_row(image_cache[image_path], row)
        if crop is None:
            continue
        features.append(extract_hog_features(crop))
        targets.append(int(row["class_id"]))

    if not features:
        return np.empty((0, 0)), np.empty((0,), dtype=int)
    return np.vstack(features), np.asarray(targets, dtype=int)


def train_svm_baseline(x_train: np.ndarray, y_train: np.ndarray) -> Pipeline:
    """Train a linear SVM baseline on HOG features."""

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("classifier", LinearSVC(class_weight="balanced", random_state=42, max_iter=10_000)),
        ]
    )
    model.fit(x_train, y_train)
    return model


def evaluate_classifier(
    model: Pipeline,
    x_test: np.ndarray,
    y_test: np.ndarray,
    class_names: dict[int, str],
    output_path: str | Path,
) -> str:
    """Save a confusion matrix and return a text classification report."""

    predictions = model.predict(x_test)
    labels = sorted(class_names)
    matrix = confusion_matrix(y_test, predictions, labels=labels)

    fig, ax = plt.subplots(figsize=(6, 5))
    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=[class_names[label] for label in labels],
    )
    display.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("HOG + Linear SVM confusion matrix")
    save_figure(fig, output_path)
    plt.close(fig)

    # Keep seaborn imported for notebook users who style follow-up plots consistently.
    sns.set_theme(style="whitegrid")
    return classification_report(
        y_test,
        predictions,
        labels=labels,
        target_names=[class_names[label] for label in labels],
        zero_division=0,
    )
