# pitch-vision

Computer Vision pipeline for football player and ball detection in football match footage.

This repository is an Unstructured Data course project that follows a full data science pipeline for image data: EDA, feature engineering, classical ML, deep learning from scratch, transfer learning, and rigorous model comparison.

## Academic Scope

The mandatory project stages are:

1. Exploratory Data Analysis before modelling.
2. Feature engineering from raw pixels.
3. Classical Machine Learning baseline.
4. Deep Learning from scratch with iterative improvements.
5. Deep Learning with transfer learning.
6. Comparison and rigorous conclusions, including overfitting analysis.

The chosen core image task is object detection for football players and the ball in match footage.

## Repository Structure

```text
pitch-vision/
├── data/
│   ├── raw/
│   │   ├── images/train/
│   │   ├── images/val/
│   │   ├── labels/train/
│   │   └── labels/val/
│   └── processed/
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_classical_ml.ipynb
│   ├── 03_yolo_scratch.ipynb
│   └── 04_yolo_transfer.ipynb
├── src/pitch_vision/
│   ├── dataset.py
│   ├── eda_utils.py
│   └── classical_ml.py
├── scripts/download_dataset.py
├── outputs/
├── runs/
├── videos/
├── data.yaml
├── pyproject.toml
└── README.md
```

## Dataset

Use the Kaggle dataset **Football Player Detection YOLOv8** through KaggleHub:

```powershell
uv run python scripts/download_dataset.py
```

The default KaggleHub slug is `iasadpanwhar/football-player-detection-yolov8`.

The expected YOLO layout is:

```text
data/raw/
├── images/
│   ├── train/
│   └── val/
└── labels/
    ├── train/
    └── val/
```

YOLO labels use normalized coordinates:

```text
<class_id> <x_center> <y_center> <width> <height>
```

Always verify the actual class IDs during EDA before training.

## Milestones

### 1. Exploratory Data Analysis

Notebook: `notebooks/01_eda.ipynb`

Required outputs in `outputs/eda/`:

- `class_distribution.png`
- `bbox_size_distribution.png`
- `bbox_aspect_ratio_scatter.png`
- `color_histograms_rgb.png`
- `annotated_samples_grid.png`

The final notebook cell must explain the main dataset characteristics, modelling challenges, and decisions for the next stages.

### 2. Classical ML Baseline

Notebook: `notebooks/02_classical_ml.ipynb`

The detection task is reframed as crop-level classification using HOG features and an SVM classifier. The goal is an honest non-DL baseline, not heavy SVM tuning.

Required output:

- `outputs/eda/classical_ml_confusion_matrix.png`

### 3. YOLOv8 From Scratch and Transfer Learning

Notebooks:

- `notebooks/03_yolo_scratch.ipynb`
- `notebooks/04_yolo_transfer.ipynb`

The scratch model starts from randomly initialized YOLOv8 weights and is improved iteratively. The transfer model starts from `yolov8n.pt`.

Required YOLO artifacts include `results.png`, `confusion_matrix.png`, `PR_curve.png`, and `F1_curve.png`.

## Setup

Install `uv`, then run:

```powershell
uv sync
uv run python -c "import torch; print(torch.cuda.is_available())"
```

For GTX 1650 training, start with:

- `imgsz=640`
- `batch=4`
- `workers=2`
- YOLOv8n before larger models

## Working Conventions

- Keep raw data, generated outputs, YOLO runs, and model weights out of git.
- Keep reusable logic in `src/pitch_vision/`.
- Keep notebooks focused on narrative, configuration, execution, and conclusions.
- Save every figure before showing it.
- Clear notebook outputs before committing.
- Run `uv run ruff check .` before review.

## Future Work

The following ideas are intentionally left out of the current implementation scope and may be explored in a later version:

- Separate players by team using shirt color: extract upper-body crops from YOLO player detections, apply HSV filtering, and cluster dominant colors with K-Means to assign each player to one of two teams.
- Detect pitch lines with HSV masks, edge detection, and Hough transforms.
- Create tactical heatmaps from accumulated player positions across frames.

## References

- Ultralytics YOLOv8: https://docs.ultralytics.com/
- Dalal, N. & Triggs, B. (2005). Histograms of Oriented Gradients for Human Detection.
- uv: https://docs.astral.sh/uv/
