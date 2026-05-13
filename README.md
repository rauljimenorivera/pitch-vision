# pitch-vision

Computer Vision pipeline for football player and ball detection in football match footage.

This repository is an Unstructured Data course project that follows a full data science pipeline for image data: EDA, feature engineering, classical ML, deep learning from scratch, transfer learning, and rigorous model comparison.

**Deliverable:** the work is organized in **four notebooks** (`01_eda` through `04_yolo_transfer`). Team assignment (which club each detection belongs to) is **out of scope** and not implemented.

## Academic Scope

The mandatory project stages are:

1. Exploratory Data Analysis before modelling.
2. Feature engineering from raw pixels.
3. Classical Machine Learning baseline.
4. Deep Learning from scratch with iterative improvements.
5. Deep Learning with transfer learning.
6. Comparison and rigorous conclusions, including overfitting analysis (integrated mainly in `notebooks/04_yolo_transfer.ipynb`, with supporting runs in `03_yolo_scratch.ipynb`).

The chosen core image task is object detection for football players and the ball in match footage.

## Project summary

This work develops an end-to-end computer vision pipeline on football match imagery: detecting and localizing players and the ball with bounding boxes, using a public YOLO-format dataset and a standard train/validation split.

The approach moves from data understanding to a simple non-deep baseline, then to YOLOv8n trained from scratch and again with transfer learning from COCO, keeping the training setup comparable between the two deep runs.

### Main conclusions

- The problem is much easier for players than for the ball: few pixels per ball, class imbalance, and ambiguous appearance. Metrics and errors follow that split.
- Hand-crafted features and a linear classifier on crops show that local appearance alone is a weak solution and does not address full-image detection.
- Training from scratch already yields strong player detection and solid overall scores, but the ball stays the bottleneck without pretrained low-level vision.
- Transfer learning gives a clear gain in convergence and validation performance, with train and validation tracking together, so the emphasis is on better representations from pretraining rather than severe overfitting.

*Takeaway: classical cues, scratch deep detection, and COCO transfer are compared on the same task; players are easy, the ball is hard, and transfer learning closes much of that gap.*

## Repository Structure

```text
pitch-vision/
├── data/
│   └── raw/                        # ignored by git, populated by download-dataset
│       ├── images/train/
│       ├── images/val/
│       ├── labels/train/
│       └── labels/val/
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_classical_ml.ipynb
│   ├── 03_yolo_scratch.ipynb
│   └── 04_yolo_transfer.ipynb
├── src/pitch_vision/
│   ├── dataset.py
│   ├── eda_utils.py
│   ├── classical_ml.py
│   └── download_dataset.py         # dataset download script (uv run download-dataset)
├── outputs/
│   └── eda/                        # plots saved by notebook 01 and 02
├── runs/
│   └── detect/runs/                # YOLO training outputs (ignored by git)
│       ├── scratch/yolov8n_scratch_v1-3/
│       └── transfer/yolov8n_transfer_v1/
├── data.yaml
├── pyproject.toml
└── README.md
```

## Dataset

Use the Kaggle dataset **Football Player Detection YOLOv8** through KaggleHub:

```powershell
uv run download-dataset
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

### 3. YOLOv8 From Scratch

Notebook: `notebooks/03_yolo_scratch.ipynb`

The scratch model starts from randomly initialized YOLOv8 weights and is improved iteratively.

### 4. YOLOv8 Transfer Learning

Notebook: `notebooks/04_yolo_transfer.ipynb`

The transfer model starts from `yolov8n.pt`. This notebook also hosts the **scratch vs transfer comparison**, conclusions, and overfitting discussion required by the course.

For both YOLO stages, required run artifacts include `results.png`, `confusion_matrix.png`, `PR_curve.png`, and `F1_curve.png` (under each experiment folder in `runs/`, not committed to git).

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

Possible extensions outside the current four-notebook scope:

- Separate players by team using shirt color: extract upper-body crops from YOLO player detections, apply HSV filtering, and cluster dominant colors with K-Means to assign each player to one of two teams.
- Detect pitch lines with HSV masks, edge detection, and Hough transforms.
- Create tactical heatmaps from accumulated player positions across frames.

## References

- Ultralytics YOLOv8: https://docs.ultralytics.com/
- Dalal, N. & Triggs, B. (2005). Histograms of Oriented Gradients for Human Detection.
- uv: https://docs.astral.sh/uv/
