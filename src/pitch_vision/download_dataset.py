"""Download and prepare a KaggleHub football detection dataset."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_DATASET = "iasadpanwhar/football-player-detection-yolov8"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def split_aliases(split: str) -> tuple[str, ...]:
    if split == "val":
        return ("val", "valid", "validation")
    return (split,)


def find_split_dir(dataset_root: Path, split: str, kind: str) -> Path | None:
    """Find a YOLO split directory in common Kaggle layouts."""

    candidates: list[Path] = []
    for split_name in split_aliases(split):
        candidates.extend(
            [
                dataset_root / kind / split_name,
                dataset_root / split_name / kind,
                dataset_root / split_name,
            ]
        )

    for candidate in candidates:
        if not candidate.exists() or not candidate.is_dir():
            continue
        if kind == "images" and any(
            path.suffix.lower() in IMAGE_EXTENSIONS for path in candidate.iterdir()
        ):
            return candidate
        if kind == "labels" and any(
            path.suffix.lower() == ".txt" for path in candidate.iterdir()
        ):
            return candidate
    return None


def copy_tree_contents(source: Path, destination: Path) -> int:
    """Copy files from one flat YOLO folder into the project raw directory."""

    destination.mkdir(parents=True, exist_ok=True)
    copied = 0
    for source_file in source.iterdir():
        if not source_file.is_file() or source_file.name == ".gitkeep":
            continue
        shutil.copy2(source_file, destination / source_file.name)
        copied += 1
    return copied


def read_yaml(path: Path) -> dict:
    """Read a YAML file with a small encoding fallback."""

    try:
        content = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        content = path.read_text()
    config = yaml.safe_load(content)
    return config if isinstance(config, dict) else {}


def find_dataset_yaml(dataset_root: Path) -> Path | None:
    """Return the first YOLO dataset YAML found in a downloaded dataset."""

    yaml_files = sorted(dataset_root.rglob("*.yaml")) + sorted(dataset_root.rglob("*.yml"))
    return yaml_files[0] if yaml_files else None


def path_from_yaml(dataset_yaml: Path, config: dict, split: str, kind: str) -> Path | None:
    """Resolve image and label directories from a Roboflow/YOLO data.yaml."""

    yaml_split = "valid" if split == "val" and "valid" in config else split
    image_entry = config.get(yaml_split) or config.get(split)
    if image_entry is None:
        return None

    image_dir = (dataset_yaml.parent / image_entry).resolve()
    if kind == "images":
        return image_dir

    parts = list(image_dir.parts)
    if "images" not in parts:
        return None
    parts[parts.index("images")] = "labels"
    return Path(*parts)


def prepare_yolo_layout(dataset_root: Path, raw_root: Path) -> None:
    """Copy a detected Kaggle YOLO layout into data/raw."""

    dataset_yaml = find_dataset_yaml(dataset_root)
    config = read_yaml(dataset_yaml) if dataset_yaml is not None else {}
    search_roots = [dataset_root]
    if dataset_yaml is not None:
        search_roots.insert(0, dataset_yaml.parent)

    for split in ("train", "val"):
        image_source = (
            path_from_yaml(dataset_yaml, config, split, "images")
            if dataset_yaml is not None
            else None
        )
        label_source = (
            path_from_yaml(dataset_yaml, config, split, "labels")
            if dataset_yaml is not None
            else None
        )

        if image_source is None or not image_source.exists():
            image_source = next(
                (
                    candidate
                    for root in search_roots
                    if (candidate := find_split_dir(root, split, "images")) is not None
                ),
                None,
            )
        if label_source is None or not label_source.exists():
            label_source = next(
                (
                    candidate
                    for root in search_roots
                    if (candidate := find_split_dir(root, split, "labels")) is not None
                ),
                None,
            )

        if image_source is None:
            print(f"Could not find image folder for split '{split}'.")
        else:
            copied = copy_tree_contents(image_source, raw_root / "images" / split)
            print(f"Copied {copied} {split} images from {image_source}")

        if label_source is None:
            print(f"Could not find label folder for split '{split}'.")
        else:
            copied = copy_tree_contents(label_source, raw_root / "labels" / split)
            print(f"Copied {copied} {split} labels from {label_source}")


def print_dataset_yaml(dataset_root: Path) -> None:
    """Print class metadata if the downloaded dataset includes a YAML file."""

    dataset_yaml = find_dataset_yaml(dataset_root)
    if dataset_yaml is None:
        print("No dataset YAML file found. Verify class IDs in the EDA notebook.")
        return

    print()
    print(f"Found dataset YAML: {dataset_yaml}")
    config = read_yaml(dataset_yaml)

    names = config.get("names")
    nc = config.get("nc")
    print(f"Dataset nc: {nc}")
    print(f"Dataset names: {names}")


def ensure_data_dirs(raw_root: Path) -> None:
    """Create the full expected data/raw directory tree if not already present."""

    for split in ("train", "val"):
        (raw_root / "images" / split).mkdir(parents=True, exist_ok=True)
        (raw_root / "labels" / split).mkdir(parents=True, exist_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download and validate a KaggleHub YOLO dataset.")
    parser.add_argument(
        "--dataset",
        default=DEFAULT_DATASET,
        help="KaggleHub dataset slug.",
    )
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=PROJECT_ROOT / "data" / "raw",
        help="Directory that contains images/{train,val} and labels/{train,val}.",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Only validate the existing local data/raw layout.",
    )
    return parser.parse_args()


def main() -> None:
    from pitch_vision.dataset import validate_layout

    args = parse_args()
    ensure_data_dirs(args.raw_root)

    if not args.skip_download:
        import kagglehub

        print(f"Downloading KaggleHub dataset: {args.dataset}")
        dataset_path = Path(kagglehub.dataset_download(args.dataset))
        print(f"Path to dataset files: {dataset_path}")
        print_dataset_yaml(dataset_path)
        print()
        prepare_yolo_layout(dataset_path, args.raw_root)
        print()

    summary = validate_layout(args.raw_root)
    print("Expected local layout:")
    print(f"  {args.raw_root / 'images' / 'train'}")
    print(f"  {args.raw_root / 'images' / 'val'}")
    print(f"  {args.raw_root / 'labels' / 'train'}")
    print(f"  {args.raw_root / 'labels' / 'val'}")
    print()
    print(summary.to_string(index=False))

    if summary["images"].sum() == 0:
        raise SystemExit(
            "No images found. Check the downloaded Kaggle layout or copy YOLO images/labels "
            "into data/raw manually before running the notebooks."
        )


if __name__ == "__main__":
    main()
