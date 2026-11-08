#!/usr/bin/env python3
"""Fine-tune a YOLO26 detector for pedestrian attribute categories.

The dataset must use the directory structure described in
data/pedestrian_attributes.yaml. See docs/RESEARCH_GUIDE.md for the annotation policy.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
import yaml
from ultralytics import YOLO


REQUIRED_CLASSES = {
    "child",
    "woman",
    "elderly_person",
    "person_with_wheelchair",
    "other_person",
}


def _class_names(raw_names: Any) -> set[str]:
    if isinstance(raw_names, dict):
        return {str(name) for name in raw_names.values()}
    if isinstance(raw_names, list):
        return {str(name) for name in raw_names}
    return set()


def validate_dataset_config(data_yaml: Path) -> None:
    """Fail early when the YAML or required dataset splits are missing."""
    if not data_yaml.is_file():
        raise FileNotFoundError(
            f"Dataset config not found: {data_yaml}\n"
            "Run this script from the repository root or pass --data."
        )

    with data_yaml.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}

    missing_classes = REQUIRED_CLASSES - _class_names(config.get("names"))
    if missing_classes:
        raise ValueError(
            "Dataset YAML is missing class names: "
            + ", ".join(sorted(missing_classes))
        )

    dataset_root_value = config.get("path")
    dataset_root = (
        Path(dataset_root_value).expanduser()
        if dataset_root_value
        else data_yaml.parent
    )
    if not dataset_root.is_absolute():
        dataset_root = (data_yaml.parent / dataset_root).resolve()

    for split in ("train", "val"):
        split_value = config.get(split)
        if not split_value:
            raise ValueError(f"Dataset YAML must define the '{split}' split.")

        split_paths = split_value if isinstance(split_value, list) else [split_value]
        for value in split_paths:
            split_path = Path(value).expanduser()
            if not split_path.is_absolute():
                split_path = dataset_root / split_path
            if not split_path.exists():
                raise FileNotFoundError(
                    f"Dataset {split} path does not exist: {split_path}"
                )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune a YOLO model on the pedestrian attribute dataset."
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/pedestrian_attributes.yaml"),
        help="Path to the dataset YAML (default: data/pedestrian_attributes.yaml).",
    )
    parser.add_argument(
        "--model",
        default="yolo26n.pt",
        help="Pretrained checkpoint used for transfer learning (default: yolo26n.pt).",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument(
        "--batch",
        type=int,
        default=8,
        help="Batch size. Lower this if you run out of GPU memory.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Training device: auto, cpu, 0, 0,1, etc. (default: auto).",
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--project", default="runs/pedestrian")
    parser.add_argument("--name", default="yolo26n-pedestrian-attributes")
    parser.add_argument(
        "--cache",
        action="store_true",
        help="Cache images in memory/disk when supported by Ultralytics.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data_yaml = args.data.expanduser().resolve()

    if args.epochs < 1 or args.imgsz < 32 or args.batch == 0:
        raise ValueError("--epochs and --imgsz must be positive; --batch cannot be 0.")

    validate_dataset_config(data_yaml)

    if args.device.lower() == "auto":
        device: str | int = 0 if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    if device == "cpu":
        print("[INFO] Training on CPU. This will be substantially slower than CUDA.")
    elif torch.cuda.is_available():
        print(f"[INFO] CUDA GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("[INFO] CUDA is not available; verify --device before starting training.")

    print(f"[INFO] Base checkpoint: {args.model}")
    print(f"[INFO] Dataset config: {data_yaml}")
    print("[INFO] Classes: child, woman, elderly_person, "
          "person_with_wheelchair, other_person")

    model = YOLO(args.model)
    results = model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        workers=args.workers,
        patience=args.patience,
        project=args.project,
        name=args.name,
        cache=args.cache,
    )

    save_dir = getattr(results, "save_dir", None)
    if save_dir:
        best_weights = Path(save_dir) / "weights" / "best.pt"
        print(f"\n[OK] Training finished. Best checkpoint: {best_weights}")
        print("[NEXT] Benchmark it with:")
        print(
            "python VideoBenchmark.py --source path/to/video.mp4 "
            f'--weights "{best_weights}"'
        )
    else:
        print("[OK] Training finished. Check the run directory for weights/best.pt.")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(f"[ERROR] {exc}")
