#!/usr/bin/env python3
"""Evaluate a YOLO detector on a labelled dataset and export metrics as JSON."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import torch
from ultralytics import YOLO


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _class_name(names: Any, class_id: int) -> str:
    if isinstance(names, dict):
        return str(names.get(class_id, names.get(str(class_id), class_id)))
    if isinstance(names, (list, tuple)) and 0 <= class_id < len(names):
        return str(names[class_id])
    return str(class_id)


def build_report(metrics: Any, names: Any, config: dict[str, Any]) -> dict[str, Any]:
    """Convert Ultralytics detection metrics into a stable JSON-ready report."""
    box = metrics.box
    class_ids = [int(value) for value in getattr(box, "ap_class_index", [])]
    precision = list(getattr(box, "p", []))
    recall = list(getattr(box, "r", []))
    f1 = list(getattr(box, "f1", []))
    ap50 = list(getattr(box, "ap50", []))
    maps = list(getattr(box, "maps", []))

    per_class = []
    for row, class_id in enumerate(class_ids):
        per_class.append(
            {
                "class_id": class_id,
                "class_name": _class_name(names, class_id),
                "precision": _float_or_none(precision[row]) if row < len(precision) else None,
                "recall": _float_or_none(recall[row]) if row < len(recall) else None,
                "f1": _float_or_none(f1[row]) if row < len(f1) else None,
                "map50": _float_or_none(ap50[row]) if row < len(ap50) else None,
                "map50_95": _float_or_none(maps[class_id]) if class_id < len(maps) else None,
            }
        )

    image_metrics = getattr(box, "image_metrics", None) or {}
    weakest_images = []
    for image_path, values in sorted(
        image_metrics.items(),
        key=lambda item: _float_or_none(item[1].get("f1")) or 0.0,
    )[: config["top_images"]]:
        weakest_images.append(
            {
                "image": str(image_path),
                "precision": _float_or_none(values.get("precision")),
                "recall": _float_or_none(values.get("recall")),
                "f1": _float_or_none(values.get("f1")),
                "tp": int(values.get("tp", 0)),
                "fp": int(values.get("fp", 0)),
                "fn": int(values.get("fn", 0)),
            }
        )

    return {
        "model": config["model"],
        "dataset": config["data"],
        "split": config["split"],
        "device": config["device"],
        "image_size": config["imgsz"],
        "confidence_threshold": config["conf"],
        "iou_threshold": config["iou"],
        "overall": {
            "precision": _float_or_none(box.mp),
            "recall": _float_or_none(box.mr),
            "map50": _float_or_none(box.map50),
            "map75": _float_or_none(box.map75),
            "map50_95": _float_or_none(box.map),
            "fitness": _float_or_none(box.fitness()),
        },
        "per_class": per_class,
        "weakest_images": weakest_images,
        "timing_ms_per_image": {
            key: _float_or_none(value)
            for key, value in getattr(metrics, "speed", {}).items()
        },
        "notes": [
            "Metrics are computed against labels in the selected dataset split.",
            "Per-image precision, recall and F1 are measured at IoU 0.5.",
            "Use a held-out test split for final reporting; do not tune on test results.",
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate YOLO detection accuracy on a labelled dataset."
    )
    parser.add_argument("--weights", required=True, help="Model checkpoint (.pt).")
    parser.add_argument("--data", default="data/pedestrian_attributes.yaml", help="Dataset YAML.")
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--device", default="auto", help="auto, cpu, 0, cuda:0, etc.")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--conf", type=float, default=0.001)
    parser.add_argument("--iou", type=float, default=0.6)
    parser.add_argument("--top-images", type=int, default=20)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    weights = Path(args.weights).expanduser().resolve()
    data = Path(args.data).expanduser().resolve()
    if not weights.is_file():
        raise FileNotFoundError(f"Model weights not found: {weights}")
    if not data.is_file():
        raise FileNotFoundError(f"Dataset YAML not found: {data}")
    if args.imgsz < 32 or args.batch < 1 or args.workers < 0 or args.top_images < 0:
        raise ValueError("--imgsz must be >=32, --batch >=1, and workers/top-images >=0.")
    if not 0.0 <= args.conf < 1.0 or not 0.0 < args.iou <= 1.0:
        raise ValueError("--conf must be in [0,1) and --iou in (0,1].")

    device = args.device
    if device.lower() == "auto":
        device = "0" if torch.cuda.is_available() else "cpu"
    if (device.startswith("cuda") or device.split(",", 1)[0].isdigit()) and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but PyTorch cannot access a CUDA GPU.")

    model = YOLO(str(weights))
    metrics = model.val(
        data=str(data),
        split=args.split,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        conf=args.conf,
        iou=args.iou,
        device=device,
        plots=not args.no_plots,
        verbose=True,
    )
    report = build_report(
        metrics,
        model.names,
        {
            "model": str(weights), "data": str(data), "split": args.split,
            "device": device, "imgsz": args.imgsz, "conf": args.conf,
            "iou": args.iou, "top_images": args.top_images,
        },
    )
    output = args.output.expanduser().resolve() if args.output else (
        Path("runs/evaluation") / f"{weights.stem}_{args.split}_metrics.json"
    ).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    overall = report["overall"]
    print("\n========== DETECTOR EVALUATION ==========")
    for key in ("precision", "recall", "map50", "map75", "map50_95"):
        value = overall[key]
        print(f"{key:12}: {value:.4f}" if value is not None else f"{key:12}: n/a")
    print(f"Per-class metrics: {len(report['per_class'])} classes")
    print(f"Weakest images:    {len(report['weakest_images'])}")
    print(f"JSON report:       {output}")
    print("=========================================")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        raise SystemExit(f"[ERROR] {exc}")
