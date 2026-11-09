#!/usr/bin/env python3
"""Export the two checkpoints the Android app needs into app/src/main/assets.

    python android/export_models.py --coco yolo26n.pt --signs path/to/ITSD/best_Fold1.pt

Writes coco.onnx / signs.onnx (raw YOLO output, 640x640) and the matching
*_names.json class lists. Weights are not committed (see .gitignore).
"""
import argparse
import json
import shutil
from pathlib import Path

from ultralytics import YOLO

ASSETS = Path(__file__).resolve().parent / "app/src/main/assets"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--coco", default="yolo26n.pt")
    parser.add_argument("--signs", required=True, help="ITSD road-sign checkpoint (.pt)")
    args = parser.parse_args()
    ASSETS.mkdir(parents=True, exist_ok=True)
    for weights, stem in ((args.coco, "coco"), (args.signs, "signs")):
        model = YOLO(weights)
        onnx = model.export(format="onnx", imgsz=640, simplify=True, opset=17)
        shutil.copy(onnx, ASSETS / f"{stem}.onnx")
        names = model.names
        (ASSETS / f"{stem}_names.json").write_text(
            json.dumps([names[i] for i in range(len(names))], ensure_ascii=False), encoding="utf-8")
        print(f"[OK] {stem}: {len(names)} classes -> {ASSETS / (stem + '.onnx')}")


if __name__ == "__main__":
    main()
