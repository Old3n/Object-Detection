#!/usr/bin/env python3
"""Run a trained Ultralytics YOLO detector on a fixed video and benchmark it.

The benchmark is intentionally detection-only: tracker startup/history and
webcam capture buffering do not contaminate model inference measurements.
It reports model-call latency and end-to-end frame processing throughput.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import torch
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark a YOLO model on a video and save annotated output."
    )
    parser.add_argument("--source", required=True, type=Path, help="Input video path.")
    parser.add_argument("--weights", required=True, help="Model checkpoint (.pt).")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Annotated output video. Default: <source>_annotated.mp4",
    )
    parser.add_argument(
        "--csv",
        dest="csv_path",
        type=Path,
        default=None,
        help="Per-frame CSV. Default: <source>_benchmark.csv",
    )
    parser.add_argument(
        "--json",
        dest="json_path",
        type=Path,
        default=None,
        help="Summary JSON. Default: <source>_benchmark.json",
    )
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size.")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold.")
    parser.add_argument("--iou", type=float, default=0.7, help="NMS IoU threshold.")
    parser.add_argument(
        "--device",
        default="auto",
        help="Inference device: auto, cpu, 0, cuda:0, etc.",
    )
    parser.add_argument(
        "--classes",
        type=int,
        nargs="+",
        default=None,
        help="Optional class IDs to keep, e.g. --classes 0 1 2.",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=10,
        help="Number of warm-up frames excluded from benchmark results.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional maximum number of measured frames (after warm-up).",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the annotated video while processing (reduces throughput).",
    )
    parser.add_argument(
        "--no-save-video",
        action="store_true",
        help="Do not write the annotated output video.",
    )
    parser.add_argument(
        "--quantize", choices=("fp32", "fp16"), default="fp32",
        help="Inference precision: fp32 (default) or fp16 (CUDA only).",
    )
    return parser.parse_args()


def is_cuda_device(device: str) -> bool:
    """Recognize both Ultralytics device syntax (0) and PyTorch syntax (cuda:0)."""
    if device.startswith("cuda"):
        return True
    parts = device.split(",")
    return bool(parts) and all(part.strip().isdigit() for part in parts)


def sync_device(device: str) -> None:
    """Wait for queued CUDA kernels so wall-clock inference timing is accurate."""
    if not is_cuda_device(device) or not torch.cuda.is_available():
        return
    if device.startswith("cuda:"):
        index = device.split(":", 1)[1].split(",", 1)[0]
        torch.cuda.synchronize(int(index)) if index.isdigit() else torch.cuda.synchronize()
    elif device.split(",", 1)[0].strip().isdigit():
        torch.cuda.synchronize(int(device.split(",", 1)[0].strip()))
    else:
        torch.cuda.synchronize()


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * pct
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def class_name(names: Any, class_id: int) -> str:
    if isinstance(names, dict):
        return str(names.get(class_id, names.get(str(class_id), class_id)))
    if isinstance(names, (list, tuple)) and 0 <= class_id < len(names):
        return str(names[class_id])
    return str(class_id)


def main() -> int:
    args = parse_args()

    source = args.source.expanduser().resolve()
    weights = Path(args.weights).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Input video does not exist: {source}")
    if not weights.is_file():
        raise FileNotFoundError(f"Model weights do not exist: {weights}")
    if args.imgsz < 32:
        raise ValueError("--imgsz must be at least 32.")
    if args.warmup < 0:
        raise ValueError("--warmup cannot be negative.")
    if args.max_frames is not None and args.max_frames < 1:
        raise ValueError("--max-frames must be positive.")
    if not 0.0 < args.conf <= 1.0 or not 0.0 < args.iou <= 1.0:
        raise ValueError("--conf and --iou must be in the range (0, 1].")

    if args.device.lower() == "auto":
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    if is_cuda_device(device) and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested but PyTorch cannot access a CUDA GPU. "
            "Install a CUDA-enabled PyTorch build or use --device cpu."
        )

    use_fp16 = args.quantize == "fp16" and is_cuda_device(device)
    if args.quantize == "fp16" and not use_fp16:
        print("[WARN] FP16 requested without CUDA; using FP32 on CPU.")
    quantize = 16 if use_fp16 else 32

    output = (
        args.output.expanduser().resolve()
        if args.output
        else source.with_name(f"{source.stem}_annotated.mp4")
    )
    if output == source:
        raise ValueError("Annotated output path must not overwrite the input video.")

    csv_path = (
        args.csv_path.expanduser().resolve()
        if args.csv_path
        else source.with_name(f"{source.stem}_benchmark.csv")
    )
    json_path = (
        args.json_path.expanduser().resolve()
        if args.json_path
        else source.with_name(f"{source.stem}_benchmark.json")
    )

    for path in (output, csv_path, json_path):
        path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Loading weights: {weights}")
    model = YOLO(str(weights))
    names = model.names
    print(f"[INFO] Device: {device}" + (", FP16" if use_fp16 else ", FP32"))
    print(f"[INFO] Classes: {names}")

    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV could not open input video: {source}")

    source_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames_hint = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if width <= 0 or height <= 0:
        cap.release()
        raise RuntimeError("Could not read video dimensions.")

    writer = None
    if not args.no_save_video:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out_fps = source_fps if source_fps > 0 else 30.0
        writer = cv2.VideoWriter(str(output), fourcc, out_fps, (width, height))
        if not writer.isOpened():
            cap.release()
            raise RuntimeError(
                f"Could not create output video: {output}. "
                "Check OpenCV video codec support or use --no-save-video."
            )

    frame_rows: list[dict[str, Any]] = []
    inference_ms: list[float] = []
    end_to_end_ms: list[float] = []
    aggregate_classes: Counter[str] = Counter()
    measured_frames = 0
    warmup_frames_done = 0
    total_detections = 0
    start_all = time.perf_counter()
    measured_start: float | None = None
    source_frame = 0

    print(
        f"[INFO] Video: {width}x{height} @ "
        f"{source_fps:.3f} FPS; frame count metadata: {total_frames_hint or 'unknown'}"
    )
    print(f"[INFO] Warm-up frames: {args.warmup}")

    try:
        while True:
            if args.max_frames is not None and measured_frames >= args.max_frames:
                break

            frame_read_start = time.perf_counter()
            ok, frame = cap.read()
            if not ok:
                break
            source_frame += 1

            # Keep warm-up frames in output but exclude them from timed metrics.
            if warmup_frames_done < args.warmup:
                warmup_result = model.predict(
                    frame,
                    imgsz=args.imgsz,
                    conf=args.conf,
                    iou=args.iou,
                    device=device,
                    quantize=quantize,
                    classes=args.classes,
                    verbose=False,
                )[0]
                sync_device(device)
                if writer is not None:
                    writer.write(warmup_result.plot())
                warmup_frames_done += 1
                continue

            if measured_start is None:
                measured_start = frame_read_start

            sync_device(device)
            inference_start = time.perf_counter()
            result = model.predict(
                frame,
                imgsz=args.imgsz,
                conf=args.conf,
                iou=args.iou,
                device=device,
                quantize=quantize,
                classes=args.classes,
                verbose=False,
            )[0]
            sync_device(device)
            inference_duration_ms = (time.perf_counter() - inference_start) * 1000.0

            boxes = result.boxes
            class_counts: Counter[str] = Counter()
            if boxes is not None and len(boxes) > 0:
                cls_ids = boxes.cls.int().cpu().tolist()
                confidences = boxes.conf.float().cpu().tolist()
                xyxy = boxes.xyxy.int().cpu().tolist()
                for cls_id in cls_ids:
                    class_counts[class_name(names, int(cls_id))] += 1
                detection_count = len(cls_ids)
            else:
                cls_ids, confidences, xyxy = [], [], []
                detection_count = 0

            annotated = result.plot()
            measured_frames += 1
            total_detections += detection_count
            inference_ms.append(inference_duration_ms)
            aggregate_classes.update(class_counts)

            if writer is not None:
                writer.write(annotated)

            frame_elapsed_ms = (time.perf_counter() - frame_read_start) * 1000.0
            end_to_end_ms.append(frame_elapsed_ms)
            frame_rows.append(
                {
                    "frame": source_frame,
                    "inference_ms": round(inference_duration_ms, 4),
                    "frame_processing_ms": round(frame_elapsed_ms, 4),
                    "detections": detection_count,
                    "class_counts": json.dumps(dict(class_counts), sort_keys=True),
                    "class_ids": json.dumps(cls_ids),
                    "confidences": json.dumps([round(float(c), 5) for c in confidences]),
                    "xyxy": json.dumps(xyxy),
                }
            )

            if measured_frames == 1 or measured_frames % 30 == 0:
                avg_inf = statistics.fmean(inference_ms)
                print(
                    f"\r[RUN] {measured_frames} frames | "
                    f"avg inference {avg_inf:.2f} ms | "
                    f"inference FPS {1000.0 / avg_inf:.2f} | "
                    f"detections {total_detections}",
                    end="",
                    flush=True,
                )

            if args.show:
                cv2.imshow("YOLO Fixed Video Benchmark (press Q to stop)", annotated)
                if (cv2.waitKey(1) & 0xFF) in (ord("q"), 27):
                    break

    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if args.show:
            cv2.destroyAllWindows()

    total_elapsed = time.perf_counter() - start_all
    measured_elapsed = (
        time.perf_counter() - measured_start if measured_start is not None else 0.0
    )

    if measured_frames == 0:
        raise RuntimeError(
            "No frames were benchmarked. Check the video and --warmup setting."
        )

    average_inference = statistics.fmean(inference_ms)
    average_frame = statistics.fmean(end_to_end_ms)
    inference_fps = 1000.0 / average_inference if average_inference > 0 else 0.0
    processing_fps = (
        measured_frames / measured_elapsed if measured_elapsed > 0 else 0.0
    )
    average_detections = total_detections / measured_frames
    realtime_factor = processing_fps / source_fps if source_fps > 0 else None

    summary = {
        "source": str(source),
        "weights": str(weights),
        "device": device,
        "precision": "fp16" if use_fp16 else "fp32",
        "image_size": args.imgsz,
        "confidence_threshold": args.conf,
        "iou_threshold": args.iou,
        "class_filter": args.classes,
        "video": {
            "width": width,
            "height": height,
            "source_fps": source_fps,
            "metadata_frame_count": total_frames_hint,
        },
        "benchmark": {
            "warmup_frames_excluded": warmup_frames_done,
            "measured_frames": measured_frames,
            "total_detections": total_detections,
            "average_detections_per_frame": average_detections,
            "average_inference_ms": average_inference,
            "median_inference_ms": statistics.median(inference_ms),
            "p95_inference_ms": percentile(inference_ms, 0.95),
            "min_inference_ms": min(inference_ms),
            "max_inference_ms": max(inference_ms),
            "average_frame_processing_ms": average_frame,
            "inference_fps": inference_fps,
            "end_to_end_processing_fps": processing_fps,
            "realtime_factor": realtime_factor,
            "measured_wall_time_seconds": measured_elapsed,
            "total_script_time_seconds": total_elapsed,
            "detections_by_class": dict(aggregate_classes),
        },
        "outputs": {
            "annotated_video": None if args.no_save_video else str(output),
            "per_frame_csv": str(csv_path),
        },
        "notes": [
            "Inference latency measures the Ultralytics model.predict call, "
            "including preprocessing and postprocessing.",
            "CUDA is synchronized before and after each measured inference call.",
            "Warm-up frames are excluded from benchmark metrics.",
            "End-to-end processing FPS includes decoding, inference, annotation, "
            "and output writing; --show can reduce this throughput.",
            "Detection counts are not precision/recall or mAP. Ground-truth labels "
            "are required to calculate accuracy metrics.",
        ],
    }

    with csv_path.open("w", newline="", encoding="utf-8") as file:
        fieldnames = [
            "frame",
            "inference_ms",
            "frame_processing_ms",
            "detections",
            "class_counts",
            "class_ids",
            "confidences",
            "xyxy",
        ]
        csv_writer = csv.DictWriter(file, fieldnames=fieldnames)
        csv_writer.writeheader()
        csv_writer.writerows(frame_rows)

    with json_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, ensure_ascii=False)

    print("\n\n========== VIDEO BENCHMARK ==========")
    print(f"Model:                    {weights.name}")
    print(f"Device / precision:       {device} / {'FP16' if use_fp16 else 'FP32'}")
    print(f"Frames measured:          {measured_frames}")
    print(f"Total detections:         {total_detections}")
    print(f"Average inference:        {average_inference:.2f} ms")
    print(f"Median / P95 inference:   {statistics.median(inference_ms):.2f} / "
          f"{percentile(inference_ms, 0.95):.2f} ms")
    print(f"Inference throughput:    {inference_fps:.2f} FPS")
    print(f"End-to-end throughput:   {processing_fps:.2f} FPS")
    print(f"Average frame processing:{average_frame:8.2f} ms")
    if realtime_factor is not None:
        print(f"Real-time factor:         {realtime_factor:.2f}x source video FPS")
    print(f"Detections by class:     {dict(aggregate_classes)}")
    if not args.no_save_video:
        print(f"Annotated video:          {output}")
    print(f"Per-frame CSV:            {csv_path}")
    print(f"Summary JSON:              {json_path}")
    print("=====================================")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        raise SystemExit(1)
