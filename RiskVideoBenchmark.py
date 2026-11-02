#!/usr/bin/env python3
"""Run detection, tracking, and advisory collision-risk estimation on a video."""

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

from adas.risk_assessment import (
    CAUTION,
    CLEAR,
    CRITICAL,
    HIGH,
    CollisionRiskAssessor,
)


RISK_RANK = {CLEAR: 0, CAUTION: 1, HIGH: 2, CRITICAL: 3}
RISK_COLORS = {
    CLEAR: (0, 150, 0),
    CAUTION: (0, 165, 255),
    HIGH: (0, 100, 255),
    CRITICAL: (0, 0, 255),
}


def class_name(names: Any, class_id: int) -> str:
    if isinstance(names, dict):
        return str(names.get(class_id, names.get(str(class_id), class_id)))
    if isinstance(names, (list, tuple)) and 0 <= class_id < len(names):
        return str(names[class_id])
    return str(class_id)


def sync_cuda(device: str) -> None:
    if torch.cuda.is_available() and (
        device.startswith("cuda") or device.split(",", 1)[0].isdigit()
    ):
        index = device.split(":", 1)[1] if device.startswith("cuda:") else device
        index = index.split(",", 1)[0]
        torch.cuda.synchronize(int(index)) if index.isdigit() else torch.cuda.synchronize()


def _blend_panel(frame: Any, x1: int, y1: int, x2: int, y2: int, alpha: float = 0.82) -> None:
    """Draw a dark translucent panel without hiding all underlying footage."""
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (12, 16, 22), -1)
    cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0, frame)


def draw_report(
    frame: Any,
    report: Any,
    tracked_objects: list[tuple[float, float, float, float, int, float, int]],
    names: Any,
    *,
    stats: dict[str, Any] | None = None,
    max_display_objects: int = 8,
) -> Any:
    """Render detections plus a readable risk, tracker, and performance dashboard."""
    stats = stats or {}
    height, width = frame.shape[:2]
    ui_scale = float(stats.get("ui_scale", 1.15))
    scale = max(0.75, min(2.0, min(width / 1280.0, height / 720.0) * ui_scale))
    font = cv2.FONT_HERSHEY_SIMPLEX
    line = max(24, round(32 * scale))
    margin = max(10, round(16 * scale))
    thickness = max(1, round(2 * scale))

    polygon = [
        (round(x * width), round(y * height))
        for x, y in report.corridor
    ]
    import numpy as np

    polygon_array = np.asarray(polygon, dtype="int32")
    overlay = frame.copy()
    cv2.fillPoly(overlay, [polygon_array], (255, 255, 0))
    cv2.addWeighted(overlay, 0.055, frame, 0.945, 0, frame)
    cv2.polylines(
        frame, [polygon_array], isClosed=True, color=(0, 255, 255),
        thickness=max(2, round(2 * scale)),
    )

    risk_by_id = {item.track_id: item for item in report.objects}
    for x1, y1, x2, y2, track_id, confidence, class_id in tracked_objects:
        risk = risk_by_id.get(track_id)
        color = RISK_COLORS[risk.level] if risk else (190, 190, 190)
        box = (int(x1), int(y1), int(x2), int(y2))
        cv2.rectangle(
            frame, box[:2], box[2:], color, max(2, round(2 * scale))
        )
        label = f"ID {track_id} | {class_name(names, class_id)} | {confidence:.2f}"
        if risk and risk.ttc_seconds is not None:
            label += f" | TTC~{risk.ttc_seconds:.1f}s"
        elif risk and risk.cut_in_eta_seconds is not None:
            label += f" | ENTRY~{risk.cut_in_eta_seconds:.1f}s"
        text_y = max(round(22 * scale), box[1] - round(8 * scale))
        cv2.putText(
            frame, label, (box[0], text_y), font, 0.62 * scale,
            (255, 255, 255), thickness, cv2.LINE_AA,
        )

    header_h = round(145 * scale)
    footer_h = round(82 * scale)
    panel_w = min(round(410 * scale), round(width * 0.43))
    panel_x = width - panel_w
    footer_y = height - footer_h

    _blend_panel(frame, 0, 0, width, min(height - 1, header_h))
    _blend_panel(
        frame, panel_x, header_h + round(8 * scale),
        width - 1, max(header_h + round(9 * scale), footer_y - round(8 * scale)),
        0.88,
    )
    _blend_panel(frame, 0, footer_y, width, height - 1, 0.88)

    risk_color = RISK_COLORS[report.level]
    frame_index = int(stats.get("frame", 0))
    total_frames = int(stats.get("total_frames", 0))
    timestamp = float(stats.get("video_seconds", 0.0))
    duration = float(stats.get("duration_seconds", 0.0))
    inference_ms = float(stats.get("inference_ms", 0.0))
    loop_ms = float(stats.get("loop_ms", 0.0))
    fps_now = float(stats.get("fps", 0.0))
    model_name = str(stats.get("model", "YOLO"))
    tracker_name = str(stats.get("tracker", "ByteTrack"))
    device = str(stats.get("device", "auto"))
    precision = str(stats.get("precision", "FP32"))
    conf = float(stats.get("confidence", 0.25))
    raw_detection_count = int(stats.get("raw_detections", len(tracked_objects)))

    title_y = round(30 * scale)
    detail_y = round(60 * scale)
    model_y = round(91 * scale)
    perf_y = round(122 * scale)
    title = f"ADAS PERCEPTION  |  COLLISION RISK: {report.level}"
    cv2.putText(
        frame, title, (margin, title_y), font, 0.90 * scale,
        risk_color, thickness + 1, cv2.LINE_AA,
    )
    position = (
        f"FRAME {frame_index:,}/{total_frames:,}  |  "
        f"VIDEO {timestamp:05.1f}s / {duration:05.1f}s"
        if total_frames > 0
        else f"FRAME {frame_index:,}  |  VIDEO {timestamp:05.1f}s"
    )
    cv2.putText(
        frame, position, (margin, detail_y), font, 0.60 * scale,
        (245, 245, 245), thickness, cv2.LINE_AA,
    )
    model_line = f"MODEL {model_name}  |  TRACKER {tracker_name}  |  DEVICE {device}  |  {precision}"
    cv2.putText(
        frame, model_line[:130], (margin, model_y), font, 0.52 * scale,
        (210, 225, 240), max(1, thickness - 1), cv2.LINE_AA,
    )
    perf = (
        f"DETECTIONS {raw_detection_count}  |  TRACKED {len(tracked_objects)}  |  "
        f"INFERENCE {inference_ms:.1f} ms  |  LOOP {loop_ms:.1f} ms  |  "
        f"{fps_now:.1f} FPS  |  CONF {conf:.2f}"
    )
    cv2.putText(
        frame, perf[:150], (margin, perf_y), font, 0.52 * scale,
        (235, 235, 235), max(1, thickness - 1), cv2.LINE_AA,
    )

    panel_y = header_h + round(8 * scale)
    panel_text_x = panel_x + margin
    panel_inner_w = panel_w - 2 * margin
    current_counts: Counter[str] = Counter(
        stats.get("class_counts") or {
            class_name(names, item[6]): sum(1 for obj in tracked_objects if obj[6] == item[6])
            for item in tracked_objects
        }
    )
    object_risks = sorted(
        report.objects,
        key=lambda item: (
            -RISK_RANK[item.level],
            -(item.ttc_seconds is not None and 1.0 / max(item.ttc_seconds, 1e-3) or 0.0),
            -item.confidence,
        ),
    )
    cv2.putText(
        frame, f"TRACKS {len(tracked_objects)}  |  DETECTIONS {raw_detection_count}",
        (panel_text_x, panel_y + line), font, 0.62 * scale,
        (255, 255, 255), thickness, cv2.LINE_AA,
    )

    y = panel_y + line * 2
    row_limit = max(1, min(max_display_objects, max(1, (footer_y - y - line) // (line * 2))))
    if not tracked_objects and raw_detection_count > 0:
        cv2.putText(
            frame, f"{raw_detection_count} DETECTIONS - WAITING FOR TRACK IDS",
            (panel_text_x, y), font, 0.60 * scale,
            (180, 210, 255), thickness, cv2.LINE_AA,
        )
        cv2.putText(
            frame, "Detector found boxes; tracker has not assigned IDs.",
            (panel_text_x, y + line), font, 0.48 * scale,
            (225, 225, 225), max(1, thickness - 1), cv2.LINE_AA,
        )
        cv2.putText(
            frame, "Risk estimation starts after track association.",
            (panel_text_x, y + line * 2), font, 0.43 * scale,
            (190, 200, 210), max(1, thickness - 1), cv2.LINE_AA,
        )
    elif not tracked_objects:
        cv2.putText(
            frame, "NO OBJECTS DETECTED",
            (panel_text_x, y), font, 0.60 * scale,
            (180, 210, 255), thickness, cv2.LINE_AA,
        )
        cv2.putText(
            frame, "Check camera framing and class filters.",
            (panel_text_x, y + line), font, 0.46 * scale,
            (225, 225, 225), max(1, thickness - 1), cv2.LINE_AA,
        )
    else:
        for item in object_risks[:row_limit]:
            item_color = RISK_COLORS[item.level]
            name = class_name(names, item.class_id).upper()
            summary = f"#{item.track_id} {name}  conf {item.confidence:.2f}"
            if len(summary) > 35:
                summary = summary[:32] + "..."
            cv2.putText(
                frame, summary, (panel_text_x, y), font, 0.49 * scale,
                (245, 245, 245), max(1, thickness - 1), cv2.LINE_AA,
            )
            if item.ttc_seconds is not None:
                extra = f"{item.level}  |  TTC ~{item.ttc_seconds:.1f}s"
            elif item.cut_in_eta_seconds is not None:
                extra = f"{item.level}  |  ENTER ~{item.cut_in_eta_seconds:.1f}s"
            else:
                extra = f"{item.level}  |  ROI {'IN' if item.in_ego_corridor else 'OUT'}"
            cv2.putText(
                frame, extra, (panel_text_x, y + round(line * 0.78)),
                font, 0.46 * scale, item_color, max(1, thickness - 1), cv2.LINE_AA,
            )
            y += line * 2

        remaining = len(object_risks) - row_limit
        if remaining > 0 and y < footer_y - line:
            cv2.putText(
                frame, f"+ {remaining} more tracked objects (see CSV)",
                (panel_text_x, y), font, 0.43 * scale,
                (185, 195, 205), max(1, thickness - 1), cv2.LINE_AA,
            )

    counts_text = "  |  ".join(
        f"{name}: {count}" for name, count in sorted(current_counts.items())
    ) or "Class counts: none"
    count_y = footer_y + round(27 * scale)
    legend_y = footer_y + round(58 * scale)
    cv2.putText(
        frame, counts_text[:150], (margin, count_y), font, 0.50 * scale,
        (245, 245, 245), max(1, thickness - 1), cv2.LINE_AA,
    )
    legend = "RISK GUIDE: CLEAR  |  CAUTION 3-5s  |  HIGH 1.5-3s  |  CRITICAL <=1.5s"
    cv2.putText(
        frame, legend, (margin, legend_y), font, 0.43 * scale,
        (190, 205, 220), max(1, thickness - 1), cv2.LINE_AA,
    )
    return frame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark YOLO tracking and collision-risk assessment on a video."
    )
    parser.add_argument("--source", required=True, type=Path, help="Input video.")
    parser.add_argument(
        "--weights",
        required=True,
        help="Local checkpoint path or Ultralytics model name such as yolo26n.pt.",
    )
    parser.add_argument(
        "--tracker",
        default="bytetrack.yaml",
        help="Ultralytics tracker YAML, e.g. bytetrack.yaml or botsort.yaml.",
    )
    parser.add_argument("--device", default="auto", help="auto, cpu, 0, cuda:0, etc.")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--classes", type=int, nargs="+", default=None)
    parser.add_argument(
        "--pedestrian-class-ids",
        type=int,
        nargs="+",
        default=[0],
        help="Class IDs treated as pedestrians by the risk estimator.",
    )
    parser.add_argument(
        "--vehicle-class-ids",
        type=int,
        nargs="+",
        default=[1, 2, 3, 5, 7],
        help="Class IDs treated as vehicles by the risk estimator.",
    )
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--csv", dest="csv_path", type=Path, default=None)
    parser.add_argument("--json", dest="json_path", type=Path, default=None)
    parser.add_argument("--no-save-video", action="store_true")
    parser.add_argument("--show", action="store_true")
    parser.add_argument(
        "--display-width", type=int, default=1280,
        help="Maximum preview-window width in pixels (default: 1280).",
    )
    parser.add_argument(
        "--display-height", type=int, default=720,
        help="Maximum preview-window height in pixels (default: 720).",
    )
    parser.add_argument(
        "--quantize", choices=("fp32", "fp16"), default="fp32",
        help="Inference precision: fp32 (default) or fp16 (CUDA only).",
    )
    parser.add_argument("--max-display-objects", type=int, default=8)
    parser.add_argument(
        "--ui-scale", type=float, default=1.15,
        help="Dashboard text/panel scale multiplier (default: 1.15).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source = args.source.expanduser().resolve()
    weights_path = Path(args.weights).expanduser()
    weights_ref = str(weights_path.resolve()) if weights_path.is_file() else args.weights
    if not source.is_file():
        raise FileNotFoundError(f"Input video not found: {source}")
    if args.imgsz < 32 or args.max_frames is not None and args.max_frames < 1:
        raise ValueError("--imgsz must be >=32 and --max-frames must be positive.")
    if args.display_width < 1 or args.display_height < 1:
        raise ValueError("--display-width and --display-height must be positive.")
    if args.max_display_objects < 1:
        raise ValueError("--max-display-objects must be positive.")
    if not 0.6 <= args.ui_scale <= 2.0:
        raise ValueError("--ui-scale must be between 0.6 and 2.0.")
    if not 0.0 < args.conf <= 1.0 or not 0.0 < args.iou <= 1.0:
        raise ValueError("--conf and --iou must be in (0,1].")

    device = args.device
    if device.lower() == "auto":
        device = "0" if torch.cuda.is_available() else "cpu"
    wants_cuda = device.startswith("cuda") or device.split(",", 1)[0].isdigit()
    if wants_cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but PyTorch cannot access a CUDA GPU.")
    use_fp16 = args.quantize == "fp16" and wants_cuda
    quantize = 16 if use_fp16 else 32
    if args.quantize == "fp16" and not wants_cuda:
        print("[WARN] FP16 requested without CUDA; using FP32.")

    output = (
        args.output.expanduser().resolve()
        if args.output
        else source.with_name(f"{source.stem}_risk.mp4")
    )
    csv_path = (
        args.csv_path.expanduser().resolve()
        if args.csv_path
        else source.with_name(f"{source.stem}_risk.csv")
    )
    json_path = (
        args.json_path.expanduser().resolve()
        if args.json_path
        else source.with_name(f"{source.stem}_risk.json")
    )
    if output == source:
        raise ValueError("Output video cannot overwrite the input video.")
    for path in (output, csv_path, json_path):
        path.parent.mkdir(parents=True, exist_ok=True)

    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise RuntimeError(f"OpenCV could not open video: {source}")
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration_seconds = total_frames / fps if fps > 0 and total_frames > 0 else 0.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if width <= 0 or height <= 0:
        capture.release()
        raise RuntimeError("Could not read video dimensions.")

    writer = None
    if not args.no_save_video:
        writer = cv2.VideoWriter(
            str(output),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps if fps > 0 else 30.0,
            (width, height),
        )
        if not writer.isOpened():
            capture.release()
            raise RuntimeError(f"Could not create output video: {output}")

    print("[INFO] Risk dashboard: v2 (HUD rendered after preview resize)")
    print(f"[INFO] Loading model: {weights_ref}")
    model = YOLO(weights_ref)
    assessor = CollisionRiskAssessor(
        pedestrian_class_ids=args.pedestrian_class_ids,
        vehicle_class_ids=args.vehicle_class_ids,
    )
    names = model.names
    frame_rows: list[dict[str, Any]] = []
    inference_times: list[float] = []
    processing_times: list[float] = []
    risk_frames: Counter[str] = Counter()
    frame_count = 0
    detection_total = 0
    raw_detection_total = 0
    max_risk = CLEAR
    start = time.perf_counter()

    print(f"[INFO] Video: {width}x{height} @ {fps:.3f} FPS")
    print(f"[INFO] Model: {weights_ref}")
    print(f"[INFO] Device: {device}, precision: {'FP16' if use_fp16 else 'FP32'}")
    print(f"[INFO] Tracker: {args.tracker}")

    window_title = "YOLO tracking + collision-risk assessment (Q/Esc to stop)"
    if args.show:
        cv2.namedWindow(window_title, cv2.WINDOW_AUTOSIZE)

    try:
        while args.max_frames is None or frame_count < args.max_frames:
            frame_start = time.perf_counter()
            ok, frame = capture.read()
            if not ok:
                break

            sync_cuda(device)
            inference_start = time.perf_counter()
            result = model.track(
                frame,
                persist=True,
                tracker=args.tracker,
                imgsz=args.imgsz,
                conf=args.conf,
                iou=args.iou,
                device=device,
                quantize=quantize,
                classes=args.classes,
                verbose=False,
            )[0]
            sync_cuda(device)
            inference_ms = (time.perf_counter() - inference_start) * 1000.0

            tracked_objects = []
            raw_class_counts: Counter[str] = Counter()
            raw_detection_count = 0
            boxes = result.boxes
            if boxes is not None and len(boxes) > 0:
                raw_class_ids = boxes.cls.int().cpu().tolist()
                raw_detection_count = len(raw_class_ids)
                raw_class_counts.update(
                    class_name(names, int(class_id)) for class_id in raw_class_ids
                )
                if boxes.id is not None:
                    coordinates = boxes.xyxy.cpu().tolist()
                    track_ids = boxes.id.int().cpu().tolist()
                    confidences = boxes.conf.float().cpu().tolist()
                    tracked_class_ids = boxes.cls.int().cpu().tolist()
                    tracked_objects = [
                        (
                            box[0], box[1], box[2], box[3],
                            int(track_id), float(confidence), int(class_id),
                        )
                        for box, track_id, confidence, class_id in zip(
                            coordinates, track_ids, confidences, tracked_class_ids
                        )
                    ]

            video_timestamp = frame_count / fps if fps > 0 else time.monotonic()
            report = assessor.update(
                tracked_objects, frame.shape, timestamp=video_timestamp
            )
            frame_count += 1
            loop_ms = (time.perf_counter() - frame_start) * 1000.0
            elapsed_so_far = max(time.perf_counter() - start, 1e-6)
            current_fps = frame_count / elapsed_so_far
            display_stats = {
                "frame": frame_count,
                "total_frames": total_frames,
                "video_seconds": video_timestamp,
                "duration_seconds": duration_seconds,
                "inference_ms": inference_ms,
                "loop_ms": loop_ms,
                "fps": current_fps,
                "model": Path(weights_ref).name if Path(weights_ref).is_file() else weights_ref,
                "tracker": args.tracker,
                "device": device,
                "precision": "FP16" if use_fp16 else "FP32",
                "confidence": args.conf,
                "raw_detections": raw_detection_count,
                "class_counts": dict(raw_class_counts),
                "ui_scale": args.ui_scale,
            }
            annotated = draw_report(
                frame.copy(), report, tracked_objects, names,
                stats=display_stats,
                max_display_objects=args.max_display_objects,
            )
            detection_total += len(tracked_objects)
            raw_detection_total += raw_detection_count
            inference_times.append(inference_ms)
            risk_frames.update([report.level])
            if RISK_RANK[report.level] > RISK_RANK[max_risk]:
                max_risk = report.level

            if writer is not None:
                writer.write(annotated)
            if args.show:
                # Resize the clean frame first, then draw the dashboard at preview
                # resolution. Otherwise the dashboard text gets shrunk along with
                # a high-resolution source video and becomes unreadable.
                frame_height, frame_width = frame.shape[:2]
                scale = min(
                    1.0,
                    args.display_width / frame_width,
                    args.display_height / frame_height,
                )
                preview_width = max(1, round(frame_width * scale))
                preview_height = max(1, round(frame_height * scale))
                preview = cv2.resize(
                    frame,
                    (preview_width, preview_height),
                    interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR,
                )
                sx = preview_width / frame_width
                sy = preview_height / frame_height
                preview_objects = [
                    (
                        x1 * sx, y1 * sy, x2 * sx, y2 * sy,
                        track_id, confidence, class_id,
                    )
                    for x1, y1, x2, y2, track_id, confidence, class_id
                    in tracked_objects
                ]
                preview_stats = dict(display_stats)
                preview_stats["ui_scale"] = args.ui_scale
                preview = draw_report(
                    preview, report, preview_objects, names,
                    stats=preview_stats,
                    max_display_objects=args.max_display_objects,
                )
                cv2.imshow(window_title, preview)
                if (cv2.waitKey(1) & 0xFF) in (ord("q"), 27):
                    break

            processing_ms = (time.perf_counter() - frame_start) * 1000.0
            processing_times.append(processing_ms)
            object_rows = [
                {
                    "track_id": item.track_id,
                    "class_id": item.class_id,
                    "class_name": class_name(names, item.class_id),
                    "object_type": item.object_type,
                    "risk": item.level,
                    "ttc_seconds": item.ttc_seconds,
                    "cut_in_eta_seconds": item.cut_in_eta_seconds,
                    "in_ego_corridor": item.in_ego_corridor,
                }
                for item in report.objects
            ]
            frame_rows.append(
                {
                    "frame": frame_count,
                    "timestamp_seconds": frame_count / fps if fps > 0 else None,
                    "inference_ms": round(inference_ms, 4),
                    "processing_ms": round(processing_ms, 4),
                    "raw_detections": raw_detection_count,
                    "tracked_objects": len(tracked_objects),
                    "class_counts": json.dumps(dict(raw_class_counts), ensure_ascii=False),
                    "risk_level": report.level,
                    "risk_reason": report.reason,
                    "objects": json.dumps(object_rows, ensure_ascii=False),
                }
            )
            if frame_count == 1 or frame_count % 30 == 0:
                elapsed = time.perf_counter() - start
                print(
                    f"\r[RUN] {frame_count} frames | "
                    f"risk={report.level} | "
                    f"avg inference={statistics.fmean(inference_times):.1f} ms | "
                    f"elapsed={elapsed:.1f}s",
                    end="",
                    flush=True,
                )
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        if args.show:
            cv2.destroyAllWindows()

    if frame_count == 0:
        raise RuntimeError("No frames were processed.")
    elapsed = time.perf_counter() - start
    summary = {
        "source": str(source),
        "weights": weights_ref,
        "tracker": args.tracker,
        "device": device,
        "precision": "fp16" if use_fp16 else "fp32",
        "video": {
            "width": width, "height": height, "fps": fps,
            "metadata_frames": total_frames,
            "metadata_duration_seconds": duration_seconds,
        },
        "frames": frame_count,
        "raw_detections": raw_detection_total,
        "tracked_detections": detection_total,
        "average_raw_detections_per_frame": raw_detection_total / frame_count,
        "average_tracked_objects_per_frame": detection_total / frame_count,
        "average_inference_ms": statistics.fmean(inference_times),
        "median_inference_ms": statistics.median(inference_times),
        "p95_inference_ms": sorted(inference_times)[
            min(len(inference_times) - 1, math.ceil(0.95 * len(inference_times)) - 1)
        ],
        "average_processing_ms": statistics.fmean(processing_times),
        "end_to_end_fps": frame_count / elapsed if elapsed > 0 else 0.0,
        "risk_frames": dict(risk_frames),
        "maximum_risk_observed": max_risk,
        "outputs": {
            "annotated_video": None if args.no_save_video else str(output),
            "per_frame_csv": str(csv_path),
            "summary_json": str(json_path),
        },
        "limitations": [
            "TTC is estimated from monocular apparent bounding-box expansion.",
            "The ego corridor is an image-space ROI, not an automatically detected lane.",
            "Risk levels are research annotations, not validated braking decisions.",
        ],
    }

    with csv_path.open("w", newline="", encoding="utf-8") as file:
        fields = [
            "frame", "timestamp_seconds", "inference_ms", "processing_ms",
            "raw_detections", "tracked_objects", "class_counts",
            "risk_level", "risk_reason", "objects",
        ]
        writer_csv = csv.DictWriter(file, fieldnames=fields)
        writer_csv.writeheader()
        writer_csv.writerows(frame_rows)
    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n\n========== VIDEO RISK EVALUATION ==========")
    print(f"Frames:                {frame_count}")
    print(f"Raw detections:        {raw_detection_total}")
    print(f"Tracked objects:       {detection_total}")
    print(f"Average inference:     {summary['average_inference_ms']:.2f} ms")
    print(f"Median / P95 inference:{summary['median_inference_ms']:.2f} / {summary['p95_inference_ms']:.2f} ms")
    print(f"End-to-end FPS:        {summary['end_to_end_fps']:.2f}")
    print(f"Risk frames:           {dict(risk_frames)}")
    print(f"Maximum observed risk:{max_risk}")
    print(f"Annotated video:       {'disabled' if args.no_save_video else output}")
    print(f"CSV:                   {csv_path}")
    print(f"JSON:                  {json_path}")
    print("===========================================")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        raise SystemExit(1)
