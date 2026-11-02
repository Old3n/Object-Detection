#!/usr/bin/env python3
"""Create a YOLO pedestrian-attribute dataset from selected JAAD clips and XML.

Expected annotation folders come from the official JAAD_2.0 annotation repository:
  annotations/video_XXXX.xml
  annotations_attributes/video_XXXX_attributes.xml

JAAD's pedestrian attributes are only available for its behavior-annotated subset.
Other pedestrian boxes are labeled other_person. JAAD does not provide wheelchair
bounding-box labels, so person_with_wheelchair is deliberately left without samples.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import cv2


CLASS_NAMES = {
    0: "child",
    1: "woman",
    2: "elderly_person",
    3: "person_with_wheelchair",
    4: "other_person",
}
IMAGE_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv", ".m4v"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert hand-picked JAAD clips and JAAD_2.0 XML to YOLO labels."
    )
    parser.add_argument("--videos-dir", type=Path, default=Path("videos"))
    parser.add_argument(
        "--annotations-dir",
        type=Path,
        required=True,
        help="JAAD_2.0 annotations directory.",
    )
    parser.add_argument(
        "--attributes-dir",
        type=Path,
        required=True,
        help="JAAD_2.0 annotations_attributes directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("datasets/jaad_pedestrian"),
    )
    parser.add_argument(
        "--frame-stride",
        type=int,
        default=5,
        help="Save every Nth frame (default: 5; JAAD is recorded at 30 FPS).",
    )
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_attributes(path: Path) -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}
    if not path.is_file():
        return result
    root = ET.parse(path).getroot()
    for person in root.findall(".//pedestrian"):
        person_id = person.attrib.get("id")
        if person_id:
            result[person_id] = {
                key: str(value).strip().lower()
                for key, value in person.attrib.items()
            }
    return result


def class_for_person(attributes: dict[str, str] | None) -> int:
    """Map JAAD age/gender labels to mutually exclusive YOLO classes."""
    if not attributes:
        return 4

    age = attributes.get("age", "").strip().lower()
    gender = attributes.get("gender", "").strip().lower()
    if age == "child":
        return 0
    if age == "senior":
        return 2
    if gender in {"female", "woman"}:
        return 1
    return 4


def track_annotations(
    xml_path: Path,
    attributes: dict[str, dict[str, str]],
) -> tuple[dict[int, list[tuple[int, float, float, float, float]]], tuple[int, int] | None]:
    """Return frame -> (class, x1, y1, x2, y2) rows and annotation dimensions."""
    root = ET.parse(xml_path).getroot()
    by_frame: dict[int, list[tuple[int, float, float, float, float]]] = defaultdict(list)

    size_node = root.find(".//original_size")
    annotation_size = None
    if size_node is not None:
        try:
            annotation_size = (
                int(float(size_node.findtext("width", "0"))),
                int(float(size_node.findtext("height", "0"))),
            )
        except (TypeError, ValueError):
            annotation_size = None

    for track in root.findall(".//track"):
        label = (track.attrib.get("label") or "").strip().lower()
        if label not in {"pedestrian", "ped"}:
            # JAAD "people" labels represent groups, not single-person boxes.
            continue

        person_id = None
        first_box = track.find("./box")
        if first_box is not None:
            for attribute in first_box.findall("./attribute"):
                if attribute.attrib.get("name") == "id":
                    person_id = (attribute.text or "").strip()
                    break
        class_id = class_for_person(attributes.get(person_id or ""))

        for box in track.findall("./box"):
            try:
                frame_index = int(box.attrib["frame"])
                outside = box.attrib.get("outside", "0") == "1"
                x1 = float(box.attrib["xtl"])
                y1 = float(box.attrib["ytl"])
                x2 = float(box.attrib["xbr"])
                y2 = float(box.attrib["ybr"])
            except (KeyError, TypeError, ValueError):
                continue
            if outside:
                continue

            occlusion = ""
            for attribute in box.findall("./attribute"):
                if attribute.attrib.get("name") == "occlusion":
                    occlusion = (attribute.text or "").strip().lower()
                    break
            if occlusion == "full":
                continue
            if x2 <= x1 or y2 <= y1:
                continue
            by_frame[frame_index].append((class_id, x1, y1, x2, y2))

    return dict(by_frame), annotation_size


def write_split(
    video_path: Path,
    split: str,
    annotations_dir: Path,
    attributes_dir: Path,
    output_dir: Path,
    frame_stride: int,
) -> dict[str, Any]:
    annotation_path = annotations_dir / f"{video_path.stem}.xml"
    attributes_path = attributes_dir / f"{video_path.stem}_attributes.xml"
    if not annotation_path.is_file():
        raise FileNotFoundError(f"Missing JAAD annotation XML: {annotation_path}")

    attributes = load_attributes(attributes_path)
    frame_annotations, annotation_size = track_annotations(annotation_path, attributes)

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if width <= 0 or height <= 0:
        capture.release()
        raise RuntimeError(f"Invalid dimensions for video: {video_path}")

    if annotation_size and annotation_size[0] > 0 and annotation_size[1] > 0:
        scale_x = width / annotation_size[0]
        scale_y = height / annotation_size[1]
    else:
        scale_x = scale_y = 1.0

    image_dir = output_dir / "images" / split
    label_dir = output_dir / "labels" / split
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)

    frame_index = 0
    saved_frames = 0
    class_counts: Counter[str] = Counter()
    image_prefix = video_path.stem

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame_index % frame_stride == 0:
                image_name = f"{image_prefix}_{frame_index:06d}.jpg"
                image_path = image_dir / image_name
                label_path = label_dir / f"{image_prefix}_{frame_index:06d}.txt"
                if not cv2.imwrite(
                    str(image_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 94]
                ):
                    raise RuntimeError(f"Could not write image: {image_path}")

                yolo_rows: list[str] = []
                for class_id, x1, y1, x2, y2 in frame_annotations.get(frame_index, []):
                    x1 = min(max(x1 * scale_x, 0.0), width)
                    x2 = min(max(x2 * scale_x, 0.0), width)
                    y1 = min(max(y1 * scale_y, 0.0), height)
                    y2 = min(max(y2 * scale_y, 0.0), height)
                    box_width = x2 - x1
                    box_height = y2 - y1
                    if box_width < 2 or box_height < 2:
                        continue
                    x_center = ((x1 + x2) * 0.5) / width
                    y_center = ((y1 + y2) * 0.5) / height
                    norm_width = box_width / width
                    norm_height = box_height / height
                    yolo_rows.append(
                        f"{class_id} {x_center:.6f} {y_center:.6f} "
                        f"{norm_width:.6f} {norm_height:.6f}"
                    )
                    class_counts[CLASS_NAMES[class_id]] += 1

                label_path.write_text(
                    "\n".join(yolo_rows) + ("\n" if yolo_rows else ""),
                    encoding="utf-8",
                )
                saved_frames += 1
            frame_index += 1
    finally:
        capture.release()

    return {
        "video": video_path.name,
        "split": split,
        "video_frames": frame_index,
        "saved_frames": saved_frames,
        "annotated_person_boxes": sum(class_counts.values()),
        "class_counts": dict(class_counts),
        "attributes_file_present": attributes_path.is_file(),
    }


def write_dataset_yaml(output_dir: Path) -> Path:
    names_block = "\n".join(
        f"  {class_id}: {name}" for class_id, name in CLASS_NAMES.items()
    )
    content = (
        f'path: "{output_dir.resolve().as_posix()}"\n'
        "train: images/train\n"
        "val: images/val\n"
        "names:\n"
        f"{names_block}\n"
    )
    yaml_path = output_dir / "pedestrian_attributes.yaml"
    yaml_path.write_text(content, encoding="utf-8")
    return yaml_path


def main() -> int:
    args = parse_args()
    videos_dir = args.videos_dir.expanduser().resolve()
    annotations_dir = args.annotations_dir.expanduser().resolve()
    attributes_dir = args.attributes_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    if not videos_dir.is_dir():
        raise FileNotFoundError(f"Video directory not found: {videos_dir}")
    if not annotations_dir.is_dir():
        raise FileNotFoundError(f"Annotations directory not found: {annotations_dir}")
    if not attributes_dir.is_dir():
        raise FileNotFoundError(f"Attributes directory not found: {attributes_dir}")
    if args.frame_stride < 1:
        raise ValueError("--frame-stride must be at least 1")
    if not 0.0 < args.val_ratio < 1.0:
        raise ValueError("--val-ratio must be between 0 and 1")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Output directory is not empty: {output_dir}. "
            "Choose a new --output-dir to avoid overwriting data."
        )

    videos = sorted(
        path for path in videos_dir.iterdir()
        if path.is_file()
        and path.suffix.lower() in IMAGE_SUFFIXES
        and path.stem.startswith("video_")
    )
    if len(videos) < 2:
        raise ValueError(
            f"Found {len(videos)} selected video(s). At least two are required "
            "to make a video-level train/validation split."
        )

    missing_xml = [
        video.name
        for video in videos
        if not (annotations_dir / f"{video.stem}.xml").is_file()
    ]
    if missing_xml:
        raise FileNotFoundError(
            "Missing annotation XML for selected clips: " + ", ".join(missing_xml[:20])
        )

    shuffled = list(videos)
    random.Random(args.seed).shuffle(shuffled)
    val_count = max(1, min(len(videos) - 1, round(len(videos) * args.val_ratio)))
    validation_videos = {path.name for path in shuffled[:val_count]}
    output_dir.mkdir(parents=True, exist_ok=True)

    summaries = []
    totals: Counter[str] = Counter()
    for index, video in enumerate(videos, start=1):
        split = "val" if video.name in validation_videos else "train"
        print(f"[{index}/{len(videos)}] {video.name} -> {split}")
        result = write_split(
            video,
            split,
            annotations_dir,
            attributes_dir,
            output_dir,
            args.frame_stride,
        )
        summaries.append(result)
        totals.update(result["class_counts"])

    yaml_path = write_dataset_yaml(output_dir)
    summary = {
        "videos_dir": str(videos_dir),
        "annotations_dir": str(annotations_dir),
        "frame_stride": args.frame_stride,
        "train_videos": [
            item["video"] for item in summaries if item["split"] == "train"
        ],
        "val_videos": [
            item["video"] for item in summaries if item["split"] == "val"
        ],
        "class_counts": dict(totals),
        "classes_without_JAAD_box_annotations": ["person_with_wheelchair"],
        "traffic_light_note": (
            "JAAD traffic labels are frame-level scene annotations, not bounding boxes. "
            "Use the COCO pretrained traffic-light detector (class 9); do not turn "
            "frame-level presence flags into fabricated object boxes."
        ),
        "videos": summaries,
        "data_yaml": str(yaml_path),
    }
    (output_dir / "conversion_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print("\n========== JAAD CONVERSION ==========")
    print(f"Videos:       {len(videos)}")
    print(f"Train videos: {len(videos) - val_count}")
    print(f"Val videos:   {val_count}")
    print(f"Class boxes:  {dict(totals)}")
    print(f"Dataset YAML: {yaml_path}")
    print("=====================================")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, FileExistsError, RuntimeError, ValueError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        raise SystemExit(1)
