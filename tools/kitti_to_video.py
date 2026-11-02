#!/usr/bin/env python3
"""Convert a KITTI odometry image sequence into an MP4 for video evaluation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a KITTI image_2 or image_3 folder into an MP4."
    )
    parser.add_argument(
        "--frames",
        required=True,
        type=Path,
        help="KITTI sequence camera folder, e.g. dataset/sequences/00/image_2.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("videos/kitti_sequence.mp4"),
        help="Output MP4 path (default: videos/kitti_sequence.mp4).",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=10.0,
        help="Playback FPS. KITTI odometry camera sequences are 10 Hz.",
    )
    parser.add_argument(
        "--start-frame",
        type=int,
        default=0,
        help="Zero-based index into the sorted image list.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional frame limit, useful for creating a shorter sample.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    frames_dir = args.frames.expanduser().resolve()
    output = args.output.expanduser().resolve()

    if not frames_dir.is_dir():
        raise FileNotFoundError(f"Frame directory does not exist: {frames_dir}")
    if args.fps <= 0:
        raise ValueError("--fps must be greater than zero.")
    if args.start_frame < 0:
        raise ValueError("--start-frame cannot be negative.")
    if args.max_frames is not None and args.max_frames < 1:
        raise ValueError("--max-frames must be positive.")

    images = sorted(
        path for path in frames_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    if not images:
        raise FileNotFoundError(f"No PNG/JPG images found in: {frames_dir}")
    if args.start_frame >= len(images):
        raise ValueError(
            f"--start-frame {args.start_frame} is outside the {len(images)} images."
        )

    selected = images[args.start_frame:]
    if args.max_frames is not None:
        selected = selected[: args.max_frames]

    first = cv2.imread(str(selected[0]), cv2.IMREAD_COLOR)
    if first is None:
        raise RuntimeError(f"Could not decode image: {selected[0]}")
    height, width = first.shape[:2]

    output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output),
        cv2.VideoWriter_fourcc(*"mp4v"),
        args.fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(
            f"Could not create MP4: {output}. Check OpenCV codec support."
        )

    written = 0
    try:
        for index, image_path in enumerate(selected):
            frame = first if index == 0 else cv2.imread(
                str(image_path), cv2.IMREAD_COLOR
            )
            if frame is None:
                raise RuntimeError(f"Could not decode image: {image_path}")
            if frame.shape[1] != width or frame.shape[0] != height:
                raise ValueError(
                    f"Image dimensions changed at {image_path}: "
                    f"expected {width}x{height}, got "
                    f"{frame.shape[1]}x{frame.shape[0]}"
                )
            writer.write(frame)
            written += 1
            if written == 1 or written % 250 == 0 or written == len(selected):
                print(
                    f"\r[WRITE] {written:,}/{len(selected):,} frames",
                    end="",
                    flush=True,
                )
    finally:
        writer.release()

    duration = written / args.fps
    print("\n\n========== KITTI VIDEO ==========")
    print(f"Input frames:  {frames_dir}")
    print(f"Output video:  {output}")
    print(f"Frame count:   {written:,}")
    print(f"Resolution:    {width}x{height}")
    print(f"FPS:           {args.fps:.2f}")
    print(f"Duration:      {int(duration // 60)}m {duration % 60:.1f}s")
    print("=================================")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        raise SystemExit(1)
