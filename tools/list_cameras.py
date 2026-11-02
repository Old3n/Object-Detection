#!/usr/bin/env python3
"""Find camera indices that OpenCV can actually open on this computer."""

from __future__ import annotations

import argparse
import time

import cv2


BACKENDS = {
    "msmf": cv2.CAP_MSMF,
    "dshow": cv2.CAP_DSHOW,
    "auto": cv2.CAP_ANY,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Probe USB and virtual webcams and print working OpenCV indices."
    )
    parser.add_argument("--max-index", type=int, default=9)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument(
        "--backends",
        nargs="+",
        choices=tuple(BACKENDS),
        default=["msmf", "dshow", "auto"],
        help="Backends to try for every camera index.",
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Show a preview for each working camera; press Q for next camera.",
    )
    return parser.parse_args()


def open_and_read(index: int, backend_name: str, width: int, height: int):
    backend = BACKENDS[backend_name]
    cap = None
    try:
        cap = (
            cv2.VideoCapture(index)
            if backend_name == "auto"
            else cv2.VideoCapture(index, backend)
        )
        if not cap.isOpened():
            return None

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        frame = None
        for _ in range(5):
            ok, candidate = cap.read()
            if ok and candidate is not None:
                frame = candidate
                break
            time.sleep(0.1)

        if frame is None:
            cap.release()
            return None

        return cap, frame
    except (cv2.error, RuntimeError, TypeError):
        if cap is not None:
            cap.release()
        return None


def main() -> int:
    args = parse_args()
    if args.max_index < 0 or args.width < 1 or args.height < 1:
        raise ValueError("--max-index must be >=0 and dimensions must be positive.")

    found = 0
    print("Probing camera indices. Close Teams, Zoom, Discord, and Camera first.")
    print(f"Trying indices 0..{args.max_index} with {', '.join(args.backends)}")
    print()

    for index in range(args.max_index + 1):
        index_found = False
        for backend_name in args.backends:
            result = open_and_read(index, backend_name, args.width, args.height)
            if result is None:
                continue

            cap, frame = result
            height, width = frame.shape[:2]
            fps = cap.get(cv2.CAP_PROP_FPS)
            print(
                f"[FOUND] --source {index} --camera-backend {backend_name} "
                f"| {width}x{height} | reported FPS: {fps:.2f}"
            )
            found += 1
            index_found = True

            if args.preview:
                window = f"Camera {index} ({backend_name}) - Q for next"
                cv2.namedWindow(window, cv2.WINDOW_NORMAL)
                cv2.imshow(window, frame)
                while True:
                    key = cv2.waitKey(30) & 0xFF
                    if key in (ord("q"), 27):
                        break
                    ok, frame = cap.read()
                    if ok:
                        cv2.imshow(window, frame)
                cv2.destroyWindow(window)

            cap.release()
            break

        if not index_found:
            print(f"[----] index {index}: no readable camera")

    print()
    if found:
        print(f"Found {found} readable camera index(es). Use the exact index above.")
    else:
        print(
            "No readable camera found. Check Windows camera permissions, "
            "Device Manager, camera privacy shutter, and whether another app "
            "is holding the camera."
        )
        return 1
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError, ValueError) as exc:
        raise SystemExit(f"[ERROR] {exc}")
