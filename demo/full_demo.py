"""Full visible demo of the Object-Detection repo on a Mac (CPU).

Uses the repo's own modules: YOLO detection + ByteTrack tracking, lane detection,
collision-risk estimation (TTC / corridor entry), Iranian road signs, night boost.

Scenes (switch with keys 1-3, they also run in order automatically):
  1 STREET    Iranian street photos: people, vehicles, traffic lights, lane, signs.
  2 NIGHT     The same photos darkened to 25%: left = camera, right = night boost.
  3 LIVE      Webcam with tracking. Walk toward the camera to raise the risk level;
              hold up a photo of a stop / speed sign on your phone.
Keys: N next photo, Space pause, D toggle simulated darkness (LIVE), Q / Esc quit.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--repo", default=str(HERE.parent), help="Repository root (default: parent of demo/)")
parser.add_argument("--sign-weights", required=True, help="ITSD checkpoint (.pt)")
parser.add_argument("--weights", default=None, help="COCO checkpoint (default: <repo>/yolo26n.pt)")
parser.add_argument("--images", default=str(HERE / "street"), help="Folder of .jpg street photos for scenes 1-2")
parser.add_argument("--camera", type=int, default=0)
parser.add_argument("--hold", type=float, default=2.5, help="Seconds per photo")
parser.add_argument("--start", type=int, choices=(1, 2, 3), default=1)
args = parser.parse_args()
sys.path.insert(0, args.repo)

import torch  # noqa: E402
from ultralytics import YOLO  # noqa: E402
from ModelBenchmark import RoadSignOverlay  # noqa: E402
from RiskVideoBenchmark import draw_report  # noqa: E402
from adas.lane_detection import LaneDetector  # noqa: E402
from adas.risk_assessment import CollisionRiskAssessor  # noqa: E402
from adas.road_signs import boost_night, is_night, sign_text  # noqa: E402

STREET_CLASSES = [0, 1, 2, 3, 5, 7, 9]  # person, bicycle, car, motorcycle, bus, truck, traffic light
LIVE_CORRIDOR = ((0.20, 0.25), (0.80, 0.25), (1.0, 1.0), (0.0, 1.0))  # webcam: person's feet are off-frame
WIN = "AutoPilot full demo  |  1 Street  2 Night  3 Live  |  N next  Space pause  D dark  Q quit"
TITLES = {1: "SCENE 1  STREET PHOTOS", 2: "SCENE 2  NIGHT BOOST", 3: "SCENE 3  LIVE CAMERA"}

coco = YOLO(args.weights or str(Path(args.repo) / "yolo26n.pt"))
DEVICE = 0 if torch.cuda.is_available() else "cpu"
signs = RoadSignOverlay(YOLO(args.sign_weights), DEVICE)
photos = sorted(Path(args.images).glob("*.jpg"))
if not photos and args.start != 3:
    print(f"[INFO] No .jpg photos in {args.images}; starting with the live camera.")
    args.start = 3


def fit(img, height=720, max_width=1600):
    scale = min(height / img.shape[0], max_width / img.shape[1])
    return cv2.resize(img, (round(img.shape[1] * scale), round(img.shape[0] * scale)))


def strip(width, scene, signs_line, extra):
    bar = np.full((96, width, 3), 18, np.uint8)
    cv2.putText(bar, TITLES[scene], (14, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 220, 255), 2)
    cv2.putText(bar, f"ROAD SIGNS: {signs_line}", (14, 66), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 140, 255), 2)
    cv2.putText(bar, extra, (14, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)
    return bar


def analyse_photo(img, lane_detector):
    """Detection + lane + risk + signs on one still photo. Returns the dashboard."""
    night = is_night(img)
    work = boost_night(img) if night else img.copy()
    res = coco.predict(work, conf=0.35, classes=STREET_CLASSES, device=DEVICE, verbose=False)[0]
    rows = [(*b, i + 1, c, k) for i, (b, c, k) in enumerate(zip(
        res.boxes.xyxy.cpu().tolist(), res.boxes.conf.cpu().tolist(), res.boxes.cls.int().cpu().tolist()))]
    lane = lane_detector.detect(work)
    corridor = lane.corridor if lane.status == "DETECTED" else None
    if corridor and any(not (0 <= x <= 1) for x, _ in corridor):
        corridor = None  # known repo issue: off-frame lane crashes the assessor
    report = CollisionRiskAssessor().update(rows, work.shape, timestamp=0, ego_corridor=corridor)
    confirmed = signs.evaluate_frame(work)
    out = draw_report(work, report, rows, res.names, stats={
        "frame": 1, "model": "yolo26n.pt + ITSD signs", "tracker": "none (photo)", "device": str(DEVICE),
        "raw_detections": len(rows), "lane_status": lane.status if corridor else "NOT_DETECTED",
        "lane_confidence": lane.confidence, "confidence": 0.35})
    people = sum(1 for r in rows if r[6] == 0)
    return out, confirmed, night, people, len(rows) - people


def photo_scene(scene):
    if not photos:
        return 3
    for n, path in enumerate(photos, 1):
        src = cv2.imread(str(path))
        if scene == 2:
            src = (src * 0.25).astype(np.uint8)
        signs.reset()
        t_end, paused = time.time() + args.hold, False
        while time.time() < t_end or paused:
            if not paused:
                out, confirmed, night, people, vehicles = analyse_photo(src, LaneDetector())
                if scene == 2:
                    raw = src.copy()
                    cv2.putText(raw, "CAMERA (no enhancement)", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (200, 200, 200), 3)
                    out = np.hstack([fit(raw), fit(out)])
                out = fit(out, 720, 1800)
                info = (f"photo {n}/{len(photos)}  |  people {people}  |  vehicles/lights {vehicles}  |  "
                        f"night boost {'ON' if night else 'off'}  |  sign needs 3 frames to confirm")
                cv2.imshow(WIN, np.vstack([out, strip(out.shape[1], scene, sign_text(confirmed), info)]))
            key = cv2.waitKey(30) & 0xFF
            if key in (ord("q"), 27):
                return "quit"
            if key in (ord("1"), ord("2"), ord("3")) and int(chr(key)) != scene:
                return int(chr(key))
            if key == ord(" "):
                paused = not paused
            if key == ord("n"):
                break
    return scene + 1


def live_scene():
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print("[WARN] Camera not available. Allow camera access for your terminal in System Settings.")
        return "quit"
    assessor, dark, start = CollisionRiskAssessor(), False, time.perf_counter()
    signs.reset()
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                return "quit"
            if dark:
                frame = (frame * 0.25).astype(np.uint8)
            night = is_night(frame)
            if night:
                frame = boost_night(frame)
            t0 = time.perf_counter()
            res = coco.track(frame, persist=True, tracker="bytetrack.yaml", conf=0.35, device=DEVICE,
                             classes=STREET_CLASSES + [67], verbose=False)[0]
            ms = (time.perf_counter() - t0) * 1000
            rows = []
            if res.boxes.id is not None:
                rows = [(*b, t, c, k) for b, t, c, k in zip(
                    res.boxes.xyxy.cpu().tolist(), res.boxes.id.int().cpu().tolist(),
                    res.boxes.conf.cpu().tolist(), res.boxes.cls.int().cpu().tolist())]
            report = assessor.update(rows, frame.shape, timestamp=time.perf_counter() - start,
                                     ego_corridor=LIVE_CORRIDOR)
            confirmed = signs.evaluate_frame(frame)
            out = draw_report(frame, report, rows, res.names, stats={
                "live": True, "source_label": "WEBCAM", "video_seconds": time.perf_counter() - start,
                "model": "yolo26n.pt + ITSD signs", "tracker": "ByteTrack", "device": str(DEVICE),
                "inference_ms": ms, "raw_detections": len(res.boxes), "lane_status": "DETECTED",
                "lane_confidence": 1.0, "confidence": 0.35})
            out = fit(out, 720, 1800)
            info = (f"walk toward the camera: CLEAR > CAUTION > HIGH  |  D = simulate darkness ({'ON' if dark else 'off'})"
                    f"  |  night boost {'ON' if night else 'off'}")
            cv2.imshow(WIN, np.vstack([out, strip(out.shape[1], 3, sign_text(confirmed), info)]))
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                return "quit"
            if key in (ord("1"), ord("2")):
                return int(chr(key))
            if key == ord("d"):
                dark = not dark
    finally:
        cap.release()


cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
scene = args.start
while scene != "quit":
    scene = live_scene() if scene == 3 else photo_scene(scene)
    if scene == 4:
        scene = 3
cv2.destroyAllWindows()
