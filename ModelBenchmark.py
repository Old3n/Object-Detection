import argparse
import time
from collections import defaultdict, deque
import cv2
import numpy as np
import torch
import yaml
from ultralytics import YOLO

from adas.risk_assessment import CLEAR, CAUTION, HIGH, CRITICAL, CollisionRiskAssessor

try:
    from boxmot import OCSORT
    HAS_BOXMOT = True
except ImportError:
    HAS_BOXMOT = False

WEBCAM_DESK_CLASSES = [0, 39, 41, 67]
IRAN_STREET_CLASSES = [0, 1, 2, 3, 5, 7]
ROAD_AND_CROSSROAD_CLASSES = [0, 1, 2, 3, 5, 7, 9, 11]
PEDESTRIAN_ATTRIBUTE_CLASSES = [0, 1, 2, 3, 4]  # child, woman, elderly, wheelchair user, other person
EXPECTED_PEDESTRIAN_NAMES = {
    0: "child",
    1: "woman",
    2: "elderly_person",
    3: "person_with_wheelchair",
    4: "other_person",
}

def write_tracker_yamls():
    bytetrack_cfg = {
        "tracker_type": "bytetrack",
        "track_high_thresh": 0.45,
        "track_low_thresh": 0.10,
        "new_track_thresh": 0.55,
        "track_buffer": 60,
        "match_thresh": 0.85,
        "fuse_score": True,
    }
    botsort_cfg = {
        "tracker_type": "botsort",
        "track_high_thresh": 0.45,
        "track_low_thresh": 0.10,
        "new_track_thresh": 0.55,
        "track_buffer": 60,
        "match_thresh": 0.85,
        "gmc_method": "orb",
        "proximity_thresh": 0.5,
        "appearance_thresh": 0.25,
        "with_reid": False,
        "fuse_score": True,
    }
    with open("iran_bytetrack.yaml", "w") as f:
        yaml.dump(bytetrack_cfg, f)
    with open("iran_botsort.yaml", "w") as f:
        yaml.dump(botsort_cfg, f)

class CollisionRiskOverlay:
    """Render advisory risk estimates; this class never commands vehicle actuators."""

    COLORS = {
        CLEAR: (0, 150, 0),
        CAUTION: (0, 165, 255),
        HIGH: (0, 100, 255),
        CRITICAL: (0, 0, 255),
    }

    def __init__(self):
        self.assessor = CollisionRiskAssessor()
        self.last_report = None

    def reset(self):
        self.assessor.reset()
        self.last_report = None

    def evaluate_frame(self, tracked_objects, frame, pedestrian_class_ids=None):
        height, width = frame.shape[:2]
        report = self.assessor.update(
            tracked_objects,
            frame.shape,
            pedestrian_class_ids=pedestrian_class_ids,
        )
        self.last_report = report

        corridor = np.array(
            [(round(x * width), round(y * height)) for x, y in report.corridor],
            dtype=np.int32,
        )
        overlay = frame.copy()
        cv2.fillPoly(overlay, [corridor], (255, 255, 0))
        cv2.addWeighted(overlay, 0.06, frame, 0.94, 0, frame)
        cv2.polylines(frame, [corridor], isClosed=True, color=(0, 255, 255), thickness=2)

        for item in report.objects:
            color = self.COLORS[item.level]
            x1, y1, x2, y2 = (int(value) for value in item.bbox)
            if item.level != CLEAR:
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                if item.ttc_seconds is not None:
                    detail = f"TTC~{item.ttc_seconds:.1f}s"
                elif item.cut_in_eta_seconds is not None:
                    detail = f"ENTRY~{item.cut_in_eta_seconds:.1f}s"
                else:
                    detail = item.level
                cv2.putText(
                    frame, f"{item.object_type.upper()} {item.track_id} {detail}",
                    (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2,
                )

        banner_color = self.COLORS[report.level]
        cv2.rectangle(frame, (0, 86), (width, 132), banner_color, -1)
        cv2.putText(
            frame, f"COLLISION RISK: {report.level} | {report.reason[:110]}",
            (12, 116), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2,
        )
        return report.level, report.reason, frame


class LiveMonitorBenchmark:
    def __init__(self, camera_id=0, pedestrian_weights=None):
        write_tracker_yamls()

        if not torch.cuda.is_available():
            raise RuntimeError(
                "PyTorch cannot see your GPU Run:\n"
                "pip uninstall -y torch torchvision torchaudio\n"
                "pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126"
            )

        self.device = "cuda:0"
        self.gpu_name = torch.cuda.get_device_name(0)
        torch.backends.cudnn.benchmark = True
        print(f"[OK] Locked onto GPU: {self.gpu_name} (FP16 Tensor Cores Enabled)")

        self.models = {
            1: {"name": "M1: YOLO26n + ByteTrack", "yolo": YOLO("yolo26n.pt").to(self.device), "tracker": "iran_bytetrack.yaml", "mode": "ultralytics"},
            2: {"name": "M2: YOLO26s + OC-SORT",   "yolo": YOLO("yolo26s.pt").to(self.device), "tracker": "iran_bytetrack.yaml", "mode": "ocsort"},
            3: {"name": "M3: YOLO26m + BoT-SORT",  "yolo": YOLO("yolo26m.pt").to(self.device), "tracker": "iran_botsort.yaml",   "mode": "ultralytics"}
        }
        self.ocsort = OCSORT(det_thresh=0.30, max_age=60, min_hits=2, iou_threshold=0.25) if HAS_BOXMOT else None

        self.pedestrian_model_cfg = None
        if pedestrian_weights:
            weights_path = str(pedestrian_weights)
            custom_model = YOLO(weights_path).to(self.device)
            custom_names = custom_model.names
            for class_id, expected_name in EXPECTED_PEDESTRIAN_NAMES.items():
                actual_name = (
                    custom_names.get(class_id, custom_names.get(str(class_id)))
                    if isinstance(custom_names, dict)
                    else custom_names[class_id] if class_id < len(custom_names) else None
                )
                if actual_name != expected_name:
                    raise ValueError(
                        f"Custom checkpoint class mismatch at ID {class_id}: "
                        f"expected '{expected_name}', found '{actual_name}'. "
                        "Train with data/pedestrian_attributes.yaml."
                    )
            self.pedestrian_model_cfg = {
                "name": "Custom Pedestrian Attributes",
                "yolo": custom_model,
                "tracker": "iran_bytetrack.yaml",
                "mode": "ultralytics",
            }
            print(f"[OK] Loaded custom pedestrian checkpoint: {weights_path}")

        self.active_slot = 1
        self.street_mode = False
        self.pedestrian_mode = False
        self.risk_overlay_enabled = False
        self.cap = cv2.VideoCapture(camera_id)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self.risk_overlay = CollisionRiskOverlay()

        self.scoreboard = {
            1: {"fps": 0.0, "ms": 0.0, "unique_ids": 0, "avg_lifespan": 0.0},
            2: {"fps": 0.0, "ms": 0.0, "unique_ids": 0, "avg_lifespan": 0.0},
            3: {"fps": 0.0, "ms": 0.0, "unique_ids": 0, "avg_lifespan": 0.0},
        }
        self.reset_current_metrics()

    def reset_current_metrics(self):
        self.fps_window = deque(maxlen=30)
        self.ms_window = deque(maxlen=30)
        self.seen_ids = set()
        self.id_frame_counts = defaultdict(int)
        self.trails = defaultdict(lambda: deque(maxlen=30))
        self.risk_overlay.reset()
        if HAS_BOXMOT:
            self.ocsort = OCSORT(det_thresh=0.30, max_age=60, min_hits=2, iou_threshold=0.25)

    def run(self):
        while self.cap.isOpened():
            t_frame_start = time.perf_counter()
            ret, frame = self.cap.read()
            if not ret:
                break

            if self.pedestrian_mode:
                active_classes = PEDESTRIAN_ATTRIBUTE_CLASSES
                cfg = self.pedestrian_model_cfg
            elif self.street_mode:
                active_classes = ROAD_AND_CROSSROAD_CLASSES if self.risk_overlay_enabled else IRAN_STREET_CLASSES
                cfg = self.models[self.active_slot]
            else:
                active_classes = WEBCAM_DESK_CLASSES
                cfg = self.models[self.active_slot]

            names_map = cfg["yolo"].names
            t_inf_start = time.perf_counter()

            tracked_objects = []

            if cfg["mode"] == "ocsort" and HAS_BOXMOT:
                res = cfg["yolo"].predict(
                    frame, classes=active_classes, imgsz=640,
                    device=self.device, half=True, conf=0.25, verbose=False
                )
                dets = res[0].boxes.data.cpu().numpy()
                if len(dets) > 0:
                    tracks = self.ocsort.update(dets, frame)
                    for trk in tracks:
                        tracked_objects.append((trk[0], trk[1], trk[2], trk[3], int(trk[4]), trk[5], int(trk[6])))
            else:
                res = cfg["yolo"].track(
                    frame, tracker=cfg["tracker"], classes=active_classes,
                    imgsz=640, device=self.device, half=True, persist=True, verbose=False
                )
                boxes = res[0].boxes
                if boxes.id is not None:
                    xyxy = boxes.xyxy.cpu().numpy()
                    tids = boxes.id.int().cpu().tolist()
                    confs = boxes.conf.cpu().tolist()
                    clss = boxes.cls.int().cpu().tolist()
                    for b, tid, c, cl in zip(xyxy, tids, confs, clss):
                        tracked_objects.append((b[0], b[1], b[2], b[3], tid, c, cl))

            inf_ms = (time.perf_counter() - t_inf_start) * 1000.0
            self.ms_window.append(inf_ms)

            for (x1, y1, x2, y2, tid, conf, cls_id) in tracked_objects:
                self.seen_ids.add(tid)
                self.id_frame_counts[tid] += 1
                cx, cy = int(0.5 * (x1 + x2)), int(0.5 * (y1 + y2))
                self.trails[tid].append((cx, cy))

                color = ((tid * 53) % 255, (tid * 97) % 255, (tid * 193) % 255)
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

                label = f"ID:{tid} {names_map.get(cls_id, cls_id)} {conf:.2f}"
                cv2.putText(frame, label, (int(x1), max(int(y1) - 8, 95)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

                pts = list(self.trails[tid])
                for i in range(1, len(pts)):
                    cv2.line(frame, pts[i - 1], pts[i], color, 2)

            if self.risk_overlay_enabled and (self.street_mode or self.pedestrian_mode):
                pedestrian_ids = (
                    PEDESTRIAN_ATTRIBUTE_CLASSES if self.pedestrian_mode else [0]
                )
                self.risk_overlay.evaluate_frame(
                    tracked_objects, frame, pedestrian_class_ids=pedestrian_ids
                )

            loop_fps = 1.0 / max(time.perf_counter() - t_frame_start, 1e-5)
            self.fps_window.append(loop_fps)
            avg_fps = sum(self.fps_window) / len(self.fps_window)
            avg_ms = sum(self.ms_window) / len(self.ms_window)
            avg_life = (sum(self.id_frame_counts.values()) / len(self.id_frame_counts)) if self.id_frame_counts else 0.0

            self.scoreboard[self.active_slot] = {
                "fps": avg_fps,
                "ms": avg_ms,
                "unique_ids": len(self.seen_ids),
                "avg_lifespan": avg_life
            }

            self.draw_hud(frame, cfg["name"], len(tracked_objects), avg_fps, avg_ms, avg_life)
            cv2.imshow("Live Object Tracking Benchmark Monitor (RTX 3060)", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key in [ord("1"), ord("2"), ord("3")]:
                self.active_slot = int(chr(key))
                self.pedestrian_mode = False
                self.reset_current_metrics()
            elif key == ord("w"):
                self.pedestrian_mode = False
                self.street_mode = False
                self.reset_current_metrics()
            elif key == ord("s"):
                self.pedestrian_mode = False
                self.street_mode = True
                self.reset_current_metrics()
            elif key == ord("p"):
                if self.pedestrian_model_cfg is None:
                    print("[WARN] Start with --pedestrian-weights path/to/best.pt to enable this mode.")
                else:
                    self.pedestrian_mode = not self.pedestrian_mode
                    if self.pedestrian_mode:
                        self.street_mode = False
                    self.reset_current_metrics()
            elif key == ord("b"):
                self.risk_overlay_enabled = not self.risk_overlay_enabled
                if not self.risk_overlay_enabled:
                    self.risk_overlay.reset()
            elif key == ord("r"):
                self.reset_current_metrics()

        self.cap.release()
        cv2.destroyAllWindows()

    def draw_hud(self, frame, model_name, active_count, avg_fps, avg_ms, avg_life):
        h, w = frame.shape[:2]
        cv2.rectangle(frame, (0, 0), (w, 85), (20, 20, 20), -1)
        if self.pedestrian_mode:
            mode_str = "PEDESTRIAN ATTRIBUTE MODE"
        else:
            mode_str = "STREET MODE" if self.street_mode else "WEBCAM DESK MODE"

        if self.risk_overlay_enabled and (self.street_mode or self.pedestrian_mode):
            mode_str += " + RISK ASSESSMENT"

        fps_color = (0, 255, 0) if avg_fps >= 28 else ((0, 200, 255) if avg_fps >= 18 else (0, 0, 255))

        cv2.putText(frame, f"ACTIVE: {model_name} | [{mode_str}] | GPU: {self.gpu_name}", (15, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2)
        cv2.putText(frame, f"Camera FPS: {avg_fps:.1f}  |  GPU Inference: {avg_ms:.1f}ms  |  In-Frame: {active_count}  |  Total IDs: {len(self.seen_ids)}  |  Stability: {avg_life:.0f}f",
                    (15, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.60, fps_color, 2)

        cv2.rectangle(frame, (0, h - 115), (w, h), (15, 15, 15), -1)
        cv2.putText(frame, "HOTKEYS: [1] M1  [2] M2  [3] M3  |  [W] Webcam  [S] Street  [P] Pedestrian Attributes  [B] Risk Assessment  [R] Reset  [Q] Quit",
                    (15, h - 90), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (180, 180, 180), 1)

        for slot in [1, 2, 3]:
            s = self.scoreboard[slot]
            m_title = self.models[slot]["name"]
            if self.pedestrian_mode and slot == self.active_slot:
                m_title = self.pedestrian_model_cfg["name"]
            y_pos = h - 65 + (slot - 1) * 22
            prefix = ">> " if slot == self.active_slot else "   "
            col = (0, 255, 255) if slot == self.active_slot else (200, 200, 200)
            row_txt = f"{prefix}{m_title:<26} | FPS: {s['fps']:5.1f} | GPU Latency: {s['ms']:5.1f}ms | Unique IDs: {s['unique_ids']:4d} | Stability: {s['avg_lifespan']:5.0f}f"
            cv2.putText(frame, row_txt, (15, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.52, col, 1 if slot != self.active_slot else 2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Live multi-model object tracking and ADAS benchmark."
    )
    parser.add_argument("--camera", type=int, default=0, help="Webcam device index.")
    parser.add_argument(
        "--pedestrian-weights",
        default=None,
        help="Optional custom checkpoint trained with data/pedestrian_attributes.yaml.",
    )
    cli_args = parser.parse_args()

    bench = LiveMonitorBenchmark(
        camera_id=cli_args.camera,
        pedestrian_weights=cli_args.pedestrian_weights,
    )
    bench.run()
