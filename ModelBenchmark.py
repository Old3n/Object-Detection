import time
from collections import defaultdict, deque
import cv2
import numpy as np
import torch
import yaml
from ultralytics import YOLO

try:
    from boxmot import OCSORT
    HAS_BOXMOT = True
except ImportError:
    HAS_BOXMOT = False

# COCO class subsets
WEBCAM_DESK_CLASSES = [0, 39, 41, 67]        # person, bottle, cup, cell phone
IRAN_STREET_CLASSES = [0, 1, 2, 3, 5, 7]     # person, bicycle, car, motorcycle, bus, truck
ROAD_AND_CROSSROAD_CLASSES = [0, 1, 2, 3, 5, 7, 9, 11]  # vehicles + traffic light (9), stop sign (11)


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


class BrakingDecisionController:
    """Level 1 ADAS braking decision module.

    Evaluates tracked obstacles in the ego-lane using proximity,
    Time-To-Collision (TTC), and lateral cut-in estimation.
    """

    PEDESTRIAN_CLASSES = {0}
    VEHICLE_CLASSES = {1, 2, 3, 5, 7}
    SIGN_CLASSES = {9, 11}

    def __init__(self, frame_w=1280, frame_h=720, ttc_brake_thresh=2.0):
        self.w = frame_w
        self.h = frame_h
        self.ttc_thresh = ttc_brake_thresh
        self.track_history = defaultdict(lambda: deque(maxlen=10))

        # Ego-lane polygon (trapezoidal corridor)
        self.ego_polygon = np.array([
            [int(0.30 * frame_w), int(0.40 * frame_h)],
            [int(0.70 * frame_w), int(0.40 * frame_h)],
            [int(0.95 * frame_w), int(0.95 * frame_h)],
            [int(0.05 * frame_w), int(0.95 * frame_h)],
        ], np.int32)

        # Fraction of frame height occupied by bbox to trigger proximity alert
        self.proximity_thresh_pedestrian = 0.35
        self.proximity_thresh_vehicle = 0.45
        self.critical_y_line = int(0.75 * frame_h)

        self.last_command = "CRUISE"
        self.last_reason = ""
        self.threat_level = 0  # 0: clear, 1: caution, 2: emergency

    def reset(self):
        self.track_history.clear()
        self.last_command = "CRUISE"
        self.last_reason = ""
        self.threat_level = 0

    def in_ego_lane(self, x, y):
        return cv2.pointPolygonTest(self.ego_polygon, (float(x), float(y)), False) >= 0

    def evaluate_frame(self, tracked_objects, frame):
        now = time.perf_counter()
        command = "CRUISE"
        trigger_reason = ""
        threat_level = 0

        # Render ego-lane overlay
        overlay = frame.copy()
        cv2.fillPoly(overlay, [self.ego_polygon], (255, 255, 0))
        cv2.addWeighted(overlay, 0.08, frame, 0.92, 0, frame)
        cv2.polylines(frame, [self.ego_polygon], isClosed=True, color=(0, 255, 255), thickness=2)

        # Critical proximity boundary line
        dash_len = 20
        x_start = int(0.05 * self.w)
        x_end = int(0.95 * self.w)
        for x in range(x_start, x_end, dash_len * 2):
            cv2.line(frame, (x, self.critical_y_line),
                     (min(x + dash_len, x_end), self.critical_y_line), (0, 0, 255), 2)

        for (x1, y1, x2, y2, tid, conf, cls_id) in tracked_objects:
            box_w = max(x2 - x1, 1.0)
            box_h = max(y2 - y1, 1.0)
            bottom_center_x = 0.5 * (x1 + x2)
            bottom_y = y2
            box_h_ratio = box_h / self.h

            self.track_history[tid].append((now, box_w, bottom_center_x, bottom_y))
            hist = self.track_history[tid]

            is_pedestrian = cls_id in self.PEDESTRIAN_CLASSES
            is_vehicle = cls_id in self.VEHICLE_CLASSES
            is_sign = cls_id in self.SIGN_CLASSES
            in_lane = self.in_ego_lane(bottom_center_x, bottom_y)

            # Rule 0: Traffic signs / signals
            if is_sign and box_h_ratio > 0.06:
                if threat_level < 1:
                    command = "SLOW - SIGN AHEAD"
                    trigger_reason = f"{'Stop Sign' if cls_id == 11 else 'Traffic Light'} (ID {tid})"
                    threat_level = 1

            # Rule 1: Proximity threshold
            if in_lane and (is_pedestrian or is_vehicle):
                prox_thresh = self.proximity_thresh_pedestrian if is_pedestrian else self.proximity_thresh_vehicle
                if box_h_ratio > prox_thresh:
                    command = "EMERGENCY BRAKE!"
                    obj_type = "Pedestrian" if is_pedestrian else "Vehicle"
                    trigger_reason = f"{obj_type} ID {tid} too close ({box_h_ratio:.0%} of frame)"
                    threat_level = 2
                    obj_overlay = frame.copy()
                    cv2.rectangle(obj_overlay, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), -1)
                    cv2.addWeighted(obj_overlay, 0.25, frame, 0.75, 0, frame)
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 3)
                    break
                elif box_h_ratio > prox_thresh * 0.6:
                    if threat_level < 1:
                        command = "CAUTION - CLOSING"
                        obj_type = "Pedestrian" if is_pedestrian else "Vehicle"
                        trigger_reason = f"{obj_type} ID {tid} approaching ({box_h_ratio:.0%})"
                        threat_level = 1
                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 165, 255), 3)

            # Rule 2: Time-To-Collision (optical expansion)
            if in_lane and len(hist) >= 5 and threat_level < 2:
                t_old, w_old, x_old, y_old = hist[0]
                dt = max(now - t_old, 1e-3)
                scale_ratio = box_w / max(w_old, 1.0)
                if scale_ratio > 1.03:
                    ttc = dt / (scale_ratio - 1.0)
                    if ttc < self.ttc_thresh:
                        command = "BRAKE - LOW TTC!"
                        trigger_reason = f"ID {tid} TTC: {ttc:.1f}s"
                        threat_level = 2
                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 3)
                        break

            # Rule 3: Lateral cut-in trajectory projection (0.8s lookahead)
            if not in_lane and len(hist) >= 5 and threat_level < 2:
                t_old, w_old, x_old, y_old = hist[0]
                dt = max(now - t_old, 1e-3)
                vx = (bottom_center_x - x_old) / dt
                future_x = bottom_center_x + (vx * 0.8)

                if bottom_y > int(0.50 * self.h) and self.in_ego_lane(future_x, bottom_y):
                    if abs(vx) > 50:
                        command = "BRAKE - CUT-IN!"
                        trigger_reason = f"ID {tid} cutting into lane!"
                        threat_level = 2
                        cv2.arrowedLine(frame, (int(bottom_center_x), int(bottom_y)),
                                        (int(future_x), int(bottom_y)), (0, 165, 255), 3)

        self.last_command = command
        self.last_reason = trigger_reason
        self.threat_level = threat_level

        # HUD threat display
        if threat_level == 2:
            cv2.rectangle(frame, (0, 86), (self.w, 135), (0, 0, 255), -1)
            cv2.putText(frame, f"!! {command}  {trigger_reason}", (15, 122),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
        elif threat_level == 1:
            cv2.rectangle(frame, (0, 86), (self.w, 128), (0, 140, 255), -1)
            cv2.putText(frame, f">> {command}  {trigger_reason}", (15, 117),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        else:
            cv2.rectangle(frame, (self.w - 180, 88), (self.w - 10, 112), (0, 140, 0), -1)
            cv2.rectangle(frame, (self.w - 180, 88), (self.w - 10, 112), (0, 180, 0), 2)
            cv2.putText(frame, "ADAS: CRUISE", (self.w - 175, 106),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1)

        return command, trigger_reason, frame


class LiveMonitorBenchmark:
    def __init__(self, camera_id=0):
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

        self.active_slot = 1
        self.street_mode = False
        self.braking_enabled = False
        self.cap = cv2.VideoCapture(camera_id)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self.braking_controller = BrakingDecisionController(frame_w=1280, frame_h=720, ttc_brake_thresh=2.0)

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
        self.braking_controller.reset()
        if HAS_BOXMOT:
            self.ocsort = OCSORT(det_thresh=0.30, max_age=60, min_hits=2, iou_threshold=0.25)

    def run(self):
        names_map = self.models[1]["yolo"].names
        while self.cap.isOpened():
            t_frame_start = time.perf_counter()
            ret, frame = self.cap.read()
            if not ret:
                break

            if self.street_mode:
                active_classes = ROAD_AND_CROSSROAD_CLASSES if self.braking_enabled else IRAN_STREET_CLASSES
            else:
                active_classes = WEBCAM_DESK_CLASSES

            cfg = self.models[self.active_slot]
            t_inf_start = time.perf_counter()

            tracked_objects = []

            if cfg["mode"] == "ocsort" and HAS_BOXMOT:
                res = cfg["yolo"].predict(
                    frame, classes=active_classes, imgsz=640,
                    device=self.device, quantize="fp16", conf=0.25, verbose=False
                )
                dets = res[0].boxes.data.cpu().numpy()
                if len(dets) > 0:
                    tracks = self.ocsort.update(dets, frame)
                    for trk in tracks:
                        tracked_objects.append((trk[0], trk[1], trk[2], trk[3], int(trk[4]), trk[5], int(trk[6])))
            else:
                res = cfg["yolo"].track(
                    frame, tracker=cfg["tracker"], classes=active_classes,
                    imgsz=640, device=self.device, quantize="fp16", persist=True, verbose=False
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

            if self.braking_enabled and self.street_mode:
                self.braking_controller.evaluate_frame(tracked_objects, frame)

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
                self.reset_current_metrics()
            elif key == ord("w"):
                self.street_mode = False
                self.reset_current_metrics()
            elif key == ord("s"):
                self.street_mode = True
                self.reset_current_metrics()
            elif key == ord("b"):
                self.braking_enabled = not self.braking_enabled
                if not self.braking_enabled:
                    self.braking_controller.reset()
            elif key == ord("r"):
                self.reset_current_metrics()

        self.cap.release()
        cv2.destroyAllWindows()

    def draw_hud(self, frame, model_name, active_count, avg_fps, avg_ms, avg_life):
        h, w = frame.shape[:2]
        cv2.rectangle(frame, (0, 0), (w, 85), (20, 20, 20), -1)
        mode_str = "STREET MODE" if self.street_mode else "WEBCAM DESK MODE"

        if self.braking_enabled and self.street_mode:
            mode_str += " + ADAS BRAKING"

        fps_color = (0, 255, 0) if avg_fps >= 28 else ((0, 200, 255) if avg_fps >= 18 else (0, 0, 255))

        cv2.putText(frame, f"ACTIVE: {model_name} | [{mode_str}] | GPU: {self.gpu_name}", (15, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2)
        cv2.putText(frame, f"Camera FPS: {avg_fps:.1f}  |  GPU Inference: {avg_ms:.1f}ms  |  In-Frame: {active_count}  |  Total IDs: {len(self.seen_ids)}  |  Stability: {avg_life:.0f}f",
                    (15, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.60, fps_color, 2)

        cv2.rectangle(frame, (0, h - 115), (w, h), (15, 15, 15), -1)
        cv2.putText(frame, "HOTKEYS: [1] M1  [2] M2  [3] M3  |  [W] Webcam  [S] Street  [B] Braking ADAS  [R] Reset  [Q] Quit",
                    (15, h - 90), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (180, 180, 180), 1)

        for slot in [1, 2, 3]:
            s = self.scoreboard[slot]
            m_title = self.models[slot]["name"]
            y_pos = h - 65 + (slot - 1) * 22
            prefix = ">> " if slot == self.active_slot else "   "
            col = (0, 255, 255) if slot == self.active_slot else (200, 200, 200)
            row_txt = f"{prefix}{m_title:<26} | FPS: {s['fps']:5.1f} | GPU Latency: {s['ms']:5.1f}ms | Unique IDs: {s['unique_ids']:4d} | Stability: {s['avg_lifespan']:5.0f}f"
            cv2.putText(frame, row_txt, (15, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.52, col, 1 if slot != self.active_slot else 2)


if __name__ == "__main__":
    bench = LiveMonitorBenchmark(camera_id=0)
    bench.run()
