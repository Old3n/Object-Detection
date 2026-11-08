"""Desktop launcher and same-origin inference server for the AutoPilot PWA."""

from __future__ import annotations

import argparse
import hmac
import json
import math
import multiprocessing
import runpy
import secrets
import sys
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

ROOT = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))
CLI_TOOLS = {
    "risk": "RiskVideoBenchmark", "video": "VideoBenchmark",
    "surround": "SurroundCameraBenchmark", "monitor": "ModelBenchmark",
    "evaluate": "EvaluateModel", "train": "train_pedestrian_model",
    "cameras": "tools.list_cameras", "jaad": "tools.prepare_jaad_dataset",
    "kitti": "tools.kitti_to_video",
}
MAX_UPLOAD = 8 * 1024 * 1024
ASSETS = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/index.html": ("index.html", "text/html; charset=utf-8"),
    "/app.js": ("app.js", "text/javascript"),
    "/style.css": ("style.css", "text/css"),
    "/sw.js": ("sw.js", "text/javascript"),
    "/manifest.webmanifest": ("manifest.webmanifest", "application/manifest+json"),
    "/icon-192.png": ("icon-192.png", "image/png"),
    "/icon-512.png": ("icon-512.png", "image/png"),
}


class Detector:
    def __init__(self, weights: str, device: str):
        self.weights = weights
        self.device = device
        self.model = None
        self.lock = threading.Lock()

    def predict(self, data: bytes, confidence: float) -> dict:
        import cv2
        import numpy as np

        frame = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("Send a valid JPEG image.")
        if max(frame.shape[:2]) > 2048:
            raise ValueError("Resize images to at most 2048 pixels per side.")
        if not self.lock.acquire(blocking=False):
            raise BlockingIOError("Detector is busy. Try again shortly.")
        try:
            if self.model is None:
                from ultralytics import YOLO
                self.model = YOLO(self.weights)
            start = time.perf_counter()
            result = self.model.predict(frame, conf=confidence, device=self.device, verbose=False)[0]
            detections = []
            for box in result.boxes:
                class_id = int(box.cls.item())
                detections.append({
                    "class_id": class_id,
                    "label": str(result.names[class_id]),
                    "confidence": round(float(box.conf.item()), 4),
                    "bbox": [round(float(v), 2) for v in box.xyxy[0].tolist()],
                })
            return {
                "detections": detections,
                "inference_ms": round((time.perf_counter() - start) * 1000, 1),
                "width": frame.shape[1], "height": frame.shape[0],
                "model": Path(self.weights).name,
            }
        finally:
            self.lock.release()


def make_server(host: str, port: int, detector: Detector, token: str) -> ThreadingHTTPServer:
    class Handler(BaseHTTPRequestHandler):
        def setup(self):
            super().setup()
            self.connection.settimeout(30)

        def log_message(self, format, *args):
            pass

        def reply(self, status: int, body: bytes, content_type: str):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store" if self.path.startswith('/api/') else "no-cache")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header("Content-Security-Policy", "default-src 'self'; img-src 'self' blob: data:; media-src 'self' blob:; frame-ancestors 'none'; object-src 'none'")
            self.end_headers()
            self.wfile.write(body)

        def json_reply(self, status: int, value: dict):
            self.reply(status, json.dumps(value).encode(), "application/json")

        def authorized(self):
            if not hmac.compare_digest(self.headers.get("Authorization", ""), f"Bearer {token}"):
                self.json_reply(401, {"error": "Enter the access key printed by AutoPilot."})
                return False
            return True

        def do_GET(self):
            path = urlsplit(self.path).path
            if path == "/api/health":
                if self.authorized():
                    self.json_reply(200, {"model": Path(detector.weights).name, "device": detector.device})
            elif path in ASSETS:
                filename, mime = ASSETS[path]
                self.reply(200, (ROOT / "web" / filename).read_bytes(), mime)
            else:
                self.json_reply(404, {"error": "Not found"})

        def do_POST(self):
            self.close_connection = True
            if urlsplit(self.path).path != "/api/detect":
                self.json_reply(404, {"error": "Not found"})
                return
            if not self.authorized():
                return
            if self.headers.get("Content-Type", "").split(";")[0] != "image/jpeg":
                self.json_reply(415, {"error": "Send image/jpeg."})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                confidence = float(parse_qs(urlsplit(self.path).query).get("confidence", ["0.35"])[0])
                if not math.isfinite(confidence) or not 0.05 <= confidence <= 0.95:
                    raise ValueError("Confidence must be between 0.05 and 0.95.")
                if not 0 < length <= MAX_UPLOAD:
                    self.json_reply(413, {"error": "Image must be between 1 byte and 8 MB."})
                    return
                data = self.rfile.read(length)
                if len(data) != length:
                    raise ValueError("Incomplete image upload.")
                self.json_reply(200, detector.predict(data, confidence))
            except (ValueError, TimeoutError) as exc:
                self.json_reply(400, {"error": str(exc)})
            except BlockingIOError as exc:
                self.json_reply(429, {"error": str(exc)})
            except Exception as exc:
                print(f"Inference failed: {exc}", file=sys.stderr)
                self.json_reply(503, {"error": "Inference unavailable. Check the server console and model installation."})

    return ThreadingHTTPServer((host, port), Handler)


def dispatch_tool(arguments: list[str]) -> bool:
    if not arguments or arguments[0] != "--tool":
        return False
    if len(arguments) < 2 or arguments[1] not in CLI_TOOLS:
        raise SystemExit("Choose --tool " + "|".join(CLI_TOOLS))
    tool = arguments[1]
    forwarded = arguments[2:]
    if tool in {"risk", "video", "surround"} and not any(arg.split("=", 1)[0] == "--weights" for arg in forwarded):
        forwarded += ["--weights", str(ROOT / "yolo26n.pt")]
    sys.argv = [CLI_TOOLS[tool]] + forwarded
    runpy.run_module(CLI_TOOLS[tool], run_name="__main__")
    return True


def main():
    if dispatch_tool(sys.argv[1:]):
        return
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--weights", default=str(ROOT / "yolo26n.pt"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    detector = Detector(args.weights, args.device)
    if args.self_test:
        import cv2
        import numpy as np
        for filename, _ in ASSETS.values():
            assert (ROOT / "web" / filename).is_file(), filename
        ok, data = cv2.imencode(".jpg", np.zeros((320, 320, 3), dtype=np.uint8))
        assert ok
        result = detector.predict(data.tobytes(), 0.35)
        assert result["width"] == 320
        print("Bundled assets and model inference passed.")
        return
    token = secrets.token_urlsafe(32)
    server = make_server(args.host, args.port, detector, token)
    url = f"http://127.0.0.1:{server.server_port}/"
    print(f"AutoPilot: {url}\nAccess key: {token}\nClose with Ctrl+C. Android needs a trusted HTTPS address.", flush=True)
    if not args.no_browser:
        webbrowser.open(f"{url}#key={token}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
