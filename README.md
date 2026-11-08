# AutoPilot · Perception Studio

A desktop and Android-friendly workspace for object detection, backed by the project's Python/Ultralytics models. Open a camera, image, or video; inspect detections; adjust confidence; export the current frame's results.

The existing command-line tools also support tracking, lane estimation, video benchmarks, pedestrian-model training, and five-camera advisory risk analysis.

## Start here

| Platform | Run | Requirements |
|---|---|---|
| Windows x64 | `AutoPilot.exe` from [Releases](https://github.com/Old3n/Object-Detection/releases) when a verified build is published | Browser; no Python installation needed |
| Python source | `python AutoPilot.py` | Python 3.11 or 3.12 and dependencies below |
| Android | Open your backend's HTTPS URL in Chrome and install the PWA | Running backend and trusted HTTPS connection |

The Windows executable includes the default YOLO26n checkpoint and CPU runtime. It starts a local server and opens the dashboard in your browser. Keep its console open; press Ctrl+C to stop. Initial extraction and model loading can take time. Models and datasets for custom pedestrian training are not included.

## Run from source

```bash
git clone https://github.com/Old3n/Object-Detection.git
cd Object-Detection
python -m venv .venv
```

Activate the environment:

```powershell
.venv\Scripts\Activate.ps1
```

On Linux/macOS, use `source .venv/bin/activate`. Then:

```bash
python -m pip install -r requirements.txt
python AutoPilot.py
```

The browser opens at `http://127.0.0.1:8765`. An access key is passed in the URL fragment and retained only for the browser session. To connect another browser, enter the key printed in the console. The first source run downloads `yolo26n.pt`, requiring internet access; later runs can use the saved checkpoint.

Use a different checkpoint, port, or device:

```bash
python AutoPilot.py --weights path/to/best.pt --device cpu --port 8765
python AutoPilot.py --weights yolo26n.pt --device 0
```

GPU inference requires a matching CUDA-enabled PyTorch installation. The release executable targets CPU for portability; build a separate GPU variant if required. The optional legacy OC-SORT comparison also requires `pip install boxmot`.

## Dashboard

- **Camera:** use the desktop webcam or Android rear camera after granting permission.
- **Open file:** analyze an image or play a browser-supported local video.
- **Confidence:** choose which detections appear; changes apply to the next submitted frame.
- **Stop:** release the camera and retain the last annotated frame.
- **Export:** download JSON for the displayed frame, including boxes, labels, confidence, model, and inference latency.

Frames are resized to a maximum side of 1280 pixels before upload. Box coordinates refer to that submitted frame. Only one inference request is sent at a time; video playback may skip frames while inference is busy. The UI reports model-call latency, not full video throughput. For complete frame-by-frame benchmarking and MP4/CSV/JSON reports, use the CLI.

The web dashboard performs detection only. Tracking, lane corridors, TTC, and multi-camera risk assessment remain in the CLI tools. The server processes frames in memory and does not save uploads. Each connected client sends frames to the machine running the backend; use trusted HTTPS when connecting remotely.

## Install on Android as a PWA

1. Run `python AutoPilot.py --no-browser` on the inference computer, or `AutoPilot.exe --no-browser` on Windows.
2. Put a trusted HTTPS reverse proxy in front of `127.0.0.1:8765`. Serve the app and `/api/` from the **same origin**. See the [complete setup](docs/PORTING.md#android-pwa-deployment).
3. Open that HTTPS address in Android Chrome and enter the console's access key.
4. Tap **Install app** when available, or use Chrome's **Add to Home screen / Install app** menu.
5. Launch AutoPilot from the home screen and grant camera permission.

A plain `http://192.168…` LAN address does not enable Android camera access or normal PWA installation. The interface is cached offline; inference is not. This port is a connected PWA, not an APK or an on-device Python runtime. Restarting the backend generates a new access key.

## Packaged CLI tools

The same executable also includes the original Python entry points:

```powershell
.\AutoPilot.exe --tool risk --source 0 --show
.\AutoPilot.exe --tool video --source videos/test.mp4
.\AutoPilot.exe --tool surround --config data/surround_cameras.json
.\AutoPilot.exe --tool cameras --max-index 8
```

Additional tools: `monitor`, `evaluate`, `train`, `jaad`, and `kitti`. Pass `--help` after the tool name for its options. Risk/video/surround use the bundled checkpoint unless `--weights` is supplied. Supply your own datasets/config files and custom weights. The legacy monitor requires CUDA and a GPU build; OC-SORT remains an optional source dependency. Training and evaluation still need external data. Source mode also accepts `python AutoPilot.py --tool ...`.

## Tests

In a fresh virtual environment:

```bash
python -m pip install -r requirements-test.txt
python -m unittest discover -s tests -v
node --check web/app.js
node --check web/sw.js
```

Tests use synthetic images/videos and a detector stand-in, without downloading model weights. They cover the risk estimators, lane detection, all five camera modes, output reports, dashboard authentication, invalid uploads, and static assets. GitHub Actions runs the suite on Windows and Linux with Python 3.11/3.12. The release build additionally runs real inference through the packaged executable.

Optional browser acceptance checks (uses a deterministic detector, not a downloaded model):

```bash
python -m pip install playwright Pillow
python -m playwright install chromium
python tools/check_dashboard.py
```

This exercises desktop/mobile layout, image results, JSON export, camera lifecycle with a synthetic browser camera, and offline shell loading. Physical Android installation and camera performance still need device testing.

## Build and publish

On Windows, using a clean Python 3.11 environment:

```powershell
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt "pyinstaller>=6.11,<7"
python -c "from ultralytics import YOLO; YOLO('yolo26n.pt')"
python -m PyInstaller --noconfirm --clean AutoPilot.spec
.\dist\AutoPilot.exe --self-test
python tools/package_pwa.py
```

Outputs: `dist/AutoPilot.exe` and `dist/AutoPilot-pwa.zip`. The ZIP includes the PWA, backend source, dependency requirements, and deployment instructions. It is not a static-only deployment.

The **Build desktop and PWA** Actions workflow produces downloadable build artifacts. A pushed `v*` tag publishes the executable, PWA archive, dependency inventory, and SHA-256 checksums to GitHub Releases only after the packaged executable passes its inference self-test:

```bash
git tag v1.0.0
git push origin v1.0.0
```

Choose an unused version and tag the reviewed commit. For native macOS/Linux builds, Android deployment, verification, and future on-device ports, follow [Porting and release instructions](docs/PORTING.md).

## CLI and research workflows

| Tool | Purpose |
|---|---|
| `ModelBenchmark.py` | GPU webcam comparison: YOLO26 variants and trackers |
| `VideoBenchmark.py` | Fixed-video inference with latency and output reports |
| `RiskVideoBenchmark.py` | Single camera/video detection, tracking, lane and advisory risk |
| `SurroundCameraBenchmark.py` | Five independent camera workers |
| `train_pedestrian_model.py` | Train a custom pedestrian checkpoint |
| `EvaluateModel.py` | Ground-truth precision, recall, F1 and mAP evaluation |
| `tools/prepare_jaad_dataset.py` | Convert selected JAAD videos and annotations |
| `tools/list_cameras.py` | Find working camera indices and backends |

Full commands, dataset-label policy, camera setup, and estimator limitations are preserved in the [Research and CLI guide](docs/RESEARCH_GUIDE.md).

This is a research project. Risk overlays are advisory image-space estimates, not validated braking or lane-change decisions. Detection counts and FPS do not establish accuracy; evaluate against labelled data. Review dependency and model licensing before redistribution, including [Ultralytics licensing](https://www.ultralytics.com/license).
