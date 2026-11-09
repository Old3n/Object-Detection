# Research and CLI guide

Real-time multi-model object detection and tracking benchmark with an advisory, track-based collision-risk estimator. Compare YOLO26 variants paired with ByteTrack, OC-SORT, and BoT-SORT.

[Back to the quick start](../README.md) · [Packaging and ports](PORTING.md)

Run all commands below from the repository root.

## Dependencies

```bash
pip install -U ultralytics boxmot opencv-python lapx pyyaml
```

### PyTorch GPU Setup (CUDA 12.6)

```bash
# 1. Remove the CPU-only version of PyTorch
pip uninstall -y torch torchvision torchaudio

# 2. Install CUDA-enabled PyTorch
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

### Verify GPU Access

```bash
python -c "import torch; print('GPU Active:', torch.cuda.is_available(), '| Card:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"
```

## Usage

```bash
python ModelBenchmark.py
```

## Hotkeys

| Key | Action |
|-----|--------|
| `1` | Switch to M1: YOLO26n + ByteTrack |
| `2` | Switch to M2: YOLO26s + OC-SORT |
| `3` | Switch to M3: YOLO26m + BoT-SORT |
| `W` | Webcam Desk Mode (person, bottle, cup, phone) |
| `S` | Street Mode (person, bicycle, car, motorcycle, bus, truck, traffic light, stop sign) |
| `B` | Toggle collision-risk assessment overlay (Street / Pedestrian mode) |
| `R` | Reset metrics |
| `Q` | Quit |

## Collision Risk Assessment

Press `B` in Street Mode or Pedestrian Attribute Mode to enable the visual risk overlay. The estimator consumes tracked detections and provides four advisory levels:

| Level | Meaning |
|---|---|
| `CLEAR` | No elevated risk estimated from the currently tracked objects |
| `CAUTION` | Approximate TTC is 3–5 seconds, or a track is predicted to enter the ego corridor |
| `HIGH` | Approximate TTC is 1.5–3 seconds |
| `CRITICAL` | Approximate TTC is 1.5 seconds or less |

### Estimation method

- **Track history:** maintains timestamped normalized bounding-box width and horizontal position per tracker ID.
- **Apparent-expansion TTC:** fits a line to log(box width) over time. For positive expansion slope (s), the approximate TTC is (1/s). The estimate is withheld until enough history is available.
- **Cut-in estimate:** extrapolates normalized horizontal track motion and checks whether the projected bottom-center enters the configured ego corridor within one second.
- **Resolution-independent geometry:** the default trapezoid is normalized to image coordinates, so it scales to the actual frame dimensions.
- **Stale-track cleanup:** track history is discarded after the track has been absent for one second.

The default corridor is a configurable image-space region. It is **not lane detection**. Replace it with a calibrated lane polygon or a lane model before interpreting it as the vehicle's actual path. The cut-in projection uses raw image-space motion and does not compensate for ego-vehicle translation, yaw, pitch, or camera vibration; those effects can create false lateral-motion estimates.

### Important limitations

This is an experimental visual risk estimator, not a braking controller and not a validated AEB function. Bounding-box expansion is affected by camera motion, object orientation, detector jitter, occlusion, and tracker ID switches. It does not provide metric distance, relative speed in m/s, road friction, stopping distance, or vehicle dynamics. The TTC value must be treated as an approximate research signal, not a guaranteed time-to-collision. The overlay does not send commands to the car.

The estimator is covered by unit tests:

```bash
python -m unittest discover -s tests -v
```

GitHub Actions also compiles the Python sources and runs this suite on pushes and pull requests.

---

## Custom Pedestrian Attribute Detector

This branch adds a YOLO fine-tuning workflow for these pedestrian categories:

| Class ID | Class name | Annotation meaning |
|---:|---|---|
| 0 | `child` | A person annotated as a child in the source dataset |
| 1 | `woman` | An adult woman, unless a higher-priority category below applies |
| 2 | `elderly_person` | A person annotated as elderly in the source dataset |
| 3 | `person_with_wheelchair` | A person actively using a wheelchair |
| 4 | `other_person` | Any other pedestrian, so the detector can still represent people outside the four requested groups |

### Important class-label policy

Standard YOLO detection assigns one class to each bounding box. Some requested attributes can overlap (for example, an elderly woman using a wheelchair). The supplied dataset template therefore uses mutually exclusive labels with this priority:

`person_with_wheelchair > child > elderly_person > woman > other_person`

Use the same policy for every image and split. Here, `woman` means an adult woman not assigned to one of the higher-priority categories. `other_person` is important: without it, training only on the four requested groups would teach the model to ignore other people.

These are dataset labels, not reliable traits that can be established from every image. Use a consistent annotation guideline and leave uncertain cases as `other_person` rather than guessing. If you need overlapping attributes (for example, both `woman` and `person_with_wheelchair` for the same box), that requires a multi-label model/training pipeline rather than ordinary single-class YOLO detection.

### Dataset layout

Create this structure and place your images and matching YOLO label files into it:

```text
datasets/pedestrian_attributes/
  images/
    train/
    val/
    test/       # optional; not used by the training command
  labels/
    train/
    val/
    test/       # optional
```

Each image needs a same-named `.txt` annotation file. Each line is:

```text
class_id x_center y_center width height
```

Coordinates must be normalized to `0..1`. Example (one child box):

```text
0 0.512 0.481 0.180 0.620
```

The class IDs and dataset paths are defined in `data/pedestrian_attributes.yaml`. Keep the train/validation split separate; do not put near-duplicate frames from the same video in both splits.

### Train / fine-tune

Install dependencies (the project already uses Ultralytics, OpenCV and PyTorch):

```bash
pip install -U ultralytics opencv-python pyyaml
```

For CUDA 12.6, use the PyTorch installation command in the GPU setup section above. Then train from the repository root:

```bash
python train_pedestrian_model.py --model yolo26n.pt --epochs 100 --imgsz 640 --batch 8
```

The script validates the YAML and train/validation directories before starting. Lower `--batch` if GPU memory is exhausted. The best checkpoint is written under:

```text
runs/pedestrian/yolo26n-pedestrian-attributes/weights/best.pt
```

Use your actual run directory if you changed `--project` or `--name`. This repository does not include the labelled training dataset or a trained custom checkpoint; detection of these new classes becomes available after you supply data and fine-tune the model.

## Live monitor integration

The existing webcam monitor can load the custom checkpoint without confusing its class IDs with COCO classes:

```bash
python ModelBenchmark.py --pedestrian-weights runs/pedestrian/yolo26n-pedestrian-attributes/weights/best.pt
```

Press `P` to toggle Pedestrian Attribute Mode. Press `B` to enable/disable the collision-risk overlay while in that mode. All five custom person classes are passed to the pedestrian risk estimator; they are not interpreted as COCO bicycle/car IDs. Press `W` or `S` to return to the existing webcam or street modes, and `1`–`3` to switch back to the original COCO model comparisons. The checkpoint class names and IDs are checked at startup.

## Iranian road signs and night boost

`ModelBenchmark.py` can run a second checkpoint for Iranian road signs, such as the ITSD YOLO weights, in Street Mode (`S`):

```bash
python ModelBenchmark.py --sign-weights path/to/ITSD/best_Fold1.pt --night-boost
```

- Only target classes are drawn (orange boxes): stop (`d-xx-1`, COCO `stop sign`), ITSD speed limits `d-r-30`..`d-r-39`, junction (`w-r- intersection`, `junction_<type>`) and route boards (`r-g-k`, logged only). Custom checkpoints may also use `speed_<km/h>`.
- Speed readings are limited to 10-50 km/h (`SPEED_MIN_KMH`/`SPEED_MAX_KMH` in `adas/road_signs.py`), so ITSD limits above 50 are ignored.
- A sign is reported only after it appears in 3 consecutive frames; each newly confirmed sign is printed once as `[SIGN] confirmed: ...` and summarised on the HUD.
- `--night-boost` applies CLAHE to the lightness channel when the frame mean is below 60 (0-255). The HUD shows `NIGHT BOOST` while it is active.

These are class-name readings from a detector. Sign text and route destinations are not read, and no vehicle command is produced.

## Labelled-dataset accuracy evaluation

Use `EvaluateModel.py` to validate a checkpoint against ground-truth annotations. This reports precision, recall, F1 and mAP metrics overall and per class, plus the lowest-F1 validation images.

Run on the validation split:

```bash
python EvaluateModel.py --weights runs/pedestrian/yolo26n-pedestrian-attributes/weights/best.pt
```

Run the final evaluation on a held-out test split:

```bash
python EvaluateModel.py --weights runs/pedestrian/yolo26n-pedestrian-attributes/weights/best.pt --split test
```

Useful options include `--imgsz`, `--batch`, `--device`, `--conf`, `--iou`, `--top-images`, and `--output`. The default JSON report is written under `runs/evaluation/`. Keep the test split untouched until model selection and tuning are complete.

## JAAD pedestrian attributes and video evaluation

Keep the JAAD clips you want to use in the repository's `videos/` directory. The scripts process only the clips you place there; they do not download or select videos automatically.

### 1. Get the JAAD annotations

The videos alone provide no ground-truth demographic labels. Download the matching JAAD_2.0 XML annotation files from the official annotation repository:

```powershell
git clone --depth 1 --branch JAAD_2.0 https://github.com/ykotseruba/JAAD.git jaad_annotations
```

The converter needs these two directories:

- `jaad_annotations/annotations/` — per-frame pedestrian boxes and track IDs.
- `jaad_annotations/annotations_attributes/` — per-pedestrian age and gender.

### 2. Convert your selected clips to YOLO training data

```powershell
python tools/prepare_jaad_dataset.py `
  --videos-dir videos `
  --annotations-dir jaad_annotations/annotations `
  --attributes-dir jaad_annotations/annotations_attributes `
  --output-dir datasets/jaad_pedestrian `
  --frame-stride 5 `
  --val-ratio 0.2
```

The converter matches filenames such as `videos/video_0001.mp4` to `video_0001.xml`, samples one frame every five video frames, converts the pedestrian boxes to YOLO labels, and splits whole videos into train/validation sets. Splitting by video prevents neighbouring frames from the same clip appearing in both sets. At least two selected clips with matching XML files are required.

JAAD age/gender annotations map to these classes:

| YOLO ID | Class | JAAD rule |
|---:|---|---|
| 0 | `child` | age = child |
| 1 | `woman` | gender = female, except child/senior |
| 2 | `elderly_person` | age = senior |
| 3 | `person_with_wheelchair` | reserved; JAAD does not supply wheelchair boxes |
| 4 | `other_person` | other pedestrians or missing demographic attributes |

JAAD provides demographic attributes only for its behavior-annotated pedestrian subset. Other pedestrian boxes are retained as `other_person`, rather than guessing age or gender. The wheelchair class will not be learned from JAAD alone; add a separately annotated wheelchair dataset before expecting this class to work.

The converter writes `datasets/jaad_pedestrian/pedestrian_attributes.yaml` and `conversion_summary.json`.

### 3. Train the pedestrian attribute detector

```powershell
python train_pedestrian_model.py `
  --data datasets/jaad_pedestrian/pedestrian_attributes.yaml `
  --model yolo26n.pt `
  --epochs 100 `
  --imgsz 640 `
  --batch 8 `
  --device 0 `
  --name jaad-pedestrian-attributes
```

The resulting custom checkpoint is normally:

```text
runs/pedestrian/jaad-pedestrian-attributes/weights/best.pt
```

### 4. Run all perception components on a selected JAAD clip

```powershell
python RiskVideoBenchmark.py `
  --source videos/video_0001.mp4 `
  --weights yolo26n.pt `
  --pedestrian-weights runs/pedestrian/jaad-pedestrian-attributes/weights/best.pt `
  --tracker bytetrack.yaml `
  --device 0 `
  --quantize fp16 `
  --show `
  --ui-scale 1.3
```

The runner uses two independent detectors:

- **COCO YOLO26:** cars (class 2), buses (5), trucks (7), and traffic lights (9).
- **JAAD-trained pedestrian model:** child, woman, elderly person, wheelchair user (only after separate training data), and other person.

The models have separate tracker ID and class-ID namespaces before their detections are combined. This prevents custom class ID 2 (elderly person) from being interpreted as COCO class 2 (car).

Without `--pedestrian-weights`, COCO only detects a generic `person`; it cannot identify child/woman/senior labels. Traffic lights are detected by the pretrained COCO model. JAAD's traffic annotations are frame-level presence labels, not traffic-light bounding boxes, so they are not converted into fabricated detection boxes.

### 5. Lane detection

The video runner also estimates the ego-lane corridor using a classical OpenCV baseline: white/yellow marking cues, Canny edges, a road-region mask, and Hough line fitting. The dashboard reports `DETECTED`, `TRACKED` (briefly held from recent frames), or `NOT_DETECTED`.

When both lane boundaries are found, the detected polygon is supplied to the collision-risk estimator. If lane markings are not visible, the system explicitly reports that lane detection failed and uses the fallback image ROI for risk estimation. That fallback is not the actual lane. The current lane detector is a research baseline, not a trained lane-segmentation network; it can fail on curves, intersections, worn markings, shadows, and rain.

### 6. Batch-run your hand-picked videos

```powershell
.\tools\run_jaad_videos.ps1 `
  -VideosDir videos `
  -PedestrianWeights runs/pedestrian/jaad-pedestrian-attributes/weights/best.pt `
  -Device 0 `
  -Quantize fp16
```

Add `-Show` to display each clip while it runs. Results are written separately for each selected video under `runs/jaad_results/`.

Each run writes an annotated MP4, per-frame CSV and JSON summary. The CSV includes lane state/confidence as well as detections, tracking, object risk, and TTC estimates. Use the CSV/JSON files to compare clips; inference FPS and risk labels are not accuracy metrics by themselves.

## Live testing with webcams and in-car cameras

`RiskVideoBenchmark.py` accepts local video paths, USB camera indices, and RTSP/HTTP network streams. On Windows it tries multiple capture backends automatically: Media Foundation (MSMF), DirectShow, then OpenCV's automatic backend. An explicitly selected backend is tried first, with fallbacks if it fails.

### 1. Find the correct camera index

Do not assume the laptop webcam is index 0. First close Windows Camera, Teams, Zoom, Discord, browsers and other apps that may be holding the camera, then run:

```powershell
python tools/list_cameras.py --max-index 8 --backends msmf dshow auto --preview
```

The script prints the working device index, backend, resolution and reported FPS. Press Q in a preview to probe the next index. If it finds the laptop webcam at index 1, use `--source 1`.

If no camera is found:

- Open the Windows Camera app and verify the webcam itself works.
- Go to Settings → Privacy & security → Camera.
- Turn on Camera access and Let desktop apps access your camera.
- Check the physical camera shutter / Fn camera switch and Device Manager → Cameras.
- Close any app that might already be using the webcam.

### 2. Run the laptop webcam

Use the index found by `tools/list_cameras.py` (example index 0):

```powershell
python RiskVideoBenchmark.py --source 0 --weights yolo26n.pt --device 0 --quantize fp16 --camera-backend auto --capture-width 1280 --capture-height 720 --max-seconds 120 --show --ui-scale 1.3
```

The runner automatically tries MSMF, DirectShow and OpenCV's automatic backend. Its startup output reports the backend that actually opened the camera.

### 3. Use an Android phone with DroidCam

**Recommended: use the current DroidCam Windows client as a virtual webcam.**

1. Install the phone app and the Windows client from [DroidCam for Windows](https://droidcam.app/windows/).
2. Connect the phone and laptop to the same Wi-Fi network.
3. Open DroidCam on the phone.
4. In the Windows client, click the centre or right-click and choose Add → DroidCam, refresh the device list, then add the phone.
5. Confirm the phone picture appears in the client. Windows apps should then see a webcam named `DroidCam Video`.
6. Find its OpenCV index:

```powershell
python tools/list_cameras.py --max-index 12 --backends msmf dshow auto --preview
```

7. Use the index printed for DroidCam. For example, if it is index 2:

```powershell
python RiskVideoBenchmark.py --source 2 --weights yolo26n.pt --device 0 --quantize fp16 --camera-backend auto --capture-width 1280 --capture-height 720 --max-seconds 300 --show --ui-scale 1.3
```

DroidCam's current Windows client documents adding the phone from the client and exposing it to applications as `DroidCam Video`. The phone and PC must be on the same network for Wi-Fi discovery. See [DroidCam Windows setup](https://droidcam.app/windows/).

### 4. DroidCam direct IP/MJPEG stream (Android)

If your DroidCam app has an **IP Cam** mode that displays a phone IP address and port, you can bypass the virtual webcam driver:

1. Start IP Cam mode on the phone.
2. From the laptop, open `http://PHONE_IP:PORT` in a browser to check that the stream is reachable.
3. Use the MJPEG endpoint `/video` as the OpenCV source:

```powershell
python RiskVideoBenchmark.py --source "http://192.168.1.50:4747/video" --weights yolo26n.pt --device 0 --quantize fp16 --camera-backend ffmpeg --max-seconds 300 --show --ui-scale 1.3
```

Replace the example IP and port with the values displayed by your phone. If you have DroidCamX and the app supports the resolution override, its documented form is `http://PHONE_IP:PORT/video/force/1280x720`. The IP Cam / MJPEG method is documented for the Android classic app; use the virtual webcam route if your app does not show IP Cam mode.

### 5. DroidCam over USB

For the classic Android DroidCam USB connection, enable USB debugging in Android Developer Options, connect with a data-capable cable, install the phone manufacturer's USB/ADB drivers if needed, approve the USB-debugging prompt, then choose USB in the DroidCam PC client and refresh the device list. The phone may need USB mode set to PTP, Camera, or Charging. After it appears as a Windows camera, run `tools/list_cameras.py` and use the detected index.

### 6. In-car camera inputs

- USB UVC camera: use the index reported by `tools/list_cameras.py`.
- DroidCam phone: use its virtual-camera index or its IP/MJPEG URL.
- IP camera: use the actual RTSP or HTTP URL supplied by that camera.

For RTSP/HTTP streams, use `--camera-backend ffmpeg` or `auto`. A sample URL is only a template; camera models and firmware use different stream paths. Do not publish URLs containing camera credentials.

Each live run accepts `--max-seconds` to stop automatically and saves an MP4, CSV and JSON under `runs/live_camera/`. Press Q or Esc to stop the preview early.

**One feed per process:** this runner tests one camera source at a time; `SurroundCameraBenchmark.py` launches five independent workers (see below). It does not yet synchronize multiple cameras or fuse them into a shared vehicle coordinate system. For lane detection and ego-lane risk estimation, use a forward-facing camera mounted securely and aligned with the vehicle. This is a research prototype; do not interact with the phone or laptop while driving.

## Fixed-video inference and benchmark

`VideoBenchmark.py` runs a trained YOLO checkpoint on a local video file. It saves an annotated MP4, a per-frame CSV, and a summary JSON by default.

Basic run:

```bash
python VideoBenchmark.py --source videos/test.mp4 --weights runs/pedestrian/yolo26n-pedestrian-attributes/weights/best.pt
```

Useful options:

```bash
# Display the annotated video while it runs
python VideoBenchmark.py --source videos/test.mp4 --weights path/to/best.pt --show

# Use FP16 on an NVIDIA GPU
python VideoBenchmark.py --source videos/test.mp4 --weights path/to/best.pt --device 0 --quantize fp16

# Test only a subset of class IDs (example: child and wheelchair user)
python VideoBenchmark.py --source videos/test.mp4 --weights path/to/best.pt --classes 0 3

# Run a shorter measured sample, excluding 5 warm-up frames
python VideoBenchmark.py --source videos/test.mp4 --weights path/to/best.pt --warmup 5 --max-frames 500

# Override output locations
python VideoBenchmark.py --source videos/test.mp4 --weights path/to/best.pt \
  --output output/annotated.mp4 --csv output/frames.csv --json output/summary.json
```

### Reported metrics

- **Average / median / P95 inference latency (ms):** time spent in the YOLO prediction call, including preprocessing and postprocessing. CUDA is synchronized before and after each timed call.
- **Inference FPS:** reciprocal of average model-call latency. It is not the same as full application throughput.
- **End-to-end processing FPS:** measured frames divided by elapsed processing time, including video decoding, annotation and output writing.
- **Real-time factor:** processing FPS divided by the source video's FPS. Values above 1.0 mean processing faster than the source playback rate.
- **Detections by class:** number of predicted boxes per class, plus per-frame class counts in the CSV.

Warm-up frames are excluded from the benchmark statistics but remain in the annotated output video. Use `--show` only for visual inspection because GUI display can reduce throughput. For fair model comparisons, use the same video, image size, confidence/IoU thresholds, device, precision and warm-up count, and benchmark models separately.

**Detection counts and FPS are not accuracy metrics.** To calculate precision, recall, or mAP, you also need ground-truth annotations for the benchmark video and an evaluation procedure that matches predictions to those labels. A video without ground truth can benchmark speed and show qualitative detections, but cannot establish detection accuracy.


## Five-camera side and rear warnings

The camera layout is `front`, `left`, `right`, `rear_left`, and `rear_right`.
The default `front` mode keeps the existing lane detector, fallback corridor,
class filters and collision-risk decisions. Existing front commands still work.

The four new modes use camera-local **image warning regions**, without running
front lane detection or front cut-in prediction:

- **Left/right:** CAUTION when a tracked pedestrian, bicycle, motorcycle, car,
  bus or truck occupies the warning region, even without measurable expansion.
- **Rear-left/rear-right:** occupancy warnings plus apparent-approach escalation.
- All four views escalate to HIGH at apparent TTC <=3s and CRITICAL at <=1.5s.
  TTC comes from bounding-box expansion, not measured range or closing speed.
- Objects outside the region and non-obstacle classes do not trigger warnings.
  A missing track removes its current warning; an empty region is not proof that
  a lane change is safe. Track history expires after one second of absence.

### Run a single side/rear camera

Find the actual device indices with `tools/list_cameras.py`, then for example:

```powershell
python RiskVideoBenchmark.py --source 1 --camera-position left --weights yolo26n.pt --max-seconds 60 --show
```

Video paths and RTSP/HTTP sources work the same way. `--pedestrian-weights`
continues to support the custom pedestrian checkpoint. Each MP4, CSV and JSON is
labelled with its camera position. CSV object records retain the existing
`in_ego_corridor` field; in surround modes this means **inside warning ROI**.
Lane status is `NOT_APPLICABLE` for side/rear views.

### Run all five cameras

Copy `data/surround_cameras.example.json` to `data/surround_cameras.json` and edit
its source assignments to match your hardware. The example indices 0–4 are
placeholders, not discovered devices. Each source can be an integer index, URL or
video filename; relative video paths are resolved relative to the config file.
Then run:

```powershell
python SurroundCameraBenchmark.py --config data/surround_cameras.json --weights yolo26n.pt --device 0 --quantize fp16 --max-seconds 60 --show
```

Model, capture and display flags are forwarded to every worker. Each worker has
its own detector, tracker and risk history, so identical track IDs from different
cameras cannot mix. Outputs go to a timestamped directory under `runs/surround/`,
with separate `front`, `left`, `right`, `rear_left` and `rear_right` MP4/CSV/JSON
files. Use `--output-dir` to choose a different destination. A failed worker stops
the launcher and remaining workers with a nonzero exit code.

Five model instances (ten with the custom pedestrian detector) need more GPU
memory and compute. Start with one camera, then lower `--imgsz` or capture
resolution if necessary. Use `--max-seconds` for live sources or `--max-frames`
for files to finish cleanly and write reports. Q/Esc finishes a preview worker;
Ctrl+C stops the launcher but forcibly stopped workers may not finish reports.

### Adjust the warning regions for your mounts

The default polygon covers x=0.05–0.95 and y=0.30–1.0 in normalized image
coordinates. Left and right use the same default because image orientation and
mirroring depend on your camera installation. This region must be adjusted for
the actual view; it is not a calibrated physical blind spot.

Save a JSON polygon such as `data/left_roi.json`:

```json
[[0.2, 0.3], [0.8, 0.3], [0.95, 1.0], [0.05, 1.0]]
```

For a single camera add `--warning-roi data/left_roi.json`. For the launcher add
`"warning_roi": "left_roi.json"` to that camera's config entry (resolved relative
to the config file). The front camera rejects this override to preserve its
existing lane logic.

The five feeds are processed independently, without synchronized timestamps,
cross-camera object fusion, steering, braking or lane-change authorization.
These are advisory university-project annotations. Camera motion, occlusion and
tracking errors affect the outputs; validate the ROI and behavior on recordings
from each mount. Automated tests cover occupancy, approach/recession, ROI scaling,
custom pedestrian IDs, track isolation/expiry, launcher configuration and all
five runner modes using synthetic videos and a detector stand-in. They do not
establish real-camera detection accuracy or five-camera GPU throughput.
