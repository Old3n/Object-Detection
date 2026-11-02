# AutoPilot - Object Detection, Tracking & ADAS Risk Estimation

Real-time multi-model object detection and tracking benchmark with an advisory, track-based collision-risk estimator. Compare YOLO26 variants paired with ByteTrack, OC-SORT, and BoT-SORT.

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
python tools/prepare_jaad_dataset.py \
  --videos-dir videos \
  --annotations-dir jaad_annotations/annotations \
  --attributes-dir jaad_annotations/annotations_attributes \
  --output-dir datasets/jaad_pedestrian \
  --frame-stride 5 \
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
python train_pedestrian_model.py \
  --data datasets/jaad_pedestrian/pedestrian_attributes.yaml \
  --model yolo26n.pt \
  --epochs 100 \
  --imgsz 640 \
  --batch 8 \
  --device 0 \
  --name jaad-pedestrian-attributes
```

The resulting custom checkpoint is normally:

```text
runs/pedestrian/jaad-pedestrian-attributes/weights/best.pt
```

### 4. Run all perception components on a selected JAAD clip

```powershell
python RiskVideoBenchmark.py \
  --source videos/video_0001.mp4 \
  --weights yolo26n.pt \
  --pedestrian-weights runs/pedestrian/jaad-pedestrian-attributes/weights/best.pt \
  --tracker bytetrack.yaml \
  --device 0 \
  --quantize fp16 \
  --show \
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
.\tools\run_jaad_videos.ps1 \
  -VideosDir videos \
  -PedestrianWeights runs/pedestrian/jaad-pedestrian-attributes/weights/best.pt \
  -Device 0 \
  -Quantize fp16
```

Add `-Show` to display each clip while it runs. Results are written separately for each selected video under `runs/jaad_results/`.

Each run writes an annotated MP4, per-frame CSV and JSON summary. The CSV includes lane state/confidence as well as detections, tracking, object risk, and TTC estimates. Use the CSV/JSON files to compare clips; inference FPS and risk labels are not accuracy metrics by themselves.

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
