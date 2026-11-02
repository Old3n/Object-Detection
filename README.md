# AutoPilot - Live Object Tracking Benchmark + ADAS Braking

Real-time multi-model object tracking benchmark with live HUD comparison and a Level 1 ADAS braking decision module. Switch between YOLO26 variants (nano/small/medium) paired with ByteTrack, OC-SORT, and BoT-SORT trackers using hotkeys - all running on GPU with FP16 inference.

## Dependencies

```bash
pip install -U ultralytics boxmot opencv-python lapx
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
| `B` | Toggle ADAS Braking Decision Module |
| `R` | Reset metrics |
| `Q` | Quit |

## ADAS Braking Decision Module

Toggle with `S` (Street Mode) then `B` to enable. The module evaluates tracked objects and issues braking commands using three independent threat detectors:

### Threat Detection Rules

| Rule | Trigger | Command |
|------|---------|---------|
| **Proximity Brake** | Object bounding box fills >35% (pedestrian) or >45% (vehicle) of frame height while inside the ego-lane | `EMERGENCY BRAKE!` |
| **TTC Brake** | Bounding box growth rate yields Time-To-Collision < 2.0 seconds | `BRAKE - LOW TTC!` |
| **Cut-In Brake** | Object's lateral velocity predicts it will enter the ego-lane within 0.8 seconds | `BRAKE - CUT-IN!` |
| **Sign Detection** | Stop sign or traffic light detected in frame | `SLOW - SIGN AHEAD` |
| **Caution** | Object approaching but not yet at emergency threshold (>21% pedestrian / >27% vehicle) | `CAUTION - CLOSING` |

### Threat Levels

- **Level 0 (CRUISE)** - Small green pill indicator in top-right corner. No threats detected.
- **Level 1 (CAUTION)** - Orange banner. Object closing in or sign detected.
- **Level 2 (EMERGENCY)** - Full-width red banner with threat highlighted. Brake immediately.

### Visual Overlays

- **Yellow trapezoid** - Ego-lane projection zone
- **Red dashed line** - Critical proximity boundary
- **Red filled box** - Object triggering emergency brake
- **Orange box** - Object triggering caution
- **Orange arrow** - Predicted cut-in trajectory

### Design Notes

- Pedestrians trigger at a **lower threshold** than vehicles (brake sooner for people)
- Proximity detection is the **primary** brake trigger (simple, reliable, no history needed)
- TTC from bounding box growth is a **secondary** trigger (needs 5+ frames of tracking history)
- Only active in **Street Mode** - ego-lane detection does not apply to Webcam Desk Mode
- Expanded class list in braking mode adds traffic light (class 9) and stop sign (class 11)

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
python VideoBenchmark.py --source videos/test.mp4 --weights path/to/best.pt --device 0 --half

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
