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

## Collision-risk testing on recorded video

Use `RiskVideoBenchmark.py` to test the complete detection → tracking → risk-estimation pipeline on a recorded drive or simulator recording. Unlike `VideoBenchmark.py`, this script uses persistent tracker IDs and feeds the tracks into the collision-risk estimator.

For a COCO-pretrained YOLO checkpoint:

```bash
python RiskVideoBenchmark.py \
  --source videos/drive.mp4 \
  --weights yolo26n.pt \
  --classes 0 1 2 3 5 7 \
  --pedestrian-class-ids 0 \
  --vehicle-class-ids 1 2 3 5 7 \
  --tracker bytetrack.yaml
```

For the custom five-class pedestrian checkpoint:

```bash
python RiskVideoBenchmark.py \
  --source videos/drive.mp4 \
  --weights runs/pedestrian/yolo26n-pedestrian-attributes/weights/best.pt \
  --classes 0 1 2 3 4 \
  --pedestrian-class-ids 0 1 2 3 4
```

The custom pedestrian checkpoint does not detect vehicles, so this run evaluates pedestrian risk only. Add a separate vehicle detector or a unified trained class set for mixed pedestrian/vehicle risk assessment.

By default the runner writes:

- `<video>_risk.mp4`: annotated detections, track IDs, risk levels and the configured ego corridor.
- `<video>_risk.csv`: frame timestamp, inference/processing latency, risk level, reason and per-object estimates.
- `<video>_risk.json`: aggregate latency, FPS and risk-frame counts.

Use `--show` for live display, `--max-frames N` for a shorter run, `--device 0 --half` for CUDA FP16, or `--no-save-video` to skip MP4 output. Risk TTC uses video frame timestamps derived from the file FPS; recordings with dropped or duplicated frames can therefore distort estimates.

## CARLA simulation workflow

CARLA is a good next test environment because you can repeat scenarios and vary traffic, weather, lighting and actor motion. Start with offline recordings before connecting the detector to a live simulation loop:

1. Start CARLA and spawn an ego vehicle with a front-facing RGB camera.
2. Enable synchronous mode and a fixed simulation step (for example, `fixed_delta_seconds = 0.05`, or 20 simulation frames per second). Keep exactly one client responsible for ticking the world.
3. Record the camera frames to an MP4 at the same nominal FPS as the fixed simulation step. Preserve the scenario settings and recording metadata.
4. Run the MP4 through `RiskVideoBenchmark.py` using the same model and thresholds for every scenario.
5. Repeat with controlled scenarios: lead vehicle braking, pedestrian crossing, vehicle cut-in, parked vehicles, intersections, night and rain. Compare risk-level timing and false alerts across scenarios.

CARLA documentation:
- [Synchrony and fixed time-step](https://carla.readthedocs.io/en/latest/adv_synchrony_timestep/)
- [ROS bridge sensor support](https://carla.readthedocs.io/en/latest/ros_documentation/)

CARLA's synchronous mode with a fixed step is important for repeatable timing. Camera sensors can arrive with rendering delay, so use frame IDs/timestamps and a queue when capturing multiple sensors. If you later connect through ROS 2, use simulation time and make sure only one client advances the world.

### What this test does and does not prove

This verifies that the detector, tracker and risk estimator execute on recorded scenarios, and lets you inspect the timing and warnings. It does **not** yet calculate risk-estimation accuracy against CARLA ground truth. For a proper research evaluation, the next addition should compare predictions against CARLA actor transforms/velocities and collision events, reporting TTC error, event recall, false-positive rate and warning lead time. A front-camera video alone cannot provide that ground truth.

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
