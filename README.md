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