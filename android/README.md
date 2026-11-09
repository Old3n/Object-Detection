# AutoPilot Live (Android)

On-device port of the live pipeline for street tests on a phone or an Android head unit. No backend or internet is needed while driving.

Per frame, in the same order as `RiskVideoBenchmark.py` / `ModelBenchmark.py`:

1. Night boost: CLAHE on LAB lightness when the frame mean is below 60 (`adas/road_signs.py`).
2. COCO detection (`yolo26n`): person, bicycle, car, motorcycle, bus, truck, traffic light.
3. Tracking: greedy IoU tracker.
4. Lane detection: port of `adas/lane_detection.py`.
5. Collision risk: port of `adas/risk_assessment.py` (TTC from box expansion, corridor entry).
6. Iranian road signs: ITSD checkpoint with 3-frame confirmation (`adas/road_signs.py`).

Both models run on the CPU through ONNX Runtime with XNNPACK (ARM NEON).

## Speed modes

| Mode | Model input | Use |
|---|---|---|
| Balanced 480 (default) | 480 px | Car head units and older phones. On the 42 demo photos it found signs in 36 (640: 35) and 24 people (640: 32). |
| Accurate 640 | 640 px | Matches the desktop setting. Use on recent phones. |

On both modes:

- The sign model runs on every 2nd frame. Confirmation still needs 3 sign-model runs, which is 6 camera frames.
- The camera is requested at 640x480, and frames are processed at most 640 px wide.

320 px was tested and rejected: it found signs in only 14 of the 42 photos and 14 people.

## Differences from the desktop version

- The tracker is a simple IoU tracker, not ByteTrack. It has no motion model, so IDs switch more often during fast motion.
- If the lane corridor falls slightly outside the frame, it is clamped to 0..1. The desktop assessor raises an error here instead.
- Speed-limit readings keep the desktop 10-50 km/h filter, so ITSD limits of 60 km/h and above are ignored.
- The output is advisory only. The app sends no command to the vehicle.

## Build

Requirements: JDK 17, Android SDK 35, Gradle 9.x. AGP 9 compiles Kotlin itself.

```bash
python android/export_models.py --coco yolo26n.pt --signs path/to/ITSD/best_Fold1.pt
cd android
echo "sdk.dir=$HOME/Library/Android/sdk" > local.properties
gradle testDebugUnitTest assembleRelease
```

One APK is built per CPU type in `app/build/outputs/apk/release/`:

- `app-armeabi-v7a-release.apk`: 32-bit ARM, which covers most car head units.
- `app-arm64-v8a-release.apk`: modern phones.
- `app-x86_64-release.apk`: emulator only.

The APKs are signed with the debug key so they install directly. Do not publish them to a store.

## Install

```bash
adb shell getprop ro.product.cpu.abi     # pick the matching APK
adb shell getprop ro.build.version.release
adb install -r app/build/outputs/apk/release/app-armeabi-v7a-release.apk
```

You can also copy the APK to the device and open it there. Allow installs from this source when Android asks.

Minimum Android version: 5.0 (API 21). This is the lowest version CameraX, ONNX Runtime and OpenCV support. Android 2.x and 4.x cannot run this app.
