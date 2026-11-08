#!/usr/bin/env python3
"""Run the five camera views in independent tracking processes.

No frame synchronization, cross-camera association or vehicle control is implied.
"""

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

from adas.surround_risk import SURROUND_POSITIONS

POSITIONS = ("front",) + SURROUND_POSITIONS


def build_commands(config_path: Path, output_dir: Path, extra_args: list[str]) -> list[list[str]]:
    """Validate sources and build commands without opening devices or loading YOLO."""
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or set(config) != set(POSITIONS):
        raise ValueError(f"Config must define exactly these views: {', '.join(POSITIONS)}")
    reserved = {"--source", "--camera-position", "--warning-roi", "--output", "--csv", "--json"}
    if any(arg.split("=", 1)[0] in reserved for arg in extra_args):
        raise ValueError("Sources, positions, ROIs and output paths must come from the camera config.")
    commands = []
    device_sources = set()
    for position in POSITIONS:
        entry = config[position]
        if not isinstance(entry, dict) or "source" not in entry or set(entry) - {"source", "warning_roi"}:
            raise ValueError(f"{position}: expected source and optional warning_roi")
        source = entry["source"]
        if isinstance(source, bool) or not isinstance(source, (str, int)) or not str(source).strip():
            raise ValueError(f"{position}: source must be a camera index, URL or video path")
        source = str(source).strip()
        if source.isdigit():
            source = str(int(source))
            if source in device_sources:
                raise ValueError(f"USB camera index {source} is assigned to multiple views")
            device_sources.add(source)
        elif not source.lower().startswith(("rtsp://", "rtsps://", "http://", "https://")):
            path = Path(source).expanduser()
            source = str((config_path.parent / path).resolve())
            if not Path(source).is_file():
                raise FileNotFoundError(f"{position}: video not found: {source}")
        stem = output_dir.resolve() / position
        command = [
            sys.executable, str(Path(__file__).with_name("RiskVideoBenchmark.py")),
            "--source", source, "--camera-position", position,
            "--output", str(stem.with_suffix(".mp4")),
            "--csv", str(stem.with_suffix(".csv")),
            "--json", str(stem.with_suffix(".json")),
        ]
        if "warning_roi" in entry:
            if position == "front":
                raise ValueError("Front ROI override is not supported; the existing lane logic is preserved")
            roi_path = (config_path.parent / entry["warning_roi"]).resolve()
            from adas.surround_risk import SurroundRiskAssessor

            SurroundRiskAssessor(position, warning_roi=json.loads(roi_path.read_text(encoding="utf-8")))
            command += ["--warning-roi", str(roi_path)]
        commands.append(command + extra_args)
    return commands


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path, default=None)
    args, extra = parser.parse_known_args()
    output_dir = args.output_dir or Path("runs/surround") / time.strftime("%Y%m%d_%H%M%S")
    commands = build_commands(args.config.expanduser().resolve(), output_dir, extra)
    output_dir.mkdir(parents=True, exist_ok=True)
    processes = []
    try:
        for position, command in zip(POSITIONS, commands):
            print(f"[INFO] Starting {position}", flush=True)
            processes.append((position, subprocess.Popen(command)))
        while True:
            statuses = [(position, process.poll()) for position, process in processes]
            failed = [(position, code) for position, code in statuses if code not in (None, 0)]
            if failed:
                print(f"[ERROR] Camera workers failed: {failed}", file=sys.stderr)
                return 1
            if all(code is not None for _, code in statuses):
                return 0
            time.sleep(0.1)
    except KeyboardInterrupt:
        return 130
    finally:
        for _, process in processes:
            if process.poll() is None:
                process.terminate()
        for _, process in processes:
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, TypeError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        raise SystemExit(1)
