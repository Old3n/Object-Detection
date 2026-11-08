"""Exercise real video I/O and reports with a deterministic detector stand-in."""

import contextlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import cv2
import numpy as np


class Tensor:
    def __init__(self, values):
        self.values = values

    def int(self):
        return self

    float = cpu = int

    def tolist(self):
        return self.values


class FakeYOLO:
    names = {0: "person", 2: "car"}

    def __init__(self, weights):
        pass

    def track(self, frame, **kwargs):
        class Boxes:
            cls = Tensor([2])
            id = Tensor([1])
            conf = Tensor([.95])
            xyxy = Tensor([[500, 300, 700, 650]])

            def __len__(self):
                return 1

        return [types.SimpleNamespace(boxes=Boxes())]


class RunnerTests(unittest.TestCase):
    def test_all_camera_modes_produce_video_csv_and_position_specific_reports(self):
        root = Path(__file__).resolve().parents[1]
        torch = types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: False))
        with patch.dict(sys.modules, {
            "torch": torch, "ultralytics": types.SimpleNamespace(YOLO=FakeYOLO),
        }):
            spec = importlib.util.spec_from_file_location("risk_runner_under_test", root / "RiskVideoBenchmark.py")
            runner = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(runner)
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            source = directory / "source.mp4"
            writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"mp4v"), 10, (1280,720))
            self.assertTrue(writer.isOpened())
            for _ in range(5):
                writer.write(np.zeros((720,1280,3), dtype=np.uint8))
            writer.release()
            for position in ("front", "left", "right", "rear_left", "rear_right"):
                output = directory / f"{position}.mp4"
                summary_path = output.with_suffix(".json")
                argv = ["RiskVideoBenchmark.py", "--source", str(source), "--weights", "fake.pt",
                        "--camera-position", position, "--output", str(output),
                        "--json", str(summary_path), "--csv", str(output.with_suffix('.csv'))]
                with self.subTest(position=position), patch.object(sys, "argv", argv), contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(runner.main(), 0)
                    summary = json.loads(summary_path.read_text())
                    self.assertEqual(summary["frames"], 5)
                    self.assertEqual(summary["camera_position"], position)
                    self.assertEqual(summary["maximum_risk_observed"], "CLEAR" if position == "front" else "CAUTION")
                    self.assertEqual(summary["lane_status_frames"], {"NOT_DETECTED" if position == "front" else "NOT_APPLICABLE": 5})
                    self.assertIn("camera_position", output.with_suffix(".csv").read_text().splitlines()[0])
                    capture = cv2.VideoCapture(str(output))
                    self.assertTrue(capture.read()[0])
                    capture.release()


if __name__ == "__main__":
    unittest.main()
