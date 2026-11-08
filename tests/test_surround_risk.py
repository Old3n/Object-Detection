import json
import math
from pathlib import Path
import tempfile
import unittest

from adas.risk_assessment import CAUTION, CLEAR, HIGH
from adas.surround_risk import SURROUND_POSITIONS, SurroundRiskAssessor
from SurroundCameraBenchmark import build_commands


class SurroundRiskTests(unittest.TestCase):
    def test_stationary_obstacle_warns_in_each_camera_without_ttc(self):
        for position in SURROUND_POSITIONS:
            with self.subTest(position=position):
                report = SurroundRiskAssessor(position).update(
                    [(400, 200, 600, 700, 1, .9, 2)], (720, 1280), timestamp=0,
                )
                self.assertEqual(report.level, CAUTION)
                self.assertIsNone(report.objects[0].ttc_seconds)
                self.assertIn(position, report.reason)

    def test_rear_approach_escalates_and_receding_object_does_not(self):
        for rate, expected in ((.5, HIGH), (-.5, CAUTION)):
            assessor = SurroundRiskAssessor("rear_left")
            for i in range(8):
                width = 100 * math.exp(rate * i * .1)
                report = assessor.update(
                    [(640-width/2, 200, 640+width/2, 650, 1, .9, 2)],
                    (720, 1280), timestamp=i*.1,
                )
            self.assertEqual(report.level, expected)
            if rate > 0:
                self.assertAlmostEqual(report.objects[0].ttc_seconds, 2.0)
            else:
                self.assertIsNone(report.objects[0].ttc_seconds)

    def test_unknown_classes_low_confidence_and_outside_roi_do_not_warn(self):
        assessor = SurroundRiskAssessor("right")
        report = assessor.update([
            (400, 100, 600, 700, 1, .9, 9),
            (400, 100, 600, 700, 2, .1, 2),
            (400, 10, 600, 100, 3, .9, 2),
        ], (720, 1280), timestamp=0)
        self.assertEqual(report.level, CLEAR)
        self.assertEqual(len(report.objects), 2)

    def test_roi_scales_and_custom_pedestrians_are_supported(self):
        roi = ((.2, .2), (.6, .2), (.6, 1.), (.2, 1.))
        for scale in (1, 2):
            assessor = SurroundRiskAssessor("left", warning_roi=roi, pedestrian_class_ids=range(100, 105))
            report = assessor.update(
                [(300*scale, 100*scale, 500*scale, 650*scale, 1, .9, 102)],
                (720*scale, 1280*scale), timestamp=0,
            )
            self.assertEqual(report.level, CAUTION)
            self.assertEqual(report.objects[0].object_type, "pedestrian")

    def test_camera_histories_are_independent_and_expire(self):
        left, right = SurroundRiskAssessor("left"), SurroundRiskAssessor("right")
        for i in range(8):
            w = 100 * math.exp(.5 * i * .1)
            left.update([(640-w/2, 200, 640+w/2, 650, 1, .9, 2)], (720, 1280), timestamp=i*.1)
        row = [(590, 200, 690, 650, 1, .9, 2)]
        self.assertIsNone(right.update(row, (720, 1280), timestamp=.8).objects[0].ttc_seconds)
        self.assertIsNone(left.update(row, (720, 1280), timestamp=2).objects[0].ttc_seconds)
        self.assertEqual(left.update([], (720, 1280), timestamp=4).level, CLEAR)
        self.assertFalse(left.estimator.history)

    def test_front_cut_in_projection_is_not_used(self):
        assessor = SurroundRiskAssessor("right", warning_roi=((.3,.3),(.7,.3),(.7,1),(.3,1)))
        for i in range(8):
            x = (1.0 - .03*i) * 1280
            report = assessor.update([(x-20, 200, x+20, 650, 1, .9, 2)], (720,1280), timestamp=i*.1)
        self.assertEqual(report.level, CLEAR)
        self.assertIsNone(report.objects[0].cut_in_eta_seconds)

    def test_invalid_position_and_roi_are_rejected(self):
        for position, roi in (("front", None), ("left", []), ("left", ((0,0),(2,0),(0,1)))):
            with self.assertRaises(ValueError):
                SurroundRiskAssessor(position, warning_roi=roi)


class LauncherTests(unittest.TestCase):
    def test_five_distinct_camera_commands_and_output_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cameras.json"
            positions = ("front",) + SURROUND_POSITIONS
            path.write_text(json.dumps({name: {"source": i} for i, name in enumerate(positions)}))
            commands = build_commands(path, Path(directory) / "out", ["--weights", "yolo26n.pt"])
            for i, (position, command) in enumerate(zip(positions, commands)):
                self.assertEqual(command[command.index("--source")+1], str(i))
                self.assertEqual(command[command.index("--camera-position")+1], position)
                self.assertEqual(Path(command[command.index("--csv")+1]).name, f"{position}.csv")
            with self.assertRaises(ValueError):
                build_commands(path, Path(directory), ["--source=0"])
            config = json.loads(path.read_text())
            config["right"]["source"] = 0
            path.write_text(json.dumps(config))
            with self.assertRaises(ValueError):
                build_commands(path, Path(directory), [])


if __name__ == "__main__":
    unittest.main()
