import math
import unittest

import numpy as np

from adas.lane_detection import LaneDetector
from adas.risk_assessment import CAUTION, CLEAR, HIGH, CollisionRiskAssessor


class CollisionRiskAssessorTests(unittest.TestCase):
    def test_corridor_is_resolution_independent(self):
        assessor = CollisionRiskAssessor()
        self.assertTrue(assessor.point_in_polygon((0.5, 0.8), assessor.ego_corridor))
        small = assessor.update(
            [(280, 100, 360, 300, 1, 0.9, 0)], (400, 640), timestamp=1.0
        )
        large = assessor.update(
            [(560, 200, 720, 600, 1, 0.9, 0)], (800, 1280), timestamp=1.1
        )
        self.assertEqual(
            small.objects[0].in_ego_corridor,
            large.objects[0].in_ego_corridor,
        )

    def test_expansion_estimates_ttc_and_sets_risk(self):
        assessor = CollisionRiskAssessor(min_history_seconds=0.25)
        report = None
        for index in range(8):
            timestamp = index * 0.1
            box_width = 100.0 * math.exp(0.5 * timestamp)
            center_x = 640.0
            report = assessor.update(
                [(
                    center_x - box_width / 2, 300,
                    center_x + box_width / 2, 620, 7, 0.95, 0
                )],
                (720, 1280),
                timestamp=timestamp,
            )
        self.assertIsNotNone(report)
        self.assertAlmostEqual(report.objects[0].ttc_seconds, 2.0, delta=0.15)
        self.assertEqual(report.level, HIGH)

    def test_non_expanding_object_has_no_ttc(self):
        assessor = CollisionRiskAssessor(min_history_seconds=0.2)
        report = None
        for index in range(7):
            report = assessor.update(
                [(590, 300, 690, 620, 2, 0.9, 0)],
                (720, 1280),
                timestamp=index * 0.1,
            )
        self.assertIsNone(report.objects[0].ttc_seconds)
        self.assertEqual(report.level, CLEAR)

    def test_custom_pedestrian_class_ids_override_coco_ids(self):
        assessor = CollisionRiskAssessor(pedestrian_class_ids=range(5))
        report = assessor.update(
            [(500, 300, 780, 650, 1, 0.9, 3)],
            (720, 1280),
            timestamp=1.0,
        )
        self.assertEqual(report.objects[0].object_type, "pedestrian")

    def test_cut_in_prediction_generates_caution(self):
        assessor = CollisionRiskAssessor(min_history_seconds=0.2)
        report = None
        for index in range(7):
            center_x = 0.99 - (0.01 * index)
            report = assessor.update(
                [(
                    center_x * 1280 - 25, 400,
                    center_x * 1280 + 25, 620, 3, 0.9, 0
                )],
                (720, 1280),
                timestamp=index * 0.1,
            )
        self.assertEqual(report.level, CAUTION)
        self.assertIsNotNone(report.objects[0].cut_in_eta_seconds)

    def test_stale_track_history_is_discarded(self):
        assessor = CollisionRiskAssessor(min_history_seconds=0.2, stale_track_seconds=0.5)
        report = None
        for index in range(7):
            timestamp = index * 0.1
            box_width = 100.0 * math.exp(0.5 * timestamp)
            report = assessor.update(
                [(640 - box_width / 2, 300, 640 + box_width / 2, 620, 9, 0.95, 0)],
                (720, 1280),
                timestamp=timestamp,
            )
        self.assertIsNotNone(report.objects[0].ttc_seconds)

        report = assessor.update(
            [(500, 300, 600, 620, 9, 0.95, 0)],
            (720, 1280),
            timestamp=2.0,
        )
        self.assertIsNone(report.objects[0].ttc_seconds)

    def test_dynamic_lane_corridor_overrides_default_roi(self):
        assessor = CollisionRiskAssessor()
        left_corridor = (
            (0.05, 0.40), (0.40, 0.40), (0.40, 0.95), (0.05, 0.95)
        )
        report = assessor.update(
            [(590, 300, 690, 620, 42, 0.9, 0)],
            (720, 1280),
            timestamp=1.0,
            ego_corridor=left_corridor,
        )
        self.assertFalse(report.objects[0].in_ego_corridor)
        self.assertEqual(report.corridor, left_corridor)

    def test_lane_detector_does_not_invent_lane_on_blank_frame(self):
        detector = LaneDetector(max_hold_frames=0)
        estimate = detector.detect(np.zeros((720, 1280, 3), dtype=np.uint8))
        self.assertEqual(estimate.status, "NOT_DETECTED")
        self.assertIsNone(estimate.corridor)

    def test_lane_detector_finds_synthetic_left_and_right_markings(self):
        import cv2

        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        cv2.line(frame, (470, 410), (180, 695), (255, 255, 255), 12)
        cv2.line(frame, (810, 410), (1100, 695), (255, 255, 255), 12)
        estimate = LaneDetector(max_hold_frames=0).detect(frame)
        self.assertEqual(estimate.status, "DETECTED")
        self.assertIsNotNone(estimate.corridor)
        self.assertLess(estimate.corridor[0][0], estimate.corridor[1][0])
        self.assertLess(estimate.corridor[3][0], estimate.corridor[2][0])

    def test_lane_detector_uses_short_track_when_markings_disappear(self):
        import cv2

        detector = LaneDetector(max_hold_frames=2)
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        cv2.line(frame, (470, 410), (180, 695), (255, 255, 255), 12)
        cv2.line(frame, (810, 410), (1100, 695), (255, 255, 255), 12)
        first = detector.detect(frame)
        self.assertEqual(first.status, "DETECTED")

        missing = detector.detect(np.zeros_like(frame))
        self.assertEqual(missing.status, "TRACKED")
        self.assertIsNotNone(missing.corridor)

        detector.detect(np.zeros_like(frame))
        expired = detector.detect(np.zeros_like(frame))
        self.assertEqual(expired.status, "NOT_DETECTED")


if __name__ == "__main__":
    unittest.main()
