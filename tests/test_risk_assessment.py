import math
import unittest

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


if __name__ == "__main__":
    unittest.main()
