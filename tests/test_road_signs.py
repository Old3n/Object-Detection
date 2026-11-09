import unittest

import numpy as np

from adas.road_signs import SignConfirmer, boost_night, is_night, sign_kind, sign_text


class SignKindTests(unittest.TestCase):
    def test_known_names_map_to_readings(self):
        cases = {
            "stop sign": ("stop", None), "stop": ("stop", None), "d-xx-1": ("stop", None),
            "d-r-30": ("speed", 30), "speed_40": ("speed", 40),
            "junction_crossroad": ("junction", None), "w-r- intersection": ("junction", None),
            "r-g-k": ("route", None),
        }
        for name, expected in cases.items():
            with self.subTest(name=name):
                self.assertEqual(sign_kind(name), expected)

    def test_other_names_and_out_of_range_speeds_are_ignored(self):
        for name in ("car", "d-r-20", "speed_5", "speed_80", "d-r-33", "speed_x"):
            with self.subTest(name=name):
                self.assertIsNone(sign_kind(name))


class SignConfirmerTests(unittest.TestCase):
    def test_reading_needs_consecutive_frames(self):
        confirmer = SignConfirmer(frames=3)
        stop = ("stop", None)
        self.assertEqual(confirmer.update({stop}), set())
        self.assertEqual(confirmer.update({stop}), set())
        self.assertEqual(confirmer.update({stop, ("speed", 30)}), {stop})
        self.assertEqual(confirmer.update(set()), set())
        confirmer.reset()
        self.assertFalse(confirmer.history)

    def test_invalid_window_is_rejected(self):
        with self.assertRaises(ValueError):
            SignConfirmer(frames=0)

    def test_sign_text(self):
        self.assertEqual(sign_text(set()), "Signs: none")
        self.assertEqual(
            sign_text({("speed", 50), ("stop", None), ("speed", 30)}),
            "STOP | SPEED 30 | SPEED 50",
        )


class NightBoostTests(unittest.TestCase):
    def test_dark_frame_detected_and_contrast_raised(self):
        rng = np.random.default_rng(0)
        dark = rng.integers(10, 40, (120, 160, 3), dtype=np.uint8)
        self.assertTrue(is_night(dark))
        self.assertFalse(is_night(np.full_like(dark, 200)))
        boosted = boost_night(dark)
        self.assertEqual(boosted.shape, dark.shape)
        self.assertEqual(boosted.dtype, np.uint8)
        self.assertGreater(boosted.std(), dark.std())


if __name__ == "__main__":
    unittest.main()
