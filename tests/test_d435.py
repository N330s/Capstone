import unittest

import numpy as np

from envs import d435

PARAMS = {"units_m": 0.001, "baseline_m": 0.050, "subpixel": 0.08, "depth_fx_px": 447.0,
          "min_range_m": 0.28, "max_range_m": 3.0, "edge_jump_m": 0.02, "edge_dilate_px": 1,
          "dropout_rate": 0.0}


class D435DepthTests(unittest.TestCase):
    def test_units_range_and_determinism(self):
        z = np.full((40, 60), 0.5)
        z[:, :5] = 0.2            # closer than min range
        z[:, -5:] = 4.0           # beyond max range
        a = d435.degrade_depth(z, np.random.default_rng(3), PARAMS)
        b = d435.degrade_depth(z, np.random.default_rng(3), PARAMS)
        self.assertEqual(a.dtype, np.uint16)
        np.testing.assert_array_equal(a, b)
        self.assertTrue((a[:, :5] == 0).all() and (a[:, -5:] == 0).all())
        mid = a[:, 10:-10]
        self.assertTrue((np.abs(mid.astype(int) - 500) <= 5).all())   # mm, ~1 mm noise at 0.5 m

    def test_noise_matches_intel_rms_formula(self):
        z = np.full((400, 400), 1.0)
        mm = d435.degrade_depth(z, np.random.default_rng(0), PARAMS).astype(float)
        expected_mm = 1e3 * d435.noise_sigma_m(1.0, PARAMS)           # 1^2*0.08/(447*0.05) m = 3.6 mm
        self.assertAlmostEqual(expected_mm, 3.58, places=2)
        self.assertAlmostEqual(np.std(mm), expected_mm, delta=0.3)

    def test_holes_at_depth_jumps_and_dropout(self):
        z = np.full((20, 20), 0.6)
        z[:, 10:] = 0.9                                               # a 30 cm step edge
        a = d435.degrade_depth(z, np.random.default_rng(1), PARAMS)
        self.assertTrue((a[:, 8:12] == 0).all())                      # edge +/- 1 px dilation
        self.assertTrue((a[:, :7] > 0).all() and (a[:, 13:] > 0).all())
        drop = dict(PARAMS, dropout_rate=0.2)
        b = d435.degrade_depth(np.full((200, 200), 0.6), np.random.default_rng(2), drop)
        self.assertAlmostEqual(float((b == 0).mean()), 0.2, delta=0.02)

    def test_intrinsics(self):
        k = d435.intrinsics(424, 240, 42.0)
        self.assertAlmostEqual(k["fy"], 120 / np.tan(np.radians(21)), places=6)
        hfov = 2 * np.degrees(np.arctan(212 / k["fx"]))
        self.assertAlmostEqual(hfov, 68.3, delta=0.5)                 # the D435 RGB stream's ~69 deg


if __name__ == "__main__":
    unittest.main()
