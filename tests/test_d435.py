import unittest

import numpy as np

from envs import d435

PARAMS = {"units_m": 0.001, "baseline_m": 0.050, "subpixel": 0.08, "depth_fx_px": 447.0,
          "min_range_m": 0.28, "max_range_m": 3.0, "edge_jump_m": 0.02, "edge_dilate_px": 1,
          "dropout_rate": 0.0}
# Wrist D405 (configs/workspace_v2_cable30.json cameras.wrist_rgb.depth), dropout off
D405_PARAMS = dict(PARAMS, baseline_m=0.018, min_range_m=0.07, max_range_m=1.0)


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

    def test_d405_short_range_params(self):
        """The wrist D405 (cameras.wrist_rgb.depth) sees 7 cm-1 m; the D435 params give nothing there."""
        plane = np.full((20, 20), 0.10)          # constant plane: no depth jumps, so no edge holes
        near = d435.degrade_depth(plane, np.random.default_rng(4), D405_PARAMS)
        self.assertTrue((near > 0).all())
        self.assertTrue((np.abs(near.astype(int) - 100) <= 2).all())           # ~0.1 mm noise at 0.10 m
        self.assertTrue((d435.degrade_depth(plane, np.random.default_rng(4), PARAMS) == 0).all())   # D435 min 0.28 m
        too_close = d435.degrade_depth(np.full((20, 20), 0.05), np.random.default_rng(4), D405_PARAMS)
        self.assertTrue((too_close == 0).all())                                # below the D405's 7 cm
        too_far = d435.degrade_depth(np.full((20, 20), 1.2), np.random.default_rng(4), D405_PARAMS)
        self.assertTrue((too_far == 0).all())                                  # beyond its 1 m cut-off

    def test_d405_noise_follows_the_same_formula(self):
        expected_mm = 1e3 * d435.noise_sigma_m(0.5, D405_PARAMS)               # 0.5^2*0.08/(447*0.018) m = 2.5 mm
        self.assertAlmostEqual(expected_mm, 2.49, places=2)
        mm = d435.degrade_depth(np.full((400, 400), 0.5), np.random.default_rng(0), D405_PARAMS).astype(float)
        self.assertAlmostEqual(np.std(mm), expected_mm, delta=0.3)
        self.assertGreater(d435.noise_sigma_m(0.5, D405_PARAMS), d435.noise_sigma_m(0.5, PARAMS))   # shorter baseline

    def test_intrinsics(self):
        k = d435.intrinsics(424, 240, 42.0)
        self.assertAlmostEqual(k["fy"], 120 / np.tan(np.radians(21)), places=6)
        hfov = 2 * np.degrees(np.arctan(212 / k["fx"]))
        self.assertAlmostEqual(hfov, 68.3, delta=0.5)                 # the D435 RGB stream's ~69 deg


if __name__ == "__main__":
    unittest.main()
