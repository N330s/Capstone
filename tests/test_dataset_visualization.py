import importlib.util
import unittest

import numpy as np

HAS_MATPLOTLIB = importlib.util.find_spec("matplotlib") is not None
if HAS_MATPLOTLIB:
    from scripts.visualize_smolvla_dataset import first_phase_frame, phase_segments


@unittest.skipUnless(HAS_MATPLOTLIB, "Install requirements-visualization.txt for plotting tests")
class DatasetVisualizationTests(unittest.TestCase):
    def test_storyboard_uses_first_contiguous_phase(self):
        phases = np.array(["survey", "approach", "approach", "close", "approach", "approach"])
        self.assertEqual(first_phase_frame(phases, "approach"), 1)
        self.assertEqual(first_phase_frame(phases, "approach", 1), 2)

    def test_missing_phase_rejected(self):
        with self.assertRaises(ValueError):
            first_phase_frame(np.array(["survey"]), "insert")

    def test_terminal_boundary_retained(self):
        segments = phase_segments(np.array(["close", "close", "lift"]), np.array([0., .02, .04, .055]))
        self.assertEqual(segments, [("close", 0., .04), ("lift", .04, .055)])


if __name__ == "__main__":
    unittest.main()
