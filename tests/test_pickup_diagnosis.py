import unittest
import numpy as np
from learning.vla.pickup_diagnosis import unfold_prefixes, first_sustained, command_errors


class PickupDiagnosisTests(unittest.TestCase):
    def test_prefixes_are_aligned_and_terminal_is_not_an_extra_anchor(self):
        chunks = np.zeros((3, 50, 8))
        chunks[1] = 1; chunks[2] = 2
        result = unfold_prefixes([0, 25, 50], chunks, 61)
        self.assertEqual(result.shape, (61, 8))
        np.testing.assert_array_equal(result[:, 0], [0]*25+[1]*25+[2]*11)
        with self.assertRaises(ValueError):
            unfold_prefixes([0, 26, 50], chunks, 61)

    def test_sustained_error_does_not_flag_single_noise_spike(self):
        self.assertIsNone(first_sustained([False, True, False], 2))
        self.assertEqual(first_sustained([False, True, True], 2), 1)

    def test_gripper_error_diagnostic_is_in_mm_and_before_close_only(self):
        expert = np.zeros((30, 8)); expert[:, 7] = .035
        predicted = expert.copy(); predicted[5:, 7] = .025
        phases = np.array(["descend"]*20+["close"]*10)
        result = command_errors(predicted, expert, phases)
        self.assertEqual(result["first_10_command_premature_close"], 5)
        self.assertAlmostEqual(result["expert_open_gripper_mae_mm"], 7.5)
