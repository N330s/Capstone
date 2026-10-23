import unittest
import numpy as np
from learning.vla.pickup_onset import filtered_targets, loss_components


class PickupOnsetTests(unittest.TestCase):
    def test_rate_limited_filter_and_replay_idempotence(self):
        initial = np.r_[np.zeros(7), .044]
        requests = np.tile(np.r_[np.ones(7), .035], (30, 1))
        targets = filtered_targets(requests, initial, -np.ones(7)*3, np.ones(7)*3)
        self.assertAlmostEqual(targets[0, 0], .01)
        self.assertAlmostEqual(targets[0, 7], .0436)
        self.assertAlmostEqual(targets[-1, 7], .035)
        np.testing.assert_allclose(filtered_targets(targets, initial, -np.ones(7)*3, np.ones(7)*3), targets, atol=1e-15)

    def test_padded_dimension_mean_does_not_hide_physical_mean_in_audit(self):
        loss = np.ones((1, 50, 32)); loss[:, :, 8:] = 0
        report = loss_components(loss, np.zeros((1, 50), bool))
        self.assertEqual(report["native_mean"], .25)
        self.assertEqual(report["actual_dimensions_mean"], 1)
        self.assertEqual(report["physical_share_of_total_loss"], 1)
        self.assertEqual(report["reconstruction_error"], 0)

    def test_terminal_padding_keeps_the_native_denominator_visible(self):
        losses = np.ones((1, 50, 32)); pad = np.ones((1, 50), bool); pad[:, :25] = False
        self.assertEqual(loss_components(losses, pad)["native_mean"], .5)
