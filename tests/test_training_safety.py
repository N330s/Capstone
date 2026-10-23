import unittest
from learning.vla.training_safety import (
    GIB, checkpoint_budget_bytes, require_checkpoint_space, training_space_available)


class TrainingSafetyTests(unittest.TestCase):
    def test_estimate_includes_model_and_two_moments(self):
        estimate = checkpoint_budget_bytes(1_392_000_000, 399_523_968)
        self.assertGreater(estimate, 1_392_000_000 + 2 * 399_523_968)
        self.assertLess(estimate, 3 * GIB)

    def test_cannot_spend_disk_reserve(self):
        with self.assertRaises(RuntimeError):
            require_checkpoint_space(26 * GIB, 2 * GIB)
        require_checkpoint_space(27 * GIB, 2 * GIB)
        with self.assertRaises(ValueError):
            require_checkpoint_space(40 * GIB, 2 * GIB, 5)

    def test_early_stop_leaves_final_checkpoint_room(self):
        self.assertFalse(training_space_available(28 * GIB, 2 * GIB))
        self.assertTrue(training_space_available(29 * GIB, 2 * GIB))
        require_checkpoint_space(28 * GIB, 2 * GIB)
