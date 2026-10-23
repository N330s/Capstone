import json
from pathlib import Path
import tempfile
import unittest

from scripts.evaluate_recovery_transfer import check_rollout, validate_pair
from scripts.probe_recovery_behavior import valid_horizon
from scripts.audit_recovery_sampling import opening_mask
from learning.vla.smolvla_data import sha256


class RecoveryEvaluationTests(unittest.TestCase):
    def test_active_reopening_distinguished_from_other_recovery(self):
        import numpy as np
        labels = np.array(["recovery", "recovery", "expert", "fault"])
        states = np.zeros((5, 16))
        states[:, 15] = .02
        actions = np.zeros((4, 8))
        actions[:, 7] = [.0204, .011, .0204, .0204]
        np.testing.assert_array_equal(opening_mask(labels, states, actions), [True, False, False, False])

    def test_probe_cannot_target_fault_or_cross_fault(self):
        labels = ["expert", "expert", "fault", "recovery", "expert"]
        self.assertEqual(valid_horizon(labels, 0), 2)
        self.assertEqual(valid_horizon(labels, 1), 1)
        self.assertEqual(valid_horizon(labels, 3), 2)
        self.assertEqual(valid_horizon(labels, 4), 1)
        for step in (-1, 2, 5, True):
            with self.assertRaises(ValueError):
                valid_horizon(labels, step)

    def test_mismatched_model_pair_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            baseline, candidate = root / "baseline", root / "candidate/pretrained_model"
            baseline.mkdir()
            candidate.mkdir(parents=True)
            (baseline / "model.safetensors").write_bytes(b"baseline")
            (candidate / "model.safetensors").write_bytes(b"candidate")
            marker = candidate.parent / "complete.json"
            marker.write_text(json.dumps({"binding": {"base_weights_sha256": sha256(baseline / "model.safetensors")}}))
            self.assertEqual(set(validate_pair(baseline, candidate)), {"baseline", "recovery_pilot"})
            (baseline / "model.safetensors").write_bytes(b"changed")
            with self.assertRaises(ValueError):
                validate_pair(baseline, candidate)

    def test_rollout_bindings_and_task_failure_are_distinct(self):
        report = {"checkpoint_weights_sha256": "hash", "source_episode": "episode", "execution_prefix": 5,
                  "policy_seed": 1000, "mode": "policy", "pipeline_passed": True,
                  "task_success": False, "outcome": "robot_collision"}
        check_rollout(report, "hash", "episode", 5, 1000)
        for field, value in (("checkpoint_weights_sha256", "wrong"), ("policy_seed", 1),
                             ("execution_prefix", 25), ("pipeline_passed", False)):
            with self.assertRaises(ValueError):
                check_rollout({**report, field: value}, "hash", "episode", 5, 1000)


if __name__ == "__main__":
    unittest.main()
