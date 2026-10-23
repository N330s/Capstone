import json
from pathlib import Path
import tempfile
import unittest

from learning.vla.smolvla_data import sha256
from scripts.report_recovery_comparison import compare, behavior_summary


class RecoveryComparisonTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.roots = [Path(self.temp.name) / name for name in ("uniform", "correction")]
        for root in self.roots:
            bundle = root / "training/checkpoints/final/pretrained_model"
            bundle.mkdir(parents=True)
            (bundle / "model.safetensors").write_bytes(root.name.encode())
            binding = {k: "same" for k in ("export_receipt_sha256", "base_weights_sha256", "base_config_sha256",
                        "sampling_index_sha256", "trainer_sha256", "sampling_implementation_sha256")}
            binding.update(steps=4, batch_size=4, learning_rate=1e-5, seed=1, probes={}, sampling=root.name + "; segmented")
            self.write(root / "training/report.json", {"binding": binding, "completed": True, "updates": 4,
                "elapsed_s": 1, "initial_evaluation": {"scores": {}}, "final_evaluation": {"scores": {}},
                "evaluations": [], "peak_allocated_gb": 3, "final_checkpoint": str(bundle.parent)})
            self.write(root / "pilot_report.json", {"passed": True})
            self.write(root / "checkpoint_verification.json", {"passed": True})
            self.write(root / "sampling_audit.json", {"counts": {"fault_targets": 0, "total_windows": 16,
                       "active_reopening_anchors": 4 if root.name == "correction" else 0}})
            self.write(root / "behavior/report.json", {"sources": [], "cases": 1, "batch_size": 4,
                "all_heldout_recovery_frames": True, "heldout_recovery_parent_count": 6,
                "weights_sha256": sha256(bundle / "model.safetensors"), "summary": {}, "state_sensitivity": {}})

    @staticmethod
    def write(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))

    def change(self, relative, field, value):
        path = self.roots[1] / relative
        report = json.loads(path.read_text())
        current = report
        for key in field[:-1]:
            current = current[key]
        current[field[-1]] = value
        self.write(path, report)

    def test_equal_budget_comparison_verified(self):
        self.assertTrue(compare(*self.roots)["matched_fields_checked"])

    def test_unequal_training_budget_rejected(self):
        self.change("training/report.json", ["binding", "steps"], 5)
        with self.assertRaisesRegex(ValueError, "Unequal"):
            compare(*self.roots)

    def test_missing_quota_exposure_rejected(self):
        self.change("sampling_audit.json", ["counts", "active_reopening_anchors"], 0)
        with self.assertRaisesRegex(ValueError, "quota"):
            compare(*self.roots)

    def test_wrong_behavior_checkpoint_rejected(self):
        self.change("behavior/report.json", ["weights_sha256"], "wrong")
        with self.assertRaisesRegex(ValueError, "another checkpoint"):
            compare(*self.roots)

    def test_unmatched_behavior_cases_rejected(self):
        self.change("behavior/report.json", ["cases"], 2)
        with self.assertRaisesRegex(ValueError, "Unequal recorded"):
            compare(*self.roots)

    def test_active_correction_summary_is_not_limited_to_reopen_phase(self):
        row = {"group": "new_recovery", "episode": "heldout", "phase": "regrasp",
               "expert_opening_delta_mm": .4, "predicted_opening_delta_mm": -.5,
               "first_gripper_mae_mm": .9, "prefix5_gripper_mae_mm": 1., "chunk_gripper_mae_mm": 2.}
        result = behavior_summary({"summary": {}, "predictions": [row, {**row, "expert_opening_delta_mm": .01}]})
        self.assertEqual(result["new_active_reopening"]["predictions"], 1)
        self.assertEqual(result["new_active_reopening"]["positive_opening_prediction_fraction"], 0)
        self.assertEqual(set(result["active_reopening_by_parent"]), {"heldout"})


if __name__ == "__main__":
    unittest.main()
