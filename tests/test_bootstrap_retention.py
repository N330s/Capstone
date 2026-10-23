import json
from pathlib import Path
import tempfile
import unittest

from scripts.retire_fp32_bootstrap import validate_target


class BootstrapRetentionTests(unittest.TestCase):
    def checkpoint(self, path, step, label):
        (path / "pretrained_model").mkdir(parents=True)
        (path / "pretrained_model/model.safetensors").write_bytes(b"fixture")
        (path / "training_state.pt").write_bytes(b"fixture")
        (path / "complete.json").write_text(json.dumps({"step": step, "label": label,
            "binding": {"trainable_storage": "float32", "export_receipt_sha256": "frozen"}}))

    def test_initial_requires_new_complete_successor_and_not_current_pointer(self):
        with tempfile.TemporaryDirectory() as d:
            campaign = Path(d).resolve()
            initial = campaign / "training/checkpoints/step_0000025_recovery_initial"
            successor = campaign / "training/checkpoints/step_0000500_best"
            self.checkpoint(initial, 25, "recovery_initial")
            self.checkpoint(successor, 500, "best")
            validate_target(initial, successor, campaign)
            pointer = campaign / "training/latest_checkpoint.json"
            pointer.write_text(json.dumps({"path": str(initial)}))
            with self.assertRaises(ValueError):
                validate_target(initial, successor, campaign)

    def test_never_accept_other_directory_or_missing_model(self):
        with tempfile.TemporaryDirectory() as d:
            campaign = Path(d).resolve()
            initial = campaign / "training/checkpoints/step_0000025_recovery_initial"
            successor = campaign / "training/checkpoints/step_0000500_best"
            self.checkpoint(initial, 25, "recovery_initial")
            self.checkpoint(successor, 500, "best")
            with self.assertRaises(ValueError):
                validate_target(campaign / "unrelated", successor, campaign)
            (successor / "training_state.pt").unlink()
            with self.assertRaises(ValueError):
                validate_target(initial, successor, campaign)
