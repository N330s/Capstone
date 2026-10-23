import json
from pathlib import Path
import tempfile
import unittest

import torch

from scripts.run_fp32_multiepisode import compare_qualification


class FP32CampaignTests(unittest.TestCase):
    def make_run(self, root, name):
        run = root / name
        checkpoint = run / "checkpoints/step_0000012_final"
        (checkpoint / "pretrained_model").mkdir(parents=True)
        (checkpoint / "pretrained_model/model.safetensors").write_bytes(b"same model bytes")
        state = {"binding": {"trainable_storage": "float32"}, "step": 12,
            "optimizer": {"state": {0: {"exp_avg": torch.zeros(2), "exp_avg_sq": torch.ones(2)}}},
            "scheduler": {"step": 12}, "rng": {"test": torch.tensor([1, 2])}, "sampler": [2, 3]}
        torch.save(state, checkpoint / "training_state.pt")
        updates = [{"type": "update", "step": i, "loss": .1, "grad_norm": .2, "lr": 1e-5,
                    "samples": [{"episode": "a", "step": i}], "elapsed_s": i * 2} for i in range(1, 13)]
        (run / "events.jsonl").write_text("\n".join(map(json.dumps, updates)))
        return run, checkpoint, state

    def test_exact_resume_checks_state_and_complete_update_history(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            a, _, _ = self.make_run(root, "reference")
            b, checkpoint, state = self.make_run(root, "split")
            self.assertTrue(compare_qualification(a, b)["passed"])
            state["sampler"] = [9]
            torch.save(state, checkpoint / "training_state.pt")
            with self.assertRaises(AssertionError):
                compare_qualification(a, b)

    def test_equal_bf16_moments_still_fail(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            runs = []
            for name in ("reference", "split"):
                run, checkpoint, state = self.make_run(root, name)
                state["optimizer"]["state"][0]["exp_avg"] = torch.zeros(2, dtype=torch.bfloat16)
                torch.save(state, checkpoint / "training_state.pt")
                runs.append(run)
            with self.assertRaises(ValueError):
                compare_qualification(*runs)

    def test_changed_action_sample_fails(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            a, _, _ = self.make_run(root, "reference")
            b, _, _ = self.make_run(root, "split")
            path = b / "events.jsonl"
            values = list(map(json.loads, path.read_text().splitlines()))
            values[-1]["samples"][0]["step"] = 99
            path.write_text("\n".join(map(json.dumps, values)))
            with self.assertRaises(AssertionError):
                compare_qualification(a, b)
