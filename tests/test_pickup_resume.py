import copy
import unittest
import json
from pathlib import Path
import tempfile
import numpy as np

from scripts.evaluate_pickup_generalization import validate_resume_plan, terminal_trace_saved


class PickupResumeTests(unittest.TestCase):
    def setUp(self):
        self.plan = {"prefix": 25, "base_weights_sha256": "frozen", "parents": [
            {"episode": "a", "evaluation_group": "control"}, {"episode": "b", "evaluation_group": "validation"}],
            "implementation": {"scripts/evaluate_pickup_generalization.py": "old-supervisor", "actor.py": "frozen-actor"}}
        self.rows = [{"episode": "a", "group": "control", "mode": "replay"}]

    def test_only_supervisor_change_allowed(self):
        current = copy.deepcopy(self.plan)
        current["implementation"]["scripts/evaluate_pickup_generalization.py"] = "resumable-supervisor"
        validate_resume_plan(self.plan, current, self.rows)
        current["implementation"]["actor.py"] = "new-actor"
        with self.assertRaises(ValueError):
            validate_resume_plan(self.plan, current, self.rows)

    def test_model_and_prefix_cannot_change(self):
        for key, value in (("prefix", 50), ("base_weights_sha256", "new")):
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_resume_plan(self.plan, {**self.plan, key: value}, self.rows)

    def test_no_skip_replace_or_duplicate_completed_trials(self):
        for rows in ([{**self.rows[0], "episode": "b"}], self.rows * 2, []):
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                validate_resume_plan(self.plan, self.plan, rows)

    def test_encoding_grace_requires_complete_terminal_trace(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            self.assertFalse(terminal_trace_saved(folder))
            np.savez(folder / "trajectory.npz", action_requested=np.zeros((25, 8)), qpos=np.zeros((26, 7)))
            (folder / "diagnostics.json").write_text(json.dumps([{}] * 26))
            (folder / "chunks.json").write_text(json.dumps([{"executed_steps": 25}]))
            self.assertTrue(terminal_trace_saved(folder))
            (folder / "chunks.json").write_text(json.dumps([{"executed_steps": 24}]))
            self.assertFalse(terminal_trace_saved(folder))
