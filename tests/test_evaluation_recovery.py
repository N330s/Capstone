import json
from pathlib import Path
import tempfile
import unittest

from learning.vla.evaluation_recovery import charged_external_seconds, validate_external_interruption
from learning.vla.smolvla_data import sha256

ROOT = Path(__file__).resolve().parents[1]


class EvaluationRecoveryTests(unittest.TestCase):
    def state(self):
        return ({"started_unix": 100.},
            {"completed": False, "state": "failed", "elapsed_s": 1900.2,
             "error": "RuntimeError('Training/evaluation heartbeat stalled for 15 minutes')"},
            {"stage": "pickup_evaluation", "state": "running", "started_unix": 1000.})

    def test_external_time_is_conservative_and_cannot_reset_budget(self):
        plan, failure, status = self.state()
        self.assertEqual(charged_external_seconds(plan, failure, status), 1003)
        for replacement in ({**failure, "completed": True}, {**failure, "error": "another failure"},
                            {**failure, "elapsed_s": 9000}, {**failure, "elapsed_s": float("nan")}):
            with self.assertRaises(ValueError):
                charged_external_seconds(plan, replacement, status)

    def fixture(self, campaign):
        evaluation = campaign / "pickup_evaluation"
        evaluation.mkdir()
        source = self.state()
        sources = {}
        for name, value in zip(("plan", "completion", "status"), source):
            path = campaign / (name + ".json")
            path.write_text(json.dumps(value))
            sources[name] = {"path": str(path), "sha256": sha256(path)}
        (evaluation / "plan.json").write_text(json.dumps({"parents": [{"episode": "control"}, {"episode": "heldout"}]}))
        artifact = evaluation / "original.npz"
        artifact.write_bytes(b"original completed artifact")
        marker = {"original_plan_sha256": sha256(evaluation / "plan.json"), "campaign_sources": sources,
            "external_elapsed_s": 1003, "preparation_s": 3, "elapsed_s": 1006,
            "completed_trials_preserved": True, "incomplete_attempt_restart_disclosed": True,
            "completed_rows": [{"episode": "control", "mode": "replay"}], "interrupted_condition": "control_policy",
            "preserved_files": {"original.npz": sha256(artifact)},
            "recovery_script_sha256": sha256(ROOT / "scripts/resume_fp32_evaluation.py"),
            "recovery_helper_sha256": sha256(ROOT / "learning/vla/evaluation_recovery.py")}
        return evaluation, artifact, marker

    def test_preserved_artifacts_and_prefix_are_bound(self):
        with tempfile.TemporaryDirectory() as d:
            folder, artifact, marker = self.fixture(Path(d))
            self.assertEqual(validate_external_interruption(folder, marker), 1006)
            marker["elapsed_s"] = 3
            with self.assertRaises(ValueError):
                validate_external_interruption(folder, marker)
            marker["elapsed_s"] = 1006
            artifact.write_bytes(b"changed")
            with self.assertRaises(ValueError):
                validate_external_interruption(folder, marker)

    def test_supervisor_evidence_or_condition_cannot_change(self):
        with tempfile.TemporaryDirectory() as d:
            folder, _, marker = self.fixture(Path(d))
            marker["interrupted_condition"] = "heldout_policy"
            with self.assertRaises(ValueError):
                validate_external_interruption(folder, marker)
            marker["interrupted_condition"] = "control_policy"
            (folder.parent / "completion.json").write_text("{}")
            with self.assertRaises(ValueError):
                validate_external_interruption(folder, marker)
