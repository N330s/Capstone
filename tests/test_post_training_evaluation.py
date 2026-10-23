import json
from pathlib import Path
import tempfile
import unittest

from scripts.evaluate_smolvla_run import checkpoint_plan, summarize
from learning.vla.smolvla_data import sha256


class EvaluationTests(unittest.TestCase):
    def test_unequal_episode_lengths_have_distinct_averages(self):
        result = summarize([{"episode": "a", "flow_loss": 1},
                            {"episode": "a", "flow_loss": 3},
                            {"episode": "b", "flow_loss": 8}])
        self.assertEqual(result["command_weighted_flow_loss"], 4)
        self.assertEqual(result["episode_balanced_flow_loss"], 5)
        self.assertEqual(result["windows"], 3)

    def test_empty_and_nonfinite_results_rejected(self):
        for rows in ([], [{"episode": "a", "flow_loss": float("nan")}],
                     [{"episode": "a", "flow_loss": float("inf")}]):
            with self.assertRaises(ValueError):
                summarize(rows)

    def test_checkpoint_identity_and_completion_guards(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run, export = root / "run", root / "export"
            run.mkdir()
            export.mkdir()
            (export / "openarm_export.json").write_text("{}")
            binding = {"export_receipt_sha256": sha256(export / "openarm_export.json")}
            checkpoint = run / "checkpoints/step_0000001_final"
            (checkpoint / "pretrained_model").mkdir(parents=True)
            (checkpoint / "pretrained_model/model.safetensors").write_bytes(b"fixture")
            marker = checkpoint / "complete.json"
            marker.write_text(json.dumps({"step": 1, "binding": binding}))
            report = {"completed": True, "binding": binding, "final_checkpoint": str(checkpoint)}
            (run / "report.json").write_text(json.dumps(report))
            (run / "best_checkpoint.json").write_text(json.dumps({"path": str(checkpoint)}))
            choices = checkpoint_plan(run, export)
            self.assertEqual(len(choices), 1)
            self.assertEqual(choices[0]["labels"], ["best", "final"])
            self.assertEqual(checkpoint_plan(run, export, "best")[0]["labels"], ["best"])
            (run / "latest_checkpoint.json").write_text(json.dumps({"path": str(checkpoint)}))
            self.assertEqual(checkpoint_plan(run, export, "latest")[0]["labels"], ["latest"])
            report["completed"] = False
            (run / "report.json").write_text(json.dumps(report))
            with self.assertRaisesRegex(ValueError, "not complete"):
                checkpoint_plan(run, export)
            report["completed"] = True
            (run / "report.json").write_text(json.dumps(report))
            (export / "openarm_export.json").write_text('{"changed": true}')
            with self.assertRaisesRegex(ValueError, "mismatch"):
                checkpoint_plan(run, export)
            (export / "openarm_export.json").write_text("{}")
            marker.unlink()
            with self.assertRaises(FileNotFoundError):
                checkpoint_plan(run, export)


if __name__ == "__main__":
    unittest.main()
