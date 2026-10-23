from datetime import datetime, timezone
import json
from pathlib import Path
import tempfile
import unittest

from learning.vla.smolvla_data import sha256
from scripts.continue_smolvla_until import deadline_timestamp, source_checkpoint


class ExtensionTests(unittest.TestCase):
    def test_deadline_is_timezone_explicit(self):
        self.assertEqual(deadline_timestamp("2026-10-13T08:00:00+07:00"),
                         datetime(2026, 10, 13, 1, tzinfo=timezone.utc).timestamp())
        with self.assertRaises(ValueError):
            deadline_timestamp("2026-10-13T08:00:00")

    def test_only_completed_matching_source_can_continue(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            export, run = root / "export", root / "run"
            export.mkdir()
            checkpoint = run / "checkpoints/final"
            checkpoint.mkdir(parents=True)
            (export / "openarm_export.json").write_text("{}")
            binding = {"export_receipt_sha256": sha256(export / "openarm_export.json")}
            marker = {"step": 123, "binding": binding}
            (checkpoint / "complete.json").write_text(json.dumps(marker))
            report = {"completed": True, "binding": binding, "final_checkpoint": str(checkpoint)}
            (run / "report.json").write_text(json.dumps(report))
            self.assertEqual(source_checkpoint(run, export), (checkpoint.resolve(), marker))
            report["completed"] = False
            (run / "report.json").write_text(json.dumps(report))
            with self.assertRaisesRegex(ValueError, "did not complete"):
                source_checkpoint(run, export)
            report["completed"] = True
            report["final_checkpoint"] = str(root / "elsewhere")
            (run / "report.json").write_text(json.dumps(report))
            with self.assertRaisesRegex(ValueError, "outside"):
                source_checkpoint(run, export)
            report["final_checkpoint"] = str(checkpoint)
            (run / "report.json").write_text(json.dumps(report))
            (export / "openarm_export.json").write_text('{"changed": true}')
            with self.assertRaisesRegex(ValueError, "dataset changed"):
                source_checkpoint(run, export)


if __name__ == "__main__":
    unittest.main()
