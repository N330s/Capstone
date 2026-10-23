import json
import os
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from learning.vla.runtime_io import write_json, read_json, write_status
from learning.vla.checkpoint_retention import prune_checkpoints


class RuntimeIOTests(unittest.TestCase):
    def test_permission_retry_and_nonfatal_status(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "status.json"
            write_json(path, {"step": 0})
            real_replace = os.replace
            calls = []
            def transient(source, target):
                calls.append(1)
                if len(calls) < 3:
                    raise PermissionError("sharing violation")
                real_replace(source, target)
            with patch("learning.vla.runtime_io.os.replace", side_effect=transient):
                write_json(path, {"step": 1}, retry_seconds=1)
            self.assertEqual(read_json(path), {"step": 1})
            with patch("learning.vla.runtime_io.os.replace", side_effect=PermissionError("held reader")):
                self.assertFalse(write_status(path, {"step": 2}))
                with self.assertRaises(PermissionError):
                    write_json(path, {"critical": True}, retry_seconds=0)
            self.assertEqual(read_json(path), {"step": 1})
            self.assertEqual(list(Path(folder).glob("*.tmp")), [])

    def test_concurrent_real_readers_never_see_partial_json(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "status.json"
            write_json(path, {"step": 0, "data": "x" * 5000})
            stop, failures = threading.Event(), []
            def reader():
                while not stop.is_set():
                    try:
                        self.assertEqual(len(read_json(path)["data"]), 5000)
                    except Exception as error:
                        failures.append(str(error))
                        stop.set()
            thread = threading.Thread(target=reader)
            thread.start()
            try:
                for step in range(200):
                    write_json(path, {"step": step, "data": "x" * 5000})
            finally:
                stop.set()
                thread.join(timeout=5)
            self.assertEqual(failures, [])
            self.assertEqual(read_json(path)["step"], 199)

    @unittest.skipUnless(os.name == "nt", "Real Windows non-delete-sharing file lock")
    def test_real_windows_lock_does_not_make_telemetry_fatal(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "status.json"
            write_json(path, {"step": 0})
            with path.open("r") as locked:
                self.assertEqual(json.load(locked), {"step": 0})
                self.assertFalse(write_status(path, {"step": 1}))
            self.assertTrue(write_status(path, {"step": 2}))
            self.assertEqual(read_json(path), {"step": 2})


class CheckpointRetentionTests(unittest.TestCase):
    def test_only_superseded_completed_checkpoints_from_same_run_pruned(self):
        with tempfile.TemporaryDirectory() as folder:
            parent = Path(folder) / "checkpoints"
            parent.mkdir()
            binding = {"run": "owned"}
            for step in range(6):
                path = parent / f"step_{step:07d}_periodic"
                path.mkdir()
                (path / "complete.json").write_text(json.dumps({"step": step,
                    "label": "recovery_initial" if step == 0 else "periodic", "binding": binding}))
            unrelated = parent / "step_0000100_user"
            unrelated.mkdir()
            (unrelated / "complete.json").write_text(json.dumps({"step": 100, "label": "periodic", "binding": {"run": "user"}}))
            incomplete = parent / "step_0000200_partial"
            incomplete.mkdir()
            protected = parent / "step_0000002_periodic"
            removed = prune_checkpoints(parent, binding, [protected], keep_recent=2)
            self.assertEqual({r["step"] for r in removed}, {1, 3})
            self.assertTrue(protected.exists())
            self.assertTrue((parent / "step_0000000_periodic").exists())
            self.assertTrue(unrelated.exists())
            self.assertTrue(incomplete.exists())


if __name__ == "__main__":
    unittest.main()
