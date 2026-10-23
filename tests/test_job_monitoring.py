import os
from pathlib import Path
import tempfile
import unittest

from learning.vla.job_monitoring import heartbeat_age_seconds, heartbeat_stalled, evaluation_heartbeat_paths


class JobMonitoringTests(unittest.TestCase):
    def test_long_no_heartbeat_stage_uses_deadline_not_false_stall(self):
        age = heartbeat_age_seconds(None, stage_elapsed_s=1800, now_unix=10000)
        self.assertIsNone(age)
        self.assertFalse(heartbeat_stalled(age))

    def test_long_evaluation_with_fresh_progress_is_alive(self):
        with tempfile.TemporaryDirectory() as d:
            folder = Path(d)
            progress = folder / "progress.json"
            progress.write_text("{}")
            os.utime(progress, (9998, 9998))
            age = heartbeat_age_seconds(evaluation_heartbeat_paths(folder), stage_elapsed_s=1800, now_unix=10000)
            self.assertEqual(age, 2)
            self.assertFalse(heartbeat_stalled(age))
            os.utime(progress, (9000, 9000))
            self.assertTrue(heartbeat_stalled(heartbeat_age_seconds(
                evaluation_heartbeat_paths(folder), stage_elapsed_s=1800, now_unix=10000)))

    def test_missing_configured_heartbeat_is_stalled_but_future_mtime_not_negative(self):
        self.assertTrue(heartbeat_stalled(heartbeat_age_seconds([], stage_elapsed_s=901, now_unix=10000)))
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "event"
            path.touch()
            os.utime(path, (10001, 10001))
            self.assertEqual(heartbeat_age_seconds([path], stage_elapsed_s=1800, now_unix=10000), 0)

    def test_resume_startup_does_not_inherit_old_session_age(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "progress.json"
            path.write_text("{}")
            os.utime(path, (1000, 1000))
            self.assertEqual(heartbeat_age_seconds([path], stage_elapsed_s=5, now_unix=10000), 5)
            self.assertFalse(heartbeat_stalled(heartbeat_age_seconds([path], stage_elapsed_s=5, now_unix=10000)))
