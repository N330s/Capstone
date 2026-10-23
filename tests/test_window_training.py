import unittest
import threading
from unittest.mock import Mock, patch

from learning.vla.window_training import CommandWindows, validate_budget


class WindowTrainingTests(unittest.TestCase):
    def test_all_commands_and_boundaries(self):
        windows = CommandWindows([{"episode": "a", "steps": 3}, {"episode": "b", "steps": 2}])
        self.assertEqual(len(windows), 5)
        self.assertEqual([windows.locate(i)["step"] for i in range(5)], [0, 1, 2, 0, 1])
        self.assertEqual(windows.locate(3)["episode"], "b")
        for bad in (-1, 5, 1.5, True):
            with self.assertRaises(IndexError):
                windows.locate(bad)

    def test_deterministic_bounded_probes(self):
        windows = CommandWindows([{"episode": str(i), "steps": 100} for i in range(10)])
        self.assertEqual(len(windows.probes(16)), 16)
        self.assertEqual(windows.probes(16), windows.probes(16))
        self.assertEqual(len(set(p["episode"] for p in windows.probes(16))), 10)

    def test_runtime_cap_and_invalid_values(self):
        validate_budget(14400, 100, 1, 1e-4)
        validate_budget(28800, 100, 1, 1e-4, ceiling=28800)
        with self.assertRaises(ValueError):
            validate_budget(28800, 100, 1, 1e-4)  # remains opt-in, not a silent default change
        for seconds in (0, -1, 14401, float("nan")):
            with self.assertRaises(ValueError):
                validate_budget(seconds, 100, 1, 1e-4)
        for records in ([], [{"episode": "a", "steps": 0}]):
            with self.assertRaises(ValueError):
                CommandWindows(records)

    def test_watchdog_targets_only_live_child(self):
        from scripts.run_smolvla_job import arm_deadline
        class Child:
            killed = False
            def poll(self):
                return None
            def kill(self):
                self.killed = True
        child, fired = Child(), threading.Event()
        timer = arm_deadline(child, .01, fired)
        timer.join(timeout=1)
        self.assertTrue(child.killed)
        self.assertTrue(fired.is_set())
        child = Child()
        child.poll = lambda: 0
        fired = threading.Event()
        timer = arm_deadline(child, .01, fired)
        timer.join(timeout=1)
        self.assertFalse(child.killed)
        self.assertFalse(fired.is_set())

    def test_deadline_kills_windows_launcher_descendants(self):
        from scripts.run_smolvla_job import kill_owned_process
        launcher = Mock(pid=123)
        launcher.poll.return_value = None
        child = Mock()
        with patch("psutil.Process") as process:
            process.return_value.children.return_value = [child]
            kill_owned_process(launcher)
        child.kill.assert_called_once()
        launcher.kill.assert_called_once()


if __name__ == "__main__":
    unittest.main()
