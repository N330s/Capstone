import threading
import unittest
from unittest.mock import Mock, patch
from learning.vla.process_control import arm_deadline, kill_owned_process


class ProcessControlTests(unittest.TestCase):
    def test_only_owned_descendants_and_live_launcher_are_killed(self):
        process = Mock(pid=123)
        process.poll.return_value = None
        children = [Mock(), Mock()]
        with patch('psutil.Process') as lookup:
            lookup.return_value.children.return_value = children
            kill_owned_process(process)
            lookup.assert_called_once_with(123)
        for child in children:
            child.kill.assert_called_once()
        process.kill.assert_called_once()

    def test_deadline_does_not_kill_completed_process(self):
        process = Mock()
        process.poll.return_value = 0
        fired = threading.Event()
        timer = arm_deadline(process, .01, fired)
        timer.join(timeout=1)
        self.assertFalse(fired.is_set())
        process.kill.assert_not_called()

    def test_deadline_signals_and_kills_only_owned_job(self):
        process = Mock()
        process.poll.return_value = None
        fired = threading.Event()
        with patch('learning.vla.process_control.kill_owned_process') as kill:
            timer = arm_deadline(process, .01, fired)
            timer.join(timeout=1)
            self.assertTrue(fired.is_set())
            kill.assert_called_once_with(process)
