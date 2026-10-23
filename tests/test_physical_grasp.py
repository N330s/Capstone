import unittest
from learning.vla.physical_grasp import PhysicalGraspTracker


class PhysicalGraspTests(unittest.TestCase):
    def sample(self, tracker, **overrides):
        return tracker.update(**({"contacts": 2, "speed": .001, "pad_min": .019,
            "command": .011, "slip": 0., "slip_limit": .01, "dt": .02} | overrides))

    def test_target_jitter_does_not_erase_real_settled_contact(self):
        tracker = PhysicalGraspTracker()
        events = [self.sample(tracker, command=.011 + i * .00001) for i in range(5)]
        self.assertEqual(events, [None] * 4 + ["latch"])

    def test_noise_below_actual_fingers_is_not_physical_release(self):
        tracker = PhysicalGraspTracker(latched=True)
        self.assertIsNone(self.sample(tracker, command=.0115))
        self.assertTrue(tracker.latched)
        self.assertEqual(self.sample(tracker, contacts=1, command=.020), "release")

    def test_contact_speed_and_squeeze_requirements_are_retained(self):
        for bad in ({"contacts": 1}, {"speed": .006}, {"command": .019}):
            tracker = PhysicalGraspTracker()
            for _ in range(20):
                self.assertIsNone(self.sample(tracker, **bad))
            self.assertFalse(tracker.latched)

    def test_slip_is_still_counted_as_drop(self):
        tracker = PhysicalGraspTracker(latched=True)
        self.assertEqual(self.sample(tracker, slip=.02), "drop")
        self.assertFalse(tracker.latched)
