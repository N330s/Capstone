import tempfile
import unittest
from pathlib import Path

import numpy as np

from controllers import faults
from data_pipeline.episodes import save_episode, validate_episode, validate_step_labels


def synthetic_episode(path, steps=6, labels=None, fault=None):
    """A tiny successful episode in the save_episode format (4x4 images)."""
    image = np.zeros((4, 4, 3), dtype=np.uint8)
    observations = [{"state": np.zeros(16, dtype=np.float32), "timestamp_s": 0.02 * i,
                     "instruction": "test", "images": {"scene_rgb": image, "wrist_rgb": image}}
                    for i in range(steps + 1)]
    action = np.zeros(8)
    infos = [{"outcome": "running"}] * (steps - 1) + [{"outcome": "success"}]
    return save_episode(path, observations, [action] * steps, [action] * steps,
                        [np.zeros(3)] * (steps + 1), [np.zeros(3)] * (steps + 1),
                        [0.] * (steps - 1) + [1.], [False] * (steps - 1) + [True], [False] * steps,
                        [0.02] * steps, ["p"] * steps, infos,
                        {"robot_config": {"image_height": 4, "image_width": 4}},
                        labels=labels, fault=fault)


class FaultDrawTests(unittest.TestCase):
    def test_deterministic_and_in_range(self):
        for seed in range(200):
            spec = faults.draw_fault(seed)
            self.assertEqual(spec, faults.draw_fault(seed))
            p = spec.params
            if spec.kind == "premature_close":
                lo, hi = faults.PREMATURE_CLOSE_FRAC
                self.assertTrue(lo <= p["descend_frac"] <= hi)
            elif spec.kind == "wrong_grasp":
                self.assertIn(p["mode"], faults.WRONG_GRASP_MODES)
                self.assertTrue(p["shift_m"] or p["yaw_deg"])
                if p["shift_m"]:
                    self.assertTrue(faults.WRONG_GRASP_SHIFT_M[0] <= abs(p["shift_m"]) <= faults.WRONG_GRASP_SHIFT_M[1])
                if p["yaw_deg"]:
                    self.assertTrue(faults.WRONG_GRASP_YAW_DEG[0] <= abs(p["yaw_deg"]) <= faults.WRONG_GRASP_YAW_DEG[1])
            else:
                lo, hi = faults.MISALIGN_OFFSET_M
                self.assertAlmostEqual(np.hypot(*p["offset_m"]), p["magnitude_m"])
                self.assertTrue(lo <= p["magnitude_m"] <= hi)

    def test_kinds_are_balanced_and_restricted(self):
        kinds = [faults.draw_fault(s).kind for s in range(300)]
        self.assertEqual({k: kinds.count(k) for k in faults.FAULT_KINDS},
                         {k: 100 for k in faults.FAULT_KINDS})
        self.assertEqual({faults.draw_fault(s, ("wrong_grasp",)).kind for s in range(10)}, {"wrong_grasp"})
        self.assertIsNone(faults.draw_fault(3, ()))

    def test_parse_kinds(self):
        self.assertEqual(faults.parse_kinds("all"), faults.FAULT_KINDS)
        self.assertEqual(faults.parse_kinds("misaligned_insert, premature_close"),
                         ("premature_close", "misaligned_insert"))
        self.assertEqual(faults.parse_kinds(None), ())
        with self.assertRaises(ValueError):
            faults.parse_kinds("grip_too_hard")


class StepLabelTests(unittest.TestCase):
    def test_labelled_episode_round_trips(self):
        labels = ["expert", "fault", "fault", "recovery", "expert", "expert"]
        fault = faults.draw_fault(1).to_json()
        with tempfile.TemporaryDirectory() as tmp:
            meta = synthetic_episode(Path(tmp) / "ep", labels=labels, fault=fault)
            self.assertEqual(meta["label_counts"], {"expert": 3, "fault": 2, "recovery": 1})
            self.assertEqual(meta["fault"], fault)
            validate_episode(Path(tmp) / "ep")
            with np.load(Path(tmp) / "ep" / "episode.npz", allow_pickle=False) as a:
                self.assertEqual(a["step_label"].tolist(), labels)

    def test_unlabelled_episode_unchanged(self):
        with tempfile.TemporaryDirectory() as tmp:
            meta = synthetic_episode(Path(tmp) / "ep")
            self.assertNotIn("label_counts", meta)
            validate_episode(Path(tmp) / "ep")
            with np.load(Path(tmp) / "ep" / "episode.npz", allow_pickle=False) as a:
                self.assertNotIn("step_label", a.files)

    def test_bad_labels_rejected(self):
        validate_step_labels(np.array(["expert", "fault", "recovery"]), 3)
        for labels, steps in ((["expert", "expert"], 3),                 # wrong length
                              (["expert", "oops", "expert"], 3),         # unknown value
                              (["expert", "fault", "expert"], 3),        # fault, never recovered
                              (["expert", "recovery", "fault"], 3)):     # ends on a fault step
            with self.assertRaises(ValueError):
                validate_step_labels(np.array(labels), steps)
        with tempfile.TemporaryDirectory() as tmp, self.assertRaises(ValueError):
            synthetic_episode(Path(tmp) / "ep", labels=["expert", "fault"] * 3)


if __name__ == "__main__":
    unittest.main()
