import tempfile
import unittest
from pathlib import Path

import numpy as np

from controllers import faults
from data_pipeline.episodes import TORQUE_KEYS, save_episode, validate_episode, validate_step_labels


def synthetic_episode(path, steps=6, labels=None, fault=None, sensors=False, torque_shape=(14,)):
    """A tiny successful episode in the save_episode format (4x4 images). ``sensors`` adds a
    chest RGB (4x6) + uint16 depth stream and per-step joint torque, as the v2_cable30 env does."""
    image = np.zeros((4, 4, 3), dtype=np.uint8)
    observations = [{"state": np.zeros(16, dtype=np.float32), "timestamp_s": 0.02 * i,
                     "instruction": "test", "images": {"scene_rgb": image, "wrist_rgb": image}}
                    for i in range(steps + 1)]
    infos = [{"outcome": "running"} for _ in range(steps)] + [{"outcome": "success"}]   # reset + T steps
    if sensors:
        for i, o in enumerate(observations):
            o["images"]["chest_rgb"] = np.full((4, 6, 3), i, dtype=np.uint8)
            o["depth"] = {"chest": np.full((4, 6), 400 + i, dtype=np.uint16)}
        for i, info in enumerate(infos):
            info.update({k: np.full(torque_shape, float(i)) for k in TORQUE_KEYS})
    action = np.zeros(8)
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
                self.assertIn(p["direction"], faults.WRONG_GRASP_DIRECTIONS)
                lo, hi = faults.WRONG_GRASP_DISTANCE_M
                self.assertTrue(lo <= p["distance_m"] <= hi)
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


class SensorArrayTests(unittest.TestCase):
    def test_torque_and_depth_saved_aligned(self):
        with tempfile.TemporaryDirectory() as tmp:
            meta = synthetic_episode(Path(tmp) / "ep", steps=5, sensors=True)
            self.assertEqual(meta["joint_torque"]["arrays"], list(TORQUE_KEYS))
            with np.load(Path(tmp) / "ep" / "episode.npz", allow_pickle=False) as a:
                self.assertEqual(a["image_chest_rgb"].shape, (6, 4, 6, 3))
                self.assertEqual(a["depth_chest_mm"].dtype, np.uint16)
                self.assertEqual(a["depth_chest_mm"][:, 0, 0].tolist(), [400, 401, 402, 403, 404, 405])
                for key in TORQUE_KEYS:
                    # row i = the interval of action i, i.e. info i+1 (info 0 is the reset)
                    self.assertEqual(a[key].shape, (5, 14))
                    self.assertEqual(a[key][:, 0].tolist(), [1., 2., 3., 4., 5.])
            frames = Path(tmp) / "ep" / "cameras"
            self.assertEqual(len(list((frames / "chest").glob("*.jpeg"))), 6)
            depth_png = sorted((frames / "chest_depth").glob("*.png"))
            self.assertEqual(len(depth_png), 6)
            from PIL import Image
            self.assertEqual(np.asarray(Image.open(frames / "chest_depth" / "1.png"))[0, 0], 400)

    def test_bad_torque_shape_rejected(self):
        with tempfile.TemporaryDirectory() as tmp, self.assertRaises(ValueError):
            synthetic_episode(Path(tmp) / "ep", sensors=True, torque_shape=(7,))


class FaultManifestTests(unittest.TestCase):
    """collect_random.build_manifest for a --faults run (regression: it once referenced removed
    fault constants and crashed every --faults --record run before the first recording)."""

    @classmethod
    def setUpClass(cls):
        import importlib.util
        import sys
        root = Path(__file__).resolve().parents[1]
        spec = importlib.util.spec_from_file_location("collect_random", root / "scripts/collect_random.py")
        sys.path.insert(0, str(root / "scripts"))
        try:
            cls.cr = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(cls.cr)
        finally:
            sys.path.pop(0)

    def test_fault_manifest_builds_and_serializes(self):
        import json
        from types import SimpleNamespace
        env = SimpleNamespace(config={"instruction": "test"}, manifest=lambda: {"robot": "stub"})
        for kinds in (faults.FAULT_KINDS, ()):
            args = SimpleNamespace(detector="noisy", start_row=0, faults=kinds)
            with tempfile.TemporaryDirectory() as tmp:
                out = Path(tmp)
                for name in ("collection_bank.json", "evaluation_bank.json"):
                    (out / name).write_text("[]", encoding="utf-8")
                manifest = self.cr.build_manifest(args, out, env, [], [], complete=False)
                json.dumps(manifest)
                if kinds:
                    ranges = manifest["fault_injection"]["ranges"]
                    self.assertEqual(set(ranges["wrong_grasp_directions"]), set(faults.WRONG_GRASP_DIRECTIONS))
                    self.assertEqual(ranges["wrong_grasp_distance_m"], list(faults.WRONG_GRASP_DISTANCE_M))
                    self.assertIn("training_note", manifest)
                else:
                    self.assertNotIn("fault_injection", manifest)


if __name__ == "__main__":
    unittest.main()
