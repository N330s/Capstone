"""make_release on tiny synthetic episodes (robot-free)."""
import copy
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from data_pipeline.episodes import save_episode, validate_episode

spec = importlib.util.spec_from_file_location("make_release", Path(__file__).resolve().parents[1] / "scripts" / "make_release.py")
make_release = importlib.util.module_from_spec(spec)
spec.loader.exec_module(make_release)

MANIFEST = {"robot_config": {"image_height": 4, "image_width": 5}, "source_hashes": {"a.py": "x"},
            "workspace": {"path": "w.json"}, "scene_sha256": "s", "contact_config": {}, "mujoco_version": "3",
            "camera_calibration": {"scene_rgb": {}}, "depth_cameras": None}


def fake_episode(path, seed, manifest=None, labels=None, steps=3):
    obs = [{"state": np.zeros(16), "timestamp_s": 0.02 * i, "instruction": "x",
            "images": {"scene_rgb": np.zeros((4, 5, 3), np.uint8), "wrist_rgb": np.zeros((4, 5, 3), np.uint8)}}
           for i in range(steps + 1)]
    act = [np.full(8, 0.1 * (i + 1)) for i in range(steps)]
    infos = [{"outcome": None}] * steps + [{"outcome": "success"}]
    m = {**copy.deepcopy(manifest or MANIFEST), "seed": seed, "reset_options": {"k": seed}}
    return save_episode(path, obs, act, act, [np.zeros(3)] * (steps + 1), [np.zeros(3)] * (steps + 1),
                        [0.0] * steps, [False] * (steps - 1) + [True], [False] * steps, [0.02] * steps,
                        ["hold"] * steps, infos, m, labels=labels, fault=None if labels is None else {"kind": "wrong_grasp"},
                        replay={"passed": True, "max_qpos_error": 0.0, "max_qvel_error": 0.0, "outcome": "success"})


COLLECTION = {"controllers/pick_insert_expert.py": "e1", "data_pipeline/scene_bank.py": "s1"}


def fake_run(root, name, episodes, collection=COLLECTION):
    run = root / name
    run.mkdir()
    for ep_id, seed, labels, manifest in episodes:
        fake_episode(run / ep_id, seed, manifest, labels)
    (run / "manifest.json").write_text(json.dumps({"complete": True, "collection_source_hashes": collection, "episodes": [{"episode": e[0]} for e in episodes]}))
    return run


class ReleaseTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        faults = ["expert", "fault", "recovery"]
        self.a = fake_run(self.tmp, "clean", [("e0", 100, None, None), ("e1", 101, None, None), ("e2", 102, None, None)])
        self.b = fake_run(self.tmp, "fault", [("f0", 100, faults, None), ("f1", 103, faults, None)])

    def tearDown(self):
        self._tmp.cleanup()

    def run_release(self, runs, out, extra=()):
        make_release.main(["--runs", *map(str, runs), "--name", "t", "--version", "1", "--output", str(out),
                           "--heldout", "0", *extra])

    def test_release_contents_and_grouping(self):
        out = self.tmp / "rel"
        self.run_release([self.a, self.b], out)
        rel = json.loads((out / "release.json").read_text())
        for key in ("name", "version", "created", "git", "source_hashes", "workspace", "robot_config",
                    "camera_calibration", "depth_cameras", "pilot_contract", "episodes"):
            self.assertIn(key, rel)
        self.assertEqual(rel["pilot_contract"]["cameras"], ["scene_rgb", "wrist_rgb"])
        self.assertEqual(len(rel["episodes"]), 5)
        split = {(e["run"], e["id"]): e["split"] for e in rel["episodes"]}
        self.assertEqual(split[("clean", "e0")], split[("fault", "f0")])     # shared seed 100
        self.assertIn("validation", split.values())
        self.assertIn("train", split.values())
        fault = [e for e in rel["episodes"] if e["id"] == "f0"][0]
        self.assertEqual(fault["fault_kind"], "wrong_grasp")
        self.assertEqual(fault["label_counts"], {"expert": 1, "fault": 1, "recovery": 1})
        self.assertTrue(fault["replay"]["passed"])
        splits = json.loads((out / "splits.json").read_text())
        self.assertEqual(sorted(splits["train"] + splits["validation"]), sorted(f"{r}__{i}" for r, i in split))
        # checksums cover every file and verify
        lines = (out / "checksums.sha256").read_text().strip().splitlines()
        files = [p for p in out.rglob("*") if p.is_file() and p.name != "checksums.sha256"]
        self.assertEqual(len(lines), len(files))
        for line in lines:
            digest, name = line.split("  ", 1)
            self.assertEqual(make_release.file_sha256(out / name), digest)
        validate_episode(out / "episodes" / "clean__e0")
        self.assertTrue((out / "heldout_resets.json").exists())

    def test_refuses_existing_output(self):
        out = self.tmp / "rel"
        self.run_release([self.a], out)
        with self.assertRaises(SystemExit):
            self.run_release([self.a], out)

    def test_refuses_mixed_source_hashes(self):
        other = copy.deepcopy(MANIFEST)
        other["source_hashes"] = {"a.py": "different"}
        c = fake_run(self.tmp, "other", [("g0", 200, None, other)])
        with self.assertRaises(SystemExit) as ctx:
            self.run_release([self.a, c], self.tmp / "rel")
        self.assertIn("mixed versions", str(ctx.exception))
        self.assertFalse((self.tmp / "rel").exists())

    def test_refuses_mixed_collection_hashes_before_replay(self):
        # episodes carry no passing replay here: the version error must come first
        c = fake_run(self.tmp, "newexpert", [("g0", 200, None, None)],
                     collection={**COLLECTION, "controllers/pick_insert_expert.py": "e2"})
        for run in (self.a, c):
            for meta in run.glob("*/metadata.json"):
                m = json.loads(meta.read_text()); m.pop("replay"); meta.write_text(json.dumps(m))
        with self.assertRaises(SystemExit) as ctx:
            self.run_release([self.a, c], self.tmp / "rel")
        self.assertIn("collection_source_hashes", str(ctx.exception))
        self.assertIn("mixed versions", str(ctx.exception))

    def test_missing_collection_hashes(self):
        old = fake_run(self.tmp, "old", [("h0", 300, None, None)], collection=None)
        with self.assertRaises(SystemExit) as ctx:
            self.run_release([old], self.tmp / "rel")
        self.assertIn("--allow-missing-collection-hashes", str(ctx.exception))
        with self.assertRaises(SystemExit):          # lacking + having cannot be compared
            self.run_release([old, self.a], self.tmp / "rel2", ["--allow-missing-collection-hashes"])
        self.run_release([old], self.tmp / "rel3", ["--allow-missing-collection-hashes"])
        rel = json.loads((self.tmp / "rel3" / "release.json").read_text())
        self.assertIsNone(rel["collection_source_hashes"])
        self.run_release([self.a], self.tmp / "rel4")
        rel = json.loads((self.tmp / "rel4" / "release.json").read_text())
        self.assertEqual(rel["collection_source_hashes"], COLLECTION)

    def test_refuses_incomplete_run(self):
        m = json.loads((self.a / "manifest.json").read_text())
        m["complete"] = False
        (self.a / "manifest.json").write_text(json.dumps(m))
        with self.assertRaises(SystemExit):
            self.run_release([self.a], self.tmp / "rel")

    def test_refuses_corrupt_episode(self):
        with open(self.a / "e0" / "episode.npz", "ab") as f:
            f.write(b"x")
        with self.assertRaises(SystemExit):
            self.run_release([self.a], self.tmp / "rel")

    def test_split_rule(self):
        seeds = list(range(1000, 1040))
        s = make_release.assign_splits(seeds, 0.15)
        self.assertEqual(s, make_release.assign_splits(seeds[::-1], 0.15))
        self.assertTrue(0 < list(s.values()).count("validation") < 20)
        two = make_release.assign_splits([1, 2], 0.0)
        self.assertIn("validation", two.values())


if __name__ == "__main__":
    unittest.main()
