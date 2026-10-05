import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from data_pipeline.episodes import SCHEMA
from learning.vla.smolvla_data import SmolVLASource, export_source, observation, decoded_actions
from learning.vla.smolvla_train import training_command


def fixture(root):
    records = []
    for n, split in enumerate(("train", "validation")):
        path = root / f"episode_{n}"
        path.mkdir(parents=True)
        t = 3
        arrays = dict(observation_state=np.full((t + 1, 16), n, dtype=float),
                      action_requested=np.full((t, 8), n + .1),
                      action_applied=np.full((t, 8), n + .2),
                      timestamp_s=np.arange(t + 1) * .02, executed_dt_s=np.full(t, .02),
                      qpos=np.zeros((t + 1, 22)), qvel=np.zeros((t + 1, 22)),
                      reward=np.array([0., 0., 1.]), phase=np.array(["approach"] * t),
                      terminated=np.array([False, False, True]), truncated=np.zeros(t, dtype=bool),
                      image_scene_rgb=np.full((t + 1, 4, 6, 3), 25, np.uint8),
                      image_wrist_rgb=np.full((t + 1, 4, 6, 3), 200, np.uint8))
        np.savez_compressed(path / "episode.npz", **arrays)
        digest = hashlib.sha256((path / "episode.npz").read_bytes()).hexdigest()
        meta = dict(dataset_schema=SCHEMA, schema="openarm_v1_joint_targets_v1", steps=t,
                    data_sha256=digest, outcome="success", instruction="Insert.",
                    action_source="privileged_scripted_expert",
                    robot={"active_arm": "right", "parked_arm": "left"},
                    robot_config={"command_hz": 50, "image_height": 4, "image_width": 6},
                    scene_sha256="test-scene", source_hashes={}, physics_timestep_s=.0005)
        (path / "metadata.json").write_text(json.dumps(meta))
        records.append(dict(episode=path.name, split=split, steps=t, data_sha256=digest))
    manifest = dict(schema=SCHEMA, instruction="Insert.", episodes=records)
    (root / "manifest.json").write_text(json.dumps(manifest))
    return manifest


class FakeDataset:
    instances = []

    @classmethod
    def create(cls, **kwargs):
        obj = cls()
        obj.kwargs, obj.frames, obj.saved = kwargs, [], 0
        obj.root = kwargs["root"]
        (obj.root / "meta").mkdir(parents=True)
        cls.instances.append(obj)
        return obj

    def add_frame(self, frame):
        self.frames.append(frame)

    def save_episode(self):
        self.saved += 1

    def finalize(self):
        (self.root / "meta/info.json").write_text(json.dumps({
            "fps": 50, "total_episodes": self.saved, "features": self.kwargs["features"]}))


class SmolVLATests(unittest.TestCase):
    def test_mask_compatibility_preserves_original_batch(self):
        from learning.vla.smolvla_compat import masked_forward
        class Policy:
            def forward(self, batch, **kwargs):
                return batch
        raw = {"action_is_pad": np.array([False, True])}
        adapted = masked_forward(Policy(), raw)
        self.assertIs(adapted["actions_id_pad"], raw["action_is_pad"])
        self.assertNotIn("actions_id_pad", raw)
        with self.assertRaises(ValueError):
            masked_forward(Policy(), {})

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "source"
        self.manifest = fixture(self.root)

    def rewrite_manifest(self):
        (self.root / "manifest.json").write_text(json.dumps(self.manifest))

    def test_raw_units_camera_mapping_and_terminal_mask(self):
        source = SmolVLASource(self.root)
        sample = source.sample("episode_0", 2, horizon=5)
        np.testing.assert_array_equal(sample["action_is_pad"], [False, True, True, True, True])
        np.testing.assert_allclose(sample["action"], .2)
        self.assertEqual(sample["observation.state"].shape, (16,))
        self.assertEqual(sample["observation.images.camera1"].shape, (3, 4, 6))
        np.testing.assert_allclose(sample["observation.images.camera2"], 200 / 255)
        self.assertEqual(set(sample), {"observation.state", "task", "action", "action_is_pad",
                                       "observation.images.camera1", "observation.images.camera2"})
        self.assertEqual(source.report()["splits"]["train"]["steps"], 3)

    def test_duplicate_path_and_content_rejected(self):
        self.manifest["episodes"].append(self.manifest["episodes"][0])
        self.rewrite_manifest()
        with self.assertRaises(ValueError):
            SmolVLASource(self.root)

    def test_path_escape_rejected(self):
        self.manifest["episodes"][0]["episode"] = "../outside"
        self.rewrite_manifest()
        with self.assertRaises(ValueError):
            SmolVLASource(self.root)

    def test_explicit_subset_leaves_source_manifest_unchanged(self):
        self.manifest["episodes"].append({"episode": "missing", "split": "train", "steps": 3})
        self.rewrite_manifest()
        before = (self.root / "manifest.json").read_bytes()
        with self.assertRaisesRegex(ValueError, "Missing selected episodes"):
            SmolVLASource(self.root)
        selection = Path(self.temp.name) / "selection.json"
        selection.write_text(json.dumps({"splits": {"train": ["episode_0"], "validation": ["episode_1"]}}))
        source = SmolVLASource(self.root, selection)
        self.assertEqual(source.report()["excluded_manifest_episodes"], ["missing"])
        self.assertEqual((self.root / "manifest.json").read_bytes(), before)
        selection.write_text(json.dumps({"splits": {"train": ["episode_0"], "validation": ["episode_0"]}}))
        with self.assertRaises(ValueError):
            SmolVLASource(self.root, selection)

    def test_v3_requires_explicit_order(self):
        for name in ("episode_0", "episode_1"):
            path = self.root / name / "metadata.json"
            meta = json.loads(path.read_text())
            meta["schema"] = "openarm_v1_joint_targets_v3"
            meta["action_names"] = [f"openarm_right_joint{i}" for i in range(1, 8)] + ["right_finger_travel"]
            meta["state_order"] = "left joint1..7, left mean finger travel, right joint1..7, right mean finger travel"
            path.write_text(json.dumps(meta))
        self.assertEqual(SmolVLASource(self.root).report()["source_robot_schemas"], ["openarm_v1_joint_targets_v3"])
        meta["action_names"].reverse()
        path.write_text(json.dumps(meta))
        with self.assertRaisesRegex(ValueError, "ordering"):
            SmolVLASource(self.root)

    def test_manifest_mismatch_and_missing_split(self):
        self.manifest["episodes"][0]["steps"] = 99
        self.rewrite_manifest()
        with self.assertRaises(ValueError):
            SmolVLASource(self.root)
        self.manifest["episodes"][0]["steps"] = 3
        self.manifest["episodes"][1]["split"] = "train"
        self.rewrite_manifest()
        with self.assertRaises(ValueError):
            SmolVLASource(self.root)

    def test_bad_timing_rejected_even_with_updated_checksum(self):
        path = self.root / "episode_0"
        with np.load(path / "episode.npz") as data:
            arrays = dict(data)
        arrays["timestamp_s"] *= 2
        arrays["executed_dt_s"] *= 2
        np.savez_compressed(path / "episode.npz", **arrays)
        digest = hashlib.sha256((path / "episode.npz").read_bytes()).hexdigest()
        meta = json.loads((path / "metadata.json").read_text())
        meta["data_sha256"] = digest
        (path / "metadata.json").write_text(json.dumps(meta))
        self.manifest["episodes"][0]["data_sha256"] = digest
        self.rewrite_manifest()
        with self.assertRaises(ValueError):
            SmolVLASource(self.root)

    def test_postprocessed_commands_are_not_renormalized_or_clipped(self):
        raw = np.arange(16).reshape(2, 8)
        np.testing.assert_array_equal(decoded_actions(raw), raw)
        for bad in (np.zeros((2, 32)), np.full((2, 8), np.nan), np.zeros((0, 8))):
            with self.assertRaises(ValueError):
                decoded_actions(bad)

    def test_partial_final_interval_allowed(self):
        path = self.root / "episode_0"
        with np.load(path / "episode.npz") as data:
            arrays = dict(data)
        arrays["executed_dt_s"][-1] = .0015
        arrays["timestamp_s"][-1] = arrays["timestamp_s"][-2] + .0015
        np.savez_compressed(path / "episode.npz", **arrays)
        digest = hashlib.sha256((path / "episode.npz").read_bytes()).hexdigest()
        meta = json.loads((path / "metadata.json").read_text())
        meta["data_sha256"] = digest
        (path / "metadata.json").write_text(json.dumps(meta))
        self.manifest["episodes"][0]["data_sha256"] = digest
        self.rewrite_manifest()
        self.assertEqual(SmolVLASource(self.root).report()["fps"], 50)

    def test_split_export_and_safe_smoke_command(self):
        source = SmolVLASource(self.root)
        destination = Path(self.temp.name) / "export"
        FakeDataset.instances = []
        receipt = export_source(source, destination, "local/test", FakeDataset)
        self.assertEqual(len(FakeDataset.instances), 2)
        train, val = FakeDataset.instances
        self.assertEqual(train.kwargs["vcodec"], "h264")
        self.assertEqual(len(train.frames), 3)  # No fabricated terminal action.
        self.assertEqual(len(val.frames), 3)
        np.testing.assert_allclose(train.frames[0]["action"], .2)
        np.testing.assert_allclose(val.frames[0]["action"], 1.2)
        self.assertNotIn("qpos", train.frames[0])
        self.assertEqual(receipt["datasets"]["train"]["episodes"][0]["episode"], "episode_0")
        with self.assertRaises(FileExistsError):
            export_source(source, destination, "local/test", FakeDataset)
        checkpoint = Path(self.temp.name) / "checkpoint"
        checkpoint.mkdir()
        (checkpoint / "config.json").write_text('{"type": "smolvla"}')
        (checkpoint / "model.safetensors").write_bytes(b"test-only")
        command = training_command(destination, checkpoint, Path(self.temp.name) / "run")
        self.assertIn("--policy.input_features={}", command)
        self.assertIn("--policy.push_to_hub=false", command)
        self.assertIn(f"--dataset.root={destination / 'train'}", command)
        self.assertFalse(any("validation" in arg for arg in command))
        with self.assertRaises(ValueError):
            training_command(destination, checkpoint, Path(self.temp.name) / "run", steps=0)


if __name__ == "__main__":
    unittest.main()
