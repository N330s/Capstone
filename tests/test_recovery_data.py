from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest

import numpy as np

from learning.vla.recovery_data import RecoveryVLASource, RECOVERY_CONTRACT, make_split_plan, prepare_view, valid_segments
from learning.vla.smolvla_data import SmolVLASource, export_source
from learning.vla.recovery_sampling import build_index, verify_index
from test_smolvla import fixture, FakeDataset


class RecoveryDataTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "source"
        self.manifest = fixture(self.root)
        for i, record in enumerate(self.manifest["episodes"]):
            path = self.root / record["episode"]
            with np.load(path / "episode.npz", allow_pickle=False) as arrays:
                raw = dict(arrays)
            raw["step_label"] = np.array(["expert", "fault", "recovery"])
            raw["action_applied"][:, 7] = [.035, 9999., .036]
            raw["depth_scene_mm"] = np.full((4, 4, 6), 65535, np.uint16)
            raw["joint_torque_applied_nm"] = np.full((3, 14), 12345.)
            np.savez_compressed(path / "episode.npz", **raw)
            digest = hashlib.sha256((path / "episode.npz").read_bytes()).hexdigest()
            metadata = json.loads((path / "metadata.json").read_text())
            metadata.update(data_sha256=digest, label_counts={"expert": 1, "fault": 1, "recovery": 1},
                            fault={"kind": "premature_close"})
            (path / "metadata.json").write_text(json.dumps(metadata))
            record.update(data_sha256=digest, seed=1000000 + i, label_counts=metadata["label_counts"],
                          collection="source", fault=metadata["fault"])
        self.manifest.update(complete=True, robot_manifest=metadata)
        self.write_manifest()

    def write_manifest(self):
        (self.root / "manifest.json").write_text(json.dumps(self.manifest))

    def test_legacy_reader_refuses_recovery_recordings(self):
        with self.assertRaisesRegex(ValueError, "RecoveryVLASource"):
            SmolVLASource(self.root)

    def test_segment_boundaries_and_unknown_labels(self):
        self.assertEqual(valid_segments(["expert", "fault", "fault", "recovery", "expert"]), [(0, 1), (3, 5)])
        self.assertEqual(valid_segments(["fault", "recovery", "fault"]), [(1, 2)])
        self.assertEqual(valid_segments(["fault"]), [])
        self.assertEqual(valid_segments(["expert", "recovery"]), [(0, 2)])
        for labels in ([], ["unknown"], [["expert"]], [1, 2]):
            with self.assertRaises(ValueError):
                valid_segments(labels)

    def test_fault_exclusion_and_masked_targets(self):
        source = RecoveryVLASource(self.root)
        self.assertEqual(len(source.records), 4)
        before = source.sample("episode_0__valid_000", 0, 50)
        after = source.sample("episode_0__valid_001", 0, 50)
        np.testing.assert_allclose(before["action"][:, 7], .035)
        np.testing.assert_allclose(after["action"][:, 7], .036)
        np.testing.assert_array_equal(before["action_is_pad"], [False] + [True] * 49)
        self.assertNotIn("step_label", before)
        self.assertNotIn("joint_torque_applied_nm", before)
        self.assertNotIn("depth_scene_mm", before)
        self.assertEqual(set(before), {"observation.state", "task", "action", "action_is_pad",
                                       "observation.images.camera1", "observation.images.camera2"})
        with self.assertRaises(KeyError):
            source.sample("episode_0", 1)
        report = source.report()
        self.assertEqual(report["contract"], RECOVERY_CONTRACT)
        self.assertEqual(report["excluded_fault_commands"], 2)
        self.assertEqual(report["parent_splits"], {"train": ["episode_0"], "validation": ["episode_1"]})

    def test_export_is_segmented_without_fault_rows(self):
        source = RecoveryVLASource(self.root)
        FakeDataset.instances = []
        receipt = export_source(source, Path(self.temp.name) / "export", "local/test", FakeDataset)
        self.assertEqual(receipt["contract"], RECOVERY_CONTRACT)
        for dataset in FakeDataset.instances:
            self.assertEqual(dataset.saved, 2)
            self.assertEqual(len(dataset.frames), 2)
            self.assertTrue(all(frame["action"][7] < 1 for frame in dataset.frames))
            self.assertEqual(set(dataset.frames[0]), {"task", "action", "observation.state",
                                                     "observation.images.camera1", "observation.images.camera2"})

    def test_mismatched_label_counts_rejected(self):
        self.manifest["episodes"][0]["label_counts"]["fault"] = 2
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, "label counts"):
            RecoveryVLASource(self.root)

    def test_correction_index_bound_to_source_and_nonfault_segments(self):
        source = RecoveryVLASource(self.root)
        destination = Path(self.temp.name) / "sampling_export"
        receipt = export_source(source, destination, "local/sampling_test", FakeDataset)
        index = build_index(self.root, destination)
        verify_index(index, self.root, destination)
        self.assertEqual(sum(map(len, index["partitions"]["train"].values())), 2)
        self.assertEqual(sum(map(len, index["partitions"]["validation"].values())), 2)
        index["partitions"]["train"]["expert"].append(999)
        with self.assertRaises(ValueError):
            verify_index(index, self.root, destination)
        receipt["datasets"]["train"]["episodes"][0]["parent_stop_exclusive"] = 2
        receipt["datasets"]["train"]["episodes"][0]["steps"] = 2
        (destination / "openarm_export.json").write_text(json.dumps(receipt))
        with self.assertRaisesRegex(ValueError, "Fault"):
            build_index(self.root, destination)

    def test_split_planning_is_parent_level_and_immutable(self):
        for record in self.manifest["episodes"]:
            record["split"] = "train"
        self.write_manifest()
        before = (self.root / "manifest.json").read_bytes()
        plan = make_split_plan([self.root / "manifest.json"])
        self.assertEqual(plan, make_split_plan([self.root / "manifest.json"]))
        self.assertEqual(Counter(r["split"] for r in plan["episodes"]), {"train": 1, "validation": 1})
        target = Path(self.temp.name) / "view"
        prepare_view(plan, [self.root], target)
        source = RecoveryVLASource(target)
        self.assertEqual((self.root / "manifest.json").read_bytes(), before)
        self.assertTrue(os.path.samefile(self.root / "episode_0/episode.npz", target / "episode_0/episode.npz"))
        for parent in source.parent_records:
            self.assertTrue(all(r["split"] == parent["split"] for r in source.records
                                if r["parent_episode"] == parent["episode"]))

    def test_missing_full_release_and_changed_manifest_refused(self):
        for record in self.manifest["episodes"]:
            record["split"] = "train"
        self.write_manifest()
        plan = make_split_plan([self.root / "manifest.json"])
        extra = {**plan["episodes"][0], "episode": "missing_parent"}
        plan["episodes"].append(extra)
        target = Path(self.temp.name) / "view"
        with self.assertRaisesRegex(ValueError, "full release"):
            prepare_view(plan, [self.root], target)
        self.assertFalse(target.exists())
        prepare_view(plan, [self.root], target, allow_partial=True)
        self.assertTrue(RecoveryVLASource(target).report()["partial_release"])
        self.manifest["complete"] = False
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, "manifest changed"):
            prepare_view(plan, [self.root], Path(self.temp.name) / "changed")


if __name__ == "__main__":
    unittest.main()
