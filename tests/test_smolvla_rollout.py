"""Lightweight provenance checks; the real model/simulator run is separate."""
import copy
from pathlib import Path
import tempfile
import unittest

from learning.vla.smolvla_data import sha256
from scripts.rollout_smolvla import verify_snapshot


class RolloutProvenanceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "snapshot"
        self.root.mkdir()
        contents = {"envs/scene.py": "scene", "configs/workspace.json": "{}",
                    "third_party/openarm_mujoco/v1/openarm_bimanual.xml": "robot"}
        for rel, content in contents.items():
            path = self.root / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content)
        self.metadata = {
            "source_hashes": {"envs/scene.py": sha256(self.root / "envs/scene.py")},
            "workspace": {"path": "configs/workspace.json",
                          "sha256": sha256(self.root / "configs/workspace.json")},
            "robot": {"robot_xml_sha256": sha256(
                self.root / "third_party/openarm_mujoco/v1/openarm_bimanual.xml")}}

    def test_matching_snapshot(self):
        self.assertEqual(verify_snapshot(self.root, self.metadata),
                         (self.root / "configs/workspace.json").resolve())

    def test_changed_source_workspace_and_robot_rejected(self):
        for path in ("envs/scene.py", "configs/workspace.json",
                     "third_party/openarm_mujoco/v1/openarm_bimanual.xml"):
            with self.subTest(path=path):
                target = self.root / path
                original = target.read_bytes()
                target.write_bytes(b"changed")
                try:
                    with self.assertRaises(ValueError):
                        verify_snapshot(self.root, self.metadata)
                finally:
                    target.write_bytes(original)

    def test_source_path_escape_rejected(self):
        metadata = copy.deepcopy(self.metadata)
        outside = self.root.parent / "outside.py"
        outside.write_text("outside")
        metadata["source_hashes"] = {"../outside.py": sha256(outside)}
        with self.assertRaises(ValueError):
            verify_snapshot(self.root, metadata)

    def test_workspace_path_escape_rejected(self):
        metadata = copy.deepcopy(self.metadata)
        outside = self.root.parent / "outside.json"
        outside.write_text("{}")
        metadata["workspace"] = {"path": "../outside.json", "sha256": sha256(outside)}
        with self.assertRaises(ValueError):
            verify_snapshot(self.root, metadata)


if __name__ == "__main__":
    unittest.main()
