import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

import numpy as np
from test_smolvla import fixture

ROOT = Path(__file__).resolve().parents[1]


class CollectionReleaseTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "source"
        manifest = fixture(self.root)
        for n in range(2, 5):
            target = self.root / f"episode_{n}"
            shutil.copytree(self.root / "episode_1", target)
            with np.load(target / "episode.npz", allow_pickle=False) as arrays:
                data = {k: arrays[k] for k in arrays.files}
            data["observation_state"] = np.full_like(data["observation_state"], n)
            np.savez_compressed(target / "episode.npz", **data)
            digest = hashlib.sha256((target / "episode.npz").read_bytes()).hexdigest()
            meta = json.loads((target / "metadata.json").read_text())
            meta["data_sha256"] = digest
            (target / "metadata.json").write_text(json.dumps(meta))
            manifest["episodes"].append({"episode": target.name, "steps": 3, "data_sha256": digest})
        for n, record in enumerate(manifest["episodes"]):
            record.update(split="train", seed=1000 + n)
        manifest["complete"] = True
        self.manifest = manifest
        (self.root / "manifest.json").write_text(json.dumps(manifest))

    def run_merge(self, output, *sources):
        return subprocess.run([sys.executable, str(ROOT / "scripts/merge_collection_release.py"),
                               "--sources", *map(str, sources or (self.root,)), "--output", str(output)],
                              capture_output=True, text=True)

    def test_explicit_holdout_leaves_original_unchanged(self):
        output = Path(self.temp.name) / "merged"
        before = (self.root / "manifest.json").read_bytes()
        result = self.run_merge(output)
        self.assertEqual(result.returncode, 0, result.stderr)
        manifest = json.loads((output / "manifest.json").read_text())
        self.assertEqual(sum(r["split"] == "validation" for r in manifest["episodes"]), 1)
        self.assertEqual((self.root / "manifest.json").read_bytes(), before)
        self.assertTrue(os.path.samefile(self.root / "episode_0/episode.npz", output / "episode_0/episode.npz"))

    def test_duplicate_names_and_evaluation_namespace_rejected(self):
        output = Path(self.temp.name) / "duplicate"
        result = self.run_merge(output, self.root, self.root)
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(output.exists())
        self.manifest["episodes"][0]["seed"] = 9000000
        (self.root / "manifest.json").write_text(json.dumps(self.manifest))
        result = self.run_merge(output)
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
