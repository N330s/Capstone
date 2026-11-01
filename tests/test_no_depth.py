"""collect_random --no-depth: the env records no depth, and the run signature only changes when it is set."""
import sys
import unittest
from argparse import Namespace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

WS_PATH = ROOT / "configs/workspace_v2_cable30.json"


class NoDepthTests(unittest.TestCase):
    def test_make_env_drops_depth(self):
        import collect_random
        full = collect_random.make_env(Namespace(workspace=WS_PATH, no_depth=False))
        self.assertTrue(full.depth_cameras)
        self.assertIsNotNone(full.manifest()["depth_cameras"])
        env = collect_random.make_env(Namespace(workspace=WS_PATH, no_depth=True))
        self.assertEqual(env.depth_cameras, {})
        self.assertIsNone(env.manifest()["depth_cameras"])

    def test_signature_key_only_when_active(self):
        import collect_random
        base = dict(workspace=WS_PATH, detector="noisy", validation_fraction=0.1, start_row=0, faults=None)
        self.assertNotIn("no_depth", collect_random.run_signature(Namespace(no_depth=False, **base)))
        self.assertTrue(collect_random.run_signature(Namespace(no_depth=True, **base))["no_depth"])


if __name__ == "__main__":
    unittest.main()
