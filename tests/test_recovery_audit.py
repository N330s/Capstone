import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/audit_recovery_collections.py"
spec = importlib.util.spec_from_file_location("recovery_audit", SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class RecoveryAuditTests(unittest.TestCase):
    def test_headers_and_streamed_frame_match(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "episode.npz"
            images = np.arange(4 * 5 * 6 * 3, dtype=np.uint8).reshape(4, 5, 6, 3)
            depth = np.arange(4 * 5 * 6, dtype=np.uint16).reshape(4, 5, 6)
            np.savez_compressed(path, image_scene_rgb=images, depth_scene=depth)
            headers = module.array_headers(path)
            self.assertEqual(headers["depth_scene"]["dtype"], "uint16")
            self.assertEqual(headers["image_scene_rgb"]["shape"], [4, 5, 6, 3])
            np.testing.assert_array_equal(module.frame_from_npz(path, "image_scene_rgb", 2), images[2])
            np.testing.assert_array_equal(module.frame_from_npz(path, "depth_scene", 3), depth[3])

    def test_object_arrays_refused(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "episode.npz"
            np.savez_compressed(path, state=np.array([{}], dtype=object))
            with self.assertRaises(ValueError):
                module.array_headers(path)


if __name__ == "__main__":
    unittest.main()
