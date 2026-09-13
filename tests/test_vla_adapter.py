import unittest
import numpy as np
from learning.vla.openpi_adapter import OpenArmCodec, OpenArmInputs, OpenArmOutputs


def fixture():
    stats = {"fit_split": "train", "transforms": {
        k: {"mean": np.arange(n).tolist(), "scale": np.linspace(.001, 1, n).tolist()}
        for k, n in (("state", 16), ("action", 8))}}
    raw = {"state": np.arange(16) + .1,
           "images": {k: np.full((24, 32, 3), i, np.uint8)
                      for k, i in (("scene_rgb", 25), ("wrist_rgb", 200))},
           "instruction": "Insert the plug into the socket.",
           "actions": np.tile(np.arange(8) + .01, (50, 1)),
           "action_mask": np.ones(50, dtype=bool)}
    return OpenArmCodec(stats), raw


class AdapterTests(unittest.TestCase):
    def test_float32_roundtrip_and_padding(self):
        codec, raw = fixture()
        sample = OpenArmInputs(codec)(raw)
        self.assertEqual(sample["state"].shape, (32,))
        self.assertEqual(sample["actions"].shape, (50, 32))
        self.assertTrue(np.all(sample["actions"][:, 8:] == 0))
        self.assertTrue(np.all(sample["state"][16:] == 0))
        np.testing.assert_allclose(OpenArmOutputs(codec)(sample)["actions"], raw["actions"], atol=1e-7, rtol=0)

    def test_camera_mapping_and_leakage(self):
        codec, raw = fixture()
        raw.update(qpos="secret", contact_force=42, timestamp_s=1)
        sample = OpenArmInputs(codec)(raw)
        self.assertEqual(set(sample), {"image", "image_mask", "state", "prompt", "actions"})
        self.assertFalse(sample["image_mask"]["left_wrist_0_rgb"])
        self.assertTrue(np.all(sample["image"]["left_wrist_0_rgb"] == 0))
        self.assertTrue(np.all(sample["image"]["right_wrist_0_rgb"] == 200))

    def test_lerobot_image_conversion(self):
        codec, raw = fixture()
        expected = OpenArmInputs(codec)(raw)
        raw["images"] = {k: v.transpose(2, 0, 1).astype(np.float32) / 255 for k, v in raw["images"].items()}
        actual = OpenArmInputs(codec)(raw)
        for key in expected["image"]:
            np.testing.assert_array_equal(expected["image"][key], actual["image"][key])

    def test_terminal_padding_rejected(self):
        codec, raw = fixture()
        raw["action_mask"][-1] = False
        with self.assertRaises(ValueError):
            OpenArmInputs(codec)(raw)

    def test_bad_inputs_rejected(self):
        codec, raw = fixture()
        for bad in (np.zeros(14), np.full(16, np.nan)):
            with self.assertRaises(ValueError):
                OpenArmInputs(codec)({**raw, "state": bad})
        for bad in (np.zeros((50, 8)), np.full((50, 32), np.nan), np.zeros((0, 32))):
            with self.assertRaises(ValueError):
                codec.decode_actions(bad)
        with self.assertRaises(ValueError):
            OpenArmCodec({**codec.stats, "fit_split": "validation"})

    def test_inference_has_no_actions_or_mask_requirement(self):
        codec, raw = fixture()
        raw.pop("actions")
        raw.pop("action_mask")
        self.assertNotIn("actions", OpenArmInputs(codec)(raw))

    def test_padding_never_commands_left_arm(self):
        codec, raw = fixture()
        encoded = codec.encode_numeric("action", raw["actions"])
        expected = codec.decode_actions(encoded)
        encoded[:, 8:] = 999
        np.testing.assert_array_equal(codec.decode_actions(encoded), expected)


if __name__ == "__main__":
    unittest.main()
