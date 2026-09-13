"""Dependency-light pi0 adapter shared by training and inference.

This adapter owns normalization and numeric padding. Configure upstream numeric
normalization as an empty dictionary; never also apply ALOHA/DROID transforms.
Image resizing, prompt tokenization and uint8-to-model conversion stay upstream.
"""
from dataclasses import dataclass
import numpy as np

SCHEMA = "openarm-v1-pi0-absolute-v1"
MODEL_DIM = 32
CAMERAS = {"base_0_rgb": "scene_rgb", "right_wrist_0_rgb": "wrist_rgb"}


def finite_array(value, shape, name):
    array = np.asarray(value, dtype=np.float64)
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite with shape {shape}")
    return array


def rgb_image(value):
    array = np.asarray(value)
    if array.dtype == np.uint8 and array.ndim == 3 and array.shape[-1] == 3:
        return array.copy()
    # LeRobot decodes images to CHW floats in [0, 1]. Do not guess other formats.
    if (array.ndim == 3 and array.shape[0] == 3
            and np.issubdtype(array.dtype, np.floating)
            and np.isfinite(array).all() and array.min() >= 0 and array.max() <= 1):
        return np.rint(array.transpose(1, 2, 0) * 255).astype(np.uint8)
    raise ValueError("Expected HWC uint8 RGB or CHW float RGB in [0, 1]")


@dataclass(frozen=True)
class OpenArmCodec:
    stats: dict

    def __post_init__(self):
        if self.stats.get("fit_split") != "train":
            raise ValueError("Normalization must be fit on training episodes only")
        for name, size in (("state", 16), ("action", 8)):
            item = self.stats["transforms"][name]
            finite_array(item["mean"], (size,), name + " mean")
            scale = finite_array(item["scale"], (size,), name + " scale")
            if np.any(scale <= 0):
                raise ValueError("Normalization scale must be positive")

    def encode_numeric(self, key, value):
        size = 16 if key == "state" else 8 if key == "action" else None
        if size is None:
            raise ValueError(key)
        value = np.asarray(value, dtype=np.float64)
        expected = (size,) if key == "state" else (len(value), size) if value.ndim == 2 else ()
        value = finite_array(value, expected, key)
        if key == "action" and (value.ndim != 2 or not len(value)):
            raise ValueError("Expected nonempty H x 8 actions")
        item = self.stats["transforms"][key]
        normalized = (value - item["mean"]) / np.asarray(item["scale"])
        result = np.zeros((*normalized.shape[:-1], MODEL_DIM), dtype=np.float32)
        result[..., :size] = normalized
        if not np.isfinite(result).all():
            raise ValueError("Normalized values exceed float32 range")
        return result

    def decode_actions(self, value):
        value = np.asarray(value)
        if value.ndim != 2 or not len(value) or value.shape[1] != MODEL_DIM or not np.isfinite(value).all():
            raise ValueError("Expected finite nonempty H x 32 model actions")
        item = self.stats["transforms"]["action"]
        # Padding is not a command for the parked arm. No actuator clipping here:
        # the shared environment applies limits and logs the applied command.
        return value[:, :8].astype(np.float64) * item["scale"] + item["mean"]


@dataclass(frozen=True)
class OpenArmInputs:
    codec: OpenArmCodec

    def __call__(self, data):
        images = {dest: rgb_image(data["images"][source]) for dest, source in CAMERAS.items()}
        if images["base_0_rgb"].shape != images["right_wrist_0_rgb"].shape:
            raise ValueError("Pilot camera shapes must agree")
        # Keep upstream camera ordering stable, including an explicitly masked left wrist.
        images = {"base_0_rgb": images["base_0_rgb"],
                  "left_wrist_0_rgb": np.zeros_like(images["base_0_rgb"]),
                  "right_wrist_0_rgb": images["right_wrist_0_rgb"]}
        prompt = data["instruction"]
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("A nonempty instruction string is required")
        result = {"image": images,
                  "image_mask": {k: np.bool_(k != "left_wrist_0_rgb") for k in images},
                  "state": self.codec.encode_numeric("state", data["state"]), "prompt": prompt}
        if "actions" in data:
            # Stock pi0 loss has no terminal mask. Only complete windows are accepted.
            mask = np.asarray(data["action_mask"])
            if mask.dtype != np.bool_ or mask.shape != (len(data["actions"]),) or not mask.all():
                raise ValueError("pi0 adapter requires complete, unpadded action windows")
            result["actions"] = self.codec.encode_numeric("action", data["actions"])
        return result  # Whitelist only: never propagate diagnostics, qpos, or timestamps.


@dataclass(frozen=True)
class OpenArmOutputs:
    codec: OpenArmCodec

    def __call__(self, data):
        return {"actions": self.codec.decode_actions(data["actions"])}
