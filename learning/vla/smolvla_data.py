"""Strict raw OpenArm data boundary; normalization/padding belong to LeRobot.

No torch, simulator, or checkpoint imports are required for validation.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from data_pipeline.episodes import SCHEMA, validate_episode

CONTRACT = "openarm-v1-smolvla-absolute-v1"
LEROBOT_VERSION = "0.4.4"
CAMERAS = {"observation.images.camera1": "scene_rgb",
           "observation.images.camera2": "wrist_rgb"}
STATE_NAMES = [name for side in ("left", "right") for name in
               [*[f"{side}_joint_{i}_rad" for i in range(1, 8)], f"{side}_finger_travel_m"]]
ACTION_NAMES = [*[f"right_joint_{i}_rad" for i in range(1, 8)], "right_finger_travel_m"]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def observation(raw):
    """Unbatched physical-unit state and CHW [0,1] images, before processors."""
    state = np.asarray(raw["state"], dtype=np.float32)
    if state.shape != (16,) or not np.isfinite(state).all():
        raise ValueError("Expected finite 16-element bimanual state")
    if not isinstance(raw["instruction"], str) or not raw["instruction"].strip():
        raise ValueError("Nonempty instruction required")
    result = {"observation.state": state.copy(), "task": raw["instruction"]}
    for dest, source in CAMERAS.items():
        image = np.asarray(raw["images"][source])
        if image.ndim != 3 or image.shape[-1] != 3 or image.dtype != np.uint8:
            raise ValueError("Expected HWC uint8 RGB")
        result[dest] = image.transpose(2, 0, 1).astype(np.float32) / 255.0
    return result  # Whitelist: never forward qpos, contacts, phases or reward.


def decoded_actions(value):
    """Validate AFTER upstream postprocessing; do not normalize or clip twice."""
    actions = np.asarray(value, dtype=np.float64)
    if actions.ndim != 2 or actions.shape[1] != 8 or not len(actions) or not np.isfinite(actions).all():
        raise ValueError("Expected postprocessed, finite H x 8 physical-unit actions")
    return actions.copy()  # Shared environment owns bounds and rate limiting.


class SmolVLASource:
    """Validate all episodes once, cache at most one episode for sampling/export.

    Existing successful-expert NPZ schema only. Full-task collectors can target
    this schema or add a separately validated reader; traces alone are not data.
    """

    supports_recovery_labels = False

    def __init__(self, root, selection=None):
        self.root = Path(root).resolve()
        self.manifest = json.loads((self.root / "manifest.json").read_text(encoding="utf-8"))
        if self.manifest.get("schema") != SCHEMA:
            raise ValueError("Unsupported source schema")
        self.instruction = self.manifest.get("instruction")
        if not isinstance(self.instruction, str) or not self.instruction.strip():
            raise ValueError("Missing instruction")
        self.records = self.manifest["episodes"]
        original_names = [r["episode"] for r in self.records]
        if len(original_names) != len(set(original_names)):
            raise ValueError("Duplicate episode in source manifest")
        self.selection = None
        if selection is not None:
            self.selection = json.loads(Path(selection).read_text(encoding="utf-8"))
            if set(self.selection["splits"]) != {"train", "validation"}:
                raise ValueError("Selection must define train and validation")
            selected = [name for names in self.selection["splits"].values() for name in names]
            if len(selected) != len(set(selected)) or not set(selected) <= set(original_names):
                raise ValueError("Selection duplicates episodes or references unknown episodes")
            by_name = {r["episode"]: r for r in self.records}
            self.records = [{**by_name[name], "split": split}
                            for split, names in self.selection["splits"].items() for name in names]
        self.excluded = [name for name in original_names if name not in {r["episode"] for r in self.records}]
        missing = [r["episode"] for r in self.records if not (self.root / r["episode"] / "episode.npz").is_file()]
        if missing:
            raise ValueError(f"Missing selected episodes: {missing}; restore files or provide an explicit smoke selection")
        if not self.records:
            raise ValueError("Empty dataset")
        self.metadata = {}
        names, hashes = set(), set()
        fingerprint = None
        self.image_shape = None
        for record in self.records:
            name = record["episode"]
            path = (self.root / name).resolve()
            if path.parent != self.root or path.name != name or name in names:
                raise ValueError("Duplicate or unsafe episode path")
            names.add(name)
            if record["split"] not in ("train", "validation"):
                raise ValueError("Unknown split; evaluation data must stay separate")
            if type(record["steps"]) is not int or record["steps"] < 1:
                raise ValueError("Episode must have positive integer steps")
            meta = validate_episode(path)
            if (meta["steps"] != record["steps"] or meta["data_sha256"] != record["data_sha256"]
                    or meta.get("dataset_schema") != SCHEMA or meta["instruction"] != self.instruction):
                raise ValueError("Manifest/episode metadata mismatch")
            if meta["data_sha256"] in hashes:
                raise ValueError("Duplicated episode contents, possibly split leakage")
            hashes.add(meta["data_sha256"])
            if meta.get("action_source") != "privileged_scripted_expert":
                raise ValueError("Unreviewed action source for behavioral cloning")
            robot, cfg = meta["robot"], meta["robot_config"]
            if (meta.get("schema") not in ("openarm_v1_joint_targets_v1", "openarm_v1_joint_targets_v3") or robot["active_arm"] != "right"
                    or robot["parked_arm"] != "left" or cfg["command_hz"] != 50):
                raise ValueError("Unsupported robot/action/timing contract")
            if meta["schema"] == "openarm_v1_joint_targets_v3":
                expected_actions = [f"openarm_right_joint{i}" for i in range(1, 8)] + ["right_finger_travel"]
                if (meta.get("action_names") != expected_actions or meta.get("state_order") !=
                        "left joint1..7, left mean finger travel, right joint1..7, right mean finger travel"):
                    raise ValueError("Unexpected v3 action/state ordering")
            # Fail closed on mixed scene/controller releases until deliberately reviewed.
            current = json.dumps({k: meta[k] for k in ("robot", "robot_config", "scene_sha256",
                                                       "source_hashes", "physics_timestep_s")} |
                                 {k: meta.get(k) for k in ("workspace", "contact_config", "command_interval_s",
                                                         "mujoco_version")}, sort_keys=True)
            if fingerprint is not None and current != fingerprint:
                raise ValueError("Mixed scene/controller versions require a separate dataset release")
            fingerprint = current
            self.metadata[name] = meta
            with np.load(path / "episode.npz", allow_pickle=False) as arrays:
                if ("step_label" in arrays.files or meta.get("fault") is not None
                        or record.get("fault") is not None) and not self.supports_recovery_labels:
                    raise ValueError("Recovery recordings require RecoveryVLASource; legacy reader must not train fault actions")
                t = record["steps"]
                if len(arrays["action_applied"]) != t:
                    raise ValueError("Recorded step count differs from arrays")
                for key in ("terminated", "truncated"):
                    if arrays[key].shape != (t,) or arrays[key].dtype != np.bool_:
                        raise ValueError("Invalid terminal flags")
                for key in ("executed_dt_s", "reward", "phase"):
                    if arrays[key].shape != (t,):
                        raise ValueError(f"Invalid transition field: {key}")
                times = arrays["timestamp_s"]
                if (times.shape != (t + 1,) or not np.isfinite(times).all()
                        or not np.isfinite(arrays["reward"]).all()
                        or not np.allclose(np.diff(times)[:-1], .02, atol=1e-9, rtol=0)
                        or not np.allclose(arrays["executed_dt_s"][:-1], .02, atol=1e-9, rtol=0)
                        or not 0 < arrays["executed_dt_s"][-1] <= .02 + 1e-9):
                    raise ValueError("Expected 50 Hz commands; only the terminal interval may be partial")
                self.image_shape = arrays["image_scene_rgb"].shape[1:]
        if {r["split"] for r in self.records} != {"train", "validation"}:
            raise ValueError("Both train and validation episodes are required")
        self._cached_name, self._cached = None, None

    def arrays(self, name):
        if name not in self.metadata:
            raise KeyError(name)
        if self._cached_name != name:
            path = self.root / name / "episode.npz"
            if sha256(path) != self.metadata[name]["data_sha256"]:
                raise ValueError("Source changed after validation")
            with np.load(path, allow_pickle=False) as data:
                self._cached = {key: data[key] for key in (
                    "observation_state", "action_applied", "image_scene_rgb", "image_wrist_rgb")}
            self._cached_name = name
        return self._cached

    def sample(self, name, step, horizon=50):
        if type(horizon) is not int or horizon < 1:
            raise ValueError("Positive integer horizon required")
        data = self.arrays(name)
        t = len(data["action_applied"])
        if type(step) is not int or not 0 <= step < t:
            raise IndexError(step)
        sample = observation({"state": data["observation_state"][step],
                              "images": {camera: data["image_" + camera][step] for camera in CAMERAS.values()},
                              "instruction": self.instruction})
        indices = np.minimum(np.arange(step, step + horizon), t - 1)
        sample["action"] = data["action_applied"][indices].astype(np.float32)
        # LeRobot uses True for PAD, the opposite of our old valid-action mask.
        sample["action_is_pad"] = np.arange(step, step + horizon) >= t
        return sample

    def report(self):
        return {"contract": CONTRACT, "source_manifest_sha256": sha256(self.root / "manifest.json"),
                "selection": self.selection, "excluded_manifest_episodes": self.excluded,
                "source_robot_schemas": sorted({m["schema"] for m in self.metadata.values()}),
                "fps": 50, "image_shape_hwc": list(self.image_shape), "camera_mapping": CAMERAS,
                "state_names": STATE_NAMES, "action_names": ACTION_NAMES,
                "normalization_owner": "upstream SmolVLA processors; train split only",
                "splits": {split: {"episodes": sum(r["split"] == split for r in self.records),
                                   "steps": sum(r["steps"] for r in self.records if r["split"] == split)}
                           for split in ("train", "validation")},
                "task_coverage": "unverified; validation does not establish full-task coverage",
                "terminal_timing": "partial final interval allowed; exact durations remain in raw source",
                "model_runtime_tested": False}

    def features(self):
        return {"observation.state": {"dtype": "float32", "shape": (16,), "names": STATE_NAMES},
                "action": {"dtype": "float32", "shape": (8,), "names": ACTION_NAMES},
                **{key: {"dtype": "video", "shape": self.image_shape,
                         "names": ["height", "width", "channels"]} for key in CAMERAS}}


def export_source(source, destination, repo_prefix, dataset_class=None):
    """Separate physical datasets prevent validation-derived normalization stats.

    No Hub upload. Terminal observations remain in the immutable source; no fake
    terminal action row is inserted into the BC dataset.
    """
    destination = Path(destination).resolve()
    if destination.exists():
        raise FileExistsError("Export destination must not exist; no overwrite/resume")
    if destination.is_relative_to(source.root) or source.root.is_relative_to(destination):
        raise ValueError("Source and export roots must be disjoint")
    if dataset_class is None:
        from importlib.metadata import version
        if version("lerobot") != LEROBOT_VERSION:
            raise RuntimeError(f"Install lerobot[smolvla]=={LEROBOT_VERSION} in a separate environment")
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        dataset_class = LeRobotDataset
    destination.mkdir(parents=True)
    receipt = {**source.report(), "lerobot_version": LEROBOT_VERSION, "datasets": {}}
    # Receipt is written only after both exports finalize. Partial dirs are not training-ready.
    for split in ("train", "validation"):
        records = [r for r in source.records if r["split"] == split]
        repo_id = f"{repo_prefix}_{split}"
        dataset = dataset_class.create(repo_id=repo_id, root=destination / split, fps=50,
                                       features=source.features(), robot_type="openarm_v1_bimanual_right",
                                       use_videos=True, vcodec="h264", image_writer_threads=0)
        for record in records:
            data = source.arrays(record["episode"])
            for step in range(record["steps"]):
                frame = {"observation.state": data["observation_state"][step].astype(np.float32),
                         "action": data["action_applied"][step].astype(np.float32),
                         "task": source.instruction}
                frame.update({key: data["image_" + camera][step].copy() for key, camera in CAMERAS.items()})
                dataset.add_frame(frame)
            dataset.save_episode()
        dataset.finalize()
        receipt["datasets"][split] = {"repo_id": repo_id, "relative_root": split,
                                      "episodes": records}
    (destination / "openarm_export.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    return receipt
