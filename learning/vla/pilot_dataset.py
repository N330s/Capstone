"""Raw physical-unit, full-window samples for stock pi0 flow-matching loss."""
import json
from pathlib import Path
import numpy as np
from data_pipeline.episodes import SCHEMA, validate_episode


class FullWindowPilotDataset:
    def __init__(self, root, split="train", horizon=50, episode_limit=None):
        if split not in ("train", "validation") or type(horizon) is not int or horizon < 1:
            raise ValueError("Invalid split or horizon")
        if episode_limit is not None and (type(episode_limit) is not int or episode_limit < 1):
            raise ValueError("Invalid episode limit")
        self.root, self.horizon = Path(root), horizon
        manifest = json.loads((self.root / "manifest.json").read_text())
        if manifest["schema"] != SCHEMA:
            raise ValueError("Unknown pilot schema")
        records = manifest["episodes"]
        names = [r["episode"] for r in records]
        if len(names) != len(set(names)) or any(r["split"] not in ("train", "validation") for r in records):
            raise ValueError("Duplicate episodes or unknown split")
        self.records = [r for r in records if r["split"] == split][:episode_limit]
        self.index = []
        self.instruction = manifest["instruction"]
        for record in self.records:
            path = (self.root / record["episode"]).resolve()
            if path.parent != self.root.resolve():
                raise ValueError("Episode path escapes dataset")
            meta = validate_episode(path)
            if meta["data_sha256"] != record["data_sha256"] or meta["steps"] != record["steps"]:
                raise ValueError("Manifest differs from episode")
            self.index.extend((record["episode"], i) for i in range(record["steps"] - horizon + 1))
        if not self.index:
            raise ValueError("No complete action windows in selected split")
        self._cached_name, self._cached = None, None

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        name, step = self.index[index]
        if name != self._cached_name:
            with np.load(self.root / name / "episode.npz", allow_pickle=False) as raw:
                self._cached = {key: raw[key] for key in (
                    "observation_state", "action_applied", "image_scene_rgb", "image_wrist_rgb")}
            self._cached_name = name
        data = self._cached
        return {"images": {camera: data["image_" + camera][step].copy()
                           for camera in ("scene_rgb", "wrist_rgb")},
                "state": data["observation_state"][step].copy(),
                "instruction": self.instruction,
                "actions": data["action_applied"][step:step + self.horizon].copy(),
                "action_mask": np.ones(self.horizon, dtype=bool)}
