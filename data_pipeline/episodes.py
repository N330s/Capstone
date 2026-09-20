"""Raw, lossless episodes with T actions and T+1 synchronized observations."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np

SCHEMA = "openarm-pilot-v1"


def clean_info(info):
    return {k:v.tolist() if isinstance(v,np.ndarray) else v for k,v in info.items()}


def save_episode(path, observations, requested, applied, states, velocities,
                 rewards, terminated, truncated, durations, phases, infos, manifest):
    path = Path(path)
    path.mkdir(parents=True,exist_ok=False)
    arrays = {
        "observation_state":np.stack([o["state"] for o in observations]),
        "timestamp_s":np.array([o["timestamp_s"] for o in observations]),
        "action_requested":np.stack(requested),
        "action_applied":np.stack(applied),
        "qpos":np.stack(states), "qvel":np.stack(velocities),
        "reward":np.array(rewards), "terminated":np.array(terminated,dtype=bool),
        "truncated":np.array(truncated,dtype=bool), "executed_dt_s":np.array(durations),
        "phase":np.array(phases),
    }
    for camera in observations[0]["images"]:
        arrays[f"image_{camera}"]=np.stack([o["images"][camera] for o in observations])
    np.savez_compressed(path/"episode.npz",**arrays)
    metadata = {**manifest,"dataset_schema":SCHEMA,"action_source":"privileged_scripted_expert",
                "instruction":observations[0]["instruction"], "steps":len(requested),
                "data_sha256":hashlib.sha256((path/"episode.npz").read_bytes()).hexdigest(),
                "outcome":infos[-1]["outcome"],
                "timing":"observation[i] precedes action[i]; observation[i+1] follows its executed interval"}
    (path/"metadata.json").write_text(json.dumps(metadata,indent=2),encoding="utf-8")
    (path/"diagnostics.json").write_text(json.dumps([clean_info(i) for i in infos],indent=2),encoding="utf-8")
    validate_episode(path)
    return metadata


def validate_episode(path):
    path=Path(path)
    meta=json.loads((path/"metadata.json").read_text())
    if hashlib.sha256((path/"episode.npz").read_bytes()).hexdigest()!=meta["data_sha256"]:
        raise ValueError("Episode checksum mismatch")
    with np.load(path/"episode.npz",allow_pickle=False) as a:
        t=len(a["action_requested"])
        if a["action_requested"].shape!=(t,8) or a["action_applied"].shape!=(t,8):
            raise ValueError("Action shape mismatch")
        if a["observation_state"].shape!=(t+1,16):
            raise ValueError("Observation/action alignment mismatch")
        if not np.allclose(np.diff(a["timestamp_s"]),a["executed_dt_s"],atol=1e-10,rtol=0):
            raise ValueError("Timestamp / executed interval mismatch")
        for key in ("observation_state","action_requested","action_applied","qpos","qvel"):
            if not np.isfinite(a[key]).all():
                raise ValueError(f"Non-finite data in {key}")
        for key in ("image_scene_rgb","image_wrist_rgb"):
            if a[key].shape!=(t+1,meta["robot_config"]["image_height"],meta["robot_config"]["image_width"],3) or a[key].dtype!=np.uint8:
                raise ValueError(f"Image schema mismatch: {key}")
        done=a["terminated"]|a["truncated"]
        if done[:-1].any() or not done[-1]:
            raise ValueError("Episode boundary mismatch")
        if np.any(a["terminated"] & a["truncated"]):
            raise ValueError("Terminal and truncation flags overlap")
        if meta.get("action_source") != "human_manual" and meta["outcome"] != "success":
            raise ValueError("Pilot BC export accepts successful expert episodes only")
    return meta


def action_chunk(path, index, horizon):
    """Return a masked, same-episode physical-unit chunk; never use reset actions."""
    if horizon<1 or index<0:
        raise ValueError("Invalid chunk index/horizon")
    with np.load(Path(path)/"episode.npz",allow_pickle=False) as a:
        if index>=len(a["action_applied"]):
            raise IndexError(index)
        chunk=a["action_applied"][index:index+horizon].copy()
    valid=len(chunk)
    mask=np.arange(horizon)<valid
    # Hold last command for padded values, but exclude padding from training loss.
    chunk=np.concatenate([chunk,np.repeat(chunk[-1:],horizon-valid,axis=0)])
    return chunk,mask

