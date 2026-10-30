"""Raw, lossless episodes with T actions and T+1 synchronized observations."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np

SCHEMA = "openarm-pilot-v1"
# Optional per-step label of injected-fault recovery episodes (controllers/faults.py). Absent in
# ordinary episodes. "fault" steps are injected bad commands: mask them before BC.
STEP_LABELS = ("expert", "fault", "recovery")
# Per-joint torque over each action interval (env info, left joint1..7 then right joint1..7, N m).
TORQUE_KEYS = ("joint_torque_applied_nm", "joint_torque_applied_peak_nm", "joint_torque_gravity_nm",
               "joint_torque_constraint_nm")


PILOT_CAMERAS = ("scene_rgb", "wrist_rgb")


def pilot_contract(manifest=None):
    """The policy-facing control contract every pilot episode follows (also stored in releases)."""
    manifest=manifest or {}
    return {"cameras":list(PILOT_CAMERAS),"state":16,
            "state_order":manifest.get("state_order","left joint1..7, left mean finger travel, "
                                       "right joint1..7, right mean finger travel"),
            "action":8,
            "action_names":manifest.get("action_names",[f"openarm_right_joint{i}" for i in range(1,8)]
                                        +["right_finger_travel"]),
            "action_units":"right joint1..7 absolute targets in rad; finger slide travel in m (0..0.044)",
            "rate_hz":50,
            "timing":"observation i precedes action i; observation i+1 follows its executed interval",
            "timing_note":"executed_dt_s is 0.02 on every step except the terminal one, which may be "
                          "shorter because the episode ends mid-interval on success",
            "left_arm":"parked",
            "label_arrays":["action_label","label_mask"],
            "extra_arrays_not_policy_inputs":["image_chest_rgb","depth_*","joint_torque_*",
                                              "visibility_pilot"]}


def clean_info(info):
    return {k:v.tolist() if isinstance(v,np.ndarray) else v for k,v in info.items()}


# Rendered camera -> folder under <episode>/cameras/ for the per-frame JPEG export.
CAMERA_FOLDERS = {"scene_rgb":"head", "wrist_rgb":"wrist", "chest_rgb":"chest"}


def export_camera_frames(path, observations, folders=CAMERA_FOLDERS, quality=95):
    """Write every recorded frame as <episode>/cameras/<folder>/<i>.jpeg, i = 1..T+1.

    Frame i is observation i-1. JPEG is lossy and for viewing / external tools only; the
    pixels the dataset and checksums use stay in episode.npz."""
    from PIL import Image
    written = {}
    for camera, folder in folders.items():
        if camera not in observations[0].get("images",{}):
            continue
        out = Path(path)/"cameras"/folder
        out.mkdir(parents=True,exist_ok=False)
        for i, o in enumerate(observations,1):
            Image.fromarray(o["images"][camera]).save(out/f"{i}.jpeg",quality=quality)
        written[folder] = camera
    depth = {}
    for name in observations[0].get("depth",{}):
        # Depth frames as lossless 16-bit PNG (mm, 0 = no data), same numbering as the JPEGs, in
        # <RGB folder>_depth (scene -> head_depth, wrist -> wrist_depth, chest -> chest_depth).
        folder = f"{folders.get(f'{name}_rgb', name)}_depth"
        out = Path(path)/"cameras"/folder
        out.mkdir(parents=True,exist_ok=False)
        for i, o in enumerate(observations,1):
            Image.fromarray(o["depth"][name]).save(out/f"{i}.png")
        depth[folder] = f"depth_{name}_mm"
    return {"folders":written,"format":"jpeg","quality":quality,"first_index":1,
            "count":len(observations),"frame_index":"file i = observation i-1",
            **({"depth_folders":depth,"depth_format":"png uint16 mm, 0 = no data"} if depth else {})}


def save_episode(path, observations, requested, applied, states, velocities,
                 rewards, terminated, truncated, durations, phases, infos, manifest,
                 labels=None, fault=None, visibility=None, quality=None, stage_outcomes=None,
                 replay=None):
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
    if labels is not None:
        arrays["step_label"]=np.array(labels)
    # Training-safe labels: action_label is the applied command where it is a usable expert label and NaN
    # on injected-fault steps. Clean episodes have no step_label, so every step is usable.
    label_mask=(np.array(labels)!="fault") if labels is not None else np.ones(len(requested),dtype=bool)
    arrays["label_mask"]=label_mask
    arrays["action_label"]=np.where(label_mask[:,None],arrays["action_applied"],np.nan)
    if visibility is not None:
        arrays["visibility_pilot"]=np.asarray(visibility,dtype=bool)
    for camera in observations[0]["images"]:
        arrays[f"image_{camera}"]=np.stack([o["images"][camera] for o in observations])
    for name in observations[0].get("depth",{}):
        arrays[f"depth_{name}_mm"]=np.stack([o["depth"][name] for o in observations])
    steps=len(requested)
    if all(k in infos[-1] for k in TORQUE_KEYS):
        # infos[0] is the reset; info i+1 closes the interval of action i.
        arrays.update({k:np.stack([np.asarray(i[k],dtype=np.float64) for i in infos[len(infos)-steps:]])
                       for k in TORQUE_KEYS})
    np.savez_compressed(path/"episode.npz",**arrays)
    frames = export_camera_frames(path, observations)
    metadata = {**manifest,"dataset_schema":SCHEMA,"action_source":"privileged_scripted_expert",
                "camera_frames":frames,
                "instruction":observations[0]["instruction"], "steps":len(requested),
                "data_sha256":hashlib.sha256((path/"episode.npz").read_bytes()).hexdigest(),
                "outcome":infos[-1]["outcome"],
                "timing":"observation[i] precedes action[i]; observation[i+1] follows its executed interval",
                "pilot_contract":pilot_contract(manifest)}
    for key,value in (("stage_outcomes",stage_outcomes),("quality",quality),("replay",replay)):
        if value is not None:
            metadata[key]=value
    if visibility is not None:
        metadata["visibility_pilot_columns"]=list(PILOT_CAMERAS)
    if TORQUE_KEYS[0] in arrays:
        metadata["joint_torque"]={"arrays":list(TORQUE_KEYS),"units":"N m","shape":"(T, 14)",
            "order":"left joint1..7, right joint1..7","row":"row i = mean (peak: max |.|) over the "
            "physics substeps of action[i]'s interval","parts":"applied = motor torque after the "
            "actuator limits; gravity = qfrc_bias (gravity + Coriolis); constraint = qfrc_constraint "
            "(contacts + joint limits)"}
    if labels is not None:
        metadata.update(fault=fault, label_counts={k:int(sum(1 for v in labels if v==k)) for k in STEP_LABELS},
                        step_label_note="step_label[i] describes action[i]; 'fault' actions are injected "
                                        "mistakes and must be masked before behaviour cloning")
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
        for key in a.files:
            if key.startswith("image_") and key not in ("image_scene_rgb","image_wrist_rgb"):
                if a[key].ndim!=4 or a[key].shape[0]!=t+1 or a[key].shape[3]!=3 or a[key].dtype!=np.uint8:
                    raise ValueError(f"Image schema mismatch: {key}")
            elif key.startswith("depth_"):
                rgb="image_"+key[len("depth_"):-len("_mm")]+"_rgb"
                if (rgb not in a.files or a[key].dtype!=np.uint16 or a[key].ndim!=3
                        or a[key].shape[0]!=t+1 or a[key].shape[1:]!=a[rgb].shape[1:3]):
                    raise ValueError(f"Depth schema mismatch: {key} (needs {rgb} of the same size)")
            elif key in TORQUE_KEYS:
                if a[key].shape!=(t,14) or not np.isfinite(a[key]).all():
                    raise ValueError(f"Joint torque schema mismatch: {key}")
        done=a["terminated"]|a["truncated"]
        if done[:-1].any() or not done[-1]:
            raise ValueError("Episode boundary mismatch")
        if np.any(a["terminated"] & a["truncated"]):
            raise ValueError("Terminal and truncation flags overlap")
        if meta.get("action_source") != "human_manual" and meta["outcome"] != "success":
            raise ValueError("Pilot BC export accepts successful expert episodes only")
        if "step_label" in a.files:
            validate_step_labels(a["step_label"], t)
        if "action_label" in a.files or "label_mask" in a.files:
            validate_label_arrays(a, t)
        if "visibility_pilot" in a.files:
            if a["visibility_pilot"].shape!=(t+1,len(PILOT_CAMERAS)) or a["visibility_pilot"].dtype!=bool:
                raise ValueError("visibility_pilot schema mismatch")
    return meta


def validate_label_arrays(a, steps):
    """action_label / label_mask: shapes, consistency with step_label, NaN exactly on masked steps."""
    if "action_label" not in a.files or "label_mask" not in a.files:
        raise ValueError("action_label and label_mask must come together")
    mask,label=a["label_mask"],a["action_label"]
    if mask.shape!=(steps,) or mask.dtype!=bool or label.shape!=(steps,8):
        raise ValueError("action_label / label_mask shape mismatch")
    if "step_label" in a.files and not np.array_equal(mask,np.asarray(a["step_label"])!="fault"):
        raise ValueError("label_mask inconsistent with step_label")
    if not np.array_equal(np.isnan(label).any(axis=1),~mask) or np.isnan(label).any(axis=1).sum()!=np.isnan(label).all(axis=1).sum():
        raise ValueError("action_label must be NaN exactly where label_mask is False")
    if not np.array_equal(label[mask],a["action_applied"][mask]):
        raise ValueError("action_label differs from action_applied on usable steps")


def validate_step_labels(labels, steps):
    """Injected-fault episodes: one label per action, a recovery after any fault, never ending on one."""
    labels=np.asarray(labels)
    if labels.shape!=(steps,):
        raise ValueError("step_label shape mismatch")
    unknown=set(labels.tolist())-set(STEP_LABELS)
    if unknown:
        raise ValueError(f"Unknown step_label values {sorted(unknown)}")
    if (labels=="fault").any() and not (labels=="recovery").any():
        raise ValueError("Fault episode without recovery steps")
    if steps and labels[-1]=="fault":
        raise ValueError("Episode ends on an injected fault step")


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

