"""Read-only recovery-release audit; never exports data or starts training."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import zipfile

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data_pipeline.episodes import validate_episode
from learning.vla.smolvla_data import sha256


def array_headers(path):
    headers = {}
    with zipfile.ZipFile(path) as archive:
        for name in archive.namelist():
            if not name.endswith(".npy"):
                raise ValueError("Unexpected NPZ member")
            with archive.open(name) as stream:
                version = np.lib.format.read_magic(stream)
                if version == (1, 0):
                    shape, fortran, dtype = np.lib.format.read_array_header_1_0(stream)
                elif version == (2, 0):
                    shape, fortran, dtype = np.lib.format.read_array_header_2_0(stream)
                else:
                    raise ValueError(f"Unsupported NPY version {version}")
                if dtype.hasobject or fortran:
                    raise ValueError("Object/Fortran arrays are not accepted")
                headers[name[:-4]] = {"shape": list(shape), "dtype": str(dtype),
                                     "nbytes": int(np.prod(shape)) * dtype.itemsize}
    return headers


def frame_from_npz(path, key, index):
    """Read through a compressed NPY stream without allocating its whole movie."""
    with zipfile.ZipFile(path) as archive, archive.open(key + ".npy") as stream:
        version = np.lib.format.read_magic(stream)
        read = (np.lib.format.read_array_header_1_0 if version == (1, 0)
                else np.lib.format.read_array_header_2_0)
        shape, fortran, dtype = read(stream)
        if fortran or dtype.hasobject or not 0 <= index < shape[0]:
            raise ValueError("Unsupported movie/frame")
        size = int(np.prod(shape[1:])) * dtype.itemsize
        skip = index * size
        while skip:
            chunk = stream.read(min(skip, 8 * 1024**2))
            if not chunk:
                raise ValueError("Truncated movie")
            skip -= len(chunk)
        payload = stream.read(size)
        if len(payload) != size:
            raise ValueError("Truncated frame")
        return np.frombuffer(payload, dtype=dtype).reshape(shape[1:]).copy()


def audit_sample(root, baseline, output):
    root = Path(root)
    inventory = json.loads((root / "archive_inventory.json").read_text())
    if "extracted" not in inventory:
        raise ValueError("Sample download did not complete")
    collection = root / inventory["root"]
    manifest = json.loads((collection / "manifest.json").read_text())
    results = []
    for episode in inventory["sampled_episodes"]:
        path = collection / episode
        meta = validate_episode(path)
        arrays = array_headers(path / "episode.npz")
        record = next(r for r in manifest["episodes"] if r["episode"] == episode)
        if meta["steps"] != record["steps"] or meta["data_sha256"] != record["data_sha256"]:
            raise ValueError("Sample and manifest disagree")
        t = meta["steps"]
        result = {"collection": collection.name, "episode": episode, "steps": t,
                  "npz_sha256": meta["data_sha256"], "legacy_episode_validation_passed": True,
                  "arrays": arrays, "new_arrays": sorted(set(arrays) - set(baseline)),
                  "missing_legacy_arrays": sorted(set(baseline) - set(arrays))}
        result["joint_torque_metadata"] = meta.get("joint_torque")
        with np.load(path / "episode.npz", allow_pickle=False) as values:
            labels = values["step_label"]
            if labels.shape != (t,) or set(labels.tolist()) - {"expert", "fault", "recovery"}:
                raise ValueError("Unexpected recovery labels")
            counts = dict(Counter(labels.tolist()))
            if counts != record["label_counts"]:
                raise ValueError("Manifest/array label counts disagree")
            times, dt = values["timestamp_s"], values["executed_dt_s"]
            if (not np.allclose(np.diff(times)[:-1], .02, atol=1e-9, rtol=0)
                    or not 0 < dt[-1] <= .02 + 1e-9):
                raise ValueError("Unexpected command timing")
            result["label_counts"] = counts
            result["phase_counts"] = dict(Counter(values["phase"].tolist()))
            result["small_array_stats"] = {}
            for key, header in arrays.items():
                if header["nbytes"] < 16 * 1024**2 and np.dtype(header["dtype"]).kind in "fiu":
                    value = values[key]
                    if not np.isfinite(value).all():
                        raise ValueError(f"Non-finite numeric array {key}")
                    result["small_array_stats"][key] = {"min": float(value.min()), "max": float(value.max())}
            for key in ("joint_torque_applied_nm", "joint_torque_applied_peak_nm",
                        "joint_torque_gravity_nm", "joint_torque_constraint_nm"):
                if arrays[key]["shape"] != [t, 14]:
                    raise ValueError(f"Unexpected torque shape {key}")
            if np.any(values["joint_torque_applied_peak_nm"] + 1e-9 < np.abs(values["joint_torque_applied_nm"])):
                raise ValueError("Peak absolute torque smaller than absolute mean")
            state, action = values["observation_state"], values["action_applied"]
            segments = []
            start = 0
            for end in range(1, t + 1):
                if end == t or labels[end] != labels[start]:
                    segments.append({"label": str(labels[start]), "start": start, "stop_exclusive": end,
                                     "observed_gripper_mm_start": float(state[start, 15] * 1000),
                                     "observed_gripper_mm_end": float(state[end, 15] * 1000),
                                     "target_gripper_mm_min": float(action[start:end, 7].min() * 1000),
                                     "target_gripper_mm_max": float(action[start:end, 7].max() * 1000)})
                    start = end
            result["label_segments"] = segments
            starts = np.flatnonzero(labels == "recovery")
            frame = int(starts[len(starts) // 2]) if len(starts) else 0
            result["review_frame"] = frame
        from PIL import Image, ImageDraw, ImageOps
        camera_keys = [key for key in arrays if key.startswith("image_")]
        depth_keys = [key for key in arrays if "depth" in key.lower()]
        sheet = Image.new("RGB", (424 * max(1, len(camera_keys)), 560), "white")
        draw = ImageDraw.Draw(sheet)
        for i, key in enumerate(camera_keys):
            header = arrays[key]
            if header["shape"][0] != t + 1 or header["dtype"] != "uint8" or header["shape"][-1] != 3:
                raise ValueError(f"Unexpected RGB schema {key}")
            pixels = frame_from_npz(path / "episode.npz", key, frame)
            draw.text((i * 424 + 4, 4), f"{key} | observation {frame}", fill="black")
            picture = ImageOps.pad(Image.fromarray(pixels), (424, 240), color="black")
            sheet.paste(picture, (i * 424, 24))
        result["depth_review_frame"] = {}
        for i, key in enumerate(depth_keys):
            header = arrays[key]
            if header["shape"][0] != t + 1 or header["dtype"] != "uint16":
                raise ValueError(f"Unexpected depth schema {key}")
            pixels = frame_from_npz(path / "episode.npz", key, frame)
            valid = pixels[pixels > 0]
            result["depth_review_frame"][key] = {
                "zero_fraction": float(np.mean(pixels == 0)),
                "valid_min_mm": int(valid.min()) if valid.size else None,
                "valid_max_mm": int(valid.max()) if valid.size else None,
                "scope": "one review frame, not every depth value"}
            if i < len(camera_keys):
                gray = np.clip(pixels.astype(float) / 3000 * 255, 0, 255).astype(np.uint8)
                draw.text((i * 424 + 4, 284), f"{key} | raw 1 mm units; zero=invalid", fill="black")
                picture = ImageOps.pad(Image.fromarray(gray).convert("RGB"), (424, 240), color="black")
                sheet.paste(picture, (i * 424, 308))
        sheet.save(output / f"{collection.name}_{episode}_rgb_depth.png")
        results.append(result)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--intake", type=Path, default=Path("data/drive_recovery_20261019"))
    parser.add_argument("--baseline", type=Path,
                        default=Path("data/drive_v2c30_20261012/merged_all_v1"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", nargs="*", type=Path, default=[])
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    old_manifest = json.loads((args.baseline / "manifest.json").read_text())
    old_meta = json.loads((args.baseline / old_manifest["episodes"][0]["episode"] / "metadata.json").read_text())
    releases, seen_names, seen_seeds, seen_hashes = [], set(), set(), set()
    for path in sorted(args.intake.glob("*_meta/recovery_*/manifest.json")):
        manifest = json.loads(path.read_text())
        records, robot = manifest["episodes"], manifest["robot_manifest"]
        labels = Counter()
        for record in records:
            if (record["episode"] in seen_names or record["seed"] in seen_seeds
                    or record["data_sha256"] in seen_hashes):
                raise ValueError("Recovery manifests overlap")
            seen_names.add(record["episode"])
            seen_seeds.add(record["seed"])
            seen_hashes.add(record["data_sha256"])
            if sum(record["label_counts"].values()) != record["steps"]:
                raise ValueError("Label counts do not sum to episode length")
            labels.update(record["label_counts"])
        releases.append({"collection": path.parent.name, "manifest_sha256": sha256(path),
                         "episodes": len(records), "steps": sum(r["steps"] for r in records),
                         "splits": dict(Counter(r["split"] for r in records)), "labels": dict(labels),
                         "complete": manifest.get("complete"), "training_note": manifest.get("training_note"),
                         "scene_sha256": robot["scene_sha256"], "source_hashes": robot["source_hashes"],
                         "robot_config_matches_baseline": robot["robot_config"] == old_meta["robot_config"],
                         "state_order": robot["state_order"], "action_names": robot["action_names"],
                         "changed_environment_fields": [k for k in
                             ("robot", "robot_config", "scene_sha256", "source_hashes", "workspace", "camera_calibration")
                             if robot.get(k) != old_meta.get(k)],
                         "joint_torque_order": robot.get("joint_torque_order"),
                         "depth_cameras": robot.get("depth_cameras"), "chest_rgbd": robot.get("chest_rgbd")})
    if len(releases) != 3:
        raise ValueError("Expected metadata for all three recovery collections")
    totals = Counter()
    for release in releases:
        totals.update(release["labels"])
    baseline_headers = array_headers(args.baseline / old_manifest["episodes"][0]["episode"] / "episode.npz")
    report = {"date": "2026-10-19", "scope": "All three manifests; only explicitly listed downloaded episodes validated",
              "releases": releases, "episodes": sum(r["episodes"] for r in releases),
              "steps": sum(r["steps"] for r in releases), "labels": dict(totals),
              "legacy_array_schema": baseline_headers,
              "samples": [r for path in args.samples for r in audit_sample(path, baseline_headers, args.output)],
              "training_started": False,
              "required_before_training": [
                  "Reject fault anchors; exclude fault targets anywhere in each future action chunk",
                  "Do not join separated valid segments across fault intervals",
                  "Create explicit whole-episode recovery validation holdout; keep original validation untouched",
                  "Review new scene/source/camera changes; obtain matching source snapshot and qualify replay",
                  "Preserve torque/depth as auxiliary channels unless a separate input contract is qualified",
                  "Torque row i summarizes action i's following interval: never expose it as a current observation at i",
                  "Validate all episodes before complete-release export or training"]}
    (args.output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("episodes", "steps", "labels", "training_started")}
                     | {"validated_samples": len(report["samples"])}), flush=True)


if __name__ == "__main__":
    main()
