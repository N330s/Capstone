"""Matched-noise predictions on old expert/noisy-state and held-out recovery inputs.

No simulator reset is reconstructed for new data: these are recorded observations.
Artificial gripper offsets are counterfactual sensitivity tests, not rollouts.
"""
import argparse
from collections import Counter
from collections import defaultdict
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ["HF_HUB_OFFLINE"] = os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ.setdefault("HF_HOME", str(ROOT / "data/hf_cache"))
os.environ.setdefault("HF_DATASETS_CACHE", str(ROOT / "data/hf_cache/datasets"))


def valid_horizon(labels, step, horizon=50):
    if type(step) is not int or not 0 <= step < len(labels) or type(horizon) is not int or horizon < 1:
        raise ValueError("Valid anchor and positive horizon required")
    if str(labels[step]) == "fault":
        raise ValueError("Fault anchor cannot be used as a correction target")
    stop = min(len(labels), step + horizon)
    for index in range(step, stop):
        if str(labels[index]) == "fault":
            return index - step
    return stop - step


def make_cases(all_recovery_frames=False, recovery_view=None):
    import numpy as np
    from learning.vla.smolvla_data import sha256
    cases, sources = [], []
    paths = [("old", ROOT / "data/drive_v2c30_20261012/merged_all_v1" / name)
             for name in ("train_20004", "train_21010", "train_21036")]
    if recovery_view:
        recovery_view = Path(recovery_view).resolve()
        manifest = json.loads((recovery_view / "manifest.json").read_text())
        for record in manifest["episodes"]:
            if record["split"] == "validation":
                path = (recovery_view / record["episode"]).resolve()
                if path.parent != recovery_view:
                    raise ValueError("Unsafe validation parent")
                paths.append(("new", path))
    else:
        paths.append(("new", ROOT / "data/drive_recovery_20261019/premature_close_holdout/recovery_premature_close/train_09009"))
    for domain, path in paths:
        meta = json.loads((path / "metadata.json").read_text())
        if sha256(path / "episode.npz") != meta["data_sha256"]:
            raise ValueError("Recording checksum mismatch")
        sources.append({"episode": path.name, "domain": domain, "data_sha256": meta["data_sha256"]})
        with np.load(path / "episode.npz", allow_pickle=False) as arrays:
            states, actions, phases = arrays["observation_state"], arrays["action_applied"], arrays["phase"]
            labels = arrays["step_label"] if domain == "new" else np.full(len(actions), "expert")
            recovery = np.flatnonzero(labels == "recovery")
            sources[-1]["recovery_commands"] = int(len(recovery))
            sources[-1]["recovery_phase_counts"] = dict(Counter(phases[recovery].tolist()))
            sources[-1]["recovery_observed_gripper_range_mm"] = (
                [float(states[recovery, 15].min() * 1000), float(states[recovery, 15].max() * 1000)]
                if len(recovery) else None)
            frames = {0, 200, 320}
            for phase in ("close", "lift"):
                found = np.flatnonzero((phases == phase) & (labels != "fault"))
                if found.size:
                    frames.add(int(found[0]))
                    if phase == "close":
                        frames.add(min(int(found[0]) + 25, int(found[-1])))
            if domain == "new":
                recovery = np.flatnonzero(labels == "recovery")
                if all_recovery_frames:
                    frames.update(map(int, recovery))
                else:
                    for offset in (0, 25, 50, 57, 100, len(recovery) - 1):
                        frames.add(int(recovery[min(offset, len(recovery) - 1)]))
            images = {camera: arrays["image_" + camera] for camera in ("scene_rgb", "wrist_rgb")}
            for step in sorted(frames):
                if str(labels[step]) == "fault":
                    continue  # A generic early probe can land inside a new disturbance.
                for offset in ((-3., 0., 3.) if domain == "old" and step in (200, 320) else (0.,)):
                    state = states[step].copy()
                    state[15] += offset / 1000
                    valid = valid_horizon(labels, step)
                    group = ("old_state_offset" if offset else "old_clean") if domain == "old" else (
                        "new_recovery" if str(labels[step]) == "recovery" else "new_clean")
                    cases.append({"episode": path.name, "domain": domain, "step": step,
                                  "phase": str(phases[step]), "label": str(labels[step]),
                                  "group": group, "state_offset_mm": offset,
                                  "observation": {"state": state, "instruction": meta["instruction"],
                                      "images": {k: v[step].copy() for k, v in images.items()}},
                                  "target": actions[step:step + valid].astype(np.float32),
                                  "observed_gripper_mm": float(state[15] * 1000)})
    return cases, sources


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--backbone", type=Path, default=ROOT / "data/models/smolvlm_backbone")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--all-recovery-frames", action="store_true")
    parser.add_argument("--recovery-view", type=Path, help="Probe every validation parent in this immutable view")
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()
    if not 1 <= args.batch_size <= 8:
        raise ValueError("Probe batch size must be 1..8")
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    import numpy as np
    import torch
    from torch.utils.data import default_collate
    from learning.vla.smolvla_data import observation, sha256
    from learning.vla.smolvla_policy import SmolVLAController
    cases, sources = make_cases(args.all_recovery_frames, args.recovery_view)
    controller = SmolVLAController(args.checkpoint, args.backbone)
    rows = []
    for start in range(0, len(cases), args.batch_size):
        group = cases[start:start + args.batch_size]
        batch = controller.pre(default_collate([observation(case["observation"]) for case in group]))
        for seed in (1000, 1001, 1002):
            controller.model.reset()
            noise = torch.randn((1, 50, controller.model.config.max_action_dim), device="cuda",
                                generator=torch.Generator(device="cuda").manual_seed(seed))
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                predictions = controller.post(controller.model.predict_action_chunk(
                    batch, noise=noise.repeat(len(group), 1, 1))).float().cpu().numpy()
            if predictions.shape != (len(group), 50, 8) or not np.isfinite(predictions).all():
                raise ValueError("Invalid physical predictions")
            for case, predicted in zip(group, predictions):
                target = case["target"]
                error = np.abs(predicted[:len(target)] - target)
                rows.append({k: v for k, v in case.items() if k not in ("observation", "target")} |
                        {"seed": seed, "valid_targets": len(target), "predicted_first_gripper_mm": float(predicted[0, 7] * 1000),
                         "expert_first_gripper_mm": float(target[0, 7] * 1000),
                         "first_gripper_mae_mm": float(error[0, 7] * 1000),
                         "prefix5_gripper_mae_mm": float(error[:5, 7].mean() * 1000),
                         "chunk_gripper_mae_mm": float(error[:, 7].mean() * 1000),
                         "first_joint_mae_rad": float(error[0, :7].mean()),
                         "predicted_opening_delta_mm": float(predicted[0, 7] * 1000 - case["observed_gripper_mm"]),
                         "expert_opening_delta_mm": float(target[0, 7] * 1000 - case["observed_gripper_mm"])})
        print(json.dumps({"cases_completed": min(start + args.batch_size, len(cases)), "total": len(cases)}), flush=True)
    groups = defaultdict(list)
    for row in rows:
        groups[row["group"]].append(row)
        if row["group"] == "new_recovery" and row["phase"] == "reopen":
            groups["new_reopen"].append(row)
    summary = {key: {"predictions": len(values),
        **{metric: float(np.mean([r[metric] for r in values])) for metric in (
            "first_gripper_mae_mm", "prefix5_gripper_mae_mm", "chunk_gripper_mae_mm", "first_joint_mae_rad",
            "predicted_opening_delta_mm", "expert_opening_delta_mm")},
        "positive_opening_prediction_fraction": float(np.mean([r["predicted_opening_delta_mm"] > 0 for r in values]))}
        for key, values in groups.items()}
    triplets = defaultdict(dict)
    for row in rows:
        if row["domain"] == "old" and row["step"] in (200, 320):
            triplets[(row["episode"], row["step"], row["seed"])][row["state_offset_mm"]] = row
    gains = [(v[3.]["predicted_first_gripper_mm"] - v[-3.]["predicted_first_gripper_mm"]) / 6
             for v in triplets.values() if set(v) == {-3., 0., 3.}]
    sensitivity = {"matched_triplets": len(gains), "mean_prediction_gain": float(np.mean(gains)),
                   "gain_std": float(np.std(gains)),
                   "minus3mm_mean_predicted_opening_delta_mm": float(np.mean([
                       v[-3.]["predicted_opening_delta_mm"] for v in triplets.values()])),
                   "definition": "Change in first predicted gripper target / 6 mm observed-state change; same expert image/joints/noise. A state-independent 35 mm hold target has gain zero.",
                   "limits": "Local counterfactual conditioning sensitivity, not a closed-loop stability proof"}
    validation_parents = {s["episode"] for s in sources if s["domain"] == "new"}
    expected_parents = set()
    if args.recovery_view:
        view_manifest = json.loads((args.recovery_view / "manifest.json").read_text())
        expected_parents = {r["episode"] for r in view_manifest["split_plan"]["episodes"] if r["split"] == "validation"}
    result = {"checkpoint": str(args.checkpoint), "weights_sha256": sha256(args.checkpoint / "model.safetensors"),
              "sources": sources, "cases": len(cases), "predictions": rows, "summary": summary,
              "all_heldout_recovery_frames": args.all_recovery_frames, "batch_size": args.batch_size,
              "heldout_recovery_parent_count": len(validation_parents),
              "full_recovery_validation_split": bool(expected_parents) and validation_parents == expected_parents,
              "validation_scope": "All recovery frames and selected clean frames, not every eligible validation command",
              "state_sensitivity": sensitivity,
              "limits": "Recorded-observation and counterfactual sensitivity probes, three noise seeds; not full validation, physical recovery or autonomous rollout success"}
    (args.output / "report.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
