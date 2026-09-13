"""Audit the real pilot and float32 command bridge without downloading a VLA."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from learning.vla.openpi_adapter import OpenArmCodec, OpenArmInputs, SCHEMA
from learning.vla.pilot_dataset import FullWindowPilotDataset

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=ROOT / "data/openarm_v1_pilot")
    parser.add_argument("--output", type=Path, default=ROOT / "results/vla_adapter_v1")
    parser.add_argument("--replay", action="store_true", help="Replay one episode through float32 numeric encoding")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    config = json.loads((ROOT / "configs/vla_pi0.json").read_text())
    upstream = ROOT / "third_party/openpi"
    revision = subprocess.check_output([
        "git", "-c", f"safe.directory={upstream.as_posix()}", "-C", str(upstream), "rev-parse", "HEAD"
    ], text=True).strip()
    if revision != config["openpi_commit"]:
        raise ValueError("Upstream revision differs from audited config")
    stats = json.loads((args.dataset / "normalization.json").read_text())
    codec = OpenArmCodec(stats)
    manifest = json.loads((args.dataset / "manifest.json").read_text())
    datasets = {split: FullWindowPilotDataset(args.dataset, split, config["action_horizon"])
                for split in ("train", "validation")}
    errors, sample_checks = [], []
    training_arrays = {"state": [], "action": []}
    for record in manifest["episodes"]:
        with np.load(args.dataset / record["episode"] / "episode.npz", allow_pickle=False) as data:
            actions = data["action_applied"]
            decoded = codec.decode_actions(codec.encode_numeric("action", actions))
            errors.append(float(np.max(np.abs(decoded - actions))))
            if record["split"] == "train":
                training_arrays["state"].append(data["observation_state"][:-1])
                training_arrays["action"].append(actions)
    stats_match = True
    for key, arrays in training_arrays.items():
        array = np.concatenate(arrays)
        std = array.std(0)
        stats_match &= np.allclose(stats["transforms"][key]["mean"], array.mean(0), atol=1e-12, rtol=0)
        stats_match &= np.allclose(stats["transforms"][key]["scale"], np.where(std < 1e-6, 1., std), atol=1e-12, rtol=0)
    for dataset in datasets.values():
        for index in (0, len(dataset) - 1):
            raw = dataset[index]
            transformed = OpenArmInputs(codec)(raw)
            sample_checks.append(transformed["actions"].shape == (50, 32) and raw["action_mask"].all())
        # Every indexed chunk ends within its own episode. Last valid chunk includes seating/hold.
        lengths = {r["episode"]: r["steps"] for r in dataset.records}
        sample_checks.extend(step + dataset.horizon <= lengths[name] for name, step in dataset.index)
    checks = {"train_only_statistics_recomputed": bool(stats_match),
              "full_window_shapes_and_boundaries": bool(all(sample_checks)),
              "float32_action_roundtrip_below_1e_6": max(errors) < 1e-6,
              "split_episodes_disjoint": not bool({n for n, _ in datasets["train"].index}
                                                  & {n for n, _ in datasets["validation"].index})}
    report = {"schema": SCHEMA, "checks": checks, "max_action_roundtrip_error": max(errors),
              "full_windows": {k: len(v) for k, v in datasets.items()},
              "excluded_tail_start_positions_per_episode": config["action_horizon"] - 1,
              "model_training_run": False, "upstream_runtime_tested": False,
              "lerobot_exported": False, "openpi_commit": revision,
              "source_hashes": {str(p.relative_to(ROOT)): sha(p) for p in (
                  ROOT / "configs/vla_pi0.json", ROOT / "learning/vla/openpi_adapter.py",
                  ROOT / "learning/vla/pilot_dataset.py", ROOT / "learning/vla/openpi_integration.py",
                  upstream / "src/openpi/models/pi0.py", upstream / "src/openpi/training/data_loader.py",
                  upstream / "src/openpi/training/config.py", upstream / "LICENSE")},
              "pilot_manifest_sha256": sha(args.dataset / "manifest.json"),
              "normalization_sha256": sha(args.dataset / "normalization.json")}
    if args.replay:
        from envs.openarm_insert import OpenArmInsertEnv
        path = args.dataset / manifest["episodes"][0]["episode"]
        meta = json.loads((path / "metadata.json").read_text())
        env = OpenArmInsertEnv(images=False)
        try:
            env.reset(seed=meta["seed"], options=meta["reset_options"])
            qpos_error = qvel_error = 0.
            with np.load(path / "episode.npz", allow_pickle=False) as data:
                decoded = codec.decode_actions(codec.encode_numeric("action", data["action_requested"]))
                for i, action in enumerate(decoded):
                    _, _, term, trunc, _ = env.step(action)
                    qpos_error = max(qpos_error, float(np.max(np.abs(env.data.qpos - data["qpos"][i + 1]))))
                    qvel_error = max(qvel_error, float(np.max(np.abs(env.data.qvel - data["qvel"][i + 1]))))
                    if term or trunc:
                        break
                checks["float32_replay_success_same_command_count"] = env.outcome == "success" and i + 1 == len(decoded)
                checks["float32_replay_state_tolerance"] = qpos_error < 1e-4 and qvel_error < 1e-3
                report["float32_replay"] = {"episode": path.name, "outcome": env.outcome,
                    "commands": i + 1, "max_qpos_error": qpos_error, "max_qvel_error": qvel_error,
                    "note": "Numeric transport audit, not learned-policy inference; qpos/qvel contain mixed joint units"}
        finally:
            env.close()
    report["passed"] = all(checks.values())
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "report.json").write_text(json.dumps(report, indent=2))
    # Copy provenance as a new immutable bundle; do not alter the source pilot.
    (args.output / "normalization.json").write_text(json.dumps(stats, indent=2))
    (args.output / "config.json").write_text(json.dumps(config, indent=2))
    (args.output / "window_index.json").write_text(json.dumps({k: v.index for k, v in datasets.items()}))
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
