"""Post-training full validation and source-matched simulation; never trains."""
import argparse
import csv
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ.setdefault("HF_HOME", str(ROOT / "data/hf_cache"))
os.environ.setdefault("HF_DATASETS_CACHE", str(ROOT / "data/hf_cache/datasets"))
from learning.vla.runtime_io import read_json, write_json
from learning.vla.window_training import CommandWindows
from learning.vla.smolvla_data import sha256, CONTRACT, LEROBOT_VERSION


def checkpoint_plan(run, export, selection="best-final"):
    """Only use complete checkpoints belonging to this completed training run."""
    report = read_json(run / "report.json")
    if not report.get("completed"):
        raise ValueError("Training is not complete; evaluation must wait")
    digest = sha256(export / "openarm_export.json")
    if report["binding"]["export_receipt_sha256"] != digest:
        raise ValueError("Training/export receipt mismatch")
    if selection not in ("best-final", "best", "final", "latest"):
        raise ValueError("Unknown checkpoint selection")
    labels = ["best", "final"] if selection == "best-final" else [selection]
    choices = [(label, Path(report["final_checkpoint"] if label == "final" else
                read_json(run / (label + "_checkpoint.json"))["path"])) for label in labels]
    result = []
    for label, path in choices:
        path = path.resolve()
        if not path.is_relative_to((run / "checkpoints").resolve()):
            raise ValueError("Checkpoint outside training run")
        marker = read_json(path / "complete.json")
        if marker["binding"] != report["binding"]:
            raise ValueError("Checkpoint binding differs from training report")
        if not (path / "pretrained_model/model.safetensors").is_file():
            raise ValueError("Missing checkpoint weights")
        existing = next((r for r in result if r["path"] == str(path)), None)
        if existing:
            existing["labels"].append(label)
        else:
            result.append({"labels": [label], "path": str(path), "step": marker["step"]})
    return result


def summarize(rows):
    """Command-weighted and episode-balanced loss; reject incomplete/nonfinite rows."""
    groups = {}
    for row in rows:
        value = float(row["flow_loss"])
        if not math.isfinite(value):
            raise ValueError("Nonfinite evaluation loss")
        groups.setdefault(row["episode"], []).append(value)
    if not groups:
        raise ValueError("No evaluation rows")
    episodes = {k: {"windows": len(v), "flow_loss": math.fsum(v) / len(v)}
                for k, v in groups.items()}
    count = sum(len(v) for v in groups.values())
    return {"windows": count, "episodes": episodes,
            "command_weighted_flow_loss": math.fsum(math.fsum(v) for v in groups.values()) / count,
            "episode_balanced_flow_loss": math.fsum(r["flow_loss"] for r in episodes.values()) / len(episodes)}


def evaluate_checkpoint(args, choice, receipt, destination):
    import torch
    from importlib.metadata import version
    from torch.utils.data import default_collate
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from learning.vla.smolvla_policy import SmolVLAController
    from learning.vla.smolvla_compat import masked_forward
    if version("lerobot") != LEROBOT_VERSION or not torch.cuda.is_bf16_supported():
        raise RuntimeError("Pinned LeRobot and BF16 CUDA required")
    destination.mkdir(parents=True, exist_ok=False)
    windows = CommandWindows(receipt["datasets"]["validation"]["episodes"])
    dataset = LeRobotDataset(receipt["datasets"]["validation"]["repo_id"],
                             root=args.export / "validation", video_backend="pyav",
                             delta_timestamps={"action": [i / 50 for i in range(50)]})
    if len(dataset) != len(windows):
        raise ValueError("Validation length differs from receipt")
    bundle = Path(choice["path"]) / "pretrained_model"
    controller = SmolVLAController(bundle, args.backbone, args.seed)
    model = controller.model
    limit = min(args.max_windows or len(windows), len(windows))
    # Eight timeline locations per held-out episode, explicitly a sampled action diagnostic.
    action_indices = {r["index"] for r in windows.probes(len(windows.records) * 8)}
    rows, actions = [], []
    started = time.monotonic()
    with (destination / "windows.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["index", "episode", "step", "flow_loss"])
        writer.writeheader()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for start in range(0, limit, args.batch_size):
                indices = list(range(start, min(start + args.batch_size, limit)))
                raw = default_collate([dataset[i] for i in indices])
                batch = controller.pre(raw)
                # Seed by global window index so comparisons do not depend on batching.
                noise = torch.stack([torch.randn((50, model.config.max_action_dim), device="cuda",
                    generator=torch.Generator(device="cuda").manual_seed(args.seed + 2000 + i))
                    for i in indices])
                losses, _ = masked_forward(model, batch, noise=noise,
                    time=torch.full((len(indices),), .5, device="cuda"), reduction="none")
                if losses.shape != (len(indices),) or not torch.isfinite(losses).all():
                    raise RuntimeError("Invalid per-window losses")
                for index, loss in zip(indices, losses.float().cpu().tolist()):
                    row = {**windows.locate(index), "flow_loss": loss}
                    rows.append(row)
                    writer.writerow(row)
                stream.flush()
                # Decode only sampled windows, separately from full flow-loss coverage.
                for index in indices:
                    if index not in action_indices:
                        continue
                    sample = default_collate([dataset[index]])
                    actor = controller.pre(sample)
                    actor = {k: v for k, v in actor.items() if k not in
                             ("action", "action_is_pad", "actions_id_pad")}
                    model.reset()
                    prediction = controller.post(model.predict_action_chunk(actor,
                                                   noise=noise[indices.index(index):indices.index(index) + 1]))
                    if tuple(prediction.shape) != (1, 50, 8) or not torch.isfinite(prediction).all():
                        raise RuntimeError("Invalid decoded action chunk")
                    valid = (~sample["action_is_pad"]).to(prediction.device)
                    error = (prediction - sample["action"].to(prediction.device)).abs()[valid]
                    actions.append({**windows.locate(index), "valid_commands": int(valid.sum()),
                                    "joint_mae_rad": error[:, :7].mean(0).float().cpu().tolist(),
                                    "gripper_mae_mm": float(error[:, 7].mean()) * 1000})
                elapsed = time.monotonic() - started
                write_json(destination / "status.json", {"evaluated": len(rows), "total": limit,
                    "elapsed_s": elapsed, "estimated_remaining_s": elapsed / len(rows) * (limit - len(rows))})
    result = {**summarize(rows), "checkpoint": choice,
              "checkpoint_weights_sha256": sha256(bundle / "model.safetensors"),
              "export_receipt_sha256": sha256(args.export / "openarm_export.json"),
              "full_validation": limit == len(windows), "expected_windows": len(windows),
              "action_probe_count": len(actions), "action_probes": actions,
              "elapsed_s": time.monotonic() - started, "seed": args.seed,
              "batch_size": args.batch_size, "robot_execution": False,
              "metric_definition": "LeRobot masked per-window flow loss, fixed t=0.5 and one seeded noise draw per window. Upstream averages zero-masked terminal padding over the full chunk. Physical action errors are sampled only.",
              "limits": "Validation split was used for checkpoint selection; not an untouched test set or task-success measurement."}
    write_json(destination / "report.json", result)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, default=ROOT / "outputs/v2c30_sft_overnight_v1")
    p.add_argument("--campaign", type=Path, default=ROOT / "outputs/v2c30_campaign_v4")
    p.add_argument("--export", type=Path, default=ROOT / "data/drive_v2c30_20261012/export_all_v1")
    p.add_argument("--collection", type=Path, default=ROOT / "data/drive_v2c30_20261012/merged_all_v1")
    p.add_argument("--backbone", type=Path, default=ROOT / "data/models/smolvlm_backbone")
    p.add_argument("--snapshot", type=Path, default=ROOT / "third_party/capstone_automateData")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--prepare-only", action="store_true", help="CPU-only inventory; allowed while training")
    p.add_argument("--simulation-only", action="store_true", help="Skip offline validation entirely")
    p.add_argument("--checkpoint-choice", choices=("best-final", "best", "final", "latest"), default="best-final")
    p.add_argument("--worker-checkpoint", type=Path, help=argparse.SUPPRESS)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-windows", type=int, help="Small smoke check; explicitly NOT full validation")
    p.add_argument("--rollout-episodes", type=int, default=3, help="0 skips simulator; 20 covers this entire held-out split")
    p.add_argument("--stage-timeout-seconds", type=int, default=14400)
    p.add_argument("--seed", type=int, default=1000)
    args = p.parse_args()
    if args.batch_size < 1 or args.rollout_episodes < 0 or args.stage_timeout_seconds < 1 or args.seed < 0:
        p.error("Positive batch/timeout and nonnegative episode count/seed required")
    if args.max_windows is not None and args.max_windows < 1:
        p.error("Positive max-windows required")
    if args.simulation_only and (args.rollout_episodes == 0 or args.max_windows or args.worker_checkpoint):
        p.error("Simulation-only requires at least one rollout and no validation worker/window limit")
    for name in ("run", "campaign", "export", "collection", "backbone", "snapshot", "output"):
        setattr(args, name, getattr(args, name).resolve())
    receipt = read_json(args.export / "openarm_export.json")
    if receipt["contract"] != CONTRACT or receipt["lerobot_version"] != LEROBOT_VERSION:
        raise ValueError("Unsupported export contract")
    train = {r["episode"] for r in receipt["datasets"]["train"]["episodes"]}
    heldout = receipt["datasets"]["validation"]["episodes"]
    if train.intersection(r["episode"] for r in heldout):
        raise ValueError("Train/validation episode overlap")
    if args.rollout_episodes > len(heldout):
        raise ValueError("Requested more rollout episodes than available")
    inventory = {"validation_episodes": len(heldout), "validation_windows": sum(r["steps"] for r in heldout),
                 "rollout_episodes_per_checkpoint": args.rollout_episodes, "batch_size": args.batch_size,
                 "full_validation_requested": not args.simulation_only and args.max_windows is None,
                 "simulation_only": args.simulation_only, "checkpoint_choice": args.checkpoint_choice,
                 "training_complete": bool(read_json(args.run / "report.json").get("completed")),
                 "robot_execution": False}
    if args.prepare_only:
        print(json.dumps(inventory, indent=2))
        return
    # Campaign includes its own final simulation check. Wait for that to finish too.
    if not (args.campaign / "completion.json").is_file():
        raise ValueError("Overnight campaign is still active or incomplete; inspect it before evaluation")
    choices = checkpoint_plan(args.run, args.export, args.checkpoint_choice)
    if args.worker_checkpoint:
        choice = next(c for c in choices if Path(c["path"]) == args.worker_checkpoint.resolve())
        evaluate_checkpoint(args, choice, receipt, args.output)
        return
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / "plan.json", {**inventory, "checkpoints": choices})
    def launch(command, log):
        from scripts.run_smolvla_job import kill_owned_process
        with log.open("w", encoding="utf-8") as stream:
            process = subprocess.Popen([str(x) for x in command], stdout=stream, stderr=subprocess.STDOUT)
            try:
                code = process.wait(timeout=args.stage_timeout_seconds)
                if code:
                    raise subprocess.CalledProcessError(code, command)
            finally:
                if process.poll() is None:
                    kill_owned_process(process)
                    process.wait(timeout=15)
    results, rollouts = [], []
    try:
        for choice in choices:
            label = "_".join(choice["labels"])
            destination = args.output / label
            command = [sys.executable, "-u", Path(__file__), "--worker-checkpoint", choice["path"],
                       "--batch-size", args.batch_size, "--seed", args.seed, "--output", destination,
                       "--rollout-episodes", 0, "--checkpoint-choice", args.checkpoint_choice]
            for name in ("run", "campaign", "export", "collection", "backbone", "snapshot"):
                command.extend(["--" + name, getattr(args, name)])
            if args.max_windows:
                command.extend(["--max-windows", args.max_windows])
            if not args.simulation_only:
                launch(command, args.output / (label + "_validation.log"))
                results.append(read_json(destination / "report.json"))
            # Deterministic spread through held-out episodes; identical resets for both models.
            selected = [heldout[int(i * len(heldout) / args.rollout_episodes)]
                        for i in range(args.rollout_episodes)]
            for record in selected:
                episode = (args.collection / record["episode"]).resolve()
                if not episode.is_relative_to(args.collection):
                    raise ValueError("Episode path outside collection")
                if sha256(episode / "episode.npz") != record["data_sha256"]:
                    raise ValueError("Rollout episode does not match held-out export")
                target = args.output / (label + "_" + record["episode"])
                launch([sys.executable, "-u", ROOT / "scripts/rollout_smolvla.py", "--snapshot", args.snapshot,
                        "--episode", episode, "--checkpoint", Path(choice["path"]) / "pretrained_model",
                        "--backbone", args.backbone, "--output", target, "--max-steps", 2000,
                        "--prefix", 5, "--seed", args.seed], target.with_suffix(".log"))
                rollouts.append({"checkpoint_labels": choice["labels"], **read_json(target / "report.json")})
                write_json(args.output / "rollouts.json", rollouts)
        write_json(args.output / "report.json", {"completed": True, "validation": results,
            "validation_performed": not args.simulation_only,
            "rollouts": rollouts, "robot_execution": False,
            "limits": "Fixed-noise flow loss; sampled action errors; finite-horizon paused-time simulation on validation resets. No hardware or real-time qualification."})
    except Exception as error:
        write_json(args.output / "failure.json", {"completed": False, "error": str(error)})
        raise


if __name__ == "__main__":
    main()
