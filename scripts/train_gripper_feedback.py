"""Controlled nominal-vs-open-hold sensor-noise SFT; one training parent only."""
import argparse
from importlib.metadata import version
import json
import os
from pathlib import Path
import random
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ["HF_HUB_OFFLINE"] = os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ.setdefault("HF_HOME", str(ROOT / "data/hf_cache"))
os.environ.setdefault("HF_DATASETS_CACHE", str(ROOT / "data/hf_cache/datasets"))
from learning.vla.runtime_io import write_json
from learning.vla.smolvla_data import sha256, observation, LEROBOT_VERSION
from learning.vla.gripper_feedback import open_hold_anchors, perturbed_gripper

BASE = ROOT / "outputs/v2c30_sft_overnight_v1/checkpoints/step_0009426_best/pretrained_model"
EXPORT = ROOT / "data/drive_v2c30_20261012/export_all_v1"
EPISODE = ROOT / "data/drive_v2c30_20261012/merged_all_v1/train_20002"
BACKBONE = ROOT / "data/models/smolvlm_backbone"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--mode", choices=("nominal", "augmented"), required=True)
    p.add_argument("--steps", type=int, default=800)
    p.add_argument("--max-seconds", type=int, default=1200)
    p.add_argument("--verify", action="store_true")
    p.add_argument("--qualify-only", action="store_true", help="At most four GPU updates, no model bundle write")
    p.add_argument("--selection", type=Path, help="Verified five-parent extension; default preserves the original single-parent recipe")
    args = p.parse_args()
    if not 1 <= args.steps <= (4000 if args.selection else 800) or not 180 <= args.max_seconds <= (6000 if args.selection else 1500):
        raise ValueError("Bounded experiment only")
    if args.qualify_only and (args.steps > (12 if args.selection else 4) or args.verify):
        raise ValueError("Qualification is bounded to four/twelve updates and is not a reload")
    args.output = args.output.resolve()
    if any(args.output.is_relative_to(v) or v.is_relative_to(args.output) for v in (BASE, EXPORT, EPISODE, BACKBONE)):
        raise ValueError("Output overlaps inputs")
    if not args.verify:
        args.output.mkdir(parents=True, exist_ok=False)
    import numpy as np
    import torch
    from torch.utils.data import default_collate
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from learning.vla.smolvla_policy import SmolVLAController
    from learning.vla.smolvla_compat import masked_forward
    from learning.vla.training_state import capture_rng, restore_rng
    from learning.vla.expert_overlay import save_overlay
    from scripts.overfit_smolvla_episode import episode_span
    from scripts.rollout_smolvla import verify_snapshot

    if version("lerobot") != LEROBOT_VERSION or not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("Pinned LeRobot/CUDA/BF16 required")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    meta = json.loads((EPISODE / "metadata.json").read_text())
    if meta["outcome"] != "success" or sha256(EPISODE / "episode.npz") != meta["data_sha256"]:
        raise ValueError("Episode changed")
    verify_snapshot(ROOT / "third_party/capstone_automateData", meta)
    receipt = json.loads((EXPORT / "openarm_export.json").read_text())
    start, end, record = episode_span(receipt, EPISODE.name)
    parents = [{**record, 'start':start, 'end':end}]
    if args.selection:
        from learning.vla.five_episode import verify_selection
        selection = json.loads(args.selection.read_text())
        verify_selection(selection, EXPORT)
        if Path(selection['collection']).resolve() != EPISODE.parent.resolve():
            raise ValueError('This extension requires the original source collection')
        parents = selection['parents']
        for parent in parents:
            other_meta = json.loads((Path(selection['collection']) / parent['episode'] / 'metadata.json').read_text())
            verify_snapshot(ROOT / 'third_party/capstone_automateData', other_meta)
    cases, observations = [], []
    for parent in parents:
        folder = EPISODE.parent / parent['episode']
        with np.load(folder / "episode.npz", allow_pickle=False) as raw:
            states, actions = raw["observation_state"].copy(), raw["action_applied"].copy()
            eligible = open_hold_anchors(states, actions)
            if not eligible or parent['end']-parent['start'] != len(actions):
                raise ValueError('Source/export/open-hold binding differs')
            parent['open_hold_local'] = eligible
            images = {camera: raw["image_" + camera] for camera in ("scene_rgb", "wrist_rgb")}
            for frame in (100, 200, 300, 400, 500, 550, 650, 1000, 1400):
                for offset in ((0., -.002, .002) if frame in (100, 200, 300, 400) else (0.,)):
                    state = states[frame].copy()
                    state[15] = perturbed_gripper(state[15], offset)
                    observations.append(observation({"state": state, "instruction": meta["instruction"],
                        "images": {camera: images[camera][frame] for camera in images}}))
                    cases.append({"episode":parent['episode'], "frame": frame, "offset_m": offset,
                        "input_gripper_mm": float(state[15] * 1000), "target": actions[frame:frame + 5].copy()})
            del images
    if meta["data_sha256"] != record["data_sha256"]:
        raise ValueError("Source/export/open-hold binding differs")
    actor = SmolVLAController(BASE, BACKBONE, seed=1000,
        expert_overlay=args.output / "expert" if args.verify else None)
    model = actor.model
    from learning.vla.training_precision import promote_trainable_fp32, parameter_dtypes, assert_fp32_optimizer
    precision_audit = promote_trainable_fp32(model) if not args.verify else {"after": parameter_dtypes(model)}
    begun = time.monotonic()

    def evaluate(step):
        rng = capture_rng()
        model.eval()
        rows = []
        try:
            for seed in (1000, 1001, 1002):
                noise = torch.randn((1, 50, model.config.max_action_dim), device="cuda",
                    generator=torch.Generator(device="cuda").manual_seed(seed))
                for offset in range(0, len(cases), 3):
                    subset = cases[offset:offset + 3]
                    batch = actor.pre(default_collate(observations[offset:offset + 3]))
                    model.reset()
                    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                        pred = actor.post(model.predict_action_chunk(batch, noise=noise.repeat(len(subset), 1, 1)))
                    values = pred.float().cpu().numpy()
                    if values.shape != (len(subset), 50, 8) or not np.isfinite(values).all():
                        raise RuntimeError("Invalid finite physical actions")
                    for n, case in enumerate(subset):
                        error = np.abs(values[n, :5] - case["target"])
                        rows.append({k: v for k, v in case.items() if k != "target"} | {
                            "seed": seed, "gripper_mae_mm": float(error[:, 7].mean() * 1000),
                            "joint_mae_rad": float(error[:, :7].mean()),
                            "predicted_gripper_mm": float(values[n, :5, 7].mean() * 1000),
                            "prediction_prefix": values[n, :5].tolist()})
        finally:
            restore_rng(rng)
        groups = {"nominal": [r for r in rows if r["offset_m"] == 0],
                  "perturbed_open_hold": [r for r in rows if r["offset_m"] != 0]}
        metrics = {label: {k: float(np.mean([r[k] for r in group])) for k in ("gripper_mae_mm", "joint_mae_rad")}
                   for label, group in groups.items()}
        gains = []
        for parent in parents:
            for frame in (100, 200, 300, 400):
                for seed in (1000, 1001, 1002):
                    pair = {r["offset_m"]: r["predicted_gripper_mm"] for r in rows
                            if r['episode'] == parent['episode'] and r["frame"] == frame and r["seed"] == seed}
                    gains.append((pair[.002] - pair[-.002]) / 4.)
        metrics["gripper_sensor_gain"] = float(np.mean(gains))
        return {"step": step, "metrics": metrics, "predictions": rows}

    if args.verify:
        report = json.loads((args.output / "report.json").read_text())
        if args.selection and (report['binding'].get('selection_sha256') != sha256(args.selection)
                or report['binding']['mode'] != args.mode or report['updates'] != args.steps):
            raise ValueError('Reload must use the exact five-parent selection/mode/final budget')
        fresh = evaluate(report["updates"])
        error = max(float(np.max(np.abs(np.asarray(a["prediction_prefix"]) - b["prediction_prefix"])))
                    for a, b in zip(fresh["predictions"], report["final"]["predictions"], strict=True))
        if error > 1e-8:
            raise ValueError("Fresh overlay prediction mismatch")
        write_json(args.output / "verification.json", {"passed": True, "prediction_max_abs_difference": error,
            "normalization": "exact immutable base tensors", "inference_binding": actor.binding})
        print(json.dumps({"verified": True, "prediction_max_abs_difference": error}), flush=True)
        return

    dataset = LeRobotDataset(receipt["datasets"]["train"]["repo_id"], root=EXPORT / "train",
        delta_timestamps={"action": [i / 50 for i in range(50)]}, video_backend="pyav")
    parameters = [v for v in model.parameters() if v.requires_grad]
    required_bytes = sum(v.numel() * v.element_size() for v in parameters)
    if shutil.disk_usage(ROOT).free < 25 * 2**30 + required_bytes + 20 * 2**20:
        raise RuntimeError("Insufficient room above 25 GiB reserve for expert overlay")
    optimizer_cfg = model.config.get_optimizer_preset()
    optimizer_cfg.lr = 1e-5
    optimizer = optimizer_cfg.build(parameters)
    assert_fp32_optimizer(model, optimizer)
    scheduler_cfg = model.config.get_scheduler_preset()
    scheduler_cfg.peak_lr, scheduler_cfg.decay_lr = 1e-5, 2.5e-7
    scheduler_cfg.num_decay_steps, scheduler_cfg.num_warmup_steps = args.steps, max(1, args.steps // 30)
    scheduler = scheduler_cfg.build(optimizer, args.steps)
    sampler, noise_sampler = random.Random(1001), random.Random(1002)
    binding = {"mode": args.mode, "episode": record, "export_sha256": sha256(EXPORT / "openarm_export.json"),
        "base_sha256": sha256(BASE / "model.safetensors"), "trainer_sha256": sha256(Path(__file__)),
        "augmentation_sha256": sha256(ROOT / "learning/vla/gripper_feedback.py"),
        "overlay_sha256": sha256(ROOT / "learning/vla/expert_overlay.py"),
        "eligible_indices": eligible, "steps": args.steps, "batch_size": 8, "learning_rate": 1e-5,
        "sampling": "4 open-hold + 4 uniform same-parent anchors per update in both conditions",
        "augmentation": "at most +/-2mm physical gripper sensor noise on first four open-hold samples only" if args.mode == "augmented" else "none",
        "deployment_observations": "unchanged live state/images; no heuristic gripper correction",
        "robot_execution": False, "normalization": "immutable original base processors"}
    binding["precision"] = {"forward": "bf16 autocast", "trainable_and_optimizer_storage": "float32", **precision_audit}
    if args.selection:
        binding.update(selection_sha256=sha256(args.selection), training_parents=parents,
            parent_schedule='Round-robin, one parent per batch; equal per-parent update count at full step budget',
            per_parent_update_target=args.steps//len(parents),
            selection_helper_sha256=sha256(ROOT / 'learning/vla/five_episode.py'))
    write_json(args.output / "binding.json", binding)
    initial = evaluate(0)
    tracked_name, tracked = next((name, p) for name, p in model.named_parameters()
        if p.requires_grad and "lm_expert.layers.0.self_attn.q_proj.weight" in name)
    tracked_before = tracked.detach().float().cpu().clone()
    update_audit = None
    checks, updates, reason = [initial], 0, "step_budget"
    write_json(args.output / "initial.json", initial)
    for step in range(1, args.steps + 1):
        if time.monotonic() - begun > args.max_seconds - 90:
            reason = "time_budget"
            break
        if shutil.disk_usage(ROOT).free < 25 * 2**30 + required_bytes + 12 * 2**20:
            reason = "disk_reserve"
            break
        current = parents[(step-1) % len(parents)]
        indices = [current['start'] + sampler.choice(current['open_hold_local']) for _ in range(4)] + [sampler.randrange(current['start'], current['end']) for _ in range(4)]
        offsets = [noise_sampler.uniform(-.002, .002) for _ in range(4)]
        raw = default_collate([dataset[i] for i in indices])
        expected_episode = next(i for i,r in enumerate(receipt['datasets']['train']['episodes']) if r['episode'] == current['episode'])
        if not torch.all(raw["episode_index"] == expected_episode):
            raise ValueError("Other-parent exposure")
        raw["observation.state"] = raw["observation.state"].clone()
        original_grip = raw["observation.state"][:4, 15].clone()
        if args.mode == "augmented":
            for n in range(4):
                raw["observation.state"][n, 15] = perturbed_gripper(float(original_grip[n]), offsets[n])
        applied_offsets = (raw["observation.state"][:4, 15] - original_grip).tolist()
        batch = actor.pre(raw)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss, _ = masked_forward(model, batch)
        if not torch.isfinite(loss):
            raise RuntimeError("Nonfinite loss")
        loss.backward()
        if any(p.grad is not None for p in model.parameters() if not p.requires_grad):
            raise RuntimeError("Frozen parameter gradient")
        norm = torch.nn.utils.clip_grad_norm_(parameters, optimizer_cfg.grad_clip_norm, error_if_nonfinite=True)
        if step == 1:
            reference_low = torch.nn.Parameter(tracked_before.to(torch.bfloat16))
            low_before = reference_low.detach().clone()
            low_optim = torch.optim.AdamW([reference_low], lr=optimizer.param_groups[0]["lr"],
                betas=optimizer_cfg.betas, eps=optimizer_cfg.eps, weight_decay=optimizer_cfg.weight_decay)
            reference_low.grad = tracked.grad.detach().to(device="cpu", dtype=torch.bfloat16)
            low_optim.step()
            update_audit = {"parameter": tracked_name,
                "gradient_nonzero_fraction": float((tracked.grad != 0).float().mean()),
                "same_gradient_bf16_changed_fraction": float((reference_low != low_before).float().mean()),
                "learning_rate": optimizer.param_groups[0]["lr"]}
        optimizer.step()
        if step == 1:
            update_audit["actual_fp32_changed_fraction"] = float((tracked.detach().cpu() != tracked_before).float().mean())
            write_json(args.output / "precision_update_audit.json", update_audit)
        if step == 1 or step % 200 == 0:
            assert_fp32_optimizer(model, optimizer)
        scheduler.step()
        updates = step
        row = {"step": step, "episode":current['episode'], "elapsed_s": time.monotonic() - begun, "indices": indices,
            "gripper_sensor_offsets_m": applied_offsets, "loss": float(loss.detach()), "grad_norm": float(norm),
            "lr": scheduler.get_last_lr()[0], "peak_allocated_gb": torch.cuda.max_memory_allocated() / 1e9}
        with (args.output / "updates.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row) + "\n")
        if step % 50 == 0:
            print(json.dumps(row), flush=True)
        if step % (800 if args.selection else 200) == 0:
            check = evaluate(step)
            checks.append(check)
            write_json(args.output / f"probes_{step:04d}.json", check)
            print(json.dumps({"evaluation_step": step, "metrics": check["metrics"]}), flush=True)
    final = evaluate(updates)
    overlay = None if args.qualify_only else save_overlay(model, BASE, args.output / "expert")
    write_json(args.output / "report.json", {"completed": True, "binding": binding, "updates": updates,
        "stop_reason": reason, "initial": initial, "final": final, "evaluations": checks,
        "overlay": overlay, "elapsed_s": time.monotonic() - begun,
        "precision_update_audit": update_audit, "qualification_only": args.qualify_only,
        "limits": "Five training-parent recipe-scaling pilot; not full validation or autonomous success." if args.selection else "One training-parent fitting/feedback test. Not autonomous success, full validation or generalization."})
    print(json.dumps({"finished": args.mode, "updates": updates, "metrics": final["metrics"]}), flush=True)


if __name__ == "__main__":
    main()
