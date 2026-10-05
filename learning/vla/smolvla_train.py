"""Fail-closed, local-only launcher for an upstream SmolVLA smoke run."""
import json
from pathlib import Path
import sys

from learning.vla.smolvla_data import CONTRACT, LEROBOT_VERSION, CAMERAS


def training_command(export, checkpoint, output, *, steps=100, batch_size=1, seed=1000):
    for value in (steps, batch_size):
        if type(value) is not int or value < 1:
            raise ValueError("Positive integer steps and batch size required")
    if type(seed) is not int or seed < 0:
        raise ValueError("Nonnegative integer seed required")
    export, checkpoint, output = (Path(p).resolve() for p in (export, checkpoint, output))
    if output.exists():
        raise FileExistsError("Use a new run directory; resume is not implemented in this launcher")
    if any(output.is_relative_to(p) or p.is_relative_to(output) for p in (export, checkpoint)):
        raise ValueError("Run output must be disjoint from export and checkpoint roots")
    receipt = json.loads((export / "openarm_export.json").read_text(encoding="utf-8"))
    if receipt["contract"] != CONTRACT or receipt["lerobot_version"] != LEROBOT_VERSION:
        raise ValueError("Incompatible export contract/version")
    train = receipt["datasets"]["train"]
    if train["relative_root"] != "train":
        raise ValueError("Unexpected training split path")
    info = json.loads((export / "train/meta/info.json").read_text(encoding="utf-8"))
    if info["fps"] != 50 or info["total_episodes"] != len(train["episodes"]):
        raise ValueError("Export metadata differs from receipt")
    expected = {"observation.state": [16], "action": [8]}
    expected.update({key: receipt["image_shape_hwc"] for key in CAMERAS})
    for key, shape in expected.items():
        if info["features"][key]["shape"] != shape:
            raise ValueError(f"Unexpected training feature: {key}")
    config = json.loads((checkpoint / "config.json").read_text(encoding="utf-8"))
    if config.get("type") != "smolvla" or not (checkpoint / "model.safetensors").is_file():
        raise ValueError("Expected a local SmolVLA checkpoint with config and weights")
    # Clear pretrained embodiment features so the factory derives OpenArm features.
    # Dataset is train-only, so upstream statistics cannot include validation data.
    return [sys.executable, "-m", "scripts.lerobot_smolvla_train",
            f"--policy.path={checkpoint}", "--policy.input_features={}",
            "--policy.device=cuda", "--policy.push_to_hub=false", "--policy.use_amp=false",
            "--policy.freeze_vision_encoder=true", "--policy.train_expert_only=true",
            "--policy.train_state_proj=true", "--policy.adapt_to_pi_aloha=false",
            "--policy.use_delta_joint_actions_aloha=false", "--policy.empty_cameras=0",
            "--policy.chunk_size=50", "--policy.n_action_steps=10",
            f"--dataset.repo_id={train['repo_id']}", f"--dataset.root={export / 'train'}",
            f"--output_dir={output}", f"--steps={steps}", f"--batch_size={batch_size}",
            f"--seed={seed}", "--num_workers=0", "--eval_freq=0", "--log_freq=10",
            f"--save_freq={steps}", "--wandb.enable=false"]
