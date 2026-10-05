"""Print an offline smoke-training command; --run explicitly executes it."""
import argparse
from importlib.metadata import version
import json
import os
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from learning.vla.smolvla_data import LEROBOT_VERSION
from learning.vla.smolvla_train import training_command


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    command = training_command(args.export, args.checkpoint, args.output,
                               steps=args.steps, batch_size=args.batch_size, seed=args.seed)
    print(json.dumps({"argv": command, "executing": args.run,
                      "warning": "Smoke training only; no closed-loop evaluation or resume validation yet"}, indent=2), flush=True)
    if args.run:
        if version("lerobot") != LEROBOT_VERSION:
            raise RuntimeError(f"Expected lerobot=={LEROBOT_VERSION}")
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("This launcher requires CUDA. Install a compatible CUDA PyTorch build in the training environment; the CPU export environment is not sufficient.")
        env = {**os.environ, "HF_HUB_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1", "WANDB_MODE": "disabled"}
        subprocess.run(command, env=env, check=True)  # No shell, upload or implicit downloads.


if __name__ == "__main__":
    main()
