import argparse
import json
import sys
import time
from pathlib import Path
import numpy as np
import mujoco
import mujoco.viewer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from envs.openarm_insert import OpenArmInsertEnv

def main():
    p = argparse.ArgumentParser(description="Visually replay a recorded episode in MuJoCo.")
    p.add_argument("--dataset", type=Path, default=Path("data/openarm_v1_varied10"))
    p.add_argument("--episode", type=str, default="varied_0000", help="Episode folder name to view")
    p.add_argument("--fps", type=float, default=50.0, help="Playback speed in Hz")
    args = p.parse_args()

    ep_path = args.dataset / args.episode
    meta = json.loads((ep_path / "metadata.json").read_text())

    # Initialize environment with the saved reset seed and randomized options
    env = OpenArmInsertEnv(images=False)
    env.reset(seed=meta["seed"], options=meta["reset_options"])

    # Load stored actions from the episode NPZ file
    with np.load(ep_path / "episode.npz") as data:
        actions = data["action_requested"]

    print(f"Visualizing {args.episode} from {args.dataset} ({len(actions)} steps)...")

    # Launch the interactive MuJoCo passive viewer window
    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        step_dt = 1.0 / args.fps
        for i, action in enumerate(actions):
            if not viewer.is_running():
                break
            
            env.step(action)
            viewer.sync()
            time.sleep(step_dt)

        print("Playback finished. Keeping window open for 3 seconds...")
        time.sleep(3.0)

    env.close()

if __name__ == "__main__":
    main()