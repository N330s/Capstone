"""Replay every command against the recorded robot state without running the expert."""
import argparse
import hashlib
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from envs.openarm_insert import OpenArmInsertEnv
from data_pipeline.episodes import validate_episode

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset",type=Path,default=Path("data/openarm_v1_pilot"))
    args=p.parse_args()
    manifest=json.loads((args.dataset/"manifest.json").read_text())
    env=OpenArmInsertEnv(images=False)
    rows=[]
    try:
        for episode in manifest["episodes"]:
            path=args.dataset/episode["episode"]
            meta=validate_episode(path)
            env.reset(seed=meta["seed"],options=meta["reset_options"])
            current=env.manifest()
            for key in ("scene_sha256","source_hashes","robot_config","contact_config","mujoco_version"):
                if current[key]!=meta[key]:
                    raise ValueError(f"Cannot replay changed {key}")
            with np.load(path/"episode.npz",allow_pickle=False) as a:
                max_error=float(np.max(np.abs(env.data.qpos-a["qpos"][0])))
                velocity_error=float(np.max(np.abs(env.data.qvel-a["qvel"][0])))
                for i,action in enumerate(a["action_requested"]):
                    _,reward,term,trunc,info=env.step(action)
                    max_error=max(max_error,float(np.max(np.abs(env.data.qpos-a["qpos"][i+1]))))
                    velocity_error=max(velocity_error,float(np.max(np.abs(env.data.qvel-a["qvel"][i+1]))))
                    if not np.array_equal(info["applied_action"],a["action_applied"][i]):
                        raise AssertionError("Action execution mismatch")
                    if (reward!=a["reward"][i] or term!=a["terminated"][i] or trunc!=a["truncated"][i]):
                        raise AssertionError("Outcome timing mismatch")
            if max_error>1e-8 or velocity_error>1e-7:
                raise AssertionError(f"Replay drift: {max_error}, {velocity_error}")
            rows.append({"episode":episode["episode"],"max_qpos_error":max_error,
                         "max_qvel_error":velocity_error,"outcome":env.outcome})
            print(rows[-1],flush=True)
    finally:
        env.close()
    (args.dataset/"replay_report.json").write_text(json.dumps({"passed":True,"episodes":rows},indent=2))

if __name__=="__main__":
    main()

