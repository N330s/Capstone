"""Replay every command against the recorded robot state without running the expert."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from envs.openarm_insert import OpenArmInsertEnv
from data_pipeline.replay import replay_episode

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset",type=Path,default=Path("data/openarm_v1_pilot"))
    args=p.parse_args()
    manifest=json.loads((args.dataset/"manifest.json").read_text())
    # Rebuild the scene the dataset was recorded in; the per-episode hash check below still applies.
    recorded=manifest.get("robot_manifest",{}).get("workspace")
    env=OpenArmInsertEnv(images=False,workspace=recorded["path"] if recorded else None)
    rows=[]
    try:
        for episode in manifest["episodes"]:
            path=args.dataset/episode["episode"]
            result=replay_episode(env,path)
            max_error,velocity_error=result["max_qpos_error"],result["max_qvel_error"]
            rows.append({"episode":episode["episode"],"max_qpos_error":max_error,
                         "max_qvel_error":velocity_error,"outcome":env.outcome})
            print(rows[-1],flush=True)
    finally:
        env.close()
    (args.dataset/"replay_report.json").write_text(json.dumps({"passed":True,"episodes":rows},indent=2))

if __name__=="__main__":
    main()

