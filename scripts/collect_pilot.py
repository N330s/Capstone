"""Collect small physical-grasp expert data; never overwrite an existing dataset."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from envs.openarm_insert import OpenArmInsertEnv
from controllers.expert import InsertionExpert
from data_pipeline.episodes import save_episode,SCHEMA


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--episodes",type=int,default=20)
    p.add_argument("--output",type=Path,default=Path("data/openarm_v1_pilot"))
    args=p.parse_args()
    if args.episodes<2:
        raise ValueError("Need at least two pilot episodes for an episode-level split")
    args.output.mkdir(parents=True,exist_ok=False)
    env=OpenArmInsertEnv(images=True)
    records=[]
    try:
        for seed in range(args.episodes):
            rng=np.random.default_rng(seed)
            options=dict(zip(("offset_y_m","offset_z_m"),rng.uniform(-.0003,.0003,2)))
            obs,info=env.reset(seed=seed,options=options)
            observations,infos=[obs],[info]
            states,velocities=[env.data.qpos.copy()],[env.data.qvel.copy()]
            requested,applied,rewards,terms,truncs,durations,phases=[],[],[],[],[],[],[]
            expert=InsertionExpert(env)
            while env.outcome is None:
                action=expert.action()
                phase=expert.phase
                obs,reward,term,trunc,info=env.step(action)
                observations.append(obs)
                infos.append(info)
                requested.append(action.copy())
                applied.append(info["applied_action"].copy())
                states.append(env.data.qpos.copy())
                velocities.append(env.data.qvel.copy())
                rewards.append(reward)
                terms.append(term)
                truncs.append(trunc)
                durations.append(info["executed_dt_s"])
                phases.append(phase)
            if env.outcome!="success":
                raise RuntimeError(f"Pilot seed {seed} failed: {env.outcome}; not exporting as expert data")
            name=f"episode_{seed:04d}"
            metadata=save_episode(args.output/name,observations,requested,applied,states,velocities,
                                  rewards,terms,truncs,durations,phases,infos,env.manifest())
            records.append({"episode":name,"seed":seed,"steps":metadata["steps"],
                            "split":"validation" if seed>=int(.8*args.episodes) else "train",
                            "data_sha256":metadata["data_sha256"]})
            print(name,env.outcome,metadata["steps"],flush=True)
        manifest={"schema":SCHEMA,"episodes":records,"purpose":"pilot collection/replay; not evidence of VLA generalization",
                  "reset_distribution":"uniform +/-0.3 mm Y/Z offsets around one held-plug approach",
                  "instruction":env.config["instruction"],"robot_manifest":env.manifest()}
        (args.output/"manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    finally:
        env.close()

if __name__=="__main__":
    main()

