"""Preflight, then record the fixed ten-reset batch; retain failed attempts separately."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from controllers.expert import InsertionExpert
from data_pipeline.collection_bank import collection_bank, evaluation_bank
from data_pipeline.episodes import SCHEMA, save_episode, clean_info
from envs.openarm_insert import OpenArmInsertEnv


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def rollout(env, row, record=False):
    obs, info = env.reset(seed=row["seed"], options=row["options"])
    start_info = clean_info(info)
    observations, infos = [obs], [info]
    qpos, qvel = [env.data.qpos.copy()], [env.data.qvel.copy()]
    requested, applied, rewards, terms, truncs, durations, phases = [], [], [], [], [], [], []
    expert = InsertionExpert(env)
    while env.outcome is None:
        action = expert.action()
        phase = expert.phase
        obs, reward, term, trunc, info = env.step(action)
        if record:
            observations.append(obs); infos.append(info)
            qpos.append(env.data.qpos.copy()); qvel.append(env.data.qvel.copy())
            requested.append(action.copy()); applied.append(info["applied_action"].copy())
            rewards.append(reward); terms.append(term); truncs.append(trunc)
            durations.append(info["executed_dt_s"]); phases.append(phase)
    summary = {**row, "outcome": env.outcome, "retries": expert.retries,
               "initial": start_info, "final": clean_info(info)}
    payload = (observations, requested, applied, qpos, qvel, rewards, terms, truncs, durations, phases, infos)
    return summary, payload


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=Path("data/openarm_v1_varied10"))
    p.add_argument("--record", action="store_true", help="Record images after all ten preflights succeed")
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    bank, heldout = collection_bank(), evaluation_bank()
    write_json(args.output/"collection_bank.json", bank)
    write_json(args.output/"evaluation_bank.json", {"status":"reserved; not executed or used for tuning", "resets":heldout})
    preflight = []
    env = OpenArmInsertEnv(images=False)
    try:
        for row in bank:
            try:
                result, _ = rollout(env,row)
            except (ValueError,RuntimeError) as error:
                result = {**row,"outcome":"reset_or_runtime_rejection","error":str(error)}
            preflight.append(result)
            write_json(args.output/"preflight.json",preflight)
            print(row["id"],result["outcome"],flush=True)
    finally:
        env.close()
    if not all(r["outcome"]=="success" for r in preflight):
        raise SystemExit("Preflight failed: retained diagnostics; no BC data exported. Do not silently resample.")
    if not args.record:
        return
    records = []
    env = OpenArmInsertEnv(images=True)
    try:
        for row in bank:
            result,payload = rollout(env,row,record=True)
            if result["outcome"] != "success":
                failed = args.output/"failed_attempts"
                failed.mkdir(exist_ok=True)
                write_json(failed/(row["id"]+".json"),result)
                np.savez_compressed(failed/(row["id"]+".npz"),
                    action_requested=np.asarray(payload[1]),action_applied=np.asarray(payload[2]),
                    qpos=np.asarray(payload[3]),qvel=np.asarray(payload[4]))
                raise RuntimeError("Recorded rollout failed; retained separately, not a BC label")
            meta = save_episode(args.output/row["id"],*payload,env.manifest())
            records.append({"episode":row["id"],"seed":row["seed"],"split":row["split"],
                            "steps":meta["steps"],"data_sha256":meta["data_sha256"],"retries":result["retries"]})
            manifest = {"schema":SCHEMA,"episodes":records,"instruction":env.config["instruction"],
                        "purpose":"ten-episode varied integration batch; not broad coverage",
                        "robot_manifest":env.manifest(),"complete":len(records)==len(bank),
                        "collection_script_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                        "reset_bank_sha256":hashlib.sha256((args.output/"collection_bank.json").read_bytes()).hexdigest(),
                        "evaluation_bank_sha256":hashlib.sha256((args.output/"evaluation_bank.json").read_bytes()).hexdigest()}
            write_json(args.output/"manifest.json",manifest)
            print("recorded",row["id"],meta["steps"],flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()
