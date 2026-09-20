"""Automated collection over randomized plug/socket scenes.

Two passes:

  preflight  images off, fast. Walks sampled scenes until --episodes of them
             succeed. Every failure is retained in rejected_scenes.json with its
             phase log and failure reason, plus a qpos trajectory under
             failed_preflight/ that scripts/replay_episode.py can play back --
             resampling is budgeted and logged, never silent.
  record     images on, replays exactly the accepted (seed, options) pairs and
             writes episodes. A rollout that fails here is retained under
             failed_attempts/ and is never labelled as BC data.

Watching and reviewing:

    python scripts/collect_random.py --episodes 20 --detector privileged
        opens a live MuJoCo window (default) and writes video of every failure

    python scripts/collect_random.py --episodes 500 --record --headless
        no window, for real collection runs

    python scripts/collect_random.py --episodes 20 --video all --realtime .3
        slow motion, keep video of successes too

    python scripts/replay_episode.py data/<out>/failed_preflight/train_00003.npz
"""
import argparse
import hashlib
import json
import time
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np

from controllers.pick_insert_expert import PickInsertExpert
from data_pipeline.scene_bank import (collection_bank, evaluation_bank, as_reset_options,
                                      COLLECTION_SEED_BASE, EVALUATION_SEED_BASE)
from data_pipeline.episodes import SCHEMA, save_episode, clean_info
from envs.openarm_insert import OpenArmInsertEnv
from rollout_viewer import LiveViewer, VideoRecorder, ViewerClosed

MAX_STEPS = 6000


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rollout(env, row, *, record=False, detector_mode="noisy", probe_offset_y_m=0.0,
            viewer=None, recorder=None):
    obs, info = env.reset(seed=row["seed"], options=as_reset_options(row["options"]))
    start_info = clean_info(info)
    if viewer is not None:
        viewer.episode(row["id"])
    if recorder is not None:
        recorder.reset()
    observations, infos = [obs], [info]
    qpos, qvel = [env.data.qpos.copy()], [env.data.qvel.copy()]
    requested, applied, rewards, terms, truncs, durations, phases = [], [], [], [], [], [], []
    # Always kept, recording or not, so a preflight failure is still replayable.
    trace_qpos, trace_phase = [env.data.qpos.copy()], []
    expert = PickInsertExpert(env, rng=np.random.default_rng(row["seed"]),
                              detector_mode=detector_mode,
                              probe_offset_y_m=probe_offset_y_m)
    steps = 0
    while env.outcome is None and expert.failure is None and steps < MAX_STEPS:
        action = expert.action()
        phase = expert.phase
        obs, reward, term, trunc, info = env.step(action)
        steps += 1
        trace_qpos.append(env.data.qpos.copy())
        trace_phase.append(phase)
        if viewer is not None:
            viewer.sync()
        if recorder is not None:
            recorder.capture(phase)
        if record:
            observations.append(obs); infos.append(info)
            qpos.append(env.data.qpos.copy()); qvel.append(env.data.qvel.copy())
            requested.append(action.copy()); applied.append(info["applied_action"].copy())
            rewards.append(reward); terms.append(term); truncs.append(trunc)
            durations.append(info["executed_dt_s"]); phases.append(phase)
    outcome = env.outcome
    if outcome is None:
        outcome = f"expert_{expert.failure}" if expert.failure else "step_budget_exhausted"
    summary = {**{k: v for k, v in row.items() if k != "sampler"},
               "outcome": outcome, "steps": steps,
               "expert": expert.diagnostics(),
               "initial": start_info, "final": clean_info(info)}
    payload = (observations, requested, applied, qpos, qvel,
               rewards, terms, truncs, durations, phases, infos)
    trace = {"qpos": np.asarray(trace_qpos), "phase": np.asarray(trace_phase)}
    return summary, payload, trace


def save_failure(directory, row_id, summary, trace, recorder=None, payload=None):
    directory.mkdir(parents=True, exist_ok=True)
    write_json(directory / (row_id + ".json"), summary)
    arrays = {"qpos": trace["qpos"], "phase": trace["phase"]}
    if payload is not None:
        arrays.update(action_requested=np.asarray(payload[1]), action_applied=np.asarray(payload[2]),
                      qvel=np.asarray(payload[4]))
    np.savez_compressed(directory / (row_id + ".npz"), **arrays)
    if recorder is not None:
        recorder.save(directory / row_id, metadata={"outcome": summary["outcome"],
                                                    "expert": summary["expert"]})


def make_viewer(env, args):
    if args.headless:
        return None
    try:
        return LiveViewer(env, realtime=args.realtime, sync_every=args.sync_every)
    except Exception as error:
        print(f"live viewer unavailable ({type(error).__name__}: {error}); continuing headless",
              flush=True)
        return None


def make_recorder(env, args):
    if args.video == "none":
        return None
    try:
        return VideoRecorder(env, camera=args.video_camera, stride=args.video_stride)
    except Exception as error:
        print(f"video recorder unavailable ({type(error).__name__}: {error})", flush=True)
        return None


def preflight_pass(args, out):
    """Walk sampled scenes until enough succeed or the attempt budget runs out."""
    accepted, rejected = [], []
    budget = int(args.episodes * args.attempt_ratio)
    env = OpenArmInsertEnv(images=False)
    viewer, recorder = make_viewer(env, args), make_recorder(env, args)
    started, cursor = time.time(), 0
    try:
        while len(accepted) < args.episodes and cursor < budget:
            batch = collection_bank(n=min(64, budget - cursor),
                                    validation_fraction=args.validation_fraction,
                                    start=cursor)
            for row in batch:
                if len(accepted) >= args.episodes:
                    break
                cursor += 1
                trace = None
                try:
                    result, _, trace = rollout(env, row, detector_mode=args.detector,
                                               viewer=viewer, recorder=recorder)
                except ViewerClosed:
                    raise SystemExit("Viewer closed; stopping. Rerun with --headless to collect.")
                except (ValueError, RuntimeError) as error:
                    result = {**row, "outcome": "reset_or_runtime_rejection",
                              "error": f"{type(error).__name__}: {error}"}
                ok = result["outcome"] == "success"
                (accepted if ok else rejected).append(result)
                if not ok and trace is not None:
                    save_failure(out / "failed_preflight", row["id"], result, trace,
                                 recorder=recorder)
                elif ok and recorder is not None and args.video == "all":
                    recorder.save(out / "video" / row["id"], metadata={"outcome": "success"})
                detail = result.get("error") or (result.get("expert") or {}).get("failure") or ""
                print(f"{row['id']} {result['outcome']} "
                      f"({len(accepted)}/{args.episodes} accepted, {cursor} attempted)"
                      f"{' | ' + str(detail) if detail else ''}", flush=True)
                if len(accepted) == 0 and cursor >= args.early_abort:
                    write_json(out / "rejected_scenes.json", rejected)
                    raise SystemExit(
                        f"Aborting: {cursor} attempts, zero successes. Last failure: "
                        f"{result['outcome']} {detail}. Diagnostics and video are in "
                        f"{out/'failed_preflight'}; replay with scripts/replay_episode.py.")
                if cursor % 10 == 0:
                    write_json(out / "preflight_accepted.json", accepted)
                    write_json(out / "rejected_scenes.json", rejected)
    finally:
        if recorder is not None:
            recorder.close()
        if viewer is not None:
            viewer.close()
        env.close()
    write_json(out / "preflight_accepted.json", accepted)
    write_json(out / "rejected_scenes.json", rejected)
    reasons = {}
    for r in rejected:
        key = r["outcome"]
        if key.startswith("expert_"):
            key = f"{key} ({(r.get('expert') or {}).get('phase')})"
        reasons[key] = reasons.get(key, 0) + 1
    write_json(out / "preflight_stats.json",
               {"requested": args.episodes, "accepted": len(accepted),
                "attempted": cursor, "attempt_budget": budget,
                "acceptance_rate": len(accepted) / max(cursor, 1),
                "failure_reasons": reasons, "wall_seconds": round(time.time() - started, 1),
                "detector_mode": args.detector})
    return accepted, cursor


def record_pass(args, out, accepted):
    records, failures = [], []
    manifest_path = out / "manifest.json"
    if args.resume and manifest_path.exists():
        records = json.loads(manifest_path.read_text(encoding="utf-8")).get("episodes", [])
    done = {r["episode"] for r in records}
    env = OpenArmInsertEnv(images=True)
    viewer, recorder = make_viewer(env, args), make_recorder(env, args)
    try:
        for row in accepted:
            if row["id"] in done:
                print("skip", row["id"], flush=True)
                continue
            try:
                result, payload, trace = rollout(env, row, record=True, detector_mode=args.detector,
                                                 viewer=viewer, recorder=recorder)
            except ViewerClosed:
                raise SystemExit("Viewer closed; stopping. Rerun with --resume --headless.")
            if result["outcome"] != "success":
                save_failure(out / "failed_attempts", row["id"], result, trace,
                             recorder=recorder, payload=payload)
                failures.append({"episode": row["id"], "outcome": result["outcome"],
                                 "expert": result["expert"]})
                print("FAILED (retained, not a BC label)", row["id"], result["outcome"], flush=True)
                if args.strict:
                    raise RuntimeError("Recorded rollout failed and --strict is set")
                continue
            if recorder is not None and args.video == "all":
                recorder.save(out / "video" / row["id"], metadata={"outcome": "success"})
            meta = save_episode(out / row["id"], *payload, env.manifest())
            records.append({"episode": row["id"], "seed": row["seed"], "split": row["split"],
                            "steps": meta["steps"], "data_sha256": meta["data_sha256"],
                            "retries": result["expert"]["retries"],
                            "regrasps": result["expert"]["regrasps"],
                            "detection_error": result["expert"]["detection_error"]})
            write_json(manifest_path, build_manifest(args, out, env, records, failures, accepted))
            print("recorded", row["id"], meta["steps"], flush=True)
    finally:
        if recorder is not None:
            recorder.close()
        if viewer is not None:
            viewer.close()
        env.close()
    write_json(manifest_path, build_manifest(args, out, env, records, failures, accepted))
    return records, failures


def build_manifest(args, out, env, records, failures, accepted):
    return {"schema": SCHEMA,
            "episodes": records,
            "failed_recordings": failures,
            "instruction": env.config["instruction"],
            "purpose": "randomized table-top pick-and-insert; plug and socket poses sampled",
            "task_stages": ["survey", "grasp", "transport", "insert"],
            "detector_mode": args.detector,
            "robot_manifest": env.manifest(),
            "complete": len(records) == len(accepted),
            "collection_script_sha256": sha256(__file__),
            "expert_sha256": sha256(sys.modules[PickInsertExpert.__module__].__file__),
            "scene_bank_module_sha256": sha256(sys.modules[collection_bank.__module__].__file__),
            "scene_bank_sha256": sha256(out / "collection_bank.json"),
            "evaluation_bank_sha256": sha256(out / "evaluation_bank.json"),
            "seed_namespaces": {"collection": COLLECTION_SEED_BASE,
                                "evaluation": EVALUATION_SEED_BASE}}


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output", type=Path, default=Path("data/openarm_v2_random"))
    p.add_argument("--episodes", type=int, default=200, help="successful episodes wanted")
    p.add_argument("--early-abort", type=int, default=8,
                   help="give up if this many attempts pass with zero successes")
    p.add_argument("--attempt-ratio", type=float, default=3.0,
                   help="preflight attempt budget as a multiple of --episodes")
    p.add_argument("--validation-fraction", type=float, default=0.1)
    p.add_argument("--detector", choices=("noisy", "privileged"), default="noisy")
    p.add_argument("--record", action="store_true", help="Record images for accepted scenes")
    p.add_argument("--resume", action="store_true", help="Reuse an existing output directory")
    p.add_argument("--strict", action="store_true",
                   help="Abort the record pass on the first failed rollout")
    p.add_argument("--headless", action="store_true",
                   help="No live window. Use this for unattended collection runs.")
    p.add_argument("--realtime", type=float, default=1.0,
                   help="Viewer pacing: 1 is real time, .25 is slow motion, 0 is uncapped")
    p.add_argument("--sync-every", type=int, default=1, help="Viewer refresh every N control steps")
    p.add_argument("--video", choices=("none", "failures", "all"), default="failures",
                   help="Keep video for later review")
    p.add_argument("--video-camera", default="scene_rgb")
    p.add_argument("--video-stride", type=int, default=2, help="Capture every Nth control step")
    args = p.parse_args()

    out = args.output
    out.mkdir(parents=True, exist_ok=args.resume)
    write_json(out / "evaluation_bank.json",
               {"status": "reserved; not executed or used for tuning", "resets": evaluation_bank()})

    accepted_path = out / "preflight_accepted.json"
    if args.resume and accepted_path.exists():
        accepted = json.loads(accepted_path.read_text(encoding="utf-8"))
        attempted = len(accepted)
        print(f"resuming with {len(accepted)} accepted scenes", flush=True)
    else:
        accepted, attempted = preflight_pass(args, out)

    write_json(out / "collection_bank.json", accepted)
    if not accepted:
        raise SystemExit("Preflight accepted nothing; check the Workspace bounds and the expert.")
    if len(accepted) < args.episodes:
        print(f"WARNING: only {len(accepted)}/{args.episodes} scenes succeeded in "
              f"{attempted} attempts. Retained diagnostics in rejected_scenes.json.", flush=True)

    if not args.record:
        return
    records, failures = record_pass(args, out, accepted)
    print(f"done: {len(records)} episodes recorded, {len(failures)} retained as failures", flush=True)


if __name__ == "__main__":
    main()