"""Automated collection over randomized plug/socket scenes.

Each sampled scene goes through two stages, one scene at a time:

  preflight  images off, fast. Every failure is retained in rejected_scenes.json
             with its phase log and failure reason, plus a qpos trajectory under
             failed_preflight/ that scripts/replay_episode.py can play back --
             resampling is budgeted and logged, never silent.
  record     only for a scene whose preflight succeeded, straight away: images
             on, the same (seed, options) again, written as an episode. A rollout
             that fails here is retained under failed_attempts/ and is never
             labelled as BC data.

The run stops once --episodes episodes are recorded (or the attempt budget is
spent). Without --record only the preflight stage runs.

Stopping and resuming (--record runs): progress.json is updated after every
stage. Rerun the same command with --resume: a scene interrupted while recording
is recorded again (its partial folder is replaced); a scene interrupted during
preflight is preflighted again. Nothing finished is redone.

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
import itertools
import json
import os
import shutil
import time
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np

from controllers.pick_insert_expert import PickInsertExpert
from data_pipeline.scene_bank import (collection_bank, evaluation_bank, as_reset_options,
                                      workspace_for_env, COLLECTION_SEED_BASE, EVALUATION_SEED_BASE,
                                      iter_collection_bank)
from data_pipeline.episodes import SCHEMA, save_episode, clean_info
from envs.openarm_insert import OpenArmInsertEnv
from rollout_viewer import LiveViewer, VideoRecorder, ViewerClosed

MAX_STEPS = 6000
# Rows are drawn in blocks of this size; the validation split is the tail of each block.
BATCH = 64


def write_json(path, value):
    # Write then rename: a run killed mid-write leaves the previous file, never a truncated one.
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, indent=2), encoding="utf-8")
    os.replace(tmp, path)


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


def preflight_pass(args, out, ws):
    """Walk sampled scenes until enough succeed or the attempt budget runs out."""
    accepted, rejected = [], []
    budget = int(args.episodes * args.attempt_ratio)
    env = OpenArmInsertEnv(images=False, workspace=args.workspace)
    viewer, recorder = make_viewer(env, args), make_recorder(env, args)
    started, cursor = time.time(), 0
    try:
        while len(accepted) < args.episodes and cursor < budget:
            # Same (n, start) grid collection_bank would use for this stretch
            # -- and therefore the same seed/split assignment -- but rows are
            # generated one at a time, so a batch is never fully materialized
            # (each row costs a live PickFeasibility/InsertFeasibility plan)
            # when the target is reached partway through it.
            batch_n = min(BATCH, budget - cursor)
            batch = iter_collection_bank(n=batch_n,
                                         validation_fraction=args.validation_fraction,
                                         ws=ws, start=cursor)
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
                expert_diag = result.get("expert") or {}
                detail = (result.get("error") or expert_diag.get("failure_detail")
                          or expert_diag.get("failure") or "")
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


def run_signature(args):
    """What decides which scenes a run draws; --resume refuses to continue under different values."""
    return {"workspace": args.workspace.as_posix() if args.workspace else None,
            "detector": args.detector, "validation_fraction": args.validation_fraction}


def row_artifacts(directory, row_id):
    """Everything save_failure / VideoRecorder.save wrote for one row (<id>, <id>.json, <id>.mp4, ...)."""
    if not directory.is_dir():
        return []
    return [p for p in directory.iterdir() if p.name == row_id or p.name.startswith(row_id + ".")]


def remove_partial(paths):
    """Delete what an interrupted stage left behind, so the stage is redone from scratch."""
    for path in paths:
        if path.is_dir():
            shutil.rmtree(path)
        elif path.exists():
            path.unlink()
        else:
            continue
        print("removed partial", path, flush=True)


def collect_pass(args, out, ws, state):
    """Preflight scene after scene; record each one straight after a successful preflight.

    state (progress.json) is saved after every stage. state["cursor"] is the next row to
    preflight; state["pending_record"] is an accepted row whose recording has not finished."""
    budget = int(args.episodes * args.attempt_ratio)
    accepted, rejected = state["accepted"], state["rejected"]
    records, failures = state["records"], state["failures"]
    # One env for both stages: observe() renders only while env.images is set, so a
    # preflight costs no rendering and the process keeps a single model in memory.
    env = OpenArmInsertEnv(images=False, workspace=args.workspace)
    viewer, recorder = make_viewer(env, args), make_recorder(env, args)
    started = time.time()

    def save_progress():
        write_json(out / "preflight_accepted.json", accepted)
        write_json(out / "rejected_scenes.json", rejected)
        write_json(out / "collection_bank.json", accepted)
        write_json(out / "manifest.json", build_manifest(args, out, env, records, failures,
                                                         complete=len(records) >= args.episodes))
        write_json(out / "progress.json", state)

    def record(row, preflight_qpos):
        print("recording", row["id"], flush=True)
        remove_partial([out / row["id"], *row_artifacts(out / "failed_attempts", row["id"]),
                        *row_artifacts(out / "video", row["id"])])
        env.images = True
        try:
            result, payload, trace = rollout(env, row, record=True, detector_mode=args.detector,
                                             viewer=viewer, recorder=recorder)
        except ViewerClosed:
            raise SystemExit("Viewer closed; stopping. Rerun with --resume --headless.")
        finally:
            env.images = False
        # Rendering only reads the state, so the recorded run must retrace its preflight
        # exactly. None = not checked (the preflight happened before a resume).
        matches = None if preflight_qpos is None else bool(np.array_equal(trace["qpos"],
                                                                          preflight_qpos))
        if matches is False:
            print("WARNING: recorded trajectory differs from its preflight", row["id"], flush=True)
        if result["outcome"] != "success":
            save_failure(out / "failed_attempts", row["id"], result, trace,
                         recorder=recorder, payload=payload)
            failures.append({"episode": row["id"], "outcome": result["outcome"],
                             "expert": result["expert"], "trace_matches_preflight": matches})
            print("FAILED (retained, not a BC label)", row["id"], result["outcome"], flush=True)
        else:
            if recorder is not None and args.video == "all":
                recorder.save(out / "video" / row["id"], metadata={"outcome": "success"})
            meta = save_episode(out / row["id"], *payload, env.manifest())
            records.append({"episode": row["id"], "seed": row["seed"], "split": row["split"],
                            "steps": meta["steps"], "data_sha256": meta["data_sha256"],
                            "retries": result["expert"]["retries"],
                            "regrasps": result["expert"]["regrasps"],
                            "detection_error": result["expert"]["detection_error"],
                            "trace_matches_preflight": matches})
            print("recorded", row["id"], meta["steps"],
                  f"({len(records)}/{args.episodes} recorded)", flush=True)
        state["pending_record"] = None
        save_progress()
        if result["outcome"] != "success" and args.strict:
            raise RuntimeError("Recorded rollout failed and --strict is set")

    try:
        if state["pending_record"] is not None:
            print("resume: the previous run stopped while recording", state["pending_record"]["id"],
                  flush=True)
            record(state["pending_record"], None)
        while len(records) < args.episodes and state["cursor"] < budget:
            # Rows come from the same BATCH-sized blocks as preflight_pass, so a row's split
            # never depends on where a run was stopped. Rows drawn before a resume come back
            # from the scene cache.
            cursor = state["cursor"]
            block = cursor - cursor % BATCH
            rows = itertools.islice(
                iter_collection_bank(n=min(BATCH, budget - block),
                                     validation_fraction=args.validation_fraction,
                                     ws=ws, start=block),
                cursor - block, None)
            for row in rows:
                if len(records) >= args.episodes:
                    break
                print("preflight", row["id"], flush=True)
                remove_partial(row_artifacts(out / "failed_preflight", row["id"]))
                trace = None
                try:
                    result, _, trace = rollout(env, row, detector_mode=args.detector,
                                               viewer=viewer, recorder=recorder)
                except ViewerClosed:
                    raise SystemExit("Viewer closed; stopping. Rerun with --resume --headless.")
                except (ValueError, RuntimeError) as error:
                    result = {**{k: v for k, v in row.items() if k != "sampler"},
                              "outcome": "reset_or_runtime_rejection",
                              "error": f"{type(error).__name__}: {error}"}
                ok = result["outcome"] == "success"
                state["cursor"] += 1
                (accepted if ok else rejected).append(result)
                if not ok and trace is not None:
                    save_failure(out / "failed_preflight", row["id"], result, trace,
                                 recorder=recorder)
                expert_diag = result.get("expert") or {}
                detail = (result.get("error") or expert_diag.get("failure_detail")
                          or expert_diag.get("failure") or "")
                print(f"{row['id']} {result['outcome']} "
                      f"({len(accepted)} accepted, {state['cursor']} attempted)"
                      f"{' | ' + str(detail) if detail else ''}", flush=True)
                state["pending_record"] = result if ok else None
                save_progress()
                if not accepted and state["cursor"] >= args.early_abort:
                    raise SystemExit(
                        f"Aborting: {state['cursor']} attempts, zero successes. Last failure: "
                        f"{result['outcome']} {detail}. Diagnostics and video are in "
                        f"{out/'failed_preflight'}; replay with scripts/replay_episode.py.")
                if ok:
                    record(result, trace["qpos"])
    finally:
        if recorder is not None:
            recorder.close()
        if viewer is not None:
            viewer.close()
        env.close()
    reasons = {}
    for r in rejected:
        key = r["outcome"]
        if key.startswith("expert_"):
            key = f"{key} ({(r.get('expert') or {}).get('phase')})"
        reasons[key] = reasons.get(key, 0) + 1
    write_json(out / "preflight_stats.json",
               {"requested": args.episodes, "accepted": len(accepted), "recorded": len(records),
                "attempted": state["cursor"], "attempt_budget": budget,
                "acceptance_rate": len(accepted) / max(state["cursor"], 1),
                "failure_reasons": reasons,
                "wall_seconds_this_session": round(time.time() - started, 1),
                "detector_mode": args.detector})
    save_progress()
    return records, failures


def build_manifest(args, out, env, records, failures, complete):
    return {"schema": SCHEMA,
            "episodes": records,
            "failed_recordings": failures,
            "instruction": env.config["instruction"],
            "purpose": "randomized table-top pick-and-insert; plug and socket poses sampled",
            "task_stages": ["survey", "grasp", "transport", "insert"],
            "detector_mode": args.detector,
            "collection_mode": "per scene: preflight, then record on success (resumable)",
            "robot_manifest": env.manifest(),
            "complete": complete,
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
    p.add_argument("--resume", action="store_true",
                   help="Continue a stopped run in the same --output (a scene that was being "
                        "recorded is recorded again; one in preflight is preflighted again)")
    p.add_argument("--strict", action="store_true",
                   help="Stop on the first recorded rollout that fails")
    p.add_argument("--headless", action="store_true",
                   help="No live window. Use this for unattended collection runs.")
    p.add_argument("--realtime", type=float, default=1.0,
                   help="Viewer pacing: 1 is real time, .25 is slow motion, 0 is uncapped")
    p.add_argument("--sync-every", type=int, default=1, help="Viewer refresh every N control steps")
    p.add_argument("--video", choices=("none", "failures", "all"), default="failures",
                   help="Keep video for later review")
    p.add_argument("--video-camera", default="scene_rgb")
    p.add_argument("--video-stride", type=int, default=2, help="Capture every Nth control step")
    p.add_argument("--workspace", type=Path, default=None,
                   help="workspace spec (default configs/workspace_v1.json)")
    p.add_argument("--eval-scenes", type=int, default=0,
                   help="size of the reserved evaluation bank written first (default 100; use a few "
                        "for quick tests, a smaller bank is a prefix of the full one)")
    args = p.parse_args()

    out = args.output
    progress_path = out / "progress.json"
    state = None
    if args.record and args.resume:
        if not progress_path.exists():
            raise SystemExit(f"--resume: {progress_path} not found. Only --record runs made by this "
                             "version of the collector can be resumed.")
        state = json.loads(progress_path.read_text(encoding="utf-8"))
        if state["run"] != run_signature(args):
            raise SystemExit(f"--resume: this run was started with {state['run']}, not "
                             f"{run_signature(args)}; those decide which scenes are drawn.")
    out.mkdir(parents=True, exist_ok=args.resume)
    # Center the sampler on the real right-arm mount, as the manual collectors do.
    probe = OpenArmInsertEnv(images=False, workspace=args.workspace)
    ws = workspace_for_env(probe)
    probe.close()
    write_json(out / "evaluation_bank.json",
               {"status": "reserved; not executed or used for tuning", "scenes": args.eval_scenes,
                "resets": evaluation_bank(args.eval_scenes, ws=ws)})

    if args.record:
        if state is None:
            state = {"run": run_signature(args), "cursor": 0, "pending_record": None,
                     "accepted": [], "rejected": [], "records": [], "failures": []}
        else:
            print(f"resuming: {len(state['records'])} recorded, {state['cursor']} scenes attempted",
                  flush=True)
        records, failures = collect_pass(args, out, ws, state)
        if len(records) < args.episodes:
            print(f"WARNING: only {len(records)}/{args.episodes} episodes recorded in "
                  f"{state['cursor']} attempts. Retained diagnostics in rejected_scenes.json.",
                  flush=True)
        print(f"done: {len(records)} episodes recorded, {len(failures)} retained as failures",
              flush=True)
        return

    accepted_path = out / "preflight_accepted.json"
    if args.resume and accepted_path.exists():
        accepted = json.loads(accepted_path.read_text(encoding="utf-8"))
        attempted = len(accepted)
        print(f"preflight already done: {len(accepted)} accepted scenes", flush=True)
    else:
        accepted, attempted = preflight_pass(args, out, ws)
    write_json(out / "collection_bank.json", accepted)
    if not accepted:
        raise SystemExit("Preflight accepted nothing; check the Workspace bounds and the expert.")
    if len(accepted) < args.episodes:
        print(f"WARNING: only {len(accepted)}/{args.episodes} scenes succeeded in "
              f"{attempted} attempts. Retained diagnostics in rejected_scenes.json.", flush=True)


if __name__ == "__main__":
    main()