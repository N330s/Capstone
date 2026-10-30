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

Row numbering: attempt k draws bank row --start-row + k (seed COLLECTION_SEED_BASE + row).
The default 0 reproduces earlier runs exactly; e.g. --start-row 5000 draws a fresh set of
scenes. Blocks and the validation split stay aligned relative to --start-row.

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

Recovery demonstrations (--faults, controllers/faults.py): every scene gets one injected
mistake -- premature_close, wrong_grasp or misaligned_insert, cycled with the seed -- that
the expert then recovers from. Episodes carry a per-step step_label (expert / fault /
recovery); fault steps are injected bad commands and must be masked before behaviour
cloning (the current VLA loaders do not). A faulted rollout that succeeds without
exercising its fault is rejected as fault_not_exercised. Keep these runs in their own
--output directory:

    python scripts/collect_random.py --workspace configs/workspace_v2_cable30.json \\
        --faults all --episodes 6 --record --headless --output data/openarm_v2_recovery
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
from controllers import faults as fault_lib
from data_pipeline.scene_bank import (collection_bank, evaluation_bank, as_reset_options,
                                      workspace_for_env, COLLECTION_SEED_BASE, EVALUATION_SEED_BASE,
                                      SEED_NAMESPACE_WIDTH,
                                      iter_collection_bank)
from data_pipeline.episodes import SCHEMA, save_episode, clean_info
from data_pipeline.quality import quality_check
from data_pipeline.replay import replay_episode
from envs.openarm_insert import OpenArmInsertEnv
from rollout_viewer import LiveViewer, VideoRecorder, ViewerClosed

MAX_STEPS = 6000
FAULT_MAX_STEPS = 9000   # 180 s: room for the recovery, still inside the env's 240 s tabletop limit
# Rows are drawn in blocks of this size; the validation split is the tail of each block.
BATCH = 64


def row_block(start_row, cursor, budget):
    """Block holding attempt `cursor`: (absolute block start row, rows in block, rows to skip).

    cursor and budget count attempts (relative); the returned start is the absolute bank row,
    for iter_collection_bank(n=n, start=abs_start). Blocks are aligned relative to start_row."""
    block = cursor - cursor % BATCH
    return start_row + block, min(BATCH, budget - block), cursor - block


def write_json(path, value):
    # Write then rename: a run killed mid-write leaves the previous file, never a truncated one.
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def make_visibility_probe(env):
    """Pilot-camera plug visibility probe, or None if data_pipeline/visibility.py is unavailable."""
    try:
        from data_pipeline.visibility import VisibilityProbe, PILOT_CAMERAS
    except ImportError:
        print("WARNING: data_pipeline.visibility unavailable; camera-occlusion gate disabled", flush=True)
        return None, ()
    return VisibilityProbe(env), tuple(PILOT_CAMERAS)


def rollout(env, row, *, record=False, detector_mode="noisy", probe_offset_y_m=0.0,
            viewer=None, recorder=None, faults=()):
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
    trace_qpos, trace_phase, trace_label = [env.data.qpos.copy()], [], []
    fault = fault_lib.draw_fault(row["seed"], faults) if faults else None
    expert = PickInsertExpert(env, rng=np.random.default_rng(row["seed"]),
                              detector_mode=detector_mode,
                              probe_offset_y_m=probe_offset_y_m, fault=fault)
    max_steps = FAULT_MAX_STEPS if fault else MAX_STEPS
    steps = 0
    # Quality bookkeeping (preflight and record alike) and the pilot-camera visibility gate.
    quality_steps, plug_z0 = [], float(env.data.qpos[env.plug_qadr + 2])
    probe, pilot_cams = make_visibility_probe(env)
    vis_rows, occluded, lift_seen = [], [], False

    def see(frame, phase_before):
        if probe is None:
            return
        seen = probe.plug()
        vis_rows.append([bool(seen[c]) for c in pilot_cams])
        if not any(vis_rows[-1]) and not lift_seen:
            occluded.append((frame, phase_before))

    see(0, "reset")
    while env.outcome is None and expert.failure is None and steps < max_steps:
        action = expert.action()
        phase = expert.phase
        lift_seen = lift_seen or phase == "lift"
        obs, reward, term, trunc, info = env.step(action)
        steps += 1
        quality_steps.append({"phase": phase, "label": expert.label,
                              "plug_finger_contacts": int(info.get("plug_finger_contacts", 0)),
                              "grasp_established": bool(info.get("grasp_established", False)),
                              "drops": int(info.get("drops", 0)),
                              "plug_z": float(env.data.qpos[env.plug_qadr + 2])})
        if record or not lift_seen:
            see(steps, phase)
        trace_qpos.append(env.data.qpos.copy())
        trace_phase.append(phase)
        trace_label.append(expert.label)
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
    if fault is not None and outcome == "success" and expert.fault_state != "done":
        # Succeeded without the mistake and its recovery happening: not a recovery demonstration.
        outcome = "fault_not_exercised"
    original_outcome = outcome
    first_lift = next((i for i, s in enumerate(quality_steps) if s["phase"] == "lift"), None)
    visibility = {"probe": probe is not None, "cameras": list(pilot_cams),
                  "frames_checked": first_lift + 1 if first_lift is not None else len(vis_rows),
                  "first_lift_step": first_lift, "occluded_frames": len(occluded),
                  "first_occluded_frame": occluded[0][0] if occluded else None,
                  "first_occluded_phase": occluded[0][1] if occluded else None}
    if outcome == "success" and occluded:
        outcome = "camera_occluded"
    quality = quality_check(quality_steps, original_outcome, start_z=plug_z0)
    if outcome == "success" and not quality["ok"]:
        outcome = quality["reasons"][0]
    summary = {**{k: v for k, v in row.items() if k != "sampler"},
               "outcome": outcome, "steps": steps,
               "original_outcome": original_outcome, "stage_outcomes": quality["stage_outcomes"],
               "quality": {"ok": quality["ok"], "reasons": quality["reasons"],
                           "min_lift_m": quality["min_lift_m"]},
               "visibility": visibility,
               "expert": expert.diagnostics(),
               "initial": start_info, "final": clean_info(info)}
    payload = (observations, requested, applied, qpos, qvel,
               rewards, terms, truncs, durations, phases, infos)
    trace = {"qpos": np.asarray(trace_qpos), "phase": np.asarray(trace_phase)}
    if record and probe is not None:
        trace["visibility"] = np.asarray(vis_rows, dtype=bool)
    if fault is not None:
        summary["fault"] = fault.to_json()
        summary["label_counts"] = fault_lib.label_counts(trace_label)
        trace["label"] = np.asarray(trace_label)
    return summary, payload, trace


def save_failure(directory, row_id, summary, trace, recorder=None, payload=None):
    directory.mkdir(parents=True, exist_ok=True)
    write_json(directory / (row_id + ".json"), summary)
    arrays = {"qpos": trace["qpos"], "phase": trace["phase"]}
    if "label" in trace:
        arrays["step_label"] = trace["label"]
    if payload is not None:
        arrays.update(action_requested=np.asarray(payload[1]), action_applied=np.asarray(payload[2]),
                      qvel=np.asarray(payload[4]))
    np.savez_compressed(directory / (row_id + ".npz"), **arrays)
    if recorder is not None:
        recorder.save(directory / row_id, metadata={"outcome": summary["outcome"],
                                                    "expert": summary["expert"]})


def verify_replay(env, path):
    """Strict replay of the episode just written, on the same env without rendering."""
    env.images = False
    try:
        result = replay_episode(env, path)
        return {"passed": True, **result}
    except (AssertionError, ValueError, RuntimeError) as error:
        return {"passed": False, "error": f"{type(error).__name__}: {error}"}


def attach_replay(path, replay_result):
    """Store the replay verdict in metadata.json (data_sha256 covers only episode.npz, so it stays valid)."""
    meta_path = path / "metadata.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["replay"] = replay_result
    write_json(meta_path, meta)
    return meta


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
            # cursor is block-aligned here (rows are consumed a full batch at a time), so skip is 0.
            abs_start, batch_n, _ = row_block(args.start_row, cursor, budget)
            batch = iter_collection_bank(n=batch_n,
                                         validation_fraction=args.validation_fraction,
                                         ws=ws, start=abs_start)
            for row in batch:
                if len(accepted) >= args.episodes:
                    break
                cursor += 1
                trace = None
                try:
                    result, _, trace = rollout(env, row, detector_mode=args.detector,
                                               viewer=viewer, recorder=recorder, faults=fault_kinds(args))
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
    write_json(out / "preflight_stats.json",
               {"requested": args.episodes, "accepted": len(accepted),
                "attempted": cursor, "attempt_budget": budget,
                "acceptance_rate": len(accepted) / max(cursor, 1),
                "failure_reasons": failure_reasons(rejected), "wall_seconds": round(time.time() - started, 1),
                "detector_mode": args.detector, "start_row": args.start_row,
                **fault_stats(args, accepted, rejected)})
    return accepted, cursor


def fault_kinds(args):
    """Injected fault kinds of a --faults run; () for ordinary runs (and args built without it)."""
    return tuple(getattr(args, "faults", None) or ())


def failure_reasons(rejected):
    reasons = {}
    for r in rejected:
        key = r["outcome"]
        if key.startswith("expert_"):
            key = f"{key} ({(r.get('expert') or {}).get('phase')})"
        reasons[key] = reasons.get(key, 0) + 1
    return reasons


def fault_stats(args, accepted, rejected):
    """--faults runs: acceptance and failure reasons per fault kind (empty for ordinary runs)."""
    if not fault_kinds(args):
        return {}
    per_kind = {}
    for kind in fault_kinds(args):
        ok = [r for r in accepted if (r.get("fault") or {}).get("kind") == kind]
        bad = [r for r in rejected if (r.get("fault") or {}).get("kind") == kind]
        per_kind[kind] = {"accepted": len(ok), "rejected": len(bad),
                          "acceptance_rate": len(ok) / max(len(ok) + len(bad), 1),
                          "failure_reasons": failure_reasons(bad)}
    return {"faults": list(fault_kinds(args)), "per_fault": per_kind}


def run_signature(args):
    """What decides which scenes a run draws; --resume refuses to continue under different values."""
    signature = {"workspace": args.workspace.as_posix() if args.workspace else None,
                 "detector": args.detector, "validation_fraction": args.validation_fraction,
                 "start_row": args.start_row}
    if fault_kinds(args):                   # only for --faults runs, so older progress.json still resume
        signature["faults"] = list(fault_kinds(args))
    return signature


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
                                             viewer=viewer, recorder=recorder, faults=fault_kinds(args))
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
        replay_result = None
        if result["outcome"] == "success":
            if recorder is not None and args.video == "all":
                recorder.save(out / "video" / row["id"], metadata={"outcome": "success"})
            labels = ({"labels": trace["label"].tolist(), "fault": result["fault"]}
                      if "label" in trace else {})
            extra = {"quality": result["quality"], "stage_outcomes": result["stage_outcomes"]}
            if "visibility" in trace:
                extra["visibility"] = trace["visibility"]
            meta = save_episode(out / row["id"], *payload, env.manifest(), **labels, **extra)
            if args.verify_replay:
                replay_result = verify_replay(env, out / row["id"])
                if replay_result["passed"]:
                    meta = attach_replay(out / row["id"], replay_result)
                else:
                    # Commands do not reproduce the physics: not usable data. Keep it for inspection.
                    result = {**result, "outcome": "replay_mismatch", "replay": replay_result}
                    dest = out / "failed_attempts" / row["id"]
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(out / row["id"]), str(dest))
                    write_json(out / "failed_attempts" / (row["id"] + ".json"), result)
        if result["outcome"] != "success":
            if result["outcome"] != "replay_mismatch":
                save_failure(out / "failed_attempts", row["id"], result, trace,
                             recorder=recorder, payload=payload)
            failures.append({"episode": row["id"], "outcome": result["outcome"],
                             "expert": result["expert"], "trace_matches_preflight": matches,
                             "stage_outcomes": result["stage_outcomes"],
                             **({"replay": replay_result} if replay_result else {})})
            print("FAILED (retained, not a BC label)", row["id"], result["outcome"], flush=True)
        else:
            records.append({"episode": row["id"], "seed": row["seed"], "split": row["split"],
                            "steps": meta["steps"], "data_sha256": meta["data_sha256"],
                            "retries": result["expert"]["retries"],
                            "regrasps": result["expert"]["regrasps"],
                            "detection_error": result["expert"]["detection_error"],
                            "trace_matches_preflight": matches,
                            "stage_outcomes": result["stage_outcomes"], "replay": replay_result,
                            **({"fault": result["fault"], "label_counts": result["label_counts"]}
                               if labels else {})})
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
            abs_start, block_n, skip = row_block(args.start_row, cursor, budget)
            rows = itertools.islice(
                iter_collection_bank(n=block_n,
                                     validation_fraction=args.validation_fraction,
                                     ws=ws, start=abs_start),
                skip, None)
            for row in rows:
                if len(records) >= args.episodes:
                    break
                print("preflight", row["id"], flush=True)
                remove_partial(row_artifacts(out / "failed_preflight", row["id"]))
                trace = None
                try:
                    result, _, trace = rollout(env, row, detector_mode=args.detector,
                                               viewer=viewer, recorder=recorder, faults=fault_kinds(args))
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
    write_json(out / "preflight_stats.json",
               {"requested": args.episodes, "accepted": len(accepted), "recorded": len(records),
                "attempted": state["cursor"], "attempt_budget": budget,
                "acceptance_rate": len(accepted) / max(state["cursor"], 1),
                "failure_reasons": failure_reasons(rejected),
                "wall_seconds_this_session": round(time.time() - started, 1),
                "detector_mode": args.detector, "start_row": args.start_row,
                **fault_stats(args, accepted, rejected)})
    save_progress()
    return records, failures


def build_manifest(args, out, env, records, failures, complete):
    manifest = _base_manifest(args, out, env, records, failures, complete)
    if fault_kinds(args):
        manifest.update(
            purpose="randomized table-top pick-and-insert with one injected fault per scene, then recovery",
            collection_mode="per scene: preflight, then record on success (resumable); "
                            "injected-fault recovery demonstrations; step_label marks fault steps",
            fault_injection={
                "kinds": list(fault_kinds(args)), "module": "controllers/faults.py",
                "module_sha256": sha256(fault_lib.__file__),
                "ranges": {"premature_close_descend_frac": list(fault_lib.PREMATURE_CLOSE_FRAC),
                           "wrong_grasp_directions": {k: list(v) for k, v in
                                                      fault_lib.WRONG_GRASP_DIRECTIONS.items()},
                           "wrong_grasp_distance_m": list(fault_lib.WRONG_GRASP_DISTANCE_M),
                           "misaligned_insert_offset_m": list(fault_lib.MISALIGN_OFFSET_M)},
                "labels": list(fault_lib.LABELS)},
            training_note="mask step_label == 'fault' before BC; current loaders do not")
    return manifest


# Code that produces the demonstrations (beyond the 11 files env.manifest() hashes): a release refuses
# to mix runs whose hashes here differ.
COLLECTION_SOURCES = ("controllers/pick_insert_expert.py", "controllers/place_insert.py",
                      "controllers/faults.py", "data_pipeline/scene_bank.py",
                      "data_pipeline/quality.py", "data_pipeline/replay.py",
                      "data_pipeline/visibility.py", "data_pipeline/episodes.py",
                      "scripts/collect_random.py", "configs/openarm_v1.json")


def collection_source_hashes():
    root = Path(__file__).resolve().parents[1]
    return {p: sha256(root / p) if (root / p).exists() else None for p in COLLECTION_SOURCES}


def _base_manifest(args, out, env, records, failures, complete):
    return {"schema": SCHEMA,
            "episodes": records,
            "failed_recordings": failures,
            "instruction": env.config["instruction"],
            "purpose": "randomized table-top pick-and-insert; plug and socket poses sampled",
            "task_stages": ["survey", "grasp", "transport", "insert"],
            "detector_mode": args.detector,
            "start_row": args.start_row,
            "collection_mode": "per scene: preflight, then record on success (resumable)",
            "robot_manifest": env.manifest(),
            "complete": complete,
            "collection_script_sha256": sha256(__file__),
            "collection_source_hashes": collection_source_hashes(),
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
    p.add_argument("--start-row", type=int, default=0,
                   help="first bank row (seed offset) drawn; --start-row 5000 draws a fresh set "
                        "of scenes, the default 0 reproduces earlier runs")
    p.add_argument("--detector", choices=("noisy", "privileged"), default="noisy")
    p.add_argument("--record", action="store_true", help="Record images for accepted scenes")
    p.add_argument("--resume", action="store_true",
                   help="Continue a stopped run in the same --output (a scene that was being "
                        "recorded is recorded again; one in preflight is preflighted again)")
    p.add_argument("--no-verify-replay", dest="verify_replay", action="store_false",
                   help="skip the strict replay of each recorded episode (replay doubles physics time)")
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
                   help="size of the reserved evaluation bank written first (default 0 = none; "
                        "release builds use 100; a smaller bank is a prefix of the full one)")
    p.add_argument("--faults", default=None,
                   help="recovery demonstrations: one injected fault per scene, then recovery. "
                        "'all' or a comma list of " + ",".join(fault_lib.FAULT_KINDS)
                        + " (kind cycles with the seed). Use a separate --output.")
    args = p.parse_args()
    if args.start_row < 0:
        p.error("--start-row must be >= 0")
    try:
        args.faults = fault_lib.parse_kinds(args.faults)
    except ValueError as error:
        p.error(str(error))
    budget = int(args.episodes * args.attempt_ratio)
    if args.start_row + budget > SEED_NAMESPACE_WIDTH:
        raise SystemExit(f"--start-row {args.start_row} + attempt budget {budget} exceeds the "
                         f"collection seed namespace ({SEED_NAMESPACE_WIDTH} rows); rows would "
                         "collide with the evaluation seeds. Lower --start-row or --episodes.")

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