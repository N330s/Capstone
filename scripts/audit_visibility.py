"""Audit plug / socket camera visibility over bank rows (physics-only expert rollouts, no recording).

Pickup frames are every frame from obs 0 until the first ``lift`` phase (fault and recovery pickup
phases included); insertion frames are the phases after it. Every --stride steps the audit records
whether the plug is visible from the pilot cameras (scene_rgb, wrist_rgb) and from every camera of
the workspace (adds chest_rgb), and whether scene_rgb sees the socket entrance.

    python scripts/audit_visibility.py --rows 4 --start-row 6100 --stride 10 --output results/vis_audit.json
    python scripts/audit_visibility.py --reset-only 100 --output results/vis_reset.json
    python scripts/audit_visibility.py --kinematic 2400 --output results/vis_kin.json
    python scripts/audit_visibility.py --rows 6 --faults all --output results/vis_faults.json

Each row also reports the start posture of the right arm, the sampler's arm_offset_rad and posture
diagnostics (fallback_zero) and the unwanted robot contact at reset. The JSON is rewritten after every
row (write then rename), so an interrupted run keeps its finished rows. ``--kinematic N`` draws N raw
sampler placements (plug, socket, jitter, +/- arm offsets), places them without any physics and reports
the reset-frame plug visibility and where the misses lie. ``--output`` must not already exist. Does not use scripts/collect_random.py.
"""
import argparse
import json
import sys
import os
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np

from controllers import faults as fault_lib
from controllers.pick_insert_expert import PickInsertExpert
from data_pipeline import scene_bank as sb
from data_pipeline.scene_bank import as_reset_options, iter_collection_bank, workspace_for_env
from data_pipeline.visibility import PILOT_CAMERAS, VisibilityProbe, pickup_coverage
from envs.openarm_insert import OpenArmInsertEnv

MAX_STEPS = 6000
FAULT_MAX_STEPS = 9000
INSERTION_PHASES = ("transit_place", "approach", "align", "insert", "hold", "withdraw")
OVERHEAD = "scene_rgb"


def all_cameras(env):
    cams = list(PILOT_CAMERAS)
    if "chest_rgbd" in env.workspace.get("cameras", {}):
        cams.append("chest_rgb")
    return tuple(cams)


def row_info(row):
    """Arm offset and posture diagnostics the sampler stored with the bank row."""
    posture = (row.get("sampler") or {}).get("posture") or {}
    return {"arm_offset_rad": row["options"].get("arm_offset_rad"),
            "posture_fallback_zero": posture.get("fallback_zero"),
            "posture_tries": posture.get("tries")}


def write_json(path, payload):
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    os.replace(tmp, path)


def kinematic(env, ws, n, seed):
    """Raw sampler draws placed without physics; plug visibility at the reset pose (kin.py)."""
    import mujoco
    pilot = VisibilityProbe(env, PILOT_CAMERAS)
    env.reset(seed=0)
    rng = np.random.default_rng(seed)
    plug_x, plug_y = ws.plug_x_range or ws.x_range, ws.plug_y_range or ws.y_range
    home = np.asarray(env.config["home_arm_rad"], float)
    rows = []
    for i in range(n):
        plug = np.array([rng.uniform(*plug_x), rng.uniform(*plug_y), ws.table_height_m + ws.plug_half_height_m])
        sock, syaw, stilt = sb.sample_socket(rng, ws)
        pyaw = float(rng.uniform(*ws.plug_yaw_range_deg))
        vis = sb._visual(rng, ws)
        off = rng.uniform(-ws.arm_offset_max_rad, ws.arm_offset_max_rad, 7)
        geom_ok = sb._check(plug, sock, sb._yaw_axis(syaw, stilt), ws) is None
        env._restore_scene()
        env.data.qpos[env.qa["right"]] = home + off
        env._place_socket(sock, syaw, stilt)
        env.place_plug(plug, env._quat(env._yaw_tilt_matrix(pyaw, 0.)))
        if vis:
            env.apply_visual(vis)
        mujoco.mj_forward(env.model, env.data)
        f = pilot.plug()
        rows.append({"i": i, "geom_ok": geom_ok, "plug_xy": [round(float(v), 4) for v in plug[:2]], "plug_yaw_deg": round(pyaw, 2),
                     "arm_offset_rad": [round(float(v), 4) for v in off], "scene": f["scene_rgb"], "wrist": f["wrist_rgb"],
                     "any": any(f.values())})
    summary = {"draws": n, "seed": seed}
    for name, sel in (("all_raw", rows), ("geom_ok", [r for r in rows if r["geom_ok"]])):
        k = max(len(sel), 1)
        summary[name] = {"n": len(sel), "pilot_visible": sum(r["any"] for r in sel) / k,
                         "scene_rgb": sum(r["scene"] for r in sel) / k, "wrist_rgb": sum(r["wrist"] for r in sel) / k}
    bad = [r for r in rows if not r["any"]]
    summary["uncovered"] = len(bad)
    if bad:
        xy = np.array([r["plug_xy"] for r in bad])
        summary["uncovered_plug_xy_min"] = xy.min(0).round(4).tolist()
        summary["uncovered_plug_xy_max"] = xy.max(0).round(4).tolist()
    return summary, rows


def start_info(env, info):
    return {"right_arm_start_rad": [round(float(v), 5) for v in env.data.qpos[env.qa["right"]]],
            "robot_unwanted_contact_n": float(info.get("robot_unwanted_contact_n", 0.0))}


def run_row(env, pilot, full, row, faults, stride):
    obs, info = env.reset(seed=row["seed"], options=as_reset_options(row["options"]))
    rec = {"id": row["id"], "seed": row["seed"], **start_info(env, info)}
    fault = fault_lib.draw_fault(row["seed"], faults) if faults else None
    expert = PickInsertExpert(env, rng=np.random.default_rng(row["seed"]), detector_mode="noisy",
                              probe_offset_y_m=0.0, fault=fault)
    max_steps = FAULT_MAX_STEPS if fault else MAX_STEPS
    pick, ins = defaultdict(list), defaultdict(list)
    pick_overhead, pick_all = defaultdict(list), defaultdict(list)
    lifted, steps = False, 0

    def sample(phase):
        nonlocal lifted
        lifted = lifted or phase == "lift"
        if not lifted:
            f = pilot.plug()
            pick[phase].append(any(f.values()))
            pick_overhead[phase].append(f[OVERHEAD])
            pick_all[phase].append(any(full.plug().values()))
        elif phase in INSERTION_PHASES:
            ins[phase].append(pilot.socket()[OVERHEAD])

    sample(expert.phase)                       # obs 0
    while env.outcome is None and expert.failure is None and steps < max_steps:
        action = expert.action()
        phase = expert.phase
        env.step(action)
        steps += 1
        if steps % stride == 0:
            sample(phase)
    outcome = env.outcome
    if outcome is None:
        outcome = f"expert_{expert.failure}" if expert.failure else "step_budget_exhausted"
    if fault is not None and outcome == "success" and expert.fault_state != "done":
        outcome = "fault_not_exercised"
    rec.update(outcome=outcome, steps=steps, fault=fault.to_json() if fault else None,
               pickup_pilot=dict(pick), pickup_overhead=dict(pick_overhead), pickup_all=dict(pick_all),
               socket_overhead=dict(ins))
    return rec


def merge(rows, key):
    out = defaultdict(list)
    for r in rows:
        for phase, flags in r[key].items():
            out[phase].extend(flags)
    return out


def summarize(rows):
    done = [r for r in rows if "outcome" in r]
    s = {"rows": len(rows)}
    for key in ("pickup_pilot", "pickup_overhead", "pickup_all", "socket_overhead"):
        s[key] = pickup_coverage(merge(done, key))
    for key in ("pickup_pilot", "pickup_all"):
        s["scenes_with_uncovered_pickup_frame_" + key.split("_")[1]] = sum(
            1 for r in done if not all(all(f) for f in r[key].values()))
    outcomes = defaultdict(int)
    for r in done:
        outcomes[r["outcome"]] += 1
    s["outcomes"] = dict(outcomes)
    s["success_rate"] = (outcomes["success"] / len(done)) if done else None
    s["unwanted_contact_resets"] = [r["id"] for r in rows if r["robot_unwanted_contact_n"] > 0]
    return s


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--workspace", default="configs/workspace_v2_cable30.json")
    p.add_argument("--rows", type=int, default=0, help="expert rollouts on this many bank rows")
    p.add_argument("--start-row", type=int, default=6100)
    p.add_argument("--reset-only", type=int, default=0, help="resets only: plug visibility at obs 0")
    p.add_argument("--kinematic", type=int, default=0, help="N raw sampler draws, no physics (reset-frame visibility)")
    p.add_argument("--kin-seed", type=int, default=12345)
    p.add_argument("--faults", default=None, help="'all' or comma list of fault kinds")
    p.add_argument("--stride", type=int, default=10, help="record every K steps")
    p.add_argument("--output", required=True)
    args = p.parse_args()
    if not (args.rows or args.reset_only or args.kinematic):
        p.error("give --rows N, --reset-only N or --kinematic N")
    out = Path(args.output)
    if out.exists():
        raise SystemExit(f"{out} already exists; choose a new --output")
    out.parent.mkdir(parents=True, exist_ok=True)
    faults = fault_lib.parse_kinds(args.faults)

    env = OpenArmInsertEnv(images=False, workspace=args.workspace)
    ws = workspace_for_env(env)
    if args.kinematic:
        summary, rows = kinematic(env, ws, args.kinematic, args.kin_seed)
        summary["workspace"] = args.workspace
        write_json(out, {"summary": summary, "rows": rows})
        print(json.dumps(summary, indent=1))
        return
    pilot, full = VisibilityProbe(env, PILOT_CAMERAS), VisibilityProbe(env, all_cameras(env))
    n = args.reset_only or args.rows
    rows, t0 = [], time.time()
    for row in iter_collection_bank(n=n, validation_fraction=0.1, ws=ws, start=args.start_row):
        try:
            if args.reset_only:
                _, info = env.reset(seed=row["seed"], options=as_reset_options(row["options"]))
                rec = {"id": row["id"], "seed": row["seed"], **start_info(env, info),
                       "plug_visible": full.plug(), "pilot_any": pilot.any_plug()}
            else:
                rec = run_row(env, pilot, full, row, faults, args.stride)
        except Exception as e:                  # a scene that cannot reset is reported, not fatal
            rec = {"id": row["id"], "seed": row["seed"], "error": f"{type(e).__name__}: {e}",
                   "robot_unwanted_contact_n": 0.0}
        rec.update(row_info(row))
        rows.append(rec)
        write_json(out, {"rows": rows, "complete": False, "workspace": args.workspace, "stride": args.stride})
        print(f"{rec['id']}: " + (rec.get("outcome") or rec.get("error") or
                                  f"pilot_any={rec.get('pilot_any')}") + f"  ({time.time() - t0:.0f}s)", flush=True)
    if args.reset_only:
        ok = [r for r in rows if "pilot_any" in r]
        summary = {"rows": len(rows), "pilot_plug_visible_at_reset": sum(r["pilot_any"] for r in ok) / max(len(ok), 1),
                   "per_camera": {c: sum(r["plug_visible"][c] for r in ok) / max(len(ok), 1)
                                  for c in (ok[0]["plug_visible"] if ok else {})},
                   "unwanted_contact_resets": [r["id"] for r in rows if r["robot_unwanted_contact_n"] > 0]}
        post = np.array([r["right_arm_start_rad"] for r in ok]) if ok else np.zeros((0, 7))
        if len(post):
            summary["right_arm_start_std_rad"] = [round(float(v), 5) for v in post.std(0)]
    else:
        summary = summarize(rows)
    summary["workspace"] = args.workspace
    summary["stride"] = args.stride
    write_json(out, {"summary": summary, "rows": rows, "complete": True})
    print(json.dumps(summary, indent=1)[:4000])


if __name__ == "__main__":
    main()
