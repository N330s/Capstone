"""Sample N workspace_v2 scenes with the current sampler and run the tabletop expert on each (with video).

usage: run_v2.py <n_scenes> <out_dir> [detector]
"""
import sys, json, time
from pathlib import Path
sys.path.insert(0, '.'); sys.path.insert(0, 'scripts')
from envs.openarm_insert import OpenArmInsertEnv
import data_pipeline.scene_bank as sb
import collect_random as cr
from rollout_viewer import VideoRecorder

n, out = int(sys.argv[1]), Path(sys.argv[2])
det = sys.argv[3] if len(sys.argv) > 3 else "privileged"
out.mkdir(parents=True, exist_ok=True)
env = OpenArmInsertEnv(images=False, workspace="configs/workspace_v2.json")
ws = sb.workspace_for_env(env)
recorder = VideoRecorder(env, camera="scene_rgb", stride=2)
seed = sb.COLLECTION_SEED_BASE
done = 0
while done < n:
    t = time.time()
    options, diag = sb.sample_scene(seed, ws)
    row = {"id": f"v2_{seed - sb.COLLECTION_SEED_BASE:05d}", "seed": seed, "split": "train", "options": options}
    (out / f"{row['id']}.json").write_text(json.dumps({**row, "sampler": diag}, indent=2))
    ts = time.time() - t
    summary, _, _ = cr.rollout(env, row, detector_mode=det, recorder=recorder)
    e = summary.get("expert", {})
    pl = e.get("place", {}) or {}
    st = pl.get("stats") or {}
    written = recorder.save(out / f"{row['id']}_{summary.get('outcome')}")
    (out / f"{row['id']}_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(row["id"], summary.get("outcome"), "|", e.get("failure_detail") or "",
          "| depth", round(st.get("final_depth_m", 0) * 1e3, 2), "peak_wall", round(st.get("peak_socket_force_n", 0), 2),
          "| sample", round(ts), "s draws", diag["draws"], "| total", round(time.time() - t), "s |", written, flush=True)
    seed += 1
    done += 1
recorder.close()
