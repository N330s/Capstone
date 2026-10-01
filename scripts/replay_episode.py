"""Replay a recorded attempt so you can see what went wrong, after the fact.

    python scripts/replay_episode.py data/openarm_v2_random/failed_preflight/train_00001.npz
    python scripts/replay_episode.py data/.../train_00003.npz --speed .25 --video out.mp4

Plays back stored qpos frames as kinematics only: no physics is re-run, so what
you see is exactly the trajectory that was recorded, not a re-simulation that
might diverge. The failed_attempts npz written by the collector already contains
qpos, qvel and both action streams.
Bug fixed here: the socket is a static body positioned via model.body_pos /
model.body_quat during reset(), not through qpos. A fresh env never calls
reset(seed=..., options=...), so the socket used to render at whatever pose
ships in the XML -- which is what you saw in the screenshot, even though the
arm and plug (both qpos-driven) played back correctly. This now reconstructs
the original options from the summary json (or regenerates them from the seed
via the scene sampler, since sampling is a pure function of the seed) and
replays reset() once before stepping through the recorded qpos frames.
"""
import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import mujoco

from envs.openarm_insert import OpenArmInsertEnv


def resolve_reset_call(summary):
    """Return (seed, options) to replay with, regenerating options from the
    seed if the summary didn't carry them (older recordings, hand-edited
    files). Returns (None, None) if there is nothing to go on."""
    seed = summary.get("seed")
    options = summary.get("options")
    if options is not None:
        return seed, options
    if seed is None:
        return None, None
    try:
        from data_pipeline.scene_bank import sample_scene, as_reset_options
    except ImportError as error:
        print(f"no options in the summary and can't import the scene sampler to "
              f"regenerate them ({error}); static bodies will render at XML defaults")
        return seed, None
    options, _ = sample_scene(seed)
    return seed, as_reset_options(options)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("npz", type=Path, help="failed_attempts/<id>.npz or an episode npz")
    p.add_argument("--speed", type=float, default=1.0, help="playback rate, 0 for as fast as possible")
    p.add_argument("--video", type=Path, default=None, help="write a video instead of opening a window")
    p.add_argument("--camera", default="scene_rgb")
    p.add_argument("--loop", action="store_true")
    args = p.parse_args()

    payload = np.load(args.npz)
    if "qpos" not in payload:
        raise SystemExit(f"{args.npz} has no qpos array; keys are {list(payload.keys())}")
    qpos = payload["qpos"]
    summary_path = args.npz.with_suffix(".json")
    summary = {}
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        print(f"outcome: {summary.get('outcome')}  expert: {summary.get('expert', {}).get('failure')}")
        for entry in summary.get("expert", {}).get("phase_log", []):
            print(f"  {entry['phase']:<14} {entry['duration_s']:>6.2f} s")
    else:
        print(f"no sidecar {summary_path.name}; socket and other static bodies "
              f"will render at their XML default pose, not the recorded one")

    env = OpenArmInsertEnv(images=False)
    try:
        seed, options = resolve_reset_call(summary)
        if seed is not None and options is not None:
            try:
                env.reset(seed=seed, options=options)
            except (ValueError, RuntimeError) as error:
                print(f"could not re-apply reset(seed={seed}, options=...): {error}\n"
                      f"continuing anyway; static bodies (the socket) may be wrong")
        dt = env.dt
        if args.video:
            try:
                from rollout_viewer import VideoRecorder
            except ImportError:
                from tools.rollout_viewer import VideoRecorder
            recorder = VideoRecorder(env, camera=args.camera, stride=1)
            for frame in qpos:
                env.data.qpos[:] = frame
                mujoco.mj_forward(env.model, env.data)
                recorder.capture()
            written = recorder.save(args.video)
            recorder.close()
            print("wrote", written)
            return
        import mujoco.viewer
        with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
            viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = 1.4, 135.0, -25.0
            while viewer.is_running():
                for frame in qpos:
                    if not viewer.is_running():
                        break
                    env.data.qpos[:] = frame
                    mujoco.mj_forward(env.model, env.data)
                    viewer.sync()
                    if args.speed > 0:
                        time.sleep(dt / args.speed)
                if not args.loop:
                    break
    finally:
        env.close()


if __name__ == "__main__":
    main()