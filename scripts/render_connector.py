"""Render the actual MuJoCo geometry and optional insertion GIF (no GUI needed)."""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import mujoco
from PIL import Image
from connector.simulation import ConnectorSimulation, ROOT, VERSION, Pose


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=ROOT / "results" / VERSION / "preview")
    p.add_argument("--offset-y-mm", type=float, default=0)
    p.add_argument("--animate", action="store_true")
    args = p.parse_args()
    sim = ConnectorSimulation()
    sim.reset(Pose(offset_y_mm=args.offset_y_mm))
    args.output.mkdir(parents=True, exist_ok=True)
    opt = mujoco.MjvOption()
    opt.sitegroup[3] = 0
    with mujoco.Renderer(sim.model, height=720, width=960) as renderer:
        renderer.update_scene(sim.data, camera="overview", scene_option=opt)
        Image.fromarray(renderer.render()).save(args.output / "scene.png")
        if args.animate:
            frames = []
            next_frame = 0.0
            while sim.outcome is None:
                sim.step()
                if sim.data.time >= next_frame or sim.outcome:
                    renderer.update_scene(sim.data, camera="overview", scene_option=opt)
                    frames.append(Image.fromarray(renderer.render()).resize((640, 480)))
                    next_frame += 1 / 20
            frames[0].save(args.output / "insertion.gif", save_all=True,
                           append_images=frames[1:], duration=50, loop=0)
            print(sim.outcome)
    print(args.output)


if __name__ == "__main__":
    main()

