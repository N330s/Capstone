"""Summarize true demonstration phases and save a two-camera sanity contact sheet."""
import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((args.dataset / "manifest.json").read_text())
    counts, episodes = Counter(), []
    for record in manifest["episodes"]:
        with np.load(args.dataset / record["episode"] / "episode.npz", allow_pickle=False) as arrays:
            phases, frequency = np.unique(arrays["phase"], return_counts=True)
            current = dict(zip(phases.tolist(), frequency.tolist()))
            counts.update(current)
            episodes.append({"episode": record["episode"], "split": record["split"], "phases": current})
    first = manifest["episodes"][0]
    with np.load(args.dataset / first["episode"] / "episode.npz", allow_pickle=False) as arrays:
        phases = arrays["phase"]
        selected = []
        for phase in ("survey", "transit_pick", "descend", "close", "lift", "transit_place", "align", "insert", "hold"):
            indices = np.flatnonzero(phases == phase)
            if len(indices):
                selected.append((phase, int(indices[len(indices) // 2])))
        columns = min(3, len(selected))
        if not columns:
            raise ValueError("No recognized task stages")
        rows = (len(selected) + columns - 1) // columns
        sheet = Image.new("RGB", (640 * columns, 280 * rows), "white")
        draw = ImageDraw.Draw(sheet)
        for n, (phase, index) in enumerate(selected):
            x, y = (n % columns) * 640, (n // columns) * 280
            draw.text((x + 5, y + 5), f"{phase} | command {index} | scene / wrist", fill="black")
            for k, camera in enumerate(("image_scene_rgb", "image_wrist_rgb")):
                sheet.paste(Image.fromarray(arrays[camera][index]).resize((320, 240)), (x + k * 320, y + 30))
        sheet.save(args.output / "camera_stage_sanity.png")
    report = {"episodes": episodes, "phase_command_counts": dict(counts),
              "stage_presence_all_episodes": {p: all(p in e["phases"] for e in episodes)
                                            for p in ("close", "lift", "insert", "hold")},
              "contact_sheet_episode": first["episode"],
              "limits": "Collector phase labels and one contact sheet, not independent physical-success verification."}
    (args.output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "episodes"}, indent=2), flush=True)
    if not all(report["stage_presence_all_episodes"].values()):
        raise ValueError("Missing grasp/lift/insert/hold stages in some episodes")


if __name__ == "__main__":
    main()
