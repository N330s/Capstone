"""Create camera contact sheets and trajectory summaries from recorded data."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from PIL import Image, ImageDraw


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("data/openarm_v1_varied10"))
    parser.add_argument("--output", type=Path, default=Path("results/varied10_inspection"))
    args = parser.parse_args()
    manifest = json.loads((args.dataset/"manifest.json").read_text())
    if not manifest.get("complete"):
        raise ValueError("Collection is incomplete")
    args.output.mkdir(parents=True, exist_ok=False)
    summary = []
    for camera in ("scene_rgb", "wrist_rgb"):
        sheet = Image.new("RGB", (960, 264*len(manifest["episodes"])), "white")
        draw = ImageDraw.Draw(sheet)
        for row, record in enumerate(manifest["episodes"]):
            path = args.dataset/record["episode"]
            infos = json.loads((path/"diagnostics.json").read_text())
            with np.load(path/"episode.npz", allow_pickle=False) as data:
                contact = next((i for i,v in enumerate(infos) if v["contact_force_n"]>.05), len(infos)//2)
                indices = (0, contact, len(infos)-1)
                frames = data["image_"+camera]
                for col, (label,index) in enumerate(zip(("start","contact","end"),indices)):
                    sheet.paste(Image.fromarray(frames[index]), (320*col,264*row+24))
                    draw.text((320*col+4,264*row+4), f"{record['episode']} {label} frame {index}", fill="black")
                if camera == "wrist_rgb":
                    # Preserve native pixels; nearest-neighbour enlargement is for inspection only.
                    Image.fromarray(frames[contact]).resize((960,720),Image.Resampling.NEAREST).save(
                        args.output/f"{record['episode']}_wrist_contact_3x.png")
                    summary.append({"episode":record["episode"],"steps":record["steps"],"split":record["split"],
                        "retries":record["retries"],"duration_s":float(data["executed_dt_s"].sum()),
                        "initial_offset_y_m":infos[0]["offset_y_m"],"initial_offset_z_m":infos[0]["offset_z_m"],
                        "initial_orientation_error_deg":infos[0]["orientation_error_deg"],
                        "peak_force_n":max(v["peak_contact_force_n"] for v in infos[1:]),
                        "peak_grasp_slip_m":max(v["grasp_slip_m"] for v in infos),
                        "phases":np.unique(data["phase"]).tolist()})
        sheet.save(args.output/f"{camera}_sheet.png")
    (args.output/"summary.json").write_text(json.dumps({"episodes":summary,
        "total_transitions":sum(r["steps"] for r in summary),
        "recovery_episodes":sum(r["retries"]>0 for r in summary),
        "evaluation_bank":"100 reserved conditions; not executed"},indent=2))


if __name__=="__main__":
    main()
