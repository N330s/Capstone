"""Robot-free calibration of the socket spring leaves (insertion resistance for one plug type).

Composes the lead-in connector named by the workspace (``--workspace``, legacy two-blade by default;
``--plug-type`` overrides the plug) with its spring leaves, drives the plug with a mocap weld at
constant speed in and back out, and records the force breakdown from ``ConnectorMetrics`` against
depth. This is a calibration harness only: the task environment never welds the plug. Writes
trace.csv, report.json and force_depth.png to --output.
"""
import argparse
import csv
import json
import sys
from pathlib import Path
import xml.etree.ElementTree as ET
import mujoco
import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from connector import catalog  # noqa: E402
from connector.geometry import connector_tree  # noqa: E402
from connector.simulation import CONFIG_PATH, ConnectorMetrics  # noqa: E402
from envs import workspace as wsp  # noqa: E402


def apply_overrides(ws, sets):
    for item in sets:
        path, value = item.split("=", 1)
        node = ws
        keys = path.split(".")
        for key in keys[:-1]:
            node = node[key]
        node[keys[-1]] = json.loads(value)
    wsp.validate(ws)
    return ws


def harness_xml(ws, spec, start_x):
    tree = connector_tree(True, spec=spec)
    tree.set("model", "leaf_calibration")
    tree.find("compiler").set("angle", "radian")
    # Match the robot scene's contact solver options (envs/scene.py), not the holder benchmark's.
    tree.find("option").set("cone", "elliptic")
    tree.find("option").set("impratio", "100")
    world = tree.find("worldbody")
    plug = world.find("./body[@name='plug']")
    visual = plug.find("./geom[@name='strain_relief_visual']")
    if visual is not None and "euler" in visual.attrib:   # legacy asset only
        visual.attrib.pop("euler")
        visual.set("quat", ".7071067811865476 0 .7071067811865476 0")
    plug.set("pos", wsp.vec((start_x, 0, .10)))
    socket = world.find("./body[@name='socket']")
    asset = tree.find("asset")
    wsp.add_retention_leaves(socket, asset, ws, wsp.leaf_openings(spec, ws), tree)
    driver = ET.SubElement(world, "body", name="driver", mocap="true", pos=wsp.vec((start_x, 0, .10)))
    ET.SubElement(driver, "site", name="driver_site", size="0.002")
    equality = ET.SubElement(tree, "equality")
    ET.SubElement(equality, "weld", name="driver_weld", body1="driver", body2="plug",
                  solref="0.002 1")
    return ET.tostring(tree, encoding="unicode")


def plot(rows, path, target, depth_range_mm):
    width, height = 900, 500
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    depth = np.array([r["depth_mm"] for r in rows])
    fx = np.array([r["socket_force_x_n"] for r in rows])
    left, right, top, bottom = 70, width - 20, 20, height - 50
    dmin, dmax = depth_range_mm
    fmax = max(2.0, float(np.abs(fx).max()) * 1.1, target * 1.3)

    def px(d, force):
        return (left + (d - dmin) / (dmax - dmin) * (right - left),
                bottom - (force + fmax) / (2 * fmax) * (bottom - top))
    draw.rectangle([left, top, right, bottom], outline="black")
    draw.line([px(dmin, 0), px(dmax, 0)], fill="gray")
    for sign in (1, -1):
        draw.line([px(dmin, sign * target), px(dmax, sign * target)], fill="green")
    for d in range(int(np.ceil(dmin / 5) * 5), int(dmax) + 1, 5):
        x, _ = px(d, 0)
        draw.line([(x, bottom), (x, bottom + 5)], fill="black")
        draw.text((x - 8, bottom + 8), f"{d}", fill="black")
    for force in np.linspace(-fmax, fmax, 7):
        _, y = px(dmin, force)
        draw.text((5, y - 6), f"{force:6.1f}", fill="black")
    for color, phase in (("red", "insert"), ("blue", "withdraw")):
        pts = [px(d, f) for d, f, r in zip(depth, fx, rows) if r["phase"] == phase]
        if len(pts) > 1:
            draw.line(pts, fill=color, width=2)
    draw.text((left + 10, top + 5), "axial force on plug along socket +X (N) vs shortest-element tip depth (mm); "
              "red = insert, blue = withdraw, green = +/- target", fill="black")
    img.save(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, default=None)
    parser.add_argument("--plug-type", choices=catalog.plug_names(), default=None,
                        help="override the workspace's connector.plug (sugar for --set connector.plug=...)")
    parser.add_argument("--set", action="append", default=[], metavar="KEY.PATH=JSON",
                        help="override a workspace value, e.g. socket_leaves.armature_kg=0.5")
    parser.add_argument("--speed-mm-s", type=float, default=5.0)
    parser.add_argument("--hold-s", type=float, default=0.5)
    args = parser.parse_args()
    ws = apply_overrides(wsp.load_workspace(args.workspace), args.set)
    if args.plug_type:
        if "connector" not in ws:
            raise SystemExit("--plug-type needs a workspace with a connector section (e.g. configs/workspace_v2.json)")
        wsp.apply_connector(ws, plug=args.plug_type)
    args.output.mkdir(parents=True, exist_ok=False)
    lv = ws["socket_leaves"]
    spec = catalog.from_workspace(ws)
    derived = spec.derived()
    start_x = derived["preinsert_x_m"]
    model = mujoco.MjModel.from_xml_string(harness_xml(ws, spec, start_x))
    data = mujoco.MjData(model)
    config = json.loads(CONFIG_PATH.read_text())
    config["leaf_penetration_limit_m"] = lv["leaf_penetration_limit_m"]
    leaf_names = wsp.leaf_names_in_model(model)
    metrics = ConnectorMetrics()
    metrics.bind(model, data, config, (), leaf_names, spec=spec)
    mujoco.mj_forward(model, data)
    leaf_joints = [model.joint(f"{n}_slide").id for n in leaf_names]
    leaf_columns = [n.replace("socket_leaf_", "leaf_") + "_travel_mm" for n in leaf_names]
    dt = model.opt.timestep
    speed = args.speed_mm_s / 1000
    travel = -start_x + 0.00003  # seat 30 um past the face gap zero like the task probe
    phases = [("settle", 0.5, 0.0), ("insert", travel / speed, speed), ("hold", args.hold_s, 0.0),
              ("withdraw", travel / speed, -speed), ("rest", 0.3, 0.0)]
    rows = []
    x = start_x
    tick = 0
    for phase, duration, velocity in phases:
        for _ in range(round(duration / dt)):
            x += velocity * dt
            data.mocap_pos[0] = [x, 0, .10]
            mujoco.mj_step(model, data)
            tick += 1
            if tick % 2 == 0:  # 1 kHz log
                mujoco.mj_forward(model, data)
                info = metrics.diagnostics()
                rows.append({"time_s": round(float(data.time), 6), "phase": phase,
                             "driver_x_m": x, "depth_mm": 1000 * info["insertion_depth_m"],
                             "socket_force_x_n": info["socket_force_x_n"],
                             "socket_lateral_force_n": info["socket_lateral_force_n"],
                             "socket_torque_nm": info["socket_torque_nm"],
                             "leaf_normal_n": info["leaf_normal_n"],
                             "leaf_friction_n": info["leaf_friction_n"],
                             "leaf_penetration_mm": 1000 * info["leaf_penetration_m"],
                             "wall_contact_force_n": info["contact_force_n"],
                             "wall_penetration_mm": 1000 * info["max_penetration_m"],
                             **{column: 1000 * float(data.qpos[model.jnt_qposadr[joint]])
                                for column, joint in zip(leaf_columns, leaf_joints)},
                             "valid_pose": info["valid_pose"]})
    with (args.output / "trace.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    def window(phase, lo, hi):
        return [r for r in rows if r["phase"] == phase and lo <= r["depth_mm"] <= hi]
    flat_start = 1000 * (lv["start_x_m"] + lv["ramp_length_m"]) + 0.5
    flat_end = 1000 * derived["flat_window_top_m"]
    engaged_in = window("insert", flat_start, flat_end)
    engaged_out = window("withdraw", flat_start, flat_end)
    report = {
        "workspace": {k: v for k, v in ws.items() if not k.startswith("_")},
        "connector": {"plug": spec.plug.name, "socket": spec.socket.name, "derived": derived,
                      "spec": spec.to_dict()},
        "overrides": args.set, "plug_type": args.plug_type, "speed_mm_s": args.speed_mm_s,
        "target_insertion_force_n": lv["insertion_force_target_n"],
        "leaf_numbers": wsp.leaf_numbers(ws, wsp.leaf_openings(spec, ws)),
        "insert_mean_axial_force_n_flat": float(np.mean([r["socket_force_x_n"] for r in engaged_in])) if engaged_in else None,
        "withdraw_mean_axial_force_n_flat": float(np.mean([r["socket_force_x_n"] for r in engaged_out])) if engaged_out else None,
        "flat_window_mm": [flat_start, flat_end],
        "peak_axial_force_n": float(max(abs(r["socket_force_x_n"]) for r in rows)),
        "peak_leaf_normal_n": float(max(r["leaf_normal_n"] for r in rows)),
        "peak_leaf_penetration_mm": float(max(r["leaf_penetration_mm"] for r in rows)),
        "peak_wall_contact_force_n": float(max(r["wall_contact_force_n"] for r in rows)),
        "peak_wall_penetration_mm": float(max(r["wall_penetration_mm"] for r in rows)),
        "leaf_rest_travel_mm_before_contact": {c: rows[0][c] for c in leaf_columns},
        "min_leaf_travel_mm": float(min(min(r[c] for c in leaf_columns) for r in rows)),
        "seated_hold_valid_pose_fraction": float(np.mean([r["valid_pose"] for r in rows if r["phase"] == "hold"])),
        "max_depth_mm": float(max(r["depth_mm"] for r in rows)),
        "warnings": int(sum(w.number for w in data.warning)),
    }
    target = lv["insertion_force_target_n"]
    mean_in = report["insert_mean_axial_force_n_flat"]
    report["passed"] = bool(mean_in is not None and abs(-mean_in - target) <= 0.25 * target
                            and report["peak_leaf_penetration_mm"] <= 1000 * lv["leaf_penetration_limit_m"]
                            and report["peak_wall_penetration_mm"] <= 1000 * config["penetration_limit_m"]
                            and report["warnings"] == 0)
    (args.output / "report.json").write_text(json.dumps(report, indent=2))
    plot(rows, args.output / "force_depth.png", target, derived["plot_depth_range_mm"])
    print(json.dumps({k: v for k, v in report.items() if k not in ("workspace",)}, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
