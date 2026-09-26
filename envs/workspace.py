"""Workspace geometry generated from configs/workspace_v1.json.

Everything here is pure ElementTree + numpy so the layout can be inspected and unit-tested
without robot assets. ``envs/scene.py`` calls these helpers while composing the robot scene;
``write_cable_qpos`` is the only function that touches a live model/data.
"""
import hashlib
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np

from connector import catalog

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PATH = ROOT / "configs/workspace_v1.json"


def f(x):
    """Deterministic float formatting so the scene XML (and its hash) is reproducible."""
    return f"{float(x):.10g}"


def vec(values):
    return " ".join(f(v) for v in values)


def workspace_sha256(path=DEFAULT_PATH):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_workspace(path=None):
    """Load and validate a workspace spec. An already-loaded dict is validated and returned."""
    if isinstance(path, dict):
        validate(path)
        return path
    path = Path(path) if path else DEFAULT_PATH
    ws = json.loads(path.read_text(encoding="utf-8"))
    ws["_path"] = str(path)
    ws["_sha256"] = workspace_sha256(path)
    validate(ws)
    return ws


def validate(ws):
    frames, table = ws["frames"], ws["table"]
    if abs(frames["table_top_z_m"] - table["height_m"] - frames["floor_z_m"]) > 1e-9:
        raise ValueError("frames.floor_z_m must equal table_top_z_m - table.height_m")
    if table["top_thickness_m"] <= 0 or table["leg_size_m"] <= 0:
        raise ValueError("table thickness and leg size must be positive")
    if ws["pedestal"]["top_z_m"] <= frames["floor_z_m"]:
        raise ValueError("pedestal top must be above the floor")
    cable = ws["cable"]
    if cable["segments"] < 2 or cable["length_m"] <= 0 or cable["radius_m"] <= 0:
        raise ValueError("cable needs >= 2 segments and positive length/radius")
    waypoints = cable.get("rest_waypoints_world", [])
    if any(len(w) != 3 for w in waypoints):
        raise ValueError("cable.rest_waypoints_world entries must be 3-vectors")
    if len(cable.get("rest_bulges_world", [])) > len(waypoints) + 1:
        raise ValueError("cable.rest_bulges_world has more entries than cable pieces")
    fixed = cable.get("rest_segments")
    if fixed is not None:
        if len(fixed) != len(waypoints) + 1:
            raise ValueError("cable.rest_segments needs one entry per cable piece")
        ints = [k for k in fixed if k is not None]
        if any((not isinstance(k, int)) or k < 1 for k in ints) or sum(ints) >= cable["segments"]:
            raise ValueError("cable.rest_segments entries must be positive ints (or null) summing below segments")
    if waypoints:
        cable_rest(ws)   # raises when a piece is shorter than its chord
    leaves = ws["socket_leaves"]
    if leaves["protrusion_m"] <= 0 or leaves["ramp_length_m"] >= leaves["length_m"]:
        raise ValueError("leaf protrusion must be positive and the ramp shorter than the leaf")
    if leaves["travel_m"] <= 0 or leaves["stiffness_n_m"] <= 0:
        raise ValueError("leaf travel and stiffness must be positive")
    for name, override in leaves.get("per_opening", {}).items():
        unknown = set(override) - {"protrusion_m", "half_height_m", "embed_m"}
        if unknown:
            raise ValueError(f"socket_leaves.per_opening.{name}: unknown keys {sorted(unknown)}")
    if "connector" in ws:
        # Generated connectors: the workspace numbers that depend on the plug/socket geometry must
        # agree with the catalog spec so nothing is duplicated by hand.
        spec = catalog.from_workspace(ws)
        d = spec.derived()
        fx = ws["fixture"]
        if abs(ws["socket"]["position_m"][0] + spec.socket.depth_m
               - (fx["center_m"][0] - fx["half_size_m"][0])) > 1e-9:
            raise ValueError("socket.position_m[0] + socket depth must equal the fixture front face")
        spawn_z = frames["table_top_z_m"] + d["spawn_height_above_table_m"]
        if abs(ws["plug_spawn"]["mating_position_m"][2] - spawn_z) > 1e-9:
            raise ValueError(f"plug_spawn.mating_position_m[2] must be {spawn_z:.6g} for plug {spec.plug.name}")
        if any(abs(a - b) > 1e-9 for a, b in zip(cable["attach_local_m"], d["cable_attach_local_m"])):
            raise ValueError(f"cable.attach_local_m must be {list(d['cable_attach_local_m'])} for plug {spec.plug.name}")
        pads = ws.get("gripper_pads") or {}
        if pads.get("enabled"):
            hx, hz = pads["half_size_xz_m"]
            # finger z runs along the hand (plug X), finger x across the housing height (plug Z)
            if 2 * hz > d["housing_depth_m"] or 2 * hx > d["housing_height_m"]:
                raise ValueError("gripper pad plate exceeds the plug housing faces")
        unknown = set(leaves.get("openings", [])) - {o.name for o in spec.socket.openings}
        if unknown:
            raise ValueError(f"socket_leaves.openings names unknown openings {sorted(unknown)}")


def apply_connector(ws, plug=None, socket=None):
    """Switch the workspace to another catalog plug/socket and re-derive the dependent numbers
    (socket x from the fixture front, plug spawn height, cable attachment). Returns ``ws``."""
    if "connector" not in ws:
        raise ValueError("workspace has no connector section (legacy two-blade workspace)")
    if plug:
        ws["connector"]["plug"] = plug
    if socket:
        ws["connector"]["socket"] = socket
    spec = catalog.from_workspace(ws)
    d = spec.derived()
    fx = ws["fixture"]
    ws["socket"]["position_m"][0] = fx["center_m"][0] - fx["half_size_m"][0] - spec.socket.depth_m
    ws["plug_spawn"]["mating_position_m"][2] = ws["frames"]["table_top_z_m"] + d["spawn_height_above_table_m"]
    ws["cable"]["attach_local_m"] = list(d["cable_attach_local_m"])
    validate(ws)
    return ws


# ---------------------------------------------------------------- static furniture

def table_boxes(ws):
    """Return {name: (center, half_size)} for the table top, four legs and the pedestal."""
    t, fr, ped = ws["table"], ws["frames"], ws["pedestal"]
    cx, cy = t["center_xy_m"]
    top_z, floor_z, thick = fr["table_top_z_m"], fr["floor_z_m"], t["top_thickness_m"]
    boxes = {"work_table": ((cx, cy, top_z - thick / 2),
                            (t["width_x_m"] / 2, t["length_y_m"] / 2, thick / 2))}
    under = top_z - thick
    leg_h = (under - floor_z) / 2
    dx = t["width_x_m"] / 2 - t["leg_inset_m"] - t["leg_size_m"] / 2
    dy = t["length_y_m"] / 2 - t["leg_inset_m"] - t["leg_size_m"] / 2
    for name, sx, sy in (("table_leg_near_left", -1, -1), ("table_leg_near_right", -1, 1),
                         ("table_leg_far_left", 1, -1), ("table_leg_far_right", 1, 1)):
        boxes[name] = ((cx + sx * dx, cy + sy * dy, floor_z + leg_h),
                       (t["leg_size_m"] / 2, t["leg_size_m"] / 2, leg_h))
    px, py = ped["center_xy_m"]
    boxes["robot_pedestal"] = ((px, py, (ped["top_z_m"] + floor_z) / 2),
                               (*ped["half_size_xy_m"], (ped["top_z_m"] - floor_z) / 2))
    return boxes


def add_floor_table_pedestal(world, ws):
    fr = ws["frames"]
    ET.SubElement(world, "geom", name="floor", type="plane", pos=vec((0, 0, fr["floor_z_m"])),
                  size=vec((*ws["floor"]["half_size_m"], .01)), rgba=vec(ws["floor"]["rgba"]))
    colors = {"work_table": ws["table"]["rgba_top"], "robot_pedestal": ws["pedestal"]["rgba"]}
    for name, (center, half) in table_boxes(ws).items():
        ET.SubElement(world, "geom", name=name, type="box", pos=vec(center), size=vec(half),
                      rgba=vec(colors.get(name, ws["table"]["rgba_legs"])))
    fx = ws["fixture"]
    ET.SubElement(world, "geom", name="fixture", type="box", pos=vec(fx["center_m"]),
                  size=vec(fx["half_size_m"]), rgba=vec(fx["rgba"]))


def anchor_world(ws):
    a = ws["appliance"]
    return np.asarray(a["center_m"], float) + np.asarray(a["anchor_local_m"], float)


def add_appliance(world, ws):
    a = ws["appliance"]
    body = ET.SubElement(world, "body", name="appliance", pos=vec(a["center_m"]))
    ET.SubElement(body, "geom", name="appliance_box", type="box", size=vec(a["half_size_m"]),
                  rgba=vec(a["rgba"]))
    ET.SubElement(body, "site", name="cable_anchor", pos=vec(a["anchor_local_m"]),
                  size="0.004", rgba="1 .5 .1 1")
    return body


# ---------------------------------------------------------------- gripper pads

def add_finger_pads(world, ws):
    """Flat pad plates on the right fingers; finger local +/-y is the pinch normal."""
    pads = ws.get("gripper_pads")
    if not pads or not pads["enabled"]:
        return []
    names = []
    for side, sign in (("right", 1), ("left", -1)):
        finger = world.find(f".//body[@name='openarm_right_{side}_finger']")
        name = f"openarm_right_{side}_finger_pad"
        cx, cz = pads["center_local_xz_m"]
        cy = sign * (pads["face_offset_m"] - pads["thickness_m"] / 2)
        hx, hz = pads["half_size_xz_m"]
        ET.SubElement(finger, "geom", name=name, type="box", pos=vec((cx, cy, cz)),
                      size=vec((hx, pads["thickness_m"] / 2, hz)), condim=str(pads["condim"]),
                      friction=vec(pads["friction"]), solref=vec(pads["solref"]),
                      priority=str(pads["priority"]), rgba=vec(pads["rgba"]))
        names.append(name)
    return names


# ---------------------------------------------------------------- cameras

def camera_frame(position, target):
    """Camera looks along -Z toward target; returns xyaxes (6 values) like the original scene."""
    position, target = np.asarray(position, float), np.asarray(target, float)
    z = position - target
    z /= np.linalg.norm(z)
    x = np.cross([1, 0, 0], z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return np.r_[x, y]


def add_cameras(world, hand, ws):
    cams = ws["cameras"]
    for name in ("scene_rgb", "inspection"):
        c = cams[name]
        ET.SubElement(world, "camera", name=name, pos=vec(c["position_m"]),
                      xyaxes=vec(c["xyaxes"]), fovy=f(c["fovy_deg"]))
    w = cams["wrist_rgb"]
    xy = camera_frame(w["position_m"], w["target_local_m"])
    ET.SubElement(hand, "camera", name="wrist_rgb", pos=vec(w["position_m"]),
                  xyaxes=vec(xy), fovy=f(w["fovy_deg"]))
    mount = w.get("visual_mount")
    if not mount:
        return
    x, y = xy[:3], xy[3:]
    z = np.cross(x, y)
    pos = np.asarray(w["position_m"], float)
    visual = dict(contype="0", conaffinity="0", group="2", mass="0")
    ET.SubElement(hand, "geom", name="wrist_cam_housing", type="box",
                  pos=vec(pos + z * mount["housing_offset_z_m"]), xyaxes=vec(xy),
                  size=vec(mount["housing_half_m"]), rgba=vec(mount["rgba_housing"]), **visual)
    ET.SubElement(hand, "geom", name="wrist_cam_lens", type="cylinder",
                  pos=vec(pos + z * mount["lens_offset_z_m"]), xyaxes=vec(xy),
                  size=vec((mount["lens_radius_m"], mount["lens_half_length_m"])),
                  rgba=vec(mount["rgba_lens"]), **visual)
    back = pos + z * (mount["housing_offset_z_m"] + mount["housing_half_m"][2])
    ET.SubElement(hand, "geom", name="wrist_cam_bracket", type="capsule",
                  fromto=vec(np.r_[back, mount["bracket_hand_point_m"]]),
                  size=f(mount["bracket_radius_m"]), rgba=vec(mount["rgba_bracket"]), **visual)


# ---------------------------------------------------------------- socket spring leaves

def slot_geometry(connector):
    """Legacy helper: outer slot wall face and blade outer face (|y|) read from the two-blade XML."""
    left = connector.find(".//geom[@name='socket_left']")
    blade = connector.find(".//geom[@name='blade_right']")
    wall = abs(float(left.get("pos").split()[1])) - float(left.get("size").split()[1])
    blade_face = abs(float(blade.get("pos").split()[1])) + float(blade.get("size").split()[1])
    return {"wall_face_m": wall, "blade_face_m": blade_face}


def leaf_openings(spec, ws=None):
    """One dict per socket opening that carries a retention leaf for this plug.

    Round pins are pressed from the hole wall along the opening's ``leaf_axis_yz`` (any direction
    works for a round hole, so the catalog can balance the leaf set). A blade only has flat slot
    walls, so its leaf uses the dominant axis component; in a keyhole it presses *outward* from the
    inner web because the blade's outer side opens into the round part. Openings the plug does not
    use get no leaf. ``ws["socket_leaves"]["openings"]`` (optional) restricts the set.
    """
    selected = None if ws is None else ws["socket_leaves"].get("openings")
    used = {p.opening for p in spec.plug.pins}
    # The catalog tilts leaf axes so the *complete* leaf set balances; a plug that leaves some
    # leaf-bearing opening empty (no earth pin) gets axis-aligned leaves instead.
    complete = all(o.name in used for o in spec.socket.openings if o.leaf_axis_yz != (0.0, 0.0))
    out = []
    for opening in spec.socket.openings:
        axis = opening.leaf_axis_yz
        pins = [p for p in spec.plug.pins if p.opening == opening.name]
        if axis == (0.0, 0.0) or not pins or (selected is not None and opening.name not in selected):
            continue
        pin = pins[0]
        part = opening.part(pin.mates)
        if pin.kind == "blade" or not complete:
            sign = 1.0 if abs(axis[0]) >= abs(axis[1]) else 0.0
            axis = (math.copysign(1.0, axis[0]) * sign + 0.0, math.copysign(1.0, axis[1]) * (1.0 - sign) + 0.0)
        if pin.kind == "blade" and opening.has("hole"):
            axis = (-axis[0] + 0.0, -axis[1] + 0.0)   # + 0.0 avoids writing -0 into the XML
        wall_half = part.wall_half_m(axis)
        element_half = pin.outer_half_m(axis)
        base = (part.center_yz[0] - axis[0] * wall_half, part.center_yz[1] - axis[1] * wall_half)
        # Leaf plate direction across the press axis, signed so the legacy XML order is kept.
        lateral = (-axis[1] + 0.0, axis[0] + 0.0)
        if lateral[1] < 0 or (lateral[1] == 0 and lateral[0] < 0):
            lateral = (-lateral[0] + 0.0, -lateral[1] + 0.0)
        if part.kind == "hole":
            max_half = 0.95 * pin.radius_m
        else:
            max_half = part.wall_half_m(lateral)
        # Leaves sit further in for longer pins so every leaf engages at the same plug travel
        # (a lone earth leaf would push the held plug onto a hole wall before the others engage).
        out.append({"name": opening.name, "kind": pin.kind, "axis_yz": axis, "lateral_yz": lateral,
                    "base_yz": base, "wall_face_m": wall_half, "element_face_m": element_half,
                    "clearance_m": wall_half - element_half, "max_half_height_m": max_half,
                    "start_offset_m": pin.length_m - spec.plug.min_pin_length_m})
    return out


def leaf_params(ws, opening):
    """Leaf dimensions for one opening: config values with optional per-opening overrides."""
    lv = ws["socket_leaves"]
    override = lv.get("per_opening", {}).get(opening["name"], {})
    params = {k: override.get(k, lv[k]) for k in ("protrusion_m", "half_height_m", "embed_m")}
    params["half_height_m"] = min(params["half_height_m"], opening["max_half_height_m"])
    return params


def leaf_numbers(ws, openings):
    """Per-opening normal force, preload at the stop and the springref that produces it."""
    lv = ws["socket_leaves"]
    n_leaves = len(openings)
    if n_leaves == 0:
        raise ValueError("no retention leaf openings for this plug")
    out = {}
    for opening in openings:
        params = leaf_params(ws, opening)
        clearance = opening["clearance_m"]
        interference = params["protrusion_m"] - clearance
        if interference <= 0:
            raise ValueError(f"leaf protrusion does not reach the element face at opening {opening['name']}")
        normal = lv["insertion_force_target_n"] / (n_leaves * lv["friction"][0])
        preload = normal - lv["stiffness_n_m"] * interference
        if preload <= 0:
            raise ValueError("leaf stiffness too high for the target force; preload would be negative")
        out[opening["name"]] = {
            "clearance_m": clearance, "interference_m": interference, "normal_n": normal,
            "preload_n": preload, "springref_m": preload / lv["stiffness_n_m"],
            "protrusion_m": params["protrusion_m"], "half_height_m": params["half_height_m"],
            "start_x_m": lv["start_x_m"] + opening["start_offset_m"],
            "engage_depth_m": lv["start_x_m"] + opening["start_offset_m"]
            + lv["ramp_length_m"] * (clearance + params["embed_m"]) / (params["protrusion_m"] + params["embed_m"])}
    return out


def add_retention_leaves(socket_body, asset, ws, openings, root=None):
    """Append one leaf body (+ mesh) per opening to the socket body.

    ``root`` receives ``<contact><exclude>`` pairs: the socket is static, so MuJoCo welds it to
    the world and the usual parent-child filter would not stop the wall from colliding with the
    leaf that is embedded in it.
    """
    lv = ws["socket_leaves"]
    nums = leaf_numbers(ws, openings)
    names = []
    for opening in openings:
        name = f"socket_leaf_{opening['name']}"
        params = leaf_params(ws, opening)
        h = params["half_height_m"]
        # Pure ramp from below the wall surface up to the contact face: there is no flat front
        # face for the element tip to hit even if the leaf sits slightly past its soft stop.
        profile = [(0, -params["embed_m"]), (lv["ramp_length_m"], params["protrusion_m"]),
                   (lv["length_m"], params["protrusion_m"]), (lv["length_m"], -params["embed_m"])]
        ay, az = opening["axis_yz"]
        ly, lz = opening["lateral_yz"]
        # Profile coordinate c runs along the press axis (toward the element), t across it.
        vertices = [(x, ay * c + ly * t, az * c + lz * t) for x, c in profile for t in (-h, h)]
        ET.SubElement(asset, "mesh", name=name,
                      vertex=" ".join(f(v) for point in vertices for v in point))
        body = ET.SubElement(socket_body, "body", name=name,
                             pos=vec((lv["start_x_m"] + opening["start_offset_m"], *opening["base_yz"])))
        ET.SubElement(body, "joint", name=f"{name}_slide", type="slide", axis=vec((0, ay, az)),
                      range=vec((-lv["travel_m"], 0)), limited="true",
                      stiffness=f(lv["stiffness_n_m"]), springref=f(nums[opening["name"]]["springref_m"]),
                      damping=f(lv["damping_ns_m"]), armature=f(lv["armature_kg"]),
                      solreflimit=vec(lv["limit_solref"]), solimplimit=vec(lv["limit_solimp"]))
        ET.SubElement(body, "geom", name=name, type="mesh", mesh=name, mass=f(lv["mass_kg"]),
                      friction=vec(lv["friction"]), priority="1", solref=vec(lv["geom_solref"]),
                      solimp=vec(lv["geom_solimp"]), rgba=vec(lv["rgba"]))
        names.append(name)
        if root is not None:
            contact = root.find("contact")
            if contact is None:
                contact = ET.SubElement(root, "contact")
            ET.SubElement(contact, "exclude", name=f"{name}_wall", body1="socket", body2=name)
    return names


def leaf_names_in_model(model):
    """Leaf geom names discovered from a compiled model (``socket_leaf_<opening>``)."""
    names = [model.geom(i).name for i in range(model.ngeom)]
    return tuple(n for n in names if n.startswith("socket_leaf_"))


# ---------------------------------------------------------------- gripper aperture

def finger_travel_for_width(ws, width_m, finger_base_gap_m):
    """Right-finger slide travel at which the pad faces touch a housing of ``width_m``.

    Pad faces sit ``face_offset_m`` inside each finger; the fingers are ``finger_base_gap_m``
    apart at zero travel, so the aperture is ``base_gap - 2*face_offset + 2*travel``.
    """
    return (width_m - finger_base_gap_m) / 2 + ws["gripper_pads"]["face_offset_m"]


# ---------------------------------------------------------------- cable

def quat_to_mat(q):
    w, x, y, z = np.asarray(q, float) / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def mat_to_quat(R):
    R = np.asarray(R, float)
    t = np.trace(R)
    if t > 0:
        s = np.sqrt(t + 1) * 2
        q = [.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s]
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        q = [(R[2, 1] - R[1, 2]) / s, .25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s]
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        q = [(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, .25 * s, (R[1, 2] + R[2, 1]) / s]
    else:
        s = np.sqrt(1 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        q = [(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, .25 * s]
    q = np.asarray(q)
    return q / np.linalg.norm(q) * (1 if q[0] >= 0 else -1)


def arc_frames(start, end, length, n, bulge):
    """Regular polygonal arc of n equal segments from start to end with total length ``length``.

    Returns (positions[n+1, 3], rotations[n, 3, 3]); rotation k has +X along segment k and +Z
    along the arc-plane normal. The arc bulges toward ``bulge`` (projected perpendicular to the
    chord). Raises ValueError when the chord is longer than the cable.
    """
    start, end = np.asarray(start, float), np.asarray(end, float)
    chord = end - start
    c = float(np.linalg.norm(chord))
    if c >= length - 1e-9:
        raise ValueError(f"cable too short: chord {c:.4f} m >= length {length:.4f} m")
    u = chord / c
    b = np.asarray(bulge, float)
    b = b - np.dot(b, u) * u
    if np.linalg.norm(b) < 1e-9:
        b = np.cross(u, [0, 0, 1]) if abs(u[2]) < .9 else np.cross(u, [1, 0, 0])
    b /= np.linalg.norm(b)
    normal = np.cross(u, b)
    ls = length / n

    def polygon_chord(phi):
        return ls * np.sin(n * phi / 2) / np.sin(phi / 2)
    lo, hi = 1e-12, 2 * np.pi / n - 1e-12
    for _ in range(200):
        mid = (lo + hi) / 2
        if polygon_chord(mid) > c:
            lo = mid
        else:
            hi = mid
    phi = (lo + hi) / 2
    thetas = (n - 1) * phi / 2 - phi * np.arange(n)
    directions = np.outer(np.cos(thetas), u) + np.outer(np.sin(thetas), b)
    positions = start + np.vstack(([0, 0, 0], np.cumsum(directions * ls, axis=0)))
    rotations = np.stack([np.column_stack((d, np.cross(normal, d), normal)) for d in directions])
    return positions, rotations


def cable_pieces(ws, start):
    """Chain of arc pieces start -> rest_waypoints_world -> anchor: [(p0, p1, n_segments)].

    Waypoint coordinates given as ``null`` copy that coordinate from ``start`` (the plug attach),
    so a fixed edge/floor route follows a randomised plug position. Segments (all of length L/n)
    are shared out by ``cable.rest_segments`` when present (an int per piece, or ``null``: such
    pieces get just enough segments to clear their chord and the *last* null piece receives every
    remaining segment, i.e. the slack coils there); otherwise in proportion to the piece chords.
    Either way segments are shifted from the slackest piece until every piece is longer than its
    chord. Without waypoints this is the single whole-cable arc used by the original layout.
    """
    cable = ws["cable"]
    n, ls = cable["segments"], cable["length_m"] / cable["segments"]
    start = np.asarray(start, float)
    waypoints = [np.array([start[i] if w[i] is None else float(w[i]) for i in range(3)])
                 for w in cable.get("rest_waypoints_world", [])]
    points = [start, *waypoints, anchor_world(ws)]
    chords = [float(np.linalg.norm(b - a)) for a, b in zip(points[:-1], points[1:])]
    if len(chords) == 1:
        return [(points[0], points[1], n)]
    if sum(chords) >= cable["length_m"] - 1e-9:
        # The plug has moved too far (e.g. held at the socket) for the cable to pass through every
        # waypoint: the slack on the far side is used up, so drop the last waypoint and retry.
        trimmed = dict(ws, cable=dict(cable, rest_waypoints_world=[w.tolist() for w in waypoints[:-1]],
                                      rest_segments=cable.get("rest_segments", [None] * len(chords))[:-1]))
        return cable_pieces(trimmed, start)
    total = sum(chords)
    fixed = cable.get("rest_segments")
    if fixed and len(fixed) == len(chords):
        counts = [int(k) if k is not None else int(math.ceil(c / ls)) + 1 for k, c in zip(fixed, chords)]
        free = [i for i, k in enumerate(fixed) if k is None]
        remaining = n - sum(counts)
        if remaining < 0 or not free:
            raise ValueError("cable.rest_segments leaves no segments for the free pieces")
        counts[free[-1]] += remaining
    else:
        counts = [max(1, int(round(n * c / total))) for c in chords]
    while sum(counts) != n:   # rounding drift: adjust the piece with the most segments
        k = int(np.argmax(counts)) if sum(counts) > n else int(np.argmin(np.array(counts) * ls - np.array(chords)))
        counts[k] += 1 if sum(counts) < n else -1
    for _ in range(n):
        slack = [k * ls - c for k, c in zip(counts, chords)]
        short = int(np.argmin(slack))
        if slack[short] > 1e-9:
            break
        donor = int(np.argmax(slack))
        if donor == short or counts[donor] <= 1:
            raise ValueError(f"cable too short for its waypoints: piece {short} chord {chords[short]:.4f} m")
        counts[donor] -= 1
        counts[short] += 1
    else:
        raise ValueError("cable segments cannot be shared out over the waypoints")
    return [(a, b, k) for a, b, k in zip(points[:-1], points[1:], counts)]


def cable_path(ws, start, first_bulge=None):
    """Positions/rotations of the whole chain laid as consecutive arcs through the waypoints.

    ``first_bulge`` overrides the bulge of the plug-side piece (the env hangs a held plug's cable
    downward); later pieces use ``rest_bulges_world[i]`` or ``rest_bulge_world``.
    """
    cable = ws["cable"]
    ls = cable["length_m"] / cable["segments"]
    bulges = cable.get("rest_bulges_world", [])
    positions, rotations = [], []
    for i, (a, b, k) in enumerate(cable_pieces(ws, start)):
        bulge = first_bulge if (i == 0 and first_bulge is not None) else (
            bulges[i] if i < len(bulges) else cable["rest_bulge_world"])
        if i > 0:
            # A waypoint piece whose bulge is nearly parallel to its chord (a floor coil whose chord
            # turned with a randomised plug) would otherwise bow along the leftover component,
            # possibly into the floor: bow it horizontally instead, away from the robot (+X).
            u = (np.asarray(b) - np.asarray(a)) / max(np.linalg.norm(np.asarray(b) - np.asarray(a)), 1e-12)
            bv = np.asarray(bulge, float)
            residual = bv - np.dot(bv, u) * u
            if np.linalg.norm(residual) < 0.5 * np.linalg.norm(bv):
                side = np.cross(u, [0, 0, 1])
                bulge = side if side[0] >= 0 else -side
        pos, rot = arc_frames(a, b, k * ls, k, bulge)
        positions.append(pos if i == 0 else pos[1:])
        rotations.append(rot)
    return np.vstack(positions), np.concatenate(rotations)


def cable_rest(ws):
    """Rest shape (baked into the XML) from the spawn pose through the waypoints to the anchor."""
    cable, spawn = ws["cable"], ws["plug_spawn"]
    r_plug = quat_to_mat(spawn["quaternion_wxyz"])
    start = np.asarray(spawn["mating_position_m"]) + r_plug @ np.asarray(cable["attach_local_m"])
    positions, rotations = cable_path(ws, start)
    return r_plug, positions, rotations


def add_cable(plug_body, ws):
    """Append the capsule/ball-joint chain to the plug body. Returns the segment body names."""
    cable = ws["cable"]
    n, ls = cable["segments"], cable["length_m"] / cable["segments"]
    r_plug, _, rotations = cable_rest(ws)
    parent, r_parent = plug_body, r_plug
    names = []
    for k in range(n):
        name = f"cable_seg_{k:02d}"
        rel = mat_to_quat(r_parent.T @ rotations[k])
        pos = cable["attach_local_m"] if k == 0 else (ls, 0, 0)
        body = ET.SubElement(parent, "body", name=name, pos=vec(pos), quat=vec(rel))
        ET.SubElement(body, "joint", name=f"cable_j_{k:02d}", type="ball",
                      stiffness=f(cable["bend_stiffness_nm_rad"]),
                      damping=f(cable["bend_damping_nms_rad"]),
                      armature=f(cable["joint_armature_kgm2"]))
        ET.SubElement(body, "geom", name=name, type="capsule", fromto=vec((0, 0, 0, ls, 0, 0)),
                      size=f(cable["radius_m"]), mass=f(cable["mass_per_metre_kg_m"] * ls),
                      friction=vec(cable["friction"]), condim="3",
                      contype=str(cable["contype"]), conaffinity=str(cable["conaffinity"]),
                      rgba=vec(cable["rgba"]))
        if k == 0:
            ET.SubElement(body, "site", name="cable_root", pos="0 0 0")
        if k == n - 1:
            ET.SubElement(body, "site", name="cable_end", pos=vec((ls, 0, 0)))
        names.append(name)
        parent, r_parent = body, rotations[k]
    return names


def add_cable_closure(root, ws):
    """Connect the last segment's tip to the appliance anchor and add the junction sensors."""
    cable = ws["cable"]
    ls = cable["length_m"] / cable["segments"]
    equality = root.find("equality")
    if equality is None:
        equality = ET.SubElement(root, "equality")
    ET.SubElement(equality, "connect", name="cable_anchor",
                  body1=f"cable_seg_{cable['segments'] - 1:02d}", body2="appliance",
                  anchor=vec((ls, 0, 0)), solref=vec(cable["anchor_solref"]))
    sensor = root.find("sensor")
    if sensor is None:
        sensor = ET.SubElement(root, "sensor")
    ET.SubElement(sensor, "force", name="cable_root_force", site="cable_root")
    ET.SubElement(sensor, "torque", name="cable_root_torque", site="cable_root")


def write_cable_qpos(model, data, ws, bulge=None):
    """Bend the cable joints so the chain runs from the plug's current pose to the anchor.

    Writes only the cable ball-joint entries of ``data.qpos``; the caller places the plug first
    and calls ``mj_forward`` afterwards. The baked rest shape is the spawn-layout arc, so the
    joint quaternions are expressed relative to the XML body orientations.
    """
    cable = ws["cable"]
    n = cable["segments"]
    adr = model.jnt_qposadr[model.joint("plug_free").id]
    plug_pos = np.array(data.qpos[adr:adr + 3])
    r_plug = quat_to_mat(data.qpos[adr + 3:adr + 7])
    start = plug_pos + r_plug @ np.asarray(cable["attach_local_m"], float)
    _, rotations = cable_path(ws, start, first_bulge=bulge)
    r_parent = r_plug
    for k in range(n):
        body = model.body(f"cable_seg_{k:02d}").id
        r_rest = quat_to_mat(model.body_quat[body])
        joint = model.jnt_qposadr[model.joint(f"cable_j_{k:02d}").id]
        data.qpos[joint:joint + 4] = mat_to_quat(r_rest.T @ r_parent.T @ rotations[k])
        r_parent = rotations[k]


def cable_body_names(ws):
    return [f"cable_seg_{k:02d}" for k in range(ws["cable"]["segments"])] if ws["cable"]["enabled"] else []


def derived(ws, openings=None):
    """Numbers implied by the spec, for docs and tests."""
    cable = ws["cable"]
    out = {"table_boxes": {k: {"center_m": list(map(float, c)), "half_size_m": list(map(float, h))}
                           for k, (c, h) in table_boxes(ws).items()},
           "cable_segment_length_m": cable["length_m"] / cable["segments"],
           "cable_segment_mass_kg": cable["mass_per_metre_kg_m"] * cable["length_m"] / cable["segments"],
           "cable_total_mass_kg": cable["mass_per_metre_kg_m"] * cable["length_m"],
           "cable_anchor_world_m": anchor_world(ws).tolist()}
    try:
        _, positions, _ = cable_rest(ws)
        out["cable_rest_chord_m"] = float(np.linalg.norm(positions[-1] - positions[0]))
        out["cable_rest_slack_m"] = cable["length_m"] - out["cable_rest_chord_m"]
        out["cable_rest_min_x_m"] = float(positions[:, 0].min())
        out["cable_rest_min_z_m"] = float(positions[:, 2].min())
        ls = cable["length_m"] / cable["segments"]
        out["cable_pieces"] = [{"chord_m": float(np.linalg.norm(b - a)), "length_m": k * ls, "segments": k,
                                "slack_m": k * ls - float(np.linalg.norm(b - a))}
                               for a, b, k in cable_pieces(ws, positions[0])]
    except ValueError as error:
        out["cable_rest_error"] = str(error)
    if openings is not None:
        out["leaves"] = leaf_numbers(ws, openings)
    return out
