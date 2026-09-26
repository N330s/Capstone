"""Turn a ``ConnectorSpec`` into the plug/socket bodies of a MuJoCo scene.

Collision: plug housing box(es), blade boxes, round-pin capsules; socket = outer frame boxes plus the
slab-decomposed opening zone (boxes / convex hull meshes) and tapered lead-in hulls. Visual-only
detail (``contype=0 conaffinity=0 group=2 mass=0``): housing edge bevels, pin sleeves, cable boot,
socket bezel and cavity back. Rotations are written as quaternions so the same bodies compile under
``angle="degree"`` (holder scene) and ``angle="radian"`` (robot scene).
"""
from __future__ import annotations

import copy
import xml.etree.ElementTree as ET
from pathlib import Path

from connector.slab import socket_pieces
from connector.spec import ConnectorSpec

ROOT = Path(__file__).resolve().parents[1]
CONNECTOR = ROOT / "assets/connector"
X_AXIS_QUAT = ".7071067811865476 0 .7071067811865476 0"   # cylinder/capsule Z axis -> +X
SOCKET_HEIGHT = 0.10                                         # holder-scene socket origin height
VISUAL = {"contype": "0", "conaffinity": "0", "group": "2", "mass": "0"}
# Collision bitmasks. Pins (contype 1) meet the decomposed socket walls (conaffinity 1); the housing
# (contype 2) meets one flat face plate (conaffinity 2) instead of every wall piece, so seating costs
# a single box-box contact rather than hundreds of housing-vs-hull pairs. Both plug groups keep
# conaffinity 1 so default geoms (fingers, pads, table, leaves) still collide with them.
PIN = {"contype": "1", "conaffinity": "0"}
HOUSING = {"contype": "2", "conaffinity": "1"}
WALL = {"contype": "0", "conaffinity": "1"}
PLATE = {"contype": "0", "conaffinity": "2"}


def f(x):
    """Deterministic float formatting (same as envs.workspace.f) so XML hashes are reproducible."""
    return f"{float(x):.10g}"


def vec(values):
    return " ".join(f(v) for v in values)


# ----------------------------------------------------------------- plug

def build_plug_body(spec: ConnectorSpec, pos=None) -> ET.Element:
    plug = spec.plug
    depth, width, height = plug.housing_size_m
    cz = plug.housing_center_z_m
    r = plug.edge_radius_m
    body = ET.Element("body", name="plug", pos=vec(pos if pos is not None else
                                                   (spec.derived()["preinsert_x_m"], 0, SOCKET_HEIGHT)))
    ET.SubElement(body, "freejoint", name="plug_free")
    ET.SubElement(body, "inertial", pos=vec(plug.com_m), mass=f(plug.mass_kg),
                  diaginertia=vec(plug.diaginertia))
    rgba = vec(plug.rgba_housing)
    # Body origin is the housing front / mating frame. The full-height box keeps the name
    # ``housing`` (table-rest contact checks look for it); the second box fills the width.
    ET.SubElement(body, "geom", name="housing", type="box", pos=vec((-depth / 2, 0, cz)),
                  size=vec((depth / 2, width / 2 - r, height / 2)), rgba=rgba, **HOUSING)
    if r > 0:
        ET.SubElement(body, "geom", name="housing_fill", type="box", pos=vec((-depth / 2, 0, cz)),
                      size=vec((depth / 2, width / 2, height / 2 - r)), rgba=rgba, **HOUSING)
        for k, (sy, sz) in enumerate(((-1, -1), (1, -1), (1, 1), (-1, 1))):
            ET.SubElement(body, "geom", name=f"housing_edge_{k}", type="cylinder",
                          pos=vec((-depth / 2, sy * (width / 2 - r), cz + sz * (height / 2 - r))),
                          quat=X_AXIS_QUAT, size=vec((r, depth / 2)), rgba=rgba, **VISUAL)
    for pin in plug.pins:
        y, z = pin.pos_yz
        if pin.kind == "blade":
            ET.SubElement(body, "geom", name=pin.geom_name, type="box",
                          pos=vec((pin.length_m / 2, y, z)),
                          size=vec((pin.length_m / 2, pin.thickness_m / 2, pin.width_m / 2)),
                          rgba=vec(pin.rgba), **PIN)
        else:
            # Capsule from the housing face to length - r so the hemispherical tip ends at length.
            ET.SubElement(body, "geom", name=pin.geom_name, type="capsule",
                          fromto=vec((0, y, z, pin.length_m - pin.radius_m, y, z)),
                          size=f(pin.radius_m), rgba=vec(pin.rgba), **PIN)
        if pin.sleeve_m > 0:
            sleeve_r = pin.outer_half_m((1, 0)) + 0.0001
            ET.SubElement(body, "geom", name=f"sleeve_{pin.name}", type="cylinder",
                          pos=vec((pin.sleeve_m / 2, y, z)), quat=X_AXIS_QUAT,
                          size=vec((sleeve_r, pin.sleeve_m / 2)), rgba=vec(plug.rgba_sleeve), **VISUAL)
    boot = plug.boot
    if boot:
        ET.SubElement(body, "geom", name="cable_boot_visual", type="cylinder",
                      pos=vec((-(depth + boot["length_m"] / 2), 0, cz)), quat=X_AXIS_QUAT,
                      size=vec((boot["radius_m"], boot["length_m"] / 2)), rgba=vec(boot["rgba"]), **VISUAL)
    ET.SubElement(body, "site", name="plug_mating_frame", pos="0 0 0", rgba="1 0.2 0.2 1")
    ET.SubElement(body, "site", name="plug_grasp_frame", pos=vec(plug.grasp_offset_m), rgba="0.2 1 0.3 1")
    for pin in plug.pins:
        ET.SubElement(body, "site", name=pin.tip_site, pos=vec((pin.length_m, *pin.pos_yz)),
                      rgba="1 0.3 0.2 1")
    return body


# ----------------------------------------------------------------- socket

def build_socket_body(spec: ConnectorSpec, leadin: bool, pos=None):
    """Return ``(body, mesh_assets)``; the meshes must be appended to the scene ``<asset>``."""
    s = spec.socket
    body = ET.Element("body", name="socket", pos=vec(pos if pos is not None else (0, 0, SOCKET_HEIGHT)))
    assets = []
    rgba = vec(s.rgba)
    frame, throat, lead = socket_pieces(s, leadin=leadin)
    for piece in frame + throat + lead:
        if piece.is_box:
            centre, half = piece.box()
            ET.SubElement(body, "geom", name=piece.name, type="box", pos=vec(centre), size=vec(half),
                          rgba=rgba, **WALL)
        else:
            mesh = ET.Element("mesh", name=piece.name,
                              vertex=" ".join(f(v) for point in piece.vertices() for v in point))
            assets.append(mesh)
            ET.SubElement(body, "geom", name=piece.name, type="mesh", mesh=piece.name, rgba=rgba,
                          **WALL)
    # Flat front face for the housing to seat against (see PLATE); pins pass through it.
    y0, y1 = s.face_y_m
    z0, z1 = s.face_z_m
    ET.SubElement(body, "geom", name="socket_face_plate", type="box",
                  pos=vec((0.0005, (y0 + y1) / 2, (z0 + z1) / 2)),
                  size=vec((0.0005, (y1 - y0) / 2, (z1 - z0) / 2)), rgba=rgba, group="3", **PLATE)
    bezel = s.bezel
    if bezel:
        w, proud = bezel["width_m"], bezel["proud_m"]
        brgba = vec(bezel["rgba"])
        for name, centre, half in (
                ("top", (-proud / 2, (y0 + y1) / 2, z1 + w / 2), (proud / 2, (y1 - y0) / 2 + w, w / 2)),
                ("bottom", (-proud / 2, (y0 + y1) / 2, z0 - w / 2), (proud / 2, (y1 - y0) / 2 + w, w / 2)),
                ("left", (-proud / 2, y0 - w / 2, (z0 + z1) / 2), (proud / 2, w / 2, (z1 - z0) / 2)),
                ("right", (-proud / 2, y1 + w / 2, (z0 + z1) / 2), (proud / 2, w / 2, (z1 - z0) / 2))):
            ET.SubElement(body, "geom", name=f"socket_bezel_{name}", type="box", pos=vec(centre),
                          size=vec(half), rgba=brgba, **VISUAL)
        ET.SubElement(body, "geom", name="socket_cavity_back", type="box",
                      pos=vec((s.depth_m - 0.0005, (y0 + y1) / 2, (z0 + z1) / 2)),
                      size=vec((0.0005, (y1 - y0) / 2, (z1 - z0) / 2)), rgba="0.12 0.12 0.12 1", **VISUAL)
    ET.SubElement(body, "site", name="socket_entry", pos="0 0 0", rgba="0.2 0.5 1 1")
    for opening in s.openings:
        for part in opening.parts:
            ET.SubElement(body, "site", name=f"opening_{opening.name}_{part.kind}",
                          pos=vec((0, *part.center_yz)))
    ET.SubElement(body, "site", name="socket_seated_frame", pos="0 0 0")
    ET.SubElement(body, "site", name="socket_preinsert_frame",
                  pos=vec((spec.derived()["preinsert_x_m"], 0, 0)))
    return body, assets


# ----------------------------------------------------------------- scene

def custom_block(spec: ConnectorSpec, leadin: bool) -> ET.Element:
    custom = ET.Element("custom")
    ET.SubElement(custom, "numeric", name="lead_length", data=f(spec.socket.leadin_m if leadin else 0.0))
    ET.SubElement(custom, "text", name="connector_spec", data=spec.to_json())
    return custom


def build_scene(spec: ConnectorSpec, leadin: bool) -> ET.Element:
    """Holder-benchmark scene: the ``plug_socket.xml`` envelope with generated bodies."""
    scene = ET.parse(CONNECTOR / "plug_socket.xml").getroot()
    scene.set("model", f"{spec.name}_{'leadin' if leadin else 'straight'}")
    world = scene.find("worldbody")
    for include in list(world.findall("include")):
        world.remove(include)
    s = spec.socket
    mount = world.find("./geom[@name='mount']")
    top = SOCKET_HEIGHT + s.face_z_m[1] + 0.004
    mount.set("pos", vec((s.depth_m + 0.006, 0, top / 2)))
    mount.set("size", vec((0.006, max(0.024, s.face_y_m[1] + 0.004), top / 2)))
    # Finer shadow map: the many small socket faces show shadow acne under the grazing key light.
    ET.SubElement(scene.find("visual"), "quality", shadowsize="4096")
    asset = ET.SubElement(scene, "asset")
    socket, meshes = build_socket_body(spec, leadin)
    asset.extend(meshes)
    scene.append(custom_block(spec, leadin))
    world.append(socket)
    world.append(build_plug_body(spec))
    return scene


def scene_xml(spec: ConnectorSpec, leadin: bool) -> str:
    return ET.tostring(build_scene(spec, leadin), encoding="unicode")
