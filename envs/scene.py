"""Compose official v1 bimanual assets with the validated connector and the workspace spec."""
import copy
import hashlib
import xml.etree.ElementTree as ET
import mujoco
from connector import catalog
from connector.geometry import connector_tree, ROOT
from envs import workspace as wsp

UPSTREAM = ROOT / "third_party/openarm_mujoco"
ROBOT_XML = UPSTREAM / "v1/openarm_bimanual.xml"
PINNED_COMMIT = "161039cd74ea8675fb8197836fe5674659825c75"


def scene_xml(ws=None):
    """Robot scene XML. ``ws`` is a loaded workspace spec (configs/workspace_v1.json by default).

    The connector comes from ``ws["connector"]`` (catalog plug/socket) or, without that section,
    the hand-written two-blade assets; the v1 workspace therefore produces the same XML as before.
    """
    if not ROBOT_XML.exists():
        raise FileNotFoundError("Run scripts/fetch_openarm.ps1 to obtain pinned OpenArm v1 assets")
    ws = ws or wsp.load_workspace()
    spec = catalog.from_workspace(ws)
    robot = ET.parse(ROBOT_XML).getroot()
    robot.set("model", "openarm_v1_bimanual_plug")
    robot.find("compiler").set("meshdir", str((UPSTREAM / "v1/meshes").resolve()))
    robot.find("compiler").set("angle", "radian")
    # Connector defaults do not alter any explicitly configured robot class.
    default = robot.find("default")
    ET.SubElement(default, "geom", condim="3", friction="0.4 0.005 0.0001",
                  solref="0.002 1", solimp="0.9 0.95 0.001")
    ET.SubElement(default, "site", size="0.0006", group="4")
    # impratio 1000 (was 100): with the firm finger servos a 10 N axial load crept ~0.3 mm/s in
    # the pads at 100; at 1000 the same load holds within 0.03 mm (PHYSICS_CHANGELOG workspace_v1).
    ET.SubElement(robot, "option", timestep="0.0005", integrator="implicitfast",
                  solver="Newton", iterations="100", gravity="0 0 -9.81", cone="elliptic", impratio="1000")
    visual = ET.SubElement(robot, "visual")
    ET.SubElement(visual, "global", offwidth="640", offheight="480")
    ET.SubElement(visual, "headlight", ambient=".3 .3 .3", diffuse=".6 .6 .6")
    ET.SubElement(visual, "map", znear=".001")
    world = robot.find("worldbody")
    ET.SubElement(world, "light", pos="0 -1 2", dir="0 0 -1", directional="true")
    # Floor, table top + legs, robot pedestal and fixture post from the workspace spec.
    wsp.add_floor_table_pedestal(world, ws)
    hand = world.find(".//body[@name='openarm_right_hand']")
    wsp.add_cameras(world, hand, ws)
    wsp.add_finger_pads(world, ws)
    # Broad finger pads resist twist about the pinch axis. V1's inherited
    # torsional coefficient is inactive with condim=3; enable it at fingertips.
    for side in ("right", "left"):
        finger_geom = world.find(f".//geom[@name='openarm_right_{side}_finger_collision']")
        finger_geom.set("condim", "4")
        finger_geom.set("solref", "0.002 1")
    # hand +Z points forward through fingers; pinch is hand +/-Y.
    ET.SubElement(hand, "site", name="robot_grasp", pos="0 .006 .085",
                  quat="0 .7071067811865476 0 .7071067811865476")
    connector = connector_tree(True, spec=spec)
    asset = robot.find("asset")
    for element in connector.find("asset"):
        asset.append(copy.deepcopy(element))
    robot.append(copy.deepcopy(connector.find("custom")))
    socket = copy.deepcopy(connector.find("./worldbody/body[@name='socket']"))
    socket.set("pos", wsp.vec(ws["socket"]["position_m"]))
    socket.set("quat", wsp.vec(ws["socket"]["quaternion_wxyz"]))
    if ws["socket_leaves"]["enabled"]:
        wsp.add_retention_leaves(socket, asset, ws, wsp.leaf_openings(spec, ws), robot)
    world.append(socket)
    plug = copy.deepcopy(connector.find("./worldbody/body[@name='plug']"))
    # qpos0 is the spawn pose so the baked cable rest arc ends exactly at the appliance anchor
    # (the connect equality samples its appliance-side point from qpos0 at compile time).
    plug.set("pos", wsp.vec(ws["plug_spawn"]["mating_position_m"]))
    plug.set("quat", wsp.vec(ws["plug_spawn"]["quaternion_wxyz"]))
    # Convert the legacy asset's only degree-valued decorative rotation to a quaternion
    # (generated connectors are written with quaternions already).
    visual_stub = plug.find("./geom[@name='strain_relief_visual']")
    if visual_stub is not None and "euler" in visual_stub.attrib:
        visual_stub.attrib.pop("euler")
        visual_stub.set("quat", ".7071067811865476 0 .7071067811865476 0")
    if ws["cable"]["enabled"]:
        wsp.add_appliance(world, ws)
        wsp.add_cable(plug, ws)
        wsp.add_cable_closure(robot, ws)
    world.append(plug)
    return ET.tostring(robot, encoding="unicode")


def build_model(ws=None):
    return mujoco.MjModel.from_xml_string(scene_xml(ws))


def source_manifest():
    return {"repository": "https://github.com/enactic/openarm_mujoco",
            "commit": PINNED_COMMIT, "variant": "v1/openarm_bimanual.xml",
            "robot_xml_sha256": hashlib.sha256(ROBOT_XML.read_bytes()).hexdigest(),
            "license": "Apache-2.0", "active_arm": "right", "parked_arm": "left"}
