"""Compose official v1 bimanual assets with the validated connector."""
import copy
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import mujoco
import numpy as np
from connector.geometry import connector_tree, ROOT

UPSTREAM = ROOT / "third_party/openarm_mujoco"
ROBOT_XML = UPSTREAM / "v1/openarm_bimanual.xml"
PINNED_COMMIT = "161039cd74ea8675fb8197836fe5674659825c75"


def scene_xml():
    if not ROBOT_XML.exists():
        raise FileNotFoundError("Run scripts/fetch_openarm.ps1 to obtain pinned OpenArm v1 assets")
    robot = ET.parse(ROBOT_XML).getroot()
    robot.set("model", "openarm_v1_bimanual_plug")
    robot.find("compiler").set("meshdir", str((UPSTREAM / "v1/meshes").resolve()))
    robot.find("compiler").set("angle", "radian")
    # Connector defaults do not alter any explicitly configured robot class.
    default = robot.find("default")
    ET.SubElement(default, "geom", condim="3", friction="0.4 0.005 0.0001",
                  solref="0.002 1", solimp="0.9 0.95 0.001")
    ET.SubElement(default, "site", size="0.0006", group="4")
    ET.SubElement(robot, "option", timestep="0.0005", integrator="implicitfast",
                  solver="Newton", iterations="100", gravity="0 0 -9.81", cone="elliptic", impratio="100")
    visual = ET.SubElement(robot, "visual")
    ET.SubElement(visual, "global", offwidth="640", offheight="480")
    ET.SubElement(visual, "headlight", ambient=".3 .3 .3", diffuse=".6 .6 .6")
    ET.SubElement(visual, "map", znear=".001")
    world = robot.find("worldbody")
    ET.SubElement(world, "light", pos="0 -1 2", dir="0 0 -1", directional="true")
    ET.SubElement(world, "geom", name="floor", type="plane", size="2 2 .01", rgba=".25 .28 .3 1")
    ET.SubElement(world, "geom", name="work_table", type="box", pos=".47 0 .29",
                  size=".22 .4 .03", rgba=".45 .43 .4 1")
    ET.SubElement(world, "geom", name="fixture", type="box", pos=".4631 -.155 .399",
                  size=".006 .024 .079", rgba=".3 .32 .35 1")
    ET.SubElement(world, "camera", name="scene_rgb", pos=".85 -.9 .85",
                  xyaxes=".8 .6 0 -.25 .333 .91", fovy="48")
    ET.SubElement(world, "camera", name="inspection", pos=".56 -.32 .59",
                  xyaxes=".75 .66 0 -.35 .4 .847", fovy="45")
    hand = world.find(".//body[@name='openarm_right_hand']")
    # Broad finger pads resist twist about the pinch axis. V1's inherited
    # torsional coefficient is inactive with condim=3; enable it at fingertips.
    for side in ("right", "left"):
        finger_geom = world.find(f".//geom[@name='openarm_right_{side}_finger_collision']")
        finger_geom.set("condim", "4")
        finger_geom.set("solref", "0.002 1")
    # hand +Z points forward through fingers; pinch is hand +/-Y.
    ET.SubElement(hand, "site", name="robot_grasp", pos="0 .006 .085",
                  quat="0 .7071067811865476 0 .7071067811865476")
    # Illustrative wrist camera mount; explicitly requires real extrinsic calibration.
    camera_pos = np.array([.05, -.04, .01])
    camera_target = np.array([0, .006, .123])
    z = camera_pos - camera_target
    z /= np.linalg.norm(z)
    x = np.cross([1, 0, 0], z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    ET.SubElement(hand, "camera", name="wrist_rgb", pos=".05 -.04 .01",
                  xyaxes=" ".join(map(str, np.r_[x, y])), fovy="55")
    connector = connector_tree(True)
    for element in connector.find("asset"):
        robot.find("asset").append(copy.deepcopy(element))
    robot.append(copy.deepcopy(connector.find("custom")))
    for name in ("plug", "socket"):
        body = connector.find(f"./worldbody/body[@name='{name}']")
        if name == "socket":
            body.set("pos", ".4391 -.155 .478")
        else:
            # Convert the only degree-valued decorative rotation to a quaternion.
            visual_stub = body.find("./geom[@name='strain_relief_visual']")
            visual_stub.attrib.pop("euler")
            visual_stub.set("quat", ".7071067811865476 0 .7071067811865476 0")
        world.append(copy.deepcopy(body))
    return ET.tostring(robot, encoding="unicode")


def build_model():
    return mujoco.MjModel.from_xml_string(scene_xml())


def source_manifest():
    return {"repository": "https://github.com/enactic/openarm_mujoco",
            "commit": PINNED_COMMIT, "variant": "v1/openarm_bimanual.xml",
            "robot_xml_sha256": hashlib.sha256(ROBOT_XML.read_bytes()).hexdigest(),
            "license": "Apache-2.0", "active_arm": "right", "parked_arm": "left"}
