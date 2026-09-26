"""Connector scene trees: the hand-written two-blade assets (optionally with the convex lead-in
patch) or a generated catalog variant (``spec=``). The legacy path never changes the saved XML."""
from pathlib import Path
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
CONNECTOR = ROOT / "assets/connector"


def connector_tree(leadin: bool = False, *, spec=None) -> ET.Element:
    """Scene tree. ``spec`` (a ``ConnectorSpec``) selects the generated builder; ``None`` or the
    legacy spec reproduces the hand-written two-blade assets byte for byte."""
    if spec is not None and not spec.is_legacy:
        from connector.builder import build_scene
        return build_scene(spec, leadin)
    scene = ET.parse(CONNECTOR / "plug_socket.xml").getroot()
    world = scene.find("worldbody")
    for include in list(world.findall("include")):
        body = ET.parse(CONNECTOR / include.attrib["file"]).getroot().find("body")
        world.remove(include)
        world.append(body)
    if not leadin:
        return scene
    scene.set("model", "two_blade_v2_leadin")
    asset = ET.SubElement(scene, "asset")
    custom = ET.SubElement(scene, "custom")
    ET.SubElement(custom, "numeric", name="lead_length", data="0.001")
    body = world.find("./body[@name='socket']")
    # Cross-section bounds at throat x=1 mm. Front opening expands 0.3 mm.
    blocks = {
        "socket_top": (-.02, .02, .0034, .015),
        "socket_bottom": (-.02, .02, -.015, -.0034),
        "socket_middle": (-.00535, .00535, -.0034, .0034),
        "socket_left": (-.02, -.00765, -.0034, .0034),
        "socket_right": (.00765, .02, -.0034, .0034),
    }
    fronts = {
        "socket_top": (-.02, .02, .0037, .015),
        "socket_bottom": (-.02, .02, -.015, -.0037),
        "socket_middle": (-.00505, .00505, -.0037, .0037),
        "socket_left": (-.02, -.00795, -.0037, .0037),
        "socket_right": (.00795, .02, -.0037, .0037),
    }
    for name, bounds in blocks.items():
        geom = body.find(f"./geom[@name='{name}']")
        pos = geom.attrib["pos"].split()
        size = geom.attrib["size"].split()
        pos[0], size[0] = ".0095", ".0085"
        geom.set("pos", " ".join(pos))
        geom.set("size", " ".join(size))
        vertices = []
        for x, section in ((0, fronts[name]), (.001, bounds)):
            yl, yh, zl, zh = section
            vertices.extend((x, y, z) for y in (yl, yh) for z in (zl, zh))
        mesh = name + "_lead"
        ET.SubElement(asset, "mesh", name=mesh,
                      vertex=" ".join(str(v) for point in vertices for v in point))
        ET.SubElement(body, "geom", name=mesh, type="mesh", mesh=mesh,
                      rgba=geom.attrib["rgba"])
    return scene


def connector_xml(leadin=False, *, spec=None):
    return ET.tostring(connector_tree(leadin, spec=spec), encoding="unicode")


def connector_spec_from_model(model):
    """Spec embedded in a compiled model (legacy models return the catalog legacy spec)."""
    from connector.spec import ConnectorSpec
    return ConnectorSpec.from_model(model)

