"""Named plug and socket variants.

All dimensions are nominal design values taken from the plug standards (TIS 166-2549 Type O,
CEE 7/16 Europlug, NEMA 1-15 / 5-15) or chosen for the simulation; none is a measurement of real
hardware. Clearances are design choices documented in docs/WORKSPACE.md. ``legacy_two_blade`` mirrors
the hand-written ``assets/connector/*.xml`` exactly so those files never need to change.
"""
from __future__ import annotations

from connector.spec import (LEGACY_NAME, ConnectorSpec, OpeningSpec, Part, PinSpec, PlugSpec,
                            SocketSpec)

BRASS = (0.72, 0.70, 0.59, 1.0)
DARK = (0.045, 0.05, 0.055, 1.0)
IVORY = (0.82, 0.81, 0.76, 1.0)
BOOT = {"radius_m": 0.0045, "length_m": 0.012, "rgba": (0.025, 0.028, 0.03, 1.0)}
BEZEL = {"width_m": 0.006, "proud_m": 0.002, "rgba": (0.90, 0.89, 0.85, 1.0)}

# Universal Thai socket geometry (mm): blades at 12.7 mm centres, round pins at 19 mm centres,
# earth 11.89 mm above the line/neutral axis (shared with the NEMA 5-15 ground pin position).
# Retention leaves: the earth leaf presses down (-Z) and the line/neutral leaves press 30 deg
# above the inward horizontal, so the three equal leaf forces and their torques about X sum to
# zero (like paired socket contacts) and the held plug is not pushed against a hole wall.
BLADE_PITCH = 0.0127
LEAF_TILT = (0.8660254037844387, 0.5)   # (cos 30, sin 30)
ROUND_PITCH = 0.019
EARTH_Z = 0.01189
HOLE_R = 0.00255          # 4.8 mm pin + 0.15 mm radial clearance
SLOT_HALF = (0.00095, 0.00345)   # 1.5 x 6.35 mm blade + 0.2 / 0.275 mm per side

_POLARISED = (0.0,)
_SYMMETRIC = (0.0, 180.0)


def _round(name, y, z, length, diameter, sleeve=0.0):
    return PinSpec(name, "round", (y, z), length, diameter_m=diameter, sleeve_m=sleeve, rgba=BRASS)


def _blade(name, y, z, length, width, thickness):
    return PinSpec(name, "blade", (y, z), length, width_m=width, thickness_m=thickness, rgba=BRASS)


def _box_inertia(mass, size):
    d, w, h = size
    return (mass / 12 * (w * w + h * h), mass / 12 * (d * d + h * h), mass / 12 * (d * d + w * w))


def _plug(name, pins, size, center_z, mass, symmetry):
    return PlugSpec(name=name, pins=pins, housing_size_m=size, housing_center_z_m=center_z,
                    grasp_offset_m=(-0.016, 0.0, center_z), mass_kg=mass,
                    com_m=(-size[0] * 0.47, 0.0, center_z * 0.8), diaginertia=_box_inertia(mass, size),
                    symmetry_rolls_deg=symmetry, edge_radius_m=0.003, boot=dict(BOOT),
                    rgba_housing=DARK)


_SMALL = (0.030, 0.034, 0.016)     # depth, width (pinch axis), height for 2-pin plugs
_EARTHED = (0.030, 0.034, 0.020)   # taller housing covering the earth pin, centre +6 mm


def legacy_plug() -> PlugSpec:
    return PlugSpec(
        name=LEGACY_NAME,
        pins=(_blade("left", -0.0065, 0.0, 0.016, 0.006, 0.0015),
              _blade("right", 0.0065, 0.0, 0.016, 0.006, 0.0015)),
        housing_size_m=(0.028, 0.024, 0.016), housing_center_z_m=0.0,
        grasp_offset_m=(-0.016, 0.0, 0.0), mass_kg=0.03, com_m=(-0.012, 0.0, 0.0),
        diaginertia=(0.0000025, 0.000006, 0.0000065), symmetry_rolls_deg=_SYMMETRIC,
        boot={"radius_m": 0.004, "length_m": 0.012, "rgba": (0.025, 0.028, 0.03, 1.0)},
        rgba_housing=DARK)


def legacy_socket() -> SocketSpec:
    return SocketSpec(
        name=LEGACY_NAME,
        openings=(OpeningSpec("left", (Part("slot", (-0.0065, 0.0), half_yz=(0.00115, 0.0034)),), (1.0, 0.0)),
                  OpeningSpec("right", (Part("slot", (0.0065, 0.0), half_yz=(0.00115, 0.0034)),), (-1.0, 0.0))),
        face_y_m=(-0.02, 0.02), face_z_m=(-0.015, 0.015), depth_m=0.018,
        leadin_m=0.001, leadin_dilation_m=0.0003, rgba=IVORY)


def universal_th() -> SocketSpec:
    def keyhole(name, sign):
        return OpeningSpec(name, (Part("slot", (sign * BLADE_PITCH / 2, 0.0), half_yz=SLOT_HALF),
                                  Part("hole", (sign * ROUND_PITCH / 2, 0.0), radius_m=HOLE_R)),
                           leaf_axis_yz=(-sign * LEAF_TILT[0], LEAF_TILT[1]))
    return SocketSpec(
        name="universal_th",
        openings=(keyhole("left", -1), keyhole("right", 1),
                  OpeningSpec("earth", (Part("hole", (0.0, EARTH_Z), radius_m=HOLE_R),), (0.0, -1.0))),
        face_y_m=(-0.02, 0.02), face_z_m=(-0.014, 0.026), depth_m=0.023,
        leadin_m=0.001, leadin_dilation_m=0.0003, polygon_sides=16, rgba=IVORY, bezel=dict(BEZEL))


def type_o() -> PlugSpec:
    return _plug("type_o", (_round("left", -ROUND_PITCH / 2, 0.0, 0.019, 0.0048, sleeve=0.010),
                            _round("right", ROUND_PITCH / 2, 0.0, 0.019, 0.0048, sleeve=0.010),
                            _round("earth", 0.0, EARTH_Z, 0.0214, 0.0048)),
                 _EARTHED, 0.006, 0.045, _POLARISED)


def type_c() -> PlugSpec:
    return _plug("type_c", (_round("left", -ROUND_PITCH / 2, 0.0, 0.019, 0.004),
                            _round("right", ROUND_PITCH / 2, 0.0, 0.019, 0.004)),
                 _SMALL, 0.0, 0.035, _SYMMETRIC)


def type_a() -> PlugSpec:
    return _plug("type_a", (_blade("left", -BLADE_PITCH / 2, 0.0, 0.0159, 0.00635, 0.0015),
                            _blade("right", BLADE_PITCH / 2, 0.0, 0.0159, 0.00635, 0.0015)),
                 _SMALL, 0.0, 0.035, _SYMMETRIC)


def type_b() -> PlugSpec:
    return _plug("type_b", (_blade("left", -BLADE_PITCH / 2, 0.0, 0.0159, 0.00635, 0.0015),
                            _blade("right", BLADE_PITCH / 2, 0.0, 0.0159, 0.00635, 0.0015),
                            _round("earth", 0.0, EARTH_Z, 0.019, 0.0048)),
                 _EARTHED, 0.006, 0.045, _POLARISED)


PLUGS = {LEGACY_NAME: legacy_plug, "type_o": type_o, "type_c": type_c, "type_a": type_a, "type_b": type_b}
SOCKETS = {LEGACY_NAME: legacy_socket, "universal_th": universal_th}
DEFAULT_PLUG, DEFAULT_SOCKET = "type_o", "universal_th"


def plug_names():
    return tuple(PLUGS)


def socket_names():
    return tuple(SOCKETS)


def plug(name: str) -> PlugSpec:
    try:
        return PLUGS[name]()
    except KeyError:
        raise KeyError(f"unknown plug {name!r}; choose from {plug_names()}") from None


def socket(name: str) -> SocketSpec:
    try:
        return SOCKETS[name]()
    except KeyError:
        raise KeyError(f"unknown socket {name!r}; choose from {socket_names()}") from None


def get(plug_name: str = DEFAULT_PLUG, socket_name: str = DEFAULT_SOCKET) -> ConnectorSpec:
    return ConnectorSpec(plug(plug_name), socket(socket_name))


def legacy() -> ConnectorSpec:
    return get(LEGACY_NAME, LEGACY_NAME)


def add_cli_arguments(parser):
    """``--plug-type`` / ``--socket`` options for the holder scripts (omit both = legacy pair)."""
    parser.add_argument("--plug-type", choices=plug_names(), default=None,
                        help="catalog plug (default: legacy two-blade; a catalog plug defaults the socket to universal_th)")
    parser.add_argument("--socket", choices=socket_names(), default=None,
                        help="catalog socket (default: universal_th when --plug-type is given)")


def from_cli(args) -> ConnectorSpec:
    """Spec named by ``add_cli_arguments`` options; ``None`` when neither flag is given."""
    if args.plug_type is None and args.socket is None:
        return None
    plug_name = args.plug_type or LEGACY_NAME
    socket_name = args.socket or (LEGACY_NAME if plug_name == LEGACY_NAME else DEFAULT_SOCKET)
    return get(plug_name, socket_name)


def from_workspace(ws: dict) -> ConnectorSpec:
    """Spec named by ``ws["connector"]``; workspaces without that section use the legacy pair."""
    section = ws.get("connector")
    if not section:
        return legacy()
    return get(section.get("plug", DEFAULT_PLUG), section.get("socket", DEFAULT_SOCKET))
