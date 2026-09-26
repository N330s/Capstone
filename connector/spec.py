"""Parametric connector description: plug pins, socket openings and every derived distance.

Pure data (no mujoco import at module level). ``connector/builder.py`` turns a ``ConnectorSpec``
into MuJoCo XML and embeds the spec as a ``<custom><text name="connector_spec">`` so that metrics,
leaves, the env and scripts read geometry back from the compiled model instead of copying numbers.
Frames: plug/socket +X is insertion, Y separates the line/neutral elements, Z is blade width.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict

LEGACY_NAME = "legacy_two_blade"


def _pair(v):
    return (float(v[0]), float(v[1]))


def _triple(v):
    return (float(v[0]), float(v[1]), float(v[2]))


def _norm(value):
    """Lists become tuples (recursively) so JSON round-trips compare equal."""
    if isinstance(value, dict):
        return {k: _norm(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return tuple(_norm(v) for v in value)
    return value


@dataclass(frozen=True)
class PinSpec:
    """One conductive element of the plug. ``kind`` is ``"blade"`` (box) or ``"round"`` (capsule)."""
    name: str
    kind: str
    pos_yz: tuple
    length_m: float
    width_m: float = 0.0        # blade: Z extent
    thickness_m: float = 0.0    # blade: Y extent
    diameter_m: float = 0.0     # round pin
    sleeve_m: float = 0.0       # insulating sleeve length from the housing face (visual only)
    opening: str = ""           # socket opening this pin mates with (default: same name)
    rgba: tuple = (0.72, 0.70, 0.59, 1.0)

    def __post_init__(self):
        if self.kind not in ("blade", "round"):
            raise ValueError(f"pin kind must be blade or round, got {self.kind!r}")
        if self.length_m <= 0:
            raise ValueError("pin length must be positive")
        if self.kind == "blade" and (self.width_m <= 0 or self.thickness_m <= 0):
            raise ValueError("blade needs positive width and thickness")
        if self.kind == "round" and self.diameter_m <= 0:
            raise ValueError("round pin needs a positive diameter")
        object.__setattr__(self, "pos_yz", _pair(self.pos_yz))
        object.__setattr__(self, "rgba", tuple(float(c) for c in self.rgba))
        if not self.opening:
            object.__setattr__(self, "opening", self.name)

    @property
    def mates(self) -> str:
        """Which part of a (key)hole opening the pin uses."""
        return "slot" if self.kind == "blade" else "hole"

    @property
    def geom_name(self) -> str:
        prefix = "blade" if self.kind == "blade" else "pin"
        return f"{prefix}_{self.name}"

    @property
    def tip_site(self) -> str:
        return f"plug_tip_{self.name}"

    @property
    def radius_m(self) -> float:
        return self.diameter_m / 2

    def outer_half_m(self, axis_yz) -> float:
        """Half extent of the element along a unit axis in the Y-Z plane."""
        if self.kind == "round":
            return self.radius_m
        return abs(axis_yz[0]) * self.thickness_m / 2 + abs(axis_yz[1]) * self.width_m / 2


@dataclass(frozen=True)
class PlugSpec:
    name: str
    pins: tuple
    housing_size_m: tuple            # full X depth, Y width (pinch axis), Z height
    housing_center_z_m: float        # housing box centre relative to the mating frame
    grasp_offset_m: tuple            # plug_grasp_frame site in the plug frame
    mass_kg: float
    com_m: tuple
    diaginertia: tuple
    symmetry_rolls_deg: tuple = (0.0,)   # (0, 180) for non-polarised plugs
    edge_radius_m: float = 0.0           # visual bevel on the four X edges (0 = plain box)
    boot: dict = field(default_factory=dict)   # visual cable boot {radius_m, length_m, rgba}
    rgba_housing: tuple = (0.045, 0.05, 0.055, 1.0)
    rgba_sleeve: tuple = (0.05, 0.05, 0.06, 1.0)

    def __post_init__(self):
        object.__setattr__(self, "pins", tuple(self.pins))
        names = [p.name for p in self.pins]
        if len(names) != len(set(names)) or not names:
            raise ValueError("plug pins must be non-empty with unique names")
        object.__setattr__(self, "housing_size_m", _triple(self.housing_size_m))
        object.__setattr__(self, "grasp_offset_m", _triple(self.grasp_offset_m))
        object.__setattr__(self, "com_m", _triple(self.com_m))
        object.__setattr__(self, "diaginertia", _triple(self.diaginertia))
        object.__setattr__(self, "symmetry_rolls_deg", tuple(float(r) for r in self.symmetry_rolls_deg))
        object.__setattr__(self, "rgba_housing", tuple(float(c) for c in self.rgba_housing))
        object.__setattr__(self, "rgba_sleeve", tuple(float(c) for c in self.rgba_sleeve))
        object.__setattr__(self, "boot", _norm(self.boot))
        if 0.0 not in self.symmetry_rolls_deg:
            raise ValueError("symmetry_rolls_deg must include 0")
        if self.edge_radius_m < 0 or 2 * self.edge_radius_m >= min(self.housing_size_m[1:]):
            raise ValueError("edge radius must be non-negative and smaller than the housing half size")

    @property
    def max_pin_length_m(self) -> float:
        return max(p.length_m for p in self.pins)

    @property
    def min_pin_length_m(self) -> float:
        return min(p.length_m for p in self.pins)

    @property
    def housing_half_m(self) -> tuple:
        return tuple(s / 2 for s in self.housing_size_m)

    def pin(self, name: str) -> PinSpec:
        for p in self.pins:
            if p.name == name:
                return p
        raise KeyError(name)


@dataclass(frozen=True)
class Part:
    """Convex part of an opening: ``kind`` ``"hole"`` (circle) or ``"slot"`` (rectangle)."""
    kind: str
    center_yz: tuple
    radius_m: float = 0.0
    half_yz: tuple = (0.0, 0.0)

    def __post_init__(self):
        if self.kind not in ("hole", "slot"):
            raise ValueError(f"opening part kind must be hole or slot, got {self.kind!r}")
        object.__setattr__(self, "center_yz", _pair(self.center_yz))
        object.__setattr__(self, "half_yz", _pair(self.half_yz))
        if self.kind == "hole" and self.radius_m <= 0:
            raise ValueError("hole needs a positive radius")
        if self.kind == "slot" and min(self.half_yz) <= 0:
            raise ValueError("slot needs positive half extents")

    def bounds(self):
        """(ylo, yhi, zlo, zhi) of the part."""
        cy, cz = self.center_yz
        hy, hz = (self.radius_m, self.radius_m) if self.kind == "hole" else self.half_yz
        return (cy - hy, cy + hy, cz - hz, cz + hz)

    def wall_half_m(self, axis_yz) -> float:
        """Distance from the part centre to its wall along a unit axis (a flat of the hole polygon)."""
        if self.kind == "hole":
            return self.radius_m
        return abs(axis_yz[0]) * self.half_yz[0] + abs(axis_yz[1]) * self.half_yz[1]


@dataclass(frozen=True)
class OpeningSpec:
    name: str
    parts: tuple
    leaf_axis_yz: tuple = (0.0, 0.0)   # unit direction the retention leaf presses along (0 = no leaf)

    def __post_init__(self):
        object.__setattr__(self, "parts", tuple(self.parts))
        if not self.parts:
            raise ValueError("opening needs at least one part")
        kinds = [p.kind for p in self.parts]
        if len(kinds) != len(set(kinds)):
            raise ValueError("an opening has at most one hole part and one slot part")
        object.__setattr__(self, "leaf_axis_yz", _pair(self.leaf_axis_yz))
        norm = (self.leaf_axis_yz[0] ** 2 + self.leaf_axis_yz[1] ** 2) ** 0.5
        if norm and abs(norm - 1) > 1e-9:
            raise ValueError("leaf_axis_yz must be a unit vector or zero")

    def part(self, kind: str) -> Part:
        for p in self.parts:
            if p.kind == kind:
                return p
        raise KeyError(f"opening {self.name} has no {kind} part")

    def has(self, kind: str) -> bool:
        return any(p.kind == kind for p in self.parts)

    def bounds(self):
        b = [p.bounds() for p in self.parts]
        return (min(x[0] for x in b), max(x[1] for x in b), min(x[2] for x in b), max(x[3] for x in b))


@dataclass(frozen=True)
class SocketSpec:
    name: str
    openings: tuple
    face_y_m: tuple                 # (lo, hi) of the front face in Y
    face_z_m: tuple                 # (lo, hi) in Z (may be asymmetric)
    depth_m: float
    leadin_m: float = 0.001
    leadin_dilation_m: float = 0.0003
    polygon_sides: int = 16
    rgba: tuple = (0.82, 0.81, 0.76, 1.0)
    bezel: dict = field(default_factory=dict)   # visual faceplate {width_m, proud_m, rgba}

    def __post_init__(self):
        object.__setattr__(self, "openings", tuple(self.openings))
        names = [o.name for o in self.openings]
        if len(names) != len(set(names)) or not names:
            raise ValueError("socket openings must be non-empty with unique names")
        object.__setattr__(self, "face_y_m", _pair(self.face_y_m))
        object.__setattr__(self, "face_z_m", _pair(self.face_z_m))
        object.__setattr__(self, "rgba", tuple(float(c) for c in self.rgba))
        object.__setattr__(self, "bezel", _norm(self.bezel))
        if self.depth_m <= 0 or self.leadin_m < 0 or self.leadin_dilation_m < 0:
            raise ValueError("socket depth must be positive; lead-in values non-negative")
        if self.leadin_m >= self.depth_m:
            raise ValueError("lead-in must be shorter than the socket depth")
        if self.polygon_sides < 8 or self.polygon_sides % 4:
            raise ValueError("polygon_sides must be a multiple of 4 and >= 8 (flats on the axes)")
        for o in self.openings:
            ylo, yhi, zlo, zhi = o.bounds()
            if (ylo <= self.face_y_m[0] or yhi >= self.face_y_m[1]
                    or zlo <= self.face_z_m[0] or zhi >= self.face_z_m[1]):
                raise ValueError(f"opening {o.name} must lie strictly inside the socket face")
        for a in self.openings:
            for b in self.openings:
                if a.name < b.name:
                    ab, bb = a.bounds(), b.bounds()
                    if ab[0] < bb[1] and bb[0] < ab[1] and ab[2] < bb[3] and bb[2] < ab[3]:
                        raise ValueError(f"openings {a.name} and {b.name} overlap")

    def opening(self, name: str) -> OpeningSpec:
        for o in self.openings:
            if o.name == name:
                return o
        raise KeyError(name)


@dataclass(frozen=True)
class ConnectorSpec:
    plug: PlugSpec
    socket: SocketSpec

    def __post_init__(self):
        for pin in self.plug.pins:
            opening = self.socket.opening(pin.opening)   # KeyError if missing
            opening.part(pin.mates)                       # KeyError if the part kind is missing
        if self.socket.depth_m < self.plug.max_pin_length_m + 0.0015:
            raise ValueError(f"socket {self.socket.name} ({self.socket.depth_m*1e3:.1f} mm) is too shallow "
                             f"for plug {self.plug.name} pins ({self.plug.max_pin_length_m*1e3:.1f} mm)")

    @property
    def name(self) -> str:
        return f"{self.plug.name}__{self.socket.name}"

    @property
    def is_legacy(self) -> bool:
        return self.plug.name == LEGACY_NAME and self.socket.name == LEGACY_NAME

    def opening_for(self, pin: PinSpec) -> OpeningSpec:
        return self.socket.opening(pin.opening)

    def derived(self) -> dict:
        """Every distance the controllers, probes and calibration used to hardcode."""
        max_pin, min_pin = self.plug.max_pin_length_m, self.plug.min_pin_length_m
        depth, width, height = self.plug.housing_size_m
        cz = self.plug.housing_center_z_m
        return {
            "preinsert_x_m": -(max_pin + 0.006),
            "retreat_x_m": -(max_pin + 0.002),
            "probe_preplug_x_m": -(max_pin + 0.009),
            "success_depth_m": min_pin - 0.001,
            "flat_window_top_m": min_pin - 0.0005,
            "plot_depth_range_mm": (-(max_pin + 0.006) * 1e3, max_pin * 1e3 + 2),
            "table_rest_z_offset_m": height / 2 - cz,
            "spawn_height_above_table_m": height / 2 - cz + 0.002,
            "cable_attach_local_m": (-(depth + 0.012), 0.0, cz),
            "housing_width_m": width,
            "housing_depth_m": depth,
            "housing_height_m": height,
            "max_pin_length_m": max_pin,
            "min_pin_length_m": min_pin,
        }

    # ------------------------------------------------------------------ serialisation
    def to_dict(self) -> dict:
        return json.loads(json.dumps(asdict(self)))

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))

    @staticmethod
    def from_dict(d: dict) -> "ConnectorSpec":
        plug = dict(d["plug"])
        plug["pins"] = tuple(PinSpec(**p) for p in plug["pins"])
        socket = dict(d["socket"])
        openings = []
        for o in socket["openings"]:
            o = dict(o)
            o["parts"] = tuple(Part(**p) for p in o["parts"])
            openings.append(OpeningSpec(**o))
        socket["openings"] = tuple(openings)
        return ConnectorSpec(PlugSpec(**plug), SocketSpec(**socket))

    @staticmethod
    def from_json(text: str) -> "ConnectorSpec":
        return ConnectorSpec.from_dict(json.loads(text))

    @staticmethod
    def from_model(model) -> "ConnectorSpec":
        """Spec embedded in a compiled model, or the legacy spec for the hand-written XML."""
        import mujoco
        text_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_TEXT, "connector_spec")
        if text_id < 0:
            from connector import catalog
            return catalog.legacy()
        adr, size = int(model.text_adr[text_id]), int(model.text_size[text_id])
        raw = bytes(model.text_data[adr:adr + size]).rstrip(b"\0")
        return ConnectorSpec.from_json(raw.decode("utf-8"))
