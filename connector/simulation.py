"""Shared dynamics, socket-relative metrics and trial runner (no robot dependency)."""
from __future__ import annotations

import csv
import hashlib
import itertools
import json
import math
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path

import mujoco
import numpy as np

from connector.spec import ConnectorSpec

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "assets/connector/plug_socket.xml"
CONFIG_PATH = ROOT / "configs/holder.json"
VERSION = "two_blade_v1_straight"


@dataclass(frozen=True)
class Pose:
    offset_y_mm: float = 0.0
    offset_z_mm: float = 0.0
    roll_deg: float = 0.0
    pitch_deg: float = 0.0
    yaw_deg: float = 0.0


def rotation(pose: Pose) -> np.ndarray:
    r, p, y = np.radians([pose.roll_deg, pose.pitch_deg, pose.yaw_deg])
    rx = np.array([[1, 0, 0], [0, np.cos(r), -np.sin(r)], [0, np.sin(r), np.cos(r)]])
    ry = np.array([[np.cos(p), 0, np.sin(p)], [0, 1, 0], [-np.sin(p), 0, np.cos(p)]])
    rz = np.array([[np.cos(y), -np.sin(y), 0], [np.sin(y), np.cos(y), 0], [0, 0, 1]])
    return rz @ ry @ rx


def roll_matrix(roll_deg: float) -> np.ndarray:
    r = math.radians(roll_deg)
    return np.array([[1, 0, 0], [0, math.cos(r), -math.sin(r)], [0, math.sin(r), math.cos(r)]])


def rotation_vector(matrix: np.ndarray) -> np.ndarray:
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, matrix.ravel())
    if q[0] < 0:
        q = -q
    n = np.linalg.norm(q[1:])
    return 2 * math.atan2(n, q[0]) * q[1:] / n if n > 1e-12 else np.zeros(3)


def limited(vector: np.ndarray, limit: float) -> np.ndarray:
    return vector * min(1.0, limit / max(float(np.linalg.norm(vector)), 1e-12))


def clip_at_throat(vertices, signs, throat):
    """Vertices of the blade volume intersected with x >= throat."""
    points = [v for v in vertices if v[0] >= throat]
    for i in range(8):
        for j in range(i + 1, 8):
            if np.count_nonzero(signs[i] != signs[j]) != 1:
                continue
            a, b = vertices[i], vertices[j]
            if (a[0] < throat <= b[0]) or (b[0] < throat <= a[0]):
                points.append(a + (b - a) * ((throat - a[0]) / (b[0] - a[0])))
    return np.asarray(points).reshape(-1, 3)


class ConnectorMetrics:
    """Read-only metrics bound to caller-owned model/data; never reset the robot.

    Geometry (elements, openings, clearances, polarity) comes from the ``ConnectorSpec`` embedded in
    the model (or the legacy catalog entry for the hand-written XML), so the same oracle serves every
    plug variant: blades use a corner-in-slot corridor test, round pins a radial axis test.
    """
    def bind(self, model, data, config, allowed_grasp_bodies=(), leaf_geoms=(), cable_sensor=None,
             *, spec: ConnectorSpec | None = None):
        """``leaf_geoms``/``cable_sensor`` describe the robot-scene spring leaves and cable
        junction sensor; the holder benchmark passes neither and reports zeros for them."""
        self.model, self.data = model, data
        self.spec = spec or ConnectorSpec.from_model(model)
        self.config = dict(config)
        if not self.spec.is_legacy:
            # holder.json's success depth is the legacy 16 mm blade value; other plugs derive it.
            self.config["success_depth_m"] = self.spec.derived()["success_depth_m"]
        self.allowed_grasp_bodies = set(allowed_grasp_bodies)
        self.leaf_geoms = {self.model.geom(n).id for n in leaf_geoms}
        self.socket_bodies = {self.model.body("socket").id} | {
            int(self.model.geom_bodyid[g]) for g in self.leaf_geoms}
        self.cable_sensor = cable_sensor
        if cable_sensor:
            self.cable_sensor_adr = [self.model.sensor_adr[self.model.sensor(n).id]
                                     for n in cable_sensor]
        numeric_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_NUMERIC, "lead_length")
        self.lead_length = (float(model.numeric_data[model.numeric_adr[numeric_id]])
                            if numeric_id >= 0 else 0.0)
        self.plug = self.model.body("plug").id
        self.socket = self.model.body("socket").id
        self.grasp = self.model.site("plug_grasp_frame").id
        self.entry = self.model.site("socket_entry").id
        self.elements = [(pin, self.model.geom(pin.geom_name).id, self.model.site(pin.tip_site).id)
                         for pin in self.spec.plug.pins]
        self.element_names = [pin.name for pin in self.spec.plug.pins]
        # Opening parts in the socket-entry frame, keyed by (opening, kind).
        self.parts = {(o.name, p.kind): p for o in self.spec.socket.openings for p in o.parts}
        # Element -> opening assignment for every symmetry roll the plug allows (180 deg for
        # non-polarised plugs swaps left/right; polarised plugs have only the identity).
        self.assignments = {}
        for roll in self.spec.plug.symmetry_rolls_deg:
            rot = roll_matrix(roll)[1:, 1:]
            mapping = {}
            for pin in self.spec.plug.pins:
                y, z = rot @ np.array(pin.pos_yz)
                best = min(((np.hypot(part.center_yz[0] - y, part.center_yz[1] - z), name)
                            for (name, kind), part in self.parts.items() if kind == pin.mates),
                           default=(math.inf, None))
                if best[0] > 0.001:
                    raise ValueError(f"plug {self.spec.plug.name}: no {pin.mates} opening for pin "
                                     f"{pin.name} at roll {roll} deg")
                mapping[pin.name] = best[1]
            self.assignments[roll] = mapping
        self.signs = np.array(list(itertools.product((-1, 1), repeat=3)))

    def _element_fits(self, pin, geom, basis, origin, plug_pose, part, tolerance):
        """Corridor test for one element against its opening part (throat region only)."""
        if pin.kind == "blade":
            corners = (self.signs * self.model.geom_size[geom]) @ self.data.geom_xmat[geom].reshape(3, 3).T
            local = (corners + self.data.geom_xpos[geom] - origin) @ basis
            if self.lead_length > 0:
                local = clip_at_throat(local, self.signs, self.lead_length)
            return bool(np.all(np.abs(local[:, 1:3] - np.array(part.center_yz)) <=
                               np.array(part.half_yz) + tolerance))
        # Round pin: the capsule axis from the housing face to the tip-sphere centre must stay
        # within (hole radius - pin radius) of the hole axis; the surface is within r of the axis.
        pos, rot = plug_pose
        y, z = pin.pos_yz
        base = basis.T @ (pos + rot @ np.array([0.0, y, z]) - origin)
        tip = basis.T @ (pos + rot @ np.array([pin.length_m - pin.radius_m, y, z]) - origin)
        points = [base, tip]
        if self.lead_length > 0:
            if tip[0] < self.lead_length:
                return True
            if base[0] < self.lead_length:
                t = (self.lead_length - base[0]) / (tip[0] - base[0])
                points = [base + t * (tip - base), tip]
        limit = part.radius_m - pin.radius_m + tolerance
        return all(math.hypot(p[1] - part.center_yz[0], p[2] - part.center_yz[1]) <= limit for p in points)

    def diagnostics(self) -> dict:
        m, d, c = self.model, self.data, self.config
        basis = d.site_xmat[self.entry].reshape(3, 3)
        origin = d.site_xpos[self.entry]
        relative = basis.T @ (d.xpos[self.plug] - origin)
        relative_rotation = basis.T @ d.xmat[self.plug].reshape(3, 3)
        # Non-polarised plugs allow a half-turn roll with swapped opening assignment.
        rolls = list(self.assignments)
        candidates = [relative_rotation @ roll_matrix(r) for r in rolls]
        angles = [np.linalg.norm(rotation_vector(r)) for r in candidates]
        chosen = int(np.argmin(angles))
        assignment = self.assignments[rolls[chosen]]
        angle = float(np.degrees(min(angles)))
        depth = [float((basis.T @ (d.site_xpos[tip] - origin))[0]) for _, _, tip in self.elements]
        plug_pose = (d.xpos[self.plug], d.xmat[self.plug].reshape(3, 3))
        fits = True
        for pin, geom, _ in self.elements:
            part = self.parts[(assignment[pin.name], pin.mates)]
            fits &= self._element_fits(pin, geom, basis, origin, plug_pose, part, c["fit_tolerance_m"])
        pair_force = 0.0
        grasp_force = 0.0
        grasp_penetration = 0.0
        other_force = 0.0
        penetration = 0.0
        count = 0
        element_forces = [0.0] * len(self.elements)
        contact_wrench = np.zeros(6)
        # Force breakdown on the plug (world frame, then expressed in the socket-entry frame):
        # rigid socket walls, spring leaves and finger pads are accumulated separately.
        wall_normal = wall_friction = leaf_normal = leaf_friction = leaf_penetration = 0.0
        leaf_count = 0
        socket_force, socket_torque = np.zeros(3), np.zeros(3)
        finger_force = np.zeros(3)
        plug_origin = d.xpos[self.plug]
        for i in range(d.ncon):
            contact = d.contact[i]
            bodies = [int(m.geom_bodyid[g]) for g in contact.geom]
            if self.plug not in bodies:
                continue
            mujoco.mj_contactForce(m, d, i, contact_wrench)
            magnitude = float(np.linalg.norm(contact_wrench[:3]))
            # mj_contactForce is the force on geom2 along the geom1->geom2 normal in the
            # contact frame; flip it when the plug is geom1 so it is the force on the plug.
            world_force = contact.frame.reshape(3, 3).T @ contact_wrench[:3]
            if int(m.geom_bodyid[contact.geom1]) == self.plug:
                world_force = -world_force
            normal = abs(float(contact_wrench[0]))
            tangential = float(np.hypot(contact_wrench[1], contact_wrench[2]))
            if any(b in self.allowed_grasp_bodies for b in bodies):
                grasp_force += magnitude
                grasp_penetration = max(grasp_penetration, -float(contact.dist))
                finger_force += world_force
                continue
            if any(g in self.leaf_geoms for g in contact.geom):
                leaf_normal += normal
                leaf_friction += tangential
                leaf_count += 1
                leaf_penetration = max(leaf_penetration, -float(contact.dist))
                socket_force += world_force
                socket_torque += np.cross(contact.pos - plug_origin, world_force)
                continue
            penetration = max(penetration, -float(contact.dist))
            if self.socket in bodies:
                pair_force += magnitude
                count += 1
                wall_normal += normal
                wall_friction += tangential
                socket_force += world_force
                socket_torque += np.cross(contact.pos - plug_origin, world_force)
                for j, (_, geom, _) in enumerate(self.elements):
                    if geom in contact.geom:
                        element_forces[j] += magnitude
            else:
                other_force += magnitude
        socket_force_local = basis.T @ socket_force
        socket_torque_local = basis.T @ socket_torque
        finger_force_local = basis.T @ finger_force
        cable_force = np.zeros(3)
        cable_torque = np.zeros(3)
        cable_tension = 0.0
        if self.cable_sensor:
            cable_force = np.array(d.sensordata[self.cable_sensor_adr[0]:self.cable_sensor_adr[0] + 3])
            cable_torque = np.array(d.sensordata[self.cable_sensor_adr[1]:self.cable_sensor_adr[1] + 3])
            # Sensor frame = cable_root site frame whose +X points along the first segment.
            cable_tension = float(cable_force[0])
        velocity = np.zeros(6)
        mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_SITE, self.grasp, velocity, 0)
        face_gap = -float(relative[0])
        relative_quaternion = np.zeros(4)
        mujoco.mju_mat2Quat(relative_quaternion, relative_rotation.ravel())
        valid = (min(depth) >= c["success_depth_m"] and fits
                 and -c["penetration_limit_m"] <= face_gap <= c["success_face_gap_m"]
                 and angle < c["success_angle_deg"]
                 and penetration <= c["penetration_limit_m"]
                 and pair_force < c["contact_abort_n"] and other_force < 0.01
                 and leaf_penetration <= c.get("leaf_penetration_limit_m", np.inf))
        return {
            "time_s": float(d.time),
            **{f"{name}_depth_m": value for name, value in zip(self.element_names, depth)},
            "insertion_depth_m": min(depth), "face_gap_m": face_gap,
            "offset_y_m": float(relative[1]), "offset_z_m": float(relative[2]),
            "orientation_error_deg": angle, "blades_fit": fits,
            "contact_force_n": pair_force, "other_contact_force_n": other_force,
            **{f"{name}_contact_force_n": value for name, value in zip(self.element_names, element_forces)},
            "contact_count": count, "max_penetration_m": penetration,
            "grasp_contact_force_n": grasp_force, "grasp_penetration_m": grasp_penetration,
            "plug_weight_n": float(m.body_mass[self.plug] * np.linalg.norm(m.opt.gravity)),
            "socket_wall_normal_n": wall_normal, "socket_wall_friction_n": wall_friction,
            "leaf_normal_n": leaf_normal, "leaf_friction_n": leaf_friction,
            "leaf_contact_force_n": float(np.hypot(leaf_normal, leaf_friction)),
            "leaf_contact_count": leaf_count, "leaf_penetration_m": leaf_penetration,
            "socket_force_x_n": float(socket_force_local[0]),
            "socket_force_y_n": float(socket_force_local[1]),
            "socket_force_z_n": float(socket_force_local[2]),
            "socket_force_total_n": float(np.linalg.norm(socket_force)),
            "socket_lateral_force_n": float(np.hypot(*socket_force_local[1:])),
            "socket_torque_x_nm": float(socket_torque_local[0]),
            "socket_torque_y_nm": float(socket_torque_local[1]),
            "socket_torque_z_nm": float(socket_torque_local[2]),
            "socket_torque_nm": float(np.linalg.norm(socket_torque)),
            "finger_force_x_n": float(finger_force_local[0]),
            "finger_force_y_n": float(finger_force_local[1]),
            "finger_force_z_n": float(finger_force_local[2]),
            "cable_force_n": float(np.linalg.norm(cable_force)),
            "cable_tension_n": cable_tension,
            "cable_torque_nm": float(np.linalg.norm(cable_torque)),
            "speed_m_s": float(np.linalg.norm(velocity[3:])),
            "angular_speed_rad_s": float(np.linalg.norm(velocity[:3])),
            "valid_pose": bool(valid),
            **{f"relative_q{axis}": float(value) for axis, value in
               zip("wxyz", relative_quaternion)},
            **{f"velocity_{axis}": float(value) for axis, value in
               zip(("wx_rad_s", "wy_rad_s", "wz_rad_s", "x_m_s", "y_m_s", "z_m_s"), velocity)},
            "elements_fit": fits, "polarity_roll_deg": float(rolls[chosen]),
        }


class ConnectorSimulation(ConnectorMetrics):
    def __init__(self, config: dict | None = None, timestep: float | None = None, leadin: bool = False,
                 *, spec: ConnectorSpec | None = None):
        holder = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        if config:
            unknown = config.keys() - holder.keys()
            if unknown:
                raise ValueError(f"Unknown config fields: {unknown}")
            holder.update(config)
        if any(not np.isfinite(v) or v <= 0 for v in holder.values()):
            raise ValueError("All holder configuration values must be finite and positive")
        from connector import catalog
        from connector.geometry import connector_xml
        self.leadin = leadin
        self.spec = spec or catalog.legacy()
        self.model = mujoco.MjModel.from_xml_string(connector_xml(leadin, spec=self.spec))
        if timestep is not None:
            if not np.isfinite(timestep) or timestep <= 0:
                raise ValueError("timestep must be finite and positive")
            self.model.opt.timestep = timestep
        self.data = mujoco.MjData(self.model)
        self.bind(self.model, self.data, holder, spec=self.spec)
        self.qadr = self.model.jnt_qposadr[self.model.joint("plug_free").id]
        self.reset()

    def reset(self, pose: Pose = Pose(), mating_x: float | None = None):
        """Place the plug ``mating_x`` along socket +X from the entry (default: the spec's
        pre-insert distance, -22 mm for the legacy plug) with the given pose offsets."""
        if mating_x is None:
            mating_x = self.spec.derived()["preinsert_x_m"]
        if not all(np.isfinite(v) for v in asdict(pose).values()) or not np.isfinite(mating_x):
            raise ValueError("Initial pose must be finite")
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        self.socket_rotation = self.data.site_xmat[self.entry].reshape(3, 3).copy()
        self.socket_origin = self.data.site_xpos[self.entry].copy()
        self.initial_position = self.socket_origin + self.socket_rotation @ np.array(
            [mating_x, pose.offset_y_mm / 1000, pose.offset_z_mm / 1000])
        self.target_rotation = self.socket_rotation @ rotation(pose)
        quaternion = np.zeros(4)
        mujoco.mju_mat2Quat(quaternion, self.target_rotation.ravel())
        self.data.qpos[self.qadr:self.qadr + 3] = self.initial_position
        self.data.qpos[self.qadr + 3:self.qadr + 7] = quaternion
        mujoco.mj_forward(self.model, self.data)
        self.initial_grasp = self.data.site_xpos[self.grasp].copy()
        self.travel = 0.0
        self.active_time = 0.0
        self.max_travel = max(0.0, -mating_x + self.config["target_overtravel_m"])
        self.hold = 0.0
        self.outcome = None
        self.history = deque()
        self.first_contact = None
        self.peak_force = 0.0
        self.peak_penetration = 0.0
        self.max_depth = -math.inf
        self.last_wrench = np.zeros(6)
        self.pose = pose
        self.info = self.diagnostics()
        if self.info["max_penetration_m"] > 1e-8:
            self.outcome = "initial_state_rejection"

    def apply_holder(self):
        m, d, c = self.model, self.data, self.config
        target = self.initial_grasp + self.socket_rotation[:, 0] * self.travel
        position = d.site_xpos[self.grasp]
        current_rotation = d.site_xmat[self.grasp].reshape(3, 3)
        velocity = np.zeros(6)
        mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_SITE, self.grasp, velocity, 0)
        force = limited(c["translation_stiffness_n_m"] * (target - position)
                        - c["translation_damping_ns_m"] * velocity[3:], c["force_limit_n"])
        torque = limited(c["rotation_stiffness_nm_rad"] *
                         rotation_vector(self.target_rotation @ current_rotation.T)
                         - c["rotation_damping_nms_rad"] * velocity[:3], c["torque_limit_nm"])
        d.qfrc_applied[:] = 0
        # Apply the holder at the grasp, so MuJoCo accounts for its moment arm.
        mujoco.mj_applyFT(m, d, force, torque, position, self.plug, d.qfrc_applied)
        # Separate ideal gravity compensation at COM: no artificial grasp torque.
        mujoco.mj_applyFT(m, d, -m.body_mass[self.plug] * m.opt.gravity, np.zeros(3),
                         d.xipos[self.plug], self.plug, d.qfrc_applied)
        self.last_wrench = np.r_[force, torque]

    def step(self, advancing: bool = True) -> dict:
        if self.outcome:
            return self.info
        m, d, c = self.model, self.data, self.config
        dt = m.opt.timestep
        if advancing:
            self.active_time += dt
            self.travel = min(self.max_travel, self.travel + c["speed_m_s"] * dt)
        self.apply_holder()
        previous_time = d.time
        mujoco.mj_step(m, d)
        # mj_step leaves some derived quantities at the previous state; synchronize
        # transforms AND contact forces for consistent trace/success timestamps.
        mujoco.mj_forward(m, d)
        self.info = info = self.diagnostics()
        self.peak_force = max(self.peak_force, info["contact_force_n"])
        self.peak_penetration = max(self.peak_penetration, info["max_penetration_m"])
        self.max_depth = max(self.max_depth, info["insertion_depth_m"])
        if info["contact_count"] and self.first_contact is None:
            self.first_contact = dict(info)
        self.hold = self.hold + dt if info["valid_pose"] else 0.0
        if (not np.isfinite(d.qpos).all() or not np.isfinite(d.qvel).all()
                or d.time <= previous_time or any(w.number for w in d.warning)
                or info["speed_m_s"] > 1 or info["angular_speed_rad_s"] > 50):
            self.outcome = "numerical_failure"
        elif info["max_penetration_m"] > c["penetration_limit_m"]:
            self.outcome = "penetration_abort"
        elif info["contact_force_n"] > c["contact_abort_n"]:
            self.outcome = "force_limit_abort"
        elif info["other_contact_force_n"] > 0.01:
            self.outcome = "prohibited_contact"
        elif self.hold >= c["hold_s"]:
            self.outcome = "success"
        if advancing:
            self.history.append((float(d.time), info["insertion_depth_m"], info["contact_force_n"]))
            while len(self.history) > 1 and self.history[1][0] <= d.time - c["jam_window_s"]:
                self.history.popleft()
            if (self.outcome is None and not info["valid_pose"] and self.history
                    and d.time - self.history[0][0] >= c["jam_window_s"]
                    and max(x[1] for x in self.history) - min(x[1] for x in self.history)
                    < c["jam_progress_m"]
                    and min(x[2] for x in self.history) > c["jam_contact_n"]):
                self.outcome = "jam"
            if self.outcome is None and self.active_time >= c["duration_s"]:
                self.outcome = "timeout"
        else:
            self.history.clear()
        return info

    def summary(self) -> dict:
        return {
            **asdict(self.pose), "success": self.outcome == "success",
            "outcome": self.outcome or "running", "final": self.info,
            "max_insertion_depth_m": self.max_depth if np.isfinite(self.max_depth)
            else self.info["insertion_depth_m"],
            "max_contact_force_n": self.peak_force,
            "max_penetration_m": self.peak_penetration,
            "hold_s": self.hold, "first_contact": self.first_contact,
            "warning_counts": [int(w.number) for w in self.data.warning],
        }

    def metadata(self) -> dict:
        paths = sorted((ROOT / "assets/connector").glob("*.xml"))
        hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
        if self.spec.is_legacy:
            version = "two_blade_v2_leadin" if self.leadin else VERSION
        else:
            version = self.spec.name + ("_leadin" if self.leadin else "_straight")
        return {"scene_version": version, "mujoco_version": mujoco.__version__,
                "model_hashes": hashes, "config": self.config,
                "connector": {"plug": self.spec.plug.name, "socket": self.spec.socket.name,
                              "spec": self.spec.to_dict()},
                "timestep_s": float(self.model.opt.timestep),
                "pose": asdict(self.pose), "seed": 0,
                "randomization": "none; deterministic initial conditions",
                "controller": "bounded grasp-frame spring damper plus ideal COM gravity compensation"}


def trace_row(sim: ConnectorSimulation) -> dict:
    return {**sim.info, "target_travel_m": sim.travel, "hold_s": sim.hold,
            **{f"holder_{name}": float(value) for name, value in
               zip(("fx_n", "fy_n", "fz_n", "tx_nm", "ty_nm", "tz_nm"), sim.last_wrench)}}


def run_trial(pose: Pose = Pose(), *, config: dict | None = None,
              timestep: float | None = None, output: Path | None = None,
              leadin: bool = False, spec: ConnectorSpec | None = None) -> dict:
    sim = ConnectorSimulation(config, timestep, leadin, spec=spec)
    sim.reset(pose)
    rows = [trace_row(sim)]
    next_sample = sim.config["trace_interval_s"]
    while sim.outcome is None:
        sim.step()
        if sim.data.time >= next_sample or sim.outcome:
            rows.append(trace_row(sim))
            next_sample += sim.config["trace_interval_s"]
    result = sim.summary()
    if output is not None:
        output.mkdir(parents=True, exist_ok=True)
        (output / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        (output / "metadata.json").write_text(json.dumps(sim.metadata(), indent=2), encoding="utf-8")
        with (output / "trace.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
    return result
