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
    """Read-only metrics bound to caller-owned model/data; never reset the robot."""
    def bind(self, model, data, config, allowed_grasp_bodies=()):
        self.model, self.data, self.config = model, data, config
        self.allowed_grasp_bodies = set(allowed_grasp_bodies)
        numeric_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_NUMERIC, "lead_length")
        self.lead_length = (float(model.numeric_data[model.numeric_adr[numeric_id]])
                            if numeric_id >= 0 else 0.0)
        self.plug = self.model.body("plug").id
        self.socket = self.model.body("socket").id
        self.grasp = self.model.site("plug_grasp_frame").id
        self.entry = self.model.site("socket_entry").id
        self.blades = [self.model.geom(n).id for n in ("blade_left", "blade_right")]
        self.slots = [self.model.site(n).id for n in ("slot_left_entry", "slot_right_entry")]
        self.tips = [self.model.site(n).id for n in ("plug_tip_left", "plug_tip_right")]
        # Read corridor bounds from the collision walls, so geometry remains the
        # source of truth for slot width/height rather than duplicated constants.
        middle = self.model.geom("socket_middle").id
        top = self.model.geom("socket_top").id
        self.slot_half = np.array([
            abs(self.model.site_pos[self.slots[0], 1]) - self.model.geom_size[middle, 1],
            self.model.geom_pos[top, 2] - self.model.geom_size[top, 2],
        ])
        self.signs = np.array(list(itertools.product((-1, 1), repeat=3)))

    def diagnostics(self) -> dict:
        m, d, c = self.model, self.data, self.config
        basis = d.site_xmat[self.entry].reshape(3, 3)
        origin = d.site_xpos[self.entry]
        relative = basis.T @ (d.xpos[self.plug] - origin)
        relative_rotation = basis.T @ d.xmat[self.plug].reshape(3, 3)
        # Equal blades allow half-turn roll with swapped slot assignment.
        candidates = [relative_rotation, relative_rotation @ np.diag([1, -1, -1])]
        angles = [np.linalg.norm(rotation_vector(r)) for r in candidates]
        swapped = int(np.argmin(angles))
        angle = float(np.degrees(min(angles)))
        depth = [float((basis.T @ (d.site_xpos[t] - origin))[0]) for t in self.tips]
        fits = True
        for index, geom in enumerate(self.blades):
            corners = (self.signs * m.geom_size[geom]) @ d.geom_xmat[geom].reshape(3, 3).T
            local = (corners + d.geom_xpos[geom] - origin) @ basis
            if self.lead_length > 0:
                local = clip_at_throat(local, self.signs, self.lead_length)
            slot = self.slots[1 - index if swapped else index]
            slot_center = basis.T @ (d.site_xpos[slot] - origin)
            fits &= bool(np.all(np.abs(local[:, 1:3] - slot_center[1:3]) <=
                                self.slot_half + c["fit_tolerance_m"]))
        pair_force = 0.0
        grasp_force = 0.0
        grasp_penetration = 0.0
        other_force = 0.0
        penetration = 0.0
        count = 0
        blade_forces = [0.0, 0.0]
        contact_wrench = np.zeros(6)
        for i in range(d.ncon):
            contact = d.contact[i]
            bodies = [int(m.geom_bodyid[g]) for g in contact.geom]
            if self.plug not in bodies:
                continue
            mujoco.mj_contactForce(m, d, i, contact_wrench)
            magnitude = float(np.linalg.norm(contact_wrench[:3]))
            if any(b in self.allowed_grasp_bodies for b in bodies):
                grasp_force += magnitude
                grasp_penetration = max(grasp_penetration, -float(contact.dist))
                continue
            penetration = max(penetration, -float(contact.dist))
            if self.socket in bodies:
                pair_force += magnitude
                count += 1
                for j, blade in enumerate(self.blades):
                    if blade in contact.geom:
                        blade_forces[j] += magnitude
            else:
                other_force += magnitude
        velocity = np.zeros(6)
        mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_SITE, self.grasp, velocity, 0)
        face_gap = -float(relative[0])
        relative_quaternion = np.zeros(4)
        mujoco.mju_mat2Quat(relative_quaternion, relative_rotation.ravel())
        valid = (min(depth) >= c["success_depth_m"] and fits
                 and -c["penetration_limit_m"] <= face_gap <= c["success_face_gap_m"]
                 and angle < c["success_angle_deg"]
                 and penetration <= c["penetration_limit_m"]
                 and pair_force < c["contact_abort_n"] and other_force < 0.01)
        return {
            "time_s": float(d.time), "left_depth_m": depth[0], "right_depth_m": depth[1],
            "insertion_depth_m": min(depth), "face_gap_m": face_gap,
            "offset_y_m": float(relative[1]), "offset_z_m": float(relative[2]),
            "orientation_error_deg": angle, "blades_fit": fits,
            "contact_force_n": pair_force, "other_contact_force_n": other_force,
            "left_contact_force_n": blade_forces[0], "right_contact_force_n": blade_forces[1],
            "contact_count": count, "max_penetration_m": penetration,
            "grasp_contact_force_n": grasp_force, "grasp_penetration_m": grasp_penetration,
            "speed_m_s": float(np.linalg.norm(velocity[3:])),
            "angular_speed_rad_s": float(np.linalg.norm(velocity[:3])),
            "valid_pose": bool(valid),
            **{f"relative_q{axis}": float(value) for axis, value in
               zip("wxyz", relative_quaternion)},
            **{f"velocity_{axis}": float(value) for axis, value in
               zip(("wx_rad_s", "wy_rad_s", "wz_rad_s", "x_m_s", "y_m_s", "z_m_s"), velocity)},
        }


class ConnectorSimulation(ConnectorMetrics):
    def __init__(self, config: dict | None = None, timestep: float | None = None, leadin: bool = False):
        self.config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        if config:
            unknown = config.keys() - self.config.keys()
            if unknown:
                raise ValueError(f"Unknown config fields: {unknown}")
            self.config.update(config)
        if any(not np.isfinite(v) or v <= 0 for v in self.config.values()):
            raise ValueError("All holder configuration values must be finite and positive")
        from connector.geometry import connector_xml
        self.leadin = leadin
        self.model = mujoco.MjModel.from_xml_string(connector_xml(leadin))
        if timestep is not None:
            if not np.isfinite(timestep) or timestep <= 0:
                raise ValueError("timestep must be finite and positive")
            self.model.opt.timestep = timestep
        self.data = mujoco.MjData(self.model)
        self.bind(self.model, self.data, self.config)
        self.qadr = self.model.jnt_qposadr[self.model.joint("plug_free").id]
        self.reset()

    def reset(self, pose: Pose = Pose(), mating_x: float = -0.022):
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
        return {"scene_version": "two_blade_v2_leadin" if self.leadin else VERSION, "mujoco_version": mujoco.__version__,
                "model_hashes": hashes, "config": self.config,
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
              leadin: bool = False) -> dict:
    sim = ConnectorSimulation(config, timestep, leadin)
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
