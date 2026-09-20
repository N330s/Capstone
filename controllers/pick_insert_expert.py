"""Scripted expert for locate -> grasp -> transport -> insert.

Outputs the SAME joint targets exposed to a policy: np.r_[7 arm targets, grip].

Motion is split into speed profiles. Free-space transit moves in ~10 mm
Cartesian steps; approach moves in ~3 mm steps; anything that can touch the
socket keeps the original 1 mm / 0.01 rad clamp. Using the contact clamp for a
0.4 m reach is what produced pregrasp timeouts.

Free-space motion always goes over the top: rise to SAFE_HEIGHT_M above the
table, translate, then descend. Straight-line Cartesian interpolation from the
home pose to a plug lying on the table drags the wrist through the table and the
socket, which is what produced robot_collision.

Two layers of information, deliberately separated:

  * ``Detector`` stands in for perception. It is latched once per attempt at the
    end of the survey phase, with a per-episode calibration bias plus per-call
    noise. Reach and grasp run off that latched estimate, so the demonstrated
    behaviour is something a camera-only policy can reproduce.
  * Privileged closed-loop pose feedback is used only once the plug is held and
    contact matters, exactly as the old insertion expert did.

The grasp pose is read from the model: the plug_grasp_frame site's placement
relative to the plug body (CAD knowledge, not state knowledge). The EULER/OFFSET
constants below are only a fallback for scenes without that site.

Env contract beyond the original expert: env.reset_options, info["grip_travel_m"],
info["grip_force_n"], config["grip_open_travel_m"].
"""
import numpy as np
import mujoco
from connector.simulation import rotation_vector

GRASP_IN_PLUG_EULER_DEG = (0.0, 90.0, 0.0)
GRASP_OFFSET_IN_PLUG_M = (-0.010, 0.0, 0.0)

SAFE_HEIGHT_M = 0.22          # tool height above the table for free-space transit
APPROACH_HEIGHT_M = 0.09      # hover height directly above the grasp pose
STANDOFF_M = 0.055            # pre-insert distance back along the mating axis

FREE = dict(gain=1.0, pos_clip=0.010, rot_clip=0.05)
APPROACH = dict(gain=0.5, pos_clip=0.003, rot_clip=0.02)
CONTACT = dict(gain=0.3, pos_clip=0.001, rot_clip=0.01)
SETTLE = dict(gain=0.15, pos_clip=0.0006, rot_clip=0.006)

PHASE_TIMEOUT_S = {"survey": 1.5, "transit_pick": 10.0, "pregrasp": 6.0, "descend": 4.0,
                   "close": 1.5, "lift": 5.0, "transit_place": 12.0, "approach": 8.0,
                   "align": 6.0, "insert": 10.0, "withdraw": 3.0, "regrasp": 6.0, "hold": 1.0}


def euler_xyz(deg):
    rx, ry, rz = np.radians(deg)
    cx, sx, cy, sy, cz, sz = np.cos(rx), np.sin(rx), np.cos(ry), np.sin(ry), np.cos(rz), np.sin(rz)
    return (np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
            @ np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
            @ np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]]))


def yaw_mat(rad):
    c, s = np.cos(rad), np.sin(rad)
    return np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])


class Detector:
    """Object localisation with calibration bias + measurement noise."""

    def __init__(self, env, rng=None, *, mode="noisy",
                 bias_pos_m=0.0015, bias_yaw_deg=1.2,
                 noise_pos_m=0.0008, noise_yaw_deg=0.4):
        self.env, self.mode = env, mode
        self.rng = rng if rng is not None else np.random.default_rng(0)
        self.noise_pos_m, self.noise_yaw_deg = noise_pos_m, noise_yaw_deg
        if mode == "privileged":
            self.bias_pos, self.bias_yaw = np.zeros(3), 0.0
        else:
            self.bias_pos = self.rng.normal(0.0, bias_pos_m, 3)
            self.bias_yaw = float(self.rng.normal(0.0, bias_yaw_deg))

    def truth(self):
        e, m, d = self.env, self.env.model, self.env.data
        return {"plug_pos": d.xpos[e.plug].copy(),
                "plug_mat": d.xmat[e.plug].reshape(3, 3).copy(),
                "socket_pos": d.site_xpos[m.site("socket_entry").id].copy(),
                "socket_mat": d.site_xmat[m.site("socket_entry").id].reshape(3, 3).copy()}

    def sense(self):
        out = self.truth()
        if self.mode == "privileged":
            return out
        for key_p, key_m in (("plug_pos", "plug_mat"), ("socket_pos", "socket_mat")):
            out[key_p] = out[key_p] + self.bias_pos + self.rng.normal(0.0, self.noise_pos_m, 3)
            dyaw = np.radians(self.bias_yaw + self.rng.normal(0.0, self.noise_yaw_deg))
            out[key_m] = yaw_mat(dyaw) @ out[key_m]
        return out


class PickInsertExpert:
    """Phase machine. ``action()`` is called once per env step."""

    def __init__(self, env, *, rng=None, detector_mode="noisy",
                 probe_offset_y_m=0.0, max_retries=2, max_regrasps=1):
        self.env = env
        self.rng = rng if rng is not None else np.random.default_rng(0)
        self.detector = Detector(env, self.rng, mode=detector_mode)
        self.phase, self.phase_time = "survey", 0.0
        self.retries, self.regrasps = 0, 0
        self.max_retries, self.max_regrasps = max_retries, max_regrasps
        self.failure = None
        self.detection = None
        self.detection_error = None
        self.grasp_signature = None
        self.phase_log = []
        self.forward = -0.022
        self.last_depth, self.stall_time = -1.0, 0.0
        self.probe_offset_y_m = probe_offset_y_m
        self.table_z = float(getattr(env, "reset_options", {}).get("table_height_m", 0.0))
        self.safe_z = self.table_z + SAFE_HEIGHT_M
        self.grip = env.config["grip_open_travel_m"]
        self.grasp_rot, self.grasp_off = self._grasp_in_plug()
        self.jacp = np.zeros((3, env.model.nv))
        self.jacr = np.zeros((3, env.model.nv))

    # ---------------------------------------------------------------- helpers
    def _grasp_in_plug(self):
        m = self.env.model
        try:
            sid = m.site("plug_grasp_frame").id
        except KeyError:
            return euler_xyz(GRASP_IN_PLUG_EULER_DEG), np.array(GRASP_OFFSET_IN_PLUG_M)
        mat = np.zeros(9)
        mujoco.mju_quat2Mat(mat, m.site_quat[sid])
        return mat.reshape(3, 3).copy(), m.site_pos[sid].copy()
        # return euler_xyz((0.0, 180.0, 0.0)), np.array([-0.010, 0.0, 0.020])

    def _goto(self, phase):
        self.phase_log.append({"phase": self.phase, "duration_s": round(self.phase_time, 3)})
        self.phase, self.phase_time = phase, 0.0

    def _grasp_frame(self):
        pos = self.detection["plug_pos"] + self.detection["plug_mat"] @ self.grasp_off
        return pos, self.detection["plug_mat"] @ self.grasp_rot

    def _socket_frame(self):
        if self.phase in ("approach", "align", "insert", "withdraw", "hold"):
            m, d = self.env.model, self.env.data          # privileged, contact phase
            sid = m.site("socket_entry").id
            return d.site_xpos[sid].copy(), d.site_xmat[sid].reshape(3, 3).copy()
        return self.detection["socket_pos"], self.detection["socket_mat"]

    def _overhead(self, pos):
        """Same x/y, raised to the transit height."""
        return np.array([pos[0], pos[1], self.safe_z])

    def _holding(self):
        info = self.env.info
        return info["grip_force_n"] > 0.5 and info.get("grasp_slip_m", 0.) < 0.01

    def _slipped(self):
        if self.grasp_signature is None:
            return False
        e, d = self.env, self.env.data
        now = d.xmat[e.plug].reshape(3, 3).T @ (d.site_xpos[e.grasp_site] - d.xpos[e.plug])
        return bool(np.linalg.norm(now - self.grasp_signature) > 0.012)

    def _servo(self, pos, mat, frame="tool", *, gain, pos_clip, rot_clip):
        """Damped-least-squares increment on the joint target. Never touches qpos."""
        e, m, d = self.env, self.env.model, self.env.data
        if frame == "tool":
            cur_pos = d.site_xpos[e.grasp_site]
            cur_mat = d.site_xmat[e.grasp_site].reshape(3, 3)
        else:
            cur_pos, cur_mat = d.xpos[e.plug], d.xmat[e.plug].reshape(3, 3)
        residual = float(np.linalg.norm(pos - cur_pos))
        delta_pos = np.clip(pos - cur_pos, -pos_clip, pos_clip)
        delta_rot = np.clip(rotation_vector(mat @ cur_mat.T), -rot_clip, rot_clip)
        mujoco.mj_jacSite(m, d, self.jacp, self.jacr, e.grasp_site)
        jac = np.vstack([self.jacp[:, e.va["right"]], self.jacr[:, e.va["right"]]])
        change = jac.T @ np.linalg.solve(jac @ jac.T + np.eye(6) * 1e-5,
                                         np.r_[delta_pos, delta_rot])
        base = d.qpos[e.qa["right"]] if self.phase in ("withdraw", "regrasp") else e.target[:7]
        return np.r_[base + gain * change, self.grip], residual

    def _timed_out(self):
        return self.phase_time > PHASE_TIMEOUT_S.get(self.phase, 8.0)

    # ------------------------------------------------------------------ phases
    def action(self):
        e, d, info = self.env, self.env.data, self.env.info
        self.phase_time += e.dt

        if self.phase == "survey":
            # Hold still so the cameras get clean frames of the untouched scene,
            # then latch one detection and commit to it for the reach.
            self.grip = e.config["grip_open_travel_m"]
            if self.phase_time > PHASE_TIMEOUT_S["survey"] * 0.6:
                self.detection = self.detector.sense()
                truth = self.detector.truth()
                self.detection_error = {
                    "plug_pos_m": float(np.linalg.norm(self.detection["plug_pos"] - truth["plug_pos"])),
                    "socket_pos_m": float(np.linalg.norm(self.detection["socket_pos"] - truth["socket_pos"]))}
                self._goto("transit_pick")
            return np.r_[e.target[:7], self.grip]

        if self.phase == "transit_pick":
            # Up and over. Never cut a diagonal across the table.
            pos, mat = self._grasp_frame()
            action, residual = self._servo(self._overhead(pos), mat, **FREE)
            if residual < 0.015:
                self._goto("pregrasp")
            elif self._timed_out():
                self.failure = "transit_pick_timeout"
            return action

        if self.phase == "pregrasp":
            pos, mat = self._grasp_frame()
            action, residual = self._servo(pos + np.array([0, 0, APPROACH_HEIGHT_M]), mat, **APPROACH)
            if residual < 0.004:
                self._goto("descend")
            elif self._timed_out():
                self.failure = "pregrasp_timeout"
            return action

        if self.phase == "descend":
            pos, mat = self._grasp_frame()
            action, residual = self._servo(pos, mat, **SETTLE)
            if residual < 0.0025 or info["grip_force_n"] > 1.5:
                self._goto("close")
            elif self._timed_out():
                self.failure = "descend_timeout"
            return action

        if self.phase == "close":
            self.grip = e.config["grip_target_travel_m"]
            pos, mat = self._grasp_frame()
            action, _ = self._servo(pos, mat, **SETTLE)
            if self.phase_time > 0.4 and self._holding():
                self.grasp_signature = (d.xmat[e.plug].reshape(3, 3).T
                                        @ (d.site_xpos[e.grasp_site] - d.xpos[e.plug]))
                self._goto("lift")
            elif self._timed_out():
                if self.regrasps < self.max_regrasps:
                    self._goto("regrasp")
                else:
                    self.failure = "grasp_failed"
            return action

        if self.phase == "lift":
            pos, mat = self._grasp_frame()
            action, residual = self._servo(self._overhead(pos), mat, **FREE)
            if self._slipped():
                self._goto("regrasp")
            elif residual < 0.02:
                self._goto("transit_place")
            elif self._timed_out():
                self.failure = "lift_timeout"
            return action

        if self.phase == "regrasp":
            # Open, retreat to transit height, re-sense, try the grasp again.
            self.grip = e.config["grip_open_travel_m"]
            pos, mat = self._grasp_frame()
            action, residual = self._servo(self._overhead(pos), mat, **FREE)
            if self.phase_time > 0.5 and residual < 0.02:
                self.regrasps += 1
                self.grasp_signature = None
                if self.regrasps > self.max_regrasps:
                    self.failure = "regrasp_budget_exhausted"
                else:
                    self.detection = self.detector.sense()
                    self._goto("transit_pick")
            elif self._timed_out():
                self.failure = "regrasp_timeout"
            return action

        if self.phase == "transit_place":
            # Carry the plug at transit height to a point above the standoff.
            entry, basis = self._socket_frame()
            standoff = entry + basis @ np.array([-STANDOFF_M, 0.0, 0.0])
            action, residual = self._servo(self._overhead(standoff), basis, frame="plug", **FREE)
            if self._slipped():
                self._goto("regrasp")
            elif residual < 0.015:
                self._goto("approach")
            elif self._timed_out():
                self.failure = "transit_place_timeout"
            return action

        if self.phase == "approach":
            entry, basis = self._socket_frame()
            standoff = entry + basis @ np.array([-STANDOFF_M, 0.0, 0.0])
            action, residual = self._servo(standoff, basis, frame="plug", **APPROACH)
            if self._slipped():
                self._goto("regrasp")
            elif residual < 0.005:
                self.forward = -0.022
                self._goto("align")
            elif self._timed_out():
                self.failure = "approach_timeout"
            return action

        # ---- from here down this is the original insertion machine: privileged
        # closed-loop pose feedback on the held plug, 1 mm clamp, stall retries.
        entry, basis = self._socket_frame()
        if self.phase == "align":
            self.forward = -0.022
            if (self.phase_time > 0.5 and abs(info["offset_y_m"]) < 1e-4
                    and abs(info["offset_z_m"]) < 1e-4 and info["orientation_error_deg"] < 0.2):
                self._goto("insert")
            elif self._timed_out():
                self.failure = "align_timeout"
        elif self.phase == "insert":
            self.forward = min(0.00015, self.forward + 0.008 * e.dt)
            depth = info["insertion_depth_m"]
            self.stall_time = (self.stall_time + e.dt
                               if abs(depth - self.last_depth) < 1e-5 and info["contact_force_n"] > 0.8
                               else 0.0)
            self.last_depth = depth
            if ((self.stall_time > 0.4 or info.get("interval_peak_contact_force_n", 0) > 0.3)
                    and not info["valid_pose"] and self.retries < self.max_retries):
                self._goto("withdraw")
                self.retries += 1
                self.forward = min(self.forward, -0.018)
            if info["valid_pose"]:
                self.phase = "hold"
            elif self._timed_out():
                self.failure = "insert_timeout"
        elif self.phase == "withdraw":
            self.forward = max(-0.022, self.forward - 0.015 * e.dt)
            if self.forward <= -0.022:
                self.stall_time = 0.0
                self._goto("align")
        elif self.phase == "hold":
            self.forward = 0.00003

        desired_plug = entry + basis @ np.array([self.forward, 0, 0])
        # Evaluation-only fault injection: intentionally probe off-axis, then
        # withdraw/re-align. Do not label the initial faulty probe as expert BC.
        if self.phase == "insert" and self.retries == 0:
            desired_plug = desired_plug + basis[:, 1] * self.probe_offset_y_m
        action, _ = self._servo(desired_plug, basis, frame="plug", **CONTACT)
        return action

    def diagnostics(self):
        return {"phase": self.phase, "failure": self.failure,
                "retries": self.retries, "regrasps": self.regrasps,
                "detector_mode": self.detector.mode,
                "detection_error": self.detection_error,
                "phase_log": self.phase_log + [{"phase": self.phase,
                                                "duration_s": round(self.phase_time, 3)}]}