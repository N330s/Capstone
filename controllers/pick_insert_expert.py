"""Scripted expert for locate -> grasp -> transport -> insert.

Outputs the SAME joint targets exposed to a policy: np.r_[7 arm targets, grip],
i.e. absolute right joint1..7 targets (rad) and finger travel (m), every step.

Two layers of information, deliberately separated:

  * ``Detector`` stands in for perception. It is latched once per attempt at the
    end of the survey phase, with a per-episode calibration bias plus per-call
    noise. Reach and grasp run off that latched estimate, so the demonstrated
    behaviour is something a camera-only policy can reproduce.
  * Privileged closed-loop pose feedback is used only once the plug is held and
    contact matters, exactly as the old insertion expert did.

Table pick (survey -> transit_pick -> descend -> close -> lift)
---------------------------------------------------------------
The pick is planned once, at the end of the survey, in joint space, on a private
MjData (``GraspPlanner``); it never touches live state. It replaces the old
Cartesian servo, which could not work: the settle profile crawled at 4.5 mm/s
into a 4 s timeout, phase exits ignored orientation, and the integrated joint
target pinned against joint limits.

  * The grasp is top-down on the middle of the housing (``plug_grasp_frame``),
    GRASP_HEIGHT_OFFSET_M above its mid-plane, the validated table_pickup recipe.
    Candidates: both hand yaws that pinch the housing sides (flip 0/1, 180 deg
    apart about the approach axis) times a small set of approach tilts about the
    pinch axis (tilting about that axis keeps both pads parallel to the housing
    faces and buys a lot of reach and joint-limit margin).
  * Each candidate is IK-solved with restarts and a null-space push away from
    joint limits, for the grasp, a straight descend line from the pregrasp
    (PREGRASP_CLEARANCE_M back along the approach axis) and a straight vertical
    lift. It is rejected on joint-limit margin, fingertip-to-table clearance or
    any penetrating contact. When the socket pose is known, candidates whose
    in-hand plug pose also makes the insertion poses reachable are preferred
    (the connector is symmetric under a half-turn about its mating axis).
  * Execution is smooth joint-space interpolation (smoothstep over a piecewise
    linear path) emitted as absolute targets, then a short hold. Nothing waits
    on a Cartesian residual, so there is no timeout path: an unreachable plug is
    reported as ``ik_infeasible`` before the arm moves.
  * Close ramps the finger command at FINGER_CLOSE_RATE_M_S to CLOSE_TRAVEL_M
    and exits only once the env has latched a physical grasp (both pads on the
    housing, pads still, command steady and squeezing). grip_force_n is a
    position-servo output, not a contact force, and is not used as a signal.
    A detector position error can let one pad touch well before the other;
    closing on is usually fine (the plug slides/rotates a little on the table
    and the second pad catches up), but if that one pad starts shoving instead
    of settling (ONE_SIDED_CLOSE_GRACE_S or GRASP_PENETRATION_BACKOFF_M), close
    backs off instead of wedging the plug into the table, and folds how far off
    that pad's own touch was into a correction (``_update_grasp_correction``,
    robot-side telemetry only -- never the plug's pose) for the next attempt.
  * The lift is checked (height gain, both pads, drift vs the latched grasp)
    before transport starts; a failed close or lift opens, retreats straight up
    and re-plans off the same latched detection plus any accumulated
    correction (max_regrasps).

Env contract beyond the original expert: env.reset_options, info["grip_travel_m"],
info["plug_finger_contacts"], info["grasp_established"], info["grasp_slip_m"],
config["grip_open_travel_m"].
"""
import numpy as np
import mujoco
from connector.simulation import rotation_vector
from controllers.place_insert import PlaceInsert, PLACE_PHASES

# ---- table pick (tabletop mode only) --------------------------------------
PLUG_HALF_HEIGHT_M = 0.008            # housing half-height: plug centre above the table
GRASP_POINT_IN_PLUG_M = np.array([-0.016, 0.0, 0.0])   # plug_grasp_frame, housing middle
GRASP_HEIGHT_OFFSET_M = 0.004         # tool site above the housing mid-plane (validated 3-4 mm)
MIN_FINGER_TABLE_CLEARANCE_M = 0.0015
PREGRASP_CLEARANCE_M = 0.07           # back-off along the approach axis
LIFT_HEIGHT_M = 0.10
GRASP_TILTS_DEG = (0., -15., 15., -30., 30.)
PRESHAPE_TRAVEL_M = 0.030             # pads ~16 mm clear of the housing on each side
CLOSE_TRAVEL_M = 0.006                # validated table_pickup closed command
HELD_TRAVEL_M = 0.0137                # measured pad travel when closed on the housing
HELD_TRAVEL_TOLERANCE_M = 0.0008      # wider => pads on an edge/corner (misaligned pinch)
FINGER_CLOSE_RATE_M_S = 0.01          # validated table_pickup finger rate
ONE_SIDED_CLOSE_GRACE_S = 0.3         # one pad only, this long: back off before it wedges the plug
GRASP_PENETRATION_BACKOFF_M = 0.0006  # one pad only and already squeezing this hard: back off now
MAX_GRASP_CORRECTION_M = 0.006        # cap on the accumulated one-sided-contact correction
MIN_JOINT_MARGIN_RAD = 0.05
GOOD_JOINT_MARGIN_RAD = 0.15
TRANSIT_SPEED_RAD_S = 0.3             # smoothstep peak = 1.5x this, env cap is 0.5
DESCEND_SPEED_RAD_S = 0.18
LIFT_SPEED_RAD_S = 0.15
ARRIVE_JOINT_ERROR_RAD = 0.004
MIN_LIFT_RISE_M = 0.05
LIFT_DRIFT_LIMIT_M = 0.001
PICK_TIMEOUT_S = {"arrive": 1.5, "close": 6.0}
PICK_PATH_PHASES = ("transit_pick", "descend", "lift", "retreat")

STANDOFF_M = 0.055            # pre-insert distance back along the mating axis

PHASE_TIMEOUT_S = {"survey": 1.5}


def euler_xyz(deg):
    rx, ry, rz = np.radians(deg)
    cx, sx, cy, sy, cz, sz = np.cos(rx), np.sin(rx), np.cos(ry), np.sin(ry), np.cos(rz), np.sin(rz)
    return (np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
            @ np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
            @ np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]]))


def yaw_mat(rad):
    c, s = np.cos(rad), np.sin(rad)
    return np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])


def rot_x(deg):
    a = np.radians(deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([[1., 0., 0.], [0., c, -s], [0., s, c]])


def rot_y(deg):
    a = np.radians(deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0., s], [0., 1., 0.], [-s, 0., c]])


TOP_DOWN_IN_PLUG = rot_y(90.)   # tool +x (approach) along plug -z, pinch (tool y) along plug y
HALF_TURN = np.diag([1., -1., -1.])


def grasp_rotation_in_plug(flip, tilt_deg):
    """Tool orientation in the plug frame for one grasp candidate."""
    return TOP_DOWN_IN_PLUG @ rot_x(180. * flip) @ rot_y(tilt_deg)


def flat_plug_pose(pos, mat, table_z):
    """A plug resting on the known table plane: keep x, y and yaw, snap z and tilt."""
    yaw = float(np.arctan2(mat[1, 0], mat[0, 0]))
    return np.array([pos[0], pos[1], table_z + PLUG_HALF_HEIGHT_M]), yaw_mat(yaw)


def upright_insert_frame(basis):
    """Equal blades make a half-turn about the mating axis equivalent (see
    ConnectorMetrics). Use the equivalent frame whose z points up, so a
    top-down grasp can reach it."""
    return basis @ HALF_TURN if basis[2, 2] < 0 else basis


def smoothstep(x):
    x = min(1., max(0., x))
    return x * x * (3. - 2. * x)


class JointPath:
    """Piecewise-linear joint path with one smoothstep time law and a final hold."""

    def __init__(self, start, waypoints, *, speed, min_duration, hold):
        self.points = [np.asarray(start, dtype=float)] + [np.asarray(q, dtype=float) for q in waypoints]
        lengths = [float(np.max(np.abs(b - a))) for a, b in zip(self.points, self.points[1:])]
        self.cum = np.r_[0., np.cumsum(lengths)]
        self.duration = max(min_duration, 1.5 * self.cum[-1] / speed)
        self.hold, self.t = hold, 0.

    @property
    def goal(self):
        return self.points[-1]

    @property
    def done(self):
        return self.t >= self.duration + self.hold

    def step(self, dt):
        self.t += dt
        s = smoothstep(self.t / self.duration) * self.cum[-1]
        if self.cum[-1] <= 0.:
            return self.goal.copy()
        i = int(np.clip(np.searchsorted(self.cum, s, side="right") - 1, 0, len(self.points) - 2))
        span = self.cum[i + 1] - self.cum[i]
        frac = 0. if span <= 0 else (s - self.cum[i]) / span
        return self.points[i] + (self.points[i + 1] - self.points[i]) * min(1., frac)


class GraspPlanner:
    """Kinematic pick planning on a private MjData. Never touches live state.

    Deterministic: fixed seeds, no dependence on wall clock or live data beyond
    the state passed to ``set_state``.
    """

    def __init__(self, model, table_z, *, seed=7):
        m = self.m = model
        self.d = mujoco.MjData(m)
        self.table_z = float(table_z)
        joints = [m.joint(f"openarm_right_joint{i}").id for i in range(1, 8)]
        self.qa = np.array([m.jnt_qposadr[j] for j in joints])
        self.va = np.array([m.jnt_dofadr[j] for j in joints])
        self.lo = m.jnt_range[joints, 0].copy()
        self.hi = m.jnt_range[joints, 1].copy()
        self.fqa = np.array([m.jnt_qposadr[m.joint(f"openarm_right_finger_joint{i}").id] for i in (1, 2)])
        self.site = m.site("robot_grasp").id
        self.plug = m.body("plug").id
        self.plug_qadr = int(m.jnt_qposadr[m.joint("plug_free").id])
        self.right_bodies = {i for i in range(m.nbody) if m.body(i).name.startswith("openarm_right")}
        self.fingers = {m.body(f"openarm_right_{s}_finger").id for s in ("right", "left")}
        hand = m.body("openarm_right_hand").id
        self.distal_geoms = [g for g in range(m.ngeom)
                             if int(m.geom_bodyid[g]) in (self.fingers | {hand})
                             and m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH
                             and (m.geom_contype[g] or m.geom_conaffinity[g])]
        self.finger_geoms = [g for g in self.distal_geoms if int(m.geom_bodyid[g]) in self.fingers]
        self.verts = {}
        for g in self.distal_geoms:
            mesh = int(m.geom_dataid[g])
            a, n = int(m.mesh_vertadr[mesh]), int(m.mesh_vertnum[mesh])
            self.verts[g] = m.mesh_vert[a:a + n].astype(float).copy()
        self.jp = np.zeros((3, m.nv))
        self.jr = np.zeros((3, m.nv))
        rng = np.random.default_rng(seed)
        self.home = np.array([0., 0., 0., np.pi / 2, 0., 0., 0.])
        # Known top-down grasp postures (flip 0, tilt 0/-15/-30 near the
        # validated plug position), then fixed pseudo-random restarts.
        self.seeds = [self.home,
                      np.array([0.86, 0.40, -0.84, 0.75, 0.90, -0.19, -1.34]),
                      np.array([0.59, 0.50, -0.65, 1.14, 0.76, -0.28, -1.18]),
                      np.array([0.30, 0.45, -0.39, 1.35, 0.54, -0.24, -0.96])]
        self.seeds += [rng.uniform(self.lo, self.hi) for _ in range(6)]

    # ----------------------------------------------------------- kinematics
    def set_state(self, qpos):
        self.d.qpos[:] = qpos
        self.d.qvel[:] = 0.

    def place_plug(self, pos, mat):
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, np.ascontiguousarray(mat, dtype=float).ravel())
        self.d.qpos[self.plug_qadr:self.plug_qadr + 3] = pos
        self.d.qpos[self.plug_qadr + 3:self.plug_qadr + 7] = quat

    def fk(self, q, travel=None):
        self.d.qpos[self.qa] = q
        if travel is not None:
            self.d.qpos[self.fqa] = travel
        mujoco.mj_kinematics(self.m, self.d)
        mujoco.mj_comPos(self.m, self.d)
        return self.d.site_xpos[self.site].copy(), self.d.site_xmat[self.site].reshape(3, 3).copy()

    def margin(self, q):
        return float(np.min(np.r_[q - self.lo, self.hi - q]))

    def ik(self, pos, rot, seed, iters=250, width=0.25, alpha=0.03):
        """Damped least squares with a null-space push away from joint limits.

        Written for speed (it runs thousands of times per plan): preallocated
        buffers, one linear solve per iteration, early abandonment of seeds
        that are stuck against a limit."""
        m, d, lo, hi = self.m, self.d, self.lo + 1e-4, self.hi - 1e-4
        q = np.minimum(np.maximum(np.asarray(seed, dtype=float), self.lo + 1e-3), self.hi - 1e-3)
        e, rhs, J = np.empty(6), np.empty((6, 2)), np.empty((6, 7))
        quat, damp = np.empty(4), 1e-4 * np.eye(6)
        settled = polish = 0
        err_p = err_r = np.inf
        for it in range(iters):
            d.qpos[self.qa] = q
            mujoco.mj_kinematics(m, d)
            mujoco.mj_comPos(m, d)
            e[:3] = pos - d.site_xpos[self.site]
            mujoco.mju_mat2Quat(quat, (rot @ d.site_xmat[self.site].reshape(3, 3).T).ravel())
            if quat[0] < 0:
                quat *= -1.
            n = float(np.sqrt(quat[1:] @ quat[1:]))
            e[3:] = 2. * np.arctan2(n, quat[0]) * quat[1:] / n if n > 1e-12 else 0.
            err_p, err_r = float(np.sqrt(e[:3] @ e[:3])), float(np.sqrt(e[3:] @ e[3:]))
            if (it == 40 and err_p > 0.03) or (it == 90 and err_p > 0.003):
                break                      # stuck against limits; let another seed try
            if err_p < 5e-5 and err_r < 5e-4:
                settled += 1
                polish += 1
                if settled > 5 and (polish > 60 or float(np.max(np.abs(step))) < 2e-4):
                    break
            else:
                settled = 0
            mujoco.mj_jacSite(m, d, self.jp, self.jr, self.site)
            J[:3], J[3:] = self.jp[:, self.va], self.jr[:, self.va]
            grad = alpha * (np.exp((lo - q) / width) - np.exp((q - hi) / width)) / width
            rhs[:3, 0] = np.minimum(np.maximum(e[:3], -.05), .05)
            rhs[3:, 0] = np.minimum(np.maximum(e[3:], -.2), .2)
            rhs[:, 1] = J @ grad
            sol = np.linalg.solve(J @ J.T + damp, rhs)
            step = J.T @ (sol[:, 0] - sol[:, 1]) + grad
            q = np.minimum(np.maximum(q + np.minimum(np.maximum(step, -.1), .1), lo), hi)
        p, R = self.fk(q)
        ok = np.linalg.norm(pos - p) < 2e-4 and np.linalg.norm(rotation_vector(rot @ R.T)) < 3e-3
        return q, bool(ok)

    def solve(self, pos, rot, seeds, good=GOOD_JOINT_MARGIN_RAD):
        best = None
        for s in seeds:
            if s is None:
                continue
            q, ok = self.ik(pos, rot, s)
            if ok and (best is None or self.margin(q) > self.margin(best)):
                best = q
                if self.margin(q) >= good:
                    break
        return best

    # ------------------------------------------------------------ checks
    def distal_clearance(self, q, travel, geoms=None):
        """Lowest hand/finger collision-mesh vertex above the table plane."""
        self.fk(q, travel)
        low = np.inf
        for g in (geoms or self.distal_geoms):
            v = self.d.geom_xpos[g] + self.verts[g] @ self.d.geom_xmat[g].reshape(3, 3).T
            low = min(low, float(v[:, 2].min()))
        return low - self.table_z

    def contacts(self, q, travel, *, ignore_plug=False):
        """Penetrating contacts between the right arm and anything else."""
        self.fk(q, travel)
        mujoco.mj_collision(self.m, self.d)
        bad = []
        for i in range(self.d.ncon):
            c = self.d.contact[i]
            if c.dist >= 0:
                continue
            b1, b2 = int(self.m.geom_bodyid[c.geom1]), int(self.m.geom_bodyid[c.geom2])
            r1, r2 = b1 in self.right_bodies, b2 in self.right_bodies
            if r1 == r2:
                continue
            if ignore_plug and self.plug in (b1, b2):
                continue
            bad.append((self.m.geom(c.geom1).name, self.m.geom(c.geom2).name, float(c.dist)))
        return bad

    def line(self, q0, start, end, rot, step=0.01):
        """IK waypoints along a straight tool path, each seeded by the previous."""
        n = max(1, int(np.ceil(np.linalg.norm(end - start) / step)))
        qs, q = [], q0
        for k in range(1, n + 1):
            q, ok = self.ik(start + (end - start) * k / n, rot, q, iters=150)
            if not ok:
                return None
            qs.append(q)
        return qs

    def segment_clear(self, q0, q1, travel, *, min_clearance=0.02, res=0.03):
        n = max(1, int(np.ceil(np.max(np.abs(q1 - q0)) / res)))
        for k in range(1, n + 1):
            q = q0 + (q1 - q0) * k / n
            if self.contacts(q, travel) or self.distal_clearance(q, travel) < min_clearance:
                return False
        return True

    # ------------------------------------------------------------ planning
    def insert_feasible(self, flip, tilt, insert, seed=None):
        """Can the hand reach the standoff and seated poses holding the plug
        with this grasp? ``insert`` = (socket entry position, upright basis)."""
        entry, basis = insert
        G = grasp_rotation_in_plug(flip, tilt)
        plug_in_tool = -G.T @ (GRASP_POINT_IN_PLUG_M + np.array([0., 0., GRASP_HEIGHT_OFFSET_M]))
        rot, q = basis @ G, seed
        saved = self.d.qpos[self.plug_qadr:self.plug_qadr + 7].copy()
        try:
            for forward in (-STANDOFF_M, 0.0):
                plug = entry + basis @ np.array([forward, 0., 0.])
                q = self.solve(plug - rot @ plug_in_tool, rot, ([q] if q is not None else []) + self.seeds,
                               good=MIN_JOINT_MARGIN_RAD)
                if q is None or self.margin(q) < MIN_JOINT_MARGIN_RAD:
                    return False
                # The hand holding the plug there must not hit the socket or table.
                self.place_plug(plug, basis)
                if self.contacts(q, HELD_TRAVEL_M, ignore_plug=True):
                    return False
            return True
        finally:
            self.d.qpos[self.plug_qadr:self.plug_qadr + 7] = saved

    @staticmethod
    def _flip_order(plug_mat, tilt):
        """Try first the hand yaw that points the tool's +z nearest the arm's
        comfortable heading. Ordering only: it never excludes a candidate."""
        def heading(flip):
            z = (plug_mat @ grasp_rotation_in_plug(flip, tilt))[:, 2]
            return abs(np.degrees(np.arctan2(np.sin(np.arctan2(z[1], z[0]) - np.radians(15.)),
                                             np.cos(np.arctan2(z[1], z[0]) - np.radians(15.)))))
        return sorted((0, 1), key=heading)

    def plan_grasp(self, plug_pos, plug_mat, q_now=None, *, insert=None, require_insert=False):
        """Best grasp plan for a plug lying flat, or None. Returns (plan, log).

        Tilt levels are tried least-tilted first and the search stops at the
        first level that yields a candidate comfortably inside the joint limits
        (and, when ``insert`` is given, able to reach the socket too)."""
        log, plans = [], []
        for tilt in GRASP_TILTS_DEG:
            for flip in self._flip_order(plug_mat, tilt):
                plan, why = self._candidate(plug_pos, plug_mat, flip, tilt, q_now)
                entry = {"flip": flip, "tilt_deg": tilt, "result": why}
                if plan is not None:
                    plan["insert_consistent"] = (insert is not None and
                                                 self.insert_feasible(flip, tilt, insert, plan["q_grasp"]))
                    entry["insert_consistent"] = plan["insert_consistent"]
                    plans.append(plan)
                log.append(entry)
            if any(p["margin"] >= GOOD_JOINT_MARGIN_RAD and (insert is None or p["insert_consistent"])
                   for p in plans):
                break
        if require_insert:
            plans = [p for p in plans if p["insert_consistent"]]
        if not plans:
            return None, log
        return max(plans, key=lambda p: (p["insert_consistent"], p["score"])), log

    def _candidate(self, plug_pos, plug_mat, flip, tilt, q_now):
        R = plug_mat @ grasp_rotation_in_plug(flip, tilt)
        approach = R[:, 0]
        grasp = plug_pos + plug_mat @ GRASP_POINT_IN_PLUG_M + np.array([0., 0., GRASP_HEIGHT_OFFSET_M])
        qg = self.solve(grasp, R, ([q_now] if q_now is not None else []) + self.seeds)
        if qg is None:
            return None, "grasp_ik"
        clearance = self.distal_clearance(qg, PRESHAPE_TRAVEL_M, self.finger_geoms)
        if clearance < MIN_FINGER_TABLE_CLEARANCE_M:
            raise_by = MIN_FINGER_TABLE_CLEARANCE_M - clearance + 1e-4
            if raise_by > 0.004:
                return None, "finger_table_clearance"
            grasp = grasp + np.array([0., 0., raise_by])
            qg, ok = self.ik(grasp, R, qg)
            clearance = self.distal_clearance(qg, PRESHAPE_TRAVEL_M, self.finger_geoms)
            if not ok or clearance < MIN_FINGER_TABLE_CLEARANCE_M:
                return None, "finger_table_clearance"
        pregrasp = grasp - approach * PREGRASP_CLEARANCE_M
        # Solve the descend line backwards from the grasp so both ends share a branch.
        up = self.line(qg, grasp, pregrasp, R)
        if up is None:
            return None, "descend_ik"
        descend = up[::-1][1:] + [qg]
        lift_pos = grasp + np.array([0., 0., LIFT_HEIGHT_M])
        lift = self.line(qg, grasp, lift_pos, R, step=0.02)
        if lift is None:
            return None, "lift_ik"
        margin = min(self.margin(q) for q in up + [qg] + lift)
        if margin < MIN_JOINT_MARGIN_RAD:
            return None, f"joint_margin_{margin:.3f}"
        for q in up + [qg]:
            if self.contacts(q, PRESHAPE_TRAVEL_M):
                return None, "descend_contact"
        for q in lift:
            if self.contacts(q, HELD_TRAVEL_M, ignore_plug=True):
                return None, "lift_contact"
        score = min(margin, GOOD_JOINT_MARGIN_RAD) + 0.1 * margin - 0.002 * abs(tilt)
        return {"flip": flip, "tilt_deg": tilt, "grasp_pos": grasp, "grasp_rot": R,
                "pregrasp_pos": pregrasp, "lift_pos": lift_pos, "q_pregrasp": up[-1],
                "descend": descend, "q_grasp": qg, "lift": lift, "margin": margin,
                "finger_clearance_m": clearance, "score": score}, "ok"

    def transit(self, q_now, plan, travel):
        """Collision-checked joint waypoints from q_now to the pregrasp, or None."""
        qp = plan["q_pregrasp"]
        if self.segment_clear(q_now, qp, travel):
            return [qp]
        # Go over the top: the pregrasp raised by up to 15 cm, same orientation.
        for rise in (0.08, 0.15):
            via, ok = self.ik(plan["pregrasp_pos"] + np.array([0., 0., rise]), plan["grasp_rot"], qp)
            if (ok and self.margin(via) >= MIN_JOINT_MARGIN_RAD
                    and self.segment_clear(q_now, via, travel)
                    and self.segment_clear(via, qp, travel, min_clearance=0.01)):
                return [via, qp]
        return None


def plan_table_pick(planner, qpos, plug_pos, plug_mat, q_now, *, insert=None, require_insert=False,
                    travel=None):
    """Shared by the expert and the scene sampler. Returns (plan or None, reason, log)."""
    planner.set_state(qpos)
    planner.place_plug(plug_pos, plug_mat)
    plan, log = planner.plan_grasp(plug_pos, plug_mat, q_now, insert=insert,
                                   require_insert=require_insert)
    if plan is None:
        reasons = sorted({entry["result"] for entry in log if entry["result"] != "ok"})
        return None, "ik_infeasible:" + ",".join(reasons or ["no_insert_consistent_grasp"]), log
    path = planner.transit(q_now, plan, planner.d.qpos[planner.fqa].max() if travel is None else travel)
    if path is None:
        return None, "transit_blocked", log
    plan["transit"] = path
    return plan, "ok", log


class PickFeasibility:
    """Deterministic scene-sampler check: can the expert plan this pick from the
    reset posture? Uses a private model (the live env is never touched)."""

    def __init__(self, table_z, *, require_insert=False):
        from envs.scene import build_model
        import json
        from pathlib import Path
        config = json.loads((Path(__file__).resolve().parents[1] / "configs/openarm_v1.json").read_text())
        self.model = build_model()
        self.planner = GraspPlanner(self.model, table_z)
        self.table_z = float(table_z)
        self.require_insert = require_insert
        m = self.model
        self.socket = int(m.site_bodyid[m.site("socket_entry").id])
        self.home = np.array(config["home_arm_rad"], dtype=float)
        self.open_travel = float(config["grip_open_travel_m"])
        self.qpos0 = m.qpos0.copy()
        for side in ("left", "right"):
            for i in range(1, 8):
                adr = m.jnt_qposadr[m.joint(f"openarm_{side}_joint{i}").id]
                self.qpos0[adr] = self.home[i - 1] if side == "right" else 0.
            for i in (1, 2):
                adr = m.jnt_qposadr[m.joint(f"openarm_{side}_finger_joint{i}").id]
                self.qpos0[adr] = self.open_travel if side == "right" else .025

    def __call__(self, options):
        from envs.openarm_insert import OpenArmInsertEnv
        m = self.model
        yaw_tilt = OpenArmInsertEnv._yaw_tilt_matrix
        basis = yaw_tilt(float(options.get("socket_yaw_deg", 0.)), float(options.get("socket_tilt_deg", 0.)))
        entry = np.asarray(options["socket_pos_m"], dtype=float)
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, basis.ravel())
        m.body_quat[self.socket], m.body_pos[self.socket] = quat, entry   # site sits at the body origin
        plug_pos, plug_mat = flat_plug_pose(np.asarray(options["plug_pos_m"], dtype=float),
                                            yaw_tilt(float(options.get("plug_yaw_deg", 0.)), 0.),
                                            self.table_z)
        insert = (entry, upright_insert_frame(basis)) if self.require_insert else None
        plan, reason, _ = plan_table_pick(self.planner, self.qpos0, plug_pos, plug_mat, self.home,
                                          insert=insert, require_insert=self.require_insert,
                                          travel=self.open_travel)
        if plan is None:
            return reason
        return None


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
                 probe_offset_y_m=0.0, max_retries=2, max_regrasps=1,
                 grasp_pose_source="top_down"):
        if grasp_pose_source != "top_down":
            raise ValueError("the table pick plans top-down grasps only; the side-on "
                             "plug_grasp_frame pose belongs to the grasped-mode scene")
        self.env = env
        self.rng = rng if rng is not None else np.random.default_rng(0)
        self.detector = Detector(env, self.rng, mode=detector_mode)
        self.phase, self.phase_time = "survey", 0.0
        self.retries, self.regrasps = 0, 0
        self.max_retries, self.max_regrasps = max_retries, max_regrasps
        self.failure = None
        self.failure_detail = None
        self.detection = None
        self.detection_error = None
        self.phase_log = []
        self.probe_offset_y_m = probe_offset_y_m
        self.table_z = float(getattr(env, "reset_options", {}).get("table_height_m", 0.0))
        self.grip = env.config["grip_open_travel_m"]
        self.planner = GraspPlanner(env.model, self.table_z)
        self.plan, self.path, self.arrive_next = None, None, None
        self.one_sided_s = 0.0        # time spent with exactly one pad on the plug during close
        self.one_sided_snapshot = None          # (travel, tool_mat, side) at first one-sided touch
        self.grasp_correction_m = np.zeros(3)   # accumulated one-sided-contact correction (world)
        self.plan_summaries, self.pick_report, self.pick_failures = [], None, []
        self.start_plug_z = float(env.data.xpos[env.plug][2])
        self.lift_drift, self.lift_angle, self.peak_support_n = 0., 0., 0.
        self.place = None             # controllers.place_insert.PlaceInsert, after the lift

    # ---------------------------------------------------------------- helpers
    def _goto(self, phase):
        self.phase_log.append({"phase": self.phase, "duration_s": round(self.phase_time, 3)})
        self.phase, self.phase_time = phase, 0.0

    def _fail(self, reason, detail=None):
        self.failure, self.failure_detail = reason, detail

    def _contact_snapshot(self):
        """(travel, tool_mat, side) for the one pad currently touching the plug,
        or None. All robot-side telemetry (which pad, its own joint travel, the
        measured wrist orientation) -- never the plug's pose."""
        e, m, d = self.env, self.env.model, self.env.data
        pads = set()
        for c in d.contact:
            bodies = (int(m.geom_bodyid[c.geom1]), int(m.geom_bodyid[c.geom2]))
            if e.plug in bodies:
                pads.update(b for b in bodies if b in e.fingers)
        if len(pads) != 1:
            return None
        pad = next(iter(pads))
        idx = 0 if pad == e.fingers[0] else 1
        travel = float(d.qpos[e.fqa["right"][idx]])
        tool_mat = d.site_xmat[e.grasp_site].reshape(3, 3).copy()
        local_y = float((tool_mat.T @ (d.xpos[pad] - d.site_xpos[e.grasp_site]))[1])
        return travel, tool_mat, (1.0 if local_y >= 0 else -1.0)

    def _update_grasp_correction(self):
        """Fold a one-sided close into a correction for the next attempt.

        Uses the snapshot taken at the first touch (before the lone pad has had
        a chance to drag the plug around), so this stays inside the
        detector-only contract for reach and grasp -- it never reads the
        plug's pose. If the housing were centred where the latched detection
        put it, both pads would touch at HELD_TRAVEL_M; a pad touching while
        still more open than that means the true housing is offset toward
        that pad, so nudge the next plan's assumed plug position the same
        way."""
        if self.one_sided_snapshot is None:
            return
        travel, tool_mat, side = self.one_sided_snapshot
        correction = self.grasp_correction_m + tool_mat[:, 1] * (side * (travel - HELD_TRAVEL_M))
        norm = float(np.linalg.norm(correction))
        if norm > MAX_GRASP_CORRECTION_M:
            correction *= MAX_GRASP_CORRECTION_M / norm
        self.grasp_correction_m = correction

    def _pick_timed_out(self):
        return self.phase_time > PICK_TIMEOUT_S[self.phase]

    # ------------------------------------------------------------ pick helpers
    def _q_now(self):
        return self.env.target[:7].copy()

    def _plan_pick(self):
        """Plan grasp + transit from the latched detection. Sets self.plan."""
        e = self.env
        plug_pos, plug_mat = flat_plug_pose(self.detection["plug_pos"] + self.grasp_correction_m,
                                            self.detection["plug_mat"], self.table_z)
        insert = (self.detection["socket_pos"], upright_insert_frame(self.detection["socket_mat"]))
        plan, reason, log = plan_table_pick(self.planner, e.data.qpos, plug_pos, plug_mat, self._q_now(),
                                            insert=insert, require_insert=True)
        summary = {"attempt": self.regrasps, "result": reason, "candidates": log}
        if plan is not None:
            summary.update(flip=plan["flip"], tilt_deg=plan["tilt_deg"],
                           joint_margin_rad=round(plan["margin"], 4),
                           finger_clearance_m=round(plan["finger_clearance_m"], 4),
                           insert_consistent=plan["insert_consistent"],
                           transit_waypoints=len(plan["transit"]))
        self.plan_summaries.append(summary)
        self.plan = plan
        return plan, reason

    def _start_path(self, phase, waypoints, *, speed, min_duration, hold):
        self.path = JointPath(self._q_now(), waypoints, speed=speed, min_duration=min_duration, hold=hold)
        self._goto(phase)

    def _begin_pick(self):
        plan, reason = self._plan_pick()
        if plan is None:
            self._fail("ik_infeasible" if reason.startswith("ik_infeasible") else reason, reason)
            return
        self.grip = PRESHAPE_TRAVEL_M
        self._start_path("transit_pick", plan["transit"], speed=TRANSIT_SPEED_RAD_S,
                         min_duration=2.0, hold=0.3)

    def _tracking_error(self):
        e = self.env
        return float(np.max(np.abs(self.path.goal - e.data.qpos[e.qa["right"]])))

    def _regrasp_or_fail(self, reason):
        self.pick_failures.append({"attempt": self.regrasps, "reason": reason, "phase": self.phase})
        if self.regrasps < self.max_regrasps:
            self._goto("regrasp")
        else:
            self._fail(reason)

    def _lift_check(self):
        e, info = self.env, self.env.info
        rise = float(e.data.xpos[e.plug][2]) - self.start_plug_z
        report = {"plug_rise_m": round(rise, 4), "finger_contacts": int(info.get("plug_finger_contacts", 0)),
                  "grasp_established": bool(info["grasp_established"]),
                  "max_drift_m": round(self.lift_drift, 6), "max_angle_deg": round(self.lift_angle, 3),
                  "peak_support_contact_n": round(self.peak_support_n, 3),
                  "drops": int(info.get("drops", 0)), "attempt": self.regrasps}
        report["firm"] = bool(rise >= MIN_LIFT_RISE_M and report["finger_contacts"] == 2
                              and report["grasp_established"] and self.lift_drift < LIFT_DRIFT_LIMIT_M)
        return report

    def _pick_action(self):
        e, info = self.env, self.env.info
        self.peak_support_n = max(self.peak_support_n, float(info.get("robot_support_contact_n", 0.)))

        if self.phase in PICK_PATH_PHASES:
            q = self.path.step(e.dt)
            if self.phase == "descend" and info["robot_support_contact_n"] > 5.0:
                self._fail("descend_table_contact")
            elif self.phase == "descend" and info["robot_unwanted_contact_n"] > 2.0:
                self._fail("descend_collision")
            elif self.phase == "lift":
                if info["grasp_established"]:
                    self.lift_drift = max(self.lift_drift, float(info["grasp_slip_m"]))
                    self.lift_angle = max(self.lift_angle, float(info["grasp_angle_deg"]))
                else:
                    self._regrasp_or_fail("lift_grasp_lost")
            if self.failure is None and self.phase in PICK_PATH_PHASES and self.path.done:
                self._path_done()
            return np.r_[q, self.grip]

        if self.phase == "arrive":
            # Path finished but the arm is still settling onto it.
            hold = self.path.goal.copy()
            if self._tracking_error() < ARRIVE_JOINT_ERROR_RAD:
                self._enter(self.arrive_next)
            elif self._pick_timed_out():
                self._fail(f"{self.arrive_next}_tracking_timeout")
            return np.r_[hold, self.grip]

        if self.phase == "close":
            contacts = int(info.get("plug_finger_contacts", 0))
            if contacts == 1:
                self.one_sided_s += e.dt
                if self.one_sided_snapshot is None:
                    self.one_sided_snapshot = self._contact_snapshot()
            else:
                self.one_sided_s = 0.0
                self.one_sided_snapshot = None
            # A small mismatch between the latched detection and the true plug
            # pose lets one pad touch before the other; closing on regardless
            # often lets the plug slide/rotate a little on the table and bring
            # the second pad in too (this is normal -- see ONE_SIDED_CLOSE_GRACE_S
            # vs. the ~0.08 s a good grasp takes to go from one pad to both).
            # But if that one pad starts to shove the plug instead of settling,
            # it will wedge the plug sideways into the table (small but sharp
            # "penetration") well before the grace timeout, so watch for that
            # too and back off (open, retreat, re-plan with a correction for
            # how far off this attempt's estimate was) at the first sign of it.
            if contacts == 1 and (self.one_sided_s > ONE_SIDED_CLOSE_GRACE_S
                                  or info.get("grasp_penetration_m", 0.) > GRASP_PENETRATION_BACKOFF_M):
                self._update_grasp_correction()
                self._regrasp_or_fail("grasp_one_sided_contact")
                return np.r_[self.plan["q_grasp"], self.grip]
            self.grip = max(CLOSE_TRAVEL_M, self.grip - FINGER_CLOSE_RATE_M_S * e.dt)
            closed_for = self.phase_time - (PRESHAPE_TRAVEL_M - CLOSE_TRAVEL_M) / FINGER_CLOSE_RATE_M_S
            if (self.grip <= CLOSE_TRAVEL_M and closed_for > 0.2 and info["grasp_established"]
                    and abs(info["grip_travel_m"] - HELD_TRAVEL_M) > HELD_TRAVEL_TOLERANCE_M):
                # Pads stopped wider (or narrower) than the housing width: they
                # pinch an edge or corner, not both side faces. That grasp squirts
                # out later, so re-grasp instead of lifting.
                self._regrasp_or_fail("grasp_misaligned")
            elif self.grip <= CLOSE_TRAVEL_M and closed_for > 0.2 and info["grasp_established"]:
                self.lift_drift = self.lift_angle = 0.
                self._start_path("lift", self.plan["lift"], speed=LIFT_SPEED_RAD_S,
                                 min_duration=2.5, hold=0.5)
            elif self._pick_timed_out():
                contacts = int(info.get("plug_finger_contacts", 0))
                self._regrasp_or_fail("grasp_no_contact" if contacts < 2 else "grasp_not_settled")
            return np.r_[self.plan["q_grasp"], self.grip]

        if self.phase == "regrasp":
            # Open in place, then retreat straight up and re-plan from there.
            self.grip = PRESHAPE_TRAVEL_M
            if self.regrasps >= self.max_regrasps:
                self._fail("regrasp_budget_exhausted")
            elif self.phase_time > 0.6:
                self.planner.set_state(e.data.qpos)
                p, R = self.planner.fk(self._q_now())
                up = self.planner.line(self._q_now(), p, p + np.array([0., 0., 0.08]), R, step=0.02)
                if up is None:
                    self._fail("regrasp_retreat_ik")
                else:
                    self.regrasps += 1
                    self._start_path("retreat", up, speed=TRANSIT_SPEED_RAD_S, min_duration=1.5, hold=0.2)
            return np.r_[self._q_now(), self.grip]
        raise RuntimeError(f"unknown pick phase {self.phase}")

    def _path_done(self):
        nxt = {"transit_pick": "descend", "descend": "close", "lift": "lift_check",
               "retreat": "replan"}[self.phase]
        if nxt in ("lift_check", "replan"):
            self._enter(nxt)
            return
        self.arrive_next = nxt
        self._goto("arrive")

    def _enter(self, phase):
        """Start a pick phase (with its side effects)."""
        if phase == "descend":
            self._start_path("descend", self.plan["descend"], speed=DESCEND_SPEED_RAD_S,
                             min_duration=2.5, hold=0.3)
        elif phase == "close":
            self.grip = min(self.grip, PRESHAPE_TRAVEL_M)
            self.one_sided_s = 0.0
            self.one_sided_snapshot = None
            self._goto("close")
        elif phase == "lift_check":
            self.pick_report = self._lift_check()
            if self.pick_report["firm"]:
                self._begin_place()
            else:
                self._regrasp_or_fail("lift_not_firm")
        elif phase == "replan":
            # Keep the latched survey detection rather than drawing a fresh noisy
            # sample: a one-sided close's correction (see _update_grasp_correction)
            # is a measurement of how far off THAT estimate was, so it only
            # converges if each retry corrects the same estimate instead of a new
            # one with its own independent noise draw.
            self._begin_pick()

    # ------------------------------------------------------------------ phases
    def action(self):
        e, info = self.env, self.env.info
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
                self.start_plug_z = float(truth["plug_pos"][2])
                self._begin_pick()
            return np.r_[e.target[:7], self.grip]

        if self.phase in PICK_PATH_PHASES + ("arrive", "close", "regrasp"):
            return self._pick_action()

        # ---- post-lift: transport + insertion (controllers/place_insert.py).
        if self.phase in PLACE_PHASES:
            return self._place_action()
        self.failure = f"unknown_phase:{self.phase}"
        return np.r_[e.target[:7], self.grip]

    # ------------------------------------------------------ transport + insert
    def _begin_place(self):
        """Hand the lifted, held plug to PlaceInsert (validated full-task recipe:
        measured in-hand transform, checked joint-space transport, compliant
        straight insertion). Planning failures are reported here, before the arm
        moves. The socket pose is the simulator's (privileged, like the contact
        phases of the original insertion expert): a detection error would only
        make the finger/socket clearance checks spuriously fail."""
        self.place = PlaceInsert(self.env, max_retries=self.max_retries,
                                 probe_offset_y_m=self.probe_offset_y_m)
        self._goto(self.place.phase)
        reason = self.place.start()
        if reason:
            self._fail(reason, self.place.failure_detail)

    def _place_action(self):
        action = self.place.action()
        if self.place.phase != self.phase:
            self._goto(self.place.phase)      # mirror phases into phase_log / recorded phase
        self.retries = self.place.retries
        if self.place.failure:
            self._fail(self.place.failure, self.place.failure_detail)
        return action

    def diagnostics(self):
        place = self.place.diagnostics() if self.place is not None else None
        failure_detail = place["failure_detail"] if place else self.failure_detail
        return {"phase": self.phase, "failure": self.failure, "failure_detail": failure_detail,
                "place": place,
                "retries": self.retries, "regrasps": self.regrasps,
                "detector_mode": self.detector.mode,
                "detection_error": self.detection_error,
                "pick_plans": [{k: v for k, v in s.items() if k != "candidates"} for s in self.plan_summaries],
                "pick_report": self.pick_report, "pick_failures": self.pick_failures,
                "phase_log": self.phase_log + [{"phase": self.phase,
                                                "duration_s": round(self.phase_time, 3)}]}
