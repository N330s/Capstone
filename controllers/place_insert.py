"""Transport and insertion of a physically held plug (tabletop mode).

This is the post-lift half of ``PickInsertExpert``: transit_place -> approach ->
align -> insert -> hold. It follows the validated full-task recipe
(docs/FULL_TASK_DEMONSTRATOR.md, scripts/probe_table_insert.py):

  * The plug-in-hand transform is *measured* once the lifted plug is held (it
    includes the rotation the closure imparted), never assumed.
  * Targets are the socket's preinsert and seated frames. Because the two blades
    are equal, a half-turn about the mating axis is an equivalent socket frame
    (``ConnectorMetrics`` accepts both); the one closer to the held plug's
    current orientation is used.
  * Every endpoint is solved with a restarted damped-least-squares IK that keeps
    a joint-limit margin (null-space push to mid-range), and checked on a
    private MjData with a rigid carried-plug proxy: no robot/plug penetration,
    and a finger/hand clearance to the socket and the table.
  * Transport is ``controllers.carry_path.plan_carry`` (checked direct segment,
    seeded RRT fallback) to a point straight above the preinsert pose, then a
    checked vertical descent. Execution is smooth joint-space interpolation that
    emits the 8-D action (absolute right joint targets + finger travel) every
    step; the finger command is held at its handoff value.
  * Insertion is the validated compliant straight line: closed-loop DLS on the
    plug pose with 0.5 mm / 0.005 rad clamps, aligned at the preinsert pose
    first, then advanced along the mating axis at INSERT_SPEED_M_S.

Nothing here waits on a blind timeout. Planning failures are reported before the
arm moves (``plan_*`` reasons); execution failures name what went wrong
(``grasp_lost``, ``socket_force``, ``insert_stalled``, ...). The planner never
writes live qpos/qvel: all kinematic queries use a private MjData.

Privileged simulator pose (plug, socket, hand) is used here, as in every
expert; it is a label source and must never enter policy observations.
"""
import numpy as np
import mujoco

from connector.simulation import rotation_vector
from controllers.carry_path import plan_carry

HALF_TURN = np.diag([1., -1., -1.])

STANDOFF_M = 0.025            # preinsert: blades 9 mm clear of the socket face (validated)
ABOVE_HEIGHTS_M = (0.06, 0.035, 0.015, 0.)   # transport end point above the preinsert pose
MAX_SEATED_BRANCHES = 3
PREINSERT_SOCKET_CLEARANCE_M = 0.002
HELD_FINGER_TRAVEL_M = 0.0137 # right finger travel when closed on the housing (measured)
SEAT_X_M = 0.00003            # final plug target along the mating axis (validated)
MIN_JOINT_MARGIN_RAD = 0.05   # hard limit on every planned configuration
GOOD_JOINT_MARGIN_RAD = 0.15  # preferred margin when choosing among IK solutions
FINGER_SOCKET_CLEARANCE_M = 0.0005
FINGER_TABLE_CLEARANCE_M = 0.002
LINE_STEP_M = 0.002           # IK chain resolution along straight lines
MAX_LINE_JUMP_RAD = 0.15      # continuity limit between chain samples

TRANSIT_SPEED_RAD_S = 0.15    # speed sweep 2026-09-25: 0.2 -> 85%, 0.15 -> 95% (rows 100-139), ~37 s episodes; slower gave no further gain
DESCEND_SPEED_RAD_S = 0.1
SETTLE_S = 0.5                # hold after each executed segment
ALIGN_S = (1.0, 4.0)          # min / max time aligning at the preinsert pose
ALIGN_POS_TOL_M = 0.0003
ALIGN_ANGLE_TOL_DEG = 0.5
INSERT_SPEED_M_S = 0.005
INSERT_MAX_S = 10.0
HOLD_MAX_S = 2.0
SERVO_GAIN = 0.2
SERVO_POS_CLIP_M = 0.0005
SERVO_ROT_CLIP_RAD = 0.005
SERVO_STEP_CLIP_RAD = 0.008
GRASP_DRIFT_LIMIT_M = 0.001   # vs the transform measured at handoff (env limit)
GRASP_ANGLE_LIMIT_DEG = 5.0
SOCKET_FORCE_LIMIT_N = 5.0
STALL_S = 1.0
TRACK_ERROR_LIMIT_M = 0.004   # plug off the planned line while advancing


def smoothstep(x):
    x = min(1., max(0., x))
    return x * x * (3. - 2. * x)


def closest_equivalent(basis, current):
    """Socket frame (or its half-turn twin) closest to the current plug rotation."""
    twin = basis @ HALF_TURN
    angle = lambda r: np.linalg.norm(rotation_vector(r @ current.T))
    return basis if angle(basis) <= angle(twin) else twin


class ArmKinematics:
    """IK and carried-plug collision queries for the right arm on a private MjData.

    Deterministic (fixed seeds, no wall clock). ``sync`` copies the live qpos so
    the parked arm, fingers and plug are where they really are; nothing is ever
    written back to live data.
    """

    def __init__(self, model, *, seed=11, n_seeds=10):
        m = self.m = model
        self.d = mujoco.MjData(m)
        joints = [m.joint(f"openarm_right_joint{i}").id for i in range(1, 8)]
        self.qa = np.array([m.jnt_qposadr[j] for j in joints])
        self.va = np.array([m.jnt_dofadr[j] for j in joints])
        self.lo = m.jnt_range[joints, 0].copy()
        self.hi = m.jnt_range[joints, 1].copy()
        self.mid = 0.5 * (self.lo + self.hi)
        self.site = m.site("robot_grasp").id
        self.plug = m.body("plug").id
        self.plug_qadr = int(m.jnt_qposadr[m.joint("plug_free").id])
        self.socket = int(m.site_bodyid[m.site("socket_entry").id])
        self.robot_bodies = {i for i in range(m.nbody) if m.body(i).name.startswith("openarm")}
        self.fingers = {m.body(f"openarm_right_{s}_finger").id for s in ("right", "left")}
        self.fqa = np.array([m.jnt_qposadr[m.joint(f"openarm_right_finger_joint{i}").id] for i in (1, 2)])
        hand = m.body("openarm_right_hand").id
        collides = lambda g: bool(m.geom_contype[g] or m.geom_conaffinity[g])
        self.distal_geoms = [g for g in range(m.ngeom)
                             if int(m.geom_bodyid[g]) in (self.fingers | {hand}) and collides(g)]
        is_box = lambda g: m.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX
        self.socket_geoms = [g for g in range(m.ngeom)
                             if int(m.geom_bodyid[g]) == self.socket and collides(g) and is_box(g)]
        self.table_geoms = [g for g in range(m.ngeom)
                            if "table" in (m.geom(g).name or "").lower() and collides(g) and is_box(g)]
        self.distal_verts = {}
        for g in self.distal_geoms:
            if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
                mesh = int(m.geom_dataid[g])
                a, n = int(m.mesh_vertadr[mesh]), int(m.mesh_vertnum[mesh])
                self.distal_verts[g] = m.mesh_vert[a:a + n].astype(float).copy()
        self.jp = np.zeros((3, m.nv))
        self.jr = np.zeros((3, m.nv))
        rng = np.random.default_rng(seed)
        self.fixed_seeds = [np.array([0., 0., 0., 1.5707963, 0., 0., 0.])] + [
            rng.uniform(self.lo + .1, self.hi - .1) for _ in range(n_seeds)]
        self.queries = 0

    # ------------------------------------------------------------ state
    def sync(self, qpos, finger_travel=None):
        self.d.qpos[:] = qpos
        if finger_travel is not None:
            self.d.qpos[self.fqa] = finger_travel
        self.d.qvel[:] = 0.
        mujoco.mj_forward(self.m, self.d)

    def margin(self, q):
        return float(np.min(np.r_[q - self.lo, self.hi - q]))

    def fk(self, q):
        self.d.qpos[self.qa] = q
        mujoco.mj_kinematics(self.m, self.d)
        return (self.d.site_xpos[self.site].copy(),
                self.d.site_xmat[self.site].reshape(3, 3).copy())

    # --------------------------------------------------------------- IK
    def _solve(self, pos, rot, q, iters):
        m, d = self.m, self.d
        span = self.hi - self.lo
        best = np.inf
        for it in range(iters):
            d.qpos[self.qa] = q
            mujoco.mj_kinematics(m, d)
            mujoco.mj_comPos(m, d)
            err = np.r_[pos - d.site_xpos[self.site],
                        rotation_vector(rot @ d.site_xmat[self.site].reshape(3, 3).T)]
            if np.linalg.norm(err[:3]) < 2e-5 and np.linalg.norm(err[3:]) < 2e-4:
                return q, True
            if it % 50 == 49:          # stalled far from the target: give up on this seed
                size = float(np.linalg.norm(err[:3]) + 0.05 * np.linalg.norm(err[3:]))
                if size > 1e-3 and size > 0.9 * best:
                    return q, False
                best = min(best, size)
            mujoco.mj_jacSite(m, d, self.jp, self.jr, self.site)
            jac = np.vstack((self.jp[:, self.va], self.jr[:, self.va]))
            inv = jac.T @ np.linalg.inv(jac @ jac.T + np.eye(6) * 1e-4)
            step = inv @ err
            # Null-space push toward mid-range keeps a joint-limit margin. The
            # projector uses the exact pseudo-inverse: the damped one leaks the
            # push into the task space and leaves a steady-state pose error.
            null = np.eye(7) - np.linalg.pinv(jac, rcond=1e-3) @ jac
            step += null @ (0.05 * (self.mid - q) / (span * span) * span.mean())
            q = np.clip(q + np.clip(step, -.08, .08), self.lo + 1e-3, self.hi - 1e-3)
        return q, False

    def ik_all(self, pos, rot, reference, *, iters=250, min_margin=MIN_JOINT_MARGIN_RAD):
        """All distinct solutions from every seed, best first (margin >= GOOD,
        then closeness to ``reference``; else by margin)."""
        reference = np.asarray(reference, dtype=float)
        found = []
        for seed in [reference, *self.fixed_seeds]:
            q, ok = self._solve(pos, rot, np.clip(np.array(seed, dtype=float), self.lo + 1e-3, self.hi - 1e-3), iters)
            if ok and self.margin(q) >= min_margin and not any(np.max(np.abs(q - f)) < 1e-2 for f in found):
                found.append(q)
        good = sorted((q for q in found if self.margin(q) >= GOOD_JOINT_MARGIN_RAD),
                      key=lambda q: float(np.max(np.abs(q - reference))))
        rest = sorted((q for q in found if self.margin(q) < GOOD_JOINT_MARGIN_RAD), key=lambda q: -self.margin(q))
        return good + rest

    def ik(self, pos, rot, reference, *, extra_seeds=(), iters=250, min_margin=MIN_JOINT_MARGIN_RAD,
           accept=None):
        """Best configuration for the tool pose, or None.

        Solutions below ``min_margin`` or rejected by ``accept(q)`` are discarded.
        Among the rest, the one closest to ``reference`` that has at least
        GOOD_JOINT_MARGIN_RAD wins (else the largest margin)."""
        reference = np.asarray(reference, dtype=float)
        seeds = [reference, *extra_seeds, *self.fixed_seeds]
        found = []
        for seed in seeds:
            q, ok = self._solve(pos, rot, np.clip(np.array(seed, dtype=float), self.lo + 1e-3, self.hi - 1e-3), iters)
            if not ok or self.margin(q) < min_margin:
                continue
            if any(np.max(np.abs(q - f)) < 1e-3 for f in found):
                continue
            if accept is not None and not accept(q):
                continue
            found.append(q)
            if self.margin(q) >= GOOD_JOINT_MARGIN_RAD and np.max(np.abs(q - reference)) < .3:
                break  # continuous with the current posture and comfortable: done
        if not found:
            return None
        good = [q for q in found if self.margin(q) >= GOOD_JOINT_MARGIN_RAD]
        if good:
            return min(good, key=lambda q: float(np.max(np.abs(q - reference))))
        return max(found, key=self.margin)

    def line(self, q0, start, end, rot, *, accept=None, min_margin=MIN_JOINT_MARGIN_RAD):
        """IK chain along a straight tool line from ``start`` to ``end`` (fixed rotation).

        Returns the list of configurations (excluding q0) or None if any sample
        fails, loses margin or jumps (a posture flip is not a straight line)."""
        count = max(1, int(np.ceil(np.linalg.norm(end - start) / LINE_STEP_M)))
        chain, q = [], np.asarray(q0, dtype=float)
        for i in range(1, count + 1):
            target = start + (end - start) * i / count
            nxt, ok = self._solve(target, rot, q.copy(), 200)
            if not ok or self.margin(nxt) < min_margin or np.max(np.abs(nxt - q)) > MAX_LINE_JUMP_RAD:
                return None
            if accept is not None and not accept(nxt):
                return None
            chain.append(nxt)
            q = nxt
        return chain

    # -------------------------------------------------------- collisions
    def place(self, q, rel_pos, rel_rot):
        """Arm at q with the plug proxy rigidly in hand. Scratch only."""
        m, d = self.m, self.d
        d.qpos[self.qa] = q
        mujoco.mj_kinematics(m, d)
        rot = d.site_xmat[self.site].reshape(3, 3)
        d.qpos[self.plug_qadr:self.plug_qadr + 3] = d.site_xpos[self.site] + rot @ rel_pos
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, (rot @ rel_rot).ravel())
        d.qpos[self.plug_qadr + 3:self.plug_qadr + 7] = quat
        mujoco.mj_forward(m, d)

    def _clearance(self, geoms_b):
        """Smallest distance from the finger/hand hull vertices to the given boxes.

        Vertex-to-box distances (exact per vertex). mj_geomDistance returns
        spurious zeros for these mesh/box pairs in MuJoCo 3.11, so it is not
        used; actual overlap is caught separately by the contact check."""
        best = np.inf
        points = [self.d.geom_xpos[g] + v @ self.d.geom_xmat[g].reshape(3, 3).T
                  for g, v in self.distal_verts.items()]
        points = np.vstack(points)
        for b in geoms_b:
            local = (points - self.d.geom_xpos[b]) @ self.d.geom_xmat[b].reshape(3, 3)
            outside = np.maximum(np.abs(local) - self.m.geom_size[b], 0.)
            best = min(best, float(np.min(np.linalg.norm(outside, axis=1))))
        return best

    def collisions(self, q, rel_pos, rel_rot, *, allow_plug_socket=False, socket_clearance=0.,
                   table_clearance=0.):
        """Names of offending contacts for this carried configuration ([] = clear)."""
        self.queries += 1
        self.place(q, rel_pos, rel_rot)
        m, d = self.m, self.d
        bad = []
        for c in d.contact:
            bodies = {int(m.geom_bodyid[g]) for g in (c.geom1, c.geom2)}
            if self.plug in bodies and bodies & self.fingers:
                continue
            if allow_plug_socket and bodies == {self.plug, self.socket}:
                continue
            if (bodies & self.robot_bodies or self.plug in bodies) and c.dist < 0.:
                bad.append(f"{m.geom(c.geom1).name}|{m.geom(c.geom2).name}")
        if socket_clearance > 0 and self.socket_geoms:
            if self._clearance(self.socket_geoms) < socket_clearance:
                bad.append("fingers_near_socket")
        if table_clearance > 0 and self.table_geoms:
            if self._clearance(self.table_geoms) < table_clearance:
                bad.append("fingers_near_table")
        return bad


class InsertionPlan:
    """Joint-space plan from the held pose to the seated pose, or a failure reason."""

    def __init__(self):
        self.failure = None
        self.detail = ""
        self.q_above = self.q_pre = self.q_seated = None
        self.descent = []      # chain above -> preinsert
        self.insertion = []    # chain preinsert -> seated (for checking; executed closed-loop)
        self.basis = None
        self.entry = None

    def fail(self, reason, detail=""):
        self.failure, self.detail = reason, detail
        return self


def tool_target(plug_pos, plug_rot, rel_pos, rel_rot):
    """Tool pose that puts the held plug at (plug_pos, plug_rot)."""
    tool_rot = plug_rot @ rel_rot.T
    return plug_pos - tool_rot @ rel_pos, tool_rot


def plan_insertion(kin, qpos, q_now, entry, basis, rel_pos, rel_rot, *, current_plug_rot=None,
                   table_z=None, standoff=STANDOFF_M, above=ABOVE_HEIGHTS_M, finger_travel=None):
    """Kinematic plan for the seated, preinsert and above-preinsert poses.

    ``rel_pos``/``rel_rot``: plug pose in the robot_grasp frame (measured).
    ``kin.sync(qpos, finger_travel)`` is called first so the scratch scene
    matches live (``finger_travel`` overrides the right finger joints, e.g. the
    closed travel when screening before a grasp exists). Several seated IK
    branches are tried; the first whose straight insertion line and preinsert
    pose check out wins. The transport end point is straight above the
    preinsert pose at the highest height in ``above`` whose vertical descent is
    feasible (0 means transport directly to the preinsert pose).
    """
    plan = InsertionPlan()
    kin.sync(qpos, finger_travel)
    if current_plug_rot is not None:
        basis = closest_equivalent(basis, current_plug_rot)
    elif basis[2, 2] < 0:
        basis = basis @ HALF_TURN
    plan.basis, plan.entry = basis, np.asarray(entry, dtype=float)
    axis = basis[:, 0]
    seated_plug = plan.entry + axis * SEAT_X_M
    pre_plug = plan.entry - axis * standoff
    pre_tool, tool_rot = tool_target(pre_plug, basis, rel_pos, rel_rot)
    seated_tool, _ = tool_target(seated_plug, basis, rel_pos, rel_rot)
    table_kw = dict(table_clearance=FINGER_TABLE_CLEARANCE_M if table_z is not None else 0.)
    contact_kw = dict(allow_plug_socket=True, socket_clearance=FINGER_SOCKET_CLEARANCE_M, **table_kw)

    seated = kin.ik_all(seated_tool, tool_rot, q_now)
    if not seated:
        return plan.fail("plan_seated_unreachable")
    reason = None
    for q_seated in seated[:MAX_SEATED_BRANCHES]:
        hit = kin.collisions(q_seated, rel_pos, rel_rot, **contact_kw)
        if hit:
            # Pose-determined (same tool pose on every branch) except for the arm links.
            reason = reason or ("plan_seated_collision", ",".join(hit))
            continue
        # Work backwards from the seated posture so the insertion line is continuous.
        insertion = kin.line(q_seated, seated_tool, pre_tool, tool_rot)
        if insertion is None:
            reason = reason or ("plan_insert_line_infeasible", "")
            continue
        q_pre = insertion[-1]
        hit = next((h for h in (kin.collisions(q, rel_pos, rel_rot, **contact_kw)
                                for q in insertion[::3] + [q_pre]) if h), None)
        if hit:
            reason = reason or ("plan_insert_line_collision", ",".join(hit))
            continue
        hit = kin.collisions(q_pre, rel_pos, rel_rot, socket_clearance=PREINSERT_SOCKET_CLEARANCE_M, **table_kw)
        if hit:
            reason = reason or ("plan_preinsert_collision", ",".join(hit))
            continue
        break
    else:
        return plan.fail(*reason)
    plan.descent, plan.q_above, plan.above_m = [], q_pre, 0.
    for height in above:
        if height <= 0:
            break
        up = kin.line(q_pre, pre_tool, pre_tool + np.array([0., 0., height]), tool_rot)
        if up is None:
            continue
        if any(kin.collisions(q, rel_pos, rel_rot, socket_clearance=PREINSERT_SOCKET_CLEARANCE_M, **table_kw)
               for q in up[::3] + [up[-1]]):
            continue
        plan.descent, plan.q_above, plan.above_m = up[::-1][1:] + [q_pre], up[-1], height
        break
    else:
        return plan.fail("plan_descent_infeasible")
    plan.q_seated, plan.q_pre = q_seated, q_pre
    plan.insertion = insertion[::-1][1:] + [q_seated]    # preinsert -> seated
    plan.pre_plug, plan.seated_plug, plan.tool_rot = pre_plug, seated_plug, tool_rot
    return plan


def _rot_x(deg):
    a = np.radians(deg)
    return np.array([[1., 0., 0.], [0., np.cos(a), -np.sin(a)], [0., np.sin(a), np.cos(a)]])


def _rot_y(deg):
    a = np.radians(deg)
    return np.array([[np.cos(a), 0., np.sin(a)], [0., 1., 0.], [-np.sin(a), 0., np.cos(a)]])


# Nominal top-down table grasp, same convention as the pick planner in
# controllers/pick_insert_expert.py: tool rotation in the plug frame is
# Ry(90) @ Rx(180*flip) @ Ry(tilt) (approach along plug -z, pinch along plug y,
# tilt about the pinch axis); the tool site sits on plug_grasp_frame (housing
# middle, plug x = -16 mm), NOMINAL_GRASP_HEIGHT_M above the housing mid-plane.
# Only used to screen socket poses before a grasp exists; execution always uses
# the transform measured after the physical grasp.
NOMINAL_GRASP_POINT_M = np.array([-0.016, 0., 0.])
NOMINAL_GRASP_HEIGHT_M = 0.004
NOMINAL_GRASP_TILTS_DEG = (-30., -15., 0., 15., 30.)


def nominal_grasp(flip, tilt_deg):
    """(plug position, plug rotation) in the robot_grasp frame for a nominal grasp."""
    tool_in_plug = _rot_y(90.) @ _rot_x(180. * flip) @ _rot_y(tilt_deg)
    tool_pos_in_plug = NOMINAL_GRASP_POINT_M + np.array([0., 0., NOMINAL_GRASP_HEIGHT_M])
    return -tool_in_plug.T @ tool_pos_in_plug, tool_in_plug.T


REFERENCE_POSTURE = np.array([0., 0., 0., 1.5707963, 0., 0., 0.])


class InsertFeasibility:
    """Deterministic kinematic screen for socket poses (used by scene_bank).

    A socket pose passes if at least one nominal top-down grasp whose hand yaw
    is plausible (tool finger-width axis within 75 deg of world +x, the only
    top-down hand yaws the right arm reaches over the table) admits the full
    insertion plan (seated, straight insertion line, preinsert, vertical
    descent; all IK with joint margin and carried-plug collision checks).
    Private env/model; fixed seeds; no dependence on live state.
    """

    def __init__(self, tilts=NOMINAL_GRASP_TILTS_DEG):
        from envs.openarm_insert import OpenArmInsertEnv   # lazy: avoid import cycles
        self.env = OpenArmInsertEnv(images=False)
        self.kin = ArmKinematics(self.env.model)
        self.tilts = tuple(tilts)

    def grasps_for(self, basis):
        upright = basis @ HALF_TURN if basis[2, 2] < 0 else basis
        out = []
        for flip in (0, 1):
            _, rel_rot = nominal_grasp(flip, 0.)
            finger_axis = (upright @ rel_rot.T)[:, 2]
            if finger_axis[0] >= np.cos(np.radians(75.)):
                out += [(flip, tilt) for tilt in self.tilts]
        return out

    def __call__(self, options):
        """(None, [(flip, tilt), ...feasible]) or (reason, [])."""
        options = {k: v for k, v in options.items() if k != "visual"}
        try:
            self.env.reset(seed=0, options=options)
        except (RuntimeError, ValueError) as error:
            return f"reset_failed:{type(error).__name__}", []
        e = self.env
        entry = e.data.site_xpos[e.socket_site].copy()
        basis = e.data.site_xmat[e.socket_site].reshape(3, 3).copy()
        table_z = options.get("table_height_m")
        first = None
        for flip, tilt in self.grasps_for(basis):
            rel_pos, rel_rot = nominal_grasp(flip, tilt)
            plan = plan_insertion(self.kin, e.data.qpos.copy(), REFERENCE_POSTURE, entry, basis,
                                  rel_pos, rel_rot, table_z=table_z, finger_travel=HELD_FINGER_TRAVEL_M)
            if plan.failure is None:
                return None, [(flip, tilt)]
            first = first or plan.failure
        return (first or "no_plausible_grasp"), []


class JointSegment:
    """Piecewise-linear joint path under one smoothstep time law, then a hold.

    All joints move synchronously along the checked segment (the validated
    execute_waypoint fix: independently saturated ramps trace a different curve)."""

    def __init__(self, start, waypoints, *, speed, hold, min_duration=1.0):
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
        if self.cum[-1] <= 0.:
            return self.goal.copy()
        s = smoothstep(self.t / self.duration) * self.cum[-1]
        i = int(np.clip(np.searchsorted(self.cum, s, side="right") - 1, 0, len(self.points) - 2))
        span = self.cum[i + 1] - self.cum[i]
        frac = 0. if span <= 0 else min(1., (s - self.cum[i]) / span)
        return self.points[i] + (self.points[i + 1] - self.points[i]) * frac


PLACE_PHASES = ("transit_place", "approach", "align", "insert", "withdraw", "hold")


class PlaceInsert:
    """Post-lift phase machine. Call ``start()`` once the lifted plug is held,
    then ``action()`` once per env step; it returns the 8-D action.

    ``failure`` / ``failure_detail`` are set instead of timing out blindly.
    ``socket_estimate`` (pos, mat) is used for planning the transport; the
    contact phases servo on the simulator's socket pose (privileged, as in the
    original insertion expert). Deterministic given the env state.
    """

    def __init__(self, env, *, kin=None, socket_estimate=None, max_retries=1, probe_offset_y_m=0.):
        self.env = env
        self.kin = kin if kin is not None else ArmKinematics(env.model)
        self.socket_estimate = socket_estimate
        self.max_retries, self.retries = max_retries, 0
        self.probe_offset_y_m = probe_offset_y_m
        self.phase, self.phase_time = "transit_place", 0.
        self.failure = self.failure_detail = None
        self.grip = float(env.target[7])
        self.log = []
        self.stats = {}
        self._pinned_s = self._contact_loss_s = 0.
        self.jp = np.zeros((3, env.model.nv))
        self.jr = np.zeros((3, env.model.nv))

    # ------------------------------------------------------------ helpers
    def _truth_socket(self):
        e = self.env
        return (e.data.site_xpos[e.socket_site].copy(),
                e.data.site_xmat[e.socket_site].reshape(3, 3).copy())

    def _plug_in_hand(self):
        e, d = self.env, self.env.data
        rot = d.site_xmat[e.grasp_site].reshape(3, 3)
        return (rot.T @ (d.xpos[e.plug] - d.site_xpos[e.grasp_site]),
                rot.T @ d.xmat[e.plug].reshape(3, 3))

    def _finger_contacts(self):
        e, m, d = self.env, self.env.model, self.env.data
        pads = set()
        for c in d.contact[:d.ncon]:
            bodies = {int(m.geom_bodyid[c.geom1]), int(m.geom_bodyid[c.geom2])}
            if e.plug in bodies:
                pads |= bodies & set(e.fingers)
        return len(pads)

    def _fail(self, reason, detail=""):
        self.failure, self.failure_detail = reason, detail
        return self._hold_action()

    def _hold_action(self):
        return np.r_[self.env.target[:7], self.grip]

    def _goto(self, phase):
        self.log.append({"phase": self.phase, "duration_s": round(self.phase_time, 3)})
        self.phase, self.phase_time = phase, 0.

    def _grasp_ok(self):
        pos, rot = self._plug_in_hand()
        drift = float(np.linalg.norm(pos - self.ref_pos))
        angle = float(np.degrees(np.linalg.norm(rotation_vector(rot @ self.ref_rot.T))))
        self.stats["max_grasp_drift_m"] = max(self.stats.get("max_grasp_drift_m", 0.), drift)
        self.stats["max_grasp_angle_deg"] = max(self.stats.get("max_grasp_angle_deg", 0.), angle)
        contacts = self._finger_contacts()
        self._contact_loss_s = 0. if contacts == 2 else self._contact_loss_s + self.env.dt
        if drift > GRASP_DRIFT_LIMIT_M or angle > GRASP_ANGLE_LIMIT_DEG:
            return f"drift={drift*1e3:.2f}mm angle={angle:.1f}deg"
        if self._contact_loss_s > 0.2:
            return f"finger_contacts={contacts}"
        return None

    # ------------------------------------------------------------ planning
    def grasp_reference(self):
        """Plug pose in the robot_grasp frame (position, rotation) that grasp
        drift is measured against.

        Prefers the env's latched tabletop grasp (``env.latched_grasp()``: the
        hand-to-plug pose latched when the physical grasp was established, same
        convention), so this controller fails fast on exactly the drift the env
        scores. Falls back to the pose measured at handoff, after the lift.
        Planning always uses the pose measured at handoff (``rel_pos/rel_rot``).

        Merge note: ``OpenArmInsertEnv.latched_grasp()`` returns
        ``self._grasp_reference`` = ``plug_in_hand()`` taken at the moment the
        physical grasp was latched -- the exact same
        ``rot.T @ (plug_pos - grasp_site_pos), rot.T @ plug_rot`` formula as
        ``_plug_in_hand()`` below, in the same ``robot_grasp`` site frame. The
        env's own ``info["grasp_slip_m"]``/``["grasp_angle_deg"]`` (used for
        ``valid_pose``/hold and the tabletop drop guard) are computed against
        that same ``_grasp_reference``, so preferring it here keeps this
        controller's grasp-lost check and the env's own success/drift checks
        reading off one reference, not two independently-measured ones. It is
        set (non-None) by the time transport starts: the lift phase only
        proceeds while ``info["grasp_established"]`` holds, which requires the
        latch to already exist."""
        latched = getattr(self.env, "latched_grasp", None)
        reference = latched() if callable(latched) else None
        if reference is not None:
            self.stats["handoff_reference"] = "env_latched_grasp"
            return np.array(reference[0], dtype=float), np.array(reference[1], dtype=float)
        self.stats["handoff_reference"] = "measured_after_lift"
        return self._plug_in_hand()

    def start(self):
        """Measure the held transform and plan. Returns the failure reason or None."""
        e, d = self.env, self.env.data
        self.grip = float(e.target[7])
        self.rel_pos, self.rel_rot = self._plug_in_hand()
        self.ref_pos, self.ref_rot = self.grasp_reference()
        if self._finger_contacts() < 2:
            self._fail("handoff_not_held", f"finger_contacts={self._finger_contacts()}")
            return self.failure
        entry, basis = self.socket_estimate if self.socket_estimate is not None else self._truth_socket()
        table_z = getattr(e, "reset_options", {}).get("table_height_m")
        q_now = e.target[:7].copy()
        plan = plan_insertion(self.kin, d.qpos.copy(), q_now, entry, basis, self.rel_pos, self.rel_rot,
                              current_plug_rot=d.xmat[e.plug].reshape(3, 3).copy(), table_z=table_z)
        if plan.failure:
            self._fail(plan.failure, plan.detail)
            return self.failure
        self.plan = plan
        try:
            path, report = plan_carry(e, q_now, plan.q_above, self.rel_pos, self.rel_rot)
        except RuntimeError as error:
            self._fail("plan_transport_failed", str(error))
            return self.failure
        self.stats["carry"] = {k: v for k, v in report.items() if k != "seed"}
        self.segment = JointSegment(q_now, path, speed=TRANSIT_SPEED_RAD_S, hold=SETTLE_S, min_duration=2.)
        return None

    # ------------------------------------------------------------ servo
    def _servo(self, desired_pos, desired_rot):
        """Validated compliant step: DLS on the plug pose, clamped, on the joint target."""
        e, m, d = self.env, self.env.model, self.env.data
        plug_pos, plug_rot = d.xpos[e.plug], d.xmat[e.plug].reshape(3, 3)
        error = np.r_[np.clip(desired_pos - plug_pos, -SERVO_POS_CLIP_M, SERVO_POS_CLIP_M),
                      np.clip(rotation_vector(desired_rot @ plug_rot.T), -SERVO_ROT_CLIP_RAD, SERVO_ROT_CLIP_RAD)]
        mujoco.mj_jacSite(m, d, self.jp, self.jr, e.grasp_site)
        jac = np.vstack([self.jp[:, e.va["right"]], self.jr[:, e.va["right"]]])
        change = jac.T @ np.linalg.solve(jac @ jac.T + np.eye(6) * 1e-5, error)
        lo, hi = self.kin.lo + 0.01, self.kin.hi - 0.01
        target = np.clip(e.target[:7] + np.clip(SERVO_GAIN * change, -SERVO_STEP_CLIP_RAD, SERVO_STEP_CLIP_RAD),
                         lo, hi)
        pinned = (target <= lo + 1e-9) | (target >= hi - 1e-9)
        self._pinned_s = self._pinned_s + e.dt if pinned.any() else 0.
        return np.r_[target, self.grip]

    def _plug_error(self, desired_pos, desired_rot):
        e, d = self.env, self.env.data
        return (float(np.linalg.norm(desired_pos - d.xpos[e.plug])),
                float(np.degrees(np.linalg.norm(rotation_vector(desired_rot @ d.xmat[e.plug].reshape(3, 3).T)))))

    # ------------------------------------------------------------ phases
    def action(self):
        e, info = self.env, self.env.info
        self.phase_time += e.dt
        if self.failure:
            return self._hold_action()
        bad = self._grasp_ok()
        if bad:
            return self._fail("grasp_lost", f"{self.phase}: {bad}")

        if self.phase == "transit_place":
            q = self.segment.step(e.dt)
            if self.segment.done:
                self.segment = JointSegment(e.target[:7], self.plan.descent, speed=DESCEND_SPEED_RAD_S,
                                            hold=SETTLE_S)
                self._goto("approach")
            return np.r_[q, self.grip]

        if self.phase == "approach":
            q = self.segment.step(e.dt)
            if self.segment.done:
                self._begin_align()
            return np.r_[q, self.grip]

        entry, basis = self._truth_socket()
        basis = closest_equivalent(basis, e.data.xmat[e.plug].reshape(3, 3))
        axis = basis[:, 0]
        force = float(info.get("contact_force_n", 0.))
        self.stats["peak_socket_force_n"] = max(self.stats.get("peak_socket_force_n", 0.), force)

        if self.phase == "align":
            desired = entry - axis * STANDOFF_M
            if self.retries == 0 and self.probe_offset_y_m:
                desired = desired + basis[:, 1] * self.probe_offset_y_m
            action = self._servo(desired, basis)
            err, ang = self._plug_error(desired, basis)
            if self.phase_time >= ALIGN_S[0] and err < ALIGN_POS_TOL_M and ang < ALIGN_ANGLE_TOL_DEG:
                self.forward = -STANDOFF_M
                self.progress_depth, self.stall_s = -1., 0.
                self._goto("insert")
            elif self.phase_time > ALIGN_S[1]:
                return self._fail("align_not_converged", f"err={err*1e3:.2f}mm angle={ang:.2f}deg")
            elif self._pinned_s > 0.5:
                return self._fail("joint_limit", "align")
            return action

        if self.phase == "insert":
            self.forward = min(SEAT_X_M, self.forward + INSERT_SPEED_M_S * e.dt)
            desired = entry + axis * self.forward
            if self.retries == 0 and self.probe_offset_y_m:
                desired = desired + basis[:, 1] * self.probe_offset_y_m
            action = self._servo(desired, basis)
            depth = float(info.get("insertion_depth_m", -1.))
            if depth > self.progress_depth + 2e-4:
                self.progress_depth, self.stall_s = depth, 0.
            elif force > 0.5 or self.forward >= SEAT_X_M:
                self.stall_s += e.dt
            lateral = float(np.hypot(info.get("offset_y_m", 0.), info.get("offset_z_m", 0.)))
            if info.get("valid_pose"):
                self._goto("hold")
            elif force > SOCKET_FORCE_LIMIT_N or self.stall_s > STALL_S or self.phase_time > INSERT_MAX_S:
                why = ("socket_force" if force > SOCKET_FORCE_LIMIT_N else
                       "insert_stalled" if self.stall_s > STALL_S else "insert_too_slow")
                detail = f"depth={depth*1e3:.2f}mm force={force:.2f}N lateral={lateral*1e3:.2f}mm"
                if self.retries < self.max_retries:
                    self.retries += 1
                    self.stats.setdefault("retry_reasons", []).append(f"{why}: {detail}")
                    self._goto("withdraw")
                else:
                    return self._fail(why, detail)
            elif self.forward < -0.004 and lateral > TRACK_ERROR_LIMIT_M:
                return self._fail("insert_off_axis", f"lateral={lateral*1e3:.2f}mm")
            elif self._pinned_s > 0.5:
                return self._fail("joint_limit", "insert")
            return action

        if self.phase == "withdraw":
            self.forward = max(-STANDOFF_M, self.forward - 3 * INSERT_SPEED_M_S * e.dt)
            action = self._servo(entry + axis * self.forward, basis)
            if self.forward <= -STANDOFF_M:
                self._begin_align()
            return action

        if self.phase == "hold":
            action = self._servo(entry + axis * SEAT_X_M, basis)
            self.stats["final_depth_m"] = float(info.get("insertion_depth_m", 0.))
            if self.phase_time > HOLD_MAX_S:
                return self._fail("hold_not_valid",
                                  f"valid_pose={info.get('valid_pose')} "
                                  f"grasp_established={info.get('grasp_established')} "
                                  f"env_slip={info.get('grasp_slip_m', 0.) * 1e3:.2f}mm")
            return action
        return self._fail("unknown_phase", self.phase)

    def _begin_align(self):
        self._pinned_s = 0.
        self._goto("align")

    def diagnostics(self):
        return {"phase": self.phase, "failure": self.failure, "failure_detail": self.failure_detail,
                "retries": self.retries, "stats": self.stats,
                "phase_log": self.log + [{"phase": self.phase, "duration_s": round(self.phase_time, 3)}]}
