"""Ray-cast visibility of the plug and the socket from the pilot cameras (read only).

A camera "sees" the plug when at least one of four sample points on the plug housing projects
inside the image and a ray from the camera reaches it first (geom group 3 is hidden, like the
renderer). The cable hanging off the plug is a child body, but a ray that ends on the cable does not
count as seeing the plug. The socket rule is the same with points on the socket entrance.

The probe only reads ``data`` (``cam_xpos``, ``cam_xmat``, ``xpos``, ``xmat``): the caller must have
run ``mj_forward``. It never writes ``qpos``/``qvel``.
"""
import math

import mujoco
import numpy as np

PILOT_CAMERAS = ("scene_rgb", "wrist_rgb")
GEOM_GROUPS = np.array([1, 1, 1, 0, 1, 1], dtype=np.uint8)  # group 3 hidden, as in the renderer
RAY_START_M = 0.01
INSET = 0.85  # sample points sit inside the housing so the ray ends on its surface


def _subtree_bodies(model, root, skip_prefix="cable"):
    """Body ids of root and its descendants, leaving out cable segments."""
    out = []
    for b in range(model.nbody):
        c = b
        while c != 0 and c != root:
            c = int(model.body_parentid[c])
        if c == root and not model.body(b).name.startswith(skip_prefix):
            out.append(b)
    return out


def _resolution(env, camera):
    if camera == "chest_rgb":
        c = env.workspace["cameras"]["chest_rgbd"]
        return int(c["width"]), int(c["height"])
    return int(env.config["image_width"]), int(env.config["image_height"])


def points_visible(model, data, pos, mat, fovy_deg, w, h, points, bodies):
    """True when any world point projects into the w x h image of a camera at ``pos`` with
    orientation ``mat`` (columns = camera x, y, z; it looks along -z) and a ray from the camera
    reaches it first on a geom of ``bodies``. ``data`` must be forwarded."""
    f = 0.5 * h / math.tan(math.radians(fovy_deg) / 2)
    geomid = np.zeros(1, dtype=np.int32)
    for p in points:
        pc = mat.T @ (p - pos)
        z = -pc[2]
        if z <= 1e-6:
            continue
        u, v = w / 2 + f * pc[0] / z, h / 2 - f * pc[1] / z
        if not (0 <= u < w and 0 <= v < h):
            continue
        dirv = p - pos
        dirv = dirv / np.linalg.norm(dirv)
        hit = mujoco.mj_ray(model, data, pos + dirv * RAY_START_M, dirv, GEOM_GROUPS, 1, -1, geomid)
        if hit >= 0 and int(geomid[0]) >= 0 and int(model.geom_bodyid[geomid[0]]) in bodies:
            return True
    return False


class VisibilityProbe:
    def __init__(self, env, cameras=PILOT_CAMERAS):
        self.env = env
        self.model = m = env.model
        self.cameras = tuple(cameras)
        self.cam_id = {c: int(m.camera(c).id) for c in self.cameras}
        self.res = {c: _resolution(env, c) for c in self.cameras}
        self.plug_body = int(env.plug)
        self.socket_body = int(env.socket)
        self.plug_bodies = frozenset(_subtree_bodies(m, self.plug_body))
        self.socket_bodies = frozenset(_subtree_bodies(m, self.socket_body))
        # Housing: the largest box geom of the plug body, in the body frame.
        boxes = [g for g in range(m.ngeom) if m.geom_bodyid[g] == self.plug_body
                 and m.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX]
        g = max(boxes, key=lambda i: float(np.prod(m.geom_size[i])))
        self.housing_geom = int(g)
        self._house_pos = m.geom_pos[g].copy()
        gm = np.zeros(9)
        mujoco.mju_quat2Mat(gm, m.geom_quat[g])
        self._house_mat = gm.reshape(3, 3)
        self._house_size = m.geom_size[g].copy()
        self._long = int(np.argmax(self._house_size))
        self.socket_site = int(env.socket_site)

    # --- sample points -------------------------------------------------------------------------
    def plug_points(self, data=None):
        """(4, 3) world points: housing centre, top-face centre, front end, back end."""
        d = data if data is not None else self.env.data
        R = d.xmat[self.plug_body].reshape(3, 3)
        origin = d.xpos[self.plug_body]
        centre = self._house_pos
        axes = self._house_mat  # columns are the housing axes in the body frame
        half = self._house_size * INSET
        # top face: the housing axis most aligned with world +z, pointing up
        up = np.array([(R @ axes[:, k])[2] for k in range(3)])
        k = int(np.argmax(np.abs(up)))
        top = centre + np.sign(up[k]) * axes[:, k] * half[k]
        ln = axes[:, self._long] * half[self._long]
        local = np.array([centre, top, centre + ln, centre - ln])
        return origin + local @ R.T

    def socket_points(self, data=None):
        """(2, 3) world points: the socket entrance (``socket_entry`` site) and the body origin."""
        d = data if data is not None else self.env.data
        return np.array([d.site_xpos[self.socket_site], d.xpos[self.socket_body]])

    # --- visibility ----------------------------------------------------------------------------
    def _sees(self, camera, points, bodies, d):
        cid = self.cam_id[camera]
        w, h = self.res[camera]
        return points_visible(self.model, d, d.cam_xpos[cid], d.cam_xmat[cid].reshape(3, 3),
                              float(self.model.cam_fovy[cid]), w, h, points, bodies)

    def plug(self, data=None):
        d = data if data is not None else self.env.data
        pts = self.plug_points(d)
        return {c: bool(self._sees(c, pts, self.plug_bodies, d)) for c in self.cameras}

    def socket(self, data=None):
        d = data if data is not None else self.env.data
        pts = self.socket_points(d)
        return {c: bool(self._sees(c, pts, self.socket_bodies, d)) for c in self.cameras}

    def any_plug(self, data=None):
        return any(self.plug(data).values())


def pickup_coverage(flags_by_phase):
    """flags_by_phase: {phase: iterable of bool}. Returns per-phase {n, covered, fraction} plus
    ``_all`` over every frame and ``_complete`` (True when no frame is uncovered)."""
    out, n_all, c_all = {}, 0, 0
    for phase, flags in flags_by_phase.items():
        flags = [bool(x) for x in flags]
        n, c = len(flags), sum(flags)
        out[phase] = {"n": n, "covered": c, "fraction": (c / n) if n else None}
        n_all, c_all = n_all + n, c_all + c
    out["_all"] = {"n": n_all, "covered": c_all, "fraction": (c_all / n_all) if n_all else None}
    out["_complete"] = bool(n_all) and c_all == n_all
    return out
