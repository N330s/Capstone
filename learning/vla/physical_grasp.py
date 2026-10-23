"""Versioned physical grasp bookkeeping; no forces, motion or guard changes."""
from dataclasses import dataclass
import math
import sys
from types import MethodType


@dataclass
class PhysicalGraspTracker:
    settle_s: float = .1
    still_m_s: float = .005
    squeeze_m: float = .001
    pending_s: float = 0.
    latched: bool = False

    def update(self, *, contacts, speed, pad_min, command, slip, slip_limit, dt):
        if (contacts not in (0, 1, 2) or not all(math.isfinite(x) for x in
                (speed, pad_min, command, slip, slip_limit, dt)) or dt <= 0 or slip_limit <= 0):
            raise ValueError("Invalid physical grasp measurement")
        if self.latched:
            if contacts < 2 and command > pad_min + 1e-4:
                self.latched, self.pending_s = False, 0.
                return "release"
            if slip > slip_limit:
                self.latched, self.pending_s = False, 0.
                return "drop"
            return None
        candidate = contacts == 2 and speed < self.still_m_s and command < pad_min - self.squeeze_m
        self.pending_s = self.pending_s + dt if candidate else 0.
        if self.pending_s + 1e-12 >= self.settle_s:
            self.latched, self.pending_s = True, 0.
            return "latch"
        return None


def install_physical_grasp_metrics(env):
    """Explicit adapter over the hash-verified collector environment.

    Keep the collector checkout immutable. This changes bookkeeping only, so the
    caller must declare the extension in its manifest/report and replay-qualify it.
    """
    import numpy as np
    from learning.vla.smolvla_data import sha256
    module = sys.modules[env.__class__.__module__]
    settle = module.TABLETOP_GRASP_SETTLE_S
    still = module.TABLETOP_FINGER_STILL_M_S
    squeeze = module.TABLETOP_SQUEEZE_M
    original_reset, original_update = env.reset, env._update_tabletop_grasp

    def reset(instance, *args, **kwargs):
        instance._physical_tracker = PhysicalGraspTracker(settle, still, squeeze)
        return original_reset(*args, **kwargs)

    def update(instance, info):
        if instance.mode != "tabletop":
            return original_update(info)
        event = instance._physical_tracker.update(contacts=info["plug_finger_contacts"],
            speed=float(np.max(np.abs(instance.data.qvel[instance.fva["right"]]))),
            pad_min=float(np.min(instance.data.qpos[instance.fqa["right"]])), command=float(instance.target[7]),
            slip=info["grasp_slip_m"], slip_limit=instance.config["grasp_slip_limit_m"], dt=instance.model.opt.timestep)
        if event == "latch":
            instance._grasp_reference = instance.plug_in_hand()
            instance._latched_finger_command = float(instance.target[7])
            instance.grasp_established = True
            info["grasp_slip_m"], info["grasp_angle_deg"] = 0., 0.
        elif event in ("drop", "release"):
            instance.grasp_established = False
            instance._grasp_reference = None
            instance._latched_finger_command = None
            if event == "drop":
                instance.drops += 1

    env.reset = MethodType(reset, env)
    env._update_tabletop_grasp = MethodType(update, env)
    return {"revision": "physical-contact-settle-v1", "extension_sha256": sha256(__file__),
        "settle_s": settle, "still_m_s": still, "squeeze_m": squeeze,
        "target_jitter_resets_timer": False, "physics_or_guard_thresholds_changed": False}
