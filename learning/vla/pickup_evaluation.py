"""Explicit pickup-only evaluation gate; never generates robot commands."""
from dataclasses import dataclass
import math


@dataclass
class PickupGate:
    initial_plug_z_m: float
    lift_m: float = .020
    hold_s: float = 1.
    consecutive_s: float = 0.
    maximum_s: float = 0.
    maximum_lift_m: float = 0.
    previous_valid: bool = False
    passed: bool = False

    def __post_init__(self):
        if not all(math.isfinite(x) for x in (self.initial_plug_z_m, self.lift_m, self.hold_s)) or min(self.lift_m, self.hold_s) <= 0:
            raise ValueError("Finite positive pickup thresholds required")

    def update(self, *, plug_z_m, grasp, contacts, drops, safe, dt):
        if not math.isfinite(plug_z_m) or not math.isfinite(dt) or dt <= 0 or contacts not in (0, 1, 2) or drops < 0:
            raise ValueError("Invalid physical pickup measurements")
        lift = plug_z_m - self.initial_plug_z_m
        self.maximum_lift_m = max(self.maximum_lift_m, lift)
        valid = bool(safe and grasp and contacts == 2 and drops == 0 and lift >= self.lift_m)
        # Both interval endpoints must qualify; don't credit the interval that
        # first crosses the height/contact threshold.
        self.consecutive_s = self.consecutive_s + dt if valid and self.previous_valid else 0.
        self.maximum_s = max(self.maximum_s, self.consecutive_s)
        self.previous_valid = valid
        self.passed = self.passed or (valid and self.consecutive_s + 1e-10 >= self.hold_s)
        return self.passed

    def definition(self):
        return {"revision": "pickup-lift-carry-v1", "initial_plug_z_m": self.initial_plug_z_m,
            "minimum_lift_m": self.lift_m, "consecutive_carry_s": self.hold_s,
            "two_finger_contacts_required": True, "no_prior_drops_required": True,
            "stationary_hold_required": False, "physics_or_commands_changed": False}


def select_validation_parents(receipt, count=5):
    """Sorted, evenly spaced selection fixed before observing any outcomes."""
    train = receipt["datasets"]["train"]["episodes"]
    heldout = sorted(receipt["datasets"]["validation"]["episodes"], key=lambda r: r["episode"])
    if (not 1 < count <= len(heldout) or len({r["episode"] for r in heldout}) != len(heldout)
            or {r["episode"] for r in train} & {r["episode"] for r in heldout}
            or any(r["split"] != "validation" for r in heldout)):
        raise ValueError("Unique, disjoint validation parents required")
    return [heldout[round(i * (len(heldout) - 1) / (count - 1))] for i in range(count)]


def wilson_interval(successes, trials, z=1.959963984540054):
    """Descriptive 95% interval by independent reset, not by video frame."""
    if trials < 1 or not 0 <= successes <= trials:
        raise ValueError("Valid nonempty binomial counts required")
    p = successes / trials
    denominator = 1 + z * z / trials
    center = (p + z * z / (2 * trials)) / denominator
    radius = z * math.sqrt(p * (1 - p) / trials + z * z / (4 * trials * trials)) / denominator
    return [max(0., center - radius), min(1., center + radius)]


def audit_pickup_trace(report, trace, diagnostics):
    """Recompute the outcome from saved physical motion, not the grasp flag alone."""
    import numpy as np
    definition = report["pickup_evaluation"]
    count = report["steps"]
    qpos = trace["qpos"]
    address = definition["plug_qpos_address"]
    if (len(diagnostics) != count + 1 or qpos.shape[0] != count + 1
            or trace["observation_state"].shape != (count + 1, 16)
            or trace["action_requested"].shape != (count, 8)
            or trace["executed_dt_s"].shape != (count,)
            or not 0 <= address < qpos.shape[1] - 2
            or not all(np.isfinite(trace[k]).all() for k in ("qpos", "observation_state", "action_requested", "executed_dt_s"))
            or not np.array_equal(trace["policy_action_raw"], trace["action_requested"])):
        raise ValueError("Invalid or filtered physical pickup trace")
    gate = PickupGate(float(qpos[0, address + 2]))
    if any(definition.get(k) != v for k, v in gate.definition().items()):
        raise ValueError("Pickup definition changed")
    first = None
    for i, info in enumerate(diagnostics[1:], 1):
        passed = gate.update(plug_z_m=float(qpos[i, address + 2]), grasp=info["grasp_established"],
            contacts=info["plug_finger_contacts"], drops=info["drops"], safe=info["outcome"] in ("running", "success"),
            dt=float(trace["executed_dt_s"][i - 1]))
        if passed and first is None:
            first = i
    if (bool(definition["passed"]) != gate.passed or gate.passed != (report["outcome"] == "pickup_success")
            or (gate.passed and first != count)
            or abs(gate.maximum_s - definition["maximum_consecutive_carry_s"]) > 1e-8
            or abs(gate.maximum_lift_m - definition["maximum_lift_m"]) > 1e-8):
        raise ValueError("Recorded pickup outcome differs from physical trace")
    return {"pickup_success": gate.passed, "first_grasp_command": next(
        (i for i, d in enumerate(diagnostics) if d["grasp_established"]), None),
        "success_command": first, "maximum_lift_mm": gate.maximum_lift_m * 1000,
        "maximum_continuous_carry_s": gate.maximum_s, "drops": diagnostics[-1]["drops"]}
