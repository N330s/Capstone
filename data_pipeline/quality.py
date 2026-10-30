"""Physical-quality check of one rollout: pickup, carry and insertion judged separately.

`steps` is one small dict per command step (accumulated by scripts/collect_random.rollout):
phase, plug_finger_contacts, grasp_established, drops, plug_z, label ("expert" if absent).
Robot-free, so it is unit-testable with synthetic lists.

 pickup     a grasp latch existed (grasp_established at some step) and the plug rose at least
            min_lift_m above its starting z by the end of the (first) lift phase.
 carry      from the first lift step to the end: both fingers touch the plug and the latch holds on
            every step, and no drop was counted. Opening the hand before the first lift (the empty
            closes of fault episodes) is not a carry failure. The expert keeps holding through
            `hold` (valid_pose needs the latch), so contact is required through the final step.
 insertion  the env outcome is "success"; any abort outcome fails it.
"""
from __future__ import annotations

LIFT_PHASE = "lift"


def quality_check(steps, outcome, *, min_lift_m=0.03, start_z=None):
    reasons = []
    steps = list(steps)
    latch_step = next((i for i, s in enumerate(steps) if s.get("grasp_established")), None)
    lift_idx = [i for i, s in enumerate(steps) if s.get("phase") == LIFT_PHASE]
    first_lift = lift_idx[0] if lift_idx else None
    z0 = start_z if start_z is not None else (steps[0]["plug_z"] if steps else None)
    lift_height = None
    if first_lift is not None and z0 is not None:
        run_end = first_lift
        while run_end + 1 < len(steps) and steps[run_end + 1].get("phase") == LIFT_PHASE:
            run_end += 1
        lift_height = float(steps[run_end]["plug_z"] - z0)

    pickup_ok = latch_step is not None and lift_height is not None and lift_height >= min_lift_m
    if latch_step is None:
        reasons.append("quality_no_grasp")
    if lift_height is None or lift_height < min_lift_m:
        reasons.append("quality_low_lift")

    drops = contact_loss = 0
    if first_lift is not None:
        carry = steps[first_lift:]
        drops = int(max(int(s.get("drops", 0)) for s in carry))
        contact_loss = sum(1 for s in carry
                           if s.get("plug_finger_contacts") != 2 or not s.get("grasp_established"))
    carry_ok = first_lift is not None and drops == 0 and contact_loss == 0
    if drops:
        reasons.append("quality_drop")
    if contact_loss:
        reasons.append("quality_contact_lost")
    insertion_ok = outcome == "success"
    if not insertion_ok:
        reasons.append("quality_insertion")
    return {"ok": not reasons, "reasons": reasons,
            "stage_outcomes": {
                "pickup": {"success": bool(pickup_ok), "latch_step": latch_step,
                           "lift_height_m": lift_height},
                "carry": {"success": bool(carry_ok), "drops": drops,
                          "contact_loss_steps": int(contact_loss)},
                "insertion": {"success": bool(insertion_ok), "outcome": outcome}},
            "min_lift_m": min_lift_m}
