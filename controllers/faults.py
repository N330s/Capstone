"""Injected faults for recovery demonstrations (scripts/collect_random.py --faults).

One deliberate mistake per episode, after which PickInsertExpert backs off and
finishes the task:

  premature_close    the fingers close part-way down the descend, above the plug;
                     an empty close is detected, the hand reopens, rises back to the
                     pregrasp and descends again.
  wrong_grasp        the first pick goes to empty table 6.5-9.5 cm away from the plug
                     (past the pin tips, or beside it) and closes on nothing; the empty
                     close is detected, the hand opens, retreats and re-plans the nominal
                     grasp. Never toward the cable end.
  misaligned_insert  the first align/insert aims a few mm off the socket holes, so the
                     pins hit the socket face; the existing force/stall rule withdraws,
                     re-aligns on the holes and inserts.

Every step of such an episode is labelled ``expert`` / ``fault`` / ``recovery``
(``PickInsertExpert.label``, saved as ``step_label``). Fault steps are injected bad
commands: they must be masked before behaviour cloning (plan.md); the current VLA
loaders do NOT do this yet.

Robot-free: drawing a fault needs only the scene seed. The draw uses its own random
stream, so the expert/detector randomness of the rollout is unchanged.
"""
from dataclasses import dataclass, field
import numpy as np

FAULT_KINDS = ("premature_close", "wrong_grasp", "misaligned_insert")
FAULT_STREAM = 0xFA17
LABELS = ("expert", "fault", "recovery")

# Parameter ranges (magnitudes; signs and directions are drawn separately).
PREMATURE_CLOSE_FRAC = (0.2, 0.6)        # fraction of the descend waypoints done before the close
# wrong_grasp misses the plug: the grasp point moves this far (from the housing grasp point) along a
# plug-frame direction. Toward the cable end (-x) the hand lands on the cable, which the pick planner
# ignores (smoke row 7101), so that direction is never used.
WRONG_GRASP_DIRECTIONS = {"past_pins": (1., 0.), "left": (0., 1.), "right": (0., -1.)}
WRONG_GRASP_DISTANCE_M = (0.065, 0.095)
# Socket-frame (y, z) aim error. Type O pins (4.8 mm) in 5.1 mm holes: >= ~1.5 mm lands on the face.
# Kept below PlaceInsert's 4 mm off-axis abort (TRACK_ERROR_LIMIT_M), which applies while approaching:
# smoke row 7201 drew 3.41 mm and measured 3.78 mm lateral at contact, so the cap is 3 mm.
MISALIGN_OFFSET_M = (0.002, 0.003)


@dataclass(frozen=True)
class FaultSpec:
    kind: str
    params: dict = field(default_factory=dict)

    def to_json(self):
        return {"kind": self.kind, "params": dict(self.params)}


def parse_kinds(text):
    """'all' or a comma-separated subset of FAULT_KINDS -> tuple (order of FAULT_KINDS)."""
    if text is None:
        return ()
    names = {s.strip() for s in text.split(",") if s.strip()}
    if names == {"all"}:
        return FAULT_KINDS
    unknown = names - set(FAULT_KINDS)
    if unknown or not names:
        raise ValueError(f"unknown fault kind(s) {sorted(unknown)}; choose from {FAULT_KINDS} or 'all'")
    return tuple(k for k in FAULT_KINDS if k in names)


def draw_fault(seed, kinds=FAULT_KINDS):
    """Deterministic fault for a scene seed. The kind cycles with the seed, so consecutive bank
    rows are balanced across ``kinds``; the parameters come from a separate stream."""
    kinds = tuple(kinds)
    if not kinds:
        return None
    kind = kinds[int(seed) % len(kinds)]
    rng = np.random.default_rng([int(seed), FAULT_STREAM])
    if kind == "premature_close":
        params = {"descend_frac": float(rng.uniform(*PREMATURE_CLOSE_FRAC))}
    elif kind == "wrong_grasp":
        names = tuple(WRONG_GRASP_DIRECTIONS)
        params = {"direction": names[int(rng.integers(len(names)))],
                  "distance_m": float(rng.uniform(*WRONG_GRASP_DISTANCE_M))}
    elif kind == "misaligned_insert":
        magnitude = float(rng.uniform(*MISALIGN_OFFSET_M))
        angle = float(rng.uniform(0., 2. * np.pi))
        params = {"offset_m": [magnitude * float(np.cos(angle)), magnitude * float(np.sin(angle))],
                  "magnitude_m": magnitude}
    else:
        raise ValueError(f"unknown fault kind {kind!r}")
    return FaultSpec(kind, params)


def label_counts(labels):
    labels = list(labels)
    return {name: labels.count(name) for name in LABELS}
