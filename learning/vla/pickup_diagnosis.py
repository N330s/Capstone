"""Pure command-index diagnostics; never changes actor observations or actions."""
import numpy as np


def unfold_prefixes(anchors, chunks, total, prefix=25):
    anchors, chunks = np.asarray(anchors), np.asarray(chunks)
    expected = np.arange(0, total, prefix)
    if not np.array_equal(anchors, expected) or chunks.shape != (len(expected), 50, 8) or not np.isfinite(chunks).all():
        raise ValueError("Complete ordered, finite 50x8 chunks required")
    return np.concatenate([c[:min(prefix, total-int(a))] for a, c in zip(anchors, chunks)])


def first_sustained(mask, minimum=10):
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 1 or minimum < 1:
        raise ValueError("One-dimensional mask and positive duration required")
    count = 0
    for i, value in enumerate(mask):
        count = count + 1 if value else 0
        if count >= minimum:
            return i-minimum+1
    return None


def command_errors(predicted, expert, phases):
    predicted, expert, phases = np.asarray(predicted), np.asarray(expert), np.asarray(phases)
    if predicted.shape != expert.shape or predicted.shape != (len(phases), 8):
        raise ValueError("Commands and phase diagnostics must align at the same index")
    before_close = np.arange(len(phases)) < int(np.flatnonzero(phases == "close")[0])
    absolute = np.abs(predicted-expert)
    opened = before_close & (expert[:, 7] >= .034)
    return {"joint_mae_rad": float(absolute[:, :7].mean()),
        "preclose_joint_mae_rad": float(absolute[before_close, :7].mean()),
        "gripper_mae_mm": float(1000*absolute[:, 7].mean()),
        "expert_open_gripper_mae_mm": float(1000*absolute[opened, 7].mean()),
        "first_10_command_premature_close": first_sustained(opened & (predicted[:, 7] < .030)),
        "first_gripper_below_30mm": next((int(i) for i in np.flatnonzero(predicted[:, 7] < .030)), None),
        "expert_first_gripper_below_30mm": next((int(i) for i in np.flatnonzero(expert[:, 7] < .030)), None)}
