"""Read-only onset diagnostics and exact environment target-filter arithmetic."""
import numpy as np


def filtered_targets(requests, initial, lower, upper, dt=.02, joint_rate=.5, finger_rate=.02):
    requests = np.asarray(requests, dtype=float)
    target = np.asarray(initial, dtype=float).copy()
    if requests.ndim != 2 or requests.shape[1] != 8 or target.shape != (8,) or not np.isfinite(requests).all():
        raise ValueError("Finite 8-target commands required")
    delta = np.r_[np.full(7, joint_rate*dt), finger_rate*dt]
    result = []
    for command in requests:
        bounded = np.r_[np.clip(command[:7], lower, upper), np.clip(command[7], 0, .044)]
        target = target + np.clip(bounded-target, -delta, delta)
        result.append(target.copy())
    return np.asarray(result)


def loss_components(losses, is_pad, physical_dimensions=8):
    losses = np.asarray(losses)
    is_pad = np.asarray(is_pad, dtype=bool)
    if losses.ndim != 3 or is_pad.shape != losses.shape[:2] or not 0 < physical_dimensions < losses.shape[-1]:
        raise ValueError("Per-element full-width losses and time padding mask required")
    valid = ~is_pad
    if not valid.any() or not np.isfinite(losses).all():
        raise ValueError("Nonempty finite loss required")
    masked = losses * valid[..., None]
    native = float(masked.mean())
    actual = float(masked[..., :physical_dimensions].mean())
    padded = float(masked[..., physical_dimensions:].mean())
    fraction = physical_dimensions/losses.shape[-1]
    return {"native_mean": native, "actual_dimensions_mean": actual,
        "padded_dimensions_mean": padded, "actual_dimension_count": physical_dimensions,
        "padded_dimension_count": losses.shape[-1]-physical_dimensions,
        "physical_dimension_weight": fraction,
        "physical_share_of_total_loss": (actual*fraction/native) if native else 0.,
        "reconstruction_error": abs(native-(fraction*actual+(1-fraction)*padded)),
        "note": "Native fixed-size time denominator retained for exact accounting; not a proposed training-loss change."}
