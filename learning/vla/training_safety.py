"""Conservative on-disk budget for full FP32 expert + Adam checkpoints."""
import math

GIB = 2**30


def checkpoint_budget_bytes(model_bytes, trainable_bytes):
    if model_bytes < 1 or not 0 < trainable_bytes <= model_bytes:
        raise ValueError("Positive model/trainable byte counts required")
    # Full model plus two FP32 Adam moments, tensor/container metadata and slack.
    return math.ceil(1.10 * (model_bytes + 2 * trainable_bytes)) + 64 * 2**20


def require_checkpoint_space(free_bytes, checkpoint_bytes, reserve_gib=25.):
    if not math.isfinite(reserve_gib) or reserve_gib < 25 or checkpoint_bytes < 1:
        raise ValueError("At least 25 GiB reserve and a positive checkpoint estimate required")
    if free_bytes < reserve_gib * GIB + checkpoint_bytes:
        raise RuntimeError("Insufficient checkpoint space above the protected disk reserve")


def training_space_available(free_bytes, checkpoint_bytes, reserve_gib=25., *, final_only=False):
    require_checkpoint_space(float("inf"), checkpoint_bytes, reserve_gib)
    if type(final_only) is not bool:
        raise ValueError("Explicit boolean checkpoint policy required")
    # Ordinary runs need room for a scheduled save plus final. A strictly
    # final-only run cannot create that scheduled copy, so reserve exactly one.
    copies = 1 if final_only else 2
    return free_bytes >= reserve_gib * GIB + copies * checkpoint_bytes
