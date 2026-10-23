"""Offline-only correction indices and deterministic, quota-based BC sampling.

Labels select examples; they never enter the policy observation. Dataset indices
refer to contiguous non-fault segments, not a time axis with faults deleted.
"""
from collections import Counter, defaultdict
import json
import math
from pathlib import Path

import numpy as np

from learning.vla.recovery_data import RECOVERY_CONTRACT
from learning.vla.smolvla_data import sha256
from learning.vla.window_training import CommandWindows

SCHEMA = "openarm-recovery-correction-index-v1"
CATEGORIES = ("active_reopening", "other_recovery", "expert")


def correction_categories(labels, states, actions, threshold_mm=.2):
    labels, states, actions = map(np.asarray, (labels, states, actions))
    if (labels.ndim != 1 or states.shape != (len(labels) + 1, 16)
            or actions.shape != (len(labels), 8)
            or set(labels.tolist()) - {"expert", "fault", "recovery"}
            or not np.isfinite(states).all() or not np.isfinite(actions).all()
            or not math.isfinite(threshold_mm) or threshold_mm <= 0):
        raise ValueError("Invalid correction labels/state/action/threshold")
    categories = np.where(labels == "expert", "expert", "other_recovery").astype("U20")
    categories[labels == "fault"] = "fault"
    opening = (labels == "recovery") & ((actions[:, 7] - states[:-1, 15]) * 1000 > threshold_mm)
    categories[opening] = "active_reopening"
    return categories


def build_index(view, export, threshold_mm=.2):
    view, export = Path(view), Path(export)
    receipt_path = export / "openarm_export.json"
    receipt = json.loads(receipt_path.read_text())
    if (receipt["contract"] != RECOVERY_CONTRACT
            or receipt["source_manifest_sha256"] != sha256(view / "manifest.json")):
        raise ValueError("Recovery view/export binding mismatch")
    parents, partitions, coverage = {}, {}, []
    parent_splits = receipt["parent_splits"]
    if set(parent_splits["train"]) & set(parent_splits["validation"]):
        raise ValueError("Parent split leakage")
    for split in ("train", "validation"):
        pools = {key: [] for key in CATEGORIES}
        offset = 0
        for record in receipt["datasets"][split]["episodes"]:
            parent = record["parent_episode"]
            if record["split"] != split or parent not in parent_splits[split]:
                raise ValueError("Segment parent split mismatch")
            if parent not in parents:
                path = view / parent / "episode.npz"
                if sha256(path) != record["parent_data_sha256"]:
                    raise ValueError("Raw recording checksum mismatch")
                with np.load(path, allow_pickle=False) as arrays:
                    parents[parent] = correction_categories(arrays["step_label"],
                        arrays["observation_state"], arrays["action_applied"], threshold_mm)
            start, stop = record["parent_start"], record["parent_stop_exclusive"]
            categories = parents[parent][start:stop]
            if stop - start != record["steps"] or len(categories) != record["steps"] or "fault" in categories:
                raise ValueError("Fault or invalid range inside eligible segment")
            for category in CATEGORIES:
                pools[category].extend((np.flatnonzero(categories == category) + offset).tolist())
            coverage.append({"split": split, "parent": parent, "segment": record["episode"],
                             "counts": dict(Counter(categories.tolist()))})
            offset += record["steps"]
        if sorted(i for pool in pools.values() for i in pool) != list(range(offset)):
            raise ValueError("Correction partitions must cover each eligible command exactly once")
        partitions[split] = pools
    return {"schema": SCHEMA, "export_receipt_sha256": sha256(receipt_path),
            "threshold_mm": threshold_mm, "parent_splits": parent_splits,
            "partitions": partitions, "segment_coverage": coverage,
            "definition": "recovery label AND applied gripper target exceeds current measured travel by > threshold; not a mechanical limit",
            "actor_inputs_changed": False}


def verify_index(index, view, export):
    if index.get("schema") != SCHEMA or index != build_index(view, export, index["threshold_mm"]):
        raise ValueError("Correction index differs from its source/export")


class CorrectionSampler:
    """At least 25% active opening, ~25% other recovery, rest expert, with replacement."""

    def __init__(self, pools, total):
        self.pools = pools
        flat = [i for category in CATEGORIES for i in pools[category]]
        if sorted(flat) != list(range(total)) or any(type(i) is not int for i in flat):
            raise ValueError("Disjoint exhaustive integer training partitions required")
        if any(not pools[category] for category in CATEGORIES):
            raise ValueError("Each training category needs examples; no silent quota fallback")
        self.categories = {i: category for category in CATEGORIES for i in pools[category]}

    def sample(self, rng, batch_size):
        if type(batch_size) is not int or batch_size < 4:
            raise ValueError("Correction-aware batches require at least four commands")
        opening = math.ceil(batch_size / 4)
        recovery = batch_size // 4
        quotas = (opening, recovery, batch_size - opening - recovery)
        indices = [rng.choice(self.pools[category]) for category, count in zip(CATEGORIES, quotas)
                   for _ in range(count)]
        rng.shuffle(indices)
        return indices

    def exposure(self, indices):
        counts = Counter(self.categories[i] for i in indices)
        return {category: counts[category] for category in CATEGORIES}


def correction_probes(index, records, split, category, per_parent=2):
    """Fixed parent-balanced probes, chosen before any candidate outcomes."""
    windows = CommandWindows(records)
    by_parent = defaultdict(list)
    parent_of = {r["episode"]: r["parent_episode"] for r in records}
    for i in index["partitions"][split][category]:
        sample = windows.locate(i)
        by_parent[parent_of[sample["episode"]]].append(sample)
    result = []
    for parent, samples in sorted(by_parent.items()):
        positions = sorted({min(len(samples) - 1, int((k + .5) * len(samples) / per_parent))
                            for k in range(per_parent)})
        result.extend({**samples[p], "parent_episode": parent, "category": category} for p in positions)
    return result
