"""Recovery BC: split by parent episode, then segment around excluded faults.

Segments are virtual contiguous recordings, never stitched across a disturbance.
The existing two-camera position-only actor and upstream padding are unchanged.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from learning.vla.smolvla_data import SmolVLASource, sha256

RECOVERY_CONTRACT = "openarm-v1-smolvla-recovery-segments-v1"
SPLIT_SCHEMA = "openarm-recovery-parent-splits-v1"


def valid_segments(labels):
    labels = np.asarray(labels)
    if labels.ndim != 1 or not len(labels) or labels.dtype.kind not in "US":
        raise ValueError("Nonempty one-dimensional string step_label required")
    if set(labels.tolist()) - {"expert", "fault", "recovery"}:
        raise ValueError("Unknown recovery label")
    good = labels != "fault"
    starts = np.flatnonzero(good & np.r_[True, ~good[:-1]])
    stops = np.flatnonzero(good & np.r_[~good[1:], True]) + 1
    return list(zip(starts.tolist(), stops.tolist()))


class RecoveryVLASource(SmolVLASource):
    supports_recovery_labels = True

    def __init__(self, root, selection=None):
        super().__init__(root, selection)
        self.parent_records = self.records
        self.segment_records, self.segment_by_name, self.label_counts = [], {}, {}
        for parent in self.parent_records:
            name = parent["episode"]
            with np.load(self.root / name / "episode.npz", allow_pickle=False) as arrays:
                if "step_label" not in arrays.files:
                    raise ValueError("Recovery source requires explicit step_label on every parent")
                labels = arrays["step_label"]
            if labels.shape != (parent["steps"],):
                raise ValueError("step_label must describe exactly T action intervals")
            segments = valid_segments(labels)
            counts = dict(Counter(labels.tolist()))
            for expected in (parent.get("label_counts"), self.metadata[name].get("label_counts")):
                if expected is None or counts != expected:
                    raise ValueError("Manifest/metadata/array label counts must agree")
            if not segments:
                raise ValueError("Parent has no eligible expert/recovery commands")
            self.label_counts[name] = counts
            for ordinal, (start, stop) in enumerate(segments):
                record = {"episode": f"{name}__valid_{ordinal:03d}", "split": parent["split"],
                          "steps": stop - start, "parent_episode": name,
                          "parent_start": start, "parent_stop_exclusive": stop,
                          "parent_data_sha256": parent["data_sha256"],
                          "collection": parent.get("collection"),
                          "boundary": "fault" if stop < parent["steps"] else "recorded_terminal"}
                self.segment_records.append(record)
                self.segment_by_name[record["episode"]] = record
        self.records = self.segment_records

    def arrays(self, name):
        record = self.segment_by_name[name]
        parent = super().arrays(record["parent_episode"])
        start, stop = record["parent_start"], record["parent_stop_exclusive"]
        return {key: array[start:stop if key == "action_applied" else stop + 1]
                for key, array in parent.items()}

    def report(self):
        report = super().report()
        totals = Counter()
        for counts in self.label_counts.values():
            totals.update(counts)
        return {**report, "contract": RECOVERY_CONTRACT,
                "parent_splits": {split: [r["episode"] for r in self.parent_records if r["split"] == split]
                                  for split in ("train", "validation")},
                "parent_label_counts": self.label_counts, "total_label_counts": dict(totals),
                "excluded_fault_commands": totals["fault"],
                "chunk_rule": "Stop at the first fault or recorded terminal; repeat last valid target and mask padding",
                "segment_boundaries_are_task_success": False,
                "auxiliary_channels": "Raw depth, chest RGB, torque and labels retained; none are actor inputs",
                "partial_release": self.manifest.get("partial_release", False)}


def make_split_plan(manifests, validation_fraction=.2):
    if not 0 < validation_fraction < .5:
        raise ValueError("Validation fraction must be between zero and one half")
    sources, records, names, seeds, hashes = [], [], set(), set(), set()
    fingerprint = None
    for path in sorted(map(lambda p: Path(p).resolve(), manifests)):
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if manifest.get("schema") != "openarm-pilot-v1" or not manifest.get("complete"):
            raise ValueError("Complete recovery source manifest required")
        collection = path.parent.name
        robot = manifest["robot_manifest"]
        current = json.dumps({k: robot[k] for k in
                              ("robot", "robot_config", "scene_sha256", "source_hashes", "physics_timestep_s")}
                             | {"instruction": manifest["instruction"]}, sort_keys=True)
        if fingerprint is not None and current != fingerprint:
            raise ValueError("Mixed recovery environment releases")
        fingerprint = current
        episodes = manifest["episodes"]
        if len(episodes) < 2:
            raise ValueError("At least two parents per recovery collection required")
        rank = sorted((r["episode"] for r in episodes), key=lambda n: hashlib.sha256(
            ("openarm-recovery-holdout-v1:" + collection + ":" + n).encode()).hexdigest())
        holdout = set(rank[:max(1, round(len(rank) * validation_fraction))])
        sources.append({"collection": collection, "manifest": str(path), "sha256": sha256(path)})
        for record in episodes:
            name, seed, digest = record["episode"], record["seed"], record["data_sha256"]
            if (Path(name).name != name or name in (".", "..") or ":" in name or "\\" in name
                    or name in names or seed in seeds or digest in hashes
                    or not 1000000 <= seed < 9000000 or record["split"] not in ("train", "validation")):
                raise ValueError("Unsafe/overlapping parent, evaluation seed, or unknown source split")
            if record["split"] == "validation" and name not in holdout:
                raise ValueError("Do not move existing validation parents into training")
            names.add(name)
            seeds.add(seed)
            hashes.add(digest)
            records.append({**record, "collection": collection,
                            "split": "validation" if name in holdout else "train"})
    if not records:
        raise ValueError("No recovery recordings")
    return {"schema": SPLIT_SCHEMA, "sources": sources,
            "validation_fraction": validation_fraction, "episodes": records,
            "rule": "SHA256-ranked whole-parent holdout independently per recovery collection",
            "instruction": json.loads(Path(sources[0]["manifest"]).read_text())["instruction"]}


def prepare_view(plan, roots, output, *, allow_partial=False):
    """Create a new hard-linked view; never alter a source manifest or recording."""
    if plan.get("schema") != SPLIT_SCHEMA:
        raise ValueError("Unknown split plan")
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    roots = [Path(root).resolve() for root in roots]
    if any(output.is_relative_to(root) or root.is_relative_to(output) for root in roots):
        raise ValueError("Input/output paths must be disjoint")
    for source in plan["sources"]:
        if sha256(source["manifest"]) != source["sha256"]:
            raise ValueError("Original manifest changed after split planning")
    resolved, missing = [], []
    for record in plan["episodes"]:
        candidates = [root / record["episode"] for root in roots
                      if root.name == record["collection"] and (root / record["episode"] / "episode.npz").is_file()]
        if len(candidates) > 1:
            raise ValueError("Ambiguous parent location")
        if not candidates:
            missing.append(record["episode"])
            continue
        path = candidates[0].resolve()
        if path.parent not in roots:
            raise ValueError("Parent resolves outside a declared collection")
        meta = json.loads((path / "metadata.json").read_text())
        if meta["data_sha256"] != record["data_sha256"] or sha256(path / "episode.npz") != record["data_sha256"]:
            raise ValueError("Incomplete or changed parent recording")
        resolved.append((record, path))
    if missing and not allow_partial:
        raise ValueError(f"Missing {len(missing)} parents; full release cannot be prepared")
    if {r["split"] for r, _ in resolved} != {"train", "validation"}:
        raise ValueError("Downloaded parents must include train and held-out validation")
    output.mkdir(parents=True)
    for record, path in resolved:
        target = output / record["episode"]
        target.mkdir()
        for name in ("episode.npz", "metadata.json", "diagnostics.json"):
            if (path / name).is_file():
                os.link(path / name, target / name)
    manifest = {"schema": "openarm-pilot-v1", "instruction": plan["instruction"],
                "complete": not missing, "partial_release": bool(missing),
                "episodes": [record for record, _ in resolved], "excluded_missing_parents": missing,
                "split_plan": plan, "source_files": "Immutable hard links; do not edit either view"}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
