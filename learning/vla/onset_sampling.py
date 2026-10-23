"""Source-bound initial-wait ablation; never trims episodes or changes actor inputs."""
from pathlib import Path
import json

import numpy as np

from learning.vla.smolvla_data import CONTRACT, sha256
from learning.vla.window_training import CommandWindows
from learning.vla.pickup_onset import loss_components

SCHEMA = "openarm-initial-survey-anchor-ablation-v1"


def anchor_partition(phases):
    phases = np.asarray(phases)
    if phases.ndim != 1 or not len(phases):
        raise ValueError("Nonempty command phases required")
    active = np.flatnonzero(phases != "survey")
    if not len(active):
        raise ValueError("Episode contains no active command")
    start = int(active[0])
    return start, list(range(start, len(phases)))


def build_index(collection, export):
    collection, export = Path(collection).resolve(), Path(export).resolve()
    receipt_path = export / "openarm_export.json"
    receipt = json.loads(receipt_path.read_text())
    records = receipt["datasets"]["train"]["episodes"]
    validation = receipt["datasets"]["validation"]["episodes"]
    names = [r["episode"] for r in records]
    if (receipt["contract"] != CONTRACT or len(set(names)) != len(names)
            or set(names) & {r["episode"] for r in validation}):
        raise ValueError("Reviewed disjoint source/export required")
    eligible, excluded, rows, probes = [], [], [], []
    offset = 0
    probe_names = {names[round(i*(len(names)-1)/min(3, len(names)-1))]
                   for i in range(min(4, len(names)))} if len(names) > 1 else set(names)
    for record in records:
        folder = (collection / record["episode"]).resolve()
        if folder.parent != collection or record["split"] != "train":
            raise ValueError("Unsafe or non-training parent")
        data_path, meta_path = folder / "episode.npz", folder / "metadata.json"
        meta = json.loads(meta_path.read_text())
        if sha256(data_path) != record["data_sha256"] or meta["data_sha256"] != record["data_sha256"]:
            raise ValueError("Recording checksum differs from export")
        with np.load(data_path, allow_pickle=False) as arrays:
            phase, state, action = arrays["phase"], arrays["observation_state"], arrays["action_applied"]
            n = record["steps"]
            if len(phase) != n or state.shape != (n+1, 16) or action.shape != (n, 8):
                raise ValueError("Command/T+1 observation alignment changed")
            start, local = anchor_partition(phase)
            if not start or start+50 > n or not np.isfinite(state).all() or not np.isfinite(action).all():
                raise ValueError("Finite source with an initial wait and a full active chunk required")
            delta = float(np.abs(state[:start+1]-state[0]).max())
            if delta > 1e-6:
                raise ValueError("Initial survey is not stationary; review before exclusion")
            if record["episode"] in probe_names:
                closes = np.flatnonzero(phase == "close")
                if not len(closes) or closes[0] < 50:
                    raise ValueError("Source close phase missing")
                anchors = {"original_wait_reset": 0, "first_active": start,
                           "approach": min(200, int(closes[0])-50),
                           "preclose": int(closes[0])-25}
                for label, step in anchors.items():
                    probes.append({"index": offset+step, "episode": record["episode"], "step": step,
                                   "category": label, "source_phase": str(phase[step])})
            row = {"episode": record["episode"], "steps": n, "global_start": offset,
                   "excluded_initial_commands": start, "data_sha256": record["data_sha256"],
                   "metadata_sha256": sha256(meta_path), "reset_to_first_active_state_max_abs": delta,
                   "first_active_requested_target": action[start].tolist()}
        rows.append(row)
        eligible.extend(offset+i for i in local)
        excluded.extend(range(offset, offset+start))
        offset += n
    if sorted(eligible+excluded) != list(range(offset)):
        raise ValueError("Training partitions must be disjoint and exhaustive")
    return {"schema": SCHEMA, "export_receipt_sha256": sha256(receipt_path),
            "total_train_commands": offset, "eligible": eligible, "excluded": excluded,
            "parents": rows, "phase_probes": probes,
            "actor_inputs_changed": False, "frames_removed": False,
            "definition": "Only leading survey anchors excluded; chunks keep original global indices, future actions and terminal mask. Later survey anchors remain eligible."}


def verify_index(index, collection, export):
    if index.get("schema") != SCHEMA or index != build_index(collection, export):
        raise ValueError("Sampling index differs from its immutable source/export")


def sample_indices(rng, count, total, eligible=None):
    if type(count) is not int or count < 1 or type(total) is not int or total < 1:
        raise ValueError("Positive batch and dataset lengths required")
    if eligible is None:
        return [rng.randrange(total) for _ in range(count)]
    if not eligible:
        raise ValueError("No eligible anchors")
    return [eligible[rng.randrange(len(eligible))] for _ in range(count)]


def audited_forward(policy, batch, forward, **kwargs):
    """Capture no-grad evaluation losses without changing the native objective."""
    original, captured = policy.model.forward, []

    def capture(*args, **inner):
        losses = original(*args, **inner)
        captured.append(losses.detach().float().cpu().numpy())
        return losses

    policy.model.forward = capture
    try:
        native, aux = forward(policy, batch, **kwargs)
    finally:
        policy.model.forward = original
    if len(captured) != 1:
        raise ValueError("Expected one native per-element loss tensor")
    audit = loss_components(captured[0], batch["action_is_pad"].detach().cpu().numpy())
    if abs(float(native)-audit["native_mean"]) > 1e-6:
        raise ValueError("Native loss reconstruction failed")
    return native, aux, audit


def chunk_errors(prediction, target, valid):
    prediction, target, valid = np.asarray(prediction), np.asarray(target), np.asarray(valid, dtype=bool)
    if prediction.shape != target.shape or prediction.shape != (50, 8) or valid.shape != (50,):
        raise ValueError("A 50 by 8 action chunk and explicit terminal mask required")
    result = {}
    for prefix in (1, 25, 50):
        error = np.abs(prediction[:prefix]-target[:prefix])[valid[:prefix]]
        if not len(error):
            raise ValueError("No valid action target")
        result[str(prefix)] = {"joint_mae_rad": error[:, :7].mean(0).tolist(),
                              "gripper_mae_mm": float(error[:, 7].mean()*1000)}
    return result
