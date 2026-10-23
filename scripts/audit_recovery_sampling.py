"""Trace actual optimizer samples to corrective targets; never trains."""
import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np


def opening_mask(labels, states, actions, threshold_mm=.2):
    """Diagnostic threshold distinguishes deliberate reopening from servo settling."""
    return (labels == "recovery") & ((actions[:, 7] - states[:-1, 15]) * 1000 > threshold_mm)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training", type=Path, default=Path("outputs/recovery_pipeline_20261019/pilot_v1/training"))
    parser.add_argument("--view", type=Path, default=Path("data/recovery_pipeline_20261019/qualification_view"))
    parser.add_argument("--export", type=Path, default=Path("data/recovery_pipeline_20261019/qualification_export"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    receipt = json.loads((args.export / "openarm_export.json").read_text())
    records = {r["episode"]: r for r in receipt["datasets"]["train"]["episodes"]}
    raw = {}
    availability = []
    for name in sorted({r["parent_episode"] for r in records.values()}):
        with np.load(args.view / name / "episode.npz", allow_pickle=False) as arrays:
            raw[name] = {key: arrays[key] for key in ("phase", "step_label", "observation_state", "action_applied")}
        values = raw[name]
        values["opening"] = opening_mask(values["step_label"], values["observation_state"], values["action_applied"])
        availability.append({"parent": name, "recovery_commands": int(np.sum(values["step_label"] == "recovery")),
                             "active_reopening_commands": int(values["opening"].sum())})
    events = [json.loads(line) for line in (args.training / "events.jsonl").read_text().splitlines()]
    counts, recovery_rows = Counter(), []
    for event in events:
        if event["type"] != "update":
            continue
        for sample in event["samples"]:
            record = records[sample["episode"]]
            start = record["parent_start"] + sample["step"]
            stop = min(start + 50, record["parent_stop_exclusive"])
            values = raw[record["parent_episode"]]
            label = str(values["step_label"][start])
            counts["total_windows"] += 1
            counts["anchor_" + label] += 1
            counts["active_reopening_anchors"] += bool(values["opening"][start])
            counts["windows_with_active_reopening_targets"] += bool(values["opening"][start:stop].any())
            counts["fault_targets"] += int(np.sum(values["step_label"][start:stop] == "fault"))
            if label == "recovery":
                recovery_rows.append({"update": event["step"], "parent": record["parent_episode"],
                    "step": start, "phase": str(values["phase"][start]),
                    "first_opening_delta_mm": float((values["action_applied"][start, 7] - values["observation_state"][start, 15]) * 1000)})
    if counts["fault_targets"]:
        raise ValueError("Injected fault target entered an optimizer sample")
    result = {"counts": dict(counts), "available_training_corrections": availability,
              "sampled_recovery_anchors": recovery_rows,
              "opening_definition": "label=recovery AND applied finger target exceeds observed travel by >0.2 mm at that interval; descriptive, not a mechanical limit",
              "scope": "Actual logged optimizer windows from " + str(args.training),
              "limits": "Exposure audit, not held-out behavior improvement or autonomous success"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "counts": result["counts"],
                      "available_training_corrections": availability}), flush=True)


if __name__ == "__main__":
    main()
