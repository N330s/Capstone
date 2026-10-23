"""Audit externally interrupted evaluation without resetting its time allowance."""
import json
import math
from pathlib import Path

from learning.vla.smolvla_data import sha256
from learning.vla.source_archive import verified_source


def charged_external_seconds(campaign_plan, completion, status):
    if (completion.get("completed") or completion.get("state") != "failed"
            or completion.get("error") != "RuntimeError('Training/evaluation heartbeat stalled for 15 minutes')"
            or status.get("stage") != "pickup_evaluation" or status.get("state") != "running"):
        raise ValueError("Expected the documented false-stall evaluation interruption")
    elapsed = campaign_plan["started_unix"] + completion["elapsed_s"] - status["started_unix"]
    if not math.isfinite(elapsed) or not 900 <= elapsed < 2700:
        raise ValueError("Invalid external interruption time ledger")
    return math.ceil(elapsed) + 2  # Conservative allowance for timestamp origin skew.


def validate_external_interruption(folder, marker):
    folder = Path(folder).resolve()
    if marker["original_plan_sha256"] != sha256(folder / "plan.json"):
        raise ValueError("Interrupted evaluation plan changed")
    for relative, key in (("scripts/resume_fp32_evaluation.py", "recovery_script_sha256"),
                          ("learning/vla/evaluation_recovery.py", "recovery_helper_sha256")):
        verified_source(relative, marker[key])
    sources = marker["campaign_sources"]
    values = {}
    for name in ("plan", "completion", "status"):
        source = sources[name]
        path = Path(source["path"])
        expected_name = "completion.json" if name == "completion" else name + ".json"
        if path.resolve() != folder.parent / expected_name or sha256(path) != source["sha256"]:
            raise ValueError("Original external supervisor evidence changed")
        values[name] = json.loads(path.read_text())
    minimum = charged_external_seconds(values["plan"], values["completion"], values["status"])
    if (marker["external_elapsed_s"] != minimum or not math.isfinite(marker["preparation_s"])
            or marker["preparation_s"] < 0
            or marker["elapsed_s"] != minimum + marker["preparation_s"]
            or marker["elapsed_s"] >= 2700 or not marker["completed_trials_preserved"]
            or not marker["incomplete_attempt_restart_disclosed"]):
        raise ValueError("Interrupted evaluation budget/provenance changed")
    plan = json.loads((folder / "plan.json").read_text())
    expected = [(r["episode"], mode) for r in plan["parents"] for mode in ("replay", "policy")]
    prior = marker["completed_rows"]
    if not 0 < len(prior) < len(expected) or [(r["episode"], r["mode"]) for r in prior] != expected[:len(prior)]:
        raise ValueError("Original completed conditions are not the frozen ordered prefix")
    if marker["interrupted_condition"] != "_".join(expected[len(prior)]):
        raise ValueError("Wrong interrupted condition")
    for relative, expected_hash in marker["preserved_files"].items():
        path = (folder / relative).resolve()
        if not path.is_relative_to(folder) or sha256(path) != expected_hash:
            raise ValueError("Original completed/interrupted artifact changed")
    return marker["elapsed_s"]
