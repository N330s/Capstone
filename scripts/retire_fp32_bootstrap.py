"""Retire two disposable NEW-run bootstrap checkpoints after verified successors.

One-off storage amendment for fp32_multiepisode_20261020; never traverses old
experiments. Preserve hashes/all small metadata and never remove a current
best/latest/final pointer. Native PowerShell performs the exact deletions.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from learning.vla.runtime_io import read_json, write_json
from learning.vla.smolvla_data import sha256
from learning.vla.checkpoint_retention import _no_reparse_points

CAMPAIGN = ROOT / "outputs/fp32_multiepisode_20261020"


def validate_target(target, successor, campaign=CAMPAIGN):
    allowed = (campaign / "qualification/reference/checkpoints/step_0000012_final",
               campaign / "training/checkpoints/step_0000025_recovery_initial")
    if target not in allowed or target.resolve() != target or successor.resolve() == target:
        raise ValueError("Not an exact disposable new bootstrap target")
    if successor.resolve().parent != (campaign / "training/checkpoints").resolve():
        raise ValueError("Successor must belong to this campaign's training checkpoints")
    _no_reparse_points(target)
    _no_reparse_points(successor)
    old, new = read_json(target / "complete.json"), read_json(successor / "complete.json")
    for checkpoint in (target, successor):
        if not all((checkpoint / name).is_file() for name in ("pretrained_model/model.safetensors", "training_state.pt")):
            raise ValueError("A complete full checkpoint successor is required")
    if old["binding"]["trainable_storage"] != "float32" or new["binding"]["trainable_storage"] != "float32":
        raise ValueError("Expected only corrected checkpoints")
    if old["binding"]["export_receipt_sha256"] != new["binding"]["export_receipt_sha256"]:
        raise ValueError("Source export changed")
    if target == allowed[1] and (new["binding"] != old["binding"] or new["step"] < 500):
        raise ValueError("Initial recovery must have a same-run successor at >=500 updates")
    for name in ("latest_checkpoint.json", "best_checkpoint.json"):
        pointer = campaign / "training" / name
        if pointer.exists() and Path(read_json(pointer)["path"]).resolve() == target:
            raise ValueError("Never retire a current best/latest pointer")
    return old, new


def retire(target, successor):
    old, new = validate_target(target, successor)
    label = "qualification_reference" if old["step"] == 12 else "initial_recovery"
    archive = CAMPAIGN / "bootstrap_retention" / label
    archive.mkdir(parents=True, exist_ok=False)
    hashes = {}
    for file in target.rglob("*"):
        if file.is_file():
            relative = file.relative_to(target)
            hashes[str(relative)] = {"sha256": sha256(file), "bytes": file.stat().st_size}
            if file.stat().st_size < 1024**2:
                (archive / relative).parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(file, archive / relative)
    receipt = {"target": str(target), "successor": str(successor), "successor_step": new["step"],
        "old_complete": old, "file_hashes": hashes, "recoverable": False,
        "exact_gpu_resume_gate": read_json(CAMPAIGN / "qualification/exact_resume.json"),
        "small_metadata_preserved": True, "removed": False}
    write_json(archive / "receipt.json", receipt)
    validate_target(target, successor)  # Check pointers and exact paths again.
    literal = str(target).replace("'", "''")
    parent = str(target.parent).replace("'", "''")
    script = ("$ErrorActionPreference='Stop'; $target=(Resolve-Path -LiteralPath '" + literal + "').Path; "
        "if ((Split-Path -Parent $target) -ne '" + parent + "') { throw 'Unexpected parent' }; "
        "$items=@(Get-Item -LiteralPath $target)+@(Get-ChildItem -LiteralPath $target -Force -Recurse); "
        "if (@($items | Where-Object { $_.Attributes -band [IO.FileAttributes]::ReparsePoint }).Count) { throw 'Reparse point' }; "
        "Remove-Item -LiteralPath $target -Recurse -Force")
    subprocess.run(["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", script],
                   check=True, timeout=60, creationflags=subprocess.CREATE_NO_WINDOW)
    receipt["removed"] = True
    write_json(archive / "receipt.json", receipt)
    print(json.dumps({"retired": str(target), "successor": str(successor), "bytes": sum(r["bytes"] for r in hashes.values())}), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--execute", action="store_true")
    args = p.parse_args()
    if os.name != "nt":
        raise RuntimeError("Windows-only native cleanup")
    plan = read_json(CAMPAIGN / "plan.json")
    gate = read_json(CAMPAIGN / "qualification/exact_resume.json")
    if not gate["passed"] or plan["campaign"] != "corrected-fp32-multiepisode-v1":
        raise ValueError("Completed corrected GPU qualification required")
    initial = CAMPAIGN / "training/checkpoints/step_0000025_recovery_initial"
    reference = CAMPAIGN / "qualification/reference/checkpoints/step_0000012_final"
    validate_target(reference, initial)
    if not args.execute:
        print(json.dumps({"dry_run": True, "targets": [str(reference), str(initial)],
            "initial_waits_for_newer_complete_step": 500, "best_latest_final_preserved": True}))
        return
    write_json(CAMPAIGN / "retention_amendment.json", {"original_plan_sha256": sha256(CAMPAIGN / "plan.json"),
        "reason": "Observed free-space reduction beyond run checkpoint sizes; retain 25 GiB reserve without shortening training",
        "targets": [str(reference), str(initial)], "new_retention": "best, latest/recent, final; bootstrap hashes and small metadata",
        "no_additional_old_experiment_cleanup": True, "implementation_sha256": sha256(Path(__file__)),
        "started_unix": time.time(), "completed": False})
    retire(reference, initial)
    import psutil
    owner = psutil.Process(plan["supervisor_pid"])
    owner_created = owner.create_time()
    deadline = plan["started_unix"] + plan["qualification_plus_training_limit_s"]
    while time.time() < deadline and owner.is_running() and owner.create_time() == owner_created:
        if (CAMPAIGN / "completion.json").exists():
            return
        pointer = CAMPAIGN / "training/latest_checkpoint.json"
        if pointer.exists():
            successor = Path(read_json(pointer)["path"])
            if read_json(successor / "complete.json")["step"] >= 500:
                retire(initial, successor)
                amendment = read_json(CAMPAIGN / "retention_amendment.json")
                write_json(CAMPAIGN / "retention_amendment.json", {**amendment, "completed": True, "ended_unix": time.time()})
                return
        time.sleep(5)


if __name__ == "__main__":
    main()
