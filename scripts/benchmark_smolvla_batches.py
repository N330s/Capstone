"""Offline end-to-end batch qualification; new processes and no artifact deletion."""
import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.finetune_smolvla import write_json
from scripts.run_smolvla_job import arm_deadline


def gpu_snapshot():
    flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    result = subprocess.run(["nvidia-smi", "--query-gpu=memory.total,memory.used",
                             "--format=csv,noheader,nounits"], capture_output=True,
                            text=True, check=True, timeout=10, creationflags=flags)
    rows = result.stdout.strip().splitlines()
    if len(rows) != 1:
        raise RuntimeError("This benchmark requires one visible GPU")
    total, used = map(float, rows[0].split(","))
    return {"total_mib": total, "used_mib": used}


def choose_batch(candidates):
    safe = [r for r in candidates if r.get("safe") and
            math.isfinite(r.get("samples_per_second", 0)) and r["samples_per_second"] > 0]
    if not safe:
        raise RuntimeError("No batch passed memory, finite-gradient and completion checks")
    return max(safe, key=lambda r: (r["samples_per_second"], -r["batch_size"]))["batch_size"]


def summarize(output, batch, snapshots, baseline, returncode, timed_out, warmup=10, updates=40):
    result = {"batch_size": batch, "returncode": returncode, "timed_out": timed_out,
              "safe": False, "output": str(output)}
    report_path = output / "report.json"
    if returncode or timed_out or not report_path.is_file():
        result["reason"] = "Trainer failed or exceeded the per-candidate deadline; inspect its log"
        return result
    report = json.loads(report_path.read_text())
    events = [json.loads(line) for line in (output / "events.jsonl").read_text().splitlines()]
    measured = [e for e in events if e["type"] == "update" and warmup <= e["step"] <= updates]
    finite = all(math.isfinite(e[k]) for e in measured for k in ("loss", "grad_norm"))
    if not report["completed"] or report["updates"] != updates or len(measured) != updates - warmup + 1 or not finite:
        result["reason"] = "Incomplete or nonfinite measured updates"
        return result
    duration = measured[-1]["elapsed_s"] - measured[0]["elapsed_s"]
    if duration <= 0 or not snapshots:
        result["reason"] = "Invalid timing or no GPU snapshots"
        return result
    # Conservative estimate also protects against transient reserved-memory peaks
    # between the one-second device snapshots. Baseline includes desktop usage.
    reserved_mib = report["peak_reserved_gb"] * 1e9 / 1024**2
    estimated_peak = max(max(s["used_mib"] for s in snapshots), baseline["used_mib"] + reserved_mib)
    headroom = baseline["total_mib"] - estimated_peak
    result.update(samples_per_second=(updates - warmup) * batch / duration,
                  mean_update_seconds=duration / (updates - warmup),
                  measured_updates=updates - warmup,
                  peak_allocated_gb=report["peak_allocated_gb"],
                  peak_reserved_gb=report["peak_reserved_gb"],
                  device_peak_used_mib=max(s["used_mib"] for s in snapshots),
                  conservative_headroom_mib=headroom, safe=headroom >= 1024,
                  reason="Passed" if headroom >= 1024 else "Less than 1 GiB conservative GPU headroom")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("export", "checkpoint", "backbone", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    candidates = []
    flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    for batch in (4, 8, 16):
        output = args.output / f"batch_{batch}"
        baseline, snapshots = gpu_snapshot(), []
        command = [sys.executable, "-u", str(ROOT / "scripts/finetune_smolvla.py"),
                   "--export", str(args.export.resolve()), "--checkpoint", str(args.checkpoint.resolve()),
                   "--backbone", str(args.backbone.resolve()), "--output", str(output),
                   "--batch-size", str(batch), "--steps", "60000", "--stop-after", "40",
                   "--max-seconds", "300", "--eval-every", "60000", "--save-every", "60000", "--probes", "4"]
        write_json(args.output / "status.json", {"stage": "benchmark", "batch_size": batch, "command": command})
        with (args.output / f"batch_{batch}.log").open("w", encoding="utf-8") as log:
            process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, creationflags=flags)
            fired = threading.Event()
            timer = arm_deadline(process, 300, fired)
            try:
                while process.poll() is None:
                    snapshots.append(gpu_snapshot())
                    time.sleep(1)
                code = process.wait()
            finally:
                timer.cancel()
                if process.poll() is None:
                    from scripts.run_smolvla_job import kill_owned_process
                    kill_owned_process(process)
            write_json(args.output / f"batch_{batch}_gpu.json", {"baseline": baseline, "snapshots": snapshots})
        candidates.append(summarize(output, batch, snapshots, baseline, code, fired.is_set()))
        write_json(args.output / "candidates.json", candidates)
    selected = choose_batch(candidates)
    write_json(args.output / "report.json", {"passed": True, "selected_batch_size": selected,
               "candidates": candidates, "elapsed_s": time.monotonic() - started,
               "export": str(args.export.resolve()), "api_tokens_used": False,
               "limits": "30 warmed end-to-end updates per candidate, not a maximum-batch search or task-success test"})


if __name__ == "__main__":
    main()
