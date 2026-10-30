"""Assemble a versioned, self-contained dataset release from finished collection runs.

    python scripts/make_release.py --runs data/run_a data/run_b --name openarm_pilot --version 1

Writes data/releases/<name>-v<version> (never overwrites) holding:
  episodes/<run>__<episode>/   hard links (copies if linking fails) of each verified episode
  release.json                 provenance: git commit + dirty flag, source hashes, workspace, controller
                               settings, camera calibration, pilot contract, per-episode records and split
  splits.json                  train / validation, grouped by scene seed
  heldout_resets.json          the reserved evaluation-bank resets (never collected from)
  checksums.sha256             sha256 of every other file in the release

Refused: an existing output, a run whose manifest.json is not complete, a run with a missing /
corrupt / non-replay-verified episode, and runs recorded with different source hashes, workspace,
scene or robot/contact config (mixed versions). Source runs are only read.

Splitting keeps related trajectories together: every episode sharing a scene seed (clean and fault
variants, across runs) goes to the same split, chosen by a hash of the seed.
"""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data_pipeline.episodes import validate_episode, pilot_contract

SPLIT_SALT = "openarm-release-split-v1"
SAME_ACROSS_RUNS = ("source_hashes", "workspace", "scene_sha256", "robot_config", "contact_config",
                    "mujoco_version")


def seed_hash(seed):
    """Deterministic value in [0, 1) for a scene seed."""
    digest = hashlib.sha256(f"{SPLIT_SALT}:{int(seed)}".encode()).hexdigest()
    return int(digest[:12], 16) / float(16 ** 12)


def assign_splits(seeds, fraction):
    """seed -> "train" | "validation". >= 1 group in each split whenever there are >= 2 groups."""
    seeds = sorted(set(int(s) for s in seeds))
    split = {s: ("validation" if seed_hash(s) < fraction else "train") for s in seeds}
    if len(seeds) >= 2 and "validation" not in split.values():
        split[min(seeds, key=seed_hash)] = "validation"
    if len(seeds) >= 2 and "train" not in split.values():
        split[max(seeds, key=seed_hash)] = "train"
    return split


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_state(root):
    def run(*cmd):
        return subprocess.run(["git", *cmd], cwd=root, capture_output=True, text=True, check=True).stdout
    try:
        return {"commit": run("rev-parse", "HEAD").strip(), "dirty": bool(run("status", "--porcelain").strip())}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}


def link_tree(src, dst):
    for path in sorted(src.rglob("*")):
        target = dst / path.relative_to(src)
        if path.is_dir():
            target.mkdir(parents=True, exist_ok=True)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.link(path, target)
        except OSError:
            shutil.copy2(path, target)


def collect_episodes(runs, replay_env_factory, allow_missing_collection_hashes=False):
    """Validate every episode of every run, check the versions agree, then require replays.

    -> list of dicts (run, id, path, meta, replay). Version checks run before the replay check so a
    mixed-version release is reported as such."""
    found, names = [], set()
    for run in runs:
        run = Path(run)
        if run.name in names:
            raise SystemExit(f"duplicate run name {run.name}")
        names.add(run.name)
        manifest_path = run / "manifest.json"
        if not manifest_path.exists():
            raise SystemExit(f"{run}: no manifest.json")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not manifest.get("complete"):
            raise SystemExit(f"{run}: manifest.json is not complete (the run did not reach its episode target)")
        run_info = {k: manifest.get(k) for k in
                    ("collection_source_hashes", "detector_mode", "start_row", "expert_sha256", "scene_bank_sha256",
                     "scene_bank_module_sha256", "collection_script_sha256", "fault_injection",
                     "task_stages", "seed_namespaces")}
        for entry in manifest["episodes"]:
            path = run / entry["episode"]
            try:
                meta = validate_episode(path)
            except Exception as error:     # missing file, checksum mismatch, bad schema
                raise SystemExit(f"{path}: {type(error).__name__}: {error}")
            found.append({"run": run.name, "run_path": run, "id": entry["episode"], "path": path,
                          "meta": meta, "replay": meta.get("replay"), "run_manifest": run_info})
    if found:
        check_same_version(found, allow_missing_collection_hashes)
    for e in found:
        if e["replay"] and e["replay"].get("passed"):
            continue
        env = replay_env_factory(e["meta"])
        if env is None:
            raise SystemExit(f"{e['path']}: no passing replay recorded; rerun with --replay or "
                             "scripts/replay_pilot.py")
        from data_pipeline.replay import replay_episode
        try:
            e["replay"] = {"passed": True, **replay_episode(env, e["path"])}
        except (AssertionError, ValueError, RuntimeError) as error:
            raise SystemExit(f"{e['path']}: replay failed: {error}")
    return found


def check_same_version(episodes, allow_missing_collection_hashes=False):
    """Episodes must share the env-manifest hashes AND the collection-code hashes of their runs.

    A run without collection_source_hashes (collected before they were recorded) is refused, unless
    --allow-missing-collection-hashes is given; then it may only mix with other runs lacking them."""
    by_run = {}
    for e in episodes:
        by_run.setdefault(e["run"], e["run_manifest"].get("collection_source_hashes"))
    missing = sorted(r for r, h in by_run.items() if h is None)
    if missing and not allow_missing_collection_hashes:
        raise SystemExit(f"runs {missing} have no collection_source_hashes in manifest.json (collected "
                         "before the expert/sampler code was hashed); the expert version cannot be "
                         "verified. Pass --allow-missing-collection-hashes to accept them anyway.")
    if missing and len(missing) != len(by_run):
        raise SystemExit(f"mixed versions: runs {missing} lack collection_source_hashes but others "
                         "have them; they cannot be compared")
    reference = next(iter(by_run.items()))
    for run, hashes in by_run.items():
        if hashes != reference[1]:
            diff = sorted(k for k in set(hashes or {}) | set(reference[1] or {})
                          if (hashes or {}).get(k) != (reference[1] or {}).get(k))
            raise SystemExit(f"mixed versions: run {run} differs from {reference[0]} in "
                             f"collection_source_hashes {diff}")
    first = episodes[0]
    for e in episodes[1:]:
        for key in SAME_ACROSS_RUNS:
            if e["meta"].get(key) != first["meta"].get(key):
                raise SystemExit(f"mixed versions: {e['run']}/{e['id']} differs from "
                                 f"{first['run']}/{first['id']} in {key}; one release = one code/scene version")


def episode_record(e, split):
    m = e["meta"]
    return {"run": e["run"], "id": e["id"], "dir": f"episodes/{e['run']}__{e['id']}",
            "seed": m["seed"], "reset_options": m["reset_options"], "data_sha256": m["data_sha256"],
            "steps": m["steps"], "fault_kind": (m.get("fault") or {}).get("kind"),
            "label_counts": m.get("label_counts") or {"expert": m["steps"]},
            "quality": m.get("quality"), "stage_outcomes": m.get("stage_outcomes"),
            "replay": e["replay"], "split": split}


def make_release(args, replay_env_factory=lambda meta: None, heldout_factory=None):
    root = Path(__file__).resolve().parents[1]
    output = args.output or Path("data/releases") / f"{args.name}-v{args.version}"
    if output.exists():
        raise SystemExit(f"{output} already exists; releases are never overwritten (bump --version)")
    episodes = collect_episodes(args.runs, replay_env_factory, args.allow_missing_collection_hashes)
    if not episodes:
        raise SystemExit("no episodes in the given runs")
    seeds = [e["meta"]["seed"] for e in episodes]
    split_of = assign_splits(seeds, args.validation_fraction)
    heldout = []
    if args.heldout > 0:
        print(f"building {args.heldout} held-out evaluation resets (scene sampling is slow)...", flush=True)
        started = time.time()
        heldout = heldout_factory(args.heldout)
        print(f"held-out resets ready in {time.time() - started:.0f}s", flush=True)
        clash = set(seeds) & {r["seed"] for r in heldout}
        if clash:
            raise SystemExit(f"release seeds overlap the held-out evaluation resets: {sorted(clash)[:5]}")
    output.mkdir(parents=True, exist_ok=False)
    first = episodes[0]["meta"]
    records = []
    for i, e in enumerate(episodes, 1):
        link_tree(e["path"], output / "episodes" / f"{e['run']}__{e['id']}")
        records.append(episode_record(e, split_of[e["meta"]["seed"]]))
        print(f"linked {i}/{len(episodes)} {e['run']}/{e['id']} -> {records[-1]['split']}", flush=True)
    runs_info = {}
    for e in episodes:
        runs_info.setdefault(e["run"], {"path": str(e["run_path"]), **e["run_manifest"]})
    release = {"name": args.name, "version": args.version,
               "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "git": git_state(root),
               "dataset_schema": first.get("dataset_schema"),
               "source_hashes": first["source_hashes"], "scene_sha256": first["scene_sha256"],
               "workspace": first["workspace"], "robot_config": first["robot_config"],
               "contact_config": first["contact_config"], "controller": first.get("controller"),
               "mujoco_version": first["mujoco_version"],
               "camera_calibration": first.get("camera_calibration"),
               "depth_cameras": first.get("depth_cameras"),
               "pilot_contract": first.get("pilot_contract") or pilot_contract(first),
               "collection_source_hashes": next(iter(runs_info.values())).get("collection_source_hashes"),
               "runs": runs_info, "validation_fraction": args.validation_fraction,
               "heldout_resets": len(heldout), "episodes": records,
               "counts": {s: sum(1 for r in records if r["split"] == s) for s in ("train", "validation")}}
    (output / "release.json").write_text(json.dumps(release, indent=2), encoding="utf-8")
    groups = {}
    for r in records:
        groups.setdefault(r["seed"], []).append(f"{r['run']}__{r['id']}")
    (output / "splits.json").write_text(json.dumps({
        "grouping": "scene seed: every episode sharing a seed lands in one split",
        "rule": f"validation iff sha256('{SPLIT_SALT}:<seed>')[:12]/16^12 < {args.validation_fraction}; "
                "lowest-hash group forced to validation when none qualifies",
        "validation_fraction": args.validation_fraction,
        "groups": {str(s): {"split": split_of[s], "episodes": eps, "hash": seed_hash(s)}
                   for s, eps in sorted(groups.items())},
        "train": [f"{r['run']}__{r['id']}" for r in records if r["split"] == "train"],
        "validation": [f"{r['run']}__{r['id']}" for r in records if r["split"] == "validation"]},
        indent=2), encoding="utf-8")
    (output / "heldout_resets.json").write_text(json.dumps({
        "status": "reserved evaluation resets; never used for collection or tuning",
        "count": len(heldout), "resets": heldout}, indent=2), encoding="utf-8")
    lines = [f"{file_sha256(p)}  {p.relative_to(output).as_posix()}"
             for p in sorted(output.rglob("*")) if p.is_file()]
    (output / "checksums.sha256").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"release written: {output} ({len(records)} episodes, {release['counts']})", flush=True)
    return output


def default_heldout_factory(workspace_path):
    def build(n):
        from envs.openarm_insert import OpenArmInsertEnv
        from data_pipeline.scene_bank import evaluation_bank, workspace_for_env
        env = OpenArmInsertEnv(images=False, workspace=workspace_path)
        try:
            ws = workspace_for_env(env)
        finally:
            env.close()
        return [{"id": r["id"], "seed": r["seed"], "split": r["split"], "options": r["options"]}
                for r in evaluation_bank(n, ws=ws)]
    return build


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", nargs="+", type=Path, required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--version", type=int, required=True)
    p.add_argument("--output", type=Path, default=None, help="default data/releases/<name>-v<version>")
    p.add_argument("--validation-fraction", type=float, default=0.15)
    p.add_argument("--heldout", type=int, default=100, help="held-out evaluation resets to list (0 = none)")
    p.add_argument("--allow-missing-collection-hashes", action="store_true",
                   help="accept runs whose manifest lacks collection_source_hashes (all runs must lack it)")
    p.add_argument("--replay", action="store_true",
                   help="rerun the strict replay for episodes without a recorded passing replay")
    args = p.parse_args(argv)
    if not 0 <= args.validation_fraction < 1:
        p.error("--validation-fraction must be in [0, 1)")
    env_cache = {}

    def replay_env(meta):
        if not args.replay:
            return None
        from envs.openarm_insert import OpenArmInsertEnv
        path = (meta.get("workspace") or {}).get("path")
        if path not in env_cache:
            env_cache[path] = OpenArmInsertEnv(images=False, workspace=path)
        return env_cache[path]

    def heldout(n):
        # Workspace of the runs (read back from the first run's manifest) so the bank matches collection.
        first = json.loads((args.runs[0] / "manifest.json").read_text(encoding="utf-8"))
        ws = (first.get("robot_manifest", {}).get("workspace") or {}).get("path")
        return default_heldout_factory(ws)(n)

    try:
        make_release(args, replay_env, heldout)
    finally:
        for env in env_cache.values():
            env.close()


if __name__ == "__main__":
    main()
