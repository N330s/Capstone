"""Heartbeat freshness is distinct from a job's hard wall-clock deadline."""
from pathlib import Path


def heartbeat_age_seconds(paths, *, stage_elapsed_s, now_unix):
    """None disables heartbeat checks; an empty configured source is not alive."""
    if paths is None:
        return None
    mtimes = []
    for path in paths:
        try:
            mtimes.append(Path(path).stat().st_mtime)
        except FileNotFoundError:
            pass  # First output may not exist during model initialization.
    # A resumed stage may inherit hours-old files. Its startup grace begins at
    # this launch, not at the previous session's last write.
    elapsed = max(0., stage_elapsed_s)
    return min(elapsed, max(0., now_unix - max(mtimes))) if mtimes else elapsed


def evaluation_heartbeat_paths(folder):
    folder = Path(folder)
    return [folder / "progress.json", *folder.glob("*_policy.log"), *folder.glob("*_replay.log")]


def heartbeat_stalled(age, timeout_s=900):
    return age is not None and age > timeout_s
