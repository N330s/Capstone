"""Opt-in bounded retention of this run's completed recovery checkpoints only."""
import json
from pathlib import Path
import shutil
import stat


def _no_reparse_points(path):
    pending = [path]
    while pending:
        child = pending.pop()
        attributes = child.lstat()
        if child.is_symlink() or getattr(attributes, "st_file_attributes", 0) & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0):
            raise ValueError("Refusing checkpoint cleanup through a symlink/junction")
        if stat.S_ISDIR(attributes.st_mode):
            pending.extend(child.iterdir())


def prune_checkpoints(parent, binding, protected=(), keep_recent=2):
    if keep_recent < 1:
        raise ValueError("At least one recent recovery checkpoint must be retained")
    parent = Path(parent).resolve()
    protected = {Path(p).resolve() for p in protected}
    candidates = []
    for path in parent.iterdir():
        marker = path / "complete.json"
        if not path.is_dir() or not marker.is_file():
            continue  # Preserve incomplete writes and anything unrelated.
        if path.resolve().parent != parent:
            continue
        metadata = json.loads(marker.read_text())
        if metadata.get("binding") != binding or not path.name.startswith("step_"):
            continue
        if metadata["label"] in ("recovery_initial", "final"):
            protected.add(path.resolve())
        candidates.append((metadata["step"], path))
    candidates.sort(key=lambda item: (item[0], item[1].name))
    protected.update(p.resolve() for _, p in candidates[-keep_recent:])
    removed = []
    for _, path in candidates:
        if path.resolve() in protected:
            continue
        # Validate exact targets immediately before recursive deletion; never
        # clean a workspace/model/data root or a checkpoint from another run.
        if path.resolve().parent != parent or path == parent:
            raise ValueError("Checkpoint cleanup target escaped its exact parent")
        _no_reparse_points(path)
        metadata = json.loads((path / "complete.json").read_text())
        if metadata["binding"] != binding:
            raise ValueError("Checkpoint binding changed during cleanup")
        shutil.rmtree(path)
        removed.append({"path": str(path), "step": metadata["step"], "recoverable": False})
    return removed
