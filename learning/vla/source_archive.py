"""Immutable local code snapshots for hash-bound historical experiment evidence."""
from pathlib import Path
import os
import shutil

from learning.vla.smolvla_data import sha256

ROOT = Path(__file__).resolve().parents[2]


def archive_root():
    """Containers keep code read-only and archives on the writable runs mount."""
    return Path(os.environ.get('OPENARM_SOURCE_ARCHIVE', ROOT / 'outputs/source_archives'))


def archive_source(relative):
    source = (ROOT / relative).resolve()
    if not source.is_relative_to(ROOT) or not source.is_file():
        raise ValueError("Workspace source file required")
    expected = sha256(source)
    destination = archive_root() / expected / source.name
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        shutil.copy2(source, destination)
    if sha256(destination) != expected:
        raise ValueError("Archived source hash differs")
    return {"source": str(relative), "sha256": expected, "archive": str(destination)}


def verified_source(relative, expected):
    source = ROOT / relative
    if source.is_file() and sha256(source) == expected:
        return source
    archived = archive_root() / expected / Path(relative).name
    if not archived.is_file() or sha256(archived) != expected:
        raise ValueError("Source version unavailable: " + str(relative))
    return archived
