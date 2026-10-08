"""Initialize only writable cache subdirectories, then preserve command signals."""
import os
from pathlib import Path
import sys


def main():
    os.umask(0o027)
    for key in ("HOME", "HF_HOME", "TORCH_HOME", "XDG_CACHE_HOME"):
        path = Path(os.environ[key])
        path.mkdir(parents=True, exist_ok=True)
        if not os.access(path, os.W_OK):
            raise PermissionError(f"Cache is not writable: {path}; check host UID/GID and storage ownership")
    if len(sys.argv) < 2:
        raise ValueError("Supply a command to execute")
    os.execvp(sys.argv[1], sys.argv[1:])


if __name__ == "__main__":
    main()
