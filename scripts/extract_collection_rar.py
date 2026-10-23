"""Validate RAR paths/size; extract lossless episodes, not redundant JPEG previews.

Run in the isolated download environment with rarfile installed. Uses Windows
bsdtar; no archive code is executed and the output directory must be new.
"""
import argparse
import json
from pathlib import Path, PurePosixPath
import shutil
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import rarfile
    archive, output = args.archive.resolve(), args.output.resolve()
    if output.exists():
        raise FileExistsError("New extraction destination required")
    if not archive.is_file() or archive.suffix.lower() != ".rar":
        raise ValueError("Completed RAR file required; no partial archive intake")
    with rarfile.RarFile(archive) as reader:
        if reader.needs_password():
            raise ValueError("Encrypted archive requires explicit handling")
        members, selected = reader.infolist(), []
        roots, names = set(), set()
        for item in members:
            name = item.filename.replace("\\", "/")
            path = PurePosixPath(name)
            if (path.is_absolute() or any(p in ("..", ".") or ":" in p for p in path.parts)
                    or item.is_symlink() or getattr(item, "file_redir", None)):
                raise ValueError(f"Unsafe archive entry: {name}")
            if not path.parts:
                raise ValueError("Empty archive path")
            key = name.casefold()
            if key in names:
                raise ValueError(f"Duplicate archive entry: {name}")
            names.add(key)
            roots.add(path.parts[0])
            if item.is_file() and ((len(path.parts) == 2 and path.suffix == ".json")
                    or (len(path.parts) == 3 and path.name in ("episode.npz", "metadata.json", "diagnostics.json"))):
                selected.append(item)
        if len(roots) != 1 or not selected:
            raise ValueError("Expected one collection root")
        expanded = sum(i.file_size for i in selected)
        output.parent.mkdir(parents=True, exist_ok=True)
        if shutil.disk_usage(output.parent).free - expanded < 25 * 1024**3:
            raise RuntimeError("Extraction would leave less than 25 GiB free")
        receipt = {"archive": str(archive), "archive_bytes": archive.stat().st_size,
                   "all_members": len(members), "all_expanded_bytes": sum(i.file_size for i in members),
                   "selected_members": len(selected), "selected_expanded_bytes": expanded,
                   "excluded": "JPEG preview frames and failed-preflight arrays; retained in original archive",
                   "root": next(iter(roots))}
    print(json.dumps(receipt), flush=True)
    output.mkdir()
    subprocess.run(["tar", "-xf", str(archive), "-C", str(output), *[i.filename for i in selected]], check=True)
    (output / "extraction_receipt.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
