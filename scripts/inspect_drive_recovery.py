"""Inspect public Drive RARs and retrieve a bounded sample, not entire releases.

Run with .venv-download. HTTP ranges avoid downloading redundant previews and
multi-gigabyte archives. Original recordings and training exports are untouched.
Only non-solid, unencrypted archives can be sampled; CRC is checked by rarfile.
"""
from __future__ import annotations

import argparse
from collections import Counter
import io
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import time
import zlib

import rarfile
import requests


class DriveRangeReader(io.RawIOBase):
    def __init__(self, file_id, *, max_bytes=4 * 1024**3, max_requests=1500):
        self.session = requests.Session()
        self.url = "https://drive.usercontent.google.com/download"
        self.params = {"id": file_id, "export": "download", "confirm": "t"}
        with self.session.get(self.url, params=self.params, headers={"Range": "bytes=0-511"},
                              timeout=(15, 20)) as response:
            response.raise_for_status()
            span = re.fullmatch(r"bytes 0-511/(\d+)", response.headers.get("Content-Range", ""))
            if response.status_code != 206 or not span or len(response.content) != 512:
                raise ValueError("Expected an exact 512-byte range probe, not a Drive interstitial")
            self.size = int(span.group(1))
            self.modified = response.headers.get("Last-Modified")
        self.position = 0
        self.page_size = 256 * 1024
        self.cache_start = 0
        self.cache = b""
        self.transferred = 0
        self.requests = 0
        self.max_bytes, self.max_requests = max_bytes, max_requests

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        position = offset + (0 if whence == 0 else self.position if whence == 1 else self.size)
        if whence not in (0, 1, 2) or position < 0:
            raise ValueError("Invalid seek")
        self.position = position
        return position

    def read(self, size=-1):
        if size is None or size < 0:
            raise ValueError("Unbounded remote reads are not allowed")
        size = min(size, max(0, self.size - self.position))
        parts = []
        while size:
            if not self.cache_start <= self.position < self.cache_start + len(self.cache):
                start = self.position
                end = min(self.size - 1, start + self.page_size - 1)
                if self.transferred + end - start + 1 > self.max_bytes or self.requests >= self.max_requests:
                    raise RuntimeError("Remote inspection transfer/request budget exhausted")
                for attempt in range(3):
                    self.requests += 1
                    try:
                        response = self.session.get(self.url, params=self.params,
                                      headers={"Range": f"bytes={start}-{end}", "Accept-Encoding": "identity"},
                                      stream=True, timeout=(15, 45))
                        response.raise_for_status()
                        break
                    except requests.RequestException:
                        if 'response' in locals():
                            response.close()
                        if attempt == 2:
                            raise
                        time.sleep(attempt + 1)
                with response:
                    response.raise_for_status()
                    if (response.status_code != 206
                            or response.headers.get("Content-Range") != f"bytes {start}-{end}/{self.size}"
                            or response.headers.get("Last-Modified") != self.modified):
                        raise ValueError("Range response mismatch or remote archive changed")
                    chunks, received = [], 0
                    for chunk in response.iter_content(256 * 1024):
                        received += len(chunk)
                        self.transferred += len(chunk)
                        if received > end - start + 1:
                            raise ValueError("Oversized range response")
                        chunks.append(chunk)
                    self.cache = b"".join(chunks)
                    self.cache_start = start
                    if len(self.cache) != end - start + 1:
                        raise ValueError("Truncated range response")
                if self.requests % 10 == 0:
                    print(json.dumps({"range_requests": self.requests,
                                      "transferred_mib": round(self.transferred / 1024**2, 1),
                                      "archive_position_gib": round(start / 1024**3, 2)}), flush=True)
            offset = self.position - self.cache_start
            chunk = self.cache[offset:offset + size]
            parts.append(chunk)
            self.position += len(chunk)
            size -= len(chunk)
        return b"".join(parts)

    def readinto(self, buffer):
        data = self.read(len(buffer))
        buffer[:len(data)] = data
        return len(data)

    def close(self):
        self.session.close()
        super().close()


class SampleHeadersComplete(Exception):
    """Stop a prefix scan after enough complete episode headers are available."""


class BSDTarSetup(rarfile.ToolSetup):
    def open_cmdline(self, pwd, rarfn, filefn=None):
        if pwd is not None:
            raise ValueError("Passwords are not supported")
        # rarfile 4.5 otherwise puts '--' between '-f' and its required value.
        # Keep the option terminator AFTER the absolute archive argument.
        command = [rarfile.BSDTAR_TOOL, "-x", "--to-stdout", "-f", rarfn, "--"]
        if filefn:
            command.append(filefn)
        return command


def quick_open_members(remote, archive):
    """Use RAR5's optional QO locator; verify selected cached headers in-place.

    Layout: https://www.rarlab.com/technote.htm (locator and quick-open records).
    QO is only an index: original headers must agree before any data is read.
    This uses the pinned rarfile 4.5 parser, not a general archive extractor.
    """
    remote.seek(8)
    initial = remote.read(32)
    length, pos = rarfile.load_vint(initial, 4)
    remote.seek(8)
    main = remote.read(pos + length)
    if zlib.crc32(main[4:]) != int.from_bytes(main[:4], "little"):
        raise ValueError("Main-header CRC mismatch")
    kind, pos = rarfile.load_vint(main, pos)
    flags, pos = rarfile.load_vint(main, pos)
    if kind != 1 or not flags & 1:
        raise ValueError("No RAR5 quick-open locator")
    extra, pos = rarfile.load_vint(main, pos)
    pos = len(main) - extra
    qo_offset = None
    while pos < len(main):
        size, pos = rarfile.load_vint(main, pos)
        end = pos + size
        record_kind, pos = rarfile.load_vint(main, pos)
        if record_kind == 1:
            locator_flags, pos = rarfile.load_vint(main, pos)
            if locator_flags & 1:
                offset, pos = rarfile.load_vint(main, pos)
                qo_offset = 8 + offset if offset else None
        pos = end
    if qo_offset is None:
        raise ValueError("Quick-open offset unavailable")
    parser = archive._file_parser
    remote.seek(qo_offset)
    qo = parser._parse_header(remote)
    if (qo.filename != "QO" or qo.compress_type != rarfile.RAR_M0
            or qo.compress_size > 64 * 1024**2):
        raise ValueError("Unsupported quick-open service block")
    data = remote.read(qo.compress_size)
    records, pos, count = [], 0, 0
    while pos < len(data):
        start = pos
        checksum = int.from_bytes(data[pos:pos + 4], "little")
        size, pos = rarfile.load_vint(data, pos + 4)
        end = pos + size
        if end > len(data) or zlib.crc32(data[start + 4:end]) != checksum:
            raise ValueError("Quick-open record CRC mismatch")
        flags, pos = rarfile.load_vint(data, pos)
        offset, pos = rarfile.load_vint(data, pos)
        length, pos = rarfile.load_vint(data, pos)
        cached = data[pos:pos + length]
        if flags or len(cached) != length or pos + length != end:
            raise ValueError("Unsupported quick-open cache record")
        count += 1
        if any(token in cached for token in (b"/episode.npz", b"/metadata.json", b"/diagnostics.json")):
            target = qo_offset - offset
            remote.seek(target)
            original = remote.read(length)
            if original != cached:
                raise ValueError("Quick-open and original archive headers differ")
            remote.seek(target)
            item = parser._parse_header(remote)
            safe_member(item)
            item.volume, item.volume_file = 0, remote
            records.append(item)
        pos = end
    print(json.dumps({"quick_open_cached_records": count, "episode_file_headers": len(records)}), flush=True)
    existing = {item.filename for item in archive.infolist()}
    for item in records:
        if item.filename not in existing:
            parser.process_entry(remote, item)
            existing.add(item.filename)
    return count


def safe_member(item):
    name = item.filename.replace("\\", "/")
    parts = PurePosixPath(name).parts
    if (not parts or name.startswith("/") or "\\" in item.filename
            or any(p in (".", "..") or re.search(r'[<>:"|?*\x00-\x1f]', p)
                   or p.endswith((" ", "."))
                   or re.fullmatch(r"(?i)(CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])(?:\..*)?", p)
                   for p in parts)
            or item.is_symlink() or getattr(item, "file_redir", None)):
        raise ValueError(f"Unsafe archive member: {name!r}")
    return PurePosixPath(name)


def main():
    if rarfile.__version__ != "4.5":
        raise RuntimeError("This bounded reader is qualified with rarfile 4.5 only")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--file-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-episodes", type=int, default=1)
    parser.add_argument("--episodes", nargs="+", help="Explicit parent names; requires --quick-open")
    parser.add_argument("--full-inventory", action="store_true",
                        help="Scan all preview headers too; substantially slower")
    parser.add_argument("--metadata-only", action="store_true")
    parser.add_argument("--quick-open", action="store_true",
                        help="Use and verify the RAR5 index, avoiding JPEG scans")
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.file_id) or not 0 <= args.sample_episodes <= 3:
        raise ValueError("Invalid file ID or sample count")
    if args.episodes and (not args.quick_open or len(args.episodes) > 3
                          or any(not re.fullmatch(r"[A-Za-z0-9_-]+", n) for n in args.episodes)):
        raise ValueError("At most three safe explicit episode names with --quick-open required")
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError("New inspection output required")
    output.mkdir(parents=True)
    rarfile.HACK_TMP_DIR = str(output)
    rarfile.BSDTAR_TOOL = shutil.which("tar") or "tar"
    unrar = shutil.which("unrar") or "C:/Program Files/WinRAR/UnRAR.exe"
    if Path(unrar).is_file():
        rarfile.UNRAR_TOOL = unrar
        rarfile.tool_setup(unrar=True, unar=False, bsdtar=False, sevenzip=False, sevenzip2=False, force=True)
    else:
        setup = BSDTarSetup(rarfile.BSDTAR_CONFIG)
        if not setup.check():
            raise RuntimeError("Working unrar or bsdtar required")
        rarfile.CURRENT_SETUP = setup
    with DriveRangeReader(args.file_id) as remote:
        remote.page_size = 64 * 1024 if args.metadata_only else 1024**2
        seen = {}

        def on_header(item):
            if not getattr(item, "filename", None):
                return
            name = safe_member(item)
            if (args.metadata_only or args.quick_open) and len(name.parts) > 2 and name.parts[1] != "failed_preflight":
                raise SampleHeadersComplete()
            if len(name.parts) == 3 and name.name in ("episode.npz", "metadata.json", "diagnostics.json"):
                fields = seen.setdefault(name.parts[1], set())
                fields.add(name.name)
                print(json.dumps({"header": str(name), "bytes": item.file_size}), flush=True)
                complete = sum(fields >= {"episode.npz", "metadata.json", "diagnostics.json"}
                               for fields in seen.values())
                if complete >= max(1, args.sample_episodes) and not args.full_inventory:
                    raise SampleHeadersComplete()

        # Keep the parser's verified prefix index when intentionally stopping its
        # callback. Entries are inserted before the callback by rarfile 4.5.
        archive = rarfile.RarFile.__new__(rarfile.RarFile)
        scan_complete = True
        try:
            rarfile.RarFile.__init__(archive, remote, errors="strict", info_callback=on_header)
        except SampleHeadersComplete:
            scan_complete = False
        quick_count = quick_open_members(remote, archive) if args.quick_open else None
        members = archive.infolist()
        names = [safe_member(item) for item in members]
        if len({str(name).casefold() for name in names}) != len(names):
            raise ValueError("Duplicate archive paths")
        roots = {name.parts[0] for name in names}
        if len(roots) != 1:
            raise ValueError("Expected one collection root")
        episode_items = [item for item, name in zip(members, names)
                         if item.is_file() and len(name.parts) == 3 and name.name == "episode.npz"]
        root = next(iter(roots))
        report = {
            "drive_file_id": args.file_id, "archive_bytes": remote.size,
            "last_modified": remote.modified, "root": root, "members": len(members),
            "full_archive_inventory": scan_complete,
            "quick_open_cached_records": quick_count,
            "solid": archive.is_solid(), "encrypted": archive.needs_password(),
            "expanded_bytes": sum(item.file_size for item in members),
            "suffix_counts": dict(Counter(name.suffix for item, name in zip(members, names) if item.is_file())),
            "episode_npz": [{"path": item.filename, "bytes": item.file_size,
                             "compressed_bytes": item.compress_size} for item in episode_items],
        }
        (output / "archive_inventory.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps({k: v for k, v in report.items() if k != "episode_npz"}
                         | {"episode_count": len(episode_items)}), flush=True)
        if archive.is_solid() or archive.needs_password():
            raise ValueError("Inventory saved; refusing selective extraction of solid/encrypted archive")
        selected_episodes = {PurePosixPath(item.filename).parts[1]
                             for item in episode_items[:args.sample_episodes]}
        if args.episodes:
            selected_episodes = set(args.episodes)
            available = {PurePosixPath(item.filename).parts[1] for item in episode_items}
            if not selected_episodes <= available:
                raise ValueError("Requested parent missing from verified archive index")
        selected = [item for item, name in zip(members, names) if item.is_file() and
                    ((len(name.parts) == 2 and name.suffix == ".json") or
                     (len(name.parts) == 3 and name.parts[1] in selected_episodes
                      and name.name in ("episode.npz", "metadata.json", "diagnostics.json")))]
        expanded = sum(item.file_size for item in selected)
        packed = sum(item.compress_size for item in selected)
        if (any(item.file_size > 2 * 1024**3 for item in selected)
                or any(item.file_size > 16 * 1024**2 for item in selected if item.filename.endswith(".json"))
                or shutil.disk_usage(output).free < expanded + packed + 25 * 1024**3):
            raise RuntimeError("Sample exceeds size/free-disk bounds; inventory remains available")
        remote.page_size = 64 * 1024 if args.metadata_only else 16 * 1024**2
        saved = []
        for item in selected:
            path = output.joinpath(*safe_member(item).parts)
            path.parent.mkdir(parents=True, exist_ok=True)
            print(json.dumps({"extract": item.filename, "bytes": item.file_size}), flush=True)
            # rarfile checks decompressed CRC; extraction emits bytes only, no archive paths or code.
            with archive.open(item) as source, path.open("xb") as destination:
                total = 0
                while chunk := source.read(1024**2):
                    total += len(chunk)
                    if total > item.file_size:
                        raise ValueError("Expanded member exceeds declared size")
                    destination.write(chunk)
                if total != item.file_size:
                    raise ValueError("Truncated extracted member")
            saved.append(str(path.relative_to(output)))
        report.update(sampled_episodes=sorted(selected_episodes), extracted=saved,
                      transferred_bytes=remote.transferred, range_requests=remote.requests,
                      scope="Bounded archive prefix/sample only unless full_archive_inventory is true. Not a full-release validation.")
        (output / "archive_inventory.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps({"sampled_episodes": report["sampled_episodes"],
                          "transferred_bytes": remote.transferred, "range_requests": remote.requests}), flush=True)
        archive.close()


if __name__ == "__main__":
    main()
