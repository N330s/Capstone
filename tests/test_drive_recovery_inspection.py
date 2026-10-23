import importlib.util
import io
from pathlib import Path
from types import SimpleNamespace
import unittest

if importlib.util.find_spec('rarfile') is None:
    raise unittest.SkipTest('Optional archive intake: install requirements-download.txt in its separate environment')

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/inspect_drive_recovery.py"
spec = importlib.util.spec_from_file_location("drive_inspection", SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class Response:
    def __init__(self, data, headers, status_code=206):
        self.data, self.headers, self.status_code = data, headers, status_code

    def raise_for_status(self):
        pass

    def iter_content(self, size):
        yield self.data

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


class Session:
    def __init__(self, *, status=206, modified="version1"):
        self.status, self.modified = status, modified

    def get(self, url, **kwargs):
        span = kwargs["headers"]["Range"].split("=")[1]
        start, end = map(int, span.split("-"))
        return Response(b"abcdefghij"[start:end + 1],
                        {"Content-Range": f"bytes {start}-{end}/10", "Last-Modified": self.modified},
                        self.status)

    def close(self):
        pass


def reader(**kwargs):
    remote = module.DriveRangeReader.__new__(module.DriveRangeReader)
    io.RawIOBase.__init__(remote)
    remote.session, remote.url, remote.params = Session(**kwargs), "unused", {}
    remote.size, remote.modified, remote.position = 10, "version1", 0
    remote.page_size, remote.cache_start, remote.cache = 4, 0, b""
    remote.transferred, remote.requests = 0, 0
    remote.max_bytes, remote.max_requests = 100, 10
    return remote


class DriveInspectionTests(unittest.TestCase):
    def test_bounded_seek_and_cache(self):
        with reader() as remote:
            self.assertEqual(remote.read(2), b"ab")
            remote.seek(1)
            self.assertEqual(remote.read(2), b"bc")
            self.assertEqual(remote.requests, 1)
            self.assertEqual(remote.read(6), b"defghi")
            remote.seek(-2, 2)
            self.assertEqual(remote.read(9), b"ij")

    def test_unbounded_or_oversized_transfer_refused(self):
        with reader() as remote:
            with self.assertRaises(ValueError):
                remote.read()
            remote.max_bytes = 3
            with self.assertRaises(RuntimeError):
                remote.read(1)

    def test_ignored_range_and_changed_version_refused(self):
        for kwargs in ({"status": 200}, {"modified": "version2"}):
            with reader(**kwargs) as remote, self.assertRaises(ValueError):
                remote.read(1)

    def test_unsafe_member_paths_refused(self):
        for path in ("../episode.npz", "/root/a", "a/CON.txt", "a/b:c", "a/b.", "a\\b"):
            item = SimpleNamespace(filename=path, file_redir=None, is_symlink=lambda: False)
            with self.assertRaises(ValueError):
                module.safe_member(item)
        item = SimpleNamespace(filename="collection/train_00001/episode.npz", file_redir=None,
                               is_symlink=lambda: False)
        self.assertEqual(str(module.safe_member(item)), item.filename)

    def test_bsdtar_archive_flag_keeps_its_argument(self):
        import rarfile
        setup = module.BSDTarSetup(rarfile.BSDTAR_CONFIG)
        command = setup.open_cmdline(None, "C:/absolute/sample.rar", "collection/metadata.json")
        self.assertEqual(command[command.index("-f") + 1], "C:/absolute/sample.rar")
        self.assertEqual(command[-2:], ["--", "collection/metadata.json"])


if __name__ == "__main__":
    unittest.main()
