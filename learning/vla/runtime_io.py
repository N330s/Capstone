"""Atomic metadata and delete-sharing reads for concurrent Windows monitoring."""
import json
import os
from pathlib import Path
import tempfile
import time


def _read_text(path):
    if os.name != "nt":
        return Path(path).read_text(encoding="utf-8")
    import ctypes
    from ctypes import wintypes
    import msvcrt
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.CreateFileW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD,
                                  wintypes.LPVOID, wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE]
    kernel.CreateFileW.restype = wintypes.HANDLE
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    handle = kernel.CreateFileW(str(Path(path).resolve()), 0x80000000, 0x1 | 0x2 | 0x4,
                               None, 3, 0x80, None)
    if handle == ctypes.c_void_p(-1).value:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        descriptor = msvcrt.open_osfhandle(handle, os.O_RDONLY | os.O_BINARY)
    except Exception:
        kernel.CloseHandle(handle)
        raise
    with os.fdopen(descriptor, "r", encoding="utf-8") as stream:
        return stream.read()


def read_json(path):
    return json.loads(_read_text(path))


def write_json(path, value, *, retry_seconds=5., durable=True):
    """Critical metadata: retry sharing violations, then raise, never write in place."""
    path = Path(path)
    payload = json.dumps(value, indent=2, allow_nan=False)
    descriptor, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(payload)
            stream.flush()
            if durable:
                os.fsync(stream.fileno())
        deadline = time.monotonic() + retry_seconds
        while True:
            try:
                os.replace(temporary, path)
                break
            except PermissionError:
                if time.monotonic() >= deadline:
                    raise
                time.sleep(min(.05, max(0, deadline - time.monotonic())))
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def write_status(path, value):
    """A transient telemetry lock must not abort optimizer updates or checkpointing."""
    try:
        write_json(path, value, retry_seconds=.25, durable=False)
        return True
    except PermissionError:
        return False  # Full event also remains in events.jsonl and process stdout.
