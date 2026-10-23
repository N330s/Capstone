"""Temporary Windows sleep prevention tied to one existing campaign process.

Does not change global power settings or keep the display on. A process handle
prevents PID reuse from extending the request to an unrelated process.
"""
import argparse
import ctypes
from ctypes import wintypes
import json
import os
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pid", type=int, required=True)
    parser.add_argument("--max-seconds", type=int, default=21600)
    args = parser.parse_args()
    if os.name != "nt" or args.pid <= 0 or not 0 < args.max_seconds <= 32400:
        raise ValueError("Windows, positive PID and <=9-hour guard required")
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    kernel.WaitForSingleObject.restype = wintypes.DWORD
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel.SetThreadExecutionState.argtypes = [wintypes.DWORD]
    kernel.SetThreadExecutionState.restype = wintypes.DWORD
    handle = kernel.OpenProcess(0x00100000, False, args.pid)
    if not handle:
        raise ctypes.WinError(ctypes.get_last_error())
    started = time.monotonic()
    try:
        if kernel.WaitForSingleObject(handle, 0) == 0:
            return
        if not kernel.SetThreadExecutionState(0x80000001):
            raise ctypes.WinError(ctypes.get_last_error())
        print(json.dumps({"awake_request_active": True, "campaign_pid": args.pid,
                          "global_settings_changed": False, "display_required": False,
                          "maximum_seconds": args.max_seconds}), flush=True)
        while time.monotonic() - started < args.max_seconds:
            result = kernel.WaitForSingleObject(handle, 1000)
            if result == 0:
                break
            if result != 0x102:
                raise ctypes.WinError(ctypes.get_last_error())
    finally:
        kernel.SetThreadExecutionState(0x80000000)
        kernel.CloseHandle(handle)
        print(json.dumps({"awake_request_active": False}), flush=True)


if __name__ == "__main__":
    main()
