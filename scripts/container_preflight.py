"""Check the container infrastructure without downloading or loading VLA weights."""
import argparse
from datetime import datetime, timezone
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def check_versions():
    expected = {"torch": "2.10.0", "torchvision": "0.25.0", "lerobot": "0.4.4",
                "mujoco": "3.11.0", "numpy": "2.2.6", "av": "15.1.0",
                "transformers": "4.57.6", "datasets": "4.8.5"}
    actual = {name: version(name) for name in expected}
    if sys.version_info[:2] != (3, 11):
        raise ValueError("Expected Python 3.11")
    for name, wanted in expected.items():
        if actual[name].split("+")[0] != wanted:
            raise ValueError(f"Expected {name}=={wanted}, found {actual[name]}")
    import torch
    if torch.version.cuda != "12.8":
        raise ValueError(f"Expected CUDA 12.8 PyTorch build, found {torch.version.cuda}")
    return actual


def check_dependencies():
    result = subprocess.run([sys.executable, "-m", "pip", "check"], capture_output=True, text=True, timeout=60)
    if result.returncode:
        raise RuntimeError((result.stdout + result.stderr).strip())
    return result.stdout.strip()


def check_smolvla_imports():
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from lerobot.policies.factory import make_pre_post_processors
    if any(os.environ.get(key) != "1" for key in ("HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE")):
        raise ValueError("Offline flags must remain enabled")
    return {"policy_class": SmolVLAPolicy.__name__, "processors": make_pre_post_processors.__name__,
            "weights_loaded": False}


def check_cuda():
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; check host driver, Docker GPU runtime and selected GPU")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16 is not supported by the selected GPU")
    x = torch.ones((64, 64), device="cuda", dtype=torch.bfloat16)
    y = x @ x
    torch.cuda.synchronize()
    if not bool(torch.isfinite(y).all()) or float(y[0, 0]) != 64.:
        raise AssertionError("CUDA BF16 matrix multiplication failed")
    free, total = torch.cuda.mem_get_info()
    return {"gpu": torch.cuda.get_device_name(), "cuda": torch.version.cuda,
            "free_gb": free / 1e9, "total_gb": total / 1e9, "bf16_matmul": True}


def make_test_video(path):
    import av
    import numpy as np
    with av.open(str(path), mode="w") as output:
        stream = output.add_stream("libx264", rate=50)
        stream.width, stream.height, stream.pix_fmt = 96, 64, "yuv420p"
        for n in range(3):
            pixels = np.full((64, 96, 3), 40 + n * 40, dtype=np.uint8)
            frame = av.VideoFrame.from_ndarray(pixels, format="rgb24")
            for packet in stream.encode(frame):
                output.mux(packet)
        for packet in stream.encode():
            output.mux(packet)


def check_pyav(path):
    import av
    with av.open(str(path)) as video:
        frames = [f.to_ndarray(format="rgb24") for f in video.decode(video=0)]
    if len(frames) != 3 or any(f.shape != (64, 96, 3) for f in frames):
        raise AssertionError("PyAV H.264 round-trip failed")
    return {"frames": len(frames), "rgb_shape": list(frames[0].shape)}


def check_torchcodec(path):
    if version("torchcodec").split("+")[0] != "0.10.0":
        raise ValueError("Expected TorchCodec 0.10.0 for PyTorch 2.10")
    from torchcodec.decoders import VideoDecoder
    frame = VideoDecoder(str(path), device="cpu")[0]
    if tuple(frame.shape) != (3, 64, 96):
        raise AssertionError("TorchCodec CPU H.264 decoding failed")
    return {"version": version("torchcodec"), "frame_shape": list(frame.shape), "device": str(frame.device)}


def check_render():
    import mujoco
    import numpy as np
    if os.environ.get("MUJOCO_GL") != "egl":
        raise ValueError("Expected MUJOCO_GL=egl; GPU visibility alone does not validate headless rendering")
    model = mujoco.MjModel.from_xml_string(
        '<mujoco><worldbody><light pos="0 0 3"/><geom type="plane" size="2 2 .1"/>'
        '<geom type="sphere" size=".15" pos="0 0 .2" rgba="1 0 0 1"/></worldbody></mujoco>')
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    with mujoco.Renderer(model, height=64, width=96) as renderer:
        renderer.update_scene(data)
        frame = renderer.render()
    if frame.shape != (64, 96, 3) or not np.isfinite(frame).all() or frame.std() == 0:
        raise AssertionError("MuJoCo offscreen render failed")
    return {"rgb_shape": list(frame.shape), "backend": "egl", "scene": "synthetic infrastructure probe, not task evaluation"}


def check_storage():
    paths = [Path(p) for p in ("/raw", "/models", "/assets", "/datasets", "/runs", "/cache")]
    for path in paths:
        if not path.is_dir():
            raise FileNotFoundError(f"Missing persistent mount: {path}")
    for path in paths[3:]:
        with tempfile.TemporaryFile(dir=path) as stream:
            stream.write(b"openarm-storage-probe")
            stream.flush()
    return {"input_mounts": [str(p) for p in paths[:3]], "writable_mounts": [str(p) for p in paths[3:]],
            "note": "Input read-only protection is configured by Compose, not tested by writing to inputs"}


def summarize(checks):
    failed = [name for name, result in checks.items() if result["status"] == "failed"]
    skipped = [name for name, result in checks.items() if result["status"] == "skipped"]
    return {"passed": not failed and not skipped, "selected_checks_passed": not failed,
            "failed": failed, "skipped": skipped}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="New report file; existing output is refused")
    for name in ("cuda", "render", "torchcodec", "storage"):
        parser.add_argument("--skip-" + name, action="store_true", help="Diagnostic only; skips prevent a full pass")
    args = parser.parse_args()
    if args.output and args.output.exists():
        raise FileExistsError(args.output)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    checks = {}
    def run(name, function, skip=False):
        if skip:
            checks[name] = {"status": "skipped"}
            return
        try:
            checks[name] = {"status": "passed", "details": function()}
        except Exception as exc:
            checks[name] = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
    run("runtime_versions", check_versions)
    run("dependency_health", check_dependencies)
    run("smolvla_imports", check_smolvla_imports)
    run("cuda", check_cuda, args.skip_cuda)
    with tempfile.TemporaryDirectory(prefix="openarm-preflight-") as temporary:
        path = Path(temporary) / "probe.mp4"
        run("video_encode", lambda: make_test_video(path))
        run("pyav_decode", lambda: check_pyav(path))
        run("torchcodec_decode", lambda: check_torchcodec(path), args.skip_torchcodec)
    run("offscreen_render", check_render, args.skip_render)
    run("persistent_storage", check_storage, args.skip_storage)
    report = {**summarize(checks), "timestamp_utc": datetime.now(timezone.utc).isoformat(),
              "platform": platform.platform(), "python": sys.version, "checks": checks,
              "limits": "Infrastructure only; no model-weight, dataset, training, task-success or resume qualification."}
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2)
    print(json.dumps(report, indent=2))
    return 0 if report["selected_checks_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
