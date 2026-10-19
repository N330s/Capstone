r"""Diagnose a collection container before a long run.

    docker run --rm --gpus all --user $(id -u):$(id -g) -v "$PWD/data:/app/data" \
        openarm-collect scripts/docker_selfcheck.py [--skip-scene]

Prints [INFO] / [PASS] / [FAIL] lines (a FAIL carries "-> fix: ...") and exits 0 when nothing failed,
1 otherwise. Needs only the stdlib, numpy and mujoco. Run it from the repository root (/app).
"""
import argparse
import ctypes
import importlib.metadata
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
WORKSPACE = "configs/workspace_v2_cable30.json"
VENDOR_DIRS = ("/usr/share/glvnd/egl_vendor.d", "/etc/glvnd/egl_vendor.d")
LIGHT_XML = ("<mujoco><worldbody><light pos='0 0 2'/>"
             "<geom type='sphere' size='.1'/></worldbody></mujoco>")
MIN_VALID_FRACTION = 0.02          # a depth stream that is almost all zeros is broken
failures = 0


def info(msg):
    print(f"[INFO] {msg}", flush=True)


def passed(msg):
    print(f"[PASS] {msg}", flush=True)


def failed(msg, fix):
    global failures
    failures += 1
    print(f"[FAIL] {msg} -> fix: {fix}", flush=True)


def can_load(name):
    try:
        ctypes.CDLL(name)
        return True
    except OSError:
        return False


def show_environment():
    try:
        mujoco_version = importlib.metadata.version("mujoco")
    except importlib.metadata.PackageNotFoundError:
        mujoco_version = "NOT INSTALLED"
    ids = f"uid={os.getuid()} gid={os.getgid()}" if hasattr(os, "getuid") else "uid=n/a"
    env = " ".join(f"{k}={os.environ.get(k, '<unset>')}" for k in
                   ("MUJOCO_GL", "PYOPENGL_PLATFORM", "NVIDIA_DRIVER_CAPABILITIES", "NVIDIA_VISIBLE_DEVICES"))
    info(f"environment: {ids} python={platform.python_version()} mujoco={mujoco_version} {env}")


def show_vendor_files():
    for d in VENDOR_DIRS:
        p = Path(d)
        if not p.is_dir():
            info(f"EGL vendor files: {d} does not exist")
            continue
        entries = []
        for f in sorted(p.iterdir()):
            try:
                entries.append(f"{f.name} = {' '.join(f.read_text().split())}")
            except OSError as e:
                entries.append(f"{f.name} (unreadable: {e})")
        info(f"EGL vendor files: {d}: " + ("; ".join(entries) if entries else "empty (EGL has no vendor)"))


def show_nvidia_libraries():
    egl = can_load("libEGL_nvidia.so.0")
    info(f"NVIDIA libraries: libEGL_nvidia.so.0 {'loads' if egl else 'NOT loadable'}; "
         f"libcuda.so.1 {'loads' if can_load('libcuda.so.1') else 'NOT loadable'}")
    smi = shutil.which("nvidia-smi")
    if not smi:
        info("nvidia-smi: not on PATH (no NVIDIA runtime in this container)")
        return egl
    try:
        out = subprocess.run([smi, "-L"], capture_output=True, text=True, timeout=30)
        info("nvidia-smi -L: " + (" | ".join((out.stdout or out.stderr).split("\n")).strip(" |") or "no output"))
    except (OSError, subprocess.TimeoutExpired) as e:
        info(f"nvidia-smi -L failed: {e}")
    return egl


def check_data_writable():
    data = ROOT / "data"
    try:
        with tempfile.NamedTemporaryFile(dir=data, prefix=".selfcheck_"):
            pass
        passed(f"{data} is writable by this uid")
    except OSError as e:
        failed(f"{data} is not writable ({e})",
               "the host folder mounted at /app/data is probably root-owned, usually because docker created it "
               "when it did not exist. Run `mkdir -p data` on the host before `docker run`, or "
               "`sudo chown $(id -u):$(id -g) data`; on SELinux hosts (Fedora/RHEL) add :Z to the -v "
               "(-v \"$PWD/data:/app/data:Z\"); pass --user $(id -u):$(id -g)")


def gl_renderer_name():
    try:
        from OpenGL import GL
        return GL.glGetString(GL.GL_RENDERER).decode()
    except Exception:           # informational only
        return "unknown"


def check_tiny_render(nvidia_egl):
    gl = os.environ.get("MUJOCO_GL", "")
    try:
        import mujoco       # imports its GL backend here, so a bad MUJOCO_GL can already fail now
        model = mujoco.MjModel.from_xml_string(LIGHT_XML)
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        renderer = mujoco.Renderer(model, 64, 64)
        try:
            renderer.update_scene(data)
            rgb = renderer.render().copy()
            name = gl_renderer_name()
            renderer.enable_depth_rendering()
            renderer.update_scene(data)
            depth = renderer.render().copy()
        finally:
            renderer.close()
        assert rgb.shape == (64, 64, 3) and depth.shape == (64, 64), (rgb.shape, depth.shape)
        passed(f"tiny render ok with MUJOCO_GL={gl or '<unset>'}: rgb {rgb.shape} depth {depth.shape}, "
               f"GL_RENDERER={name}")
        return True
    except Exception as e:      # ImportError/AttributeError too: a GL mismatch drops mujoco.Renderer
        err = f"{type(e).__name__}: {e}"
        pyopengl = os.environ.get("PYOPENGL_PLATFORM", "")
        if pyopengl and pyopengl.lower() != gl:
            fix = (f"PYOPENGL_PLATFORM={pyopengl} does not match MUJOCO_GL={gl or '<unset>'}: unset "
                   "PYOPENGL_PLATFORM (MuJoCo derives it from MUJOCO_GL; a mismatch drops mujoco.Renderer)")
        elif gl == "egl" and not nvidia_egl:
            fix = ("NVIDIA EGL library not injected. Run with `--gpus all` on a Linux host with the NVIDIA "
                   "Container Toolkit; check that `NVIDIA_DRIVER_CAPABILITIES` includes graphics. "
                   "Docker Desktop/WSL2 can't do this; use `openarm-collect:cpu` with `-e MUJOCO_GL=osmesa`")
        elif gl == "egl":
            fix = ("libEGL_nvidia loads but EGL still fails: check the EGL vendor files listed above "
                   "(10_nvidia.json must be present), that `NVIDIA_DRIVER_CAPABILITIES` includes graphics, "
                   "and that the host driver and NVIDIA Container Toolkit are current")
        elif gl == "osmesa":
            fix = "OSMesa is only in the CPU image: use `openarm-collect:cpu` with `-e MUJOCO_GL=osmesa`"
        else:
            fix = ("set `-e MUJOCO_GL=egl` (GPU image, `--gpus all`) or `-e MUJOCO_GL=osmesa` (`:cpu` image); "
                   "do not set PYOPENGL_PLATFORM")
        failed(f"tiny render failed with MUJOCO_GL={gl or '<unset>'} ({err})", fix)
        return False


def depth_streams(manifest):
    declared = manifest.get("depth_cameras")
    if declared is None:
        return {"chest"} if manifest.get("chest_rgbd") else set()
    return set(declared) if isinstance(declared, dict) else {
        x if isinstance(x, str) else x.get("name") for x in declared}


def check_scene():
    import numpy as np
    from data_pipeline.scene_bank import as_reset_options, iter_collection_bank, workspace_for_env
    from envs.openarm_insert import OpenArmInsertEnv
    env = None
    try:
        env = OpenArmInsertEnv(images=True, workspace=WORKSPACE)
        row = next(iter(iter_collection_bank(n=1, validation_fraction=0.1, ws=workspace_for_env(env), start=6100)))
        obs, _ = env.reset(seed=row["seed"], options=as_reset_options(row["options"]))
        images, depth = obs.get("images", {}), obs.get("depth", {})
        problems, fractions = [], []
        for name in ("scene_rgb", "wrist_rgb", "chest_rgb"):
            img = images.get(name)
            if img is None:
                problems.append(f"images[{name!r}] missing")
            elif img.ndim != 3 or img.shape[2] != 3 or img.dtype != np.uint8 or img.min() == img.max():
                problems.append(f"images[{name!r}] is {img.shape} {img.dtype}, blank or malformed")
        expected = depth_streams(env.manifest())
        for stream in sorted(expected):
            z, rgb = depth.get(stream), images.get(f"{stream}_rgb")
            if z is None:
                problems.append(f"depth[{stream!r}] missing")
                continue
            valid = float((z > 0).mean())
            fractions.append(f"{stream}={valid:.2f}")
            if z.dtype != np.uint16:
                problems.append(f"depth[{stream!r}] dtype {z.dtype}, expected uint16")
            if rgb is not None and z.shape != rgb.shape[:2]:
                problems.append(f"depth[{stream!r}] {z.shape} does not match its RGB {rgb.shape[:2]}")
            if valid < MIN_VALID_FRACTION:
                problems.append(f"depth[{stream!r}] valid fraction {valid:.3f} < {MIN_VALID_FRACTION}")
        t0 = time.perf_counter()
        for _ in range(10):
            env.observe()
        ms = (time.perf_counter() - t0) / 10 * 1e3
        summary = (f"seed {row['seed']}; images {sorted(images)}; depth streams {sorted(expected) or 'none'} "
                   f"valid fractions {' '.join(fractions) or 'n/a'}; observe() mean {ms:.0f} ms over 10 calls")
        if problems:
            failed("collection scene: " + "; ".join(problems) + f" ({summary})",
                   "rebuild the image from the current repo (`docker build -f docker/collect.Dockerfile -t "
                   "openarm-collect .`) so that it matches the sources that define the camera streams")
        else:
            passed(f"collection scene {WORKSPACE}: {summary}")
    except Exception as e:
        failed(f"collection scene could not be built or observed ({type(e).__name__}: {e})",
               "a missing-file error means the image was built without third_party/openarm_mujoco "
               "(run scripts/fetch_openarm.ps1, then rebuild); a render error means the tiny-render fix applies")
    finally:
        if env is not None:
            env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--skip-scene", action="store_true", help="skip the (slow) collection-scene check")
    args = parser.parse_args()
    sys.unraisablehook = lambda unraisable: None    # mujoco's GLContext.__del__ spams after a failed EGL init
    show_environment()
    show_vendor_files()
    nvidia_egl = show_nvidia_libraries()
    check_data_writable()
    rendered = check_tiny_render(nvidia_egl)
    if args.skip_scene:
        info("collection scene: skipped (--skip-scene)")
    elif not rendered:
        info("collection scene: skipped because the tiny render failed")
    else:
        check_scene()
    info(f"result: {failures} check(s) failed" if failures else "result: all checks passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
