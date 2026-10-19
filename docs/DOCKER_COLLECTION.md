# Automated collection in Docker

`docker/collect.Dockerfile` builds a slim image that contains only what `scripts/collect_random.py`
needs. It does not contain `data/`, `results/`, `learning/`, docs, the full upstream robot checkout
or openpi.

| In the image | Size |
| --- | --- |
| `python:3.13-slim` + GLVND EGL/GL loaders | ~110 MB |
| `mujoco==3.11.0`, `numpy==2.5.3`, `Pillow==12.3.0` (same versions as the dev venv) | ~200 MB |
| OpenArm v1 bimanual XML + meshes (`third_party/openarm_mujoco/v1`), upstream LICENSE | ~47 MB |
| `connector/ envs/ controllers/ data_pipeline/ configs/ assets/connector/` (includes `controllers/faults.py`, `envs/d435.py`) | < 1 MB |
| `scripts/collect_random.py`, `rollout_viewer.py`, `replay_pilot.py`, `docker_selfcheck.py`, 9 unit-test files | < 0.2 MB |
| GPU image only: GLVND EGL vendor file for the NVIDIA driver (`docker/10_nvidia.json`) | < 1 kB |

The default (GPU) image is **507 MB**, and **~244 MB** as the `docker save | gzip` file you copy. A CPU variant (`--build-arg CPU_RENDER=1`) keeps Mesa and adds
OSMesa, at **796 MB**. `.dockerignore` whitelists the paths above, so the multi-GB `data/` never enters the
build context.

## Build and move to the collection machine

```bash
# on a machine with the repo (Docker installed; third_party/openarm_mujoco fetched)
docker build -f docker/collect.Dockerfile -t openarm-collect .
docker save openarm-collect | gzip > openarm-collect.tar.gz
# copy the .tar.gz, then on the collection machine:
docker load < openarm-collect.tar.gz
```

## Linux PC with an NVIDIA GPU

Cameras render with EGL on the GPU (`MUJOCO_GL=egl` is the image default). **Status: this GPU/EGL path is
still not verified on a native Linux NVIDIA machine.** On the Windows development machine the image could
only be checked statically and through Mesa's EGL (see "What was and was not verified" below), so run the
self-check first.

### Host prerequisites (once)

1. An NVIDIA driver **with its graphics libraries**. A compute-only / headless driver package may lack
   `libEGL_nvidia.so.0` (on Ubuntu it ships in `libnvidia-gl-<driver version>`). Check with `nvidia-smi`
   and `ldconfig -p | grep libEGL_nvidia`.
2. Docker Engine plus the
   [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html),
   registered with Docker:
   ```bash
   sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker
   docker run --rm --gpus all ubuntu nvidia-smi        # sanity check: lists your GPU
   ```
3. The image: `docker load < openarm-collect.tar.gz` (see above) or build it on that machine.

The image sets `NVIDIA_DRIVER_CAPABILITIES=graphics,compute,utility`; `graphics` is what makes the toolkit
inject the driver's EGL library. It also ships GLVND's EGL vendor file `10_nvidia.json` (GPU image only), for
toolkit versions that inject `libEGL_nvidia.so.0` but not the vendor file; the `:cpu` image keeps Mesa's
vendor file only.

### 1. Run the self-check first

`mkdir -p data` **before** the first `docker run`: if the folder does not exist, Docker creates it
root-owned and a `--user` container cannot write to it. Pass `--user $(id -u):$(id -g)` on every run that
writes to `data`.

```bash
mkdir -p data
docker run --rm --gpus all --user $(id -u):$(id -g) -v "$PWD/data:/app/data" \
  openarm-collect scripts/docker_selfcheck.py          # add --skip-scene for a ~2 s run
```

It prints `[INFO]` lines (environment, GLVND EGL vendor files, whether `libEGL_nvidia.so.0` / `libcuda.so.1`
load, `nvidia-smi -L`), then `[PASS]` / `[FAIL] ... -> fix: ...` lines, and exits 1 if anything failed. On a
working machine the render line ends with `GL_RENDERER=` naming your NVIDIA GPU; `llvmpipe` would mean
software rendering. The last check builds the `workspace_v2_cable30` scene (about a minute on a GPU host, one to
two minutes under OSMesa), reads RGB from
the three cameras and every depth stream the manifest declares (uint16, same size as its RGB, valid
fraction), and prints the mean time of 10 `observe()` calls.

| Failing line | Meaning and fix |
| --- | --- |
| `/app/data is not writable` | The host `data` folder is root-owned, usually because Docker auto-created it. `mkdir -p data` before `docker run`, or `sudo chown $(id -u):$(id -g) data`. On SELinux hosts add `:Z` to the `-v`. |
| `tiny render failed with MUJOCO_GL=egl`, and `[INFO] NVIDIA libraries` says `libEGL_nvidia.so.0 NOT loadable` | The toolkit did not inject the NVIDIA EGL library. Use `--gpus all`, keep `NVIDIA_DRIVER_CAPABILITIES` including `graphics` (the image default; do not override it with a shorter value), and check host prerequisites 1 and 2. Docker Desktop on Windows/WSL2 cannot do this: use the `:cpu` image. |
| `tiny render failed` but `libEGL_nvidia.so.0` loads | EGL found the library but no usable vendor or device. The `[INFO] EGL vendor files` line should list `10_nvidia.json`. Update the host driver and toolkit so they match. |
| `tiny render failed` with `MUJOCO_GL=osmesa` | OSMesa exists only in the `:cpu` image. |
| `PYOPENGL_PLATFORM=... does not match MUJOCO_GL` | Unset `PYOPENGL_PLATFORM`; MuJoCo derives it. A mismatch makes MuJoCo drop `mujoco.Renderer`. |
| `collection scene: ...` | A camera or depth stream is missing or malformed: the image does not match the repo sources. Rebuild it. |

### 2. Collect

```bash
# unit tests (75 in the image; 3-4 min under OSMesa on 2026-10-16; peaks at about 4.5 GiB, so run it alone):
docker run --rm --gpus all openarm-collect -m unittest discover -s tests
# collect (episodes, the scene cache and failed attempts go to ./data on the host):
docker run --rm --gpus all --user $(id -u):$(id -g) -v "$PWD/data:/app/data" openarm-collect \
  scripts/collect_random.py --workspace configs/workspace_v2_cable30.json \
  --episodes 50 --record --headless --output data/run1
# strict replay check of what was recorded:
docker run --rm --gpus all --user $(id -u):$(id -g) -v "$PWD/data:/app/data" openarm-collect \
  scripts/replay_pilot.py --dataset data/run1
```

All `collect_random.py` options work unchanged. `python` is the entrypoint, so pass the script path and its
arguments. Without a display the live viewer is skipped automatically; `--headless` just makes that
explicit. A headless server needs no X server or display: EGL renders on the GPU without one.

### Variations

- **SELinux (Fedora/RHEL):** `-v "$PWD/data:/app/data:Z"`.
- **Podman, or Docker with CDI:** generate the CDI spec once (`sudo nvidia-ctk cdi generate
  --output=/etc/cdi/nvidia.yaml`) and use `--device nvidia.com/gpu=all` instead of `--gpus all`. Podman also
  needs `--security-opt=label=disable` unless the volume uses `:Z`.
- **One GPU of several:** `--gpus '"device=1"'`; the container then sees it as its only GPU. Without that,
  MuJoCo uses the first EGL device that initialises; `-e MUJOCO_EGL_DEVICE_ID=<n>` pins an index into EGL's
  device list.
- **Permissions:** the image creates `/app/data` with mode 1777, and Docker applies that to an anonymous or
  named volume on every start, so `--user` works with those. A **bind mount** (`-v "$PWD/data:..."`) keeps
  the host folder's owner and mode: that is the case the self-check's writable test guards.

### No NVIDIA GPU, or EGL cannot be made to work

Use the CPU variant (`--build-arg CPU_RENDER=1`, tag `openarm-collect:cpu`), run it with
`-e MUJOCO_GL=osmesa` and no `--gpus`.

A preflight-only run (without `--record`) renders nothing **only with `--video none`**, and then works in
either image. The default `--video failures` renders a 640×480 review frame every 2nd step and keeps up to
3000 of them in RAM, about 1.2 GiB for a 2700-step scene, even in preflight. In `:cpu` that is slow, and it
OOM-killed a 4 GiB container (2026-10-16). With `--video none` the same fault preflight peaked at 2.9 GiB.

**Recording is much slower:** with the three cameras and three depth streams the self-check measured a mean
`observe()` of 2.1 to 4.7 s under OSMesa/llvmpipe on the Docker Desktop VM (2026-10-16, depending on other
containers running), against 62 ms with the RTX 4050 on the Windows host. Use `:cpu` for tests and small
runs; plan long recordings on a GPU.

```bash
docker run --rm --user $(id -u):$(id -g) -v "$PWD/data:/app/data" -e MUJOCO_GL=osmesa \
  openarm-collect:cpu scripts/docker_selfcheck.py
```

**Docker Desktop on Windows (WSL2) counts as "no GPU" for rendering.** It passes the NVIDIA GPU through for
compute only: `nvidia-smi` works in the container, but no NVIDIA EGL library is injected. The GPU image
therefore fails with `Cannot initialize a EGL device display` (checked 2026-10-15 and again 2026-10-16 on an
RTX 4050 laptop), so use the `:cpu` image there.
Do not set `PYOPENGL_PLATFORM`. MuJoCo derives it from `MUJOCO_GL`, and a mismatch makes it silently drop
`mujoco.Renderer` (error: `module 'mujoco' has no attribute 'Renderer'`).

### What was and was not verified (2026-10-16, Windows, Docker Desktop, no Linux host)

Checked:

- The GPU image contains a valid `/usr/share/glvnd/egl_vendor.d/10_nvidia.json`; the `:cpu` image does not
  (its only vendor file is Mesa's `50_mesa.json`). `/app/data` is mode 1777.
- **Mesa EGL proxy:** the `:cpu` image with `-e MUJOCO_GL=egl` passes the whole self-check, so the GLVND ->
  EGL -> MuJoCo path works, including the tiny render, depth readback and the collection scene with all RGB
  and depth streams. This exercises the code path with Mesa's software device (`llvmpipe`), not the NVIDIA
  driver.
- The GPU image with `--gpus all` on Docker Desktop fails the self-check with the expected diagnosis: `libcuda`
  loads and `nvidia-smi -L` lists the RTX 4050, but `libEGL_nvidia.so.0` is not loadable.
- `--user 1000:1000` with an anonymous volume passes (no permission errors, no shader-cache warnings). A
  root-owned bind mount produces the `/app/data is not writable` FAIL with its fix text.

**Not verified:** `libEGL_nvidia.so.0` being injected and used on a native Linux host with the NVIDIA
Container Toolkit, and therefore GPU rendering speed in the container. Run the self-check there before a
long recording.

## Recovery demonstrations and the chest RGB-D camera

The image contains the 2026-10-14/15 collection features (`docs/DATA_COLLECTION.md`):

- **Injected-fault recovery demonstrations:** add `--faults all` (or a comma list of
  `premature_close`, `wrong_grasp`, `misaligned_insert`) and use a separate `--output`. Episodes then
  carry `step_label`. The `fault` steps must be masked before behaviour cloning; the current VLA loaders
  do not do this.
- **Depth on every camera**, in `workspace_v2_cable30` only (uint16 mm, aligned to each RGB):
  - the chest RealSense D435-like camera (2026-10-15);
  - since 2026-10-16, the overhead camera (D435 model) and the wrist camera (D405 model).
- **Joint torque arrays** in every recorded episode.

```bash
docker run --rm --gpus all --user $(id -u):$(id -g) -v "$PWD/data:/app/data" openarm-collect   scripts/collect_random.py --workspace configs/workspace_v2_cable30.json   --faults all --episodes 30 --record --headless --output data/recovery_run1
```

Rebuild the image after pulling these changes. Datasets recorded with an older image or code no longer
pass strict replay against the new sources; their physics is unchanged.

## Stop, resume, pause

`collect_random.py --record` handles one scene at a time. It preflights the scene without images and,
if the preflight succeeds, records it right away. `progress.json` in the output folder is updated after
every stage.

- **Resume** after a crash, Ctrl+C, reboot or `docker stop`: run the **same command** again with
  `--resume` added. A scene that was being recorded is recorded again, and its half-written folder is
  replaced. A scene that was in preflight is preflighted again. Finished episodes are never redone.
  Workspace, detector and validation fraction must match the first run; `--episodes` may be raised to
  extend a run.
- **Pause** without stopping: start the container with `-d --name collect1`, then
  `docker pause collect1` / `docker unpause collect1`. This freezes the process in place; nothing is
  lost or redone. It does not survive a reboot (use `--resume` then).

```bash
docker run -d --name collect1 --gpus all --user $(id -u):$(id -g) -v "$PWD/data:/app/data" openarm-collect \
  scripts/collect_random.py --workspace configs/workspace_v2_cable30.json --episodes 50 --record --headless --output data/run1
docker logs -f collect1          # watch progress (Ctrl+C only stops watching)
# after a stop: docker rm collect1, then the same docker run line with --resume appended
```

## Notes

- **RAM while recording `workspace_v2_cable30` (all cameras with depth): plan on about 10 GB per
  recording container.**
  - **Why:** a recording keeps every observation in memory until the episode is saved, at 1.22 MiB per
    step (3 RGB + 3 depth). `save_episode` then stacks them into arrays, which copies them once more.
  - **Measured 2026-10-16** on the Windows host, with a synthetic 2700-step episode (a real cable30
    observation copied 2701 times, saved to a temporary folder):
    - 1.6 GiB with the env built;
    - 4.8 GiB holding the observations;
    - **8.0 GiB peak working set, 9.7 GiB peak commit, during the save.**
  - **Video:** the default `--video failures` adds about 1.2 GiB more for its review frames; pass
    `--video none` to drop it.
  - **The 7.6 GiB Docker Desktop VM cannot record a full-length cable30 episode.** Record on the Windows
    host or on the Linux PC, or raise the VM's memory limit.
  - On the Linux PC, run as many recording containers as free RAM / 10 GB allows.
  - **Other loads:**
    - a physics-only preflight peaked at 2.8–2.9 GiB (`--faults`, `--video none`, 2026-10-16);
    - the full in-image unit suite at 4.3–4.5 GiB. Two suites in parallel wedged the 7.6 GiB VM, and
      Docker had to be restarted.
  - Before the chest camera the measured peak was 2.96 GiB per collector.
  - To run several in parallel, start several containers, each with its own `--output`. They can share
    `./data/.scene_cache`, whose writes are atomic.
  - Exit code 137 means out of memory: rerun with `--resume`.
- **Scene cache:** the first run on a new machine generates the scenes (slow; see
  `docs/AUTOMATED_COLLECTION_LOG.md`). Keep `./data` mounted so later runs reuse the cache. Don't copy a
  cache across operating systems.
- **Replay and provenance:** all source-file hashes in `env.manifest()` are identical in the container and
  on the Windows dev machine. The generated scene XML hash is not: cable rest quaternions differ at
  the 1e-19 level between Linux and Windows floating-point libraries. Strict replay (`replay_pilot.py`)
  is therefore exact **within the same image/platform**. Replay container-recorded datasets in the
  container, and don't mix container and Windows recordings in one strict-replay check.
- **Disk:** without the chest camera a recorded episode is about 155 MB: `episode.npz` plus about 71 MB of
  JPEG frames (`cameras/head`, `cameras/wrist`), so 50 episodes need about 8 GB. **With the chest camera
  (`workspace_v2_cable30`) an episode is about 0.4–0.6 GB** (measured on Windows), because it adds the
  chest JPEGs and the 16-bit depth PNGs.
  **With depth on all three cameras (2026-10-16) it is about 0.34–0.38 MB per step**, about 0.75 GB for
  a 2000-step episode and 0.9–1 GB for a 2700-step fault episode. 50 episodes then need about 40–50 GB.
- **Rebuild** the image after any code or config change; the image is a snapshot of the repo at build time.
- Variance of the sampled scenes is unchanged from the repo; the image only packages the collector.
