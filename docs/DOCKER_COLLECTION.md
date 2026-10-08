# Automated collection in Docker

`docker/collect.Dockerfile` builds a slim image that contains only what `scripts/collect_random.py`
needs. It does not contain `data/`, `results/`, `learning/`, docs, the full upstream robot checkout
or openpi.

| In the image | Size |
| --- | --- |
| `python:3.13-slim` + GLVND EGL/GL loaders | ~110 MB |
| `mujoco==3.11.0`, `numpy==2.5.3`, `Pillow==12.3.0` (same versions as the dev venv) | ~200 MB |
| OpenArm v1 bimanual XML + meshes (`third_party/openarm_mujoco/v1`), upstream LICENSE | ~47 MB |
| `connector/ envs/ controllers/ data_pipeline/ configs/ assets/connector/` | < 1 MB |
| `scripts/collect_random.py`, `rollout_viewer.py`, `replay_pilot.py`, 6 unit-test files | < 0.2 MB |

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

## Run (Linux + NVIDIA GPU)

Prerequisites: an NVIDIA driver plus the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).
Cameras render with EGL on the GPU (`MUJOCO_GL=egl` is the image default). The toolkit injects the
driver's EGL library because the image sets `NVIDIA_DRIVER_CAPABILITIES=graphics,compute,utility`.

```bash
mkdir -p data
# check GPU rendering first (should print (64, 64, 3)):
docker run --rm --gpus all openarm-collect -c "import mujoco; m=mujoco.MjModel.from_xml_string('<mujoco><worldbody><light/></worldbody></mujoco>'); print(mujoco.Renderer(m,64,64).render().shape)"
# unit tests (38, about 1 min):
docker run --rm --gpus all openarm-collect -m unittest discover -s tests
# collect (episodes, the scene cache and failed attempts go to ./data on the host):
docker run --rm --gpus all --user $(id -u):$(id -g) -v $PWD/data:/app/data openarm-collect \
  scripts/collect_random.py --workspace configs/workspace_v2_cable30.json \
  --episodes 50 --record --headless --output data/run1
# strict replay check of what was recorded:
docker run --rm --gpus all -v $PWD/data:/app/data openarm-collect scripts/replay_pilot.py --dataset data/run1
```

All `collect_random.py` options work unchanged. `python` is the entrypoint, so pass the script path and its
arguments. Without a display the live viewer is skipped automatically; `--headless` just makes that
explicit.

No NVIDIA GPU: build with `--build-arg CPU_RENDER=1` and run with `-e MUJOCO_GL=osmesa` and no `--gpus`.
Recording is then slower. If the EGL check above fails on a GPU machine, use the CPU variant until the
toolkit is fixed. A preflight-only run (without `--record`) renders nothing and works in either image.

## Notes

- **RAM:** about 1.5 GB per collector process. To run several in parallel, start several containers, each
  with its own `--output`. They can share `./data/.scene_cache`, whose writes are atomic.
- **Scene cache:** the first run on a new machine generates the scenes (slow; see
  `docs/AUTOMATED_COLLECTION_LOG.md`). Keep `./data` mounted so later runs reuse the cache. Don't copy a
  cache across operating systems.
- **Replay and provenance:** all source-file hashes in `env.manifest()` are identical in the container and
  on the Windows dev machine. The generated scene XML hash is not: cable rest quaternions differ at
  the 1e-19 level between Linux and Windows floating-point libraries. Strict replay (`replay_pilot.py`)
  is therefore exact **within the same image/platform**. Replay container-recorded datasets in the
  container, and don't mix container and Windows recordings in one strict-replay check.
- **Disk:** a recorded episode is `episode.npz` plus about 71 MB of JPEG frames
  (`cameras/head`, `cameras/wrist`).
- **Rebuild** the image after any code or config change; the image is a snapshot of the repo at build time.
- Variance of the sampled scenes is unchanged from the repo; the image only packages the collector.
