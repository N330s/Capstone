# Lab runtime setup

Updated 2026-10-23. Start with [LAB_HANDOFF.md](LAB_HANDOFF.md) for the exact data
and training experiment. These are host/container requirements, not a claim that
the container has already run successfully on the lab 4090.

## Host requirements

- Approved persistent SSD allocation: **150–200 GB free** is a practical initial
  allocation, including input copies, checkpoints and headroom. Check Docker's
  separate image/build-cache disk too. Do not prune other users' data/images.
- A working NVIDIA host driver (`nvidia-smi`), Docker Engine/Desktop and Compose v2.
  Linux needs the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).
  Windows GPU containers use the [Docker WSL2 backend](https://docs.docker.com/desktop/features/gpu/).
  Keep WSL training data on its Linux filesystem where possible. Do not install a
  GPU display driver inside the container/WSL or replace lab drivers without IT.
- Confirm GPU booking, sleep/reboot policy, persistent storage ownership and backup.
  Use approved SSH/VPN access; AnyDesk is optional host software, not part of the
  container. Test remote access before leaving. No public training ports needed.

## Prepare storage and environment

Choose an approved storage root; examples below do not authorize writes to a
system-owned directory. Create the directories shown in the handoff, stage inputs,
then copy `docker/lab.env.example` to repository-root `.env` and edit:

```dotenv
OPENARM_STORAGE=/srv/openarm-storage
OPENARM_UID=1000
OPENARM_GID=1000
OPENARM_GPU_ID=0
```

Use actual `id -u` / `id -g` values. The owner/admin should grant that account write
access to `datasets`, `runs`, `cache`; do not use broad `chmod 777`. All seven bind
sources must already exist. Raw/model/assets mounts are read-only; root filesystem
is read-only; the process is non-root with dropped capabilities, no host socket,
no exposed ports. Compose reserves [one selected GPU](https://docs.docker.com/compose/how-tos/gpu-support/).
Only persistent output/cache/data mounts and temporary filesystems are writable.

## Build and infrastructure qualification

From the repository root on Linux/WSL:

```bash
nvidia-smi
docker version
docker compose version
docker compose config --quiet
docker compose build smolvla
docker compose run --rm smolvla python scripts/container_preflight.py --output /runs/lab_preflight_001.json
```

Require report `passed=true` with no skipped checks. It checks package pins,
dependencies, CUDA BF16, video encode/decode (PyAV/TorchCodec), EGL rendering and
persistent storage. It loads no model and does not qualify native training or
task physics; those are subsequent handoff gates. A CUDA pass alone is not EGL.

The image uses Python 3.11, PyTorch 2.10.0 / torchvision 0.25.0 cu128, LeRobot 0.4.4,
MuJoCo 3.11.0, NumPy 2.2.6, PyAV 15.1.0 and Transformers 4.57.6. TorchCodec 0.10.0
uses the CPU decoding wheel; training remains CUDA. Critical pins are in
`docker/constraints-smolvla.txt`. The base image/OS packages/transitive requirements
are not a fully reproducible lock; retain image ID/digest and
`/opt/openarm/requirements.resolved.txt` with run reports.

Runtime downloads/uploads are disabled. Code is baked into the image: **rebuild
after edits or switching branches**. Datasets, model weights, credentials, Git
metadata and third-party snapshots are excluded from the image build context.
Cache and source archives use `/cache` and `/runs/source_archives`, not read-only
`/workspace`. A saved model's old Windows tokenizer path is overridden at runtime.

## Status and caveats

Local Windows source/configuration tests do not qualify Linux CUDA/EGL or
cross-platform physics. The Docker engine is unavailable on the development PC;
the image is **not yet Linux GPU build/run qualified**. Do not bypass failed
checks, alter source hashes, loosen physics limits or describe the recipe as a
completed lab setup.

The previous four-episode/custom-training guide is retained only in
[the archive](archive/LAB_SETUP_20261008.md). Follow the native full-release handoff
instead. Host reboots/container removal stop jobs; persistent reports survive,
but no automatic restart/resume is configured.
