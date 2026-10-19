# Slim image for automated data collection only (scripts/collect_random.py + replay_pilot.py).
# Build from the repo root:  docker build -f docker/collect.Dockerfile -t openarm-collect .
# Usage: docs/DOCKER_COLLECTION.md
FROM python:3.13-slim

# Default (GPU): only the GLVND EGL/GL loaders. The NVIDIA container toolkit injects the driver's
# EGL vendor library at run time, so Debian's Mesa vendor stack (~200 MB: LLVM, gallium, z3) that
# libegl1 drags in is removed. --build-arg CPU_RENDER=1 keeps Mesa and adds OSMesa for machines
# without an NVIDIA GPU (run those with -e MUJOCO_GL=osmesa).
ARG CPU_RENDER=0
RUN apt-get update \
 && apt-get install -y --no-install-recommends libegl1 libgl1 libglvnd0 libopengl0 \
    $(if [ "$CPU_RENDER" = "1" ]; then echo libosmesa6; fi) \
 && if [ "$CPU_RENDER" != "1" ]; then \
      dpkg --remove --force-depends libegl-mesa0 libglx-mesa0 libgl1-mesa-dri libgbm1 \
           mesa-libgallium libllvm19 libz3-4 2>/dev/null || true; \
    fi \
 && rm -rf /var/lib/apt/lists/*

# Same versions as the development venv, so recordings are reproducible.
RUN pip install --no-cache-dir mujoco==3.11.0 numpy==2.5.3 Pillow==12.3.0

# GPU build only: GLVND's EGL vendor file for the NVIDIA driver. Older NVIDIA Container Toolkit
# versions inject libEGL_nvidia.so.0 but not this file, and with Mesa's vendor file removed above
# EGL would then find no vendor at all. A toolkit that does mount its own copy simply replaces it.
# A plain COPY cannot be conditional, so it is staged and moved only when CPU_RENDER != 1; the
# :cpu image keeps Mesa as its only EGL vendor.
COPY docker/10_nvidia.json /tmp/10_nvidia.json
RUN if [ "$CPU_RENDER" != "1" ]; then \
      mkdir -p /usr/share/glvnd/egl_vendor.d \
      && mv /tmp/10_nvidia.json /usr/share/glvnd/egl_vendor.d/10_nvidia.json \
      && chmod 0644 /usr/share/glvnd/egl_vendor.d/10_nvidia.json; \
    else \
      rm -f /tmp/10_nvidia.json; \
    fi

# PYOPENGL_PLATFORM is deliberately unset: mujoco sets it from MUJOCO_GL. A fixed value here
# would make -e MUJOCO_GL=osmesa fail, and mujoco then silently drops mujoco.Renderer.
# NVIDIA_DRIVER_CAPABILITIES must contain "graphics": that is what makes the toolkit inject the
# driver's EGL library (compute alone, e.g. Docker Desktop on WSL2, gives CUDA but no EGL).
# XDG_CACHE_HOME: a --user container has HOME=/, which is not writable; Mesa and the NVIDIA
# driver would otherwise warn about (or silently skip) their shader cache there.
ENV MUJOCO_GL=egl \
    NVIDIA_DRIVER_CAPABILITIES=graphics,compute,utility \
    XDG_CACHE_HOME=/tmp/.cache \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Paths mirror the repository: modules resolve files relative to the repo root.
WORKDIR /app
# Only the v1 bimanual model is used (45 MB of the 134 MB upstream checkout).
COPY third_party/openarm_mujoco/LICENSE third_party/openarm_mujoco/LICENSE
COPY third_party/openarm_mujoco/v1/openarm_bimanual.xml third_party/openarm_mujoco/v1/openarm_bimanual.xml
COPY third_party/openarm_mujoco/v1/meshes third_party/openarm_mujoco/v1/meshes
COPY assets/connector assets/connector
COPY configs configs
COPY connector connector
COPY envs envs
COPY controllers controllers
COPY data_pipeline data_pipeline
COPY scripts/collect_random.py scripts/rollout_viewer.py scripts/replay_pilot.py scripts/docker_selfcheck.py scripts/
COPY tests tests

# Collected episodes and the scene cache (data/.scene_cache) live on the host. The directory is
# created here, world-writable with the sticky bit (like /tmp), so that a container started with
# --user $(id -u):$(id -g) can write to an anonymous or fresh named volume (which inherits these
# permissions). A host bind mount keeps the host folder's own owner: create it first (mkdir -p data).
RUN mkdir -p /app/data && chmod 1777 /app/data
VOLUME /app/data

ENTRYPOINT ["python"]
CMD ["scripts/collect_random.py", "--help"]
