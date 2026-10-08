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

ENV MUJOCO_GL=egl \
    PYOPENGL_PLATFORM=egl \
    NVIDIA_DRIVER_CAPABILITIES=graphics,compute,utility \
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
COPY scripts/collect_random.py scripts/rollout_viewer.py scripts/replay_pilot.py scripts/
COPY tests tests

# Collected episodes and the scene cache (data/.scene_cache) live on the host.
VOLUME /app/data

ENTRYPOINT ["python"]
CMD ["scripts/collect_random.py", "--help"]
