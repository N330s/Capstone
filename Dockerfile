# syntax=docker/dockerfile:1
ARG PYTHON_IMAGE=python:3.11-slim-bookworm
FROM ${PYTHON_IMAGE}

# CUDA libraries come from the PyTorch wheels. The GPU driver stays on the host.
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HOME=/cache/home HF_HOME=/cache/huggingface TORCH_HOME=/cache/torch \
    XDG_CACHE_HOME=/cache/xdg HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
    HF_HUB_DISABLE_TELEMETRY=1 WANDB_MODE=disabled TOKENIZERS_PARALLELISM=false \
    MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MPLBACKEND=Agg \
    NVIDIA_VISIBLE_DEVICES=all NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates ffmpeg git libegl1 libgl1 libglx0 libglfw3 libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
COPY requirements-smolvla.txt requirements-visualization.txt /opt/openarm/
COPY docker/constraints-smolvla.txt /opt/openarm/constraints-smolvla.txt
RUN python -m pip install --no-cache-dir torch==2.10.0 torchvision==0.25.0 \
        --index-url https://download.pytorch.org/whl/cu128 \
    && python -m pip install --no-cache-dir --no-deps torchcodec==0.10.0 \
        --index-url https://download.pytorch.org/whl/cpu \
    && python -m pip install --no-cache-dir -c /opt/openarm/constraints-smolvla.txt \
        -r /opt/openarm/requirements-smolvla.txt -r /opt/openarm/requirements-visualization.txt \
    && python -m pip check \
    && python -m pip freeze --all > /opt/openarm/requirements.resolved.txt

# .dockerignore allowlists code/config/assets only: never datasets or credentials.
COPY . /workspace
COPY docker/entrypoint.py /opt/openarm/entrypoint.py
USER 1000:1000
ENTRYPOINT ["python", "/opt/openarm/entrypoint.py"]
CMD ["python", "scripts/container_preflight.py"]
