# Frigate Anomaly Detector (DINOv2 + OpenVINO) — container image
#
# Builds the app plus everything it needs at runtime:
#   * ffmpeg            -> RTSP frame grabs
#   * Intel GPU userspace (OpenCL + Level Zero) -> lets the OpenVINO GPU plugin
#     see /dev/dri from inside the container
#   * Python deps       -> CPU torch (inference runs on the iGPU, not CUDA)
#
# The image carries NO configuration: config.json / baselines / anomalies live in
# a volume (DINO_DATA_DIR, default /data) so camera names, IPs and RTSP
# credentials never end up baked into a layer.
#
# Ubuntu 26.04 LTS (resolute) on purpose: it ships the current Intel graphics
# compute stack — intel-opencl-icd / libze-intel-gpu1 26.05, i.e. the same
# userspace this project is developed against. The GPU plugin needs Level Zero,
# and no Debian release carries that userspace in its own archive (bookworm's
# 22.43 cannot even initialise a Meteor Lake iGPU, and OpenVINO then reports
# CPU-only without an error). 26.04 also gives us Python 3.14, which torch
# (cpu), openvino, nncf and scikit-learn all publish wheels for.
FROM ubuntu:26.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DINO_DATA_DIR=/data

# ffmpeg is required for RTSP sources. The Intel compute runtime is what lets
# OpenVINO's GPU plugin talk to /dev/dri; Level Zero (libze1 + libze-intel-gpu1)
# is the path modern OpenVINO uses.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        ffmpeg \
        python3 \
        python3-venv \
        intel-opencl-icd \
        libze-intel-gpu1 \
        libze1 \
    && rm -rf /var/lib/apt/lists/*

# Ubuntu marks system Python as externally managed; a venv keeps pip happy and
# keeps the app's deps away from the distro packages.
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}"

WORKDIR /app

# Dependency layer first, so code edits don't reinstall the world. CPU wheels on
# purpose: inference goes through OpenVINO on the iGPU, so the multi-GB CUDA
# build of torch would be dead weight.
COPY requirements.txt ./
RUN pip install --upgrade pip \
    && pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu \
    && pip install -r requirements.txt

# Application files only — read-only at runtime.
COPY server.py ./
COPY templates ./templates
COPY static ./static

# Mutable state: config.json, baselines/, anomalies/, profiles/, model_cache/,
# logs/, rtsp_schedule.json. Mount a host directory here to keep it all.
RUN mkdir -p /data
VOLUME ["/data"]

EXPOSE 8080
CMD ["python", "server.py"]
