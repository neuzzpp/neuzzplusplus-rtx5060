FROM nvidia/cuda:12.8.0-cudnn-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /workspace

# System deps (добавили zlib1g-dev чтобы собирать/линковать zlib таргеты без apt-get в runtime)
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-venv python3-pip python3-dev \
    git make build-essential clang pkg-config \
    ca-certificates curl \
    zlib1g-dev \
 && rm -rf /var/lib/apt/lists/*

# Create venv (PEP 668 safe) and install tools inside it
ENV VENV_PATH=/opt/venv
RUN python3 -m venv $VENV_PATH
ENV PATH="$VENV_PATH/bin:$PATH"

RUN pip install --no-cache-dir -U pip setuptools wheel
RUN pip install --no-cache-dir poetry

# TensorFlow nightly (лучше шанс на RTX 50xx / Blackwell)
RUN pip install --no-cache-dir -U --pre tf-nightly
