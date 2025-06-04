##############################################################################
# Multi-stage build
FROM ghcr.io/openfaas/of-watchdog:0.9.6 as watchdog
FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04

# Set environment variables
ENV PYTHON_VERSION=3.11.1
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

# Install system packages with better error handling
RUN apt-get update && apt-get install -y \
    --no-install-recommends \
    software-properties-common \
    ca-certificates \
    gnupg \
    lsb-release \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Add deadsnakes PPA and install Python 3.11
RUN add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y \
    --no-install-recommends \
    python3.11 \
    python3.11-dev \
    python3.11-distutils \
    python3.11-venv \
    curl \
    wget \
    build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install CUDA packages separately to avoid conflicts
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb \
    && dpkg -i cuda-keyring_1.1-1_all.deb \
    && rm cuda-keyring_1.1-1_all.deb \
    && apt-get update \
    && apt-get install -y \
    --no-install-recommends \
    cudnn \
    cudnn-cuda-12 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install pip properly
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11
RUN python3.11 -m pip install --no-cache-dir --upgrade pip setuptools wheel

WORKDIR /workspace

##############################################################################
# Of-watchdog Setup
USER root
COPY --from=watchdog /fwatchdog /usr/bin/fwatchdog
RUN chmod +x /usr/bin/fwatchdog

WORKDIR /workspace
COPY backend_onnx.py backend_onnx.py
COPY requirements.txt requirements.txt 

# Function Installation
# Force install without dependencies first
RUN python3.11 -m pip install --no-cache-dir --ignore-installed blinker
RUN python3.11 -m pip install --no-cache-dir -r requirements.txt

# Env Setup
EXPOSE 8080
EXPOSE 5000

ENV cgi_headers="true"
ENV mode="http"
ENV upstream_url="http://127.0.0.1:5000"
ENV fprocess="python3.11 backend_onnx.py"

HEALTHCHECK --interval=5s CMD [ -e /tmp/.lock ] || exit 1
CMD ["fwatchdog"]
