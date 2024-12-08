##############################################################################
# Multi-stage build
FROM ghcr.io/openfaas/of-watchdog:0.9.6 as watchdog
FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04

# Set environment variables
ENV PYTHON_VERSION 3.11.1

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

RUN apt-get update && apt-get install -y \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3.11-distutils \
    curl \
    wget \
    && rm -rf /var/lib/apt/lists/*

RUN   wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
RUN  dpkg -i cuda-keyring_1.1-1_all.deb
RUN  apt-get update
RUN  apt-get -y install cudnn


RUN  apt-get -y install cudnn-cuda-12

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
