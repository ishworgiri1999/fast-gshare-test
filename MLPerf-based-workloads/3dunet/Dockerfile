# FaaS watchdog image
FROM --platform=${TARGETPLATFORM:-linux/amd64}  ghcr.io/openfaas/of-watchdog:0.9.6 as watchdog
# Base Image
FROM tensorflow/tensorflow:2.2.0-gpu 
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub \
 && apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub
RUN apt-get update && apt-get install -y git

WORKDIR /workspace
COPY requirements.txt .
RUN pip3 install --disable-pip-version-check -U -r requirements.txt

## Prepare src code 
WORKDIR /workspace
COPY server server
RUN git clone --recursive https://github.com/mlcommons/inference \
    && cd inference/vision/medical_imaging/3d-unet-kits19 \
    && git checkout r2.1 \
    && cp /workspace/server/* ./

## Setup FaaS components
WORKDIR /workspace/inference/vision/medical_imaging/3d-unet-kits19
COPY --from=watchdog /fwatchdog /usr/bin/fwatchdog
RUN chmod +x /usr/bin/fwatchdog
ENV mode="http"
ENV upstream_url="http://127.0.0.1:5000"
ENV fprocess="python3 index.py"
CMD ["fwatchdog"]
