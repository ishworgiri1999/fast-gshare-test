# FaaS watchdog image
FROM --platform=${TARGETPLATFORM:-linux/amd64}  ghcr.io/openfaas/of-watchdog:0.9.6 as watchdog
# Compiler image




FROM --platform=${TARGETPLATFORM:-linux/amd64} pytorch/pytorch:2.1.0-cuda12.1-cudnn8-devel as compiler

RUN apt-get update 

RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub \
 && apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub
RUN apt-get update -y
RUN apt-get install -y libsndfile1
RUN apt-get install -y git
RUN apt-get install -y sox
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y cmake
RUN apt-get install -y build-essential
RUN rm -rf /var/lib/apt/lists/*
COPY patch/setup.py /setup.py
ENV CUDA_VERSION=12.1
ENV COMMIT_SHA=2aca0c546302228917bcfa3009b6186b64501a7b
RUN git clone https://github.com/b-flo/warp-transducer deps/warp-transducer
RUN cd deps/warp-transducer && git checkout $COMMIT_SHA

RUN cd deps/warp-transducer && mkdir -p build
RUN cd deps/warp-transducer/build && \
    cmake -DCMAKE_POLICY_VERSION_MINIMUM=3.5 .. && \
    make VERBOSE=1
RUN cd deps/warp-transducer/build && \
    export CUDA_HOME="/usr/local/cuda" && \
    export WARP_RNNT_PATH=`pwd` && \
    export CUDA_TOOLKIT_ROOT_DIR=$CUDA_HOME && \
    export LD_LIBRARY_PATH="$CUDA_HOME/extras/CUPTI/lib64:$LD_LIBRARY_PATH" && \
    export LIBRARY_PATH=$CUDA_HOME/lib64:$LIBRARY_PATH && \
    export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH && \
    export CFLAGS="-std=c++17 -I$CUDA_HOME/include $CFLAGS" && \
    export CXXFLAGS="-std=c++17 -I$CUDA_HOME/include $CXXFLAGS" && \
    cd ../pytorch_binding && \
    cp /setup.py ./setup.py && \
    python3 setup.py install --user

# Base Image
FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime
RUN apt-get update && apt-get install -y libsndfile1 git sox && rm -rf /var/lib/apt/lists/*
WORKDIR /workspace
COPY --from=compiler /root/.local /root/.local 
COPY requirements.txt .
RUN pip install --disable-pip-version-check -U -r requirements.txt

## Prepare src code 
COPY . .
RUN git clone --recursive https://github.com/mlcommons/inference 

RUN cd /workspace && \
    cd inference && \
    git checkout r2.1 && \
    cd speech_recognition/rnnt/pytorch && \
    git apply /workspace/patch/*.patch && \ 
    cp /workspace/server/* ./

## Setup FaaS components
COPY --from=watchdog /fwatchdog /usr/bin/fwatchdog
RUN chmod +x /usr/bin/fwatchdog
ENV mode="http"
ENV upstream_url="http://127.0.0.1:5000"
ENV fprocess="python /workspace/inference/speech_recognition/rnnt/pytorch/index.py"
CMD ["fwatchdog"]
