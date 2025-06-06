FROM --platform=${TARGETPLATFORM:-linux/amd64}  ghcr.io/openfaas/of-watchdog:0.9.6 as watchdog
FROM nvidia/cuda:10.2-cudnn8-runtime-ubuntu18.04
RUN apt-get update && \ 
    apt-get install -y wget python3.7-dev python3.7-distutils g++ libsndfile1 -y
RUN wget https://bootstrap.pypa.io/get-pip.py && \
    python3.7 get-pip.py && \
    rm get-pip.py
COPY requirements.txt requirements.txt
RUN python3.7 -m pip install -r requirements.txt
# Just have to maintain the softlink manually
RUN cp -s /usr/lib/x86_64-linux-gnu/libcudnn.so.8 /usr/lib/x86_64-linux-gnu/libcudnn.so && \
    cp -s /usr/lib/x86_64-linux-gnu/libcublas.so.10 /usr/lib/x86_64-linux-gnu/libcublas.so && \
    cp -s /usr/local/cuda/lib64/libcufft.so.10 /usr/local/cuda/lib64/libcufft.so && \
    echo "export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu" >> /root/.bashrc

# Prepare for Dataset (for more details: paddlespeech/resource/pretrained_models.py)
RUN wget https://paddlespeech.bj.bcebos.com/s2t/wenetspeech/asr1_conformer_wenetspeech_ckpt_0.1.1.model.tar.gz -P /root/.paddlespeech/models/conformer_wenetspeech-zh-16k/1.0/ 

# Of-watchdog Setup
USER root
COPY --from=watchdog /fwatchdog /usr/bin/fwatchdog
RUN chmod +x /usr/bin/fwatchdog

# Env Setup
WORKDIR /home/app
COPY application.yaml application.yaml
ADD probe probe
EXPOSE 8080

ENV cgi_headers="true"
ENV mode="http"
ENV upstream_url="http://127.0.0.1:8090"
ENV fprocess="paddlespeech_server start --config_file /home/app/application.yaml"

HEALTHCHECK --interval=5s CMD [ -e /tmp/.lock ] || exit 1
CMD ["fwatchdog"]
