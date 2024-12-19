##############################################################################
# Multi-stage build
FROM ghcr.io/openfaas/of-watchdog:0.9.6 as watchdog
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime


# Set environment variables
ENV PYTHON_VERSION 3.11.1

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC


WORKDIR /workspace


##############################################################################
# Of-watchdog Setup
USER root
COPY --from=watchdog /fwatchdog /usr/bin/fwatchdog
RUN chmod +x /usr/bin/fwatchdog



WORKDIR /workspace

RUN pip install transformers datasets evaluate accelerate flask waitress
COPY backend.py backend.py


# Env Setup
EXPOSE 8080

EXPOSE 5000


ENV cgi_headers="true"
ENV mode="http"
ENV upstream_url="http://127.0.0.1:5000"
ENV fprocess="python3 backend.py"

HEALTHCHECK --interval=5s CMD [ -e /tmp/.lock ] || exit 1
CMD ["fwatchdog"]
