# One definition for both roles. docker-compose.yml builds it twice:
#   FLWR_ROLE=superlink  -> the server container (SuperLink + ServerApp)
#   FLWR_ROLE=supernode  -> one container per hospital (SuperNode + ClientApp)
#
# Ubuntu (glibc) variant on purpose: the default Alpine tag is musl-based, which has no
# PyTorch wheels. Flower runs the ServerApp/ClientApp as a subprocess of these processes
# (isolation mode "subprocess"), so the app's dependencies must be installed here.
# The app code itself is delivered by `flwr run`; data and config are mounted, not baked in.
ARG FLWR_VERSION=1.42.0
ARG FLWR_ROLE=supernode
FROM flwr/${FLWR_ROLE}:${FLWR_VERSION}-py3.13-ubuntu24.04

USER root
COPY docker/requirements.txt /tmp/requirements.txt
# CPU-only PyTorch from PyTorch's own index: the PyPI wheel for Linux/arm64 drags in the whole
# CUDA toolkit (several GB), which is useless here. Same version as the project's requirements.txt.
RUN pip install --no-cache-dir torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r /tmp/requirements.txt && rm /tmp/requirements.txt
USER app

# src.data_loader reads data and config below FL_ROOT (see the mounts in docker-compose.yml)
ENV FL_ROOT=/app
