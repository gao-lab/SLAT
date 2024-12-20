FROM nvcr.io/nvidia/rapidsai/base:24.04-cuda11.8-py3.11

# Update and install dependencies
USER root
RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install gcc g++ git -y \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Copy package and install
USER rapids
WORKDIR /app
COPY ./scSLAT ./scSLAT
COPY ./README.md ./README.md
COPY ./pyproject.toml ./pyproject.toml
RUN pip --no-cache-dir install -e "." && install_pyg_dependencies && rm -rf /tmp/*
