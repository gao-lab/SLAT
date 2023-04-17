FROM nvidia/cuda:11.7.0-cudnn8-runtime-ubuntu20.04

# Update and install dependencies
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y wget bzip2 ca-certificates curl git

# Download and install Miniconda
ENV CONDA_AUTO_UPDATE_CONDA=false
RUN wget -q https://repo.anaconda.com/miniconda/Miniconda3-py38_4.10.3-Linux-x86_64.sh -O ~/miniconda.sh \
    && /bin/bash ~/miniconda.sh -b -p /opt/conda \
    && rm ~/miniconda.sh \
    && /opt/conda/bin/conda clean -tipsy \
    && ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh \
    && echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc \
    && echo "conda activate base" >> ~/.bashrc
ENV PATH=/opt/conda/bin:$PATH

# Install scSLAT
RUN conda install -c conda-forge -y mamba
RUN mamba install -c conda-forge -c bioconda jupyterlab snakemake==7.12.0 tabulate==0.8.10 -y

RUN pip3 install --upgrade pip
RUN git clone https://github.com/gao-lab/SLAT \
    && cd SLAT && pip install -e ".[dev, docs]"  \
    && pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.0.0+cu117.html
