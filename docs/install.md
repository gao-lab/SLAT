# Installation guide

## Install from PyPI
```{note}
Installing `scSLAT` within a new [conda environment](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) is recommended.
```
First, create a clean environment and activate it. Then we install `scSLAT`. We recommend you install on a machine equipped CUDA-enabled GPU, `scSLAT` will be 10x faster than running on CPU.

```{warning}
Install in machine with old NVIDIA driver may raise error, please update NVIDIA driver to the latest version.
```

```bash
conda create -n scSLAT python=3.8 -y && conda activate scSLAT

pip install "scSLAT[torch]"
pip install "scSLAT[pyg]"
```

## Accelerate the install
Some dependencies such as `torch-scatter` need to compile from source, which may take a long time. We provide solutions to accelerate the install on different platforms:

### CPU-only
We install cpu version of dependencies from pre-built wheel, which is faster than compiling from source. Then install `scSLAT` from PyPI:

```bash
pip install torch==1.11.0+cpu --extra-index-url https://download.pytorch.org/whl/cpu
pip install https://data.pyg.org/whl/torch-1.11.0%2Bcpu/torch_cluster-1.6.0-cp38-cp38-linux_x86_64.whl
pip install https://data.pyg.org/whl/torch-1.11.0%2Bcpu/torch_scatter-2.0.9-cp38-cp38-linux_x86_64.whl
pip install https://data.pyg.org/whl/torch-1.11.0%2Bcpu/torch_sparse-0.6.15-cp38-cp38-linux_x86_64.whl
pip install torch-geometric
pip install scSLAT
```

### GPU-enabled
```{warning}
Install in machine with old NVIDIA driver may raise error, please update NVIDIA driver to the latest version.
```

PyG Team provides pre-built wheels for specific CUDA version ([here](https://data.pyg.org/whl/)). If your cuda version is in the list, please install the corresponding version torch and pyg dependencies. At last install `scSLAT`. We provide an example for CUDA 10.2:

```bash
pip install torch==1.11.0+cu102 --extra-index-url https://download.pytorch.org/whl/cu102
pip install https://data.pyg.org/whl/torch-1.11.0%2Bcu102/torch_cluster-1.6.0-cp38-cp38-linux_x86_64.whl
pip install https://data.pyg.org/whl/torch-1.11.0%2Bcu102/torch_sparse-0.6.13-cp38-cp38-linux_x86_64.whl
pip install https://data.pyg.org/whl/torch-1.11.0%2Bcu102/torch_cluster-1.6.0-cp38-cp38-linux_x86_64.whl
pip install torch-geometric
pip install scSLAT
```

## Install from Github
You can also install `scSLAT` from Github for development purpose, clone this repo and install:

```bash
conda create -n scSLAT python=3.8 -y && conda activate scSLAT

git clone git@github.com:gao-lab/SLAT.git
cd SLAT
pip install -e ".[torch]"
pip install -e ".[pyg,dev,doc]"
```

## Docker
Dockerfile of `scSLAT` is available at [`env/Dockerfile`](https://github.com/gao-lab/SLAT/blob/main/env/Dockerfile). You can also pull the docker image from [here](https://hub.docker.com/repository/docker/huhansan666666/slat) by :
```
docker pull huhansan666666/slat:latest
```

## Install from Conda (Ongoing)
We plan to provide a conda package of `scSLAT` in the future.