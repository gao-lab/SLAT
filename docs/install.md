# Installation guide

## Install from PyPI
```{note}
Installing `scSLAT` within a new [conda environment](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) is recommended.
```
First, create a clean environment and activate it. Then we install `scSLAT`. We give two examples on GPU and CPU respectively. We recommend you install on a machine equipped CUDA-enabled GPU, `scSLAT` will be 5x-10x faster than running on CPU.

### CUDA-enabled

```{warning}
Install in machine with old NVIDIA driver may raise error, please update NVIDIA driver to the latest version.
```

PyG Team provides pre-built wheels for specific CUDA version ([here](https://data.pyg.org/whl/)). If your CUDA version is in the list, please install the corresponding version torch and pyg dependencies. At last install `scSLAT`. We provide an example for CUDA 11.7:

```bash
conda create -n scSLAT python=3.8 -y && conda activate scSLAT
pip install scSLAT
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.0.0+cu117.html
```

### CPU-only

```bash
conda create -n scSLAT python=3.8 -y && conda activate scSLAT

pip install scSLAT
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.0.0+cpu.html

```


## Install from Github
You can also install latest version of `scSLAT` from Github, clone the repo and install:

```bash
conda create -n scSLAT python=3.8 -y && conda activate scSLAT

git clone git@github.com:gao-lab/SLAT.git
cd SLAT
pip install -e ".[dev, docs]"
```

## Docker
Dockerfile of `scSLAT` is available at [`env/Dockerfile`](https://github.com/gao-lab/SLAT/blob/main/env/Dockerfile). You can also pull the docker image from [here](https://hub.docker.com/repository/docker/huhansan666666/slat) by :
```
docker pull huhansan666666/slat:0.2.1
```

## Install from Conda (Ongoing)
We plan to provide a conda package of `scSLAT` in the future.