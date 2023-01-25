# Installation guide

## Install from PyPI
```{note}
Installing `scSLAT` within a new [conda environment](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) is recommended.
```
First, create a clean environment and activate it. Then we install `scSLAT`. We recommend you install on a machine equipped CUDA-enabled GPU, `scSLAT` will be 10x faster than running on CPU.

```{warning}
Install in machine with old NVIDIA driver may raise error, please update NVIDIA driver to the latest version or use Docker.
```

```bash
conda create -n scSLAT python=3.8 -y && conda activate scSLAT

git clone git@github.com:gao-lab/SLAT.git
cd SLAT
pip install "scSLAT[torch]"
pip install "scSLAT[pyg]"
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