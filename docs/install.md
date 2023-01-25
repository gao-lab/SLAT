# Installation guide

## Install from Github
```{note}
Installing `scSLAT` within a new [conda environment](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) is recommended.
```

Fist, create a clean environment and activate it. Then we install `scSLAT` from Github.

```{note}
We strong recommend you install SLAT with CUDA-enabled GPU, it will 10x faster than running on CPU.
```
```{warning}
Install in machine with old NVIDIA driver may raise error, please update NVIDIA driver to the latest version or use Docker.
```

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

## Install from PyPI (Ongoing)
We plan to provide a PyPI package of `scSLAT` in the future.

## Install from Conda (Ongoing)
We plan to provide a conda package of `scSLAT` in the future.