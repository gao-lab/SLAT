# Installation guide

## Install from PyPI
```{note}
Installing `scSLAT` within a new [conda environment](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) is recommended.
```


First, create a clean environment to install the `scSLAT`. We recommend you install on a machine equipped CUDA-enabled GPU, `scSLAT` will be 5x-10x faster than running on CPU.

```bash
conda create -n scSLAT python=3.11 -y && conda activate scSLAT
pip install scSLAT
install_pyg_dependencies
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
You can pull the docker image from [here](https://hub.docker.com/repository/docker/huhansan666666/slat) or build it from [`Dockerfile`](https://github.com/gao-lab/SLAT/blob/main/Dockerfile):
```
docker pull huhansan666666/slat:latest
```