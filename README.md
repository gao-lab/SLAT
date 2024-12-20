[![stars-badge](https://img.shields.io/github/stars/gao-lab/SLAT?logo=GitHub&color=yellow)](https://github.com/gao-lab/SLAT/stargazers)
[![pypi-badge](https://img.shields.io/pypi/v/scslat)](https://pypi.org/project/scslat)
[![Downloads](https://static.pepy.tech/badge/scSLAT)](https://pepy.tech/project/scSLAT)
[![build-badge](https://github.com/gao-lab/SLAT/actions/workflows/build.yml/badge.svg)](https://github.com/gao-lab/SLAT/actions/workflows/build.yml)
[![docs-badge](https://readthedocs.org/projects/slat/badge/?version=latest)](https://slat.readthedocs.io/en/latest/?badge=latest)
[![license-badge](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)


# scSLAT: single cell spatial alignment tools
**scSLAT** package implements the **SLAT** (**S**patial **L**inked **A**lignment **T**ool) model to align single cell spatial omics data.

![Model architecture](docs/_static/Model.png)

## Directory structure

```
.
├── scSLAT/                  # Main Python package
├── env/                     # Extra environment
├── data/                    # Data files
├── evaluation/              # SLAT evaluation pipeline
├── benchmark/               # Benchmark pipeline
├── case/                    # Case studies in paper
├── docs/                    # Documentation files
├── resource/                # Other useful resource
├── pyproject.toml           # Python package metadata
└── README.md
```

## Tutorial
Tutorial of `scSLAT` is [here](https://slat.readthedocs.io/en/latest/tutorials.html), if you have any question please open an issue on github

<img src='docs/_static/imgalignment.gif' width='400'>


## Installation

### PyPI
> **Note**
> Installing `scSLAT` within a new [conda environment](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) is recommended.

> **Warning**
> Old NVIDIA driver may raise error.

First, we create a clean environment and install `scSLAT` from PyPI. Then we also need install dependencies for `pyg` manually via `install_pyg_dependencies`.

```bash
conda create -n scSLAT python=3.11 -y && conda activate scSLAT
pip install scSLAT
install_pyg_dependencies
```

### Docker
You can pull the docker image directly from [Docker Hub](https://hub.docker.com/repository/docker/huhansan666666/slat) or refer to the [`Dockerfile`](Dockerfile) to build it.

``` bash
docker pull huhansan666666/slat:latest
```

### Development version
For development purpose, clone this repo and install:

```bash
git clone git@github.com:gao-lab/SLAT.git && cd SLAT
pip install -e ".[dev,docs]"
install_pyg_dependencies
```

### Conda (Ongoing)
We plan to provide a conda package of `scSLAT` in the near future.


## Reproduce manuscript results
1. Please follow the [`env/README.md`](env/README.md) to install all dependencies. Please checkout the repository to v0.2.1 before install `scSLAT`.
2. Download and pre-process data follow the [`data/README.md`](data/README.md).
3. Whole benchmark and evaluation procedure can be found in [`/benchmark`](benchmark/README.md) and [`/evaluation`](evaluation/README.md), respectively.
4. Every case study is recorded in the [`/case`](case/README.md) directory in the form of jupyter notebook.
