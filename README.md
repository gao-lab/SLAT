[![stars-badge](https://img.shields.io/github/stars/gao-lab/SLAT?logo=GitHub&color=yellow)](https://github.com/gao-lab/SLAT/stargazers)
[![dev-badge](https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/xiachenrui/bc835db052fde5bd731a09270b42006c/raw/slat_version.json)](https://gist.github.com/xiachenrui/bc835db052fde5bd731a09270b42006c)
[![build-badge](https://github.com/gao-lab/SLAT/actions/workflows/build.yml/badge.svg)](https://github.com/gao-lab/SLAT/actions/workflows/build.yml)
[![license-badge](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![docs-badge](https://readthedocs.org/projects/slat/badge/?version=latest)](https://slat.readthedocs.io/en/latest/?badge=latest)
[![pypi-badge](https://img.shields.io/pypi/v/scslat)](https://pypi.org/project/scslat)
<!-- [![conda-badge](https://anaconda.org/bioconda/<name>/badges/version.svg)](https://anaconda.org/bioconda/<name>) -->

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

### Docker
Dockerfile of `scSLAT` is available at [`env/Dockerfile`](env/Dockerfile). You can also pull the docker image directly from [here](https://hub.docker.com/repository/docker/huhansan666666/slat) by:

``` bash
docker pull huhansan666666/slat:latest
```

### PyPI
> **Note**
> Installing `scSLAT` within a new [conda environment](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) is recommended.

First, we create a clean environment and install `scSLAT` from PyPI. Then we also need install dependencies for `pyg` manually. We default install with CUDA 11.7. Please refer [here](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html#quick-start) for CPU version or different CUDA versions.

> **Warning**
> old NVIDIA driver may raise error, please update NVIDIA driver to the latest version.

```bash
conda create -n scSLAT python=3.8 -y && conda activate scSLAT
pip install scSLAT
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.0.0+cu117.html
```

### Development version
For development purpose, clone this repo and install:

```bash
git clone git@github.com:gao-lab/SLAT.git
cd SLAT
pip install -e ".[dev,docs]"
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.0.0+cu117.html
```

### Conda (Ongoing)
We plan to provide a conda package of `scSLAT` in the near future.


## Reproduce manuscript results
1. Please follow the [`env/README.md`](env/README.md) to install all dependencies. Please checkout the repository to v0.2.1 before install `scSLAT`.
2. Download and pre-process data follow the [`data/README.md`](data/README.md).
3. Whole benchmark and evaluation procedure can be found in [`/benchmark`](benchmark/README.md) and [`/evaluation`](evaluation/README.md), respectively.
4. Every case study is recorded in the [`/case`](case/README.md) directory in the form of jupyter notebook.
