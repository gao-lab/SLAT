[![stars-badge](https://img.shields.io/github/stars/gao-lab/SLAT?logo=GitHub&color=yellow)](https://github.com/gao-lab/SLAT/stargazers)
[![dev-badge](https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/xiachenrui/bc835db052fde5bd731a09270b42006c/raw/version.json)](https://gist.github.com/xiachenrui/bc835db052fde5bd731a09270b42006c)
[![build-badge](https://github.com/gao-lab/SLAT/actions/workflows/build.yml/badge.svg)](https://github.com/gao-lab/SLAT/actions/workflows/build.yml)
[![license-badge](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
<!-- [![pypi-badge](https://img.shields.io/pypi/v/<name>)](https://pypi.org/project/<name>) -->
<!-- [![conda-badge](https://anaconda.org/bioconda/<name>/badges/version.svg)](https://anaconda.org/bioconda/<name>) -->
<!-- [![docs-badge](https://readthedocs.org/projects/<name>/badge/?version=latest)](https://<name>.readthedocs.io/en/latest/?badge=latest) -->

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
Tutorial of `scSLAT` is [here](TBD), if you have any question please open an issue on github

## Installation
### Install from PyPI
> Installing `scSLAT` within a new [conda environment](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) is recommended.

Fist we create a clean environment and activate it:
```bash
conda create -n scSLAT python=3.8 -y && conda activate scSLAT
```
Then we install `scSLAT` from PyPI.
If you have a CUDA-enabled GPU, we strong recommend you install SLAT with GPU support, it will 5x faster than CPU. You should install [PyG](https://pytorch-geometric.readthedocs.io/en/latest/index.html) first because it can not correctly install from PyPI.

```bash
conda install pytorch=1.11.0 torchvision torchaudio cudatoolkit=11.3 -c pytorch -y
conda install pyg -c pyg -y
pip install scSLAT
```
However, if you only want to run on CPU, you can directly install CPU version:
```bash
pip install scSLAT[cpu]
```
### Docker container
Dockerfile of `scSLAT` is available at [`env/Dockerfile`](env/Dockerfile)

### Install from Conda (Ongoing)
We plan to provide a conda package of `scSLAT` in the near future.

### Development
For development purpose, clone this repo and run:
```bash
pip install -e ".[dev]"
```

## Reproduce results
1. Please follow the [`env/README.md`](env/README.md) to install all dependencies. Please checkout the repository to v0.1.0 before install `scSLAT`:

```
git checkout tags/v0.1.0
pip install -e '.'
```

2. Download files via links in [`data/README.md`](data/README.md)

2. Whole benchmark and evaluation procedure can be found in `/benchmark` and `/evaluation`, respectively.

3. Every case study is recorded in the `/case` directory in the form of jupyter notebook.

