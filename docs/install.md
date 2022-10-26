# Installation guide
 
## Install from PyPI
```{note}
Installing `scSLAT` within a new [conda environment](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) is recommended.
```

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
## Docker container
Dockerfile of `scSLAT` is available at [`env/Dockerfile`](https://github.com/gao-lab/SLAT/blob/main/env/Dockerfile)

## Install from Conda (Ongoing)
We plan to provide a conda package of `scSLAT` in the future.