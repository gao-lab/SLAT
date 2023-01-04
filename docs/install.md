# Installation guide
 
## Install from Github
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

git clone git@github.com:gao-lab/SLAT.git
cd SLAT
pip install -e ".[torch]"
pip install -e ".[pyg,dev,doc]"
```
## Docker
Dockerfile of `scSLAT` is available at [`env/Dockerfile`](https://github.com/gao-lab/SLAT/blob/main/env/Dockerfile). You can also pull the docker image from [here](https://hub.docker.com/repository/docker/huhansan666666/slat) by :
```
docker push huhansan666666/slat:latest
```

## Install from Conda (Ongoing)
We plan to provide a conda package of `scSLAT` in the future.