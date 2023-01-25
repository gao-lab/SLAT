# More complex env config
## Benchmark env
If you want repeat our whole benchmark and evaluation workflow, please configure environment as follow, you need to install extra packages such as:
- snakemake
- papermill
- jupyter
- dill
- parse

> NOTE: Do **NOT** change install order !
```
mamba create -p ./conda python==3.8 -y && conda activate ./conda
mamba install pytorch=1.11.0 torchvision torchaudio cudatoolkit=11.3 -c pytorch -y
mamba install pyg -c pyg -y

mamba install -c conda-forge papermill parse dill jupyter -y 
mamba install -c bioconda -c conda-forge snakemake==7.12.0 tabulate==0.8.10 -y

git clone git@github.com:gao-lab/SLAT.git
cd SLAT
pip install -e ".[torch]"
pip install -e ".[pyg,dev,doc]"
```
And then you also should configure `R` environment as below. 

## R env
We strong recommend to compile a **new** R without rather than use environment manage tools (conda, pakrat, renv). We use `R-4.1.3`, but `R` version > 4 may be enough.

> NOTE: Please make sure you have **deactivated** any conda env before using `R`

```
cd resource
wget https://cran.r-project.org/src/base/R-4/R-4.1.3.tar.gz
tar -xzvf R-4.1.3.tar.gz && cd R-4.1.3 && 
  ./configure --without-x --with-cairo --with-libpng --with-libtiff --with-jpeglib --enable-R-shlib --prefix={your path} &&
   make && make install
```

Please install R packages in specific version from `renv.lock` (nedd install `renv` first)
```R
install.packages('renv')
renv::restore()
```
You can manually install following packages:

- Seurat (4.1.1)
- reticulate (any version)
- IRkernel (any version)
- tidyverse (any version)
- data.table (any version)
- yaml (any version)
- ggpubr (any version)


Then register the jupyter kernel for `R` so that we can call `R` in benchmark workflow.
```
IRkernel::installspec(name = 'slat_r', displayname = 'slat_r')
```

Before running snakemake, you should install `IRkernel` again outside of `renv` env, so that `papermill` can call `R` kernel before `renv` env initialization.


## Slurm env
We failed to install `torch_geometric` via conda in slurm cluster. If you meet similar problem please use `pip` instead.
```
mamba create -p ./conda python==3.8 -y && conda activate ./conda
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
pip install pytorch-lightning
pip install torch-scatter -f https://data.pyg.org/whl/torch-1.11.0+cu113.html
pip install torch-sparse -f https://data.pyg.org/whl/torch-1.11.0+cu113.html
pip install torch-cluster -f https://data.pyg.org/whl/torch-1.11.0+cu113.html
pip install torch-geometric

mamba install -c conda-forge papermill parse jupyter dill -y 
mamba install -c bioconda -c conda-forge snakemake==7.12.0 tabulate==0.8.10 -y

git clone git@github.com:gao-lab/SLAT.git
cd SLAT
pip install -e ".[torch]"
pip install -e ".[pyg,dev,doc]"
```