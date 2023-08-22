# Env for reproducing
If you want repeat our whole benchmark and evaluation workflow, please configure environment as follow.

## Python env
You can build the python env for snakemake workflow by `conda` or `mamba`. We recommend to use `mamba` to speed up the installation process.
- snakemake

```bash
mamba create -p ./conda python==3.8 -y && conda activate ./conda
mamba install -c bioconda -c conda-forge snakemake==7.12.0 tabulate==0.8.10 pandoc -y

git clone git@github.com:gao-lab/SLAT.git
cd SLAT
pip install -e ".[dev,docs]"
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.0.0+cu117.html
```

## R env
### STEP 1: Install R
You also should configure `R` environment. We strong recommend to compile a **new** `R-4.1.3` rather than install R in conda.

> **Warning**
> Please make sure you have **deactivated** any conda env before using `R`

```bash
cd SLAT/resource
wget https://cran.r-project.org/src/base/R-4/R-4.1.3.tar.gz
tar -xzvf R-4.1.3.tar.gz && cd R-4.1.3 && 
  ./configure --without-x --with-cairo --with-libpng --with-libtiff --with-jpeglib --enable-R-shlib --prefix={YOUR_PATH} &&
   make && make install
```

### STEP 2: Register R kernel
Then register the jupyter kernel for `R` so snakemake can call `R` in benchmark workflow.
```R
install.packages('IRkernel')
IRkernel::installspec(name = 'slat_r', displayname = 'slat_r')
```

### STEP 3: Install R packages
At last, please install all R packages we used from `renv.lock` (see [`renv`](https://rstudio.github.io/renv/articles/renv.html)).
```R
install.packages('renv')
install.packages('IRkernel') # install IRkernel again inside renv env
renv::restore()
```

## System env
### Singularity
You also need install [`singularity`](https://docs.sylabs.io/guides/3.0/user-guide/index.html), because we use container to ensure the repeatability of benchmark results.