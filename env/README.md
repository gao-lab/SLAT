# Env for reproducing
If you want repeat our whole benchmark and evaluation workflow, please configure environment as follow.

## Python env
First, you need to install extra Python packages such as:
- snakemake
- papermill
- jupyter

> **Warning**
> Do **NOT** change install order !

```bash
mamba create -p ./conda python==3.8 -y && conda activate ./conda
mamba install pytorch=1.11.0 torchvision torchaudio cudatoolkit=11.3 -c pytorch -y
mamba install pyg -c pyg -y

mamba install -c conda-forge papermill parse dill jupyter -y 
mamba install -c bioconda -c conda-forge snakemake==7.12.0 tabulate==0.8.10 -y

git clone git@github.com:gao-lab/SLAT.git
cd SLAT
git checkout tags/v0.2.0
pip install -e ".[dev,doc]"
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
install.packages('IRkernel')
renv::restore()
```

## System env
### Singularity
You also need install [`singularity`](https://docs.sylabs.io/guides/3.0/user-guide/index.html), because we use container to ensure the repeatability.