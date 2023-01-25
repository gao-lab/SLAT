FROM gpuci/miniconda-cuda:11.3-base-ubuntu20.04

RUN conda install -c conda-forge -y mamba
RUN mamba install -c pytorch pytorch=1.11.0 torchvision torchaudio cudatoolkit=11.3 -y
RUN mamba install -c pyg pyg -y
RUN mamba install -c conda-forge scanpy python-igraph leidenalg flit jupyterlab papermill parse dill -y 
RUN mamba install -c conda-forge -c bioconda snakemake==7.12.0 tabulate==0.8.10 -y

RUN pip3 install --upgrade pip
RUN git clone https://github.com/gao-lab/SLAT && cd SLAT && pip install -e ".[torch]" && pip install -e ".[pyg, dev, doc]"
