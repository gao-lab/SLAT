FROM gpuci/miniconda-cuda:11.3-base-ubuntu20.04

RUN conda install -c conda-forge -y mamba
RUN mamba install pytorch=1.11.0 torchvision torchaudio cudatoolkit=11.3 -c pytorch -y
RUN mamba install -c conda-forge pytorch-lightning -y
RUN mamba install pyg -c pyg -y
RUN mamba install -c conda-forge scanpy python-igraph leidenalg -y

RUN mamba install -c conda-forge flit jupyterlab papermill parse dill -y 
RUN mamba install -c conda-forge -c bioconda  snakemake -y 

RUN conda clean -a -y
RUN pip install scSLAT
