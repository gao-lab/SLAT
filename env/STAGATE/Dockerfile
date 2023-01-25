FROM gpuci/miniconda-cuda:11.3-base-ubuntu20.04

RUN conda install -c conda-forge -y mamba python=3.7
RUN mamba install -c conda-forge -y tensorflow-gpu=1.15.0 scanpy python-igraph scikit-misc leidenalg jupyterlab papermill -y
RUN rm /etc/apt/sources.list.d/cuda.list && rm /etc/apt/sources.list.d/nvidia-ml.list
RUN apt update && apt install ssh git -y && apt clean
ENV http_proxy http://xiachenrui:xiachenrui%40GaoLab@202.38.126.18:7777
ENV https_proxy http://xiachenrui:xiachenrui%40GaoLab@202.38.126.18:7777
RUN git clone https://github.com/QIFEIDKN/STAGATE \
    && cd STAGATE \
    && python3 setup.py build \
    && python3 setup.py install
RUN ipython kernel install --name "python3" --user