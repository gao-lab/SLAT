FROM gpuci/miniconda-cuda:11.3-base-ubuntu20.04
RUN conda install -c conda-forge -y mamba
RUN mamba install -c pytorch pytorch=1.11.0 torchvision torchaudio cudatoolkit=11.3 -y
RUN mamba install -c pyg pyg -y
RUN mamba install -c conda-forge scanpy python-igraph leidenalg flit jupyterlab papermill parse dill -y 

ENV http_proxy http://xiachenrui:xiachenrui%40GaoLab@202.38.126.18:7777
ENV https_proxy http://xiachenrui:xiachenrui%40GaoLab@202.38.126.18:7777
RUN git clone https://github.com/QIFEIDKN/STAGATE_pyG \
    && cd STAGATE_pyG \
    && python setup.py build \
    && python setup.py install

RUN pip install scikit-misc