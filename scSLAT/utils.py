r"""
Miscellaneous utilities
"""
import random
from subprocess import run
from typing import Optional, Union

import numpy as np
import pynvml
import torch
from anndata import AnnData


def norm_to_raw(
    adata: AnnData,
    library_size: Optional[Union[str, np.ndarray]] = "total_counts",
    check_size: Optional[int] = 100,
) -> AnnData:
    r"""
    Convert normalized adata.X to raw counts

    Parameters
    ----------
    adata
        adata to be convert
    library_size
        raw library size of every cells, can be a key of `adata.obs` or a array
    check_size
        check the head `[0:check_size]` row and column to judge if adata normed

    Note
    ----------
    Adata must follow scanpy official norm step
    """
    check_chunk = adata.X[0:check_size, 0:check_size].todense()
    assert not all(isinstance(x, int) for x in check_chunk)

    from scipy import sparse

    scale_size = np.array(adata.X.expm1().sum(axis=1).round()).flatten()
    if isinstance(library_size, str):
        scale_factor = np.array(adata.obs[library_size]) / scale_size
    elif isinstance(library_size, np.ndarray):
        scale_factor = library_size / scale_size
    else:
        try:
            scale_factor = np.array(library_size) / scale_size
        except ValueError:
            raise ValueError("Invalid `library_size`")
    scale_factor.resize((scale_factor.shape[0], 1))
    raw_count = sparse.csr_matrix.multiply(
        sparse.csr_matrix(adata.X).expm1(), sparse.csr_matrix(scale_factor)
    )
    raw_count = sparse.csr_matrix(np.round(raw_count))
    adata.X = raw_count
    # adata.layers['counts'] = raw_count
    return adata


def get_free_gpu() -> int:
    r"""
    Get index of GPU with least memory usage

    Ref
    ----------
    https://stackoverflow.com/questions/58216000/get-total-amount-of-free-gpu-memory-and-available-using-pytorch
    """
    index = 0
    if torch.cuda.is_available():
        pynvml.nvmlInit()
        max = 0
        for i in range(torch.cuda.device_count()):
            h = pynvml.nvmlDeviceGetHandleByIndex(i)
            info = pynvml.nvmlDeviceGetMemoryInfo(h)
            index = i if info.free > max else index
            max = info.free if info.free > max else max
    return index


def global_seed(seed: int) -> None:
    r"""
    Set seed

    Parameters
    ----------
    seed
        int
    """
    if seed > 2**32 - 1 or seed < 0:
        seed = 0

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    print(f"Global seed set to {seed}.")


def install_pyg_dep(torch_version: str = None, cuda_version: str = None) -> None:
    r"""
    Automatically install PyG dependencies

    Parameters
    ----------
    torch_version
        torch version, e.g. 2.2.1
    cuda_version
        cuda version, e.g. 12.1
    """
    if torch_version is None:
        torch_version = torch.__version__
        torch_version = torch_version.split("+")[0]

    if cuda_version is None:
        cuda_version = torch.version.cuda

    if torch_version < "2.0":
        raise ValueError(f"PyG only support torch>=2.0, but get {torch_version}")
    elif "2.0" <= torch_version < "2.1":
        torch_version = "2.0.0"
    elif "2.1" <= torch_version < "2.2":
        torch_version = "2.1.0"
    elif "2.2" <= torch_version < "2.3":
        torch_version = "2.2.0"
    elif "2.3" <= torch_version < "2.4":
        torch_version = "2.3.0"
    else:
        raise ValueError(f"Automatic install only support torch<=2.3, but get {torch_version}")

    if "cu" in cuda_version and not torch.cuda.is_available():
        print("CUDA is not available, try install CPU version, but may raise error.")
        cuda_version = "cpu"
    elif cuda_version >= "12.1":
        cuda_version = "cu121"
    elif "11.8" <= cuda_version < "12.1":
        cuda_version = "cu118"
    elif "11.7" <= cuda_version < "11.8":
        cuda_version = "cu117"
    else:
        raise ValueError(f"PyG only support cuda>=11.7, but get {cuda_version}")

    url = "https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html"
    if torch_version in ["2.2.0", "2.1.0"] and cuda_version == "cu117":
        raise ValueError(
            f"PyG not support torch-{torch_version} with cuda-11.7, please check {url}"
        )
    if torch_version == "2.0.0" and cuda_version == "cu121":
        raise ValueError(f"PyG not support torch-2.0.* with cuda-12.1, please check {url}")

    print(f"Installing PyG dependencies for torch-{torch_version} and cuda-{cuda_version}")
    cmd = f"pip --no-cache-dir install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-{torch_version}+{cuda_version}.html"
    run(cmd, shell=True)
