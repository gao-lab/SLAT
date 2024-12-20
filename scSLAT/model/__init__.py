r"""
Core functions of SLAT model
"""
from .loaddata import load_anndata, load_anndatas  # noqa
from .prematch import icp  # noqa
from .preprocess import Cal_Spatial_Net, scanpy_workflow  # noqa
from .utils import * # noqa