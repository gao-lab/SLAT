r"""
SLAT (Spatial-Linked Alignment Tool)
"""

try:
    from importlib.metadata import version
except ModuleNotFoundError:
    from pkg_resources import get_distribution
    version = lambda name: get_distribution(name).version

from . import viz
from . import model

name = "scSLAT"
__version__ = version(name)
__author__ = 'Chen-Rui Xia'