r"""
SLAT (Spatial-Linked Alignment Tool)
"""

from importlib.metadata import version

from . import model  # noqa
from . import viz  # noqa

name = "scSLAT"
__version__ = version(name)
__author__ = "Chen-Rui Xia"
