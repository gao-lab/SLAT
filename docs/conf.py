# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from docs.utils import _get_thumbnails

copyright = 'Gao Lab@2022'
author = 'Xia Chen-rui'


# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.mathjax',
    'sphinx_autodoc_typehints',
    'sphinx_copybutton',
    'nbsphinx',
    'sphinx.ext.githubpages',
    'myst_parser',
    'sphinx_gallery.load_style',
]

templates_path = ['_templates']
source_suffix = ['.rst', '.md']
master_doc = 'index'
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']
# nbsphinx_thumbnails = {**_get_thumbnails("tutorials")}
nbsphinx_thumbnails = {
    "tutorials/basic_usage": "_static/gallery_thumb/basic_usage.png",
    "tutorials/multi_datasets": "_static/gallery_thumb/multi_datasets.png",
    "tutorials/cross_technology": "_static/gallery_thumb/cross_technology.png",
    "tutorials/times_series": "_static/gallery_thumb/times_series.png",
    "tutorials/pre_match": "_static/gallery_thumb/pre_match.png",
}


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_favicon = '_static/SLAT.ico'
html_static_path = ['_static']
html_css_files = ['custom.css']
html_logo = "_static/SLAT_logo.png"


intersphinx_mapping = dict(
    python=('https://docs.python.org/3/', None),
    numpy=('https://numpy.org/doc/stable/', None),
    scipy=('https://docs.scipy.org/doc/scipy/', None),
    pandas=('https://pandas.pydata.org/pandas-docs/stable/', None),
    sklearn=('https://scikit-learn.org/stable/', None),
    matplotlib=('https://matplotlib.org/stable/', None),
    seaborn=('https://seaborn.pydata.org/', None),
    anndata=('https://anndata.readthedocs.io/en/stable/', None),
    scanpy=('https://scanpy.readthedocs.io/en/stable/', None),
    torch=('https://pytorch.org/docs/stable/', None),
    plotly=('https://plotly.com/python-api-reference/', None),
    scglue=('https://scglue.readthedocs.io/en/latest/',None),
    torch_geometric=('https://pytorch-geometric.readthedocs.io/en/latest/',None)
    )

qualname_overrides = {
    'anndata._core.anndata.AnnData': 'anndata.AnnData',
    'matplotlib.axes._axes.Axes': 'matplotlib.axes.Axes',
    'numpy.random.mtrand.RandomState': 'numpy.random.RandomState',
    'pandas.core.frame.DataFrame': 'pandas.DataFrame',
    'scipy.sparse.base.spmatrix': 'scipy.sparse.spmatrix',
    'seaborn.axisgrid.JointGrid': 'seaborn.JointGrid',
    'torch.device': 'torch.torch.device',
    'torch.nn.modules.module.Module': 'torch.nn.Module'
}