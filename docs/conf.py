import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

copyright = 'Gao Lab@2024'
author = 'Chen-rui Xia'


extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
    'sphinx.ext.napoleon',
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

html_show_sourcelink = True
set_type_checking_flag = True
typehints_fully_qualified = True
napoleon_use_rtype = False
autosummary_generate = True
autosummary_generate_overwrite = True
autodoc_preserve_defaults = True
autodoc_inherit_docstrings = True
autodoc_default_options = {
    'autosummary': True
}

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

# For interactive plot in plotly
html_js_files = [
    "require.min.js",  # Add to your _static
    "custom.js",
]