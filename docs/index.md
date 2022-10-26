```{include} ../README.md
:end-line: 10
```

```{eval-rst}
.. .. image:: _static/SLAT.png
..    :width: 140
..    :alt: SLAT icon
..    :align: right
```

```{eval-rst}
.. toctree::
   :maxdepth: 2
   :caption: Contents:
```

**scSLAT** package implements the **SLAT** (**S**patially-**L**inked **A**lignment **T**ool) model, which aims to align single cell spatial data fast and accurate. scSLAT can extend to large dataset and various omics data easily.

```{eval-rst}
.. image:: _static/Model.png
   :width: 600
   :alt: Model architecture
```

## Key applications of scSLAT
- Accurately align large (100,000+) single cell spatial data in few **seconds**.
- Align **multi-platform**(such as Stereo-seq and MERFISH) and **multi-modalities** (such as ATAC and RNA) single cell spatial data.
- Revealing spatial-temporal changes by time-series developmental spatial data alignment.

## Manuscript
Preparing

## Getting started with scSLAT
To get started with ``scSLAT``, check out our {doc}`installation guide <install>` and {doc}`tutorials <tutorials>`.

## Contributing to scSLAT
Single-cell spatial omics is still developing rapidly. We are happy about your contributions! Please submit Pull requests on our [Github Repo](https://github.com/gao-lab/SLAT) .We think your contribution will make a big difference to community!


```{eval-rst}
.. toctree::
   :maxdepth: 2
   :caption: Contents:

   install
   tutorials
   data
   api
   release
   credits

```