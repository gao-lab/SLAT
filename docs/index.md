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

- **Heterogeneous** alignment 
   - **Cross technologies align**: such as Stereo-seq and MERFISH, Visium and Xenium.
   - **Multi modalities align**: such as spatial-ATAC-seq and Stereo-seq. 
   - **Define spatial dynamics**: such as revealing spatial-temporal changes in time-series developmental datasets.
- **Fast**: SLAT can precisely align large single cell spatial dataset with 100,000+ cells  in few **seconds**.
- **3D reconstruction** of multiple continuous spatial datasets

```{eval-rst}
.. image:: _static/imgalignment.gif
   :width: 300
   :alt: Alignment result
```

## Manuscript
Preparing

## Getting started with scSLAT
To get started with ``scSLAT``, check out our {doc}`installation guide <install>` and {doc}`tutorials <tutorials>`.

## Contributing to scSLAT
Single-cell spatial omics is still developing rapidly. We are happy about your contributions! Please submit Pull requests on our [Github Repo](https://github.com/gao-lab/SLAT). We think your contribution will make a big difference to community!


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