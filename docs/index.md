```{include} ../README.md
:end-line: 9
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

``scSLAT`` package implements the **SLAT** (**S**patially-**L**inked **A**lignment **T**ool) model, which aims to align heterogeneous single cell spatial data fast and accurately. scSLAT works well on various spatial technologies (such as Visium, MERFISH, Stereo-seq, seqFISH, Xenium) and can extend to large datasets easily.

```{eval-rst}
.. image:: _static/Model.png
   :width: 600
   :alt: Model architecture
```

## Key applications of scSLAT

1. **Heterogeneous alignment**: as the first algorithm designed for heterogeneous spatial alignment, scSLAT enable

    * **Cross technologies alignment**: such as Stereo-seq and MERFISH, Visium and Xenium.
    * **Multi modalities alignment**: such as spatial-ATAC-seq and Stereo-seq (cooperate with our previous work [scglue](https://scglue.readthedocs.io/en/latest/)). 
    * **Define spatial dynamics**: such as revealing spatial-temporal changes in time-series developmental datasets.

2. **Atlas alignment**: precisely align large single cell spatial atlas containing 200,000+ cells less than 3 minutes.

3. **3D reconstruction**: align multiple continuous spatial slices in parallel to rebuild 3D.

```{eval-rst}
.. image:: _static/imgalignment.gif
   :width: 300
   :alt: Alignment result
```

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