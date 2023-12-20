# Tutorials

``SLAT`` has a wide range of application scenarios in spatial alignment, especially in heterogeneous alignment. We provide following tutorials for you to get started with it.

## Pre-match (Recommended)
We recommend you to pre-match your data before alignment, which can significantly improve the alignment performance.

```{eval-rst}
.. nbgallery::
    tutorials/pre_match.ipynb
```

## Basic Usage

### Two slices alignment
We  show how to use ``SLAT`` to align two spatial slices. You can find basic usages of ``SLAT`` in this tutorial.

### Multiple slices alignment
``SLAT`` can also align multiple datasets once, which is important for 3D structure reconstruction.

```{eval-rst}
.. nbgallery::
    tutorials/basic_usage.ipynb
    tutorials/multi_datasets.ipynb
```


## Heterogeneous alignment

### Cross-technology alignment
We show how to use ``SLAT`` to align heterogeneous 10x Visium and 10x Xenium data in this tutorial.

### Define spatial dynamics regions
In this tutorial, we focus on finding dynamic regions in embryo development via ``SLAT``.

```{eval-rst}
.. nbgallery::
    tutorials/cross_technology.ipynb
    tutorials/times_series.ipynb
```