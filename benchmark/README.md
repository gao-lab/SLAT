# Methods
We benchmark with following methods
| Method  | Graph based | For spatial | Cross platform |
| ------- | ----------- | ----------- | -------------- |
| SLAT    | yes         | yes         | yes            |
| PASTE   | no          | yes         | no             |
| STAGATE | yes         | yes         | no             |
| Seurat  | no          | no          | yes            |
| Harmony | no          | no          | yes            |



# Metric and datasets


Due to we can not know ground truth between real spatial datasets, so we newly design CRI (Celltype and Region matching Index) score to measure the performance of spatial alignment. First, we annotate every dataset by its histology region and cell type, according to source literature, then check how do alignment method recover corresponding celltype and histology region simultaneously.

<img src = "../docs/_static/formula_benchmark1.png" width="400" height="150"/>

We benchmark on following datasets:
> NOTE: Dataset download links are available at [`../data/README.md`](../data/README.md)

| Index | Paper                                                        | Species | Tissue                                       | Technology   | Resolution | Cells/Spots | Genes | Download                                                     |
| ----- | ------------------------------------------------------------ | ------- | -------------------------------------------- | ------------ | ---------- | ----------- | ----- | ------------------------------------------------------------ |
| 1     | [Kristen et al.](https://www.nature.com/articles/s41593-020-00787-0) | Human   | Brain(dorsolateral prefrontal cortex, DLPFC) | 10x Visium   | 50μm       | ~3500       |  >20,000    | [website](https://github.com/LieberInstitute/spatialLIBD)    |
| 2     | [Jeffrey et al.](https://www.science.org/doi/10.1126/science.aau5324) | Mouse   | Brain(hypothalamic preoptic)                 | MERFISH      | subcellular    | ~6,500      | 151   | [website](https://datadryad.org/stash/dataset/doi:10.5061/dryad.8t8s248) |
| 3     | [Chen et al.](https://doi.org/10.1016/j.cell.2022.04.003)    | Mouse   | Whole embryo                                 | Stereo-seq   | 0.2μm      | 5000-100,000     |  >20,000    | [website](https://db.cngb.org/stomics/mosta/download.html)   |



# Run
> NOTE: you may need install `snakemake` and other dependencies following [`env/README.md`](../env/README.md)

To repeat our benchmark, just run:
```shell
snakemake --profile profiles/local -p
```
