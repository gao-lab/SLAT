# Data preparation
## Datasets download
Here are all published dataset used in our paper. 

| Index | Paper                                                        | Species | Tissue                                       | Technology   | Resolution | Cells/Spots | Genes | Download                                                     |
| ----- | ------------------------------------------------------------ | ------- | -------------------------------------------- | ------------ | ---------- | ----------- | ----- | ------------------------------------------------------------ |
| 1     | [Chen et al.](https://doi.org/10.1016/j.cell.2022.04.003)    | Mouse   | Whole embryo                                 | Stereo-seq   | 0.2μm      | 5000-100,000     |  >20,000    | [website](https://db.cngb.org/stomics/mosta/download.html)   |
| 2     | [Lohoff et al.](https://www.biorxiv.org/content/10.1101/2020.11.20.391896v1) | Mouse   | Whole embryo                                 | seqFISH      | subcellular        | ~10,000     | 351   | [website](https://marionilab.cruk.cam.ac.uk/SpatialMouseAtlas/ ) |
| 3     | [Deng et al.](https://www.nature.com/articles/s41586-022-05094-1) | Mouse   | Whole embryo                                 | spatial-ATAC-seq | 20μm       | 2099      | >20,000    | [website](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE171943) |
| 4     | [Jeffrey et al.](https://www.science.org/doi/10.1126/science.aau5324) | Mouse   | Brain(hypothalamic preoptic)                 | MERFISH      | subcellular    | ~6,500      | 151   | [website](https://datadryad.org/stash/dataset/doi:10.5061/dryad.8t8s248) |
| 5     | [Kristen et al.](https://www.nature.com/articles/s41593-020-00787-0) | Human   | Brain(dorsolateral prefrontal cortex, DLPFC) | 10x Visium   | 50μm       | ~3500       |  >20,000    | [website](https://github.com/LieberInstitute/spatialLIBD)    |
| 6 | [Amanda et al.](https://www.biorxiv.org/content/10.1101/2022.10.06.510405v1) | Human | Breast cancer | 10x Visium | 50μm        | ~3500 | >20,000 | [website](https://www.10xgenomics.com/products/xenium-in-situ/preview-dataset-human-breast?utm_medium=other&utm_source=none&utm_campaign=xenium-explorer-software&useroffertype=website-page&userresearcharea=ra_g&userregion=multi&userrecipient=customer) |
| 7 | [Amanda et al.](https://www.biorxiv.org/content/10.1101/2022.10.06.510405v1) | Human | Breast cancer | 10x Xenium | subcellular | > 140,000 | 311 | [website](https://www.10xgenomics.com/products/xenium-in-situ/preview-dataset-human-breast?utm_medium=other&utm_source=none&utm_campaign=xenium-explorer-software&useroffertype=website-page&userresearcharea=ra_g&userregion=multi&userrecipient=customer) |

## Data preprocessing
We do some necessary pre-processing on some of them, you can repeat them by:
1. For Stereo-seq, please follow the `./Stereo_process.ipynb` to do pre-processing.
2. For spatial-ATAC-seq, we filter out the spots which not locate in tissue as authors did.
