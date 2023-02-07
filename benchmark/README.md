# Methods
We benchmark with following methods:

| Method  | Graph based | Spatial aware | Cross platform |
| ------- | ----------- | ------------- | -------------- |
| SLAT    | yes         | yes           | yes            |
| PASTE   | no          | yes           | no             |
| STAGATE | yes         | yes           | no             |
| Seurat  | no          | no            | yes            |
| Harmony | no          | no            | yes            |

# Metric
Due to we can not know ground truth between real spatial datasets, we newly design CRI (Celltype and Region matching Index) metric to measure the performance of spatial alignment. CRI checks how much alignment method recover corresponding celltype and histology region simultaneously.

$$
CRI= \frac{1}{M} \sum_{v_i,v_j \in M} I(i,j),\\

f(x)=\left\lbrace
    \begin{aligned}
    1 &,\ c_1^{i}=c_2^{j} \ \mathbf{and} \ r_1^{i}=r_2^{j} \\
    0 &,\ otherwise, \\
    \end{aligned}
\right.
$$

We also use Euclidean distance to measure the performance of spatial alignment:

$$
Euclidean\ distance = \frac{1}{M} \sum_{v_i,v_j \in M} ||\mathbf{s_i} - \mathbf{s_j}||_F^2
$$

# Datasets
> **Note**
> Dataset download links are available at [`here`](../data/README.md)

We do benchmark on following datasets:
| Index | Paper                                                        | Species | Tissue                                       | Technology   | Resolution | Cells/Spots | Genes | Download                                                     |
| ----- | ------------------------------------------------------------ | ------- | -------------------------------------------- | ------------ | ---------- | ----------- | ----- | ------------------------------------------------------------ |
| 1     | [Kristen et al.](https://www.nature.com/articles/s41593-020-00787-0) | Human   | Brain(dorsolateral prefrontal cortex, DLPFC) | 10x Visium   | 50μm       | ~3500       |  >20,000    | [website](https://github.com/LieberInstitute/spatialLIBD)    |
| 2     | [Jeffrey et al.](https://www.science.org/doi/10.1126/science.aau5324) | Mouse   | Brain(hypothalamic preoptic)                 | MERFISH      | subcellular    | ~6,500      | 151   | [website](https://datadryad.org/stash/dataset/doi:10.5061/dryad.8t8s248) |
| 3     | [Chen et al.](https://doi.org/10.1016/j.cell.2022.04.003)    | Mouse   | Whole embryo                                 | Stereo-seq   | 0.2μm      | 5000-100,000     |  >20,000    | [website](https://db.cngb.org/stomics/mosta/download.html)   |

# Run benchmark pipeline
> **Note**
> You need install extra dependencies following [`env/README.md`](../env/README.md).

To repeat our benchmark, just run:
```shell
snakemake --profile profiles/local -p
```
