r"""
Benchmark in different subsampling sizes and different datasets
"""
import os
from utils import target_directories, target_files


rule summarize:
    input:
        target_files(target_directories(config, sample=config["sample"]))
    output:
        "results/benchmark.csv"
    params:
        pattern=lambda wildcards: "results/{dataset}/cells:{cells}/seed:{seed}/{method}/metrics_all.yaml"
    threads: 1
    script:
        "scripts/summarize.py"