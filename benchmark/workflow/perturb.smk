r"""
Benchmark in perturbed datasets
"""
import os
from utils import target_directories, target_files


rule summarize:
    input:
        metrics = expand(
            "perturb/{dataset}/cells:{cells}/inverse_noise:{inverse_noise}/seed:{seed}/{method}/metrics.yaml",
            cells=0,
            dataset=config['dataset_perturb'],
            seed=[*config['seed']], 
            method=config['method_perturb'],
            inverse_noise=config['inverse_noise']
        ),
    output:
        "perturb/benchmark_purterb.csv"
    params:
        pattern=lambda wildcards: "perturb/{dataset}/cells:{cells}/inverse_noise:{inverse_noise}/seed:{seed}/{method}/metrics.yaml"
    threads: 1
    script:
        "scripts/summarize.py"