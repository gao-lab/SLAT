import os
from utils import target_directories, target_files


rule plot:
    input:
        "results/evaluation.csv"
    output:
        evaluation_plot='results/pic/evaluation_noise_plot.pdf',
    script:
        "scripts/evaluation_plot.R"


rule summarize:
    input:
        metrics=expand("results/{dataset}/cells:{cells}/seed:{seed}/SLAT_noise/nosie:{noise}/metrics.yaml",
                        dataset=config['dataset'], cells=config['evaluation_size'], 
                        seed=config['seed'], noise=config['noise'])
    output:
        "results/evaluation.csv"
    params:
        pattern=lambda wildcards: "results/{dataset}/cells:{cells}/seed:{seed}/SLAT_noise/nosie:{noise}/metrics.yaml"
    threads: 1
    script:
        "scripts/summarize.py"