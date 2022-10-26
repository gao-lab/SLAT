import os
from utils import target_directories, target_files


rule plot:
    input:
        "results/benchmark.csv"
    output:
        local_metric='results/pic/local_metric.pdf',
        global_metric='results/pic/noisglobal_metrice_h5.pdf',
    script:
        "scripts/benchmark_plot.R"


rule summarize:
    input:
        target_files(target_directories(config, sample=config["sample"]))
    output:
        "results/benchmark.csv"
    params:
        pattern=lambda wildcards: "results/{dataset}/cells:{cells}/seed:{seed}/{method}/metrics.yaml"
    threads: 1
    script:
        "scripts/summarize.py"