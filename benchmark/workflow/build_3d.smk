r"""
Benchmark in 3D data reconstruction
"""

rule summarize:
    input:
        metrics = expand(
            "multi/{dataset}/cells:{cells}/seed:{seed}/{method}/metrics_all.yaml",
            cells=0,
            dataset=config['dataset_3d'],
            seed=[*config['seed']], 
            method=config['method_3d'],
        ),
    output:
        "multi/build_3d.csv"
    params:
        pattern=lambda wildcards: "multi/{dataset}/cells:{cells}/seed:{seed}/{method}/metrics_all.yaml"
    threads: 1
    script:
        "scripts/summarize.py"