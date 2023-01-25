r"""
Evluation in different edge corrpurtion 
"""

rule summarize:
    input:
        metrics=expand("results/{dataset}/SLAT_noise/seed:{seed}/nosie:{noise}/metrics.yaml",
                        dataset=config['noise_dataset'],
                        seed=[*config['seed']], 
                        noise=config['noise'])
    output:
        "results/noise.csv"
    params:
        pattern=lambda wildcards: "results/{dataset}/SLAT_noise/seed:{seed}/nosie:{noise}/metrics.yaml"
    threads: 1
    script:
        "scripts/summarize.py"