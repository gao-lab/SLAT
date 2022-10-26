import os


rule summarize:
    input:
        expand("results/{dataset}/cells:{datasize}/LGCN_layer:{LGCN_layer}-k_neighbors:{k_neighbors}-feature_type:{feature_type}-smooth:{smooth}/metrics.yaml",
                dataset=config["dataset"], datasize=config["datasize"]["cells"]["choices"], LGCN_layer=config["LGCN_layer"],
                k_neighbors=config["spatial_graph"]["k_neighbors"]["choices"], feature_type=config["feature_embed"]["feature_type"],
                smooth=config["smooth"])
    output:
        "results/LGCN.csv"
    params:
        pattern=lambda wildcards: "results/{dataset}/cells:{datasize}/LGCN_layer:{LGCN_layer}-k_neighbors:{k_neighbors}-feature_type:{feature_type}-smooth:{smooth}/metrics.yaml"
    threads: 1
    script:
        "scripts/summarize.py"