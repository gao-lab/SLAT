import os
from utils import target_directories, target_files


rule summarize:
    input:
        target_files(target_directories(config, sample=config["sample"], hyperparameter='SLAT_hyperparam'))
    output:
        "results/hyperparam.csv"
    params:
        pattern=lambda wildcards: "results/{dataset}/cells:{datasize}/k_neighbors:{k_neighbors}-feature_type:{feature_type}-epochs:{epochs}-LGCN_layer:{LGCN_layer}-mlp_hidden:{mlp_hidden}-hidden_size:{hidden_size}-alpha:{alpha}-anchor_scale:{anchor_scale}-lr_mlp:{lr_mlp}-lr_wd:{lr_wd}-lr_recon:{lr_recon}-batch_d_per_iter:{batch_d_per_iter}-batch_r_per_iter:{batch_r_per_iter}-smooth:{smooth}/seed:{seed}/metrics.yaml"
    threads: 1
    script:
        "scripts/summarize.py"

