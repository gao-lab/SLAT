import os
from utils import target_directories, target_files


rule summarize_slat:
    input:
        target_files(target_directories(config, sample=config["sample"], hyperparameter='SLAT_hyperparam', method='SLAT'))
    output:
        "results/hyperparam_slat.csv"
    params:
        pattern=lambda wildcards: "results/{dataset}/cells:{datasize}/SLAT/k_neighbors:{k_neighbors}-feature_type:{feature_type}-feature_dim:{feature_dim}-epochs:{epochs}-LGCN_layer:{LGCN_layer}-mlp_hidden:{mlp_hidden}-hidden_size:{hidden_size}-alpha:{alpha}-anchor_scale:{anchor_scale}-lr_mlp:{lr_mlp}-lr_wd:{lr_wd}-lr_recon:{lr_recon}-batch_d_per_iter:{batch_d_per_iter}-batch_r_per_iter:{batch_r_per_iter}-smooth:{smooth}/seed:{seed}/metrics.yaml"
    threads: 1
    script:
        "scripts/summarize.py"


rule summarize_harmony:
    input:
        target_files(target_directories(config, sample=config["sample"], hyperparameter='Harmony_hyperparam', method='Harmony'))
    output:
        "results/hyperparam_harmony.csv"
    params:
        pattern=lambda wildcards: "results/{dataset}/cells:{datasize}/Harmony/feature_dim:{feature_dim}-theta:{theta}-lambda:{lambda}/seed:{seed}/metrics.yaml"
    threads: 1
    script:
        "scripts/summarize.py"


rule summarize_seurat:
    input:
        target_files(target_directories(config, sample=config["sample"], hyperparameter='Seurat_hyperparam', method='Seurat'))
    output:
        "results/hyperparam_seurat.csv"
    params:
        pattern=lambda wildcards: "results/{dataset}/cells:{datasize}/Seurat/anchor_feature:{anchor_feature}-dim:{dim}-k_anchor:{k_anchor}-k_filter:{k_filter}-k_score:{k_score}/seed:{seed}/metrics.yaml"
    threads: 1
    script:
        "scripts/summarize.py"


rule summarize_stagate:
    input:
        target_files(target_directories(config, sample=config["sample"], hyperparameter='STAGATE_hyperparam', method='STAGATE'))
    output:
        "results/hyperparam_stagate.csv"
    params:
        pattern=lambda wildcards: "results/{dataset}/cells:{datasize}/STAGATE/hidden_dim1:{hidden_dim1}-hidden_dim2:{hidden_dim2}-n_epochs:{n_epochs}-lr:{lr}/seed:{seed}/metrics.yaml"
    threads: 1
    script:
        "scripts/summarize.py"


rule summarize_paste:
    input:
        target_files(target_directories(config, sample=config["sample"], hyperparameter='PASTE_hyperparam', method='PASTE'))
    output:
        "results/hyperparam_paste.csv"
    params:
        pattern=lambda wildcards: "results/{dataset}/cells:{datasize}/PASTE/alpha:{alpha}/seed:{seed}/metrics.yaml"
    threads: 1
    script:
        "scripts/summarize.py"