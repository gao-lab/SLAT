rule run_SLAT:
    input:
        dataset1="{path}/dataset1.h5ad",
        dataset2="{path}/dataset2.h5ad",
        notebook="workflow/notebooks/run_SLAT.ipynb"
    output:
        metrics="{path}/k_neighbors:{k_neighbors}-feature_type:{feature_type}-epochs:{epochs}-LGCN_layer:{LGCN_layer}-mlp_hidden:{mlp_hidden}-hidden_size:{hidden_size}-alpha:{alpha}-anchor_scale:{anchor_scale}-lr_mlp:{lr_mlp}-lr_wd:{lr_wd}-lr_recon:{lr_recon}-batch_d_per_iter:{batch_d_per_iter}-batch_r_per_iter:{batch_r_per_iter}-smooth:{smooth}/seed:{seed}/metrics.yaml",
        embd0="{path}/k_neighbors:{k_neighbors}-feature_type:{feature_type}-epochs:{epochs}-LGCN_layer:{LGCN_layer}-mlp_hidden:{mlp_hidden}-hidden_size:{hidden_size}-alpha:{alpha}-anchor_scale:{anchor_scale}-lr_mlp:{lr_mlp}-lr_wd:{lr_wd}-lr_recon:{lr_recon}-batch_d_per_iter:{batch_d_per_iter}-batch_r_per_iter:{batch_r_per_iter}-smooth:{smooth}/seed:{seed}/embd0.csv",
        embd1="{path}/k_neighbors:{k_neighbors}-feature_type:{feature_type}-epochs:{epochs}-LGCN_layer:{LGCN_layer}-mlp_hidden:{mlp_hidden}-hidden_size:{hidden_size}-alpha:{alpha}-anchor_scale:{anchor_scale}-lr_mlp:{lr_mlp}-lr_wd:{lr_wd}-lr_recon:{lr_recon}-batch_d_per_iter:{batch_d_per_iter}-batch_r_per_iter:{batch_r_per_iter}-smooth:{smooth}/seed:{seed}/embd1.csv",
    params:
        notebook_result="{path}/k_neighbors:{k_neighbors}-feature_type:{feature_type}-epochs:{epochs}-LGCN_layer:{LGCN_layer}-mlp_hidden:{mlp_hidden}-hidden_size:{hidden_size}-alpha:{alpha}-anchor_scale:{anchor_scale}-lr_mlp:{lr_mlp}-lr_wd:{lr_wd}-lr_recon:{lr_recon}-batch_d_per_iter:{batch_d_per_iter}-batch_r_per_iter:{batch_r_per_iter}-smooth:{smooth}/seed:{seed}/run_SLAT.ipynb",
    log:
        "{path}/k_neighbors:{k_neighbors}-feature_type:{feature_type}-epochs:{epochs}-LGCN_layer:{LGCN_layer}-mlp_hidden:{mlp_hidden}-hidden_size:{hidden_size}-alpha:{alpha}-anchor_scale:{anchor_scale}-lr_mlp:{lr_mlp}-lr_wd:{lr_wd}-lr_recon:{lr_recon}-batch_d_per_iter:{batch_d_per_iter}-batch_r_per_iter:{batch_r_per_iter}-smooth:{smooth}/seed:{seed}/run_SLAT.log",
    threads: 8
    resources: gpu=1
    shell:
        """
        timeout {config[timeout]} papermill \
        -p adata1_file  {input.dataset1} \
        -p adata2_file  {input.dataset2} \
        -p k_cutoff {wildcards.k_neighbors} \
        -p feature_type  {wildcards.feature_type} \
        -p epochs  {wildcards.epochs} \
        -p LGCN_layer  {wildcards.LGCN_layer} \
        -p mlp_hidden  {wildcards.mlp_hidden} \
        -p hidden_size  {wildcards.hidden_size} \
        -p alpha  {wildcards.alpha} \
        -p anchor_scale {wildcards.anchor_scale} \
        -p lr_mlp {wildcards.lr_mlp} \
        -p lr_wd  {wildcards.lr_wd} \
        -p lr_recon  {wildcards.lr_recon} \
        -p batch_d_per_iter {wildcards.batch_d_per_iter} \
        -p batch_r_per_iter {wildcards.batch_r_per_iter} \
        -p smooth  {wildcards.smooth} \
        -p seed  {wildcards.seed} \
        -p emb0_file  {output.embd0} \
        -p emb1_file  {output.embd1} \
        -p metrics_file  {output.metrics} \
        {input.notebook} {params.notebook_result} \
        > {log} 2>&1
        """


rule run_LGCN:
    input:
        dataset1="{path}/dataset1.h5ad",
        dataset2="{path}/dataset2.h5ad",
        notebook="workflow/notebooks/run_LGCN.ipynb"
    output:
        metrics="{path}/LGCN_layer:{LGCN_layer}-k_neighbors:{k_neighbors}-feature_type:{feature_type}-smooth:{smooth}/metrics.yaml",
        embd0="{path}/LGCN_layer:{LGCN_layer}-k_neighbors:{k_neighbors}-feature_type:{feature_type}-smooth:{smooth}/embd0.csv",
        embd1="{path}/LGCN_layer:{LGCN_layer}-k_neighbors:{k_neighbors}-feature_type:{feature_type}-smooth:{smooth}/embd1.csv",
    params:
        notebook_result="{path}/LGCN_layer:{LGCN_layer}-k_neighbors:{k_neighbors}-feature_type:{feature_type}-smooth:{smooth}/run_SLAT.ipynb",
    log:
        "{path}/LGCN_layer:{LGCN_layer}-k_neighbors:{k_neighbors}-feature_type:{feature_type}-smooth:{smooth}/run_SLAT.log",
    threads: 8
    resources: gpu=1
    shell:
        """
        timeout {config[timeout]} papermill \
        -p adata1_file  {input.dataset1} \
        -p adata2_file  {input.dataset2} \
        -p k_cutoff {wildcards.k_neighbors} \
        -p feature_type  {wildcards.feature_type} \
        -p LGCN_layer  {wildcards.LGCN_layer} \
        -p smooth  {wildcards.smooth} \
        -p emb0_file  {output.embd0} \
        -p emb1_file  {output.embd1} \
        -p metrics_file  {output.metrics} \
        {input.notebook} {params.notebook_result} \
        > {log} 2>&1
        """