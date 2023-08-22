rule run_SLAT:
    input:
        adata1="{path}/dataset1.h5ad",
        adata2="{path}/dataset2.h5ad",
        notebook="workflow/notebooks/run_SLAT.ipynb"
    output:
        metrics="{path}/SLAT/k_neighbors:{k_neighbors}-feature_type:{feature_type}-feature_dim:{feature_dim}-epochs:{epochs}-LGCN_layer:{LGCN_layer}-mlp_hidden:{mlp_hidden}-hidden_size:{hidden_size}-alpha:{alpha}-anchor_scale:{anchor_scale}-lr_mlp:{lr_mlp}-lr_wd:{lr_wd}-lr_recon:{lr_recon}-batch_d_per_iter:{batch_d_per_iter}-batch_r_per_iter:{batch_r_per_iter}-smooth:{smooth}/seed:{seed}/metrics.yaml",
        embd0="{path}/SLAT/k_neighbors:{k_neighbors}-feature_type:{feature_type}-feature_dim:{feature_dim}-epochs:{epochs}-LGCN_layer:{LGCN_layer}-mlp_hidden:{mlp_hidden}-hidden_size:{hidden_size}-alpha:{alpha}-anchor_scale:{anchor_scale}-lr_mlp:{lr_mlp}-lr_wd:{lr_wd}-lr_recon:{lr_recon}-batch_d_per_iter:{batch_d_per_iter}-batch_r_per_iter:{batch_r_per_iter}-smooth:{smooth}/seed:{seed}/embd0.csv",
        embd1="{path}/SLAT/k_neighbors:{k_neighbors}-feature_type:{feature_type}-feature_dim:{feature_dim}-epochs:{epochs}-LGCN_layer:{LGCN_layer}-mlp_hidden:{mlp_hidden}-hidden_size:{hidden_size}-alpha:{alpha}-anchor_scale:{anchor_scale}-lr_mlp:{lr_mlp}-lr_wd:{lr_wd}-lr_recon:{lr_recon}-batch_d_per_iter:{batch_d_per_iter}-batch_r_per_iter:{batch_r_per_iter}-smooth:{smooth}/seed:{seed}/embd1.csv",
    params:
        notebook_result="{path}/SLAT/k_neighbors:{k_neighbors}-feature_type:{feature_type}-feature_dim:{feature_dim}-epochs:{epochs}-LGCN_layer:{LGCN_layer}-mlp_hidden:{mlp_hidden}-hidden_size:{hidden_size}-alpha:{alpha}-anchor_scale:{anchor_scale}-lr_mlp:{lr_mlp}-lr_wd:{lr_wd}-lr_recon:{lr_recon}-batch_d_per_iter:{batch_d_per_iter}-batch_r_per_iter:{batch_r_per_iter}-smooth:{smooth}/seed:{seed}/run_SLAT.ipynb",
    log:
        "{path}/SLAT/k_neighbors:{k_neighbors}-feature_type:{feature_type}-feature_dim:{feature_dim}-epochs:{epochs}-LGCN_layer:{LGCN_layer}-mlp_hidden:{mlp_hidden}-hidden_size:{hidden_size}-alpha:{alpha}-anchor_scale:{anchor_scale}-lr_mlp:{lr_mlp}-lr_wd:{lr_wd}-lr_recon:{lr_recon}-batch_d_per_iter:{batch_d_per_iter}-batch_r_per_iter:{batch_r_per_iter}-smooth:{smooth}/seed:{seed}/run_SLAT.log",
    threads: 8
    # resources: gpu=1
    shell:
        """
        timeout {config[timeout]} papermill \
        -p adata1_file  {input.adata1} \
        -p adata2_file  {input.adata2} \
        -p k_cutoff {wildcards.k_neighbors} \
        -p feature_type  {wildcards.feature_type} \
        -p feature_dim  {wildcards.feature_dim} \
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
        adata1="{path}/dataset1.h5ad",
        adata2="{path}/dataset2.h5ad",
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
    # resources: gpu=1
    shell:
        """
        timeout {config[timeout]} papermill \
        -p adata1_file  {input.adata1} \
        -p adata2_file  {input.adata2} \
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


rule run_SLAT_noise:
    input:
        adata1="{path}/original/raw_dataset1.h5ad",
        adata2="{path}/original/raw_dataset2.h5ad",
        notebook="workflow/notebooks/run_SLAT_noise.ipynb",
    output:
        metric="{path}/SLAT_noise/seed:{seed}/nosie:{noise}/metrics.yaml",
        emb0="{path}/SLAT_noise/seed:{seed}/nosie:{noise}/emb0.csv",
        emb1="{path}/SLAT_noise/seed:{seed}/nosie:{noise}/emb1.csv",
        matching="{path}/SLAT_noise/seed:{seed}/nosie:{noise}/matching.csv",
    params:
        notebook_result="{path}/SLAT_noise/seed:{seed}/nosie:{noise}/run_SLAT_noise.ipynb",
    log:
        "{path}/SLAT_noise/seed:{seed}/nosie:{noise}/run_SLAT_noise.log"
    threads:8
    # resources: gpu=1
    shell:
        """
        timeout {config[timeout]} papermill \
        -p adata1_file {input.adata1} \
        -p adata2_file {input.adata2} \
        -p noise_level {wildcards.noise} \
        -p seed {wildcards.seed} \
        -p metric_file {output.metric} \
        -p emb0_file {output.emb0} \
        -p emb1_file {output.emb1} \
        -p matching_file {output.matching} \
        {input.notebook} {params.notebook_result} \
        > {log} 2>&1
        """


rule run_Harmony:
    input:
        adata1="{path}/dataset1.h5ad",
        adata2="{path}/dataset2.h5ad",
        notebook="workflow/notebooks/run_Harmony.ipynb"
    output:
        metric="{path}/Harmony/feature_dim:{feature_dim}-theta:{theta}-lambda:{lambda}/seed:{seed}/metrics.yaml",
        emb0="{path}/Harmony/feature_dim:{feature_dim}-theta:{theta}-lambda:{lambda}/seed:{seed}/emb0.csv",
        emb1="{path}/Harmony/feature_dim:{feature_dim}-theta:{theta}-lambda:{lambda}/seed:{seed}/emb1.csv",
    params:
        notebook_result="{path}/Harmony/feature_dim:{feature_dim}-theta:{theta}-lambda:{lambda}/seed:{seed}/run_Harmony.ipynb",
    log:
        "{path}/Harmony/feature_dim:{feature_dim}-theta:{theta}-lambda:{lambda}/seed:{seed}/run_Harmony.log"
    threads:8
    # resources: gpu=1
    shell:
        """
        timeout {config[timeout]} papermill \
        -p adata1_file {input.adata1} \
        -p adata2_file {input.adata2} \
        -p seed {wildcards.seed} \
        -p feature_dim {wildcards.feature_dim} \
        -p theta {wildcards.theta} \
        -p lamb {wildcards.lambda} \
        -p metrics_file {output.metric} \
        -p emb0_file {output.emb0} \
        -p emb1_file {output.emb1} \
        {input.notebook} {params.notebook_result} \
        > {log} 2>&1
        """


rule run_Seurat:
    input:
        adata1="{path}/dataset1.h5ad",
        adata2="{path}/dataset2.h5ad",
        notebook="workflow/notebooks/run_Seurat.ipynb",
    output:
        emb0="{path}/Seurat/anchor_feature:{anchor_feature}-dim:{dim}-k_anchor:{k_anchor}-k_filter:{k_filter}-k_score:{k_score}/seed:{seed}/emb0.csv",
        emb1="{path}/Seurat/anchor_feature:{anchor_feature}-dim:{dim}-k_anchor:{k_anchor}-k_filter:{k_filter}-k_score:{k_score}/seed:{seed}/emb1.csv",
    params:
        notebook_result="{path}/Seurat/anchor_feature:{anchor_feature}-dim:{dim}-k_anchor:{k_anchor}-k_filter:{k_filter}-k_score:{k_score}/seed:{seed}/run_Seurat.ipynb",
    log:
        "{path}/Seurat/anchor_feature:{anchor_feature}-dim:{dim}-k_anchor:{k_anchor}-k_filter:{k_filter}-k_score:{k_score}/seed:{seed}/run_Seurat.log"
    threads:8
    shell:
        """
        timeout {config[timeout]} papermill -k slat_r \
        -p adata1_file {input.adata1} \
        -p adata2_file {input.adata2} \
        -p emb0_file {output.emb0} \
        -p emb1_file {output.emb1} \
        -p anchor_feature {wildcards.anchor_feature} \
        -p dim {wildcards.dim} \
        -p k_anchor {wildcards.k_anchor} \
        -p k_filter {wildcards.k_filter} \
        -p k_score {wildcards.k_score} \
        {input.notebook} {params.notebook_result} \
        > {log} 2>&1
        """


rule Seurat_metrics:
    input:
        emb0="{path}/Seurat/anchor_feature:{anchor_feature}-dim:{dim}-k_anchor:{k_anchor}-k_filter:{k_filter}-k_score:{k_score}/seed:{seed}/emb0.csv",
        emb1="{path}/Seurat/anchor_feature:{anchor_feature}-dim:{dim}-k_anchor:{k_anchor}-k_filter:{k_filter}-k_score:{k_score}/seed:{seed}/emb1.csv",
        adata1="{path}/dataset1.h5ad",
        adata2="{path}/dataset2.h5ad",
        notebook="workflow/notebooks/emb2metrics.ipynb"
    output:
        metric="{path}/Seurat/anchor_feature:{anchor_feature}-dim:{dim}-k_anchor:{k_anchor}-k_filter:{k_filter}-k_score:{k_score}/seed:{seed}/metrics.yaml",
        matching="{path}/Seurat/anchor_feature:{anchor_feature}-dim:{dim}-k_anchor:{k_anchor}-k_filter:{k_filter}-k_score:{k_score}/seed:{seed}/matching.csv",
    params:
        notebook_result="{path}/Seurat/anchor_feature:{anchor_feature}-dim:{dim}-k_anchor:{k_anchor}-k_filter:{k_filter}-k_score:{k_score}/seed:{seed}/emb2metrics.ipynb",
    log:
        "{path}/Seurat/anchor_feature:{anchor_feature}-dim:{dim}-k_anchor:{k_anchor}-k_filter:{k_filter}-k_score:{k_score}/seed:{seed}/emb2metrics.log"
    threads:4
    shell:
        """
        papermill\
        -p emb0_file {input.emb0} \
        -p emb1_file {input.emb1} \
        -p adata1_file {input.adata1} \
        -p adata2_file {input.adata2} \
        -p metrics_file {output.metric} \
        -p matching_file {output.matching} \
        {input.notebook} {params.notebook_result} \
        > {log} 2>&1
        """


rule run_STAGATE:
    input:
        adata1="{path}/dataset1.h5ad",
        adata2="{path}/dataset1.h5ad",
        notebook="workflow/notebooks/run_STAGATE.ipynb",
    output:
        emb0="{path}/STAGATE/hidden_dim1:{hidden_dim1}-hidden_dim2:{hidden_dim2}-n_epochs:{n_epochs}-lr:{lr}/seed:{seed}/emb0.csv",
        emb1="{path}/STAGATE/hidden_dim1:{hidden_dim1}-hidden_dim2:{hidden_dim2}-n_epochs:{n_epochs}-lr:{lr}/seed:{seed}/emb1.csv",
    params:
        notebook_result="{path}/STAGATE/hidden_dim1:{hidden_dim1}-hidden_dim2:{hidden_dim2}-n_epochs:{n_epochs}-lr:{lr}/seed:{seed}/run_STAGATE.ipynb",
    container: 
        "docker://huhansan666666/stagate_pyg:v0.2"
    log:
        "{path}/STAGATE/hidden_dim1:{hidden_dim1}-hidden_dim2:{hidden_dim2}-n_epochs:{n_epochs}-lr:{lr}/seed:{seed}/run_STAGATE.log"
    threads:8
    resources: gpu=1
    shell:
        """
        timeout {config[timeout]} papermill \
        -p adata1_file {input.adata1} \
        -p adata2_file {input.adata2} \
        -p emb0_file {output.emb0} \
        -p emb1_file {output.emb1} \
        -p hidden_dim1 {wildcards.hidden_dim1} \
        -p hidden_dim2 {wildcards.hidden_dim2} \
        -p n_epochs {wildcards.n_epochs} \
        -p lr {wildcards.lr} \
        {input.notebook} {params.notebook_result} \
        > {log} 2>&1
        """

rule STAGATE_metrics:
    input:
        emb0="{path}/STAGATE/hidden_dim1:{hidden_dim1}-hidden_dim2:{hidden_dim2}-n_epochs:{n_epochs}-lr:{lr}/seed:{seed}/emb0.csv",
        emb1="{path}/STAGATE/hidden_dim1:{hidden_dim1}-hidden_dim2:{hidden_dim2}-n_epochs:{n_epochs}-lr:{lr}/seed:{seed}/emb1.csv",
        adata1="{path}/dataset1.h5ad",
        adata2="{path}/dataset1.h5ad",
        notebook="workflow/notebooks/emb2metrics.ipynb"
    output:
        metric="{path}/STAGATE/hidden_dim1:{hidden_dim1}-hidden_dim2:{hidden_dim2}-n_epochs:{n_epochs}-lr:{lr}/seed:{seed}/metrics.yaml",
        matching="{path}/STAGATE/hidden_dim1:{hidden_dim1}-hidden_dim2:{hidden_dim2}-n_epochs:{n_epochs}-lr:{lr}/seed:{seed}/matching.csv",
    params:
        notebook_result="{path}/STAGATE/hidden_dim1:{hidden_dim1}-hidden_dim2:{hidden_dim2}-n_epochs:{n_epochs}-lr:{lr}/seed:{seed}/emb2metrics.ipynb",
    log:
        "{path}/STAGATE/hidden_dim1:{hidden_dim1}-hidden_dim2:{hidden_dim2}-n_epochs:{n_epochs}-lr:{lr}/seed:{seed}/emb2metrics.log"
    threads:4
    shell:
        """
        papermill\
        -p emb0_file {input.emb0} \
        -p emb1_file {input.emb1} \
        -p adata1_file {input.adata1} \
        -p adata2_file {input.adata2} \
        -p metrics_file {output.metric} \
        -p matching_file {output.matching} \
        {input.notebook} {params.notebook_result} \
        > {log} 2>&1
        """


rule run_PASTE:
    input:
        adata1="{path}/dataset1.h5ad",
        adata2="{path}/dataset2.h5ad",
        notebook="workflow/notebooks/run_PASTE.ipynb",
    output:
        metric="{path}/PASTE/alpha:{alpha}/seed:{seed}/metrics.yaml",
        matching="{path}/PASTE/alpha:{alpha}/seed:{seed}/matching.csv",
    params:
        notebook_result="{path}/PASTE/alpha:{alpha}/seed:{seed}/run_PASTE.ipynb",
    log:
        "{path}/PASTE/alpha:{alpha}/seed:{seed}/run_PASTE.log"
    container: 
        "docker://huhansan666666/paste:v0.1"
    threads:8
    # resources: gpu=1
    shell:
        """
        timeout {config[timeout]} papermill \
        -p adata1_file {input.adata1} \
        -p adata2_file {input.adata2} \
        -p metrics_file {output.metric} \
        -p matching_file {output.matching} \
        -p alpha {wildcards.alpha} \
        {input.notebook} {params.notebook_result} \
        > {log} 2>&1
        """