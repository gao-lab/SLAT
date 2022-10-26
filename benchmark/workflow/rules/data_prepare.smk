# Data to models input format

rule link_data:
    input:
        dataset1=lambda wildcards: config["dataset"][wildcards.dataset][0],
        dataset2=lambda wildcards: config["dataset"][wildcards.dataset][1],
    output:
        dataset1="{path}/{dataset}/original/dataset1.h5ad",
        dataset2="{path}/{dataset}/original/dataset2.h5ad",
    log:
        "{path}/{dataset}/original/link_data.log"
    threads: 1
    shell:
        """
        ln -frs {input.dataset1} {output.dataset1} > {log}
        ln -frs {input.dataset2} {output.dataset2} >> {log}
        """


rule data2input:
    input:
        dataset1="{path}/{dataset}/original/dataset1.h5ad",
        dataset2="{path}/{dataset}/original/dataset2.h5ad",
        notebook='workflow/notebooks/prepare_data.ipynb',
    output:
        adata1='{path}/{dataset}/cells:{cells}/seed:{seed}/adata1.h5ad',
        adata2='{path}/{dataset}/cells:{cells}/seed:{seed}/adata2.h5ad',
    params:    
        notebook_result='{path}/{dataset}/cells:{cells}/seed:{seed}/prepare_data.ipynb',
    log:
        "{path}/{dataset}/cells:{cells}/seed:{seed}/prepare_data.log"
    threads: 8
    shell:
        """
        papermill \
        -p dataset1_file {input.dataset1} \
        -p dataset2_file {input.dataset2} \
        -p cells {wildcards.cells} \
        -p seed {wildcards.seed} \
        -p adata1_out {output.adata1} \
        -p adata2_out {output.adata2} \
        {input.notebook} {params.notebook_result} \
        > {log} 2>&1
        """