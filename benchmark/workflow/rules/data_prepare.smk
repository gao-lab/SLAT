r"""
Data to models input format
"""

rule link_data:
    input:
        dataset1=lambda wildcards: config["dataset"][wildcards.dataset][0],
        dataset2=lambda wildcards: config["dataset"][wildcards.dataset][1],
    output:
        dataset1="results/{dataset}/original/dataset1.h5ad",
        dataset2="results/{dataset}/original/dataset2.h5ad",
    log:
        "results/{dataset}/original/link_data.log"
    threads: 1
    shell:
        """
        ln -frs {input.dataset1} {output.dataset1} > {log}
        ln -frs {input.dataset2} {output.dataset2} >> {log}
        """


rule data2input:
    input:
        dataset1="results/{dataset}/original/dataset1.h5ad",
        dataset2="results/{dataset}/original/dataset2.h5ad",
        notebook='workflow/notebooks/prepare_data.ipynb',
    output:
        adata1='results/{dataset}/cells:{cells}/seed:{seed}/adata1.h5ad',
        adata2='results/{dataset}/cells:{cells}/seed:{seed}/adata2.h5ad',
    params:    
        notebook_result='results/{dataset}/cells:{cells}/seed:{seed}/prepare_data.ipynb',
    log:
        "results/{dataset}/cells:{cells}/seed:{seed}/prepare_data.log"
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


rule data_split:
    input:
        adata1='results/{dataset}/cells:{cells}/seed:{seed}/adata1.h5ad',
        notebook='workflow/notebooks/data_split.ipynb',
    output:
        adata1='split/{dataset}/cells:{cells}/seed:{seed}/split_ratio:{split_ratio}/adata1.h5ad',
        adata2='split/{dataset}/cells:{cells}/seed:{seed}/split_ratio:{split_ratio}/adata2.h5ad',
    params:
        notebook_result='split/{dataset}/cells:{cells}/seed:{seed}/split_ratio:{split_ratio}/data_split.ipynb',
    log:
        "split/{dataset}/cells:{cells}/seed:{seed}/split_ratio:{split_ratio}/split_data.log"
    threads: 8
    shell:
        """
        papermill \
        -p adata1_file {input.adata1} \
        -p adata1_out {output.adata1} \
        -p adata2_out {output.adata2} \
        -p seed {wildcards.seed} \
        {input.notebook} {params.notebook_result} \
        > {log} 2>&1
        """


rule link_data3D:
    input:
        dataset1=lambda wildcards: config["dataset_3d"][wildcards.dataset][0],
        dataset2=lambda wildcards: config["dataset_3d"][wildcards.dataset][1],
    output:
        dataset1="multi/{dataset}/original/dataset1.h5ad",
        dataset2="multi/{dataset}/original/dataset2.h5ad",
    log:
        "multi/{dataset}/original/link_data.log"
    threads: 1
    shell:
        """
        ln -frs {input.dataset1} {output.dataset1} > {log}
        ln -frs {input.dataset2} {output.dataset2} >> {log}
        """


rule data2input3d:
    input:
        dataset1="multi/{dataset}/original/dataset1.h5ad",
        dataset2="multi/{dataset}/original/dataset2.h5ad",
        notebook='workflow/notebooks/prepare_data.ipynb',
    output:
        adata1='multi/{dataset}/cells:{cells}/seed:{seed}/adata1.h5ad',
        adata2='multi/{dataset}/cells:{cells}/seed:{seed}/adata2.h5ad',
    params:    
        notebook_result='multi/{dataset}/cells:{cells}/seed:{seed}/prepare_data.ipynb',
    log:
        "multi/{dataset}/cells:{cells}/seed:{seed}/prepare_data.log"
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


rule link_data_perturb:
    input:
        dataset1=lambda wildcards: config["dataset_perturb"][wildcards.dataset][0],
        dataset2=lambda wildcards: config["dataset_perturb"][wildcards.dataset][1],
    output:
        dataset1="perturb/{dataset}/original/dataset1.h5ad",
        dataset2="perturb/{dataset}/original/dataset2.h5ad",
    log:
        "perturb/{dataset}/original/link_data.log"
    threads: 1
    shell:
        """
        ln -frs {input.dataset1} {output.dataset1} > {log}
        ln -frs {input.dataset2} {output.dataset2} >> {log}
        """

rule data2input_perturb:
    input:
        dataset1="perturb/{dataset}/original/dataset1.h5ad",
        dataset2="perturb/{dataset}/original/dataset2.h5ad",
        notebook='workflow/notebooks/prepare_data.ipynb',
    output:
        adata1='perturb/{dataset}/cells:{cells}/inverse_noise:{inverse_noise}/seed:{seed}/adata1.h5ad',
        adata2='perturb/{dataset}/cells:{cells}/inverse_noise:{inverse_noise}/seed:{seed}/adata2.h5ad',
    params:    
        notebook_result='perturb/{dataset}/cells:{cells}/inverse_noise:{inverse_noise}/seed:{seed}/prepare_data.ipynb',
    log:
        "perturb/{dataset}/cells:{cells}/inverse_noise:{inverse_noise}/seed:{seed}/prepare_data.log"
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
        -p rotation True \
        -p perturb True \
        -p inverse_noise {wildcards.inverse_noise} \
        {input.notebook} {params.notebook_result} \
        > {log} 2>&1
        """