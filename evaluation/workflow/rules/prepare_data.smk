import os

rule link_datas:
    input:
        dataset1=lambda wildcards: config["dataset"][wildcards.dataset][0],
        dataset2=lambda wildcards: config["dataset"][wildcards.dataset][1],
    output:
        dataset1="{path}/{dataset}/original/raw_dataset1.h5ad",
        dataset2="{path}/{dataset}/original/raw_dataset2.h5ad",
    log:
        "{path}/{dataset}/original/link_data.log"
    threads: 1
    shell:
        """
        ln -frs {input.dataset1} {output.dataset1} > {log}
        ln -frs {input.dataset2} {output.dataset2} >> {log}
        """

rule subsample:
    input:
        dataset1="{path}/{dataset}/original/raw_dataset1.h5ad",
        dataset2="{path}/{dataset}/original/raw_dataset2.h5ad",
        notebook="workflow/notebooks/subsample.ipynb"
    output:
        dataset1="{path}/{dataset}/cells:{cells}/dataset1.h5ad",
        dataset2="{path}/{dataset}/cells:{cells}/dataset2.h5ad",
    params:
        notebook_result="{path}/{dataset}/cells:{cells}/subsample.ipynb"
    log:
        "{path}/{dataset}/cells:{cells}/subsample.log"
    threads: 8
    shell:
        """
        papermill \
        -p dataset1_file  {input.dataset1} \
        -p dataset2_file  {input.dataset2} \
        -p output1_file  {output.dataset1} \
        -p output2_file  {output.dataset2} \
        -p cells  {wildcards.cells}\
        {input.notebook} {params.notebook_result} \
        > {log} 2>&1
        """

