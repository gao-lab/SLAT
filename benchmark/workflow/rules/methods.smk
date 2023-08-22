r"""
Benchmark different methods
"""
import os

##############################################################################################
#
#  Run methods
#
##############################################################################################
rule run_SLAT_dpca:
    input:
        adata1="{path}/adata1.h5ad",
        adata2="{path}/adata2.h5ad",
        notebook="workflow/notebooks/run_SLAT_dpca.ipynb",
    output:
        metric="{path}/SLAT_dpca/metrics.yaml",
        emb0="{path}/SLAT_dpca/emb0.csv",
        emb1="{path}/SLAT_dpca/emb1.csv",
        matching="{path}/SLAT_dpca/matching.csv",
        graphs="{path}/SLAT_dpca/graph.pkl",
    params:
        notebook_result="{path}/SLAT_dpca/run_SLAT_dpca.ipynb",
    log:
        "{path}/SLAT_dpca/run_SLAT_dpca.log"
    threads:8
    resources: gpu=1
    shell:
        """
        timeout {config[timeout]} papermill \
        -p adata1_file {input.adata1} \
        -p adata2_file {input.adata2} \
        -p metric_file {output.metric} \
        -p emb0_file {output.emb0} \
        -p emb1_file {output.emb1} \
        -p graphs_file {output.graphs} \
        -p matching_file {output.matching} \
        {input.notebook} {params.notebook_result} \
        > {log} 2>&1
        """


rule run_SLAT_dpca_one2many:
    input:
        adata1="{path}/adata1.h5ad",
        adata2="{path}/adata2.h5ad",
        notebook="workflow/notebooks/run_SLAT_dpca_one2many.ipynb",
    output:
        metric="{path}/SLAT_dpca_one2many/metrics.yaml",
        emb0="{path}/SLAT_dpca_one2many/emb0.csv",
        emb1="{path}/SLAT_dpca_one2many/emb1.csv",
        matching="{path}/SLAT_dpca_one2many/matching.csv",
        graphs="{path}/SLAT_dpca_one2many/graph.pkl",
    params:
        notebook_result="{path}/SLAT_dpca_one2many/run_SLAT_dpca_one2many.ipynb",
    log:
        "{path}/SLAT_dpca_one2many/run_SLAT_dpca_one2many.log"
    threads:8
    resources: gpu=1
    shell:
        """
        timeout {config[timeout]} papermill \
        -p adata1_file {input.adata1} \
        -p adata2_file {input.adata2} \
        -p metric_file {output.metric} \
        -p emb0_file {output.emb0} \
        -p emb1_file {output.emb1} \
        -p graphs_file {output.graphs} \
        -p matching_file {output.matching} \
        {input.notebook} {params.notebook_result} \
        > {log} 2>&1
        """


rule run_PASTE:
    input:
        adata1="{path}/adata1.h5ad",
        adata2="{path}/adata2.h5ad",
        notebook="workflow/notebooks/run_PASTE.ipynb",
    output:
        metric="{path}/PASTE/run_time.yaml",
        matching="{path}/PASTE/matching.csv",
    params:
        notebook_result="{path}/PASTE/run_PASTE.ipynb",
    log:
        "{path}/PASTE/run_PASTE.log"
    container: 
        "docker://huhansan666666/paste:v0.1"
    threads:8
    resources: gpu=1
    shell:
        """
        timeout {config[timeout]} papermill \
        -p adata1_file {input.adata1} \
        -p adata2_file {input.adata2} \
        -p metric_file {output.metric} \
        -p matching_file {output.matching} \
        {input.notebook} {params.notebook_result} \
        > {log} 2>&1
        """


rule run_PASTE2:
    input:
        adata1="{path}/adata1.h5ad",
        adata2="{path}/adata2.h5ad",
        notebook="workflow/notebooks/run_PASTE2.ipynb",
    output:
        metric="{path}/PASTE2/run_time.yaml",
        matching="{path}/PASTE2/matching.csv",
    params:
        notebook_result="{path}/PASTE2/run_PASTE2.ipynb",
    log:
        "{path}/PASTE2/run_PASTE2.log"
    container: 
        "docker://huhansan666666/paste:v0.1"
    threads:8
    resources: gpu=1
    shell:
        """
        timeout {config[timeout]} papermill \
        -p adata1_file {input.adata1} \
        -p adata2_file {input.adata2} \
        -p metric_file {output.metric} \
        -p matching_file {output.matching} \
        {input.notebook} {params.notebook_result} \
        > {log} 2>&1
        """


rule run_STAGATE:
    input:
        adata1="{path}/adata1.h5ad",
        adata2="{path}/adata2.h5ad",
        notebook="workflow/notebooks/run_STAGATE.ipynb",
    output:
        emb0="{path}/STAGATE/emb0.csv",
        emb1="{path}/STAGATE/emb1.csv",
    params:
        notebook_result="{path}/STAGATE/run_STAGATE.ipynb",
    container: 
        "docker://huhansan666666/stagate_pyg:v0.2"
    log:
        "{path}/STAGATE/run_STAGATE.log"
    threads:8
    resources: gpu=1
    shell:
        """
        timeout {config[timeout]} papermill \
        -p adata1_file {input.adata1} \
        -p adata2_file {input.adata2} \
        -p emb0_file {output.emb0} \
        -p emb1_file {output.emb1} \
        {input.notebook} {params.notebook_result} \
        > {log} 2>&1
        """


rule run_Harmony:
    input:
        adata1="{path}/adata1.h5ad",
        adata2="{path}/adata2.h5ad",
        notebook="workflow/notebooks/run_Harmony.ipynb",
    output:
        emb0="{path}/Harmony/emb0.csv",
        emb1="{path}/Harmony/emb1.csv",
    params:
        notebook_result="{path}/Harmony/run_Harmony.ipynb",
    log:
        "{path}/Harmony/run_Harmony.log"
    threads:8
    shell:
        """
        timeout {config[timeout]} papermill \
        -p adata1_file {input.adata1} \
        -p adata2_file {input.adata2} \
        -p emb0_file {output.emb0} \
        -p emb1_file {output.emb1} \
        {input.notebook} {params.notebook_result} \
        > {log} 2>&1
        """


rule run_Seurat:
    input:
        adata1="{path}/adata1.h5ad",
        adata2="{path}/adata2.h5ad",
        notebook="workflow/notebooks/run_Seurat.ipynb",
    output:
        emb0="{path}/Seurat/emb0.csv",
        emb1="{path}/Seurat/emb1.csv",
        # seurat_RDS_file="{path}/Seurat/seurat_combine.rds",
    params:
        notebook_result="{path}/Seurat/run_Seurat.ipynb",
    log:
        "{path}/Seurat/run_Seurat.log"
    threads:8
    shell:
        """
        timeout {config[timeout]} papermill -k slat_r \
        -p adata1_file {input.adata1} \
        -p adata2_file {input.adata2} \
        -p emb0_file {output.emb0} \
        -p emb1_file {output.emb1} \
        {input.notebook} {params.notebook_result} \
        > {log} 2>&1
        """


##############################################################################################
#
# Metircs for one2one
#
##############################################################################################
rule matching2metrics:
    input:
        adata1="{path}/adata1.h5ad",
        adata2="{path}/adata2.h5ad",
        matching="{path}/{method,PASTE|PASTE2}/matching.csv",
        notebook="workflow/notebooks/emb2metrics.ipynb"
    output:
        metric="{path}/{method,PASTE|PASTE2}/metrics.yaml",
    params:
        notebook_result="{path}/{method,PASTE|PASTE2}/emb2metrics.ipynb",
    log:
        "{path}/{method,PASTE|PASTE2}/emb2metrics.log"
    threads:4
    shell:
        """
        papermill\
        -p adata1_file {input.adata1} \
        -p adata2_file {input.adata2} \
        -p matching_file {input.matching} \
        -p metric_file {output.metric} \
        {input.notebook} {params.notebook_result} \
        > {log} 2>&1
        """

rule emb2metrics:
    input:
        emb0="{path}/{method,Seurat|STAGATE|Harmony}/emb0.csv",
        emb1="{path}/{method,Seurat|STAGATE|Harmony}/emb1.csv",
        adata1="{path}/adata1.h5ad",
        adata2="{path}/adata2.h5ad",
        notebook="workflow/notebooks/emb2metrics.ipynb"
    output:
        metric="{path}/{method,Seurat|STAGATE|Harmony}/metrics.yaml",
        matching="{path}/{method,Seurat|STAGATE|Harmony}/matching.csv",
    params:
        notebook_result="{path}/{method,Seurat|STAGATE|Harmony}/emb2metrics.ipynb",
    log:
        "{path}/{method,Seurat|STAGATE|Harmony}/emb2metrics.log"
    threads:4
    shell:
        """
        papermill\
        -p emb0_file {input.emb0} \
        -p emb1_file {input.emb1} \
        -p adata1_file {input.adata1} \
        -p adata2_file {input.adata2} \
        -p metric_file {output.metric} \
        -p matching_file {output.matching} \
        {input.notebook} {params.notebook_result} \
        > {log} 2>&1
        """

##############################################################################################
#
# Metircs for one2many
#
##############################################################################################
rule emb2metrics_one2many:
    input:
        emb0="{path}/{method,Seurat|STAGATE|Harmony}/emb0.csv",
        emb1="{path}/{method,Seurat|STAGATE|Harmony}/emb1.csv",
        adata1="{path}/adata1.h5ad",
        adata2="{path}/adata2.h5ad",
        notebook="workflow/notebooks/emb2metrics_one2many.ipynb"
    output:
        metric="{path}/{method,Seurat|STAGATE|Harmony}_one2many/metrics.yaml",
        matching="{path}/{method,Seurat|STAGATE|Harmony}_one2many/matching.csv",
    params:
        notebook_result="{path}/{method,Seurat|STAGATE|Harmony}_one2many/emb2metrics_one2many.ipynb",
    log:
        "{path}/{method,Seurat|STAGATE|Harmony}_one2many/emb2metrics_one2many.log"
    threads:4
    shell:
        """
        papermill\
        -p emb0_file {input.emb0} \
        -p emb1_file {input.emb1} \
        -p adata1_file {input.adata1} \
        -p adata2_file {input.adata2} \
        -p metric_file {output.metric} \
        -p matching_file {output.matching} \
        -p method {wildcards.method} \
        {input.notebook} {params.notebook_result} \
        > {log} 2>&1
        """