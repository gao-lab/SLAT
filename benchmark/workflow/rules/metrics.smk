r"""
Extra metrics for evaluating the performance
"""

rule egde_socre:
    input:
        graphs="{path}/SLAT/graph.pkl",
        matching="{path}/{method}/matching.csv",
        metrics="{path}/{method}/metrics.yaml",
        notebook="workflow/notebooks/edge_score.ipynb",
    output:
        metrics_all="{path}/{method}/metrics_all.yaml",
    params:
        notebook_result="{path}/{method}/edge_score.ipynb",
    log:
        "{path}/{method}/edge_score.log",
    threads:4
    shell:
        """
        papermill \
        -p graphs_file {input.graphs} \
        -p metrics_input {input.metrics} \
        -p matching_file {input.matching} \
        -p metrics_output {output.metrics_all} \
        {input.notebook} {params.notebook_result} \
        > {log} 2>&1
        """