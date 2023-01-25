r"""
Split one data into two parts so we know the ground truth
"""
import os


def filter_rules(rules):
    filtered_rules = []
    for rule in rules:
        if 'paste' in rule.lower() and 'stereo_mouse_embryo' in rule.lower():
            continue
        filtered_rules.append(rule)
    return filtered_rules


rule summarize:
    input:
        metrics = filter_rules(expand(
            "split/{dataset}/cells:{cells}/seed:{seed}/split_ratio:{split_ratio}/{method}/metrics_all.yaml",
            cells=0,
            dataset=config['dataset'],
            split_ratio=config['split_ratio'],
            seed=[*config['seed']], 
            method=config['method_split'],
        )),
    output:
        "split/split_data.csv"
    params:
        pattern=lambda wildcards: "split/{dataset}/cells:{cells}/seed:{seed}/split_ratio:{split_ratio}/{method}/metrics_all.yaml"
    threads: 1
    script:
        "scripts/summarize.py"