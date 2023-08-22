r"""
Help functions for benchmark workflow.
"""
from functools import reduce
from operator import add
from pathlib import Path
import copy

import parse
import numpy as np


def conf_expand_pattern(conf, placeholder="null"):
    r"""
    expand the str by config, otherwise by default placeholder
    """
    expand_pattern = "-".join(f"{key}:{{{key}}}" for key in conf)       
    return expand_pattern if expand_pattern else placeholder


def expand(pattern, **wildcards):
    from snakemake.io import expand

    has_default_choices = False
    for val in wildcards.values():  # Sanity check
        if isinstance(val, dict):
            if "default" not in val or "choices" not in val:
                print(val)
                raise ValueError("Invalid default choices!")
            has_default_choices = True

    if not has_default_choices:
        return expand(pattern, **wildcards)

    expand_set = set()
    for key, val in wildcards.items():
        if isinstance(val, dict):
            wildcards_use = {key: val["choices"]}
            for other_key, other_val in wildcards.items():
                if other_key == key:
                    continue
                if isinstance(other_val, dict):
                    wildcards_use[other_key] = other_val["default"]
                else:
                    wildcards_use[other_key] = other_val
            expand_set = expand_set.union(expand(pattern, **wildcards_use))
    return list(expand_set)


def seed2range(config):
    for key, val in config.items():
        if isinstance(val, dict):
            seed2range(val)
        elif key.endswith("seed") and val != 0:
            config[key] = range(val)


def target_directories(config, sample:int = 0):
    r"""
    Resolve snakemake config to str
    
    Parameters
    ----------
    config
        snakemake config
    sample
        number of rules to sample
    fix_sample
        use fixed sample of rules
    """
    np.random.seed(seed=0)
    seed2range(config)
    
    def per_method_dataset(method_dataset):
        method = method_dataset[0]
        dataset = method_dataset[1]
        # print(method)
        # seed = 0 if method in ['Harmony', 'Seurat', 'PCA'] else config["seed"] # change seed strategy
        seed = config["seed"]
        
        data_conf = copy.deepcopy(config["datasize"]) or {}
        if dataset in ['visium_human_DLPFC']:
            data_conf['cells']['choices'] = list(filter(lambda x: int(x) < 3600, data_conf['cells']['choices'] ))
        if dataset in ['merfish_mouse_hypothalamic']:
                data_conf['cells']['choices'] = list(filter(lambda x: int(x) < 7000, data_conf['cells']['choices'] )) 
        if method in ['PASTE', 'PASTE2'] and 'stereo_mouse_embryo' in dataset:
            data_conf['cells']['choices'] = list(filter(lambda x: 0 < int(x) < 30001, data_conf['cells']['choices'] ))
        # if method == 'STAGATE' and dataset == 'stereo_mouse_embryo':
        #     data_conf['cells']['choices'] = list(filter(lambda x: 0 < int(x) < 2000000, data_conf['cells']['choices'] ))
        if method == 'Seuart' and dataset == 'stereo_mouse_embryo':
            data_conf['cells']['choices'] = list(filter(lambda x: 100 < int(x), data_conf['cells']['choices'] ))
        # print(f"{method}--{dataset}--{data_conf['cells']['choices']}")
        data_conf = expand(
            conf_expand_pattern(data_conf, placeholder="default"),
            **data_conf
        )
        # dataset = list(config["dataset"].keys())
        # if method in ['PASTE']:
        #     dataset.remove('stereo_mouse_embryo') # these methods can not run on such big dataset
        
        pool = expand(
            "results/{dataset}/{data_conf}/seed:{seed}/{method}/",
            dataset=dataset,
            data_conf=data_conf,
            method=method,
            seed=seed
        )
        # print(pool)
        return np.random.choice(pool, size=sample, replace=False) if sample > 0 else pool

    return reduce(add, map(per_method_dataset, ((x,y) for x in config["method"] for y in config["dataset"].keys())))


def target_files(directories):
    r"""
    Check if target file exist, other return them
    
    Parameters
    ----------
    directories
        list of dirs
    """
    def per_directory(directory):
        directory = Path(directory)
        if (directory / "metrics_all.yaml").exists():
            return [directory / "metrics_all.yaml"]
        return [
            directory / "metrics_all.yaml",
        ]

    return reduce(add, map(per_directory, directories))