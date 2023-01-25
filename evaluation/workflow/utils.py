r"""
Utility functions for snakemake files
"""
from functools import reduce
from operator import add
from pathlib import Path

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


def target_directories(config,
                       sample:int = 0,
                       hyperparameter:str = 'SLAT_hyperparam'):
    r"""
    Resolve snakemake config to str
    
    Parameters
    ----------
    config
        snakemake config
    sample
        number of rules to sample
    hyperparameter
        which hyperparameter to use
    """
    np.random.seed(seed=0)
    seed2range(config)

    dataset = config["dataset"].keys()
    
    data_conf = config["datasize"] or {}
    data_conf = expand(
        conf_expand_pattern(data_conf, placeholder="default"),
        **data_conf
    )
    
    hyperparam_conf = config[hyperparameter] or {}
    hyperparam_conf = expand(
        conf_expand_pattern(hyperparam_conf, placeholder="default"),
        **hyperparam_conf
    )
    
    seed = config["seed"] 
    
    pool = expand(
        "results/{dataset}/{data_conf}/{hyperparam_conf}/seed:{seed}",
        dataset=dataset,
        data_conf=data_conf,
        hyperparam_conf=hyperparam_conf,
        seed=seed
    )
    return np.random.choice(pool, size=sample, replace=False) if sample > 0 else pool


def target_files(directories):
    r"""
    Check if tagert file exist, other return them
    
    Parameters
    ----------
    directories
        list of dirs
    """
    def per_directory(directory):
        directory = Path(directory)
        if (directory / "metrics.yaml").exists():
            return [directory / "metrics.yaml"]
        return [
            directory / "metrics.yaml"
        ]

    return reduce(add, map(per_directory, directories))