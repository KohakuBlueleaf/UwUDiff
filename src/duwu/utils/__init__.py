import os
import sys
import importlib
import omegaconf
import logging
from typing import Literal
from pathlib import Path
from inspect import isfunction
from random import shuffle

import torch
import torch.nn as nn
from lightning.pytorch.utilities import rank_zero_only
from hydra.utils import instantiate


def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def instantiate_class(obj):
    if isinstance(obj, omegaconf.DictConfig):
        obj = omegaconf.OmegaConf.to_container(obj, resolve=True)
    if isinstance(obj, dict) and "class" in obj:
        obj_factory = instantiate_class(obj.pop("class"))
        if "factory" in obj:
            obj_factory = getattr(obj_factory, obj.pop("factory"))
        if "args" in obj or "kwargs" in obj:
            return obj_factory(*obj.get("args", []), **obj.get("kwargs", {}))
        else:
            return obj_factory(**obj)
    if isinstance(obj, str):
        return get_obj_from_str(obj)
    return obj


def instantiate_any(obj):
    '''
    Instantiate utils for both hydra format and custom format
    '''
    if (
        isinstance(obj, dict) or isinstance(obj, omegaconf.DictConfig)
    ) and "_target_" in obj:
        return instantiate(obj)
    else:
        return instantiate_class(obj)


def exists(val):
    return val is not None


def uniq(arr):
    return {el: True for el in arr}.keys()


def default(val, d):
    if val is not None:
        return val
    return d() if isfunction(d) else d


def zero_module(module: nn.Module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


def random_choice(
    x: torch.Tensor,
    num: int,
):
    rand_x = list(x)
    shuffle(rand_x)

    return torch.stack(rand_x[:num])


def count_params(model, verbose=False):
    total_params = sum(p.numel() for p in model.parameters())
    if verbose:
        print(f"{model.__class__.__name__} has {total_params * 1.e-6:.2f} M params.")
    return total_params


def remove_none(list_x):
    return [i for i in list_x if i is not None]


def balance_sharding_index(total, shards):
    prev = 0
    for i in range(shards):
        this_shard = total // shards
        yield prev, this_shard
        shards -= 1
        total -= this_shard
        prev += this_shard


def balance_sharding(datas, shards):
    total = len(datas)
    for prev, this_shard in balance_sharding_index(total, shards):
        yield datas[prev : prev + this_shard]


def balance_sharding_max_size(datas, max_size):
    total = len(datas)
    shards = total // max_size + int(bool(total % max_size))
    return balance_sharding(datas, shards)


def truncate_or_pad_to_length(
    list_x: list,
    target_length: int,
    padding_mode: Literal["repeat_last", "cycling", "uniform_expansion"],
):
    if len(list_x) > target_length:
        return list_x[:target_length]
    if len(list_x) == target_length:
        return list_x
    if padding_mode == "repeat_last":
        return repeat_last(list_x, target_length)
    if padding_mode == "cycling":
        return cycling(list_x, target_length)
    if padding_mode == "uniform_expansion":
        return uniform_expansion(list_x, target_length)


def repeat_last(list_x, target_length):
    return list_x + [list_x[-1]] * (target_length - len(list_x))


def cycling(list_x, target_length):
    return (
        list_x * (target_length // len(list_x)) + list_x[: target_length % len(list_x)]
    )


def uniform_expansion(list_x, target_length):
    result = []
    for idx, ref in enumerate(
        balance_sharding(list(range(target_length)), len(list_x))
    ):
        result.extend([list_x[idx]] * len(ref))
    return result


def get_duwu_logger() -> logging.Logger:
    """
    Get the logger for duwu training.

    Returns
    -------
    logging.Logger
        The logger instance for duwu training.
    """
    return logging.getLogger("duwu")


@rank_zero_only
def setup_duwu_logger(level: int = logging.DEBUG) -> logging.Logger:
    """
    Set up the duwu logger with a specified logging level.

    Parameters
    ----------
    level
        The logging level, by default logging.DEBUG.
    """
    logger = get_duwu_logger()
    logger.setLevel(level)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


def get_images_recursively(folder_path: str) -> list[str]:
    """
    Get all images recursively from a folder.

    Parameters
    ----------
    folder_path
        The path to the folder.

    Returns
    -------
    list[str]
        The list of paths to images found recursively in the folder.

    Raises
    ------
    ValueError
        If the provided path does not exist.
    """
    if not os.path.exists(folder_path):
        raise ValueError(f"The path {folder_path} does not exist.")

    allowed_patterns = [
        "*.[Pp][Nn][Gg]",
        "*.[Jj][Pp][Gg]",
        "*.[Jj][Pp][Ee][Gg]",
        "*.[Ww][Ee][Bb][Pp]",
        "*.[Gg][Ii][Ff]",
    ]

    image_path_list = [
        str(path)
        for pattern in allowed_patterns
        for path in Path(folder_path).rglob(pattern)
    ]

    return image_path_list
