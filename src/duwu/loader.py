from typing import Any
import omegaconf
from hydra.utils import instantiate
from dataclasses import dataclass

import torch
import torch.nn as nn
from lightning import LightningDataModule, LightningModule

from duwu.utils import instantiate_any


@dataclass
class ModelLoadingConfig:
    ckpt_path: str | None = None
    state_dict_key: str | None = None
    state_dict_prefix: str | None = None
    precision: str | None = None
    device: str | None = None
    to_compile: bool = False
    to_freeze: bool = False


def extract_state_dict(state_dict: dict[str, Any], key: str | None, prefix: str | None):
    if key is not None:
        state_dict = state_dict[key]
    if prefix is None:
        return state_dict
    extracted_state_dict = {}
    for key, params in state_dict.items():
        if key.startswith(prefix):
            extracted_state_dict[key[len(prefix) :]] = params
    return extracted_state_dict


def prepare_model(model: nn.Module, model_loading_config: ModelLoadingConfig):
    if model_loading_config.ckpt_path is not None:
        state_dict = torch.load(
            model_loading_config.ckpt_path, map_location=lambda storage, loc: storage
        )
        state_dict = extract_state_dict(
            state_dict,
            model_loading_config.state_dict_key,
            model_loading_config.state_dict_prefix,
        )
        model.load_state_dict(state_dict)
    if model_loading_config.precision is not None:
        model = model.to(eval(model_loading_config.precision))
    if model_loading_config.device is not None:
        model = model.to(model_loading_config.device)
    if model_loading_config.to_compile:
        model = torch.compile(model)
    if model_loading_config.to_freeze:
        model.requires_grad_(False).eval()
    return model


def load_any(obj):
    load_config = None
    if isinstance(obj, dict) or isinstance(obj, omegaconf.DictConfig):
        if "_load_config_" in obj:
            load_config = obj.pop("_load_config_")
            load_config = ModelLoadingConfig(**load_config)
    obj = instantiate_any(obj)
    if load_config is not None:
        obj = prepare_model(obj, load_config)
    return obj


def load_all(
    conf: omegaconf.DictConfig | dict,
    trainer: LightningModule | None = None,
    data_module: LightningDataModule | None = None,
):
    trainer = trainer or instantiate_any(conf.pop("trainer"))
    data_module = data_module or instantiate_any(conf.pop("data"))
    # TODO: we need a better way to handle this
    data_module.set_tokenizers(trainer.te.tokenizers)
    return data_module, trainer


if __name__ == "__main__":
    from objprint import objprint

    conf = omegaconf.OmegaConf.load("./configs/demo.yaml")
    objprint(conf)
    dataset, trainer = load_all(conf)
    objprint(trainer.hparams)
    print(dataset)
