import argparse
import toml
from omegaconf import OmegaConf, DictConfig
from hydra.utils import instantiate

import torch
import lightning.pytorch as pl
from lightning.pytorch.callbacks import LearningRateMonitor
from lightning.pytorch.loggers import WandbLogger

from duwu.utils import get_duwu_logger, instantiate_any
from duwu.loader import load_all


if __name__ == "__main__":

    print(f"PyTorch version: {torch.__version__}")

    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", type=str, nargs="+", default=None)
    args = parser.parse_args()

    configs = []
    toml_configs = []
    for config in args.configs:
        if config.endswith(".yaml"):
            conf = OmegaConf.load(config)
            configs.append(conf)
        else:
            conf = toml.load(config)
            toml_configs.append(conf)
    config = OmegaConf.merge(*configs)
    config = OmegaConf.merge(config, *toml_configs)

    logger = get_duwu_logger()
    data_module, trainer_wrapper = load_all(config)

    if config.get("unet_gradient_checkpointing", False):
        trainer_wrapper.unet.enable_gradient_checkpointing()

    # Configure lightning trainer

    lightning_config = {
        "accelerator": "gpu",
        "precision": "16-true",
        "devices": 1,
        "fast_dev_run": True,
        "deterministic": True,
        # This wrapper would further subsample the index of custom sampler
        # We do not use it because each node has different data
        # https://lightning.ai/docs/fabric/stable/_modules/lightning/fabric/utilities/distributed.html  # noqa
        "use_distributed_sampler": False,
        "callbacks": [],
        "logger": [],
        "plugins": [],
    }
    if "lightning_config" in config:
        lightning_config.update(instantiate_any(config["lightning_config"]))

    # Configure callbacks
    lightning_config["callbacks"].append(LearningRateMonitor(logging_interval="step"))

    # Configure logger
    lightning_config["logger"].append(WandbLogger())
    trainer = pl.Trainer(**lightning_config)

    # can only get rank after trainer is defined
    if "seed" in config:
        pl.seed_everything(config.seed + trainer.global_rank)

    ckpt_path = config.get("resume_from_checkpoint", None)
    if isinstance(ckpt_path, DictConfig):
        ckpt_path = instantiate_any(ckpt_path)
    if ckpt_path is not None:
        logger.info(f"Resume from {ckpt_path}...")

    trainer.fit(trainer_wrapper, data_module, ckpt_path=ckpt_path)
