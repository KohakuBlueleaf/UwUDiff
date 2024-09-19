from typing import Any
from omegaconf import DictConfig

import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_sch
from warmup_scheduler import GradualWarmupScheduler

from duwu.trainer.trainer import DMTrainer


class NNWeightedLossTrainer(DMTrainer):

    def __init__(
        self,
        model_config: DictConfig | dict,
        te_use_normed_ctx: bool = False,
        vae_std: float | None = None,
        vae_mean: float | None = None,
        lycoris_model: nn.Module | None = None,
        *args,
        name: str = "",
        lr: float = 1e-5,
        optimizer: type[optim.Optimizer] = optim.AdamW,
        opt_config: dict[str, Any] = {
            "weight_decay": 0.01,
            "betas": (0.9, 0.999),
        },
        lr_scheduler: type[lr_sch.LRScheduler] | None = lr_sch.CosineAnnealingLR,
        lr_scheduler_config: dict[str, Any] = {
            "T_max": 100_000,
            "eta_min": 1e-7,
        },
        use_warm_up: bool = True,
        warm_up_period: int = 1000,
        loss_config: DictConfig | dict | None = None,
        loss_opt_config: dict[str, Any] = {
            "lr": 1e-3,
            "weight_decay": 0,
            "betas": (0.9, 0.999),
        },
    ):
        super(NNWeightedLossTrainer, self).__init__(
            model_config=model_config,
            te_use_normed_ctx=te_use_normed_ctx,
            vae_std=vae_std,
            vae_mean=vae_mean,
            lycoris_model=lycoris_model,
            *args,
            name=name,
            lr=lr,
            optimizer=optimizer,
            opt_config=opt_config,
            lr_scheduler=lr_scheduler,
            lr_scheduler_config=lr_scheduler_config,
            use_warm_up=use_warm_up,
            warm_up_period=warm_up_period,
            loss_config=loss_config,
        )
        self.loss.requires_grad_(True).train()
        self.loss_params = self.loss.parameters()
        self.loss_opt_config = loss_opt_config

    def configure_optimizers(self):

        optimizer = self.optimizer(
            [
                {"params": self.loss_params, **self.loss_opt_config},
                {"params": self.train_params, "lr": self.lr, **self.opt_config},
            ]
        )

        lr_sch = None
        if self.lr_sch is not None:
            lr_sch = self.lr_sch(optimizer, **self.lr_sch_config)

        # https://github.com/ildoonet/pytorch-gradual-warmup-lr/issues/27
        if self.use_warm_up:
            lr_scheduler = GradualWarmupScheduler(
                optimizer, 1, self.warm_up_period, lr_sch
            )
        else:
            lr_scheduler = lr_sch

        if lr_scheduler is None:
            return optimizer
        else:
            return {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": lr_scheduler, "interval": "step"},
            }
