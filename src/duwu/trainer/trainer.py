import os
import toml
from typing import Any
from collections.abc import Iterator
from omegaconf import DictConfig
from hydra.utils import instantiate

import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_sch
import lightning.pytorch as pl
from warmup_scheduler import GradualWarmupScheduler
from lycoris import LycorisNetwork, create_lycoris

from duwu.utils import instantiate_any
from duwu.utils.aggregation import aggregate_embeddings
from duwu.loader import load_any
from duwu.modules.text_encoders import BaseTextEncoder
from duwu.modules.attn_masks import convert_mask_dict, convert_layer_attn_meta_dict


class BaseTrainer(pl.LightningModule):
    def __init__(
        self,
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
        **kwargs,
    ):
        super().__init__()
        self.name = name
        self.train_params: Iterator[nn.Parameter] = None
        self.optimizer = instantiate_any(optimizer)
        self.opt_config = opt_config
        self.lr = lr
        self.lr_sch = instantiate_any(lr_scheduler)
        self.lr_sch_config = lr_scheduler_config
        self.use_warm_up = use_warm_up
        self.warm_up_period = warm_up_period

    def configure_optimizers(self):
        assert self.train_params is not None
        optimizer = self.optimizer(self.train_params, lr=self.lr, **self.opt_config)

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

    def on_train_start(self):
        """
        A hack to fix the load_state_dict issue of GradualWarmupScheduler
        https://github.com/ildoonet/pytorch-gradual-warmup-lr/issues/27

        Otherwise, the fact that optimizer state is repeatedly loaded in
        ``after_scheduler`` causes the checkpoint to become twice larger
        if we resume from a checkpoint.

        Note that ``on_fit_start`` is run before the resumed model is loaded,
        so we use ``on_train_start`` here.
        """
        if self.lr_schedulers() is not None:
            if isinstance(self.lr_schedulers(), GradualWarmupScheduler):
                self.lr_schedulers().after_scheduler.optimizer = (
                    self.optimizers().optimizer
                )


class DMTrainer(BaseTrainer):
    def __init__(
        self,
        model_config: DictConfig | dict,
        te_use_normed_ctx: bool = False,
        vae_std: float | None = None,
        vae_mean: float | None = None,
        lycoris_config: dict | str | None = None,
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
        # This is not compatible with dynamic batch size
        use_flex_attention_for_region: bool = False,
        use_flex_attention_for_layer: bool = False,
    ):
        super(DMTrainer, self).__init__(
            *args,
            name=name,
            lr=lr,
            optimizer=optimizer,
            opt_config=opt_config,
            lr_scheduler=lr_scheduler,
            lr_scheduler_config=lr_scheduler_config,
            use_warm_up=use_warm_up,
            warm_up_period=warm_up_period,
        )
        # Do not save hyperparameters to logger
        self.save_hyperparameters(ignore=["lycoris_model"], logger=False)

        unet = load_any(model_config["unet"])
        te = load_any(model_config["te"])
        vae = load_any(model_config["vae"])

        self.unet = unet
        self.te = te
        self.vae = vae

        self.te_use_normed_ctx = te_use_normed_ctx
        self.vae_std = vae_std
        self.vae_mean = vae_mean or 0
        if self.vae_std is None and self.vae is not None:
            self.vae_std = 1 / self.vae.config.scaling_factor

        if isinstance(lycoris_config, str):
            lycoris_config = toml.load(lycoris_config)

        if lycoris_config is not None:
            LycorisNetwork.apply_preset(lycoris_config["preset"])
            lycoris_model = create_lycoris(unet, **lycoris_config["config"])
            lycoris_model.apply_to()
        else:
            lycoris_model = None

        self.lycoris_model = lycoris_model

        self.register_buffer("ema_loss", torch.tensor(0.0))
        self.ema_decay = 0.99

        if lycoris_model is not None:
            self.lycoris_model.train()
            self.unet.requires_grad_(False)
            self.train_params = self.lycoris_model.parameters()
        else:
            self.unet.requires_grad_(True).train()
            self.train_params = self.unet.parameters()

        if loss_config is None:
            from duwu.loss import DiffusionLoss
            from diffusers import EulerDiscreteScheduler

            scheduler = EulerDiscreteScheduler.from_pretrained(
                "stabilityai/stable-diffusion-xl-base-1.0", subfolder="scheduler"
            )
            self.loss = DiffusionLoss(scheduler)
        else:
            self.loss = instantiate_any(loss_config)

        self.n_diffusion_time_steps = self.loss.n_diffusion_time_steps

        # for region and layer attention
        self.use_flex_attention_for_region = use_flex_attention_for_region
        self.use_flex_attention_for_layer = use_flex_attention_for_layer
        # https://github.com/pytorch/pytorch/issues/104674
        if self.use_flex_attention_for_region or self.use_flex_attention_for_layer:
            torch._dynamo.config.optimize_ddp = False

    def merge_lycoris(self):
        # For inference, load ckpt than merge lycoris back to unet
        self.lycoris_model.restore()
        self.lycoris_model.merge_to()

    def on_train_epoch_end(self) -> None:
        if self.lycoris_model is not None:
            dir = "./lycoris_weight"
            epoch = self.current_epoch
            if self._trainer is not None:
                trainer = self._trainer
                epoch = trainer.current_epoch
                if len(trainer.loggers) > 0:
                    if trainer.loggers[0].save_dir is not None:
                        save_dir = trainer.loggers[0].save_dir
                    else:
                        save_dir = trainer.default_root_dir
                    name = trainer.loggers[0].name
                    version = trainer.loggers[0].version
                    version = (
                        version if isinstance(version, str) else f"version_{version}"
                    )
                    dir = os.path.join(save_dir, str(name), version, "lycoris_weight")
                else:
                    # if no loggers, use default_root_dir
                    dir = os.path.join(trainer.default_root_dir, "lycoris_weight")
            os.makedirs(dir, exist_ok=True)
            model_weight = {
                k: v for k, v in self.unet.named_parameters() if v.requires_grad
            }
            lycoris_weight = self.lycoris_model.state_dict() | model_weight
            torch.save(lycoris_weight, os.path.join(dir, f"epoch={epoch}.pt"))

    def on_save_checkpoint(self, checkpoint):
        # See https://github.com/Lightning-AI/pytorch-lightning/issues/18060
        fit_loop_dict = checkpoint["loops"]["fit_loop"]
        batch_progress_dict = fit_loop_dict["epoch_loop.batch_progress"]
        batch_progress_dict["current"]["completed"] = batch_progress_dict["current"][
            "processed"
        ]
        batch_progress_dict["total"]["completed"] = batch_progress_dict["total"][
            "processed"
        ]
        # Note that `_batches_that_stepped` is used by wandb logger and should
        # be set to total batch instead of current epoch batch
        fit_loop_dict["epoch_loop.state_dict"]["_batches_that_stepped"] = (
            batch_progress_dict["total"]["processed"]
        )

    def get_latent_and_conditioning(self, batch):

        x, captions, tokenizer_outputs, added_cond, cross_attn_kwargs = batch
        # print(type(x), type(captions), type(tokenizer_outputs), type(added_cond))

        with torch.no_grad():

            # Encode latent
            if self.vae is not None:
                latent_dist = self.vae.encode(x).latent_dist
                x = latent_dist.sample()
                x = (x - self.vae_mean) / self.vae_std

            # Encode text conditioning
            if isinstance(self.te, BaseTextEncoder):
                embedding, normed_embedding, pooled_embedding, attn_mask = self.te(
                    tokenizer_outputs
                )
            else:
                normed_embedding, pooled_embedding, *embeddings = self.te(
                    **tokenizer_outputs[0], return_dict=False, output_hidden_states=True
                )
                embedding = embeddings[-1][-1]
            if self.te_use_normed_ctx:
                ctx = normed_embedding
            else:
                ctx = embedding

        # Convert masks
        if "region_mask_dict" in cross_attn_kwargs:
            if cross_attn_kwargs["region_mask_dict"] is None:
                cross_attn_kwargs.pop("region_mask_dict")
            else:
                sequence_length = ctx.size(1)
                cross_attn_kwargs["region_mask_dict"] = convert_mask_dict(
                    cross_attn_kwargs["region_mask_dict"],
                    sequence_length=sequence_length,
                    encoder_attn_mask=attn_mask,
                    use_flex_attention=self.use_flex_attention_for_region,
                )

        if "layer_attn_meta_dict" in cross_attn_kwargs:
            if cross_attn_kwargs["layer_attn_meta_dict"] is None:
                cross_attn_kwargs.pop("layer_attn_meta_dict")
            else:
                cross_attn_kwargs["layer_attn_meta_dict"] = (
                    convert_layer_attn_meta_dict(
                        cross_attn_kwargs["layer_attn_meta_dict"],
                        use_flex_attention=self.use_flex_attention_for_layer,
                    )
                )

        # Aggregate embeddings
        if "region_mask_dict" in cross_attn_kwargs:
            n_elements_per_image = cross_attn_kwargs["n_elements_per_image"]
            ctx = aggregate_embeddings(ctx, n_elements_per_image, mode="concat")
            if attn_mask is not None:
                attn_mask = aggregate_embeddings(attn_mask, n_elements_per_image)
            if "layer_attn_meta_dict" in cross_attn_kwargs:
                ctx = ctx.repeat_interleave(n_elements_per_image, dim=0)
                if attn_mask is not None:
                    attn_mask = attn_mask.repeat_interleave(n_elements_per_image, dim=0)
            elif pooled_embedding is not None:
                pooled_embedding = aggregate_embeddings(
                    pooled_embedding, n_elements_per_image, mode="first"
                )
        added_cond["text_embeds"] = pooled_embedding
        # print(f"rank {self.global_rank}: {ctx.size()}")
        return x, ctx, attn_mask, added_cond, cross_attn_kwargs

    def training_step(self, batch, idx):
        x, ctx, attn_mask, added_cond, cross_attn_kwargs = (
            self.get_latent_and_conditioning(batch)
        )
        loss, aux_output = self.loss(
            x,
            self.unet,
            encoder_hidden_states=ctx,
            encoder_attention_mask=attn_mask,
            added_cond_kwargs=added_cond,
            cross_attention_kwargs=cross_attn_kwargs,
        )

        ema_decay = min(self.global_step / (10 + self.global_step), self.ema_decay)
        # We would get memory leak if we don't detach here
        self.ema_loss = ema_decay * self.ema_loss + (1 - ema_decay) * loss.detach()

        if self._trainer is not None:
            self.log(
                "train/loss",
                loss.item(),
                on_step=True,
                logger=True,
            )
            self.log(
                "train/ema_loss",
                self.ema_loss.item(),
                on_step=True,
                logger=True,
                prog_bar=True,
            )
        return {"loss": loss, "aux_output": aux_output}

    def validation_step(self, batch, idx):
        x, ctx, attn_mask, added_cond, cross_attn_kwargs = (
            self.get_latent_and_conditioning(batch)
        )
        loss, aux_output = self.loss(
            x,
            self.unet,
            encoder_hidden_states=ctx,
            encoder_attention_mask=attn_mask,
            added_cond_kwargs=added_cond,
            cross_attention_kwargs=cross_attn_kwargs,
        )
        self.log(
            "val/loss",
            loss.item(),
            logger=True,
            prog_bar=True,
            sync_dist=True,
            on_step=False,
            on_epoch=True,
            batch_size=batch[0].shape[0],
        )
        return loss, aux_output
