from typing import Any, NamedTuple

import torch
import torch.nn as nn

from duwu.loss.diffusion import DiffusionLoss, DiffusionLossAuxOutput
from duwu.loss.t_distributions import Exp, Cosh


class RectifiedFlowLoss(DiffusionLoss):

    def __init__(
        self,
        time_sampling_type: str = "uniform_time",
        time_sampling_kwargs: dict[str, Any] = {},
        rescale_image: bool = False,
        rescale_noise: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.target_type = "rectified_flow"
        self.time_sampling_type = time_sampling_type
        self.time_sampling_kwargs = time_sampling_kwargs
        self.rescale_image = rescale_image
        self.rescale_noise = rescale_noise

    def sample_timesteps_and_sigmas(self, ref_params: torch.Tensor):

        batch_size = ref_params.size(0)
        scheduler_sigma_max = self.scheduler.sigmas[0]
        max_time = scheduler_sigma_max / (1 + scheduler_sigma_max)

        if self.time_sampling_type == "uniform_timestep":
            return super().sample_timesteps_and_sigmas(ref_params)

        elif self.time_sampling_type == "uniform_time":

            time = torch.rand(batch_size, device=ref_params.device) * max_time
            sigmas = time / (1 - time).to(ref_params)
            # inverse_sigma_min = 1 / scheduler_sigma_max
            # sigmas = time / torch.maximum((1 - time), inverse_sigma_min)
            timesteps = self.sigma_to_timestep(sigmas)
            return timesteps, sigmas

        elif self.time_sampling_type == "exp":
            exp_dist = Exp(**self.time_sampling_kwargs)
            time = exp_dist.sample(bs=batch_size).to(ref_params) * max_time
            sigmas = time / (1 - time)
            timesteps = self.sigma_to_timestep(sigmas)
            return timesteps, sigmas

        elif self.time_sampling_type == "cosh":
            cosh_dist = Cosh(**self.time_sampling_kwargs)
            time = cosh_dist.sample(bs=batch_size).to(ref_params) * max_time
            sigmas = time / (1 - time)
            timesteps = self.sigma_to_timestep(sigmas)
            return timesteps, sigmas

        else:
            raise ValueError(
                f"Unsupported time sampling type: {self.time_sampling_type}"
            )

    def get_x0_and_noises(self, x: torch.Tensor):
        # [batch, 2, channels, height, width], get sample and noise
        if len(x.shape) == 5:
            noises = x[:, 1, ...]
            x = x[:, 0, ...]
        else:
            noises = torch.randn_like(x)
        if self.rescale_image:
            x = x / x.std([1, 2, 3], keepdim=True)
            x = x * 0.937
        if self.rescale_noise:
            noises = noises / noises.std([1, 2, 3], keepdim=True)
        return x, noises

    def forward(self, x: torch.Tensor, unet: nn.Module, **unet_kwargs):

        x, noises = self.get_x0_and_noises(x)
        timesteps, sigmas = self.sample_timesteps_and_sigmas(ref_params=x)
        while len(sigmas.shape) < len(x.shape):
            sigmas = sigmas.unsqueeze(-1)
        scales = 1 / (sigmas**2 + 1) ** 0.5

        noisy_latent = scales * (x + noises * sigmas)

        model_output = unet(
            noisy_latent,
            timesteps,
            **unet_kwargs,
        )[0]

        target = noises - x
        pred_x0, pred_eps = self.get_x0_eps_from_pred_with_sigmas(
            noisy_latent, model_output, sigmas
        )
        pred = pred_eps - pred_x0

        losses = self.loss(pred, target)
        if len(losses.shape) > 1:
            losses = losses.flatten(start_dim=1).mean(dim=1)

        aux_outputs = DiffusionLossAuxOutput(
            losses=losses,
            timesteps=timesteps,
            pred=pred,
            target=target,
            noisy_latent=noisy_latent,
        )
        return losses.mean(), aux_outputs

    def sigma_to_timestep(self, sigmas):
        """
        Taken from
        https://github.com/huggingface/diffusers/blob/v0.30.2/src/diffusers/schedulers/scheduling_euler_discrete.py#L422  # noqa``
        """
        # get log sigma
        log_sigmas = torch.log(sigmas.clamp(min=1e-10))
        # get log sigmas from small to large
        # remove 0 from sigmas
        log_scheduler_sigmas = (
            torch.log(self.scheduler.sigmas[:-1]).flip(0).to(log_sigmas)
        )

        # get distribution, size [log_sigmas.shape[0], log_scheduler_sigmas.shape[0]]
        dists = log_sigmas - log_scheduler_sigmas[:, None]
        # get sigmas range
        low_idx = (
            dists.ge(0)
            .cumsum(dim=0)
            .argmax(dim=0)
            .clamp(max=log_scheduler_sigmas.shape[0] - 2)
        )
        high_idx = low_idx + 1
        low = log_scheduler_sigmas[low_idx]
        high = log_scheduler_sigmas[high_idx]

        # interpolate sigmas
        w = (low - log_sigmas) / (low - high)
        w = torch.clamp(w, 0, 1)
        # transform interpolation to time range
        t = (1 - w) * low_idx + w * high_idx
        return t.view(sigmas.shape)


class NNWeightedRFLossAuxOutput(NamedTuple):

    losses: torch.Tensor
    rescaled_losses: torch.Tensor
    pred_losses: torch.Tensor
    loss_pred_losses: torch.Tensor
    timesteps: torch.Tensor
    pred: torch.Tensor
    target: torch.Tensor
    noisy_latent: torch.Tensor


class NNWeightedRFLoss(RectifiedFlowLoss):

    def __init__(
        self,
        loss_pred_module: nn.Module,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.loss_pred_module = loss_pred_module

    def forward(self, x: torch.Tensor, unet: nn.Module, **unet_kwargs):

        x, noises = self.get_x0_and_noises(x)
        timesteps, sigmas = self.sample_timesteps_and_sigmas(ref_params=x)
        while len(sigmas.shape) < len(x.shape):
            sigmas = sigmas.unsqueeze(-1)
        scales = 1 / (sigmas**2 + 1) ** 0.5

        noisy_latent = scales * (x + noises * sigmas)

        model_output = unet(
            noisy_latent,
            timesteps,
            **unet_kwargs,
        )[0]

        # Loss prediction
        target = noises - x
        pred_x0, pred_eps = self.get_x0_eps_from_pred_with_sigmas(
            noisy_latent, model_output, sigmas
        )
        pred = pred_eps - pred_x0
        rf_losses = self.loss(pred, target)
        if len(rf_losses.shape) > 1:
            rf_losses = rf_losses.flatten(start_dim=1).mean(dim=1)

        # Important, loss prediction module takes sigmas as input
        log_ls_pred = self.loss_pred_module(
            noisy_latent, sigmas.flatten(), **unet_kwargs
        ).flatten()
        log_ls = rf_losses.detach().log()
        ls_pred_loss = (log_ls - log_ls_pred).square()

        # need to use eval() and recompute if we have dropout or normalization
        pred_loss = log_ls_pred.detach().exp().clamp(min=1e-4)

        rescaled_losses = rf_losses / pred_loss
        losses = rescaled_losses + ls_pred_loss

        aux_outputs = NNWeightedRFLossAuxOutput(
            losses=rf_losses,
            rescaled_losses=rescaled_losses,
            pred_losses=pred_loss,
            loss_pred_losses=ls_pred_loss,
            timesteps=timesteps,
            pred=pred,
            target=target,
            noisy_latent=noisy_latent,
        )
        return losses.mean(), aux_outputs
