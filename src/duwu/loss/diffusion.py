from typing import NamedTuple

import torch
import torch.nn as nn

from diffusers import EulerDiscreteScheduler


class DiffusionLossAuxOutput(NamedTuple):

    losses: torch.Tensor
    timesteps: torch.Tensor
    pred: torch.Tensor
    target: torch.Tensor
    noisy_latent: torch.Tensor


class DiffusionLoss(nn.Module):

    def __init__(
        self,
        scheduler: EulerDiscreteScheduler,
        # TODO: define weighting class instead
        use_snr_weight: bool = False,
        min_snr_gamma: float = 5.0,
        use_debiased_estimation: bool = False,
        prediction_type: str | None = None,
        target_type: str | None = None,
        loss: nn.Module = nn.MSELoss(reduction="none"),
    ):
        super().__init__()
        self.scheduler = scheduler
        self.prepare_scheduler_for_custom_training()
        self.use_snr_weight = use_snr_weight
        self.min_snr_gamma = min_snr_gamma
        self.use_debiased_estimation = use_debiased_estimation
        self.prediction_type = prediction_type or self.scheduler.config.prediction_type
        self.target_type = target_type or self.scheduler.config.prediction_type
        self.loss = loss
        self.n_diffusion_time_steps = self.scheduler.config.num_train_timesteps

    def prepare_scheduler_for_custom_training(self):
        if hasattr(self.scheduler, "all_snr"):
            return
        alphas_cumprod = self.scheduler.alphas_cumprod
        sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
        alpha = sqrt_alphas_cumprod
        sigma = sqrt_one_minus_alphas_cumprod
        all_snr = (alpha / sigma) ** 2
        self.scheduler.all_snr = all_snr

    def get_sigmas_for_timesteps(
        self, timesteps: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # See also the `index_for_timestep` of the scheduler
        scheduler_timesteps = self.scheduler.timesteps.to(device=timesteps.device)
        step_indices = [(scheduler_timesteps == t).nonzero().item() for t in timesteps]
        # scale is `sqrt_alpha_prod`
        # sigma is `sqrt((1-alphas_cumprod)/alphas_cumprod)`, and flip
        sigmas = self.scheduler.sigmas[step_indices].flatten()
        return sigmas

    def sample_timesteps_and_sigmas(self, ref_params: torch.Tensor):
        batch_size = ref_params.size(0)
        min_timestep = 0
        max_timestep = self.scheduler.config.num_train_timesteps
        timesteps = torch.randint(
            min_timestep, max_timestep, (batch_size,), device=ref_params.device
        )
        sigmas = self.get_sigmas_for_timesteps(timesteps).to(ref_params)
        return timesteps, sigmas

    def get_noise_noisy_latents_and_timesteps(self, latents: torch.Tensor):
        noises = torch.randn_like(latents)
        timesteps, sigmas = self.sample_timesteps_and_sigmas(ref_params=latents)
        while len(sigmas.shape) < len(latents.shape):
            sigmas = sigmas.unsqueeze(-1)
        scales = 1 / (sigmas**2 + 1) ** 0.5
        # Diffusion Forward process
        noisy_samples = latents + noises * sigmas
        return noisy_samples * scales, noises, timesteps

    def get_target(
        self, x0: torch.Tensor, noise: torch.Tensor, timesteps: torch.Tensor
    ):
        if self.target_type == "epsilon":
            return noise
        elif self.target_type == "v_prediction":
            return self.scheduler.get_velocity(x0, noise, timesteps)
        elif self.target_type == "sample":
            return x0
        # rectified flow https://arxiv.org/abs/2209.03003
        # o-prediction is the opposite of this https://arxiv.org/abs/2303.00848
        elif self.target_type == "rectified_flow":
            return noise - x0
        else:
            raise ValueError(f"Unsupported target type {self.target_type}")

    def get_x0_eps_from_pred_with_sigmas(
        self, xt: torch.Tensor, model_output: torch.Tensor, sigmas: torch.Tensor
    ):
        """
        xt: Tensor of shape [batch_size, channels, height, width]
        model_output: Tensor of shape [batch_size, channels, height, width]
        sigmas: Tensor of shape [batch_size]
        """
        while len(sigmas.shape) < len(xt.shape):
            sigmas = sigmas.unsqueeze(-1)
        scales = 1 / (sigmas**2 + 1) ** 0.5
        if self.prediction_type == "sample":
            x0 = model_output
            eps = (xt / scales - x0) / sigmas
        elif self.prediction_type == "epsilon":
            eps = model_output
            x0 = xt / scales - sigmas * eps
        elif self.prediction_type == "v_prediction":
            x0 = scales * (xt - sigmas * model_output)
            eps = (xt / scales - x0) / sigmas
        elif self.prediction_type == "rectified_flow":
            x0 = (xt / scales - sigmas * model_output) / (1 + sigmas)
            eps = (xt / scales + model_output) / (1 + sigmas)
        else:
            raise ValueError(f"Unsupported prediction type {self.prediction_type}")
        return x0, eps

    def get_x0_eps_from_pred(
        self, xt: torch.Tensor, model_output: torch.Tensor, timesteps: torch.Tensor
    ):
        sigmas = self.get_sigmas_for_timesteps(timesteps).to(xt)
        return self.get_x0_eps_from_pred_with_sigmas(xt, model_output, sigmas)

    def get_prediction_for_training(
        self, xt: torch.Tensor, model_output: torch.Tensor, timesteps: torch.Tensor
    ):
        if self.prediction_type == self.target_type:
            return model_output
        x0, eps = self.get_x0_eps_from_pred(xt, model_output, timesteps)
        return self.get_target(x0, eps, timesteps)

    def apply_snr_weight(self, loss: torch.Tensor, timesteps: torch.Tensor):

        assert self.prediction_type == self.target_type
        assert self.prediction_type in ["epsilon", "v_prediction"]

        snr = torch.stack([self.scheduler.all_snr[t] for t in timesteps])
        min_snr_gamma = torch.minimum(snr, torch.full_like(snr, self.min_snr_gamma))
        if self.prediction_type == "v_prediction":
            snr_weight = torch.div(min_snr_gamma, snr + 1).float().to(loss.device)
        else:
            snr_weight = torch.div(min_snr_gamma, snr).float().to(loss.device)
        loss = loss * snr_weight
        return loss

    def apply_debiased_estimation(self, loss: torch.Tensor, timesteps: torch.Tensor):

        assert self.prediction_type == self.target_type == "epsilon"

        snr_t = torch.stack(
            [self.scheduler.all_snr[t] for t in timesteps]
        )  # batch_size
        snr_t = torch.minimum(
            snr_t, torch.ones_like(snr_t) * 1000
        )  # if timestep is 0, snr_t is inf, so limit it to 1000
        weight = (1 / torch.sqrt(snr_t)).to(loss.device)
        loss = weight * loss
        return loss

    def forward(self, x: torch.Tensor, unet: nn.Module, **unet_kwargs):

        noisy_latent, noise, timesteps = self.get_noise_noisy_latents_and_timesteps(x)
        model_output = unet(
            noisy_latent,
            timesteps,
            **unet_kwargs,
        )[0]
        pred = self.get_prediction_for_training(x, model_output, timesteps)
        target = self.get_target(x, noise, timesteps)
        losses = self.loss(pred, target)
        if len(losses.shape) > 1:
            losses = losses.flatten(start_dim=1).mean(dim=1)
        if self.use_snr_weight:
            losses = self.apply_snr_weight(losses, timesteps)
        if self.use_debiased_estimation:
            losses = self.apply_debiased_estimation(losses, timesteps)
        aux_outputs = DiffusionLossAuxOutput(
            losses=losses,
            timesteps=timesteps,
            pred=pred,
            target=target,
            noisy_latent=noisy_latent,
        )
        return losses.mean(), aux_outputs
