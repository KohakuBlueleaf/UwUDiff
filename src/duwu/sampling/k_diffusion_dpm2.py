from tqdm import trange

import torch

from k_diffusion.sampling import to_d


@torch.no_grad()
@torch.inference_mode()
def sample_dpm2(
    model,
    x,
    sigmas,
    extra_args=None,
    disable=None,
    s_churn=0.0,
    s_tmin=0.0,
    s_tmax=float("inf"),
    s_noise=1.0,
    single_call: bool = False,
    image_to_noise: bool = False,  # unused
):
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    d_cached = None
    for i in trange(len(sigmas) - 1, disable=disable):
        gamma = (
            min(s_churn / (len(sigmas) - 1), 2**0.5 - 1)
            if s_tmin <= sigmas[i] <= s_tmax
            else 0.0
        )
        eps = torch.randn_like(x) * s_noise
        sigma_hat = sigmas[i] * (gamma + 1)
        if gamma > 0:
            x = x + eps * (sigma_hat**2 - sigmas[i] ** 2) ** 0.5
        if sigmas[i + 1] == 0:
            # Euler method
            denoised, _ = model(x, sigma_hat * s_in, **extra_args)
            d = to_d(x, sigma_hat, denoised)
            dt = sigmas[i + 1] - sigma_hat
            x = x + d * dt
        else:
            if single_call and d_cached is not None:
                d = d_cached
            else:
                denoised, _ = model(x, sigma_hat * s_in, **extra_args)
                d = to_d(x, sigma_hat, denoised)
            # DPM-Solver-2
            sigma_mid = sigma_hat.log().lerp(sigmas[i + 1].log(), 0.5).exp()
            dt_1 = sigma_mid - sigma_hat
            dt_2 = sigmas[i + 1] - sigma_hat
            x_2 = x + d * dt_1
            denoised_2, _ = model(x_2, sigma_mid * s_in, **extra_args)
            d_2 = to_d(x_2, sigma_mid, denoised_2)
            d_cached = d_2
            x = x + d_2 * dt_2
    return x


@torch.no_grad()
@torch.inference_mode()
def sample_dpm2_cfgpp(
    model,
    x,
    sigmas,
    extra_args=None,
    disable=None,
    s_churn=0.0,
    s_tmin=0.0,
    s_tmax=float("inf"),
    s_noise=1.0,
    single_call: bool = False,
    image_to_noise: bool = False,  # unused
):
    # TODO: cfg++ does not work with single-call
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    cfg_denoised_cached = None
    uncond_d_cached = None
    for i in trange(len(sigmas) - 1, disable=disable):
        gamma = (
            min(s_churn / (len(sigmas) - 1), 2**0.5 - 1)
            if s_tmin <= sigmas[i] <= s_tmax
            else 0.0
        )
        eps = torch.randn_like(x) * s_noise
        sigma_hat = sigmas[i] * (gamma + 1)
        if gamma > 0:
            x = x + eps * (sigma_hat**2 - sigmas[i] ** 2) ** 0.5
        if sigmas[i + 1] == 0:
            # Euler method
            cfg_denoised, _ = model(x, sigma_hat * s_in, **extra_args)
            x = cfg_denoised
        else:
            if single_call and cfg_denoised_cached is not None:
                cfg_denoised = cfg_denoised_cached
                uncond_d = uncond_d_cached
            else:
                cfg_denoised, uncond_denoised = model(x, sigma_hat * s_in, **extra_args)
                uncond_d = to_d(x, sigma_hat, uncond_denoised)
            # DPM-Solver-2
            sigma_mid = sigma_hat.log().lerp(sigmas[i + 1].log(), 0.5).exp()
            x_2 = cfg_denoised + uncond_d * sigma_mid
            cfg_denoised_2, uncond_denoised_2 = model(
                x_2, sigma_mid * s_in, **extra_args
            )
            uncond_d_2 = to_d(x_2, sigma_mid, uncond_denoised_2)
            x = cfg_denoised_2 + uncond_d_2 * sigmas[i + 1]
            cfg_denoised_cached = cfg_denoised_2
            uncond_d_cached = uncond_d_2
    return x
