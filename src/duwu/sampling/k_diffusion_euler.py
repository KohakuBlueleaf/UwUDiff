from tqdm import trange

import torch

from k_diffusion.sampling import to_d, get_ancestral_step, default_noise_sampler


@torch.no_grad()
@torch.inference_mode()
def sample_euler_ancestral(
    model,
    x,
    sigmas,
    extra_args=None,
    callback=None,
    disable=None,
    eta=1.0,
    s_noise=1.0,
    noise_sampler=None,
    image_to_noise: bool = False,
):
    """Ancestral sampling with Euler method steps."""
    extra_args = {} if extra_args is None else extra_args
    noise_sampler = default_noise_sampler(x) if noise_sampler is None else noise_sampler
    s_in = x.new_ones([x.shape[0]])
    for i in trange(len(sigmas) - 1, disable=disable):
        sigma_cond = sigmas[i + 1] if image_to_noise else sigmas[i]
        denoised, _ = model(
            x, sigmas[i] * s_in, sigma_cond=sigma_cond * s_in, **extra_args
        )
        sigma_down, sigma_up = get_ancestral_step(sigmas[i], sigmas[i + 1], eta=eta)
        if callback is not None:
            callback(
                {
                    "x": x,
                    "i": i,
                    "sigma": sigmas[i],
                    "sigma_hat": sigmas[i],
                    "denoised": denoised,
                }
            )
        d = to_d(x, sigmas[i], denoised)
        # Euler method
        dt = sigma_down - sigmas[i]
        x = x + d * dt
        if sigmas[i + 1] > 0:
            x = x + noise_sampler(sigmas[i], sigmas[i + 1]) * s_noise * sigma_up
    return x


@torch.no_grad()
@torch.inference_mode()
def sample_euler_ancestral_cfgpp(
    model,
    x,
    sigmas,
    extra_args=None,
    callback=None,
    disable=None,
    eta=1.0,
    s_noise=1.0,
    noise_sampler=None,
    image_to_noise: bool = False,
):
    """
    Ancestral sampling with Euler method steps with cfg++.
    https://arxiv.org/pdf/2406.08070v1
    """

    extra_args = {} if extra_args is None else extra_args
    noise_sampler = default_noise_sampler(x) if noise_sampler is None else noise_sampler
    s_in = x.new_ones([x.shape[0]])
    for i in trange(len(sigmas) - 1, disable=disable):
        # if image_to_noise:
        #     x = x + math.sqrt(sigmas[i + 1] ** 2 - sigmas[i] ** 2) * noise_sampler(
        #         sigmas[i], sigmas[i + 1]
        #     )
        #     cfg_denoised, uncond_denoised = model(x, sigmas[i + 1] * s_in, **extra_args)  # noqa
        # else:
        #     cfg_denoised, uncond_denoised = model(x, sigmas[i] * s_in, **extra_args)
        sigma_cond = sigmas[i + 1] if image_to_noise else sigmas[i]
        cfg_denoised, uncond_denoised = model(
            x, sigmas[i] * s_in, sigma_cond=sigma_cond * s_in, **extra_args
        )
        sigma_down, sigma_up = get_ancestral_step(sigmas[i], sigmas[i + 1], eta=eta)
        if callback is not None:
            callback(
                {
                    "x": x,
                    "i": i,
                    "sigma": sigmas[i],
                    "sigma_hat": sigmas[i],
                    "cfg_denoised": cfg_denoised,
                    "uncond_denoised": uncond_denoised,
                }
            )
        if image_to_noise:
            d = to_d(x, sigmas[i], cfg_denoised)
            # d = to_d(x, sigmas[i + 1], cfg_denoised)
            x = uncond_denoised + d * sigma_down
        else:
            d = to_d(x, sigmas[i], uncond_denoised)
            x = cfg_denoised + d * sigma_down
        if sigmas[i + 1] > 0:
            x = x + noise_sampler(sigmas[i], sigmas[i + 1]) * s_noise * sigma_up
    return x
