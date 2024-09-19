import os

import torch
from diffusers import UNet2DConditionModel, EulerDiscreteScheduler

from duwu.modules.text_encoders import ConcatTextEncoders
from duwu.sampling.cfg import cfg_wrapper, cond_text_wrapper
from duwu.sampling.k_diffusion_euler import (
    sample_euler_ancestral,
    sample_euler_ancestral_cfgpp,
)
from duwu.sampling.k_diffusion_wrapper import DiscreteEpsDDPMDenoiser


@torch.no_grad()
@torch.inference_mode()
def euler_latent_sampling(
    x_init: torch.Tensor,
    unet: UNet2DConditionModel,
    te: ConcatTextEncoders,
    scheduler: EulerDiscreteScheduler,
    prompt: str | list[str],
    neg_prompt: str | list[str],
    image_to_noise: bool = False,
    cfg_scale: float = 3.0,
    use_cfgpp: float = False,
    num_steps: int = 16,
    time_ids: torch.Tensor | None = None,
):
    if isinstance(prompt, str):
        prompt = [prompt]
    if isinstance(neg_prompt, str):
        neg_prompt = [neg_prompt]

    prompt = list(prompt)
    neg_prompt = list(neg_prompt)
    assert len(prompt) == len(neg_prompt) == x_init.size(0)

    reference_param = next(unet.parameters())
    model_wrapper = DiscreteEpsDDPMDenoiser(
        lambda *args, **kwargs: unet(*args, **kwargs)[0],
        scheduler.alphas_cumprod,
        False,
    ).to(reference_param)
    width, height = x_init.size(2) * 8, x_init.size(3) * 8

    if cfg_scale == 0.0:
        cfg_fn = cond_text_wrapper(
            prompt=neg_prompt,
            width=width,
            height=height,
            unet=model_wrapper,
            te=te,
            time_ids=time_ids,
        )
    elif cfg_scale == 1.0 and not use_cfgpp:
        cfg_fn = cond_text_wrapper(
            prompt=prompt,
            width=width,
            height=height,
            unet=model_wrapper,
            te=te,
            time_ids=time_ids,
        )
    else:
        cfg_fn = cfg_wrapper(
            prompt=prompt,
            neg_prompt=neg_prompt,
            width=width,
            height=height,
            unet=model_wrapper,
            te=te,
            cfg=cfg_scale,
            time_ids=time_ids,
        )
    indices = torch.linspace(
        0, scheduler.config.num_train_timesteps - 1, num_steps
    ).long()
    sigmas = scheduler.sigmas[indices]
    if image_to_noise:
        sigmas = sigmas.flip(0)
        # -1 is zero, so we use -2 here
        sigmas = torch.cat([scheduler.sigmas[-2] * sigmas.new_ones(1), sigmas])
    else:
        sigmas = torch.cat([sigmas, scheduler.sigmas[-1] * sigmas.new_ones(1)])

    x_init = x_init.to(reference_param)
    sigmas = sigmas.to(reference_param)

    if image_to_noise:
        x_init = x_init + sigmas[0] * torch.randn_like(x_init)

    if use_cfgpp and cfg_scale != 0.0:
        generated_latents = sample_euler_ancestral_cfgpp(
            cfg_fn, x_init, sigmas, image_to_noise=image_to_noise, eta=0
        )
    else:
        generated_latents = sample_euler_ancestral(
            cfg_fn, x_init, sigmas, image_to_noise=image_to_noise, eta=0
        )
    return generated_latents


if __name__ == "__main__":

    from PIL import Image
    from omegaconf import OmegaConf

    from duwu.data.utils import resize_and_crop_image, vae_image_postprocess
    from duwu.loader import load_any

    image = Image.open("data/images/wiki/Magdeburg.jpg").convert("RGB")
    image_tensor = resize_and_crop_image(
        image, target_size=(1024, 1024), random_crop=False
    )[0]
    prompt = (
        "A stone clock tower with intricate carvings and a large clock face, "
        "encircled by a black fence, stands in front of vibrant red "
        "and yellow flowers under a clear blue sky."
    )
    neg_prompt = ""

    config = OmegaConf.load("configs/model/pretrained_sdxl.yaml")
    unet = load_any(config.model_config.unet)
    te = load_any(config.model_config.te)
    vae = load_any(config.model_config.vae)
    scheduler = load_any(config.model_config.scheduler)

    reference_param = next(unet.parameters())

    with torch.no_grad():
        latents = vae.encode(
            image_tensor.unsqueeze(0).to(reference_param)
        ).latent_dist.sample()

    # VAE decoding

    use_cfgpp = False
    num_steps_list = [500]
    cfg_scales = [1]

    print(f"use_cfgpp: {use_cfgpp}")
    print(f"num_steps_list: {num_steps_list}")
    print(f"cfg_scales: {cfg_scales}")

    output_dir = "test_scripts/outputs/euler_latent/sdxl"
    if use_cfgpp:
        output_dir = os.path.join(output_dir, "cfgpp")
    os.makedirs(output_dir, exist_ok=True)

    decoded_image_tensor = vae.decode(latents).sample[0]
    decoded_image = vae_image_postprocess(decoded_image_tensor)
    image_path = os.path.join(output_dir, "decoded_image.png")
    print(f"Saving decoded image to {image_path}")
    decoded_image.save(image_path)

    # Inversion

    print(f"VAE scaling factor: {vae.config.scaling_factor}")

    noise_scale = torch.sqrt(1 + scheduler.sigmas[0] ** 2)
    print(f"Noise scale: {noise_scale}")

    for num_steps in num_steps_list:

        for cfg_scale in cfg_scales:

            inverted_latent = euler_latent_sampling(
                latents * vae.config.scaling_factor,
                unet,
                te,
                scheduler,
                prompt,
                neg_prompt,
                image_to_noise=True,
                cfg_scale=cfg_scale,
                num_steps=num_steps,
                use_cfgpp=use_cfgpp,
            )
            decoded_inverted = vae.decode(
                inverted_latent / noise_scale / vae.config.scaling_factor
            ).sample[0]
            decoded_inverted_image = vae_image_postprocess(decoded_inverted)
            image_path = os.path.join(
                output_dir,
                f"inverted_image_cfg{cfg_scale}" f"_numsteps{num_steps}.png",
            )
            print(f"Saving decoded inverted image to {image_path}")
            decoded_inverted_image.save(image_path)

            for cfg_scale_sampled in cfg_scales:
                sampled_from_inverted_latent = euler_latent_sampling(
                    inverted_latent,
                    unet,
                    te,
                    scheduler,
                    prompt,
                    neg_prompt,
                    image_to_noise=False,
                    cfg_scale=cfg_scale_sampled,
                    num_steps=num_steps,
                    use_cfgpp=use_cfgpp,
                )
                decoded_sampled_from_inverted = vae.decode(
                    sampled_from_inverted_latent / vae.config.scaling_factor
                ).sample[0]
                decoded_sampled_from_inverted_image = vae_image_postprocess(
                    decoded_sampled_from_inverted
                )
                image_path = os.path.join(
                    output_dir,
                    f"sampled_from_inverted_image_cfg{cfg_scale}-"
                    f"{cfg_scale_sampled}_numsteps{num_steps}.png",
                )
                print(f"Saving decoded sampled image to {image_path}")
                decoded_sampled_from_inverted_image.save(image_path)

    x_init = (
        torch.randn(1, unet.config.in_channels, 128, 128).to(reference_param)
        * noise_scale
    )
    for cfg_scale in cfg_scales:
        for num_steps in num_steps_list:
            sampled_latent = euler_latent_sampling(
                x_init,
                unet,
                te,
                scheduler,
                prompt,
                neg_prompt,
                image_to_noise=False,
                cfg_scale=cfg_scale,
                num_steps=num_steps,
                use_cfgpp=use_cfgpp,
            )
            decoded_sampled = vae.decode(
                sampled_latent / vae.config.scaling_factor
            ).sample[0]
            decoded_sampled_image = vae_image_postprocess(decoded_sampled)
            image_path = os.path.join(
                output_dir,
                f"sampled_image_cfg{cfg_scale}_numsteps{num_steps}.png",
            )
            print(f"Saving decoded sampled image to {image_path}")
            decoded_sampled_image.save(image_path)
