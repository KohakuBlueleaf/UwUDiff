from collections.abc import Callable
from typing import Literal

import torch
import lightning.pytorch as pl
from diffusers.models.attention_processor import AttnProcessor2_0
from diffusers import UNet2DConditionModel, AutoencoderKL, EulerDiscreteScheduler

from duwu.utils import truncate_or_pad_to_length
from duwu.data.utils import vae_image_postprocess
from duwu.sampling.cfg import cfg_wrapper, region_cfg_wrapper, layer_cfg_wrapper
from duwu.sampling.k_diffusion_euler import sample_euler_ancestral
from duwu.modules.text_encoders import ConcatTextEncoders
from duwu.sampling.k_diffusion_wrapper import DiscreteEpsDDPMDenoiser
from duwu.modules.attention import (
    CombinedAttnProcessor,
    RegionAttnProcessor,
    LayerAttnProcessor,
)


@torch.no_grad()
@torch.inference_mode()
def diffusion_sampling(
    unet: UNet2DConditionModel,
    te: ConcatTextEncoders,
    vae: AutoencoderKL,
    train_scheduler: EulerDiscreteScheduler,
    prompt: str | list[str] | list[list[str]],
    neg_prompt: str | list[str],
    bboxes: list[list[tuple[float, float, float, float]]] | None = None,
    pad_to_n_bboxes: int | None = None,  # for padding
    adjacency: list[list[list[int]]] | None = None,
    num_steps: int = 16,
    sample_scheduler: EulerDiscreteScheduler | None = None,
    get_sigma_func: Callable[[int], list[float]] | None = None,
    num_samples: int = 1,
    padding_mode: Literal["repeat_last", "cycling", "uniform_expansion"] = "cycling",
    cfg_scale: float = 3.0,
    seed: int = 42,
    width: int = 1024,
    height: int = 1024,
    rescale: bool = False,
    vae_std: float | None = None,
    vae_mean: float | None = None,
    internal_sampling_func: Callable | None = None,
    use_region_attn: bool = False,
    use_layer_attn: bool = False,
    use_flex_attention_for_region: bool = False,
    use_flex_attention_for_layer: bool = True,
):
    pl.seed_everything(seed)

    internal_sampling_func = internal_sampling_func or sample_euler_ancestral

    vae_std = vae_std or 1 / vae.config.scaling_factor
    vae_mean = vae_mean or 0.0

    if isinstance(prompt, str):
        prompt = [prompt]
    if isinstance(neg_prompt, str):
        neg_prompt = [neg_prompt]

    prompt = list(prompt)
    neg_prompt = list(neg_prompt)
    assert len(prompt) == len(neg_prompt)

    prompt = truncate_or_pad_to_length(prompt, num_samples, padding_mode=padding_mode)
    neg_prompt = truncate_or_pad_to_length(
        neg_prompt, num_samples, padding_mode=padding_mode
    )

    reference_param = next(unet.parameters())
    model_wrapper = DiscreteEpsDDPMDenoiser(
        lambda *args, **kwargs: unet(*args, **kwargs)[0],
        train_scheduler.alphas_cumprod,
        False,
    ).to(reference_param)

    num_layers = num_samples

    if bboxes is not None:
        bboxes = list(bboxes)
        if not isinstance(bboxes[0], torch.Tensor):
            bboxes = [torch.tensor(bboxes_per_image) for bboxes_per_image in bboxes]
        bboxes = [bboxes_per_image.to(reference_param) for bboxes_per_image in bboxes]
        bboxes = truncate_or_pad_to_length(
            bboxes, num_samples, padding_mode=padding_mode
        )

    if use_layer_attn and bboxes is not None and adjacency is not None:
        adjacency = list(adjacency)
        adjacency = truncate_or_pad_to_length(
            adjacency, num_samples, padding_mode=padding_mode
        )
        for bboxes_per_image, prompt_per_image in zip(bboxes, prompt):
            assert len(bboxes_per_image) == len(prompt_per_image)
        num_layers = sum([len(bboxes_per_image) for bboxes_per_image in bboxes])

        cross_attn_processor = (
            RegionAttnProcessor(use_flex_attention=use_flex_attention_for_region)
            if use_region_attn
            else AttnProcessor2_0()
        )
        attn_processor = CombinedAttnProcessor(
            self_attn_processor=LayerAttnProcessor(
                use_flex_attention=use_flex_attention_for_layer
            ),
            cross_attn_processor=cross_attn_processor,
        )
        unet.set_attn_processor(attn_processor)
        n_unet_levels = len(unet.down_blocks)
        cfg_fn = layer_cfg_wrapper(
            prompt=prompt,
            neg_prompt=neg_prompt,
            bboxes=bboxes,
            adjacency=adjacency,
            pad_to_n_bboxes=pad_to_n_bboxes,
            width=width,
            height=height,
            unet=model_wrapper,
            te=te,
            cfg=cfg_scale,
            n_unet_levels=n_unet_levels,
            with_region_attention=use_region_attn,
            use_flex_attention_for_region=use_flex_attention_for_region,
            use_flex_attention_for_layer=use_flex_attention_for_layer,
        )
    elif use_region_attn and bboxes is not None:
        for bboxes_per_image, prompt_per_image in zip(bboxes, prompt):
            assert len(bboxes_per_image) == len(prompt_per_image)

        attn_processor = CombinedAttnProcessor(
            self_attn_processor=AttnProcessor2_0(),
            cross_attn_processor=RegionAttnProcessor(
                use_flex_attention=use_flex_attention_for_region
            ),
        )
        unet.set_attn_processor(attn_processor)
        # no downsampling at last down block
        n_unet_levels = len(unet.down_blocks)
        cfg_fn = region_cfg_wrapper(
            prompt=prompt,
            neg_prompt=neg_prompt,
            bboxes=bboxes,
            pad_to_n_bboxes=pad_to_n_bboxes,
            width=width,
            height=height,
            unet=model_wrapper,
            te=te,
            cfg=cfg_scale,
            n_unet_levels=n_unet_levels,
            use_flex_attention=use_flex_attention_for_region,
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
        )
    # for laplace scheduler sigmas[0] would be too large
    sample_scheduler = sample_scheduler or train_scheduler
    # print(train_scheduler.sigmas)
    # print(sample_scheduler.sigmas)

    if get_sigma_func is None:
        # # Either is fine as long as we do not have two consecutive 0s at the end
        # # but the first one is much worse when we have few steps
        # sigmas = sample_scheduler.sigmas[
        #     torch.linspace(
        #         0, sample_scheduler.config.num_train_timesteps - 1, num_steps
        #     ).long()
        # ]
        # sigmas = torch.concat([sigmas, sigmas.new_zeros(1)], 0).to(reference_param)
        sigmas = sample_scheduler.sigmas[
            torch.linspace(
                0, sample_scheduler.config.num_train_timesteps, num_steps + 1
            ).long()
        ]
    else:
        sigmas = get_sigma_func(num_steps)
    if not isinstance(sigmas, torch.Tensor):
        sigmas = torch.tensor(sigmas)
    sigmas = sigmas.to(reference_param)

    # print(
    #     train_scheduler._sigma_to_t(
    #         sigmas.cpu(), train_scheduler.sigmas.log().flip(0)[1:]
    #     )
    # )

    init_x = torch.randn(
        num_layers, unet.config.in_channels, height // 8, width // 8
    ).to(reference_param) * torch.sqrt(1 + sigmas[0] ** 2)
    generated_latents = internal_sampling_func(cfg_fn, init_x, sigmas)
    # print(generated_latents.std([1, 2, 3]))
    if rescale:
        generated_latents /= generated_latents.std([1, 2, 3], keepdim=True)
    generated_latents = generated_latents * vae_std + vae_mean
    image_tensors = []
    for generated_latent in generated_latents:
        image_tensors.append(vae.decode(generated_latent.unsqueeze(0)).sample)
    image_tensors = torch.concat(image_tensors)
    torch.cuda.empty_cache()
    images = []
    for image_tensor in image_tensors:
        image = vae_image_postprocess(image_tensor)
        images.append(image)
    return images
