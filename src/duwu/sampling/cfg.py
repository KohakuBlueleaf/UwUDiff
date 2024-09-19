import torch
import torch.nn.functional as F
from k_diffusion.external import DiscreteSchedule

from duwu.modules.text_encoders import ConcatTextEncoders
from duwu.modules.attn_masks import (
    make_region_mask_dict,
    make_layer_attn_meta_dict,
    make_layer_region_mask_dict,
)
from duwu.utils.aggregation import aggregate_embeddings


def cond_text_wrapper(
    prompt: str | list[str],
    width: int,
    height: int,
    unet: DiscreteSchedule,  # should be k_diffusion wrapper
    te: ConcatTextEncoders,
    time_ids: torch.Tensor | None = None,
):
    emb, normed_emb, pool, mask = te.encode(prompt, padding=True, truncation=True)

    if te.use_normed_ctx:
        emb = normed_emb

    if time_ids is None:
        time_ids = (
            torch.tensor([height, width, 0, 0, height, width])
            .repeat(emb.size(0), 1)
            .to(emb)
        )
    else:
        time_ids = time_ids.to(emb.device)

    # sdxl
    if pool is not None:
        added_cond = {
            "time_ids": time_ids,
            "text_embeds": pool,
        }
    else:
        added_cond = None

    def model_fn(x, sigma, sigma_cond=None):
        denoised = unet(
            x,
            sigma,
            sigma_cond=sigma_cond,
            encoder_hidden_states=emb,
            encoder_attention_mask=mask,
            added_cond_kwargs=added_cond,
        )
        return denoised, None

    return model_fn


def cfg_wrapper(
    prompt: str | list[str],
    neg_prompt: str | list[str],
    width: int,
    height: int,
    unet: DiscreteSchedule,  # should be k_diffusion wrapper
    te: ConcatTextEncoders,
    cfg: float = 5.0,
    time_ids: torch.Tensor | None = None,
):
    # Set truncation to True for simplicity
    # TODO: split into multiple sentences, encode separately,
    # use all for cross attention

    emb, normed_emb, pool, mask = te.encode(prompt, padding=True, truncation=True)
    neg_emb, normed_neg_emb, neg_pool, neg_mask = te.encode(
        neg_prompt, padding=True, truncation=True
    )

    if te.use_normed_ctx:
        emb = normed_emb
        neg_emb = normed_neg_emb

    if time_ids is None:
        time_ids = (
            torch.tensor([height, width, 0, 0, height, width])
            .repeat(2 * emb.size(0), 1)
            .to(emb)
        )
    else:
        time_ids = time_ids.repeat(2, 1).to(emb)

    # sdxl
    if pool is not None:
        added_cond = {
            "time_ids": time_ids,
            "text_embeds": torch.concat([pool, neg_pool]),
        }
    else:
        added_cond = None

    # emb is of size (batch_size, seq_len, emb_dim)
    if emb.size(1) > neg_emb.size(1):
        pad_setting = (0, 0, 0, emb.size(1) - neg_emb.size(1))
        neg_emb = F.pad(neg_emb, pad_setting)
        if neg_mask is not None:
            neg_mask = F.pad(neg_mask, pad_setting[2:])
    if neg_emb.size(1) > emb.size(1):
        pad_setting = (0, 0, 0, neg_emb.size(1) - emb.size(1))
        emb = F.pad(emb, pad_setting)
        if mask is not None:
            mask = F.pad(mask, pad_setting[2:])

    if mask is not None and neg_mask is not None:
        attn_mask = torch.concat([mask, neg_mask])
    else:
        attn_mask = None
    text_ctx_emb = torch.concat([emb, neg_emb])

    def cfg_fn(x, sigma, sigma_cond=None):
        if sigma_cond is not None:
            sigma_cond = torch.cat([sigma_cond, sigma_cond])
        cond, uncond = unet(
            torch.cat([x, x]),
            torch.cat([sigma, sigma]),
            sigma_cond=sigma_cond,
            encoder_hidden_states=text_ctx_emb,
            encoder_attention_mask=attn_mask,
            added_cond_kwargs=added_cond,
        ).chunk(2)
        cfg_output = uncond + (cond - uncond) * cfg
        return cfg_output, uncond

    return cfg_fn


def region_cfg_wrapper(
    prompt: list[list[str]],
    neg_prompt: list[str],
    bboxes: list[torch.FloatTensor],
    width: int,
    height: int,
    unet: DiscreteSchedule,  # should be k_diffusion wrapper
    te: ConcatTextEncoders,
    use_flex_attention: bool = False,
    cfg: float = 5.0,
    pad_to_n_bboxes: int | None = None,
    time_ids: torch.Tensor | None = None,
    n_unet_levels: int = 3,
):
    assert len(prompt) == len(neg_prompt) == len(bboxes)

    emb, normed_emb, pool, mask = te.encode(
        prompt,
        nested=True,
        padding=True,
        truncation=True,
        pad_to_n_elements=pad_to_n_bboxes,
    )
    neg_emb, normed_neg_emb, neg_pool, neg_mask = te.encode(
        neg_prompt,
        padding=True,
        truncation=True,
    )

    if te.use_normed_ctx:
        emb = normed_emb
        neg_emb = normed_neg_emb

    if time_ids is None:
        time_ids = (
            torch.tensor([height, width, 0, 0, height, width])
            .repeat(2 * emb.size(0), 1)
            .to(emb)
        )
    else:
        time_ids = time_ids.repeat(2, 1).to(emb)

    # sdxl
    if pool is not None:
        added_cond = {
            "time_ids": time_ids,
            "text_embeds": torch.concat([pool, neg_pool]),
        }
    else:
        added_cond = None

    # emb is of size (batch_size, seq_len, emb_dim)
    if emb.size(1) > neg_emb.size(1):
        pad_setting = (0, 0, 0, emb.size(1) - neg_emb.size(1))
        neg_emb = F.pad(neg_emb, pad_setting)
        if neg_mask is not None:
            neg_mask = F.pad(neg_mask, pad_setting[2:])
    if neg_emb.size(1) > emb.size(1):
        pad_setting = (0, 0, 0, neg_emb.size(1) - emb.size(1))
        emb = F.pad(emb, pad_setting)
        if mask is not None:
            mask = F.pad(mask, pad_setting[2:])

    # text attention mask
    if mask is not None and neg_mask is not None:
        attn_mask = torch.concat([mask, neg_mask])
    else:
        attn_mask = None
    text_ctx_emb = torch.concat([emb, neg_emb])

    # bboxes conditioning
    n_captions_per_image = [len(p) for p in prompt]
    max_n_captions = pad_to_n_bboxes or max(n_captions_per_image)
    # TODO: this does not work with general neg_emb, need to do as in layer_cfg_wrapper
    sequence_length = emb.shape[1] // max_n_captions

    # add entire image bbox for negative caption
    bboxes = bboxes + [torch.tensor([[0, 0, 1, 1]]).to(bboxes[0])] * neg_emb.shape[0]
    region_mask_dict = make_region_mask_dict(
        bboxes,
        latent_width=width // 8,
        latent_height=height // 8,
        sequence_length=sequence_length,
        encoder_attn_mask=attn_mask,
        n_levels=n_unet_levels,
        skip_top_level=True,
        pad_to_n_bboxes=pad_to_n_bboxes,
        use_flex_attention=use_flex_attention,
    )

    def cfg_fn(x, sigma, sigma_cond=None):
        if sigma_cond is not None:
            sigma_cond = torch.cat([sigma_cond, sigma_cond])
        cond, uncond = unet(
            torch.cat([x, x]),
            torch.cat([sigma, sigma]),
            sigma_cond=sigma_cond,
            encoder_hidden_states=text_ctx_emb,
            encoder_attention_mask=attn_mask,
            added_cond_kwargs=added_cond,
            cross_attention_kwargs={"region_mask_dict": region_mask_dict},
        ).chunk(2)
        cfg_output = uncond + (cond - uncond) * cfg
        return cfg_output, uncond

    return cfg_fn


def layer_cfg_wrapper(
    prompt: list[list[str]],
    neg_prompt: list[str],
    bboxes: list[torch.FloatTensor],
    adjacency: list[list[list[int]]],
    width: int,
    height: int,
    unet: DiscreteSchedule,  # should be k_diffusion wrapper
    te: ConcatTextEncoders,
    with_region_attention: bool = False,
    use_flex_attention_for_region: bool = False,
    use_flex_attention_for_layer: bool = True,
    cfg: float = 5.0,
    pad_to_n_bboxes: int | None = None,
    time_ids: torch.Tensor | None = None,
    n_unet_levels: int = 3,
):
    assert len(prompt) == len(neg_prompt) == len(bboxes) == len(adjacency)

    n_captions_per_image = torch.tensor([len(p) for p in prompt]).to(te.device)

    # Encode separately
    prompt_flattend = [item for sublist in prompt for item in sublist]
    n_valid_layers = len(prompt_flattend) * 2

    emb, normed_emb, pool, mask = te.encode(
        prompt_flattend, padding=True, truncation=True
    )
    neg_emb, normed_neg_emb, neg_pool, neg_mask = te.encode(
        neg_prompt, padding=True, truncation=True
    )

    if te.use_normed_ctx:
        emb = normed_emb
        neg_emb = normed_neg_emb

    # emb is of size (batch_size, seq_len, emb_dim)
    # Make sure emb and neg_emb have the same sequence length for individual prompt
    if emb.size(1) > neg_emb.size(1):
        pad_setting = (0, 0, 0, emb.size(1) - neg_emb.size(1))
        neg_emb = F.pad(neg_emb, pad_setting)
        if neg_mask is not None:
            neg_mask = F.pad(neg_mask, pad_setting[2:])
    if neg_emb.size(1) > emb.size(1):
        pad_setting = (0, 0, 0, neg_emb.size(1) - emb.size(1))
        emb = F.pad(emb, pad_setting)
        if mask is not None:
            mask = F.pad(mask, pad_setting[2:])

    if with_region_attention:
        sequence_length = emb.shape[1]
        emb = aggregate_embeddings(
            emb, n_captions_per_image, mode="concat", pad_to_n_elements=pad_to_n_bboxes
        )
        emb = emb.repeat_interleave(n_captions_per_image, dim=0)
        if mask is not None:
            mask = aggregate_embeddings(
                mask,
                n_captions_per_image,
                mode="concat",
                pad_to_n_elements=pad_to_n_bboxes,
            )
            mask = mask.repeat_interleave(n_captions_per_image, dim=0)
        to_pad = emb.size(1) - neg_emb.size(1)
        # Pad negative
        neg_emb = F.pad(neg_emb, (0, 0, 0, to_pad))
        if neg_mask is not None:
            neg_mask = F.pad(neg_mask, (0, to_pad))

    neg_emb = neg_emb.repeat_interleave(n_captions_per_image, dim=0)

    if neg_pool is not None:
        neg_pool = neg_pool.repeat_interleave(n_captions_per_image, dim=0)
    if neg_mask is not None:
        neg_mask = neg_mask.repeat_interleave(n_captions_per_image, dim=0)

    # text attention mask
    if mask is not None and neg_mask is not None:
        attn_mask = torch.concat([mask, neg_mask])
    else:
        attn_mask = None
    text_ctx_emb = torch.concat([emb, neg_emb])

    if time_ids is None:
        bboxes_flattend = [item for sublist in bboxes for item in sublist]
        time_ids = []
        # Use cropping coordinate
        for bbox in bboxes_flattend:
            time_ids.append([height, width, bbox[1], bbox[0], height, width])
        time_ids = torch.tensor(time_ids)
    time_ids = time_ids.repeat(2, 1).to(emb)

    if pad_to_n_bboxes is not None:
        pad_to_n_total_bboxes = pad_to_n_bboxes * len(prompt) * 2
        to_pad = pad_to_n_total_bboxes - text_ctx_emb.size(0)
        text_ctx_emb = F.pad(text_ctx_emb, (0, 0, 0, 0, 0, to_pad))
        if attn_mask is not None:
            attn_mask = F.pad(attn_mask, (0, 0, 0, to_pad))
        if neg_pool is not None:
            neg_pool = F.pad(neg_pool, (0, 0, 0, to_pad))
        if time_ids is not None:
            time_ids = F.pad(time_ids, (0, 0, 0, to_pad))
    else:
        to_pad = None

    # sdxl
    if pool is not None:
        added_cond = {
            "time_ids": time_ids,
            "text_embeds": torch.concat([pool, neg_pool]),
        }
    else:
        added_cond = None

    # add entire image bbox for negative caption
    bboxes = bboxes + [torch.tensor([[0, 0, 1, 1]]).to(bboxes[0])] * neg_emb.shape[0]
    adjacency = adjacency + [[[]]] * neg_emb.shape[0]
    n_captions_per_image = torch.cat(
        [n_captions_per_image, n_captions_per_image.new_ones(neg_emb.shape[0])]
    )

    if to_pad is not None:
        # print("to_pad", to_pad)
        bboxes = bboxes + [torch.tensor([[0, 0, 1, 1]]).to(bboxes[0])] * to_pad
        adjacency = adjacency + [[[]]] * to_pad
        n_captions_per_image = torch.cat(
            [n_captions_per_image, n_captions_per_image.new_ones(to_pad)]
        )

    layer_attn_meta_dict = make_layer_attn_meta_dict(
        bboxes,
        adjacency,
        latent_width=width // 8,
        latent_height=height // 8,
        n_levels=n_unet_levels,
        skip_top_level=True,
        pad_to_n_bboxes=pad_to_n_bboxes,
        use_flex_attention=use_flex_attention_for_layer,
    )

    if with_region_attention:
        region_mask_dict = make_layer_region_mask_dict(
            bboxes,
            adjacency,
            latent_width=width // 8,
            latent_height=height // 8,
            sequence_length=sequence_length,
            encoder_attn_mask=attn_mask,
            n_levels=n_unet_levels,
            skip_top_level=True,
            pad_to_n_bboxes=pad_to_n_bboxes,
            use_flex_attention=use_flex_attention_for_region,
        )
    else:
        region_mask_dict = None

    # This is wrong as this should correspond to bboxes
    # # n_captions_per_image = n_captions_per_image.repeat(2)

    def cfg_fn(x, sigma, sigma_cond=None):
        if sigma_cond is not None:
            sigma_cond = torch.cat([sigma_cond, sigma_cond])
        x = torch.cat([x, x])
        sigma = torch.cat([sigma, sigma])
        if to_pad is not None:
            x = F.pad(x, (0, 0, 0, 0, 0, 0, 0, to_pad))
            sigma = F.pad(sigma, (0, to_pad))
            if sigma_cond is not None:
                sigma_cond = F.pad(sigma_cond, (0, to_pad))
        cond, uncond = unet(
            x,
            sigma,
            sigma_cond=sigma_cond,
            encoder_hidden_states=text_ctx_emb,
            encoder_attention_mask=attn_mask,
            added_cond_kwargs=added_cond,
            cross_attention_kwargs={
                "layer_attn_meta_dict": layer_attn_meta_dict,
                "n_elements_per_image": n_captions_per_image,
                "pad_to_n_elements": pad_to_n_bboxes,
                "region_mask_dict": region_mask_dict,
            },
        )[:n_valid_layers].chunk(2)
        cfg_output = uncond + (cond - uncond) * cfg
        return cfg_output, uncond

    return cfg_fn
