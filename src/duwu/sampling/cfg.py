import torch
import torch.nn.functional as F
from k_diffusion.external import DiscreteSchedule

from duwu.modules.text_encoders import ConcatTextEncoders
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
