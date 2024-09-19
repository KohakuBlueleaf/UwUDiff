from omegaconf import OmegaConf

import torch
from diffusers.models.attention_processor import AttnProcessor2_0

from duwu.loader import load_any
from duwu.modules.attn_masks import make_region_mask_dict, make_layer_region_mask_dict
from duwu.modules.attention import CombinedAttnProcessor, RegionAttnProcessor


attn_processor = CombinedAttnProcessor(
    self_attn_processor=AttnProcessor2_0(), cross_attn_processor=RegionAttnProcessor()
)

config = OmegaConf.load("configs/model/pretrained_sdxl_te_mask.yaml")

unet = load_any(config.model_config.unet)
te = load_any(config.model_config.te)
# vae = load_any(config.model_config.vae)

unet.set_attn_processor(attn_processor)

# print(unet)


ref_params = next(unet.parameters())
device = ref_params.device
image_width, image_height = 1024, 1024

batch_size = 2
n_channels = 4
latent_width, latent_height = image_width // 8, image_height // 8

# Define latent and timestpes

latent = torch.randn(batch_size, n_channels, latent_width, latent_height).to(ref_params)
timesteps = torch.randint(0, 1000, (batch_size,), device=device)

# Encode prompt

prompts = [
    [
        (
            "A photograph of an astronaut riding a horse through a dense forest, "
            "with sunlight filtering through the trees."
        ),
        (
            "The horse is a majestic brown mare with a flowing mane, "
            "galloping confidently over the leafy ground."
        ),
    ],
    [
        (
            "A photograph of a tiny elephant balancing on "
            "a tightrope strung between two skyscrapers."
        ),
        (
            "The skyscrapers below are sleek and modern, "
            "with glass facades reflecting the golden glow of the setting sun."
        ),
        "The elephant is wearing a bright red circus hat.",
    ],
]

emb, normed_emb, pool, mask = te.encode(
    prompts,
    nested=True,
    padding=True,
    truncation=True,
)
print(f"Embedding shape: {emb.shape}")
print(f"Mask shape: {mask.shape}")
print(mask)
print(f"Pool shape: {pool.shape}")

n_captions_per_image = [len(p) for p in prompts]
max_n_captions = max(n_captions_per_image)
max_caption_length = emb.shape[1] // max_n_captions

# sequence_length = emb.shape[1]
# to_pad = 128 - (sequence_length % 128)
# if to_pad > 0:
#     emb = F.pad(emb, (0, 0, 0, to_pad))

# Additional conditioning

added_cond = {
    "time_ids": torch.tensor(
        [image_height, image_width, 0, 0, image_height, image_width]
    )
    .repeat(emb.size(0), 1)
    .to(emb),
    "text_embeds": pool.to(emb),
}

# Region masks

bboxes = [
    torch.tensor([[0, 0, 1, 1], [0.3, 0.5, 0.7, 1]]).to(ref_params),
    torch.tensor([[0, 0, 1, 1], [0, 0, 1, 0.5], [0.5, 0.5, 1, 1]]).to(ref_params),
]
n_levels = 3

region_mask_dict = make_region_mask_dict(
    bboxes,
    latent_width,
    latent_height,
    sequence_length=max_caption_length,
    encoder_attn_mask=mask.bool(),
    n_levels=n_levels,
    skip_top_level=True,
    use_flex_attention=True,
)

denoised = unet(
    latent,
    timesteps,
    encoder_hidden_states=emb,
    encoder_attention_mask=mask,
    added_cond_kwargs=added_cond,
    cross_attention_kwargs={"region_mask_dict": region_mask_dict},
)

print("denoised shape:", denoised[0].shape)


# Test layer region attn

adjacencies = [[[1], []], [[1, 2], [], []]]

n_bboxes_per_image = torch.tensor([len(b) for b in bboxes]).to(ref_params.device)
latent = torch.randn(
    n_bboxes_per_image.sum(), n_channels, latent_width, latent_height
).to(ref_params)
timesteps = torch.randint(0, 1000, (n_bboxes_per_image.sum(),), device=device)

mask = mask.repeat_interleave(n_bboxes_per_image, dim=0)
emb = emb.repeat_interleave(n_bboxes_per_image, dim=0)

print("latent shape:", latent.shape)
print("mask shape:", mask.shape)
print("emb shape:", emb.shape)

added_cond = {
    "time_ids": torch.tensor(
        [image_height, image_width, 0, 0, image_height, image_width]
    )
    .repeat(emb.size(0), 1)
    .to(emb),
    "text_embeds": pool.repeat_interleave(n_bboxes_per_image, dim=0).to(emb),
}

region_mask_dict = make_layer_region_mask_dict(
    bboxes,
    adjacencies,
    latent_width,
    latent_height,
    sequence_length=max_caption_length,
    encoder_attn_mask=mask.bool(),
    n_levels=n_levels,
    skip_top_level=True,
    use_flex_attention=True,
)

denoised = unet(
    latent,
    timesteps,
    encoder_hidden_states=emb,
    encoder_attention_mask=mask,
    added_cond_kwargs=added_cond,
    cross_attention_kwargs={"region_mask_dict": region_mask_dict},
)

print("denoised shape:", denoised[0].shape)
