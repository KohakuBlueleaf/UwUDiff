from omegaconf import OmegaConf

import torch
from diffusers.models.attention_processor import AttnProcessor2_0

from duwu.loader import load_any
from duwu.modules.attn_masks import make_layer_attn_meta_dict
from duwu.modules.attention import CombinedAttnProcessor, LayerAttnProcessor


attn_processor = CombinedAttnProcessor(
    self_attn_processor=LayerAttnProcessor(),
    cross_attn_processor=AttnProcessor2_0(),
)

config = OmegaConf.load("configs/model/pretrained_sdxl_te_mask.yaml")

unet = load_any(config.model_config.unet)
te = load_any(config.model_config.te)
# vae = load_any(config.model_config.vae)

unet.set_attn_processor(attn_processor)
# unet = torch.compile(unet)

# print(unet)


ref_params = next(unet.parameters())
device = ref_params.device
image_width, image_height = 1024, 1024

batch_size = 5
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

n_captions_per_image = [len(p) for p in prompts]
prompts_flattened = [p for ps in prompts for p in ps]

emb, normed_emb, pool, mask = te.encode(
    prompts_flattened, nested=False, padding=True, truncation=True
)
print(f"Embedding shape: {emb.shape}")
print(f"Encoder mask shape: {mask.shape}")
print(mask)
print(f"Pool shape: {pool.shape}")

# Additional conditioning

added_cond = {
    "time_ids": torch.tensor(
        [image_height, image_width, 0, 0, image_height, image_width]
    )
    .repeat(emb.size(0), 1)
    .to(emb),
    "text_embeds": pool.to(emb),
}

# Layer masks

bboxes = [
    torch.tensor([[0, 0, 1, 1], [0.3, 0.5, 0.7, 1]]).to(ref_params),
    torch.tensor([[0, 0, 1, 1], [0, 0, 1, 0.5], [0.5, 0.5, 1, 1]]).to(ref_params),
]
adjacencies = [[[1], []], [[1, 2], [], []]]
n_levels = 3

layer_attn_meta_dict = make_layer_attn_meta_dict(
    bboxes,
    adjacencies,
    latent_width,
    latent_height,
    n_levels,
    skip_top_level=True,
    use_flex_attention=True,
)

denoised = unet(
    latent,
    timesteps,
    encoder_hidden_states=emb,
    encoder_attention_mask=mask,
    added_cond_kwargs=added_cond,
    cross_attention_kwargs={
        "layer_attn_meta_dict": layer_attn_meta_dict,
        "n_elements_per_image": n_captions_per_image,
    },
)

print(denoised[0].shape)
