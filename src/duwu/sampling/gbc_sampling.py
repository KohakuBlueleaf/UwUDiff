from dataclasses import dataclass

import torch

from duwu.sampling.sampling import diffusion_sampling


@dataclass
class GbcGraphPrompt:
    prompts: list[str]
    bboxes: list[list[tuple[float, float, float, float]]] | list[torch.Tensor]
    adjacency: list[list[int]]

    def __str__(self):
        repr = []
        for prompt, bbox, adjacency in zip(self.prompts, self.bboxes, self.adjacency):
            bbox_str = ",".join([f"{x:.2f}" for x in bbox])
            repr.append(f"- {prompt} [{bbox_str}] [target: {adjacency}]")
        return "\n".join(repr)

    def get_layer_prompt(self, layer: int) -> str:
        bbox = self.bboxes[layer]
        bbox_str = ",".join([f"{x:.2f}" for x in bbox])
        return (
            f"Layer {layer}: {self.prompts[layer]} [{bbox_str}] "
            f"[target: {self.adjacency[layer]}]"
        )


def gbc_diffusion_sampling(
    prompt: list[GbcGraphPrompt],
    neg_prompt: list[str],
    use_region_attn: bool = True,
    use_layer_attn: bool = True,
    num_samples: int = 1,
    **kwargs,
):
    gbc_prompts = prompt
    prompts = [gbc_prompt.prompts for gbc_prompt in gbc_prompts]
    bboxes = [gbc_prompt.bboxes for gbc_prompt in gbc_prompts]
    adjacency = [gbc_prompt.adjacency for gbc_prompt in gbc_prompts]

    # Still need max n bbox padding
    n_total_prompts = sum(len(gbc_prompt.prompts) for gbc_prompt in gbc_prompts)

    images = diffusion_sampling(
        prompt=prompts,
        neg_prompt=neg_prompt,
        bboxes=bboxes,
        adjacency=adjacency,
        use_region_attn=use_region_attn,
        use_layer_attn=use_layer_attn,
        num_samples=num_samples,
        **kwargs,
    )

    if use_layer_attn:
        # regroup images
        start = 0
        stacked_images = []
        for gbc_prompt in gbc_prompts:
            stacked_images.append(images[start : start + len(gbc_prompt.prompts)])
            start += len(gbc_prompt.prompts)
        images = stacked_images
    return images
