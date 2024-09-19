from tqdm import tqdm
from collections.abc import Sequence

import torch
from torchmetrics.multimodal import CLIPScore


def compute_clip_score(
    generated: Sequence[tuple[torch.Tensor, str]],  # Image text pairs
    batch_size: int = 256,
    device: str = "cuda",
    disable_tqdm: bool = False,
    normalize: bool = True,
    **clip_kwargs,
):

    clip_score = CLIPScore(**clip_kwargs).to(device)

    for start in tqdm(
        range(0, len(generated), batch_size),
        disable=disable_tqdm,
    ):
        end_index = min(len(generated), start + batch_size)
        images = torch.stack([generated[idx][0] for idx in range(start, end_index)]).to(
            device
        )
        if normalize:
            # clipscore expects values in [0, 255]
            images = images * 255
        texts = [generated[idx][1] for idx in range(start, end_index)]
        clip_score.update(images, texts)

    return clip_score.compute()
