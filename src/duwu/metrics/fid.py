from tqdm import tqdm
from collections.abc import Sequence

import torch
from torchmetrics.image.fid import FrechetInceptionDistance


def compute_fid(
    generated: Sequence[torch.Tensor],
    reference: Sequence[torch.Tensor],
    batch_size: int = 256,
    device: str = "cuda",
    disable_tqdm: bool = False,
    **fid_kwargs,
):

    fid = FrechetInceptionDistance(**fid_kwargs).to(device)

    for start in tqdm(
        range(0, len(reference), batch_size),
        disable=disable_tqdm,
        desc="Update FID with reference images",
    ):
        end_index = min(len(reference), start + batch_size)
        images = torch.stack([reference[idx] for idx in range(start, end_index)]).to(
            device
        )
        fid.update(images, real=True)

    for start in tqdm(
        range(0, len(generated), batch_size),
        disable=disable_tqdm,
        desc="Update FID with generated images",
    ):
        end_index = min(len(generated), start + batch_size)
        images = torch.stack([generated[idx] for idx in range(start, end_index)]).to(
            device
        )
        fid.update(images, real=False)

    return fid.compute()
