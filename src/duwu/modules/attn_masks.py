import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch.nn.attention.flex_attention import create_block_mask, BlockMask
from einops import rearrange

from duwu.utils.aggregation import get_batch_and_position_indices_for_concat_aggregate


Mask = torch.BoolTensor | BlockMask


def convert_and_stretch_mask(
    attn_mask: torch.BoolTensor,
    sequence_length: int,
    encoder_attn_mask: torch.BoolTensor | None = None,
    use_flex_attention: bool = False,
) -> Mask:
    # print(attn_mask.shape)
    if use_flex_attention:
        attn_mask = convert_to_flex_attn_mask(
            attn_mask,
            sequence_length=sequence_length,
            encoder_attn_mask=encoder_attn_mask,
        )
    else:
        attn_mask = attn_mask.repeat_interleave(sequence_length, dim=-1)
        if encoder_attn_mask is not None:
            attn_mask = torch.logical_and(attn_mask, encoder_attn_mask)
    return attn_mask


def convert_to_flex_attn_mask(
    mask: torch.BoolTensor,
    sequence_length: int,
    encoder_attn_mask: torch.BoolTensor | None = None,
) -> BlockMask:
    """
    Converts a binary mask to a flex attention mask
    """

    # must be multiple of 128
    kv_length = sequence_length * mask.size(2)
    final_kv_length = math.ceil(kv_length / 128) * 128
    to_pad = math.ceil((final_kv_length - kv_length) / mask.size(2))
    mask = F.pad(mask, (0, to_pad))
    if encoder_attn_mask is not None:
        encoder_attn_mask = F.pad(encoder_attn_mask, (0, final_kv_length - kv_length))

    def mask_value(b, h, q_idx, kv_idx):
        mask_value = mask[b, q_idx, kv_idx // sequence_length]
        if encoder_attn_mask is not None:
            mask_value = mask_value & encoder_attn_mask[b, kv_idx]
        return mask_value

    block_mask = create_block_mask(
        mask_value,
        B=mask.size(0),
        H=None,
        Q_LEN=mask.size(1),
        KV_LEN=final_kv_length,
        # device=mask.device,   # This does not seem to help creating on cpu
        _compile=True,
    )
    return block_mask


def convert_to_flex_attn_score_mod(
    mask: torch.Tensor, sequence_length: int
) -> BlockMask:

    def score_func(score, b, h, q_idx, kv_idx):
        return torch.where(
            mask[b, q_idx, kv_idx // sequence_length], score, -float("inf")
        )

    return score_func


# For region-attention


def convert_mask_dict(
    mask_dict: dict[int, torch.BoolTensor],
    sequence_length: int,
    encoder_attn_mask: torch.BoolTensor | None = None,
    use_flex_attention: bool = False,
) -> dict[int, torch.BoolTensor] | dict[int, BlockMask]:
    return {
        key: convert_and_stretch_mask(
            mask, sequence_length, encoder_attn_mask, use_flex_attention
        )
        for key, mask in mask_dict.items()
    }


def make_region_mask_dict(
    bboxes: list[torch.Tensor],
    latent_width: int,
    latent_height: int,
    n_levels: int,
    sequence_length: int = 77,
    pad_to_n_bboxes: int | None = None,
    # For sdxl, there is no attention at first down block
    skip_top_level: bool = False,
    encoder_attn_mask: torch.BoolTensor | None = None,
    use_flex_attention: bool = True,
    no_conversion: bool = False,
) -> dict[int, torch.BoolTensor] | dict[int, BlockMask]:
    """
    Converts a set of relative bounding boxes to a dictionary mapping
    feature map size to binary mask of size
    [batch_size, feature_height*feature_width, max_n_bboxes(*sequence_length)]
    - The length of the last dimension is only of max_n_bboxes
      if ``no_conversion`` is True
    - Otherwise it is of max_n_bboxes * sequence_length
    """
    assert (
        latent_width * latent_height % 128 == 0
    ), "Latent size must be divisible by 128"

    if encoder_attn_mask is not None and not use_flex_attention:
        encoder_attn_mask = encoder_attn_mask.unsqueeze(1)

    region_mask_dict = {}
    start_idx = 1 if skip_top_level else 0

    for idx in range(n_levels):
        if idx >= start_idx:
            region_masks = [
                bboxes_to_mask(bbox, latent_width, latent_height) for bbox in bboxes
            ]
            size = latent_width * latent_height
            attn_mask = aggregate_region_masks(
                region_masks, max_n_masks=pad_to_n_bboxes
            )
            if not no_conversion:
                attn_mask = convert_and_stretch_mask(
                    attn_mask, sequence_length, encoder_attn_mask, use_flex_attention
                )
            region_mask_dict[size] = attn_mask
        latent_width = latent_width // 2
        latent_height = latent_height // 2
    return region_mask_dict


def aggregate_region_masks(
    region_masks: list[torch.FloatTensor], max_n_masks: int | None = None
) -> torch.BoolTensor:
    """
    Parameters
    ----------
    region_masks
        A batch of region masks. For each image we can have different number
        of region masks, but the region masks should be of the same size.
        The elements of the list are tensor of size [n_masks, height, width]
    max_n_masks
        maximum number of region masks, should be given in the case of padding
    """
    batch_size = len(region_masks)
    max_n_masks = max_n_masks or max(len(m) for m in region_masks)
    height, width = region_masks[0].shape[-2:]
    device = region_masks[0].device
    mask = torch.zeros([batch_size, height, width, max_n_masks], dtype=torch.bool).to(
        device
    )

    # Should we avoid for here? -> we shouldn't call it at each attention
    # but should be ok if we only call it at the beginning
    for b, region_mask in enumerate(region_masks):
        n_masks = region_mask.shape[0]
        mask[b, ..., :n_masks] = rearrange(region_mask, "n h w -> h w n")

    return mask.flatten(start_dim=1, end_dim=2)


# For layer-attention


@dataclass
class LayerAttnMeta:
    # For embedding aggregation
    agg_batch_indices_flat: torch.LongTensor
    agg_positions_flat: torch.LongTensor
    # mask of size
    # [n_total_bboxes, height*width, max_n_bboxes*height*width]
    layer_attn_mask: Mask
    # Memory allocation for the concatenated embedding
    # [batch_size, max_n_bboxes*height*width, *embedding_dim[2:]]
    # It should be cached within attention layers
    cat_embeddings: torch.FloatTensor | None = None


def convert_layer_attn_meta_dict(
    layer_attn_mask_dict: dict[int, LayerAttnMeta],
    use_flex_attention: bool = False,
) -> dict[int, torch.BoolTensor] | dict[int, BlockMask]:
    # This performs in-place operation
    for latent_size, meta in layer_attn_mask_dict.items():
        meta.layer_attn_mask = convert_and_stretch_mask(
            meta.layer_attn_mask,
            sequence_length=latent_size,
            encoder_attn_mask=None,
            use_flex_attention=use_flex_attention,
        )
    return layer_attn_mask_dict


def make_layer_attn_meta_dict(
    bboxes: list[torch.Tensor],
    adjacency: list[list[list[int]]],
    latent_width: int,
    latent_height: int,
    n_levels: int,
    pad_to_n_bboxes: int | None = None,
    pad_to_n_total_bboxes: int | None = None,
    # For sdxl, there is no attention at first down block
    skip_top_level: bool = False,
    use_flex_attention: bool = True,
    no_conversion: bool = False,
) -> dict[int, LayerAttnMeta]:

    assert (
        latent_width * latent_height % 128 == 0
    ), "Latent size must be divisible by 128"

    layer_attn_meta_dict = {}
    n_bboxes_for_image = torch.tensor([b.shape[0] for b in bboxes])
    start_idx = 1 if skip_top_level else 0

    for idx in range(n_levels):
        if idx >= start_idx:
            attn_mask = get_layer_attention_mask(
                bboxes,
                adjacency,
                latent_width,
                latent_height,
                pad_to_n_bboxes=pad_to_n_bboxes,
                pad_to_n_total_bboxes=pad_to_n_total_bboxes,
            )
            size = latent_width * latent_height
            if not no_conversion:
                attn_mask = convert_and_stretch_mask(
                    attn_mask,
                    sequence_length=size,
                    encoder_attn_mask=None,
                    use_flex_attention=use_flex_attention,
                )
            batch_indices_flat, positions_flat = (
                get_batch_and_position_indices_for_concat_aggregate(
                    n_bboxes_for_image, size
                )
            )
            meta = LayerAttnMeta(
                agg_batch_indices_flat=batch_indices_flat,
                agg_positions_flat=positions_flat,
                layer_attn_mask=attn_mask,
            )
            layer_attn_meta_dict[size] = meta
        latent_width = latent_width // 2
        latent_height = latent_height // 2
    return layer_attn_meta_dict


def get_layer_mask_per_image(
    bboxes: torch.Tensor, adjacency: list[list[int]], width: int, height: int
) -> torch.BoolTensor:
    """
    Parameters
    ----------
    bboxes
        A tensor of shape [n_bboxes, 4] representing the bounding boxes

    Returns
    -------
    The mask is of size [n_bboxes, height x width, n_bboxes]
    """

    device = bboxes.device
    n_bboxes = bboxes.shape[0]
    mask = torch.zeros([n_bboxes, height, width, n_bboxes], dtype=torch.bool).to(device)

    # Attention within each layer
    for i in range(len(mask)):
        mask[i, :, :, i] = True

    # get relative bbox for each edge
    # source is query (larger image) and target is key (smaller image)
    relative_bboxes = []
    for source, targets in enumerate(adjacency):
        for target in targets:
            relative_bbox = convert_to_relative_bbox(bboxes[target], bboxes[source])
            relative_bboxes.append(relative_bbox)
    relative_bboxes = torch.tensor(relative_bboxes)
    edge_masks = bboxes_to_mask(relative_bboxes, width, height)

    edge_counter = 0
    for source, targets in enumerate(adjacency):
        for target in targets:
            mask[source, :, :, target] = edge_masks[edge_counter]
            edge_counter += 1

    return mask.flatten(start_dim=1, end_dim=2)


def get_layer_attention_mask(
    bboxes: list[torch.Tensor],
    adjacency: list[list[list[int]]],
    width: int,
    height: int,
    pad_to_n_bboxes: int | None = None,
    pad_to_n_total_bboxes: int | None = None,
) -> torch.BoolTensor:
    """
    Parameters
    ----------
    bboxes
        A list of tensors of shape [n_bboxes, 4] representing the bounding boxes
    adjacency
        A list of adjacency lists. Each adjacency list is a list of lists representing
        the neighbors of the corresponding bounding box in bboxes.

    Returns
    -------
    The mask is of size
    [n_total_bboxes, height x width, n_bboxes]
    """
    max_n_bboxes = pad_to_n_bboxes or max(len(bbox) for bbox in bboxes)
    n_total_bboxes = pad_to_n_total_bboxes or sum(len(bbox) for bbox in bboxes)
    device = bboxes[0].device

    mask = torch.zeros(
        [n_total_bboxes, height * width, max_n_bboxes], dtype=torch.bool
    ).to(device)

    bbox_start = 0
    for i, bboxes_i in enumerate(bboxes):
        sequence_length_i = len(bboxes_i)
        mask[bbox_start : bbox_start + len(bboxes_i), :, :sequence_length_i] = (
            get_layer_mask_per_image(bboxes_i, adjacency[i], width, height)
        )
        bbox_start += len(bboxes_i)

    return mask


# For layer-region-attention


def make_layer_region_mask_dict(
    bboxes: list[torch.Tensor],
    adjacency: list[list[list[int]]],
    latent_width: int,
    latent_height: int,
    n_levels: int,
    sequence_length: int = 77,
    pad_to_n_bboxes: int | None = None,
    pad_to_n_total_bboxes: int | None = None,
    # For sdxl, there is no attention at first down block
    skip_top_level: bool = False,
    encoder_attn_mask: torch.BoolTensor | None = None,
    use_flex_attention: bool = True,
    no_conversion: bool = False,
) -> dict[int, Mask]:
    """
    Converts a set of relative bounding boxes and adjacency lists to a dictionary
    mapping feature map size to binary mask of size
    [n_total_bboxes, feature_height*feature_width, max_n_bboxes*sequence_length]
    """
    assert (
        latent_width * latent_height % 128 == 0
    ), "Latent size must be divisible by 128"

    if encoder_attn_mask is not None and not use_flex_attention:
        encoder_attn_mask = encoder_attn_mask.unsqueeze(1)

    layer_attn_meta_dict = {}
    descendants = [get_descendants(adj) for adj in adjacency]
    start_idx = 1 if skip_top_level else 0

    for idx in range(n_levels):
        if idx >= start_idx:
            attn_mask = get_layer_attention_mask(
                bboxes,
                descendants,
                latent_width,
                latent_height,
                pad_to_n_bboxes=pad_to_n_bboxes,
                pad_to_n_total_bboxes=pad_to_n_total_bboxes,
            )
            size = latent_width * latent_height
            if not no_conversion:
                attn_mask = convert_and_stretch_mask(
                    attn_mask, sequence_length, encoder_attn_mask, use_flex_attention
                )
            layer_attn_meta_dict[size] = attn_mask
        latent_width = latent_width // 2
        latent_height = latent_height // 2
    return layer_attn_meta_dict


def get_descendants(adjacency: list[list[int]]) -> list[list[int]]:
    """
    This only works for DAG
    """
    descendants = [None] * len(adjacency)

    def dfs(node):
        if descendants[node] is not None:
            return descendants[node]
        descendants[node] = set()
        for child in adjacency[node]:
            descendants[node].update(dfs(child))
        descendants[node].add(node)
        return descendants[node]

    for node in range(len(adjacency)):
        dfs(node)

    descendants = [sorted(x) for x in descendants]
    return descendants


# bbox utilities


Bbox = tuple[float, float, float, float]


def convert_to_relative_bbox(inner_bbox_global: Bbox, outer_bbox: Bbox) -> Bbox:
    """
    Converts a bounding box (inner_bbox_global) that is relative to
    the entire image into coordinates that are relative to
    another bounding box (outer_bbox).

    Args:
        outer_bbox:
            The outer bounding box relative to the entire image.
        inner_bbox_global:
            The inner bounding box relative to the entire image.

    Returns:
        A new Bbox instance representing the inner bounding box
        relative to the outer bounding box.
    """
    inner_left, inner_top, inner_right, inner_bottom = inner_bbox_global
    outer_left, outer_top, outer_right, outer_bottom = outer_bbox
    relative_left = (inner_left - outer_left) / (outer_right - outer_left)
    relative_top = (inner_top - outer_top) / (outer_bottom - outer_top)
    relative_right = (inner_right - outer_left) / (outer_right - outer_left)
    relative_bottom = (inner_bottom - outer_top) / (outer_bottom - outer_top)
    return relative_left, relative_top, relative_right, relative_bottom


def interpolate_mask(
    mask: torch.Tensor, target_width: int, target_height: int
) -> torch.BoolTensor:
    """
    Interpolates a binary mask to a target width and height
    using nearest interpolation.

    Parameters
    ----------
    mask : torch.Tensor
        A tensor of shape [batch_size, height, width] representing the input masks.
    target_width : int
        The desired width of the output mask.
    target_height : int
        The desired height of the output mask.

    Returns
    -------
    torch.Tensor
        A tensor of shape [batch_size, target_height, target_width],
        with interpolated masks, returned as a boolean tensor.

    Example
    -------
    >>> mask = torch.rand(4, 50, 50) > 0.5
    >>> result = interpolate_mask(mask, 100, 100)
    >>> result.shape
    torch.Size([4, 100, 100])
    """
    return (
        F.interpolate(mask.unsqueeze(1).half(), size=(target_height, target_width))
        .squeeze(1)
        .bool()
    )


def bboxes_to_mask(
    bboxes: torch.Tensor, target_width: int, target_height: int
) -> torch.BoolTensor:
    """
    Converts a set of relative bounding boxes to a binary mask of size
    [n_bboxes, target_height, target_width].

    Parameters
    ----------
    bboxes : torch.Tensor
        A tensor of shape [n_bboxes, 4] where each bounding box is represented
        by four relative coordinates [left, top, right, bottom].
        Each coordinate is a relative value between 0 and 1,
        where 0 corresponds to the left/top and 1 corresponds to the right/bottom.
    target_width : int
        The desired width of the output mask.
    target_height : int
        The desired height of the output mask.

    Returns
    -------
    torch.Tensor
        A tensor of shape [n_bboxes, target_height, target_width],
        where the regions corresponding to each bounding box are set to True.
        The rest of the mask is set to False.

    Example
    -------
    >>> bboxes = torch.tensor([[0.1, 0.2, 0.5, 0.6], [0.3, 0.4, 0.7, 0.8]])
    >>> result = bboxes_to_mask(bboxes, 100, 100)
    >>> result.shape
    torch.Size([2, 100, 100])
    """
    n_bboxes = bboxes.shape[0]
    mask = bboxes.new_zeros((n_bboxes, target_height, target_width)).bool()

    for i, bbox in enumerate(bboxes):
        left = int(target_width * bbox[0])
        top = int(target_height * bbox[1])
        right = int(target_width * bbox[2])
        bottom = int(target_height * bbox[3])

        if right > left and bottom > top:
            mask[i, top:bottom, left:right] = True

    return mask


if __name__ == "__main__":

    region_masks = [torch.randint(0, 2, (2, 8, 8)), torch.randint(0, 2, (4, 8, 8))]
    print(region_masks)
    mask = aggregate_region_masks(region_masks)
    print(mask)
    print(mask.shape)

    bboxes = [
        torch.tensor([[0, 0, 1, 1], [0.3, 0.5, 0.7, 1]]),
        torch.tensor([[0, 0, 1, 1], [0, 0, 1, 0.5], [0.5, 0.5, 1, 1]]),
    ]
    bbox_masks = [bboxes_to_mask(bbox, 8, 8) for bbox in bboxes]
    print(bbox_masks)
    mask = aggregate_region_masks(bbox_masks)
    print(mask)
    print(mask.shape)

    adjacency = [[[1], []], [[1, 2], [], []]]
    mask = get_layer_attention_mask(bboxes, adjacency, 2, 2)
    print(mask)
    print(mask.shape)

    mask = make_layer_region_mask_dict(
        bboxes, adjacency, 8, 8, n_levels=2, sequence_length=5, use_flex_attention=False
    )
    print(mask)
    print(mask[64].shape)
