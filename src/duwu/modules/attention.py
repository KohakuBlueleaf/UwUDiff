import torch
import torch.nn.functional as F
from torch.nn.attention.flex_attention import flex_attention
from diffusers.models.attention_processor import Attention
from einops import rearrange

from duwu.utils.aggregation import concat_aggregate_embeddings_vectorize
from duwu.modules.attn_masks import LayerAttnMeta


class CombinedAttnProcessor:

    def __init__(self, self_attn_processor, cross_attn_processor):
        self.self_attn_processor = self_attn_processor
        self.cross_attn_processor = cross_attn_processor

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor,
        attention_mask: torch.FloatTensor | None = None,
        temb: torch.FloatTensor | None = None,
        n_elements_per_image: list[int] | None = None,
        pad_to_n_elements: int | None = None,
        region_mask_dict: dict[int, torch.BoolTensor] | None = None,
        layer_attn_meta_dict: dict[int, LayerAttnMeta] | None = None,
    ):
        if encoder_hidden_states is None:
            return self.self_attn_processor(
                attn,
                hidden_states,
                n_elements_per_image=n_elements_per_image,
                pad_to_n_elements=pad_to_n_elements,
                temb=temb,
                layer_attn_meta_dict=layer_attn_meta_dict,
            )
        else:
            return self.cross_attn_processor(
                attn,
                hidden_states,
                encoder_hidden_states,
                attention_mask=attention_mask,
                temb=temb,
                region_mask_dict=region_mask_dict,
            )


class LayerAttnProcessor:

    def __init__(self, use_flex_attention: bool = True):
        if use_flex_attention:
            # dynamic needs to be set to False
            # compilation is needed
            self.flex_attention = torch.compile(flex_attention, dynamic=False)
        self.use_flex_attention = use_flex_attention

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        n_elements_per_image: list[int] | torch.LongTensor,
        layer_attn_meta_dict: dict[int, LayerAttnMeta],
        pad_to_n_elements: int | None = None,
        temb: torch.FloatTensor | None = None,
    ):
        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(
                batch_size, channel, height * width
            ).transpose(1, 2)

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(
                1, 2
            )

        batch_size, sequence_length, _ = hidden_states.shape
        meta = layer_attn_meta_dict[sequence_length]

        # Put layers of the same image together, which affects sequence_length
        hidden_states_agg = concat_aggregate_embeddings_vectorize(
            hidden_states,
            n_elements_per_image,
            pad_to_n_elements=pad_to_n_elements,
            batch_indices_flat=meta.agg_batch_indices_flat,
            positions_flat=meta.agg_positions_flat,
            cat_embeddings=meta.cat_embeddings,
        )
        # Allocate memory for the concatenated embedding
        if meta.cat_embeddings is None:
            meta.cat_embeddings = hidden_states_agg

        if not isinstance(n_elements_per_image, torch.Tensor):
            n_elements_per_image = torch.tensor(n_elements_per_image).to(
                hidden_states.device
            )

        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states_agg)
        value = attn.to_v(hidden_states_agg)

        key = key.repeat_interleave(n_elements_per_image, dim=0)
        value = value.repeat_interleave(n_elements_per_image, dim=0)

        query = rearrange(query, "b n (nh d) -> b nh n d", nh=attn.heads)
        key = rearrange(key, "b n (nh d) -> b nh n d", nh=attn.heads)
        value = rearrange(value, "b n (nh d) -> b nh n d", nh=attn.heads)

        attention_mask = meta.layer_attn_mask

        if self.use_flex_attention:
            hidden_states = self.flex_attention(
                query, key, value, block_mask=attention_mask
            )
        else:
            # Add head dimension
            # print(query.shape, key.shape, value.shape, attention_mask.shape)
            hidden_states = F.scaled_dot_product_attention(
                query, key, value, attn_mask=attention_mask.unsqueeze(1)
            )

        hidden_states = rearrange(hidden_states, "b nh n d -> b n (nh d)")

        out_proj, dropout = attn.to_out
        hidden_states = out_proj(hidden_states)
        hidden_states = dropout(hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(
                batch_size, channel, height, width
            )

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


class RegionAttnProcessor:
    """
    Region mask-based cross-attention
    Adapted from https://github.com/Birch-san/regional-attn/blob/main/src/attention/regional_attn.py  # noqa
    """

    def __init__(self, use_flex_attention: bool = True):
        if use_flex_attention:
            # dynamic needs to be set to False
            # compilation is needed
            self.flex_attention = torch.compile(flex_attention, dynamic=False)
        self.use_flex_attention = use_flex_attention

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor,
        attention_mask: torch.FloatTensor | None = None,
        temb: torch.FloatTensor | None = None,
        region_mask_dict: dict[int, torch.BoolTensor] | None = None,
    ) -> torch.FloatTensor:
        """
        Parameters
        ----------
        attention
            Attention module
        hidden_states
            Input tensor of size [batch_size, sequence_length_q, hidden_size_q]
            This represents the flattened image patches, where sequence_length_q
            is height * width
        encoder_hidden_states
            Text conditioning of size [batch_size, sequence_length_k, hidden_size_k]
            This represents the concatenated text embedding, where sequence_length_k
            is n_captions * max_caption_length
        attention_mask
            Encoder attention mask of size [batch_size, 1, sequence_length_k]
            It takes values in {0, -inf}
        temb
            time embedding
        region_mask_dict
            A dictionary mapping from sequence_length_q to region masks.
            Region mask should be of size
            [batch_size, sequence_length_q, sequence_length_k].
            Region mask should take boolean values.

        Returns
        -------
        torch.FloatTensor
        """

        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(
                batch_size, channel, height * width
            ).transpose(1, 2)

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(
                1, 2
            )

        if attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(
                encoder_hidden_states
            )

        query = attn.to_q(hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = rearrange(query, "b n (nh d) -> b nh n d", nh=attn.heads)
        key = rearrange(key, "b n (nh d) -> b nh n d", nh=attn.heads)
        value = rearrange(value, "b n (nh d) -> b nh n d", nh=attn.heads)

        if self.use_flex_attention:
            # Ignore attention mask input and assume it is already taken
            # into account in region_mask_dict
            if region_mask_dict is not None:
                attention_mask = region_mask_dict[hidden_states.shape[1]]
            else:
                attention_mask = None
            hidden_states = self.flex_attention(
                query, key, value, block_mask=attention_mask
            )
        else:
            if attention_mask is not None:
                # Convert from {0, -inf} to {True, False}
                attention_mask = (attention_mask == 0).bool()
            if region_mask_dict is not None:
                region_attention_mask = region_mask_dict[hidden_states.shape[1]].bool()
                if attention_mask is None:
                    attention_mask = region_attention_mask
                else:
                    attention_mask = torch.logical_and(
                        attention_mask, region_attention_mask
                    )
            # Add head dimension
            if attention_mask is not None:
                attention_mask = attention_mask.unsqueeze(1)
            hidden_states = F.scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask=attention_mask,
                dropout_p=0.0,
                is_causal=False,
            )
        hidden_states = rearrange(hidden_states, "b nh n d -> b n (nh d)")

        out_proj, dropout = attn.to_out
        hidden_states = out_proj(hidden_states)
        hidden_states = dropout(hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(
                batch_size, channel, height, width
            )

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states
