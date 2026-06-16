# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from collections.abc import Iterable
from functools import lru_cache
from math import prod
from typing import TYPE_CHECKING, Any

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

# TODO replace this with vLLM implementation
from diffusers.models.embeddings import TimestepEmbedding, Timesteps
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.normalization import AdaLayerNormContinuous
from vllm.logger import init_logger
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader

if TYPE_CHECKING:
    from vllm.model_executor.layers.quantization.base_config import (
        QuantizationConfig,
    )

from vllm_omni.diffusion.attention.backends.abstract import (
    AttentionMetadata,
)
from vllm_omni.diffusion.attention.backends.ring_flash_attn import (
    ring_flash_attn_varlen_replicated_prefix_plan,
    ring_flash_attn_varlen_with_replicated_prefix_func,
)
from vllm_omni.diffusion.attention.layer import Attention
from vllm_omni.diffusion.attention.parallel.ulysses import (
    ulysses_flat_varlen_attention,
    ulysses_flat_varlen_attention_with_replicated_prefix,
)
from vllm_omni.diffusion.cache.base import CachedTransformer
from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.diffusion.distributed.hsdp_utils import is_transformer_block_module
from vllm_omni.diffusion.distributed.sp_plan import (
    SequenceParallelInput,
    SequenceParallelOutput,
)
from vllm_omni.diffusion.forward_context import get_forward_context, is_forward_context_available
from vllm_omni.diffusion.layers.adalayernorm import AdaLayerNorm
from vllm_omni.diffusion.layers.rope import RotaryEmbedding

logger = init_logger(__name__)


def _join_prefix(prefix: str, suffix: str) -> str:
    return f"{prefix}.{suffix}" if prefix else suffix


def _get_qwen_image_attention_metadata(attention_id: str | None) -> AttentionMetadata | None:
    if attention_id is None or not is_forward_context_available():
        return None
    metadata = get_forward_context().attn_metadata
    if isinstance(metadata, dict):
        return metadata.get(attention_id)
    return None


def _qwen_image_token_to_req(lengths: list[int], device: torch.device) -> torch.Tensor:
    return torch.repeat_interleave(
        torch.arange(len(lengths), dtype=torch.long, device=device),
        torch.tensor(lengths, dtype=torch.long, device=device),
    )


def _qwen_image_packed_indices(
    image_lens: list[int],
    text_lens: list[int],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    total_text = sum(text_lens)
    text_offset = 0
    image_offset = 0
    joint_offset = 0
    gather_indices: list[torch.Tensor] = []
    text_indices: list[torch.Tensor] = []
    image_indices: list[torch.Tensor] = []
    for text_len, image_len in zip(text_lens, image_lens, strict=True):
        gather_indices.append(torch.arange(text_offset, text_offset + text_len, dtype=torch.long, device=device))
        gather_indices.append(
            torch.arange(
                total_text + image_offset,
                total_text + image_offset + image_len,
                dtype=torch.long,
                device=device,
            )
        )
        text_indices.append(torch.arange(joint_offset, joint_offset + text_len, dtype=torch.long, device=device))
        image_indices.append(
            torch.arange(
                joint_offset + text_len,
                joint_offset + text_len + image_len,
                dtype=torch.long,
                device=device,
            )
        )
        text_offset += text_len
        image_offset += image_len
        joint_offset += text_len + image_len

    return {
        "image_token_to_req": _qwen_image_token_to_req(image_lens, device),
        "text_token_to_req": _qwen_image_token_to_req(text_lens, device),
        "joint_gather_idx": torch.cat(gather_indices, dim=0),
        "joint_text_idx": torch.cat(text_indices, dim=0),
        "joint_image_idx": torch.cat(image_indices, dim=0),
    }


def _qwen_image_sp_context() -> tuple[dist.ProcessGroup, int, int] | None:
    if not is_forward_context_available():
        return None
    cfg = get_forward_context().omni_diffusion_config
    if cfg is None:
        return None
    sp_size = int(getattr(cfg.parallel_config, "sequence_parallel_size", 1) or 1)
    if sp_size <= 1:
        return None
    if int(getattr(cfg.parallel_config, "ring_degree", 1) or 1) != 1:
        raise ValueError("Qwen-Image packed dynamic batching only supports pure Ulysses SP (ring_degree=1).")
    from vllm_omni.diffusion.distributed.parallel_state import get_sp_group

    sp_group = get_sp_group()
    return sp_group.ulysses_group, int(sp_group.ulysses_world_size), int(sp_group.ulysses_rank)


def _qwen_image_dynamic_sp_context() -> tuple[str, dist.ProcessGroup, int, int] | None:
    if not is_forward_context_available():
        return None
    cfg = get_forward_context().omni_diffusion_config
    if cfg is None:
        return None
    parallel_config = cfg.parallel_config
    sp_size = int(getattr(parallel_config, "sequence_parallel_size", 1) or 1)
    if sp_size <= 1:
        return None
    ulysses_degree = int(getattr(parallel_config, "ulysses_degree", 1) or 1)
    ring_degree = int(getattr(parallel_config, "ring_degree", 1) or 1)
    if ulysses_degree > 1 and ring_degree > 1:
        raise ValueError("Qwen-Image packed dynamic batching does not support hybrid Ulysses+Ring SP yet.")

    from vllm_omni.diffusion.distributed.parallel_state import get_sp_group

    sp_group = get_sp_group()
    if ring_degree > 1:
        return "ring", sp_group.ring_group, int(sp_group.ring_world_size), int(sp_group.ring_rank)
    return "ulysses", sp_group.ulysses_group, int(sp_group.ulysses_world_size), int(sp_group.ulysses_rank)


def _qwen_image_balanced_ranges(total_len: int, parts: int) -> list[tuple[int, int]]:
    base, remainder = divmod(int(total_len), int(parts))
    ranges: list[tuple[int, int]] = []
    start = 0
    for rank in range(parts):
        length = base + (1 if rank < remainder else 0)
        end = start + length
        ranges.append((start, end))
        start = end
    return ranges


def _qwen_image_source_order_indices(
    *,
    total_text: int,
    total_image: int,
    text_ranges: list[tuple[int, int]],
    image_ranges: list[tuple[int, int]],
    joint_gather_idx: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    concat_to_source = torch.empty(total_text + total_image, dtype=torch.long, device=device)
    source_offset = 0
    for text_range, image_range in zip(text_ranges, image_ranges, strict=True):
        text_start, text_end = text_range
        text_len = text_end - text_start
        if text_len:
            concat_to_source[text_start:text_end] = torch.arange(
                source_offset,
                source_offset + text_len,
                dtype=torch.long,
                device=device,
            )
        source_offset += text_len

        image_start, image_end = image_range
        image_len = image_end - image_start
        if image_len:
            concat_start = total_text + image_start
            concat_to_source[concat_start : concat_start + image_len] = torch.arange(
                source_offset,
                source_offset + image_len,
                dtype=torch.long,
                device=device,
            )
        source_offset += image_len

    source_to_joint = concat_to_source.index_select(0, joint_gather_idx)
    joint_to_source = torch.empty_like(source_to_joint)
    joint_to_source.scatter_(0, source_to_joint, torch.arange(source_to_joint.numel(), dtype=torch.long, device=device))
    return source_to_joint, joint_to_source


def _qwen_image_rank_local_indices(
    lengths: list[int],
    *,
    rank: int,
    world_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, list[int]]:
    pieces: list[torch.Tensor] = []
    local_lens: list[int] = []
    offset = 0
    for length in lengths:
        ranges = _qwen_image_balanced_ranges(int(length), world_size)
        start, end = ranges[rank]
        local_lens.append(end - start)
        if end > start:
            pieces.append(torch.arange(offset + start, offset + end, dtype=torch.long, device=device))
        offset += int(length)
    if not pieces:
        return torch.empty(0, dtype=torch.long, device=device), local_lens
    return torch.cat(pieces, dim=0), local_lens


def _qwen_image_cu_seqlens(lengths: list[int], device: torch.device) -> tuple[torch.Tensor, int]:
    cu_seqlens = torch.tensor(
        [0, *torch.tensor(lengths, dtype=torch.int32).cumsum(0).tolist()],
        dtype=torch.int32,
        device=device,
    )
    return cu_seqlens, max(lengths) if lengths else 0


def _qwen_image_ring_flat_varlen_attention_with_replicated_prefix(
    pg: dist.ProcessGroup,
    prefix_query: torch.Tensor,
    prefix_key: torch.Tensor,
    prefix_value: torch.Tensor,
    suffix_query: torch.Tensor,
    suffix_key: torch.Tensor,
    suffix_value: torch.Tensor,
    *,
    text_lens: list[int],
    suffix_lens_by_rank: list[list[int]],
    rank: int,
    softmax_scale: float | None,
    varlen_plan: dict[str, Any] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    return ring_flash_attn_varlen_with_replicated_prefix_func(
        prefix_query,
        prefix_key,
        prefix_value,
        suffix_query,
        suffix_key,
        suffix_value,
        prefix_lens=text_lens,
        suffix_lens_by_rank=suffix_lens_by_rank,
        rank=rank,
        softmax_scale=softmax_scale,
        causal=False,
        group=pg,
        varlen_plan=varlen_plan,
    )


def _qwen_image_sp_source_order_indices(
    *,
    text_lens: list[int],
    image_lens: list[int],
    world_size: int,
    joint_gather_idx: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    total_text = sum(text_lens)
    total_image = sum(image_lens)
    source_pieces: list[torch.Tensor] = []
    image_source_pieces: list[torch.Tensor] = []
    for rank in range(world_size):
        text_indices, _ = _qwen_image_rank_local_indices(
            text_lens,
            rank=rank,
            world_size=world_size,
            device=device,
        )
        image_indices, _ = _qwen_image_rank_local_indices(
            image_lens,
            rank=rank,
            world_size=world_size,
            device=device,
        )
        source_pieces.append(text_indices)
        source_pieces.append(total_text + image_indices)
        image_source_pieces.append(image_indices)

    source_order = torch.cat(source_pieces, dim=0)
    concat_to_source = torch.empty(total_text + total_image, dtype=torch.long, device=device)
    concat_to_source.scatter_(0, source_order, torch.arange(source_order.numel(), dtype=torch.long, device=device))
    source_to_joint = concat_to_source.index_select(0, joint_gather_idx)
    joint_to_source = torch.empty_like(source_to_joint)
    joint_to_source.scatter_(0, source_to_joint, torch.arange(source_to_joint.numel(), dtype=torch.long, device=device))

    image_source_order = torch.cat(image_source_pieces, dim=0)
    image_packed_to_source = torch.empty(total_image, dtype=torch.long, device=device)
    image_packed_to_source.scatter_(
        0,
        image_source_order,
        torch.arange(image_source_order.numel(), dtype=torch.long, device=device),
    )
    return source_to_joint, joint_to_source, image_packed_to_source


def _qwen_image_sp_replicated_text_source_order_indices(
    *,
    text_lens: list[int],
    image_lens: list[int],
    world_size: int,
    joint_gather_idx: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    total_text = sum(text_lens)
    total_image = sum(image_lens)
    image_source_pieces: list[torch.Tensor] = []
    for rank in range(world_size):
        image_indices, _ = _qwen_image_rank_local_indices(
            image_lens,
            rank=rank,
            world_size=world_size,
            device=device,
        )
        image_source_pieces.append(image_indices)

    image_source_order = torch.cat(image_source_pieces, dim=0)
    source_order = torch.cat(
        [
            torch.arange(total_text, dtype=torch.long, device=device),
            total_text + image_source_order,
        ],
        dim=0,
    )
    concat_to_source = torch.empty(total_text + total_image, dtype=torch.long, device=device)
    concat_to_source.scatter_(0, source_order, torch.arange(source_order.numel(), dtype=torch.long, device=device))
    source_to_joint = concat_to_source.index_select(0, joint_gather_idx)
    joint_to_source = torch.empty_like(source_to_joint)
    joint_to_source.scatter_(0, source_to_joint, torch.arange(source_to_joint.numel(), dtype=torch.long, device=device))

    image_packed_to_source = torch.empty(total_image, dtype=torch.long, device=device)
    image_packed_to_source.scatter_(
        0,
        image_source_order,
        torch.arange(image_source_order.numel(), dtype=torch.long, device=device),
    )
    return source_to_joint, joint_to_source, image_packed_to_source


def _qwen_image_all_gather_varlen(x: torch.Tensor, group: dist.ProcessGroup) -> torch.Tensor:
    world_size = dist.get_world_size(group)
    if world_size == 1:
        return x

    local_len = torch.tensor([int(x.shape[0])], dtype=torch.int64, device=x.device)
    gathered_lens = [torch.empty_like(local_len) for _ in range(world_size)]
    dist.all_gather(gathered_lens, local_len, group=group)
    lengths = [int(length.item()) for length in gathered_lens]
    max_len = max(lengths)

    if int(x.shape[0]) < max_len:
        pad_shape = (max_len - int(x.shape[0]), *x.shape[1:])
        x = torch.cat([x, torch.zeros(pad_shape, dtype=x.dtype, device=x.device)], dim=0)

    gathered = [torch.empty_like(x) for _ in range(world_size)]
    dist.all_gather(gathered, x.contiguous(), group=group)
    return torch.cat([part[:length] for part, length in zip(gathered, lengths, strict=True)], dim=0)


def _qwen_image_unwrap_module_output(output: torch.Tensor | tuple[torch.Tensor, Any]) -> torch.Tensor:
    return output[0] if isinstance(output, tuple) else output


def _qwen_image_joint_cat(
    text: torch.Tensor,
    image: torch.Tensor,
    text_lens: list[int],
    image_lens: list[int],
) -> torch.Tensor:
    if len(text_lens) == 1:
        return torch.cat([text, image], dim=0)

    pieces: list[torch.Tensor] = []
    text_offset = 0
    image_offset = 0
    for text_len, image_len in zip(text_lens, image_lens, strict=True):
        pieces.append(text[text_offset : text_offset + text_len])
        pieces.append(image[image_offset : image_offset + image_len])
        text_offset += text_len
        image_offset += image_len
    return torch.cat(pieces, dim=0)


def _qwen_image_split_joint(
    joint: torch.Tensor,
    text_lens: list[int],
    image_lens: list[int],
) -> tuple[torch.Tensor, torch.Tensor]:
    if len(text_lens) == 1:
        text_len = int(text_lens[0])
        return joint[:text_len], joint[text_len:]

    text_parts: list[torch.Tensor] = []
    image_parts: list[torch.Tensor] = []
    offset = 0
    for text_len, image_len in zip(text_lens, image_lens, strict=True):
        text_parts.append(joint[offset : offset + text_len])
        offset += text_len
        image_parts.append(joint[offset : offset + image_len])
        offset += image_len
    return torch.cat(text_parts, dim=0), torch.cat(image_parts, dim=0)


class ImageRopePrepare(nn.Module):
    """Prepares image hidden_states and RoPE embeddings for sequence parallel.

    This module encapsulates the input linear projection and RoPE computation.
    Similar to Z-Image's UnifiedPrepare, this creates a module boundary where
    _sp_plan can shard outputs via split_output=True.

    The key insight is that hidden_states and vid_freqs must be sharded together
    to maintain dimension alignment for RoPE computation in attention layers.

    Note: Our _sp_plan corresponds to diffusers' _cp_plan (Context Parallelism).
    """

    def __init__(self, img_in: nn.Module, pos_embed: nn.Module):
        super().__init__()
        self.img_in = img_in
        self.pos_embed = pos_embed

    def forward(
        self,
        hidden_states: torch.Tensor,
        img_shapes: list[tuple[int, int, int]],
        txt_seq_lens: list[int],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Prepare hidden_states and RoPE for SP.

        Args:
            hidden_states: [batch, img_seq_len, channels]
            img_shapes: List of (frame, height, width) tuples
            txt_seq_lens: List of text sequence lengths

        Returns:
            hidden_states: Processed hidden states [batch, img_seq_len, dim]
            vid_freqs: Image RoPE frequencies [img_seq_len, rope_dim]
            txt_freqs: Text RoPE frequencies [txt_seq_len, rope_dim]

        Note: _sp_plan will shard hidden_states and vid_freqs via split_output=True
              txt_freqs is kept replicated for dual-stream attention
        """
        # Apply input projection
        hidden_states = self.img_in(hidden_states)

        # Compute RoPE embeddings
        image_rotary_emb = self.pos_embed(img_shapes, txt_seq_lens, device=hidden_states.device)
        vid_freqs, txt_freqs = image_rotary_emb

        return hidden_states, vid_freqs, txt_freqs


class ModulateIndexPrepare(nn.Module):
    """Prepares modulate_index for sequence parallel when zero_cond_t is enabled.

    This module encapsulates the creation of modulate_index tensor, which is used
    to select different conditioning parameters (shift/scale/gate) for different
    token positions in image editing tasks.

    Similar to Z-Image's UnifiedPrepare and ImageRopePrepare, this creates a module
    boundary where _sp_plan can shard the output via split_output=True.

    The modulate_index must be sharded along the sequence dimension to match the
    sharded hidden_states in SP mode.

    Note: Our _sp_plan corresponds to diffusers' _cp_plan (Context Parallelism).
    """

    def __init__(self, zero_cond_t: bool = False):
        super().__init__()
        self.zero_cond_t = zero_cond_t

    def forward(
        self,
        timestep: torch.Tensor,
        img_shapes: list[list[tuple[int, int, int]]],
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Prepare timestep and modulate_index for SP.

        Args:
            timestep: Timestep tensor [batch]
            img_shapes: List of image shape tuples per batch item.
                Each item is a list of (frame, height, width) tuples.
                For edit models: [[source_shape], [target_shape1, target_shape2, ...]]

        Returns:
            timestep: Doubled timestep if zero_cond_t, else original [batch] or [2*batch]
            modulate_index: Token condition index [batch, seq_len] if zero_cond_t, else None
                - index=0: source image tokens (use normal timestep conditioning)
                - index=1: target image tokens (use zero timestep conditioning)

        Note: _sp_plan will shard modulate_index via split_output=True when SP is enabled.
              The modulate_index sequence dimension must match hidden_states after sharding.
        """
        if self.zero_cond_t:
            # Double the timestep: [timestep, timestep * 0]
            # This creates two sets of conditioning parameters in AdaLayerNorm
            timestep = torch.cat([timestep, timestep * 0], dim=0)

            # Create modulate_index to select conditioning per token position
            # - First image (sample[0]): source image, use index=0 (normal timestep)
            # - Remaining images (sample[1:]): target images, use index=1 (zero timestep)
            modulate_index = torch.tensor(
                [[0] * prod(sample[0]) + [1] * sum([prod(s) for s in sample[1:]]) for sample in img_shapes],
                device=timestep.device,
                dtype=torch.int,
            )
            return timestep, modulate_index

        return timestep, None


class QwenTimestepProjEmbeddings(nn.Module):
    def __init__(
        self,
        embedding_dim,
        use_additional_t_cond: bool = False,
        quant_config: QuantizationConfig | None = None,
    ):
        super().__init__()

        self.time_proj = Timesteps(num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=0, scale=1000)
        self.timestep_embedder = TimestepEmbedding(in_channels=256, time_embed_dim=embedding_dim)
        self.timestep_embedder.linear_1 = ReplicatedLinear(
            256,
            embedding_dim,
            bias=True,
            return_bias=False,
            quant_config=None,
            prefix="timestep_embedder.linear_1",
        )
        self.timestep_embedder.linear_2 = ReplicatedLinear(
            embedding_dim,
            embedding_dim,
            bias=True,
            return_bias=False,
            quant_config=None,
            prefix="timestep_embedder.linear_2",
        )
        self.use_additional_t_cond = use_additional_t_cond
        if use_additional_t_cond:
            self.addition_t_embedding = nn.Embedding(2, embedding_dim)

    def forward(self, timestep, hidden_states, addition_t_cond=None):
        timesteps_proj = self.time_proj(timestep)
        timesteps_emb = self.timestep_embedder(timesteps_proj.to(dtype=hidden_states.dtype))  # (N, D)

        conditioning = timesteps_emb
        if self.use_additional_t_cond:
            if addition_t_cond is None:
                raise ValueError("When additional_t_cond is True, addition_t_cond must be provided.")
            addition_t_emb = self.addition_t_embedding(addition_t_cond)
            addition_t_emb = addition_t_emb.to(dtype=hidden_states.dtype)
            conditioning = conditioning + addition_t_emb

        return conditioning


class QwenEmbedLayer3DRope(nn.Module):
    def __init__(self, theta: int, axes_dim: list[int], scale_rope=False):
        super().__init__()
        self.theta = theta
        self.axes_dim = axes_dim
        pos_index = torch.arange(4096)
        neg_index = torch.arange(4096).flip(0) * -1 - 1
        self.pos_freqs = torch.cat(
            [
                self.rope_params(pos_index, self.axes_dim[0], self.theta),
                self.rope_params(pos_index, self.axes_dim[1], self.theta),
                self.rope_params(pos_index, self.axes_dim[2], self.theta),
            ],
            dim=1,
        )
        self.neg_freqs = torch.cat(
            [
                self.rope_params(neg_index, self.axes_dim[0], self.theta),
                self.rope_params(neg_index, self.axes_dim[1], self.theta),
                self.rope_params(neg_index, self.axes_dim[2], self.theta),
            ],
            dim=1,
        )

        self.scale_rope = scale_rope

    def rope_params(self, index, dim, theta=10000):
        """
        Args:
            index: [0, 1, 2, 3] 1D Tensor representing the position index of the token
        """
        assert dim % 2 == 0
        freqs = torch.outer(index, 1.0 / torch.pow(theta, torch.arange(0, dim, 2).to(torch.float32).div(dim)))
        freqs = torch.polar(torch.ones_like(freqs), freqs)
        return freqs

    def forward(self, video_fhw, txt_seq_lens, device):
        """
        Args: video_fhw: [frame, height, width] a list of 3 integers representing the shape of the video Args:
        txt_length: [bs] a list of 1 integers representing the length of the text
        """
        if self.pos_freqs.device != device:
            self.pos_freqs = self.pos_freqs.to(device)
            self.neg_freqs = self.neg_freqs.to(device)

        if isinstance(video_fhw, list):
            video_fhw = video_fhw[0]
        if not isinstance(video_fhw, list):
            video_fhw = [video_fhw]

        vid_freqs = []
        max_vid_index = 0
        layer_num = len(video_fhw) - 1
        for idx, fhw in enumerate(video_fhw):
            frame, height, width = fhw
            if idx != layer_num:
                video_freq = self._compute_video_freqs(frame, height, width, idx)
            else:
                ### For the condition image, we set the layer index to -1
                video_freq = self._compute_condition_freqs(frame, height, width)
            video_freq = video_freq.to(device)
            vid_freqs.append(video_freq)

            if self.scale_rope:
                max_vid_index = max(height // 2, width // 2, max_vid_index)
            else:
                max_vid_index = max(height, width, max_vid_index)

        max_vid_index = max(max_vid_index, layer_num)
        max_len = max(txt_seq_lens)
        txt_freqs = self.pos_freqs[max_vid_index : max_vid_index + max_len, ...]
        vid_freqs = torch.cat(vid_freqs, dim=0)

        return vid_freqs, txt_freqs

    @lru_cache(maxsize=16)
    def _compute_video_freqs(self, frame, height, width, idx=0):
        seq_lens = frame * height * width
        freqs_pos = self.pos_freqs.split([x // 2 for x in self.axes_dim], dim=1)
        freqs_neg = self.neg_freqs.split([x // 2 for x in self.axes_dim], dim=1)

        freqs_frame = freqs_pos[0][idx : idx + frame].view(frame, 1, 1, -1).expand(frame, height, width, -1)
        if self.scale_rope:
            freqs_height = torch.cat([freqs_neg[1][-(height - height // 2) :], freqs_pos[1][: height // 2]], dim=0)
            freqs_height = freqs_height.view(1, height, 1, -1).expand(frame, height, width, -1)
            freqs_width = torch.cat([freqs_neg[2][-(width - width // 2) :], freqs_pos[2][: width // 2]], dim=0)
            freqs_width = freqs_width.view(1, 1, width, -1).expand(frame, height, width, -1)
        else:
            freqs_height = freqs_pos[1][:height].view(1, height, 1, -1).expand(frame, height, width, -1)
            freqs_width = freqs_pos[2][:width].view(1, 1, width, -1).expand(frame, height, width, -1)

        freqs = torch.cat([freqs_frame, freqs_height, freqs_width], dim=-1).reshape(seq_lens, -1)
        return freqs.clone().contiguous()

    @lru_cache(maxsize=16)
    def _compute_condition_freqs(self, frame, height, width):
        seq_lens = frame * height * width
        freqs_pos = self.pos_freqs.split([x // 2 for x in self.axes_dim], dim=1)
        freqs_neg = self.neg_freqs.split([x // 2 for x in self.axes_dim], dim=1)

        freqs_frame = freqs_neg[0][-1:].view(frame, 1, 1, -1).expand(frame, height, width, -1)
        if self.scale_rope:
            freqs_height = torch.cat([freqs_neg[1][-(height - height // 2) :], freqs_pos[1][: height // 2]], dim=0)
            freqs_height = freqs_height.view(1, height, 1, -1).expand(frame, height, width, -1)
            freqs_width = torch.cat([freqs_neg[2][-(width - width // 2) :], freqs_pos[2][: width // 2]], dim=0)
            freqs_width = freqs_width.view(1, 1, width, -1).expand(frame, height, width, -1)
        else:
            freqs_height = freqs_pos[1][:height].view(1, height, 1, -1).expand(frame, height, width, -1)
            freqs_width = freqs_pos[2][:width].view(1, 1, width, -1).expand(frame, height, width, -1)

        freqs = torch.cat([freqs_frame, freqs_height, freqs_width], dim=-1).reshape(seq_lens, -1)
        return freqs.clone().contiguous()


class QwenEmbedRope(nn.Module):
    def __init__(self, theta: int, axes_dim: list[int], scale_rope=False):
        super().__init__()
        self.theta = theta
        self.axes_dim = axes_dim
        pos_index = torch.arange(4096)
        neg_index = torch.arange(4096).flip(0) * -1 - 1
        self.pos_freqs = torch.cat(
            [
                self.rope_params(pos_index, self.axes_dim[0], self.theta),
                self.rope_params(pos_index, self.axes_dim[1], self.theta),
                self.rope_params(pos_index, self.axes_dim[2], self.theta),
            ],
            dim=1,
        )
        self.neg_freqs = torch.cat(
            [
                self.rope_params(neg_index, self.axes_dim[0], self.theta),
                self.rope_params(neg_index, self.axes_dim[1], self.theta),
                self.rope_params(neg_index, self.axes_dim[2], self.theta),
            ],
            dim=1,
        )

        # DO NOT USING REGISTER BUFFER HERE, IT WILL CAUSE COMPLEX NUMBERS LOSE ITS IMAGINARY PART
        self.scale_rope = scale_rope

    def rope_params(self, index: torch.Tensor, dim: int, theta: int = 10000):
        """
        Args:
            index (`torch.Tensor`): [0, 1, 2, 3] 1D Tensor representing the position index of the token
            dim (`int`): Dimension for the rope parameters
            theta (`int`): Theta parameter for rope
        """
        assert dim % 2 == 0
        freqs = torch.outer(
            index,
            1.0 / torch.pow(theta, torch.arange(0, dim, 2).to(torch.float32).div(dim)),
        )
        freqs = torch.polar(torch.ones_like(freqs), freqs)
        return freqs

    def forward(self, video_fhw, txt_seq_lens, device):
        """
        Args: video_fhw: [frame, height, width] a list of 3 integers representing the shape of the video Args:
        txt_length: [bs] a list of 1 integers representing the length of the text
        """
        if self.pos_freqs.device != device:
            self.pos_freqs = self.pos_freqs.to(device)
            self.neg_freqs = self.neg_freqs.to(device)

        if isinstance(video_fhw, list):
            video_fhw = video_fhw[0]
        if not isinstance(video_fhw, list):
            video_fhw = [video_fhw]

        vid_freqs = []
        max_vid_index = 0
        for idx, fhw in enumerate(video_fhw):
            frame, height, width = fhw
            video_freq = self._compute_video_freqs(frame, height, width, idx)
            video_freq = video_freq.to(device)
            vid_freqs.append(video_freq)

            if self.scale_rope:
                max_vid_index = max(height // 2, width // 2, max_vid_index)
            else:
                max_vid_index = max(height, width, max_vid_index)

        max_len = max(txt_seq_lens)
        txt_freqs = self.pos_freqs[max_vid_index : max_vid_index + max_len, ...]
        vid_freqs = torch.cat(vid_freqs, dim=0)

        return vid_freqs, txt_freqs

    @lru_cache(maxsize=16)
    def _compute_video_freqs(self, frame, height, width, idx=0):
        seq_lens = frame * height * width
        freqs_pos = self.pos_freqs.split([x // 2 for x in self.axes_dim], dim=1)
        freqs_neg = self.neg_freqs.split([x // 2 for x in self.axes_dim], dim=1)

        freqs_frame = freqs_pos[0][idx : idx + frame].view(frame, 1, 1, -1).expand(frame, height, width, -1)
        if self.scale_rope:
            freqs_height = torch.cat(
                [freqs_neg[1][-(height - height // 2) :], freqs_pos[1][: height // 2]],
                dim=0,
            )
            freqs_height = freqs_height.view(1, height, 1, -1).expand(frame, height, width, -1)
            freqs_width = torch.cat(
                [freqs_neg[2][-(width - width // 2) :], freqs_pos[2][: width // 2]],
                dim=0,
            )
            freqs_width = freqs_width.view(1, 1, width, -1).expand(frame, height, width, -1)
        else:
            freqs_height = freqs_pos[1][:height].view(1, height, 1, -1).expand(frame, height, width, -1)
            freqs_width = freqs_pos[2][:width].view(1, 1, width, -1).expand(frame, height, width, -1)

        freqs = torch.cat([freqs_frame, freqs_height, freqs_width], dim=-1).reshape(seq_lens, -1)
        return freqs.clone().contiguous()


class ColumnParallelApproxGELU(nn.Module):
    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        *,
        approximate: str,
        bias: bool = True,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()
        self.proj = ColumnParallelLinear(
            dim_in,
            dim_out,
            bias=bias,
            gather_output=False,
            return_bias=False,
            quant_config=quant_config,
            prefix=prefix,
        )
        self.approximate = approximate

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        return F.gelu(x, approximate=self.approximate)


class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        dim_out: int | None = None,
        mult: int = 4,
        activation_fn: str = "gelu-approximate",
        inner_dim: int | None = None,
        bias: bool = True,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()

        assert activation_fn == "gelu-approximate", "Only gelu-approximate is supported."

        inner_dim = inner_dim or int(dim * mult)
        dim_out = dim_out or dim

        layers: list[nn.Module] = [
            ColumnParallelApproxGELU(
                dim,
                inner_dim,
                approximate="tanh",
                bias=bias,
                quant_config=quant_config,
                prefix=_join_prefix(prefix, "net.0.proj"),
            ),
            nn.Identity(),  # placeholder for weight loading
            RowParallelLinear(
                inner_dim,
                dim_out,
                input_is_parallel=True,
                return_bias=False,
                quant_config=quant_config,
                prefix=_join_prefix(prefix, "net.2"),
            ),
        ]

        self.net = nn.ModuleList(layers)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        for module in self.net:
            hidden_states = module(hidden_states)
        return hidden_states


class QwenImageCrossAttention(nn.Module):
    def __init__(
        self,
        dim: int,  # query_dim
        num_heads: int,
        head_dim: int,
        added_kv_proj_dim: int,
        window_size: tuple[int, int] = (-1, -1),
        out_bias: bool = True,
        qk_norm: bool = True,
        eps: float = 1e-6,
        pre_only: bool = False,
        context_pre_only: bool = False,
        out_dim: int | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0

        self.dim = dim
        self.head_dim = head_dim
        self.total_num_heads = num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.eps = eps

        self.to_qkv = QKVParallelLinear(
            hidden_size=dim,
            head_size=self.head_dim,
            total_num_heads=num_heads,
            quant_config=quant_config,
            prefix=_join_prefix(prefix, "to_qkv"),
        )
        self.query_num_heads = self.to_qkv.num_heads
        self.kv_num_heads = self.to_qkv.num_kv_heads

        self.norm_q = nn.RMSNorm(head_dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = nn.RMSNorm(head_dim, eps=eps) if qk_norm else nn.Identity()

        self.inner_dim = out_dim if out_dim is not None else head_dim * self.total_num_heads

        assert context_pre_only is not None
        self.add_kv_proj = QKVParallelLinear(
            hidden_size=added_kv_proj_dim,
            head_size=head_dim,
            total_num_heads=num_heads,
            quant_config=quant_config,
            prefix=_join_prefix(prefix, "add_kv_proj"),
        )
        self.add_query_num_heads = self.add_kv_proj.num_heads
        self.add_kv_num_heads = self.add_kv_proj.num_kv_heads

        assert not context_pre_only
        self.to_add_out = RowParallelLinear(
            self.inner_dim,
            self.dim,
            bias=out_bias,
            input_is_parallel=True,
            return_bias=False,
            quant_config=quant_config,
            prefix=_join_prefix(prefix, "to_add_out"),
        )

        assert not pre_only
        self.to_out = RowParallelLinear(
            self.inner_dim,
            self.dim,
            bias=out_bias,
            input_is_parallel=True,
            return_bias=False,
            quant_config=quant_config,
            prefix=_join_prefix(prefix, "to_out"),
        )

        self.norm_added_q = nn.RMSNorm(head_dim, eps=eps)
        self.norm_added_k = nn.RMSNorm(head_dim, eps=eps)

        self.attn = Attention(
            num_heads=self.query_num_heads,
            head_size=self.head_dim,
            softmax_scale=1.0 / (self.head_dim**0.5),
            causal=False,
            num_kv_heads=self.kv_num_heads,
        )
        self.rope = RotaryEmbedding(is_neox_style=False)

        try:
            config = get_forward_context().omni_diffusion_config
            self.parallel_config = config.parallel_config
        except Exception:
            self.parallel_config = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        vid_freqs: torch.Tensor,
        txt_freqs: torch.Tensor,
        hidden_states_mask: torch.Tensor | None = None,
        encoder_hidden_states_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        img_qkv, _ = self.to_qkv(hidden_states)
        q_size = self.query_num_heads * self.head_dim
        kv_size = self.kv_num_heads * self.head_dim
        img_query, img_key, img_value = img_qkv.split([q_size, kv_size, kv_size], dim=-1)

        txt_qkv, _ = self.add_kv_proj(encoder_hidden_states)
        add_q_size = self.add_query_num_heads * self.head_dim
        add_kv_size = self.add_kv_num_heads * self.head_dim
        txt_query, txt_key, txt_value = txt_qkv.split([add_q_size, add_kv_size, add_kv_size], dim=-1)

        img_query = img_query.unflatten(-1, (self.query_num_heads, self.head_dim))
        img_key = img_key.unflatten(-1, (self.kv_num_heads, self.head_dim))
        img_value = img_value.unflatten(-1, (self.kv_num_heads, self.head_dim))

        txt_query = txt_query.unflatten(-1, (self.add_query_num_heads, self.head_dim))
        txt_key = txt_key.unflatten(-1, (self.add_kv_num_heads, self.head_dim))
        txt_value = txt_value.unflatten(-1, (self.add_kv_num_heads, self.head_dim))

        img_query = self.norm_q(img_query)
        img_key = self.norm_k(img_key)
        txt_query = self.norm_added_q(txt_query)
        txt_key = self.norm_added_k(txt_key)

        img_cos = vid_freqs.real.to(img_query.dtype)
        img_sin = vid_freqs.imag.to(img_query.dtype)
        txt_cos = txt_freqs.real.to(txt_query.dtype)
        txt_sin = txt_freqs.imag.to(txt_query.dtype)

        img_query = self.rope(img_query, img_cos, img_sin)
        img_key = self.rope(img_key, img_cos, img_sin)
        txt_query = self.rope(txt_query, txt_cos, txt_sin)
        txt_key = self.rope(txt_key, txt_cos, txt_sin)

        seq_len_txt = encoder_hidden_states.shape[1]
        joint_query = torch.cat([txt_query, img_query], dim=1)
        joint_key = torch.cat([txt_key, img_key], dim=1)
        joint_value = torch.cat([txt_value, img_value], dim=1)

        parallel_config = self.parallel_config
        if parallel_config is None and is_forward_context_available():
            diffusion_config = get_forward_context().omni_diffusion_config
            parallel_config = None if diffusion_config is None else diffusion_config.parallel_config
        use_dense_ring_sp = (
            parallel_config is not None
            and int(getattr(parallel_config, "sequence_parallel_size", 1) or 1) > 1
            and int(getattr(parallel_config, "ring_degree", 1) or 1) > 1
            and not get_forward_context().split_text_embed_in_sp
        )
        if use_dense_ring_sp:
            attn_metadata = AttentionMetadata(
                joint_query=txt_query,
                joint_key=txt_key,
                joint_value=txt_value,
                joint_strategy="front",
            )
            if hidden_states_mask is not None:
                attn_metadata.attn_mask = hidden_states_mask
            if encoder_hidden_states_mask is not None:
                attn_metadata.joint_attn_mask = encoder_hidden_states_mask

            joint_hidden_states = self.attn(img_query, img_key, img_value, attn_metadata)
        else:
            sp_context = _qwen_image_sp_context()
            if sp_context is not None and not get_forward_context().split_text_embed_in_sp:
                if hidden_states_mask is None and encoder_hidden_states_mask is None:
                    sp_group, sp_world_size, _ = sp_context
                    batch_size = int(hidden_states.shape[0])
                    image_len = int(hidden_states.shape[1]) * sp_world_size
                    text_len = int(encoder_hidden_states.shape[1])
                    image_lens = [image_len] * batch_size
                    text_lens = [text_len] * batch_size
                    flat_indices = _qwen_image_packed_indices(image_lens, text_lens, hidden_states.device)
                    source_to_joint_idx, joint_to_source_idx, _ = _qwen_image_sp_replicated_text_source_order_indices(
                        text_lens=text_lens,
                        image_lens=image_lens,
                        world_size=sp_world_size,
                        joint_gather_idx=flat_indices["joint_gather_idx"],
                        device=hidden_states.device,
                    )
                    joint_lens = [text_len + image_len] * batch_size
                    cu_seqlens = torch.tensor(
                        [0, *torch.tensor(joint_lens, dtype=torch.int32).cumsum(0).tolist()],
                        dtype=torch.int32,
                        device=hidden_states.device,
                    )
                    attn_metadata = AttentionMetadata(
                        q_cu_seqlens=cu_seqlens,
                        kv_cu_seqlens=cu_seqlens,
                        max_q_len=text_len + image_len,
                        max_kv_len=text_len + image_len,
                        padded_tokens=0,
                    )

                    def _attention_fn(query, key, value, metadata):
                        metadata = self.attn._with_kv_cache_dtype(metadata)
                        return self.attn._run_local_attention(query, key, value, metadata)

                    txt_hidden_states, img_hidden_states = ulysses_flat_varlen_attention_with_replicated_prefix(
                        sp_group,
                        txt_query.flatten(0, 1).contiguous(),
                        txt_key.flatten(0, 1).contiguous(),
                        txt_value.flatten(0, 1).contiguous(),
                        img_query.flatten(0, 1).contiguous(),
                        img_key.flatten(0, 1).contiguous(),
                        img_value.flatten(0, 1).contiguous(),
                        attn_metadata,
                        source_to_joint_idx=source_to_joint_idx,
                        joint_to_source_idx=joint_to_source_idx,
                        local_suffix_len=int(img_query.shape[0] * img_query.shape[1]),
                        attention_fn=_attention_fn,
                    )
                    joint_hidden_states = torch.cat(
                        [
                            txt_hidden_states.unflatten(0, (batch_size, text_len)),
                            img_hidden_states.unflatten(0, (batch_size, int(hidden_states.shape[1]))),
                        ],
                        dim=1,
                    )
                else:
                    attn_metadata = AttentionMetadata(
                        joint_query=txt_query,
                        joint_key=txt_key,
                        joint_value=txt_value,
                        joint_strategy="front",
                    )
                    if hidden_states_mask is not None:
                        attn_metadata.attn_mask = hidden_states_mask
                    if encoder_hidden_states_mask is not None:
                        attn_metadata.joint_attn_mask = encoder_hidden_states_mask

                    joint_hidden_states = self.attn(img_query, img_key, img_value, attn_metadata)
            else:
                attn_metadata = None
                if hidden_states_mask is not None or encoder_hidden_states_mask is not None:
                    mask_list: list[torch.Tensor] = []
                    if encoder_hidden_states_mask is not None:
                        mask_list.append(encoder_hidden_states_mask)
                    else:
                        mask_list.append(
                            torch.ones(
                                encoder_hidden_states.shape[:2],
                                dtype=torch.bool,
                                device=encoder_hidden_states.device,
                            )
                        )
                    if hidden_states_mask is not None:
                        mask_list.append(hidden_states_mask)
                    else:
                        mask_list.append(
                            torch.ones(
                                hidden_states.shape[:2],
                                dtype=torch.bool,
                                device=hidden_states.device,
                            )
                        )
                    joint_mask = torch.cat(mask_list, dim=1) if len(mask_list) > 1 else mask_list[0]
                    attn_metadata = AttentionMetadata(attn_mask=joint_mask)

                joint_hidden_states = self.attn(joint_query, joint_key, joint_value, attn_metadata)

        joint_hidden_states = joint_hidden_states.flatten(2, 3).to(joint_query.dtype)
        txt_attn_output = joint_hidden_states[:, :seq_len_txt, :]
        img_attn_output = joint_hidden_states[:, seq_len_txt:, :]

        img_attn_output = self.to_out(img_attn_output)
        txt_attn_output = self.to_add_out(txt_attn_output)

        return img_attn_output, txt_attn_output

    def forward_dynamic(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        vid_freqs: torch.Tensor,
        txt_freqs: torch.Tensor,
        attn_metadata: AttentionMetadata,
        packed_indices: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        q_size = self.query_num_heads * self.head_dim
        kv_size = self.kv_num_heads * self.head_dim
        add_q_size = self.add_query_num_heads * self.head_dim
        add_kv_size = self.add_kv_num_heads * self.head_dim
        sp_group = packed_indices.get("sp_group")

        img_qkv = _qwen_image_unwrap_module_output(self.to_qkv(hidden_states))
        img_query, img_key, img_value = img_qkv.split([q_size, kv_size, kv_size], dim=-1)

        txt_qkv = _qwen_image_unwrap_module_output(self.add_kv_proj(encoder_hidden_states))
        txt_query, txt_key, txt_value = txt_qkv.split([add_q_size, add_kv_size, add_kv_size], dim=-1)

        img_query = img_query.unflatten(-1, (self.query_num_heads, self.head_dim))
        img_key = img_key.unflatten(-1, (self.kv_num_heads, self.head_dim))
        img_value = img_value.unflatten(-1, (self.kv_num_heads, self.head_dim))

        txt_query = txt_query.unflatten(-1, (self.add_query_num_heads, self.head_dim))
        txt_key = txt_key.unflatten(-1, (self.add_kv_num_heads, self.head_dim))
        txt_value = txt_value.unflatten(-1, (self.add_kv_num_heads, self.head_dim))

        img_query = self.norm_q(img_query)
        img_key = self.norm_k(img_key)
        txt_query = self.norm_added_q(txt_query)
        txt_key = self.norm_added_k(txt_key)

        img_cos = vid_freqs.real.to(img_query.dtype)
        img_sin = vid_freqs.imag.to(img_query.dtype)
        txt_cos = txt_freqs.real.to(txt_query.dtype)
        txt_sin = txt_freqs.imag.to(txt_query.dtype)

        img_query = self.rope(img_query, img_cos, img_sin)
        img_key = self.rope(img_key, img_cos, img_sin)
        txt_query = self.rope(txt_query, txt_cos, txt_sin)
        txt_key = self.rope(txt_key, txt_cos, txt_sin)

        if sp_group is not None:
            sp_mode = packed_indices.get("sp_mode", "ulysses")
            if sp_mode == "ring":
                txt_hidden_states, img_hidden_states = _qwen_image_ring_flat_varlen_attention_with_replicated_prefix(
                    sp_group,
                    txt_query,
                    txt_key,
                    txt_value,
                    img_query,
                    img_key,
                    img_value,
                    text_lens=packed_indices["sp_text_lens"],
                    suffix_lens_by_rank=packed_indices["sp_image_lens_by_rank"],
                    rank=int(packed_indices["sp_rank"]),
                    softmax_scale=self.attn.softmax_scale,
                    varlen_plan=packed_indices.get("sp_ring_varlen_plan"),
                )
                txt_attn_output = txt_hidden_states.flatten(1, 2).to(txt_query.dtype)
                img_attn_output = img_hidden_states.flatten(1, 2).to(img_query.dtype)
                return self.to_out(img_attn_output), self.to_add_out(txt_attn_output)

            def _attention_fn(query, key, value, metadata):
                metadata = self.attn._with_kv_cache_dtype(metadata)
                return self.attn._run_local_attention(query, key, value, metadata)

            if packed_indices.get("sp_replicated_text", False):
                txt_hidden_states, img_hidden_states = ulysses_flat_varlen_attention_with_replicated_prefix(
                    sp_group,
                    txt_query,
                    txt_key,
                    txt_value,
                    img_query,
                    img_key,
                    img_value,
                    attn_metadata,
                    source_to_joint_idx=packed_indices["sp_source_to_joint_idx"],
                    joint_to_source_idx=packed_indices["sp_joint_to_source_idx"],
                    local_suffix_len=int(img_query.shape[0]),
                    attention_fn=_attention_fn,
                    use_sync=bool(packed_indices.get("sp_use_sync", False)),
                )
                txt_attn_output = txt_hidden_states.flatten(1, 2).to(txt_query.dtype)
                img_attn_output = img_hidden_states.flatten(1, 2).to(img_query.dtype)
            else:
                local_query = torch.cat([txt_query, img_query], dim=0).contiguous()
                local_key = torch.cat([txt_key, img_key], dim=0).contiguous()
                local_value = torch.cat([txt_value, img_value], dim=0).contiguous()
                local_hidden_states = ulysses_flat_varlen_attention(
                    sp_group,
                    local_query,
                    local_key,
                    local_value,
                    attn_metadata,
                    source_to_joint_idx=packed_indices["sp_source_to_joint_idx"],
                    joint_to_source_idx=packed_indices["sp_joint_to_source_idx"],
                    local_seq_len=int(local_query.shape[0]),
                    attention_fn=_attention_fn,
                    use_sync=bool(packed_indices.get("sp_use_sync", False)),
                )
                local_hidden_states = local_hidden_states.flatten(1, 2).to(local_query.dtype)
                text_len = int(packed_indices["sp_local_text_len"])
                txt_attn_output = local_hidden_states[:text_len]
                img_attn_output = local_hidden_states[text_len:]
            return self.to_out(img_attn_output), self.to_add_out(txt_attn_output)

        text_lens = packed_indices["text_lens"]
        image_lens = packed_indices["image_lens"]
        joint_query = _qwen_image_joint_cat(txt_query, img_query, text_lens, image_lens)
        joint_key = _qwen_image_joint_cat(txt_key, img_key, text_lens, image_lens)
        joint_value = _qwen_image_joint_cat(txt_value, img_value, text_lens, image_lens)

        joint_hidden_states = self.attn(joint_query, joint_key, joint_value, attn_metadata)
        joint_hidden_states = joint_hidden_states.flatten(1, 2).to(joint_query.dtype)

        txt_attn_output, img_attn_output = _qwen_image_split_joint(joint_hidden_states, text_lens, image_lens)
        return self.to_out(img_attn_output), self.to_add_out(txt_attn_output)


class QwenImageTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        qk_norm: str = "rms_norm",
        eps: float = 1e-6,
        zero_cond_t: bool = False,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()

        self.dim = dim
        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim

        # Image processing modules.
        self.img_mod = nn.Sequential(
            nn.SiLU(),
            ColumnParallelLinear(
                dim,
                6 * dim,
                bias=True,
                gather_output=True,
                return_bias=False,
                quant_config=None,
                prefix=_join_prefix(prefix, "img_mod.1"),
            ),
        )
        self.img_norm1 = AdaLayerNorm(dim, elementwise_affine=False, eps=eps)
        self.attn = QwenImageCrossAttention(
            dim=dim,
            num_heads=num_attention_heads,
            added_kv_proj_dim=dim,
            context_pre_only=False,
            head_dim=attention_head_dim,
            quant_config=quant_config,
            prefix=_join_prefix(prefix, "attn"),
        )
        self.img_norm2 = AdaLayerNorm(dim, elementwise_affine=False, eps=eps)
        self.img_mlp = FeedForward(
            dim=dim,
            dim_out=dim,
            quant_config=quant_config,
            prefix=_join_prefix(prefix, "img_mlp"),
        )

        # Text processing modules.
        self.txt_mod = nn.Sequential(
            nn.SiLU(),
            ColumnParallelLinear(
                dim,
                6 * dim,
                bias=True,
                gather_output=True,
                return_bias=False,
                quant_config=None,
                prefix=_join_prefix(prefix, "txt_mod.1"),
            ),
        )
        self.txt_norm1 = AdaLayerNorm(dim, elementwise_affine=False, eps=eps)
        # Text doesn't need separate attention - it's handled by img_attn joint computation
        self.txt_norm2 = AdaLayerNorm(dim, elementwise_affine=False, eps=eps)
        self.txt_mlp = FeedForward(
            dim=dim,
            dim_out=dim,
            quant_config=quant_config,
            prefix=_join_prefix(prefix, "txt_mlp"),
        )

        self.zero_cond_t = zero_cond_t

    def _modulate(self, mod_params, index=None):
        """Apply modulation to input tensor"""
        # shift: b d, scale: b d, gate: b d
        shift, scale, gate = mod_params.chunk(3, dim=-1)

        if index is not None:
            # Assuming mod_params batch dim is 2*actual_batch (chunked into 2 parts)
            # So shift, scale, gate have shape [2*actual_batch, d]
            actual_batch = shift.size(0) // 2
            shift_0, shift_1 = shift[:actual_batch], shift[actual_batch:]  # each: [actual_batch, d]
            scale_0, scale_1 = scale[:actual_batch], scale[actual_batch:]
            gate_0, gate_1 = gate[:actual_batch], gate[actual_batch:]

            # index: [b, l] where b is actual batch size
            # Expand to [b, l, 1] to match feature dimension
            index_expanded = index.unsqueeze(-1)  # [b, l, 1]

            # Expand chunks to [b, 1, d] then broadcast to [b, l, d]
            shift_0_exp = shift_0.unsqueeze(1)  # [b, 1, d]
            shift_1_exp = shift_1.unsqueeze(1)  # [b, 1, d]
            scale_0_exp = scale_0.unsqueeze(1)
            scale_1_exp = scale_1.unsqueeze(1)
            gate_0_exp = gate_0.unsqueeze(1)
            gate_1_exp = gate_1.unsqueeze(1)

            # Use torch.where to select based on index
            shift_result = torch.where(index_expanded == 0, shift_0_exp, shift_1_exp)
            scale_result = torch.where(index_expanded == 0, scale_0_exp, scale_1_exp)
            gate_result = torch.where(index_expanded == 0, gate_0_exp, gate_1_exp)
        else:
            shift_result = shift.unsqueeze(1)
            scale_result = scale.unsqueeze(1)
            gate_result = gate.unsqueeze(1)

        return scale_result, shift_result, gate_result

    @staticmethod
    def _modulate_packed(mod_params: torch.Tensor, token_to_req: torch.Tensor):
        shift, scale, gate = mod_params.chunk(3, dim=-1)
        return (
            scale.index_select(0, token_to_req),
            shift.index_select(0, token_to_req),
            gate.index_select(0, token_to_req),
        )

    @staticmethod
    def _modulate_shared(mod_params: torch.Tensor):
        shift, scale, gate = mod_params.chunk(3, dim=-1)
        return scale, shift, gate

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_hidden_states_mask: torch.Tensor,
        temb: torch.Tensor,
        image_rotary_emb: tuple[torch.Tensor, torch.Tensor],
        joint_attention_kwargs: dict[str, Any] | None = None,
        modulate_index: list[int] | None = None,
        hidden_states_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Get modulation parameters for both streams
        img_mod_params = self.img_mod(temb)  # [B, 6*dim]

        if self.zero_cond_t:
            temb = torch.chunk(temb, 2, dim=0)[0]

        txt_mod_params = self.txt_mod(temb)  # [B, 6*dim]

        # Split modulation parameters for norm1 and norm2
        img_mod1, img_mod2 = img_mod_params.chunk(2, dim=-1)  # Each [B, 3*dim]
        txt_mod1, txt_mod2 = txt_mod_params.chunk(2, dim=-1)  # Each [B, 3*dim]

        # Process image stream - norm1 + modulation
        img_scale1, img_shift1, img_gate1 = self._modulate(img_mod1, modulate_index)
        img_modulated = self.img_norm1(hidden_states, img_scale1, img_shift1)

        # Process text stream - norm1 + modulation
        txt_scale1, txt_shift1, txt_gate1 = self._modulate(txt_mod1)
        txt_modulated = self.txt_norm1(encoder_hidden_states, txt_scale1, txt_shift1)

        # Use QwenAttnProcessor2_0 for joint attention computation
        # This directly implements the DoubleStreamLayerMegatron logic:
        # 1. Computes QKV for both streams
        # 2. Applies QK normalization and RoPE
        # 3. Concatenates and runs joint attention
        # 4. Splits results back to separate streams
        attn_output = self.attn(
            hidden_states=img_modulated,  # Image stream (will be processed as "sample")
            encoder_hidden_states=txt_modulated,  # Text stream (will be processed as "context")
            vid_freqs=image_rotary_emb[0],
            txt_freqs=image_rotary_emb[1],
            hidden_states_mask=hidden_states_mask,
            encoder_hidden_states_mask=encoder_hidden_states_mask,
        )

        # QwenAttnProcessor2_0 returns (img_output, txt_output) when encoder_hidden_states is provided
        img_attn_output, txt_attn_output = attn_output

        # Apply attention gates and add residual (like in Megatron)
        hidden_states = hidden_states + img_gate1 * img_attn_output
        encoder_hidden_states = encoder_hidden_states + txt_gate1 * txt_attn_output

        # Process image stream - norm2 + MLP
        img_scale2, img_shift2, img_gate2 = self._modulate(img_mod2, modulate_index)
        img_modulated2 = self.img_norm2(hidden_states, img_scale2, img_shift2)

        img_mlp_output = self.img_mlp(img_modulated2)
        hidden_states = hidden_states + img_gate2 * img_mlp_output

        # Process text stream - norm2 + MLP
        txt_scale2, txt_shift2, txt_gate2 = self._modulate(txt_mod2)
        txt_modulated2 = self.txt_norm2(encoder_hidden_states, txt_scale2, txt_shift2)

        txt_mlp_output = self.txt_mlp(txt_modulated2)
        encoder_hidden_states = encoder_hidden_states + txt_gate2 * txt_mlp_output

        # Clip to prevent overflow for fp16
        if encoder_hidden_states.dtype == torch.float16:
            encoder_hidden_states = encoder_hidden_states.clip(-65504, 65504)
        if hidden_states.dtype == torch.float16:
            hidden_states = hidden_states.clip(-65504, 65504)

        return encoder_hidden_states, hidden_states

    def forward_dynamic(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        temb_silu: torch.Tensor,
        image_rotary_emb: tuple[torch.Tensor, torch.Tensor],
        attn_metadata: AttentionMetadata,
        packed_indices: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.zero_cond_t:
            raise ValueError("Qwen-Image dynamic step batching does not support zero_cond_t/edit blocks yet.")

        image_token_to_req = packed_indices["image_token_to_req"]
        text_token_to_req = packed_indices["text_token_to_req"]
        shared_temb = bool(packed_indices.get("shared_temb", False))
        img_mod1, img_mod2 = _qwen_image_unwrap_module_output(self.img_mod[1](temb_silu)).chunk(2, dim=-1)
        txt_mod1, txt_mod2 = _qwen_image_unwrap_module_output(self.txt_mod[1](temb_silu)).chunk(2, dim=-1)
        if shared_temb:
            img_scale1, img_shift1, img_gate1 = self._modulate_shared(img_mod1)
            txt_scale1, txt_shift1, txt_gate1 = self._modulate_shared(txt_mod1)
        else:
            img_scale1, img_shift1, img_gate1 = self._modulate_packed(img_mod1, image_token_to_req)
            txt_scale1, txt_shift1, txt_gate1 = self._modulate_packed(txt_mod1, text_token_to_req)
        img_modulated = self.img_norm1(hidden_states, img_scale1, img_shift1)
        txt_modulated = self.txt_norm1(encoder_hidden_states, txt_scale1, txt_shift1)

        img_attn_output, txt_attn_output = self.attn.forward_dynamic(
            hidden_states=img_modulated,
            encoder_hidden_states=txt_modulated,
            vid_freqs=image_rotary_emb[0],
            txt_freqs=image_rotary_emb[1],
            attn_metadata=attn_metadata,
            packed_indices=packed_indices,
        )

        hidden_states = hidden_states + img_gate1 * img_attn_output
        encoder_hidden_states = encoder_hidden_states + txt_gate1 * txt_attn_output

        if shared_temb:
            img_scale2, img_shift2, img_gate2 = self._modulate_shared(img_mod2)
            txt_scale2, txt_shift2, txt_gate2 = self._modulate_shared(txt_mod2)
        else:
            img_scale2, img_shift2, img_gate2 = self._modulate_packed(img_mod2, image_token_to_req)
            txt_scale2, txt_shift2, txt_gate2 = self._modulate_packed(txt_mod2, text_token_to_req)
        img_mlp_input = self.img_norm2(hidden_states, img_scale2, img_shift2)
        txt_mlp_input = self.txt_norm2(encoder_hidden_states, txt_scale2, txt_shift2)
        img_mlp_output = self.img_mlp(img_mlp_input)
        txt_mlp_output = self.txt_mlp(txt_mlp_input)
        hidden_states = hidden_states + img_gate2 * img_mlp_output
        encoder_hidden_states = encoder_hidden_states + txt_gate2 * txt_mlp_output

        if encoder_hidden_states.dtype == torch.float16:
            encoder_hidden_states = encoder_hidden_states.clip(-65504, 65504)
        if hidden_states.dtype == torch.float16:
            hidden_states = hidden_states.clip(-65504, 65504)

        return encoder_hidden_states, hidden_states


# Note: inheriting from CachedTransformer only when we support caching
class QwenImageTransformer2DModel(CachedTransformer):
    """
    The Transformer model introduced in Qwen.

    Args:
        patch_size (`int`, defaults to `2`):
            Patch size to turn the input data into small patches.
        in_channels (`int`, defaults to `64`):
            The number of channels in the input.
        out_channels (`int`, *optional*, defaults to `None`):
            The number of channels in the output. If not specified, it defaults to `in_channels`.
        num_layers (`int`, defaults to `60`):
            The number of layers of dual stream DiT blocks to use.
        attention_head_dim (`int`, defaults to `128`):
            The number of dimensions to use for each attention head.
        num_attention_heads (`int`, defaults to `24`):
            The number of attention heads to use.
        joint_attention_dim (`int`, defaults to `3584`):
            The number of dimensions to use for the joint attention (embedding/channel dimension of
            `encoder_hidden_states`).
        guidance_embeds (`bool`, defaults to `False`):
            Whether to use guidance embeddings for guidance-distilled variant of the model.
        axes_dims_rope (`tuple[int]`, defaults to `(16, 56, 56)`):
            The dimensions to use for the rotary positional embeddings.
    """

    # the small and frequently-repeated block(s) of a model
    # -- typically a transformer layer
    # used for torch compile optimizations
    _repeated_blocks = ["QwenImageTransformerBlock"]
    _layerwise_offload_blocks_attrs = ["transformer_blocks"]
    packed_modules_mapping = {
        "to_qkv": ["to_q", "to_k", "to_v"],
        "add_kv_proj": ["add_q_proj", "add_k_proj", "add_v_proj"],
    }

    _hsdp_shard_conditions = [is_transformer_block_module]

    # Sequence Parallelism plan (following diffusers' _cp_plan pattern)
    # Similar to Z-Image's UnifiedPrepare, we use ImageRopePrepare to create
    # a module boundary where _sp_plan can shard hidden_states and vid_freqs together.
    #
    # Key insight: hidden_states and vid_freqs MUST be sharded together to maintain
    # dimension alignment for RoPE computation in attention layers.
    #
    # auto_pad=True enables automatic padding when sequence length is not divisible
    # by SP world size. This creates an attention mask stored in ForwardContext
    # that attention layers can use to ignore padding positions.
    #
    # Note: _sp_plan corresponds to diffusers' _cp_plan (Context Parallelism)
    _sp_plan = {
        # Shard ImageRopePrepare outputs (hidden_states and vid_freqs must be sharded together)
        "image_rope_prepare": {
            # hidden_states: auto_pad=True for variable sequence length support
            0: SequenceParallelInput(split_dim=1, expected_dims=3, split_output=True, auto_pad=True),
            # vid_freqs: auto_pad=True to match hidden_states padding
            1: SequenceParallelInput(split_dim=0, expected_dims=2, split_output=True, auto_pad=True),
            # txt_freqs (index 2) is NOT sharded - kept replicated for dual-stream attention
        },
        # Shard ModulateIndexPrepare output (modulate_index must be sharded to match hidden_states)
        # This is only active when zero_cond_t=True (image editing models)
        # Output index 1 is modulate_index [batch, seq_len], needs sharding along dim=1
        "modulate_index_prepare": {
            1: SequenceParallelInput(split_dim=1, expected_dims=2, split_output=True, auto_pad=True),
        },
        # Gather output at proj_out
        "proj_out": SequenceParallelOutput(gather_dim=1, expected_dims=3),
    }

    def __init__(
        self,
        od_config: OmniDiffusionConfig,
        patch_size: int = 2,
        in_channels: int = 64,
        out_channels: int | None = 16,
        num_layers: int = 60,
        attention_head_dim: int = 128,
        num_attention_heads: int = 24,
        joint_attention_dim: int = 3584,
        guidance_embeds: bool = False,
        axes_dims_rope: tuple[int, int, int] = (16, 56, 56),
        zero_cond_t: bool = False,
        use_additional_t_cond: bool = False,
        use_layer3d_rope: bool = False,
        quant_config: QuantizationConfig | None = None,
    ):
        super().__init__()
        self.parallel_config = od_config.parallel_config
        self.in_channels = in_channels
        self.out_channels = out_channels or in_channels
        self.inner_dim = num_attention_heads * attention_head_dim
        self.guidance_embeds = guidance_embeds

        if not use_layer3d_rope:
            self.pos_embed = QwenEmbedRope(theta=10000, axes_dim=list(axes_dims_rope), scale_rope=True)
        else:
            self.pos_embed = QwenEmbedLayer3DRope(theta=10000, axes_dim=list(axes_dims_rope), scale_rope=True)

        self.time_text_embed = QwenTimestepProjEmbeddings(
            embedding_dim=self.inner_dim,
            use_additional_t_cond=use_additional_t_cond,
            quant_config=quant_config,
        )

        self.txt_norm = nn.RMSNorm(joint_attention_dim, eps=1e-6)

        self.img_in = ReplicatedLinear(
            in_channels,
            self.inner_dim,
            bias=True,
            return_bias=False,
            quant_config=None,
            prefix="img_in",
        )
        self.txt_in = ReplicatedLinear(
            joint_attention_dim,
            self.inner_dim,
            bias=True,
            return_bias=False,
            quant_config=None,
            prefix="txt_in",
        )

        self.transformer_blocks = nn.ModuleList(
            [
                QwenImageTransformerBlock(
                    dim=self.inner_dim,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    zero_cond_t=zero_cond_t,
                    quant_config=quant_config,
                    prefix=f"transformer_blocks.{i}",
                )
                for i in range(num_layers)
            ]
        )

        self.norm_out = AdaLayerNormContinuous(self.inner_dim, self.inner_dim, elementwise_affine=False, eps=1e-6)
        self.norm_out.linear = ReplicatedLinear(
            self.inner_dim,
            2 * self.inner_dim,
            bias=True,
            return_bias=False,
            quant_config=None,
            prefix="norm_out.linear",
        )
        self.proj_out = ReplicatedLinear(
            self.inner_dim,
            patch_size * patch_size * self.out_channels,
            bias=True,
            return_bias=False,
            quant_config=None,
            prefix="proj_out",
        )

        self.gradient_checkpointing = False
        self.zero_cond_t = zero_cond_t

        # ImageRopePrepare module for _sp_plan to shard hidden_states and vid_freqs together
        # This ensures RoPE dimensions align with hidden_states after sharding
        self.image_rope_prepare = ImageRopePrepare(self.img_in, self.pos_embed)

        # ModulateIndexPrepare module for _sp_plan to shard modulate_index
        # This ensures modulate_index dimensions align with hidden_states after sharding
        # Only active when zero_cond_t=True (image editing models)
        self.modulate_index_prepare = ModulateIndexPrepare(zero_cond_t=zero_cond_t)

    def _forward_dynamic(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_hidden_states_mask: torch.Tensor | None,
        timestep: torch.LongTensor,
        img_shapes: list[tuple[int, int, int]] | None,
        img_seq_lens: list[int] | None,
        txt_seq_lens: list[int] | None,
        guidance: torch.Tensor | None,
        attention_kwargs: dict[str, Any] | None,
        additional_t_cond=None,
    ) -> Transformer2DModelOutput:
        if self.zero_cond_t:
            raise ValueError("Qwen-Image dynamic step batching does not support zero_cond_t/edit mode yet.")

        attention_id = None if attention_kwargs is None else attention_kwargs.get("attention_id")
        attn_metadata = _get_qwen_image_attention_metadata(attention_id)
        if attn_metadata is None or not attn_metadata.is_varlen:
            raise ValueError("Qwen-Image dynamic forward requires varlen AttentionMetadata.")
        if attn_metadata.padded_tokens != 0:
            raise ValueError("Qwen-Image dynamic forward requires padded_tokens=0.")
        if encoder_hidden_states is None:
            raise ValueError("Qwen-Image dynamic forward requires encoder_hidden_states.")
        if timestep is None:
            raise ValueError("Qwen-Image dynamic forward requires timestep.")

        if not torch.is_tensor(hidden_states) or hidden_states.ndim != 2:
            raise ValueError("Packed dynamic Qwen-Image hidden states must be [total_image_tokens, channels].")
        if img_seq_lens is None:
            if img_shapes is None:
                position = attn_metadata.position if isinstance(attn_metadata.position, dict) else {}
                img_shapes = position.get("img_shapes")
            if img_shapes is None:
                raise ValueError("Packed dynamic Qwen-Image forward requires img_seq_lens or img_shapes.")
            img_seq_lens = [sum(prod(shape) for shape in image_shape) for image_shape in img_shapes]
        batch_size = len(img_seq_lens)
        if batch_size == 0:
            raise ValueError("Qwen-Image dynamic forward received an empty batch.")
        if txt_seq_lens is None:
            if encoder_hidden_states_mask is None:
                if not torch.is_tensor(encoder_hidden_states):
                    raise ValueError("txt_seq_lens is required when dynamic text embeddings are provided as a list.")
                txt_seq_lens = [int(encoder_hidden_states.shape[1])] * batch_size
            else:
                txt_seq_lens = encoder_hidden_states_mask.sum(dim=1, dtype=torch.int32).tolist()
        if len(txt_seq_lens) != batch_size:
            raise ValueError(f"Expected {batch_size} text sequence lengths, got {len(txt_seq_lens)}.")

        if img_shapes is None:
            position = attn_metadata.position if isinstance(attn_metadata.position, dict) else {}
            img_shapes = position.get("img_shapes")
        if img_shapes is None or len(img_shapes) != batch_size:
            raise ValueError(f"Expected {batch_size} image shape entries for dynamic Qwen-Image forward.")

        device = hidden_states.device
        dtype = hidden_states.dtype
        image_lens = [int(value) for value in img_seq_lens]
        if sum(image_lens) != int(hidden_states.shape[0]):
            raise ValueError(
                f"Packed dynamic Qwen-Image hidden token count {hidden_states.shape[0]} "
                f"does not match img_seq_lens sum {sum(image_lens)}."
            )
        packed_hidden_states = hidden_states.contiguous()
        image_states = _qwen_image_unwrap_module_output(self.img_in(packed_hidden_states))
        image_freqs: list[torch.Tensor] = []
        text_freqs: list[torch.Tensor] = []
        for image_len, image_shape, txt_len in zip(image_lens, img_shapes, txt_seq_lens, strict=True):
            vid_freq, txt_freq = self.pos_embed(image_shape, [int(txt_len)], device=image_states.device)
            if int(vid_freq.shape[0]) != image_len:
                raise ValueError(
                    f"Qwen-Image dynamic RoPE image length {vid_freq.shape[0]} "
                    f"does not match hidden length {image_len}."
                )
            image_freqs.append(vid_freq)
            text_freqs.append(txt_freq[: int(txt_len)])
        image_freqs_packed = torch.cat(image_freqs, dim=0).contiguous()
        text_freqs_packed = torch.cat(text_freqs, dim=0).contiguous()

        text_values: list[torch.Tensor] = []
        if not torch.is_tensor(encoder_hidden_states) or encoder_hidden_states.shape[0] != batch_size:
            got = "non-tensor" if not torch.is_tensor(encoder_hidden_states) else int(encoder_hidden_states.shape[0])
            raise ValueError(f"Expected packed dynamic encoder tensor with {batch_size} rows, got {got}.")
        for index, txt_len in enumerate(txt_seq_lens):
            text_values.append(encoder_hidden_states[index : index + 1, : int(txt_len)])
        text_states = torch.cat([tensor.squeeze(0) for tensor in text_values], dim=0).contiguous()
        text_states = self.txt_norm(text_states)
        text_states = _qwen_image_unwrap_module_output(self.txt_in(text_states))

        timestep = timestep.to(device=device, dtype=dtype)
        if timestep.ndim == 0:
            timestep = timestep.reshape(1).expand(batch_size)
        elif int(timestep.numel()) != batch_size:
            raise ValueError(f"Expected {batch_size} timesteps, got {int(timestep.numel())}.")
        timestep = timestep.reshape(batch_size)

        if guidance is not None:
            guidance = guidance.to(device=device, dtype=dtype) * 1000

        tensor_parallel_size = int(getattr(self.parallel_config, "tensor_parallel_size", 1) or 1)
        shared_temb = (
            tensor_parallel_size == 1
            and guidance is None
            and additional_t_cond is None
            and bool(torch.equal(timestep, timestep[:1].expand_as(timestep)))
        )
        if shared_temb:
            temb = self.time_text_embed(timestep[:1], image_states, None)
        else:
            temb = (
                self.time_text_embed(timestep, image_states, additional_t_cond)
                if guidance is None
                else self.time_text_embed(timestep, guidance, image_states, additional_t_cond)
            )

        text_lens = [int(value) for value in txt_seq_lens]
        packed_indices = _qwen_image_packed_indices(image_lens, text_lens, device)
        packed_indices.update(
            {
                "image_lens": image_lens,
                "text_lens": text_lens,
                "shared_temb": shared_temb,
            }
        )
        sp_group: dist.ProcessGroup | None = None
        sp_context = _qwen_image_dynamic_sp_context()
        if sp_context is not None:
            sp_mode, sp_group, sp_world_size, sp_rank = sp_context
            total_image = int(image_states.shape[0])
            total_text = int(text_states.shape[0])
            image_local_idx, image_local_lens = _qwen_image_rank_local_indices(
                image_lens,
                rank=sp_rank,
                world_size=sp_world_size,
                device=device,
            )
            source_to_joint_idx, joint_to_source_idx, image_packed_to_source_idx = (
                _qwen_image_sp_replicated_text_source_order_indices(
                    text_lens=text_lens,
                    image_lens=image_lens,
                    world_size=sp_world_size,
                    joint_gather_idx=packed_indices["joint_gather_idx"],
                    device=device,
                )
            )
            if int(total_image) != sum(image_lens) or int(total_text) != sum(text_lens):
                raise ValueError(
                    "Qwen-Image SP dynamic token count mismatch: "
                    f"image={total_image}/{sum(image_lens)}, text={total_text}/{sum(text_lens)}."
                )
            image_token_to_req = _qwen_image_token_to_req(image_local_lens, device)
            text_token_to_req = _qwen_image_token_to_req(text_lens, device)
            sp_image_lens_by_rank = [
                _qwen_image_rank_local_indices(
                    image_lens,
                    rank=rank,
                    world_size=sp_world_size,
                    device=device,
                )[1]
                for rank in range(sp_world_size)
            ]
            packed_indices = dict(packed_indices)
            packed_indices.update(
                {
                    "image_token_to_req": image_token_to_req,
                    "text_token_to_req": text_token_to_req,
                    "sp_group": sp_group,
                    "sp_source_to_joint_idx": source_to_joint_idx,
                    "sp_joint_to_source_idx": joint_to_source_idx,
                    "sp_image_packed_to_source_idx": image_packed_to_source_idx,
                    "sp_mode": sp_mode,
                    "sp_rank": sp_rank,
                    "sp_replicated_text": True,
                    "sp_local_text_len": int(total_text),
                    "sp_local_image_len": int(image_local_idx.numel()),
                    "sp_image_lens": image_local_lens,
                    "sp_image_lens_by_rank": sp_image_lens_by_rank,
                    "sp_text_lens": text_lens,
                    "sp_ring_varlen_plan": (
                        ring_flash_attn_varlen_replicated_prefix_plan(
                            prefix_lens=text_lens,
                            suffix_lens_by_rank=sp_image_lens_by_rank,
                            rank=sp_rank,
                            device=device,
                        )
                        if sp_mode == "ring"
                        else None
                    ),
                    "image_lens": None,
                    "text_lens": None,
                }
            )

            image_states = image_states.index_select(0, image_local_idx).contiguous()
            image_freqs_packed = image_freqs_packed.index_select(0, image_local_idx).contiguous()

        image_rotary_emb = (image_freqs_packed, text_freqs_packed)
        temb_silu = F.silu(temb)
        for block in self.transformer_blocks:
            text_states, image_states = block.forward_dynamic(
                hidden_states=image_states,
                encoder_hidden_states=text_states,
                temb=temb,
                temb_silu=temb_silu,
                image_rotary_emb=image_rotary_emb,
                attn_metadata=attn_metadata,
                packed_indices=packed_indices,
            )

        norm_out_input = self.norm_out.silu(temb).to(image_states.dtype)
        emb = _qwen_image_unwrap_module_output(self.norm_out.linear(norm_out_input))
        scale, shift = torch.chunk(emb, 2, dim=1)
        image_token_to_req = packed_indices["image_token_to_req"]
        if shared_temb:
            image_states = self.norm_out.norm(image_states) * (1 + scale) + shift
        else:
            image_states = self.norm_out.norm(image_states) * (
                1 + scale.index_select(0, image_token_to_req)
            ) + shift.index_select(0, image_token_to_req)
        sample = _qwen_image_unwrap_module_output(self.proj_out(image_states))
        if sp_group is not None:
            sample = _qwen_image_all_gather_varlen(sample, sp_group)
            sample = sample.index_select(0, packed_indices["sp_image_packed_to_source_idx"]).contiguous()
        return Transformer2DModelOutput(sample=sample)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
        encoder_hidden_states_mask: torch.Tensor = None,
        timestep: torch.LongTensor = None,
        img_shapes: list[tuple[int, int, int]] | None = None,
        img_seq_lens: list[int] | None = None,
        txt_seq_lens: list[int] | None = None,
        guidance: torch.Tensor = None,  # TODO: this should probably be removed
        attention_kwargs: dict[str, Any] | None = None,
        additional_t_cond=None,
        return_dict: bool = True,
    ) -> torch.Tensor | Transformer2DModelOutput:
        """
        The [`QwenTransformer2DModel`] forward method.

        Args:
            hidden_states (`torch.Tensor` of shape `(batch_size, image_sequence_length, in_channels)`):
                Input `hidden_states`.
            encoder_hidden_states (`torch.Tensor` of shape `(batch_size, text_sequence_length, joint_attention_dim)`):
                Conditional embeddings (embeddings computed from the input conditions such as prompts) to use.
            encoder_hidden_states_mask (`torch.Tensor` of shape `(batch_size, text_sequence_length)`):
                Mask of the input conditions.
            timestep ( `torch.LongTensor`):
                Used to indicate denoising step.
            attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~models.transformer_2d.Transformer2DModelOutput`] instead of a plain
                tuple.

        Returns:
            If `return_dict` is True, an [`~models.transformer_2d.Transformer2DModelOutput`] is returned, otherwise a
            `tuple` where the first element is the sample tensor.
        """
        # if attention_kwargs is not None:
        #     attention_kwargs = attention_kwargs.copy()
        #     lora_scale = attention_kwargs.pop("scale", 1.0)
        # else:
        #     lora_scale = 1.0

        dynamic_attention_id = None if attention_kwargs is None else attention_kwargs.get("attention_id")
        if torch.is_tensor(hidden_states) and hidden_states.ndim == 2 and dynamic_attention_id is not None:
            return self._forward_dynamic(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                encoder_hidden_states_mask=encoder_hidden_states_mask,
                timestep=timestep,
                img_shapes=img_shapes,
                img_seq_lens=img_seq_lens,
                txt_seq_lens=txt_seq_lens,
                guidance=guidance,
                attention_kwargs=attention_kwargs,
                additional_t_cond=additional_t_cond,
            )

        # Set split_text_embed_in_sp = False for dual-stream attention
        # QwenImage uses *dual-stream* (text + image) and runs a *joint attention*.
        # Text embeddings must be replicated across SP ranks for correctness.
        if self.parallel_config.sequence_parallel_size > 1:
            get_forward_context().split_text_embed_in_sp = False

        # Prepare hidden_states and RoPE via ImageRopePrepare module
        # _sp_plan will shard hidden_states and vid_freqs together via split_output=True
        # txt_freqs is kept replicated for dual-stream attention
        hidden_states, vid_freqs, txt_freqs = self.image_rope_prepare(hidden_states, img_shapes, txt_seq_lens)
        image_rotary_emb = (vid_freqs, txt_freqs)

        # Ensure timestep tensor is on the same device and dtype as hidden_states
        timestep = timestep.to(device=hidden_states.device, dtype=hidden_states.dtype)

        # Prepare timestep and modulate_index via ModulateIndexPrepare module
        # _sp_plan will shard modulate_index via split_output=True (when zero_cond_t=True)
        # This ensures modulate_index sequence dimension matches sharded hidden_states
        timestep, modulate_index = self.modulate_index_prepare(timestep, img_shapes)

        encoder_hidden_states = self.txt_norm(encoder_hidden_states)
        encoder_hidden_states = self.txt_in(encoder_hidden_states)

        if guidance is not None:
            guidance = guidance.to(hidden_states.dtype) * 1000

        temb = (
            self.time_text_embed(timestep, hidden_states, additional_t_cond)
            if guidance is None
            else self.time_text_embed(timestep, guidance, hidden_states, additional_t_cond)
        )

        # Check for SP auto_pad: create attention mask dynamically if padding was applied
        # In Ulysses mode, attention is computed on the FULL sequence (after All-to-All)
        hidden_states_mask = None  # default
        if self.parallel_config is not None and self.parallel_config.sequence_parallel_size > 1:
            ctx = get_forward_context()
            if ctx.sp_original_seq_len is not None and ctx.sp_padding_size > 0:
                # Create mask for the full (padded) sequence
                # valid positions = True, padding positions = False
                batch_size = hidden_states.shape[0]
                padded_seq_len = ctx.sp_original_seq_len + ctx.sp_padding_size
                hidden_states_mask = torch.ones(
                    batch_size,
                    padded_seq_len,
                    dtype=torch.bool,
                    device=hidden_states.device,
                )
                hidden_states_mask[:, ctx.sp_original_seq_len :] = False

        # if mask is all true, set it to None
        if hidden_states_mask is not None and hidden_states_mask.all():
            hidden_states_mask = None
        if encoder_hidden_states_mask is not None and encoder_hidden_states_mask.all():
            encoder_hidden_states_mask = None

        for index_block, block in enumerate(self.transformer_blocks):
            encoder_hidden_states, hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                encoder_hidden_states_mask=encoder_hidden_states_mask,
                temb=temb,
                image_rotary_emb=image_rotary_emb,
                joint_attention_kwargs=attention_kwargs,
                modulate_index=modulate_index,
                hidden_states_mask=hidden_states_mask,
            )

        if self.zero_cond_t:
            temb = temb.chunk(2, dim=0)[0]
        # Use only the image part (hidden_states) from the dual-stream blocks
        hidden_states = self.norm_out(hidden_states, temb)
        output = self.proj_out(hidden_states)

        # Note: SP gather is handled automatically by _sp_plan's SequenceParallelGatherHook
        # on proj_out output. No manual all_gather needed here.

        return Transformer2DModelOutput(sample=output)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            # self-attn
            (".to_qkv", ".to_q", "q"),
            (".to_qkv", ".to_k", "k"),
            (".to_qkv", ".to_v", "v"),
            # cross-attn
            (".add_kv_proj", ".add_q_proj", "q"),
            (".add_kv_proj", ".add_k_proj", "k"),
            (".add_kv_proj", ".add_v_proj", "v"),
        ]
        # Expose packed shard mappings for LoRA handling of fused projections.
        self.stacked_params_mapping = stacked_params_mapping

        params_dict = dict(self.named_parameters())

        # we need to load the buffers for beta and eps (XIELU)
        for name, buffer in self.named_buffers():
            if name.endswith(".beta") or name.endswith(".eps"):
                params_dict[name] = buffer

        loaded_params: set[str] = set()
        for name, loaded_weight in weights:
            original_name = name
            lookup_name = name
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in original_name or param_name in original_name:
                    continue
                lookup_name = original_name.replace(weight_name, param_name)
                param = params_dict[lookup_name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                if lookup_name not in params_dict and ".to_out.0." in lookup_name:
                    lookup_name = lookup_name.replace(".to_out.0.", ".to_out.")
                param = params_dict[lookup_name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
            loaded_params.add(original_name)
            loaded_params.add(lookup_name)
        return loaded_params
