# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Qwen-Image attention metadata builders.

This module keeps Qwen-Image's step-local layout construction out of the
transformer forward path. ``InputBatch`` owns the packed image tokens for the
current denoise step; this builder derives the varlen joint-attention contract
that attention layers consume.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.distributed as dist

from vllm_omni.diffusion.attention.backends.abstract import (
    AttentionMetadata,
    AttentionSegment,
)
from vllm_omni.diffusion.attention.backends.ring_flash_attn import (
    ring_flash_attn_varlen_replicated_prefix_plan,
)
from vllm_omni.diffusion.worker.input_batch import InputBatch


def qwen_image_token_to_req(lengths: list[int], device: torch.device) -> torch.Tensor:
    return torch.repeat_interleave(
        torch.arange(len(lengths), dtype=torch.long, device=device),
        torch.tensor(lengths, dtype=torch.long, device=device),
    )


def qwen_image_packed_indices(
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
        "image_token_to_req": qwen_image_token_to_req(image_lens, device),
        "text_token_to_req": qwen_image_token_to_req(text_lens, device),
        "joint_gather_idx": torch.cat(gather_indices, dim=0),
        "joint_text_idx": torch.cat(text_indices, dim=0),
        "joint_image_idx": torch.cat(image_indices, dim=0),
    }


def qwen_image_balanced_ranges(total_len: int, parts: int) -> list[tuple[int, int]]:
    base, remainder = divmod(int(total_len), int(parts))
    ranges: list[tuple[int, int]] = []
    start = 0
    for rank in range(parts):
        length = base + (1 if rank < remainder else 0)
        end = start + length
        ranges.append((start, end))
        start = end
    return ranges


def qwen_image_rank_local_indices(
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
        ranges = qwen_image_balanced_ranges(int(length), world_size)
        start, end = ranges[rank]
        local_lens.append(end - start)
        if end > start:
            pieces.append(torch.arange(offset + start, offset + end, dtype=torch.long, device=device))
        offset += int(length)
    if not pieces:
        return torch.empty(0, dtype=torch.long, device=device), local_lens
    return torch.cat(pieces, dim=0), local_lens


def qwen_image_cu_seqlens(lengths: list[int], device: torch.device) -> tuple[torch.Tensor, int]:
    cu_seqlens = torch.tensor(
        [0, *torch.tensor(lengths, dtype=torch.int32).cumsum(0).tolist()],
        dtype=torch.int32,
        device=device,
    )
    return cu_seqlens, max(lengths) if lengths else 0


def qwen_image_sp_replicated_text_source_order_indices(
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
        image_indices, _ = qwen_image_rank_local_indices(
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


class QwenImageJointMetadataBuilder:
    """Build Qwen-Image packed joint-attention metadata from ``InputBatch``."""

    def __init__(self, transformer: Any | None = None, parallel_config: Any | None = None) -> None:
        self.transformer = transformer
        self.parallel_config = parallel_config

    def build(self, input_batch: InputBatch, *, branch: str) -> AttentionMetadata:
        device, image_lens = self._image_lens(input_batch)
        row_requests = self._row_requests(input_batch)
        text_lens = self._text_lens(input_batch, branch=branch, row_count=len(row_requests))

        lengths: list[int] = []
        segments: list[AttentionSegment] = []
        offset = 0
        for (request_id, request_index, row_in_request), text_len, image_len in zip(
            row_requests, text_lens, image_lens, strict=True
        ):
            text_len = int(text_len)
            image_len = int(image_len)
            position_suffix = f"{request_id}:{row_in_request}"
            if text_len:
                segments.append(
                    AttentionSegment(
                        request_id=request_id,
                        request_index=request_index,
                        branch=branch,
                        role="text" if branch == "positive" else "negative_text",
                        packed_start=offset,
                        length=text_len,
                        request_start=0,
                        position_id=f"{position_suffix}:{branch}:text",
                    )
                )
            if image_len:
                segments.append(
                    AttentionSegment(
                        request_id=request_id,
                        request_index=request_index,
                        branch=branch,
                        role="image",
                        packed_start=offset + text_len,
                        length=image_len,
                        request_start=0,
                        position_id=f"{position_suffix}:{branch}:image",
                    )
                )
            total_len = text_len + image_len
            lengths.append(total_len)
            offset += total_len

        cu_seqlens, max_len = qwen_image_cu_seqlens(lengths, device)
        packed_indices = qwen_image_packed_indices(image_lens, text_lens, device)
        extra: dict[str, Any] = {
            "attention_id": f"qwen_image.joint.{branch}",
            "qwen_image_dynamic": input_batch.is_dynamic,
            "request_ids": tuple(input_batch.request_ids),
            "lengths": tuple(lengths),
            "num_rows": len(row_requests),
            "image_lens": image_lens,
            "text_lens": text_lens,
            "token_to_req": packed_indices["image_token_to_req"],
            **packed_indices,
        }
        extra.update(self._sequence_parallel_plan(image_lens, text_lens, packed_indices, device))

        return AttentionMetadata(
            q_cu_seqlens=cu_seqlens,
            kv_cu_seqlens=cu_seqlens,
            max_q_len=max_len,
            max_kv_len=max_len,
            q_segments=segments,
            kv_segments=segments,
            position=self._rope_plan(input_batch, text_lens=text_lens, branch=branch),
            padded_tokens=0,
            extra=extra,
        )

    def _image_lens(self, input_batch: InputBatch) -> tuple[torch.device, list[int]]:
        if input_batch.packed_latents is not None:
            return input_batch.packed_latents.device, [int(value) for value in input_batch.img_seq_lens]
        return input_batch.latents.device, [int(input_batch.latents.shape[1])] * int(input_batch.latents.shape[0])

    @staticmethod
    def _row_requests(input_batch: InputBatch) -> list[tuple[str, int, int]]:
        row_requests: list[tuple[str, int, int]] = []
        for span in input_batch.request_spans:
            for row_in_request in range(span.row_count):
                row_requests.append((span.request_id, span.request_index, row_in_request))
        return row_requests

    @staticmethod
    def _text_lens(input_batch: InputBatch, *, branch: str, row_count: int) -> list[int]:
        if branch == "negative":
            text_lens = input_batch.negative_txt_seq_lens
            embeds = input_batch.negative_prompt_embeds
        else:
            text_lens = input_batch.txt_seq_lens
            embeds = input_batch.prompt_embeds

        if text_lens is None:
            text_lens = [int(embeds.shape[1])] * row_count
        elif len(text_lens) == input_batch.num_reqs and row_count != input_batch.num_reqs:
            text_lens = [
                int(text_lens[span.request_index]) for span in input_batch.request_spans for _ in range(span.row_count)
            ]
        return [int(value) for value in text_lens]

    def _rope_plan(
        self,
        input_batch: InputBatch,
        *,
        text_lens: list[int],
        branch: str,
    ) -> dict[str, Any]:
        """Per-request RoPE layout; the transformer recomputes the frequencies."""
        return {
            "kind": "qwen_image_rope_plan",
            "branch": branch,
            "img_shapes": input_batch.img_shapes,
            "txt_seq_lens": text_lens,
        }

    def _sp_context(self) -> tuple[str, dist.ProcessGroup, int, int] | None:
        parallel_config = self.parallel_config
        if parallel_config is None:
            return None
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

    def _sequence_parallel_plan(
        self,
        image_lens: list[int],
        text_lens: list[int],
        packed_indices: dict[str, torch.Tensor],
        device: torch.device,
    ) -> dict[str, Any]:
        sp_context = self._sp_context()
        if sp_context is None:
            return {}
        sp_mode, sp_group, sp_world_size, sp_rank = sp_context
        image_local_idx, image_local_lens = qwen_image_rank_local_indices(
            image_lens,
            rank=sp_rank,
            world_size=sp_world_size,
            device=device,
        )
        source_to_joint_idx, joint_to_source_idx, image_packed_to_source_idx = (
            qwen_image_sp_replicated_text_source_order_indices(
                text_lens=text_lens,
                image_lens=image_lens,
                world_size=sp_world_size,
                joint_gather_idx=packed_indices["joint_gather_idx"],
                device=device,
            )
        )
        image_lens_by_rank = [
            qwen_image_rank_local_indices(
                image_lens,
                rank=rank,
                world_size=sp_world_size,
                device=device,
            )[1]
            for rank in range(sp_world_size)
        ]
        return {
            "image_token_to_req": qwen_image_token_to_req(image_local_lens, device),
            "text_token_to_req": qwen_image_token_to_req(text_lens, device),
            "sp_group": sp_group,
            "sp_source_to_joint_idx": source_to_joint_idx,
            "sp_joint_to_source_idx": joint_to_source_idx,
            "sp_image_packed_to_source_idx": image_packed_to_source_idx,
            "sp_image_local_idx": image_local_idx,
            "sp_mode": sp_mode,
            "sp_rank": sp_rank,
            "sp_replicated_text": True,
            "sp_local_text_len": sum(text_lens),
            "sp_local_image_len": int(image_local_idx.numel()),
            "sp_image_lens": image_local_lens,
            "sp_image_lens_by_rank": image_lens_by_rank,
            "sp_text_lens": text_lens,
            "sp_ring_varlen_plan": (
                ring_flash_attn_varlen_replicated_prefix_plan(
                    prefix_lens=text_lens,
                    suffix_lens_by_rank=image_lens_by_rank,
                    rank=sp_rank,
                    device=device,
                )
                if sp_mode == "ring"
                else None
            ),
            "image_lens": None,
            "text_lens": None,
        }
