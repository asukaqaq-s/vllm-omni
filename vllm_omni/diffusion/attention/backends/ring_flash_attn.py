# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024, Jiarui Fang.
# Adapted from https://github.com/feifeibear/long-context-attention


from typing import Any

import torch
import torch.distributed as dist

from vllm_omni.diffusion.attention.backends.ring.ring_selector import AttnType, select_flash_attn_impl
from vllm_omni.diffusion.attention.backends.ring.ring_utils import flatten_varlen_lse, update_out_and_lse
from vllm_omni.diffusion.distributed.comm import RingComm


def ring_flash_attn_forward(
    process_group,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    softmax_scale,
    dropout_p=0,
    causal=True,
    window_size=(-1, -1),
    softcap=0.0,
    alibi_slopes=None,
    deterministic=False,
    attn_type: AttnType = AttnType.FA,
    attn_processor=None,
    joint_tensor_key=None,
    joint_tensor_value=None,
    joint_strategy="front",
):
    # Validate causal + joint_strategy combination
    # When causal=True and joint_strategy="rear", the causal mask would incorrectly
    # prevent local query tokens from attending to joint key tokens (which are
    # concatenated at the end). This breaks the semantics where joint tokens
    # (e.g., text conditioning) should be visible to all local tokens.
    if causal and joint_tensor_key is not None and joint_strategy == "rear":
        raise ValueError(
            "joint_strategy='rear' is not compatible with causal=True in Ring Attention. "
            "When using causal attention with joint tokens, use joint_strategy='front' "
            "to ensure joint tokens act as a visible prefix for all local tokens. "
            "With 'rear' strategy, the causal mask would incorrectly block local tokens "
            "from seeing the joint tokens."
        )

    comm = RingComm(process_group)

    out = None
    lse = None

    next_k, next_v = None, None

    # Check and adjust q, k, v to be contiguous
    if not q.is_contiguous():
        q = q.contiguous()
    if not k.is_contiguous():
        k = k.contiguous()
    if not v.is_contiguous():
        v = v.contiguous()

    for step in range(comm.world_size):
        if step + 1 != comm.world_size:
            next_k: torch.Tensor
            next_v: torch.Tensor
            next_k = comm.send_recv(k)
            next_v = comm.send_recv(v)
            comm.commit()

        if not causal or step <= comm.rank:
            step_k = k
            step_v = v
            if step == 0 and joint_tensor_key is not None:
                if joint_strategy == "front":
                    step_k = torch.cat([joint_tensor_key, step_k], dim=1)
                    step_v = torch.cat([joint_tensor_value, step_v], dim=1)
                else:
                    step_k = torch.cat([step_k, joint_tensor_key], dim=1)
                    step_v = torch.cat([step_v, joint_tensor_value], dim=1)

            fn = select_flash_attn_impl(attn_type, stage="fwd-only", attn_processor=attn_processor)
            block_out, block_lse = fn(
                q,
                step_k,
                step_v,
                dropout_p=dropout_p,
                softmax_scale=softmax_scale,
                causal=causal and step == 0,
                window_size=window_size,
                softcap=softcap,
                alibi_slopes=alibi_slopes,
                return_softmax=True and dropout_p > 0,
            )

            # Ensure block_out is contiguous if needed, though usually it is from FA

            if attn_type == AttnType.SPARSE_SAGE:
                out, lse = block_out, block_lse
            else:
                out, lse = update_out_and_lse(out, lse, block_out, block_lse)

        if step + 1 != comm.world_size:
            comm.wait()
            k = next_k
            v = next_v

    out = out.to(q.dtype)
    if attn_type != AttnType.SPARSE_SAGE:
        lse = lse.squeeze(dim=-1).transpose(1, 2)
    return out, lse


def _ring_varlen_cu_seqlens(lengths: list[int], device: torch.device) -> tuple[torch.Tensor, int]:
    cu_values = [0]
    max_len = 0
    for length in lengths:
        length = int(length)
        cu_values.append(cu_values[-1] + length)
        max_len = max(max_len, length)
    return torch.tensor(cu_values, dtype=torch.int32, device=device), max_len


def _ring_varlen_replicated_prefix_indices(
    suffix_lens: list[int],
    prefix_lens: list[int],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    total_prefix = sum(prefix_lens)
    prefix_offset = 0
    suffix_offset = 0
    packed_offset = 0
    gather_indices: list[torch.Tensor] = []
    prefix_restore_indices: list[torch.Tensor] = []
    suffix_restore_indices: list[torch.Tensor] = []
    for prefix_len, suffix_len in zip(prefix_lens, suffix_lens, strict=True):
        gather_indices.append(torch.arange(prefix_offset, prefix_offset + prefix_len, dtype=torch.long, device=device))
        prefix_restore_indices.append(
            torch.arange(packed_offset, packed_offset + prefix_len, dtype=torch.long, device=device)
        )
        packed_offset += prefix_len
        gather_indices.append(
            torch.arange(
                total_prefix + suffix_offset,
                total_prefix + suffix_offset + suffix_len,
                dtype=torch.long,
                device=device,
            )
        )
        suffix_restore_indices.append(
            torch.arange(packed_offset, packed_offset + suffix_len, dtype=torch.long, device=device)
        )
        packed_offset += suffix_len
        prefix_offset += prefix_len
        suffix_offset += suffix_len
    return torch.cat(gather_indices, dim=0), torch.cat([*prefix_restore_indices, *suffix_restore_indices], dim=0)


def ring_flash_attn_varlen_replicated_prefix_plan(
    *,
    prefix_lens: list[int],
    suffix_lens_by_rank: list[list[int]],
    rank: int,
    device: torch.device,
) -> dict[str, Any]:
    local_suffix_lens = suffix_lens_by_rank[rank]
    q_gather_idx, q_restore_idx = _ring_varlen_replicated_prefix_indices(
        local_suffix_lens,
        prefix_lens,
        device,
    )
    q_lens = [prefix_len + suffix_len for prefix_len, suffix_len in zip(prefix_lens, local_suffix_lens, strict=True)]
    q_cu_seqlens, max_q_len = _ring_varlen_cu_seqlens(q_lens, device)

    kv_cu_seqlens_by_owner: list[torch.Tensor] = []
    max_kv_len_by_owner: list[int] = []
    for owner, suffix_lens in enumerate(suffix_lens_by_rank):
        if owner == rank:
            kv_cu_seqlens_by_owner.append(q_cu_seqlens)
            max_kv_len_by_owner.append(max_q_len)
        else:
            kv_cu_seqlens, max_kv_len = _ring_varlen_cu_seqlens(suffix_lens, device)
            kv_cu_seqlens_by_owner.append(kv_cu_seqlens)
            max_kv_len_by_owner.append(max_kv_len)

    return {
        "q_gather_idx": q_gather_idx,
        "q_restore_idx": q_restore_idx,
        "q_cu_seqlens": q_cu_seqlens,
        "max_q_len": max_q_len,
        "kv_cu_seqlens_by_owner": kv_cu_seqlens_by_owner,
        "max_kv_len_by_owner": max_kv_len_by_owner,
    }


def _normalize_varlen_lse(
    lse: torch.Tensor,
    q_cu_seqlens: torch.Tensor,
    query: torch.Tensor,
) -> torch.Tensor:
    query_len = int(query.shape[0])
    head_count = int(query.shape[1])
    if lse.ndim == 2:
        if int(lse.shape[0]) == head_count and int(lse.shape[1]) == query_len:
            return lse.contiguous()
        if int(lse.shape[0]) == query_len and int(lse.shape[1]) == head_count:
            return lse.transpose(0, 1).contiguous()
    if lse.ndim == 3:
        if int(lse.shape[1]) == head_count:
            return flatten_varlen_lse(lse, q_cu_seqlens).contiguous()
        if int(lse.shape[0]) == head_count:
            pieces = []
            for i in range(int(q_cu_seqlens.numel()) - 1):
                start = int(q_cu_seqlens[i].item())
                end = int(q_cu_seqlens[i + 1].item())
                pieces.append(lse[:, i, : end - start])
            return torch.cat(pieces, dim=1).contiguous()
    raise ValueError(f"Unsupported varlen LSE shape {tuple(lse.shape)} for query shape {tuple(query.shape)}.")


def _flash_varlen_with_lse(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    q_cu_seqlens: torch.Tensor,
    kv_cu_seqlens: torch.Tensor,
    max_q_len: int,
    max_kv_len: int,
    softmax_scale: float | None,
    causal: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    from vllm_omni.diffusion.attention.backends.utils.fa import flash_attn_varlen_func

    if flash_attn_varlen_func is None:
        raise ImportError("Ring varlen attention requires flash_attn_varlen_func.")

    kwargs = {
        "q": query,
        "k": key,
        "v": value,
        "cu_seqlens_q": q_cu_seqlens,
        "cu_seqlens_k": kv_cu_seqlens,
        "max_seqlen_q": max_q_len,
        "max_seqlen_k": max_kv_len,
        "causal": causal,
        "softmax_scale": softmax_scale,
    }
    try:
        result = flash_attn_varlen_func(**kwargs, return_softmax_lse=True)
    except TypeError:
        result = flash_attn_varlen_func(**kwargs, return_attn_probs=True)
    if not isinstance(result, tuple) or len(result) < 2:
        raise RuntimeError("flash_attn_varlen_func did not return softmax_lse.")

    out, lse = result[0], result[1]
    return out, _normalize_varlen_lse(lse, q_cu_seqlens, query)


def _update_flat_varlen_out_and_lse(
    out: torch.Tensor | None,
    lse: torch.Tensor | None,
    block_out: torch.Tensor,
    block_lse: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    block_out = block_out.to(torch.float32)
    block_lse = block_lse.transpose(0, 1).unsqueeze(-1).to(torch.float32)
    if out is None or lse is None:
        return block_out, block_lse

    out = out - torch.sigmoid(block_lse - lse) * (out - block_out)
    lse = lse - torch.nn.functional.logsigmoid(lse - block_lse)
    return out, lse


def ring_flash_attn_varlen_with_replicated_prefix_func(
    prefix_query: torch.Tensor,
    prefix_key: torch.Tensor,
    prefix_value: torch.Tensor,
    suffix_query: torch.Tensor,
    suffix_key: torch.Tensor,
    suffix_value: torch.Tensor,
    *,
    prefix_lens: list[int],
    suffix_lens_by_rank: list[list[int]],
    rank: int | None = None,
    dropout_p: float = 0.0,
    softmax_scale: float | None = None,
    causal: bool = False,
    group: dist.ProcessGroup | None = None,
    return_attn_probs: bool = False,
    varlen_plan: dict[str, Any] | None = None,
) -> tuple[torch.Tensor, torch.Tensor] | tuple[tuple[torch.Tensor, torch.Tensor], torch.Tensor, None]:
    """Ring varlen attention for flat requests with a replicated prefix.

    Query tokens are packed per request as ``[prefix, local suffix]``.  K/V
    suffix shards rotate around the ring; the replicated prefix K/V is visible
    only on the local step, matching the dense Ring joint-prefix path.
    """
    if dropout_p != 0.0:
        raise ValueError("Ring varlen attention is inference-only and requires dropout_p=0.")
    if softmax_scale is None:
        softmax_scale = prefix_query.shape[-1] ** (-0.5)

    world_size = dist.get_world_size(group)
    if rank is None:
        rank = dist.get_rank(group)

    if varlen_plan is None:
        varlen_plan = ring_flash_attn_varlen_replicated_prefix_plan(
            prefix_lens=prefix_lens,
            suffix_lens_by_rank=suffix_lens_by_rank,
            rank=rank,
            device=prefix_query.device,
        )
    source_query = torch.cat([prefix_query, suffix_query], dim=0)
    q_gather_idx = varlen_plan["q_gather_idx"]
    q_restore_idx = varlen_plan["q_restore_idx"]
    query = source_query.index_select(0, q_gather_idx)
    q_cu_seqlens = varlen_plan["q_cu_seqlens"]
    max_q_len = int(varlen_plan["max_q_len"])

    if world_size == 1:
        source_key = torch.cat([prefix_key, suffix_key], dim=0)
        source_value = torch.cat([prefix_value, suffix_value], dim=0)
        key = source_key.index_select(0, q_gather_idx)
        value = source_value.index_select(0, q_gather_idx)
        joint_out, joint_lse = _flash_varlen_with_lse(
            query,
            key,
            value,
            q_cu_seqlens=q_cu_seqlens,
            kv_cu_seqlens=q_cu_seqlens,
            max_q_len=max_q_len,
            max_kv_len=max_q_len,
            softmax_scale=softmax_scale,
            causal=causal,
        )
        source_out = joint_out.index_select(0, q_restore_idx)
        prefix_len = int(prefix_query.shape[0])
        output = (source_out[:prefix_len], source_out[prefix_len:])
        return output if not return_attn_probs else (output, joint_lse, None)

    comm = RingComm(group)
    current_owner = rank
    current_key = suffix_key.contiguous()
    current_value = suffix_value.contiguous()
    out: torch.Tensor | None = None
    lse: torch.Tensor | None = None

    for step in range(world_size):
        if step + 1 != world_size:
            recv_owner = (current_owner - 1) % world_size
            recv_len = sum(suffix_lens_by_rank[recv_owner])
            next_key = current_key.new_empty((recv_len, *current_key.shape[1:]))
            next_value = current_value.new_empty((recv_len, *current_value.shape[1:]))
            next_key = comm.send_recv(current_key, recv_tensor=next_key)
            next_value = comm.send_recv(current_value, recv_tensor=next_value)
            comm.commit()

        if step == 0:
            source_key = torch.cat([prefix_key, current_key], dim=0)
            source_value = torch.cat([prefix_value, current_value], dim=0)
            key = source_key.index_select(0, q_gather_idx)
            value = source_value.index_select(0, q_gather_idx)
        else:
            key = current_key
            value = current_value

        kv_cu_seqlens = varlen_plan["kv_cu_seqlens_by_owner"][current_owner]
        max_kv_len = int(varlen_plan["max_kv_len_by_owner"][current_owner])
        block_out, block_lse = _flash_varlen_with_lse(
            query,
            key,
            value,
            q_cu_seqlens=q_cu_seqlens,
            kv_cu_seqlens=kv_cu_seqlens,
            max_q_len=max_q_len,
            max_kv_len=max_kv_len,
            softmax_scale=softmax_scale,
            causal=causal and step == 0,
        )
        out, lse = _update_flat_varlen_out_and_lse(out, lse, block_out, block_lse)

        if step + 1 != world_size:
            comm.wait()
            current_key = next_key
            current_value = next_value
            current_owner = (current_owner - 1) % world_size

    if out is None or lse is None:
        raise RuntimeError("Ring varlen attention produced no output.")

    source_out = out.to(query.dtype).index_select(0, q_restore_idx)
    prefix_len = int(prefix_query.shape[0])
    output = (source_out[:prefix_len], source_out[prefix_len:])
    softmax_lse = lse.squeeze(-1).transpose(0, 1)
    return output if not return_attn_probs else (output, softmax_lse, None)


class RingFlashAttnFunc(torch.autograd.Function):
    """Ring Flash Attention autograd function (inference only, no backward)."""

    @staticmethod
    def forward(
        ctx,
        q,
        k,
        v,
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        softcap,
        alibi_slopes,
        deterministic,
        return_softmax,
        group,
        attn_type,
        attn_processor,
        joint_tensor_key=None,
        joint_tensor_value=None,
        joint_strategy="front",
    ):
        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** (-0.5)

        assert alibi_slopes is None
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()

        out, softmax_lse = ring_flash_attn_forward(
            group,
            q,
            k,
            v,
            softmax_scale=softmax_scale,
            dropout_p=dropout_p,
            causal=causal,
            window_size=window_size,
            softcap=softcap,
            alibi_slopes=alibi_slopes,
            deterministic=False,
            attn_type=attn_type,
            attn_processor=attn_processor,
            joint_tensor_key=joint_tensor_key,
            joint_tensor_value=joint_tensor_value,
            joint_strategy=joint_strategy,
        )
        return out if not return_softmax else (out, softmax_lse, None)


def ring_flash_attn_qkvpacked_func(
    qkv,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),
    softcap=0.0,
    alibi_slopes=None,
    deterministic=False,
    return_attn_probs=False,
    group=None,
    attn_type: AttnType = AttnType.FA,
):
    return RingFlashAttnFunc.apply(
        qkv[:, :, 0],
        qkv[:, :, 1],
        qkv[:, :, 2],
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        softcap,
        alibi_slopes,
        deterministic,
        return_attn_probs,
        group,
        attn_type,
        None,  # attn_processor
        None,  # joint_tensor_key
        None,  # joint_tensor_value
        "front",  # joint_strategy
    )


def ring_flash_attn_kvpacked_func(
    q,
    kv,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),
    softcap=0.0,
    alibi_slopes=None,
    deterministic=False,
    return_attn_probs=False,
    group=None,
    attn_type: AttnType = AttnType.FA,
):
    return RingFlashAttnFunc.apply(
        q,
        kv[:, :, 0],
        kv[:, :, 1],
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        softcap,
        alibi_slopes,
        deterministic,
        return_attn_probs,
        group,
        attn_type,
        None,  # attn_processor
        None,  # joint_tensor_key
        None,  # joint_tensor_value
        "front",  # joint_strategy
    )


def ring_flash_attn_func(
    q,
    k,
    v,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),
    softcap=0.0,
    alibi_slopes=None,
    deterministic=False,
    return_attn_probs=False,
    group=None,
    attn_type: AttnType = AttnType.FA,
    attn_processor=None,
    joint_tensor_key=None,
    joint_tensor_value=None,
    joint_strategy="front",
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, None]:
    """Ring Attention forward pass using Flash Attention backend.

    Implements Ring Attention with sequence parallelism using a ring-based P2P
    communication pattern. The sequence dimension is sharded across devices, and
    Key/Value blocks are circulated through the ring to accumulate attention results.

    Args:
        q (torch.Tensor): Query tensor of shape (batch, seq_len, num_heads, head_dim).
            Sequence dimension is sharded across the ring group.
        k (torch.Tensor): Key tensor of shape (batch, seq_len, num_heads, head_dim).
            Sequence dimension is sharded across the ring group.
        v (torch.Tensor): Value tensor of shape (batch, seq_len, num_heads, head_dim).
            Sequence dimension is sharded across the ring group.
        dropout_p (float): Dropout probability. Defaults to 0.0.
        softmax_scale (float | None): Scaling factor for softmax.
            If None, computed as head_dim^(-0.5).
        causal (bool): Whether to apply causal masking. Defaults to False.
        window_size (tuple[int, int]): Sliding window size for attention.
            (-1, -1) means no windowing.
        softcap (float): Soft capping value for attention logits. Defaults to 0.0.
        alibi_slopes (torch.Tensor | None): ALiBi slopes for positional bias.
            Not supported.
        deterministic (bool): Whether to use deterministic algorithms.
            Defaults to False.
        return_attn_probs (bool): If True, returns (out, softmax_lse, None).
            Defaults to False.
        group (ProcessGroup | None): Process group for ring communication.
            Defaults to None.
        attn_type (AttnType): Flash Attention implementation type
            (AttnType.FA, AttnType.FA3, etc.).
        attn_processor (Callable | None): Custom attention processor for sparse
            attention. Defaults to None.
        joint_tensor_key (torch.Tensor | None): Additional key tensor for joint
            attention (e.g., text + image). Concatenated only at step=0.
            Defaults to None.
        joint_tensor_value (torch.Tensor | None): Additional value tensor for
            joint attention (e.g., text + image). Concatenated only at step=0.
            Defaults to None.
        joint_strategy (str): Concatenation strategy ("front" or "back").
            Defaults to "front".

    Returns:
        Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, None]]:
            - If return_attn_probs is False: Output tensor (batch, seq_len, num_heads, head_dim).
            - If return_attn_probs is True: A tuple (out, softmax_lse, None).
    """
    return RingFlashAttnFunc.apply(
        q,
        k,
        v,
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        softcap,
        alibi_slopes,
        deterministic,
        return_attn_probs,
        group,
        attn_type,
        attn_processor,
        joint_tensor_key,
        joint_tensor_value,
        joint_strategy,
    )
