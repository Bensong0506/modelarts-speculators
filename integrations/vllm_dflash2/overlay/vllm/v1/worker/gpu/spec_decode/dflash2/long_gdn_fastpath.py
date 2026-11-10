# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Keep the stock <=8-token GDN kernel usable with a wider rollback cache."""

import copy

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _rebase_ssm_state(
    indices,
    accepted,
    state,
    slots: tl.constexpr,
    state_size: tl.constexpr,
    state_stride: tl.constexpr,
    state_shape: tl.constexpr,
    inner_strides: tl.constexpr,
    index_stride: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    column = tl.minimum(tl.maximum(tl.load(accepted + row) - 1, 0), slots - 1)
    source = tl.load(indices + row * index_stride + column).to(tl.int64)
    destination = tl.load(indices + row * index_stride).to(tl.int64)
    offsets = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    valid = (source >= 0) & (destination >= 0) & (offsets < state_size)
    remainder = offsets
    inner = tl.full((BLOCK,), 0, tl.int64)
    for dimension in tl.static_range(len(state_shape) - 1, -1, -1):
        inner += (remainder % state_shape[dimension]) * inner_strides[dimension]
        remainder //= state_shape[dimension]
    values = tl.load(state + source * state_stride + inner, valid, other=0)
    tl.store(state + destination * state_stride + inner, values, valid)


def rebase_short_ssm_metadata(metadata, state):
    """Preserve the previous accepted state before compacting output slots."""
    indices = metadata.spec_state_indices_tensor
    accepted = metadata.num_accepted_tokens
    if state.is_cuda:
        assert indices.stride(1) == 1
        state_size = state[0].numel()
        _rebase_ssm_state[(indices.size(0), triton.cdiv(state_size, 256))](
            indices,
            accepted,
            state,
            indices.size(1),
            state_size,
            state.stride(0),
            state.shape[1:],
            state.stride()[1:],
            indices.stride(0),
            256,
        )
    else:
        source_column = (accepted.to(torch.int64) - 1).clamp(0, indices.size(1) - 1)
        source = indices.gather(1, source_column[:, None]).flatten().to(torch.int64)
        destination = indices[:, 0].to(torch.int64)
        state.index_copy_(0, destination, state.index_select(0, source))
    result = copy.copy(metadata)
    result.spec_state_indices_tensor = indices[:, :8].contiguous()
    result.num_accepted_tokens = torch.ones_like(accepted)
    return result


def install_long_short_gdn_fastpath():
    from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
        QwenGatedDeltaNetAttention,
    )

    cls = QwenGatedDeltaNetAttention
    if getattr(cls, "_swx_long_short_installed", False):
        return
    original_check = cls._can_use_fused_gdn_mtp_decode
    original_post_conv = cls._forward_core_decode_spec_post_conv_fused_norm

    def is_short_wide(metadata):
        indices = metadata.spec_state_indices_tensor
        return (
            indices is not None
            and indices.size(1) > 8
            and metadata.num_prefills == 0
            and metadata.num_spec_decodes > 0
            and (
                metadata.uniform_spec_sequence_length is not None
                and metadata.uniform_spec_sequence_length <= 8
                or indices.size(0) == 1
                and metadata.num_actual_tokens <= 8
            )
        )

    def can_use(self, metadata):
        if is_short_wide(metadata):
            compact = copy.copy(metadata)
            compact.spec_state_indices_tensor = metadata.spec_state_indices_tensor[
                :, :8
            ]
            result = original_check(self, compact)
            if result:
                self._swx_short_fused_enabled = True
            return result
        return original_check(self, metadata)

    def post_conv(self, *args, **kwargs):
        metadata = kwargs["attn_metadata"]
        if is_short_wide(metadata):
            kwargs["attn_metadata"] = rebase_short_ssm_metadata(
                metadata, self.kv_cache[1]
            )
        return original_post_conv(self, *args, **kwargs)

    cls._can_use_fused_gdn_mtp_decode = can_use
    cls._forward_core_decode_spec_post_conv_fused_norm = post_conv
    cls._swx_long_short_installed = True
