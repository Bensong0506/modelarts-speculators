# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exact suffix retrieval from a request's confirmed token history."""

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _find_matches(
    history,
    lengths,
    mapping,
    partials,
    stride: tl.constexpr,
    max_len: tl.constexpr,
    steps: tl.constexpr,
    min_n: tl.constexpr,
    max_n: tl.constexpr,
    BLOCK: tl.constexpr,
):
    block = tl.program_id(0)
    row = tl.program_id(1)
    req = tl.load(mapping + row)
    length = tl.load(lengths + req)
    ends = block * BLOCK + tl.arange(0, BLOCK) + 1
    valid = (ends <= length - steps) & (ends <= max_len)
    matched = valid
    score = tl.full((BLOCK,), 0, tl.int32)
    for offset in tl.static_range(max_n):
        candidate = tl.load(
            history + req * stride + ends - 1 - offset,
            mask=valid & (ends > offset),
            other=-1,
        )
        suffix = tl.load(
            history + req * stride + length - 1 - offset,
            mask=length > offset,
            other=-2,
        )
        matched &= (ends > offset) & (length > offset) & (candidate == suffix)
        if offset + 1 >= min_n:
            score = tl.where(matched, (offset + 1) * (max_len + 1) + ends, score)
    tl.store(partials + row * tl.num_programs(0) + block, tl.max(score, 0))


@triton.jit
def _choose_match(
    history,
    mapping,
    partials,
    best,
    tokens,
    stride: tl.constexpr,
    max_len: tl.constexpr,
    num_blocks: tl.constexpr,
    steps: tl.constexpr,
    BLOCK: tl.constexpr,
    TOKEN_BLOCK: tl.constexpr,
):
    offsets = tl.arange(0, BLOCK)
    score = tl.max(tl.load(partials + offsets, offsets < num_blocks, 0), 0)
    tl.store(best, score)
    end = score % (max_len + 1)
    req = tl.load(mapping)
    cols = tl.arange(0, TOKEN_BLOCK)
    values = tl.load(
        history + req * stride + end + cols,
        mask=(score > 0) & (cols < steps),
        other=0,
    )
    tl.store(tokens + cols, values, cols < steps)


@triton.jit
def _cache_point_mass(
    logits,
    cached_ids,
    tokens,
    mapping,
    stride_req: tl.constexpr,
    stride_step: tl.constexpr,
    steps: tl.constexpr,
    top_k: tl.constexpr,
    BLOCK: tl.constexpr,
):
    step = tl.program_id(0)
    req = tl.load(mapping)
    cols = tl.arange(0, BLOCK)
    base = (req * steps + step) * top_k
    old_ids = tl.load(cached_ids + base + cols, cols < top_k, 0)
    row = logits + req * stride_req + step * stride_step
    tl.store(row + old_ids, -float("inf"), cols < top_k)
    tl.debug_barrier()
    token = tl.load(tokens + step)
    tl.store(row + token, 0.0)
    # Keep the retrieval token in the sparse cache's next clearing set.
    tl.store(cached_ids + base, token)


class ConfirmedNgramLookup:
    """Retrieve K known continuation tokens; prefer longer then newer matches.

    This initial policy serves single-request decode batches. The host reads
    one scalar to decide whether the neural drafter can be skipped.
    """

    def __init__(self, max_len, steps, device, min_n=3, max_n=8):
        self.max_len = max_len
        self.steps = steps
        self.min_n = min_n
        self.max_n = max_n
        self.num_blocks = triton.cdiv(max_len, 256)
        self.partials = torch.empty(self.num_blocks, dtype=torch.int32, device=device)
        self.best = torch.empty((), dtype=torch.int32, device=device)

    def lookup_async(self, history, lengths, mapping, tokens):
        _find_matches[(self.num_blocks,)](
            history,
            lengths,
            mapping,
            self.partials,
            history.stride(0),
            self.max_len,
            self.steps,
            self.min_n,
            self.max_n,
            256,
        )
        _choose_match[(1,)](
            history,
            mapping,
            self.partials,
            self.best,
            tokens,
            history.stride(0),
            self.max_len,
            self.num_blocks,
            self.steps,
            triton.next_power_of_2(self.num_blocks),
            triton.next_power_of_2(self.steps),
        )
        return self.best

    def lookup(self, history, lengths, mapping, tokens):
        return int(self.lookup_async(history, lengths, mapping, tokens).item())

    def cache_proposal(self, logits, cached_ids, tokens, mapping):
        if logits is not None:
            _cache_point_mass[(self.steps,)](
                logits,
                cached_ids,
                tokens,
                mapping,
                logits.stride(0),
                logits.stride(1),
                self.steps,
                cached_ids.shape[-1],
                triton.next_power_of_2(cached_ids.shape[-1]),
            )
