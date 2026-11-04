# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any

import torch

from vllm.config import VllmConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.triton_utils import tl, triton
from vllm.v1.worker.gpu.sample.gumbel import gumbel_noised_argmax
from vllm.v1.worker.gpu.spec_decode.dflash.speculator import DFlashSpeculator


@triton.jit
def _selector_walk_kernel(
    scores_ptr,
    candidate_ptr,
    sample_pos_ptr,
    req_state_ptr,
    temperature_ptr,
    seeds_ptr,
    tokens_ptr,
    realized_scores_ptr,
    num_steps: tl.constexpr,
    top_k: tl.constexpr,
    BLOCK_K: tl.constexpr,
    SAMPLE_PROBABILISTIC: tl.constexpr,
    USE_FP64: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_K)
    mask = offsets < top_k
    req_state = tl.load(req_state_ptr + row * num_steps)
    valid = req_state >= 0
    temperature = tl.load(temperature_ptr + req_state, mask=valid, other=0.0)
    seed = tl.load(seeds_ptr + req_state, mask=valid, other=0)
    previous = 0
    for step in range(num_steps):
        flat = row * num_steps + step
        score_base = (flat * top_k + previous) * top_k
        scores = tl.load(
            scores_ptr + score_base + offsets,
            mask=mask & valid,
            other=float("-inf"),
        ).to(tl.float64 if USE_FP64 else tl.float32)
        candidate_base = flat * top_k
        candidates = tl.load(
            candidate_ptr + candidate_base + offsets,
            mask=mask & valid,
            other=0,
        )

        # sample_pos is the predicted token's position P. Sampling keys a draw
        # by the position before the sampled token, P-1.
        sample_pos = tl.load(sample_pos_ptr + flat) - 1
        _, index = gumbel_noised_argmax(
            scores,
            candidates,
            mask & valid,
            seed,
            sample_pos,
            temperature if SAMPLE_PROBABILISTIC else 0.0,
            IS_DRAFTING=True,
            USE_FP64=USE_FP64,
        )

        tl.store(
            realized_scores_ptr + candidate_base + offsets,
            scores,
            mask=mask & valid,
        )
        token = tl.load(candidate_ptr + candidate_base + index, mask=valid, other=0)
        tl.store(tokens_ptr + flat, token, mask=valid)
        previous = index


@triton.jit
def _cache_draft_logits_kernel(
    draft_logits_ptr,
    cached_candidate_ptr,
    candidate_ptr,
    scores_ptr,
    req_state_ptr,
    draft_logits_stride_0,
    draft_logits_stride_1,
    num_steps: tl.constexpr,
    top_k: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    flat = tl.program_id(0)
    req_state = tl.load(req_state_ptr + flat)
    step = flat % num_steps
    offsets = tl.arange(0, BLOCK_K)
    mask = (req_state >= 0) & (offsets < top_k)
    candidate_base = flat * top_k
    cache_base = (req_state * num_steps + step) * top_k
    old_token_ids = tl.load(cached_candidate_ptr + cache_base + offsets, mask=mask)
    logits_base = (
        draft_logits_ptr
        + req_state * draft_logits_stride_0
        + step * draft_logits_stride_1
    )
    tl.store(logits_base + old_token_ids, -float("inf"), mask=mask)
    token_ids = tl.load(candidate_ptr + candidate_base + offsets, mask=mask)
    scores = tl.load(scores_ptr + candidate_base + offsets, mask=mask)
    tl.store(logits_base + token_ids, scores, mask=mask)
    tl.store(cached_candidate_ptr + cache_base + offsets, token_ids, mask=mask)


class CandidateSampler:
    """The shared DFlash2/LiLiCorr candidate walk and realized proposal cache."""

    def __init__(
        self, max_num_reqs: int, num_steps: int, top_k: int, device: torch.device
    ):
        self.num_steps = num_steps
        self.top_k = top_k
        self.scores = torch.empty(
            max_num_reqs, num_steps, top_k, dtype=torch.float32, device=device
        )
        self.cached_candidate_ids = torch.zeros(
            self.scores.shape, dtype=torch.int64, device=device
        )

    def sample(
        self,
        candidate_ids: torch.Tensor,
        scores: torch.Tensor,
        num_reqs: int,
        sample_pos: torch.Tensor,
        idx_mapping: torch.Tensor,
        temperature: torch.Tensor,
        seeds: torch.Tensor,
        draft_tokens: torch.Tensor,
        draft_logits: torch.Tensor | None,
        use_fp64: bool,
    ) -> None:
        candidate_ids = candidate_ids.contiguous()
        sample_pos = sample_pos.contiguous()
        idx_mapping = idx_mapping.contiguous()
        block_k = triton.next_power_of_2(self.top_k)
        _selector_walk_kernel[(num_reqs,)](
            scores.contiguous(),
            candidate_ids.contiguous(),
            sample_pos,
            idx_mapping,
            temperature,
            seeds,
            draft_tokens,
            self.scores,
            num_steps=self.num_steps,
            top_k=self.top_k,
            BLOCK_K=block_k,
            SAMPLE_PROBABILISTIC=draft_logits is not None,
            USE_FP64=use_fp64,
            num_warps=1,
        )

        if draft_logits is not None:
            self._cache_draft_logits(
                candidate_ids, num_reqs * self.num_steps, idx_mapping, draft_logits
            )

    def _cache_draft_logits(
        self,
        candidate_ids: torch.Tensor,
        num_sample: int,
        idx_mapping: torch.Tensor,
        draft_logits: torch.Tensor,
    ) -> None:
        block_k = triton.next_power_of_2(self.top_k)
        _cache_draft_logits_kernel[(num_sample,)](
            draft_logits,
            self.cached_candidate_ids,
            candidate_ids,
            self.scores,
            idx_mapping,
            draft_logits.stride(0),
            draft_logits.stride(1),
            num_steps=self.num_steps,
            top_k=self.top_k,
            BLOCK_K=block_k,
            num_warps=1,
        )


class DFlash2Speculator(DFlashSpeculator):
    _speculator_name = "DFlash2"
    _candidate_top_k_key = "selector_top_k"

    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        super().__init__(vllm_config, device)
        draft_config = self.draft_model_config.hf_config.dflash_config
        self.top_k = int(draft_config[self._candidate_top_k_key])
        self.candidate_sampler = CandidateSampler(
            self.max_num_reqs, self.num_speculative_steps, self.top_k, device
        )
        self.ngram_enabled = False
        self.ngram_lookups = 0
        self.ngram_hits = 0
        self.ngram_trace_enabled = False
        self.ngram_match_lengths = []
        self.ngram_request_state = None
        self.ngram_context_graph = None
        self.ngram_context_graph_replays = 0
        self.ngram_miss_backoff = 0
        self.ngram_cooldown = 0
        self.ngram_active_req = None
        self.ngram_skipped_lookups = 0
        self.ngram_native_conditional = False
        self.ngram_conditional_graphs = {}
        self.ngram_native_counts = torch.zeros(2, dtype=torch.int64, device=device)

    def init_cudagraph_manager(self, cudagraph_mode):
        super().init_cudagraph_manager(cudagraph_mode)
        self.query_cudagraph_manager.keep_raw_graph = True

    def _uses_native_ngram(self, input_batch, num_target_tokens):
        return (
            self.ngram_enabled
            and self.ngram_native_conditional
            and not self.ngram_miss_backoff
            and input_batch.num_reqs == 1
            and not input_batch.has_prefill
            and self.dp_size == self.dcp_size == 1
            and not self.enable_adaptive_verification
            and bool(self.query_cudagraph_manager.graphs)
            and num_target_tokens == self.num_query_per_req
        )

    def _publish_ngram_context(self, num_target_tokens):
        self._precompute_context_kv(0, num_target_tokens)
        self.ngram_lookup.cache_proposal(
            self.draft_logits,
            self.candidate_sampler.cached_candidate_ids,
            self.draft_tokens[0],
            self.sample_idx_mapping,
        )

    def _capture_ngram_context(self, num_target_tokens):
        if self.ngram_context_graph is None:
            self.ngram_context_graph = torch.cuda.CUDAGraph(keep_graph=True)
            with torch.cuda.graph(self.ngram_context_graph):
                self._publish_ngram_context(num_target_tokens)

    def run_ngram_conditional(self, batch_desc, input_batch, num_target_tokens):
        if not self._uses_native_ngram(input_batch, num_target_tokens):
            return False
        from dflash2_conditional import ConditionalGraph

        key = (self.ngram_lookup.min_n, self.ngram_lookup.max_n, batch_desc)
        if key not in self.ngram_conditional_graphs:
            self._capture_ngram_context(num_target_tokens)

            def lookup():
                self.ngram_lookup.lookup_async(
                    self.ngram_request_state.all_token_ids.gpu,
                    self.ngram_request_state.total_len.gpu,
                    self.sample_idx_mapping,
                    self.draft_tokens[0],
                )

            lookup()
            lookup_graph = torch.cuda.CUDAGraph(keep_graph=True)
            with torch.cuda.graph(lookup_graph):
                lookup()
            self.ngram_conditional_graphs[key] = ConditionalGraph(
                lookup_graph,
                self.ngram_context_graph,
                self.query_cudagraph_manager.graphs[batch_desc],
                self.ngram_lookup.best,
                self.ngram_native_counts,
            )
        self.ngram_conditional_graphs[key].replay()
        return True

    def bind_ngram_request_state(self, request_state):
        from .ngram import ConfirmedNgramLookup

        self.ngram_request_state = request_state
        self.ngram_lookup = ConfirmedNgramLookup(
            request_state.all_token_ids.gpu.shape[1],
            self.num_speculative_steps,
            self.device,
        )

    def try_ngram_draft(self, input_batch, num_target_tokens):
        if self._uses_native_ngram(input_batch, num_target_tokens):
            return False
        if (
            not self.ngram_enabled
            or input_batch.num_reqs != 1
            or self.dp_size != 1
            or self.dcp_size != 1
            or self.enable_adaptive_verification
        ):
            return False
        if input_batch.req_ids[0] != self.ngram_active_req:
            self.ngram_active_req = input_batch.req_ids[0]
            self.ngram_cooldown = 0
        if input_batch.has_prefill:
            return False
        if self.ngram_cooldown:
            self.ngram_cooldown -= 1
            self.ngram_skipped_lookups += 1
            return False
        assert self.ngram_request_state is not None
        self.ngram_lookups += 1
        score = self.ngram_lookup.lookup(
            self.ngram_request_state.all_token_ids.gpu,
            self.ngram_request_state.total_len.gpu,
            input_batch.idx_mapping,
            self.draft_tokens[0],
        )
        if self.ngram_trace_enabled:
            self.ngram_match_lengths.append(score // (self.ngram_lookup.max_len + 1))
        if not score:
            self.ngram_cooldown = self.ngram_miss_backoff
            return False
        self.ngram_hits += 1
        # A skipped query forward must still publish newly confirmed target
        # features to every draft layer's context KV cache.
        use_graph = (
            self.query_cudagraph_manager is not None
            and bool(self.query_cudagraph_manager.graphs)
            and num_target_tokens == self.num_query_per_req
        )

        if use_graph:
            self._capture_ngram_context(num_target_tokens)
            self.ngram_context_graph.replay()
            self.ngram_context_graph_replays += 1
        else:
            self._publish_ngram_context(num_target_tokens)
        return True

    def draft_logits_spec(self, vllm_config: VllmConfig) -> tuple[torch.dtype, float]:
        # fp32 so the walk and the rejection that checks it read the same
        # distribution; -inf because the cache kernel writes only the K
        # candidates.
        return torch.float32, -float("inf")

    def _generate_draft(
        self,
        num_reqs: int,
        num_tokens_padded: int,
        attn_metadata: dict[str, Any] | None,
        slot_mappings: dict[str, torch.Tensor] | None,
        num_tokens_across_dp: torch.Tensor | None,
        cudagraph_runtime_mode: CUDAGraphMode = CUDAGraphMode.NONE,
    ) -> None:
        last_hidden_states = self._run_model(
            num_tokens_padded,
            attn_metadata,
            slot_mappings,
            num_tokens_across_dp,
            cudagraph_runtime_mode,
        )
        num_sample = num_reqs * self.num_speculative_steps
        hidden_states = last_hidden_states[self.sample_indices[:num_sample]].view(
            num_reqs, self.num_speculative_steps, -1
        )
        candidate_ids, unary_logits = self.model.compute_candidates(
            hidden_states.flatten(0, 1)
        )
        candidate_ids = candidate_ids.view(
            num_reqs, self.num_speculative_steps, self.top_k
        )
        unary_logits = unary_logits.view_as(candidate_ids)
        scores = self._score_candidates(candidate_ids, unary_logits, hidden_states)
        self.candidate_sampler.sample(
            candidate_ids,
            scores,
            num_reqs,
            self.sample_pos,
            self.sample_idx_mapping,
            self.temperature,
            self.seeds,
            self.draft_tokens,
            self.draft_logits,
            self.use_fp64_gumbel,
        )
        if self.enable_adaptive_verification:
            self._maybe_predict_acceptance(
                self.candidate_sampler.scores[:num_reqs].flatten(0, 1),
                self.sample_idx_mapping[:num_sample],
                self.sample_col[:num_sample],
            )

    def _score_candidates(
        self,
        candidate_ids: torch.Tensor,
        unary_logits: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        num_reqs = candidate_ids.shape[0]
        anchor_token_ids = self.input_buffers.input_ids[
            : num_reqs * self.num_query_per_req : self.num_query_per_req
        ]
        return self.model.model.candidate_selector(
            candidate_ids, unary_logits, hidden_states, anchor_token_ids
        )
