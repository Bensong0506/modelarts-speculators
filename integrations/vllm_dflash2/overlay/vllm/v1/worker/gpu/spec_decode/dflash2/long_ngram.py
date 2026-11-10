# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Long retrieved drafts with an unchanged 7-token DFlash2 query network."""

import copy

import torch

from vllm.triton_utils import tl, triton
from vllm.v1.worker.gpu.spec_decode.dflash2.ngram import _find_matches
from vllm.v1.worker.gpu.spec_decode.dflash2.speculator import DFlash2Speculator


@triton.jit
def _select_long_match(
    history,
    lengths,
    mapping,
    partials,
    status,
    tokens,
    neural,
    stride: tl.constexpr,
    max_len: tl.constexpr,
    num_blocks: tl.constexpr,
    min_steps: tl.constexpr,
    max_steps: tl.constexpr,
    BLOCK: tl.constexpr,
    TOKEN_BLOCK: tl.constexpr,
    CONFIRM_NEURAL: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK)
    score = tl.max(
        tl.load(partials + row * num_blocks + offsets, offsets < num_blocks, 0), 0
    )
    end = score % (max_len + 1)
    req = tl.load(mapping + row)
    available = tl.load(lengths + req) - end
    count = tl.where(available >= max_steps, max_steps, min_steps)
    count = tl.where(score > 0, count, 0)
    cols = tl.arange(0, TOKEN_BLOCK)
    values = tl.load(
        history + req * stride + end + cols,
        mask=(score > 0) & (cols < count),
        other=-1,
    )
    if CONFIRM_NEURAL:
        proposed = tl.load(neural + row * 7 + cols, cols < 7, other=-2)
        agrees = tl.sum(((cols < 7) & (values == proposed)).to(tl.int32), 0) == 7
        count = tl.where(agrees, count, 0)
    tl.store(status + row, score * 64 + count)
    tl.store(tokens + row * max_steps + cols, values, cols < max_steps)


@triton.jit
def _assemble_long_drafts(
    output,
    neural,
    retrieved,
    status,
    cu_logits,
    num_sampled,
    stats,
    max_steps: tl.constexpr,
    RECORD: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    count = tl.load(status + row) % 64
    cols = tl.arange(0, BLOCK)
    old = tl.load(neural + row * 7 + cols, cols < 7, other=0)
    copied = tl.load(retrieved + row * max_steps + cols, cols < count, other=0)
    values = tl.where(count > 0, copied, old)
    tl.store(output + row * max_steps + cols, values, cols < max_steps)
    if RECORD:
        width = tl.load(cu_logits + row + 1) - tl.load(cu_logits + row) - 1
        accepted = tl.minimum(tl.maximum(tl.load(num_sampled + row) - 1, 0), width)
        group = tl.where(
            width == 7, 0, tl.where(width == 15, 1, tl.where(width == 31, 2, 3))
        )
        if width > 0:
            tl.atomic_add(stats + group * 5, 1)
            tl.atomic_add(stats + group * 5 + 1, width)
            tl.atomic_add(stats + group * 5 + 2, accepted)
            tl.atomic_add(stats + group * 5 + 3, (accepted == 0).to(tl.int64))
            tl.atomic_add(stats + group * 5 + 4, (accepted == width).to(tl.int64))


class LongConfirmedNgramLookup:
    def __init__(
        self,
        max_len,
        device,
        min_steps=15,
        max_steps=31,
        min_n=8,
        max_n=16,
        max_reqs=1,
    ):
        assert 7 < min_steps <= max_steps < 64
        self.max_len = max_len
        self.min_steps = min_steps
        self.max_steps = max_steps
        self.min_n = min_n
        self.max_n = max_n
        self.num_blocks = triton.cdiv(max_len, 256)
        self.partials = torch.empty(
            max_reqs, self.num_blocks, dtype=torch.int32, device=device
        )
        self.status = torch.empty(max_reqs, dtype=torch.int64, device=device)

    def lookup(self, history, lengths, mapping, tokens):
        return self.lookup_batch(history, lengths, mapping[:1], tokens)[0]

    def lookup_batch(self, history, lengths, mapping, tokens, neural=None):
        num_reqs = mapping.numel()
        _find_matches[(self.num_blocks, num_reqs)](
            history,
            lengths,
            mapping,
            self.partials,
            history.stride(0),
            self.max_len,
            self.min_steps,
            self.min_n,
            self.max_n,
            256,
        )
        _select_long_match[(num_reqs,)](
            history,
            lengths,
            mapping,
            self.partials,
            self.status,
            tokens,
            tokens if neural is None else neural,
            history.stride(0),
            self.max_len,
            self.num_blocks,
            self.min_steps,
            self.max_steps,
            triton.next_power_of_2(self.num_blocks),
            triton.next_power_of_2(self.max_steps),
            neural is not None,
        )
        values = self.status[:num_reqs].cpu().tolist()
        self.last_raw_hits = sum(value // 64 > 0 for value in values)
        return [(value // 64 if value % 64 else 0, value % 64) for value in values]


class LongNgramDFlash2Speculator(DFlash2Speculator):
    def __init__(self, vllm_config, device):
        config = vllm_config.additional_config["swx_long_ngram"]
        if config.get("fast_short_gdn", False):
            from .long_gdn_fastpath import install_long_short_gdn_fastpath

            install_long_short_gdn_fastpath()
        self.long_max_reqs = vllm_config.scheduler_config.max_num_seqs
        self.long_neural_confirm = bool(config.get("neural_confirm", False))
        self.long_min_n = int(config.get("min_n", 8))
        if vllm_config.scheduler_config.async_scheduling:
            raise ValueError(
                "Host-dispatched long drafts require async_scheduling=False."
            )
        self.long_max_steps = vllm_config.num_speculative_tokens
        self.long_min_steps = int(config.get("min_long_steps", 15))
        if not 7 < self.long_min_steps <= self.long_max_steps < 64:
            raise ValueError("Expected 7 < min_long_steps <= max_steps < 64.")
        private_config = copy.copy(vllm_config)
        private_config.speculative_config = copy.copy(vllm_config.speculative_config)
        private_config.speculative_config.num_speculative_tokens = 7
        private_config.additional_config = dict(vllm_config.additional_config)
        private_config.additional_config.pop("swx_long_ngram")
        if private_config.speculative_config.draft_sample_method != "greedy":
            raise ValueError("This private experiment only supports greedy drafts.")
        self._long_context_graphs = {}
        self._long_context_graphs_ready = False
        super().__init__(private_config, device)
        self.long_enabled = False
        self.last_long_proposal_len = 7
        self.long_output_tokens = torch.empty(
            self.long_max_reqs, self.long_max_steps, dtype=torch.int64, device=device
        )
        self.long_retrieved_tokens = torch.empty_like(self.long_output_tokens)
        self.long_acceptance_stats = torch.zeros(4, 5, dtype=torch.int64, device=device)
        self.last_long_proposal_lens = [7]
        self.long_lookups = 0
        self.long_hits_by_length = {15: 0, 31: 0}
        self.long_raw_hits = 0
        self._long_hit_this_step = False

    def init_cudagraph_manager(self, cudagraph_mode):
        self.vllm_config.compilation_config = copy.copy(
            self.vllm_config.compilation_config
        )
        self.vllm_config.compilation_config.cudagraph_capture_sizes = list(
            range(8, 8 * self.long_max_reqs + 1, 8)
        )
        self.vllm_config.compilation_config.max_cudagraph_capture_size = (
            8 * self.long_max_reqs
        )
        super().init_cudagraph_manager(cudagraph_mode)

    def bind_ngram_request_state(self, request_state):
        super().bind_ngram_request_state(request_state)
        self.long_lookup = LongConfirmedNgramLookup(
            request_state.all_token_ids.gpu.shape[1],
            self.device,
            min_steps=self.long_min_steps,
            max_steps=self.long_max_steps,
            min_n=self.long_min_n,
            max_reqs=self.long_max_reqs,
        )

    def _precompute_context_kv(self, start, end, dummy_run=False):
        if (
            self._long_context_graphs_ready
            and not dummy_run
            and 0 < end <= self.long_max_steps + 1
            and not torch.cuda.is_current_stream_capturing()
        ):
            key = (start, end)
            if key not in self._long_context_graphs:
                super()._precompute_context_kv(start, end)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    super()._precompute_context_kv(start, end)
                self._long_context_graphs[key] = graph
            self._long_context_graphs[key].replay()
            return
        super()._precompute_context_kv(start, end, dummy_run)

    def try_ngram_draft(self, input_batch, num_target_tokens):
        if (
            not self.long_enabled
            or input_batch.has_prefill
            or self.long_neural_confirm
            or input_batch.num_reqs != 1
        ):
            return False
        self.long_lookups += 1
        score, count = self.long_lookup.lookup(
            self.ngram_request_state.all_token_ids.gpu,
            self.ngram_request_state.total_len.gpu,
            input_batch.idx_mapping,
            self.long_output_tokens[0],
        )
        self.long_raw_hits += self.long_lookup.last_raw_hits
        if not score:
            return False
        self.last_long_proposal_len = count
        self.last_long_proposal_lens = [count]
        self.long_hits_by_length[count] = self.long_hits_by_length.get(count, 0) + 1
        self._long_hit_this_step = True
        self._precompute_context_kv(0, num_target_tokens)
        return True

    @torch.inference_mode()
    def propose(self, *args, **kwargs):
        input_batch = args[0] if args else kwargs["input_batch"]
        num_reqs = input_batch.num_reqs
        self.last_long_proposal_len = 7
        self.last_long_proposal_lens = [7] * num_reqs
        self._long_hit_this_step = False
        neural = super().propose(*args, **kwargs)
        output = self.long_output_tokens[:num_reqs]
        num_sampled = args[5] if len(args) > 5 else kwargs.get("num_sampled")
        retrieved = output if self._long_hit_this_step else self.long_retrieved_tokens
        if not self._long_hit_this_step:
            # Startup warmup may schedule the maximum width independently of
            # the real handler length. Unused ids must remain in vocabulary.
            if num_sampled is None:
                output.zero_()
                output[:, :7].copy_(neural)
            else:
                self.long_lookup.status[:num_reqs].zero_()
            if (
                self.long_enabled
                and not kwargs.get("dummy_run", False)
                and not input_batch.has_prefill
                and (self.long_neural_confirm or num_reqs > 1)
            ):
                self.long_lookups += num_reqs
                matches = self.long_lookup.lookup_batch(
                    self.ngram_request_state.all_token_ids.gpu,
                    self.ngram_request_state.total_len.gpu,
                    input_batch.idx_mapping[:num_reqs],
                    self.long_retrieved_tokens,
                    neural=neural if self.long_neural_confirm else None,
                )
                self.long_raw_hits += self.long_lookup.last_raw_hits
                for row, (score, count) in enumerate(matches):
                    if score:
                        self.last_long_proposal_lens[row] = count
                        self.long_hits_by_length[count] += 1
                self.last_long_proposal_len = max(self.last_long_proposal_lens)
        if num_sampled is not None:
            _assemble_long_drafts[(num_reqs,)](
                output,
                neural,
                retrieved,
                self.long_lookup.status,
                input_batch.cu_num_logits,
                num_sampled,
                self.long_acceptance_stats,
                self.long_max_steps,
                not kwargs.get("dummy_run", False),
                triton.next_power_of_2(self.long_max_steps),
            )
        return output
