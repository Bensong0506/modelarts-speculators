# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.models.qwen3_dflash import (
    _add_global_draft_layer_exclusions,
)
from vllm.model_executor.models.qwen3_dflash2 import _grouped_conv, _score_edges
from vllm.platforms import current_platform
from vllm.v1.worker.gpu.spec_decode.dflash.speculator import DFlashSpeculator
from vllm.v1.worker.gpu.spec_decode.dflash2.speculator import DFlash2Speculator


def test_dmtp_architecture_survives_eagle_config_and_requires_hybrid_proposer():
    from transformers import Qwen3Config

    from vllm.transformers_utils.configs.eagle import EAGLEConfig
    from vllm.v1.worker.gpu.spec_decode import init_speculator

    config = EAGLEConfig(
        Qwen3Config(architectures=["DMTPDFlash2DraftModel"]), method="dflash"
    )
    assert config.architectures == ["DMTPDFlash2DraftModel"]
    with pytest.raises(ValueError, match="require additional_config.swx_dmtp"):
        init_speculator(
            SimpleNamespace(
                additional_config={},
                speculative_config=SimpleNamespace(
                    method="dflash", draft_model_config=config
                ),
            ),
            torch.device("cpu"),
        )


@pytest.mark.skipif(not current_platform.is_cuda(), reason="Requires CUDA")
def test_dmtp_captures_prediction_hidden_before_hc_feedback_and_replays(monkeypatch):
    from vllm.v1.worker.gpu.spec_decode.autoregressive.speculator import (
        AutoRegressiveSpeculator,
    )
    from vllm.v1.worker.gpu.spec_decode.dmtp.speculator import FirstTokenMTP

    # Native MTP returns a single-stream sample plus a distinct HC4 feedback.
    # Fusion must capture exactly the former, including after graph replay.
    sample = torch.arange(8 * 16, device="cuda", dtype=torch.float32).view(8, 16)
    feedback = torch.full((8, 64), -99.0, device="cuda")
    monkeypatch.setattr(
        AutoRegressiveSpeculator,
        "_run_model",
        lambda *args, **kwargs: (sample, feedback),
    )
    spec = object.__new__(FirstTokenMTP)
    spec.first_hidden = torch.zeros(4, 16, device="cuda")
    spec.last_token_indices = torch.tensor([2, 6, 0, 0], device="cuda")
    spec.input_buffers = SimpleNamespace(positions=torch.arange(8, device="cuda"))
    spec.idx_mapping = torch.tensor([3, 1, 0, 0], device="cuda")
    spec.temperature = torch.zeros(4, device="cuda")
    spec.seeds = torch.zeros(4, dtype=torch.int64, device="cuda")
    spec.current_draft_step = torch.tensor(0, device="cuda")
    spec.draft_logits = None
    spec.draft_tokens = torch.zeros(4, 1, dtype=torch.int64, device="cuda")
    spec.hidden_states = torch.zeros_like(feedback)
    spec.sample_src_positions = torch.zeros(4, dtype=torch.int64, device="cuda")
    spec.pcp_manager = None
    spec.sample_draft = lambda h, *args: h[:, 0].long()
    spec._prefill(2, 8, None, None, None)
    torch.testing.assert_close(spec.first_hidden[:2], sample[[2, 6]])
    torch.testing.assert_close(spec.hidden_states[:2], feedback[[2, 6]])
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        spec._prefill(2, 8, None, None, None)
    # A different acceptance length/request order changes the selected rows.
    spec.last_token_indices[:2].copy_(torch.tensor([7, 1], device="cuda"))
    sample.add_(1000)
    graph.replay()
    torch.testing.assert_close(spec.first_hidden[:2], sample[[7, 1]])
    torch.testing.assert_close(spec.draft_tokens[:2, 0], sample[[7, 1], 0].long())


@pytest.mark.parametrize("accepted", [1, 7, 15, 31, 32])
def test_short_gdn_rebase_preserves_long_previous_acceptance_state(accepted):
    from vllm.v1.worker.gpu.spec_decode.dflash2.long_gdn_fastpath import (
        rebase_short_ssm_metadata,
    )

    state = torch.arange(64 * 6, dtype=torch.float32).view(64, 2, 3)
    before = state.clone()
    metadata = SimpleNamespace(
        spec_state_indices_tensor=torch.arange(3, 35, dtype=torch.int32).view(1, 32),
        num_accepted_tokens=torch.tensor([accepted], dtype=torch.int32),
    )
    result = rebase_short_ssm_metadata(metadata, state)
    torch.testing.assert_close(state[3], before[3 + accepted - 1])
    torch.testing.assert_close(state[:3], before[:3])
    torch.testing.assert_close(state[4:], before[4:])
    assert result.spec_state_indices_tensor.tolist() == [list(range(3, 11))]
    assert result.num_accepted_tokens.tolist() == [1]
    assert metadata.num_accepted_tokens.tolist() == [accepted]
    assert metadata.spec_state_indices_tensor.shape == (1, 32)


def test_long_adapter_pads_warmup_with_valid_ids_without_extending_neural_draft(
    monkeypatch,
):
    from vllm.v1.worker.gpu.spec_decode.dflash2.long_ngram import (
        LongNgramDFlash2Speculator,
    )

    neural = torch.arange(7).view(1, 7)
    monkeypatch.setattr(DFlash2Speculator, "propose", lambda _self, _batch: neural)
    spec = object.__new__(LongNgramDFlash2Speculator)
    spec.long_output_tokens = torch.empty(1, 31, dtype=torch.int64)
    spec.long_retrieved_tokens = torch.empty_like(spec.long_output_tokens)
    spec.long_enabled = False
    actual = spec.propose(SimpleNamespace(num_reqs=1))
    assert spec.last_long_proposal_len == 7
    torch.testing.assert_close(actual[:, :7], neural)
    assert actual[:, 7:].tolist() == [[0] * 24]


@pytest.mark.skipif(not current_platform.is_cuda(), reason="Requires CUDA")
def test_long_ngram_batch_confirms_neural_prefix_without_mixing_request_slots():
    from vllm.v1.worker.gpu.spec_decode.dflash2.long_ngram import (
        LongConfirmedNgramLookup,
    )

    sequences = []
    for row in range(3):
        key = list(range(100 + row * 100, 112 + row * 100))
        sequences.append(key + list(range(1000, 1040)) + key)
    history = torch.zeros(3, 128, dtype=torch.int32, device="cuda")
    for row, sequence in enumerate(sequences):
        history[row, : len(sequence)] = torch.tensor(sequence, device="cuda")
    mapping = torch.tensor([2, 0, 1], dtype=torch.int32, device="cuda")
    lengths = torch.tensor([len(x) for x in sequences], device="cuda")
    neural = torch.tensor([list(range(1000, 1007))] * 3, device="cuda")
    neural[1, 0] = 9999
    result = torch.zeros(3, 31, dtype=torch.int64, device="cuda")
    lookup = LongConfirmedNgramLookup(128, "cuda", min_n=12, max_reqs=3)
    matches = lookup.lookup_batch(history, lengths, mapping, result, neural)
    assert [count for _, count in matches] == [31, 0, 31]
    assert result[0].tolist() == sequences[2][12:43]
    assert result[2].tolist() == sequences[1][12:43]


@pytest.mark.skipif(not current_platform.is_cuda(), reason="Requires CUDA")
@pytest.mark.parametrize("strided", [False, True, "transposed"])
def test_batched_short_gdn_rebase_ignores_padding_and_captures_in_cuda_graph(strided):
    from vllm.v1.worker.gpu.spec_decode.dflash2.long_gdn_fastpath import (
        rebase_short_ssm_metadata,
    )

    state = torch.arange(96 * 6, device="cuda", dtype=torch.float32).view(96, 2, 3)
    if strided:
        state = torch.arange(96 * 12, device="cuda", dtype=torch.float32).view(
            96, 4, 3
        )[:, :2]
    if strided == "transposed":
        state = state.transpose(-1, -2)
    before = state.clone()
    indices = torch.cat(
        [
            torch.arange(96, device="cuda", dtype=torch.int32).view(3, 32),
            torch.full((1, 32), -1, device="cuda", dtype=torch.int32),
        ]
    )
    metadata = SimpleNamespace(
        spec_state_indices_tensor=indices,
        num_accepted_tokens=torch.tensor([31, 7, 15, 0], device="cuda"),
    )
    rebase_short_ssm_metadata(metadata, state)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        result = rebase_short_ssm_metadata(metadata, state)
    state.copy_(before)
    graph.replay()
    expected = before.clone()
    expected[0] = before[30]
    expected[32] = before[38]
    expected[64] = before[78]
    torch.testing.assert_close(state, expected)
    assert result.spec_state_indices_tensor.shape == (4, 8)
    assert result.num_accepted_tokens.tolist() == [1, 1, 1, 1]


def test_long_scheduler_receives_individual_request_draft_lengths():
    from vllm.v1.outputs import DraftTokenIds
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner

    drafts = DraftTokenIds(["a", "b", "c"], [[-1] * 31 for _ in range(3)])
    runner = SimpleNamespace(
        draft_tokens_handler=SimpleNamespace(get_draft_tokens=lambda: drafts),
        speculator=SimpleNamespace(last_long_proposal_lens=[7, 31, 15]),
    )
    actual = GPUModelRunner.take_draft_token_ids(runner)
    assert actual.req_ids == ["a", "b", "c"]
    assert [len(x) for x in actual.draft_token_ids] == [7, 31, 15]


@pytest.mark.skipif(not current_platform.is_cuda(), reason="Requires CUDA")
def test_long_batch_assembly_preserves_neural_rows_and_counts_actual_acceptance():
    from vllm.v1.worker.gpu.spec_decode.dflash2.long_ngram import (
        _assemble_long_drafts,
    )

    neural = torch.arange(21, device="cuda").view(3, 7)
    retrieved = torch.arange(93, device="cuda").view(3, 31) + 100
    output = torch.empty_like(retrieved)
    status = torch.tensor([0, 64 + 15, 64 + 31], device="cuda")
    boundaries = torch.tensor([0, 8, 24, 56], device="cuda")
    sampled = torch.tensor([5, 1, 32], device="cuda")
    stats = torch.zeros(4, 5, device="cuda", dtype=torch.int64)
    _assemble_long_drafts[(3,)](
        output, neural, retrieved, status, boundaries, sampled, stats, 31, True, 32
    )
    torch.testing.assert_close(output[0, :7], neural[0])
    assert output[0, 7:].tolist() == [0] * 24
    torch.testing.assert_close(output[1, :15], retrieved[1, :15])
    assert output[1, 15:].tolist() == [0] * 16
    torch.testing.assert_close(output[2], retrieved[2])
    assert stats.tolist() == [
        [1, 7, 4, 0, 0],
        [1, 15, 0, 1, 0],
        [1, 31, 31, 0, 1],
        [0] * 5,
    ]


def test_long_context_graphs_do_not_capture_startup_cache_addresses(monkeypatch):
    from vllm.v1.worker.gpu.spec_decode.dflash2.long_ngram import (
        LongNgramDFlash2Speculator,
    )

    calls = []
    monkeypatch.setattr(
        DFlash2Speculator,
        "_precompute_context_kv",
        lambda _self, start, end, dummy_run=False: calls.append((start, end)),
    )
    spec = object.__new__(LongNgramDFlash2Speculator)
    spec._long_context_graphs_ready = False
    spec._long_context_graphs = {}
    spec._precompute_context_kv(0, 8)
    assert calls == [(0, 8)]
    assert not spec._long_context_graphs


@pytest.mark.skipif(not current_platform.is_cuda(), reason="Requires CUDA")
@pytest.mark.parametrize(
    ("known_future", "expected_count"), [(14, 0), (15, 15), (30, 15), (31, 31)]
)
def test_long_ngram_respects_confirmed_future_and_request_slot(
    known_future, expected_count
):
    """Long proposals must neither read unconfirmed tokens nor assume slot zero."""
    from vllm.v1.worker.gpu.spec_decode.dflash2.long_ngram import (
        LongConfirmedNgramLookup,
    )

    key = list(range(100, 108))
    sequence = key + list(range(1000, 1000 + known_future - len(key))) + key
    history = torch.full((2, 128), 9999, dtype=torch.int32, device="cuda")
    history[1, : len(sequence)] = torch.tensor(sequence, device="cuda")
    lengths = torch.tensor([128, len(sequence)], dtype=torch.int32, device="cuda")
    mapping = torch.tensor([1], dtype=torch.int32, device="cuda")
    tokens = torch.zeros(31, dtype=torch.int64, device="cuda")
    lookup = LongConfirmedNgramLookup(128, "cuda")
    score, count = lookup.lookup(history, lengths, mapping, tokens)
    assert count == expected_count
    assert bool(score) == bool(expected_count)
    if count:
        assert tokens[:count].tolist() == sequence[8 : 8 + count]
        assert score // 129 == 8
    assert tokens[count:].tolist() == [-1] * (31 - count)


@pytest.mark.skipif(not current_platform.is_cuda(), reason="Requires CUDA")
@pytest.mark.parametrize(("steps", "max_n"), [(3, 8), (7, 8), (7, 16)])
def test_confirmed_ngram_prefers_longer_then_recent_known_continuations(steps, max_n):
    from vllm.v1.worker.gpu.spec_decode.dflash2.ngram import ConfirmedNgramLookup

    torch.manual_seed(73)
    lookup = ConfirmedNgramLookup(768, steps, "cuda", min_n=3, max_n=max_n)
    cases = [
        [1, 2, 3, *range(10, 10 + steps), 1, 2, 3],
        [
            11,
            1,
            2,
            3,
            *range(20, 20 + steps),
            22,
            1,
            2,
            3,
            *range(30, 30 + steps),
            11,
            1,
            2,
            3,
        ],
        [1, 2, 3, *range(40, 40 + steps), 1, 2, 3, *range(50, 50 + steps), 1, 2, 3],
        list(range(300)),
        [1, 2],
        [0] * 32,
    ]
    for _ in range(12):
        case = torch.randint(0, 19, (530,)).tolist()
        case[-max_n:] = case[254 : 254 + max_n]
        cases.append(case)
    mapping = torch.tensor([2], dtype=torch.int32, device="cuda")
    history = torch.full((3, 768), -999, dtype=torch.int32, device="cuda")
    lengths = torch.zeros(3, dtype=torch.int32, device="cuda")
    tokens = torch.empty(steps, dtype=torch.int64, device="cuda")
    for case in cases:
        history[2, : len(case)] = torch.tensor(case, device="cuda")
        lengths[2] = len(case)
        matches = [
            (n, end)
            for n in range(3, max_n + 1)
            for end in range(n, len(case) - steps + 1)
            if case[end - n : end] == case[-n:]
        ]
        expected = max(matches, default=None)
        score = lookup.lookup(history, lengths, mapping, tokens)
        if expected is None:
            assert score == 0
        else:
            n, end = expected
            assert score == n * 769 + end
            assert tokens.tolist() == case[end : end + steps]


@pytest.mark.skipif(not current_platform.is_cuda(), reason="Requires CUDA")
def test_ngram_point_mass_clears_neural_proposal_and_is_clearable_next_round():
    from vllm.v1.worker.gpu.spec_decode.dflash2.ngram import ConfirmedNgramLookup

    lookup = ConfirmedNgramLookup(32, 3, "cuda")
    logits = torch.full((2, 3, 64), -float("inf"), device="cuda")
    cached = torch.arange(4, device="cuda").repeat(2, 3, 1)
    logits[:, :, :4] = 2.0
    mapping = torch.tensor([1], dtype=torch.int32, device="cuda")
    for ids in ([8, 9, 10], [30, 31, 32]):
        tokens = torch.tensor(ids, device="cuda")
        lookup.cache_proposal(logits, cached, tokens, mapping)
        probs = logits[1].softmax(-1)
        expected = torch.nn.functional.one_hot(tokens, 64).float()
        torch.testing.assert_close(probs, expected, rtol=0, atol=0)
        assert cached[1, :, 0].tolist() == ids
    assert torch.all(logits[0, :, :4] == 2.0)
    from vllm.v1.worker.gpu.spec_decode.dflash2.speculator import CandidateSampler

    sampler = CandidateSampler(2, 3, 4, torch.device("cuda"))
    sampler.cached_candidate_ids = cached
    sampler.scores.zero_()
    neural_ids = torch.arange(40, 44, device="cuda").repeat(1, 3, 1)
    sampler._cache_draft_logits(neural_ids, 3, mapping.repeat(3), logits)
    expected = torch.zeros(3, 64, device="cuda")
    expected[:, 40:44] = 0.25
    torch.testing.assert_close(logits[1].softmax(-1), expected, rtol=0, atol=0)


@pytest.mark.parametrize("block_size", [5, 8])
def test_grouped_conv_matches_reference(block_size: int):
    torch.manual_seed(0)
    batch, taps, num_groups, group_size = 3, 3, 4, 2
    hidden = torch.randn(batch * block_size, num_groups * group_size)
    delta = torch.randn(batch * block_size, taps, num_groups)
    base = torch.randn(taps, num_groups * group_size)

    actual = _grouped_conv(
        hidden, delta, base, block_size, num_groups, group_size, taps
    )
    hidden_blocks = hidden.view(batch, block_size, num_groups, group_size)
    expected = torch.zeros_like(hidden_blocks)
    base = base.view(taps, num_groups, group_size)
    delta = delta.view(batch, block_size, taps, num_groups)
    for position in range(block_size):
        for tap in range(min(taps, position + 1)):
            expected[:, position] += (
                base[tap] + delta[:, position, tap, :, None]
            ) * hidden_blocks[:, position - tap]

    torch.testing.assert_close(actual, expected.flatten(0, 1).flatten(-2))


@pytest.mark.skipif(not current_platform.is_cuda(), reason="This test requires CUDA")
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize(
    "batch,num_groups,group_size,block_size,taps",
    [
        (7, 13, 11, 5, 1),
        (7, 13, 11, 5, 3),
        (7, 13, 11, 8, 2),
        (16, 64, 16, 8, 2),
        (16, 160, 16, 8, 2),
    ],
)
def test_grouped_conv_triton_matches_reference(
    dtype: torch.dtype,
    batch: int,
    num_groups: int,
    group_size: int,
    block_size: int,
    taps: int,
):
    torch.manual_seed(0)
    rows = batch * block_size
    hidden = torch.randn(rows, num_groups * group_size, device="cuda", dtype=dtype)
    base = torch.randn(taps, num_groups * group_size, device="cuda", dtype=dtype)
    projected = torch.randn(rows, 2, taps, num_groups, device="cuda", dtype=dtype)
    delta = projected[:, 1]

    actual = _grouped_conv(
        hidden, delta, base, block_size, num_groups, group_size, taps
    )

    hidden_blocks = hidden.float().view(batch, block_size, num_groups, group_size)
    expected = torch.zeros_like(hidden_blocks)
    base_blocks = base.float().view(taps, num_groups, group_size)
    delta_blocks = delta.float().view(batch, block_size, taps, num_groups)
    for position in range(block_size):
        for tap in range(min(taps, position + 1)):
            expected[:, position] += (
                base_blocks[tap] + delta_blocks[:, position, tap, :, None]
            ) * hidden_blocks[:, position - tap]

    torch.testing.assert_close(
        actual,
        expected.flatten(0, 1).flatten(-2).to(dtype),
        rtol=1e-2 if dtype is torch.bfloat16 else 1e-5,
        atol=1e-2 if dtype is torch.bfloat16 else 1e-5,
    )


def test_draft_quant_exclusions_include_global_layer_indices():
    quant_config = SimpleNamespace(
        exclude_modules=[
            "layers.0.mlp_conv*",
            "*layers.4.self_attn.q_proj",
            "layers.88.already_global",
            "lilicorr.layers.0.mlp.0",
        ]
    )

    _add_global_draft_layer_exclusions(quant_config, 88, 5)

    assert "layers.88.mlp_conv*" in quant_config.exclude_modules
    assert "*layers.92.self_attn.q_proj" in quant_config.exclude_modules
    assert quant_config.exclude_modules.count("layers.88.already_global") == 1
    assert "lilicorr.layers.88.mlp.0" not in quant_config.exclude_modules


def test_selector_edges_match_sequential_reference():
    torch.manual_seed(1)
    batch, steps, top_k, rank = 2, 4, 3, 5
    vocab = 17
    predecessors = torch.randn(vocab, rank)
    successors = torch.randn(vocab, rank)
    candidate_ids = torch.randint(vocab, (batch, steps, top_k))
    unary = torch.randn(batch, steps, top_k)
    hidden = torch.randn(batch, steps, rank)
    anchors = torch.randint(vocab, (batch,))

    actual = _score_edges(
        predecessors,
        successors,
        candidate_ids,
        unary,
        hidden,
        anchors,
        top_k,
    )
    expected = torch.empty_like(actual)
    for step in range(steps):
        pred = (
            anchors[:, None].expand(-1, top_k)
            if step == 0
            else candidate_ids[:, step - 1]
        )
        expected[:, step] = unary[:, step, None] + torch.einsum(
            "bpr,bcr->bpc",
            predecessors[pred] * hidden[:, step, None],
            successors[candidate_ids[:, step]],
        )

    torch.testing.assert_close(actual, expected)


def _stub_base(monkeypatch, draft_logits):
    """A DFlashSpeculator.__init__ that allocates only what the base class would.

    The real base class fills draft_logits from draft_logits_spec, so callers
    pass a tensor already in that state.
    """

    def init_base(self, _vllm_config, device):
        self.draft_model_config = SimpleNamespace(
            hf_config=SimpleNamespace(dflash_config={"selector_top_k": 3})
        )
        self.max_num_reqs = 2
        self.num_query_per_req = 5
        self.num_speculative_steps = 4
        self.vocab_size = 17
        self.draft_tokens = torch.empty((2, 4), dtype=torch.int64, device=device)
        self.draft_logits = draft_logits

    monkeypatch.setattr(DFlashSpeculator, "__init__", init_base)


def test_selector_leaves_greedy_drafting_without_proposal_logits(monkeypatch):
    """Greedy is the default, and it caches no proposal distribution.

    The base class allocates draft_logits only for "probabilistic"; verification
    reads `draft_logits is None` to decide whether a distribution is on offer, so
    allocating one here would claim a proposal the walk never sampled from.
    """
    _stub_base(monkeypatch, None)
    speculator = DFlash2Speculator(None, torch.device("cpu"))

    assert speculator.draft_logits is None


def test_selector_asks_for_fp32_proposal_logits():
    """The spec the base class allocates from: fp32, filled -inf.

    Not the head dtype -- rounding selector scores to bf16 moves the argmax of a
    candidate row often enough that the walk and the rejection sampler checking it
    would no longer read the same distribution.
    """
    dtype, fill = DFlash2Speculator.draft_logits_spec(None, None)

    assert dtype is torch.float32
    assert fill == float("-inf")


@pytest.mark.skip_global_cleanup
@pytest.mark.parametrize("variant", ["dflash2", "lilicorr", "lilicorr_plain"])
def test_candidate_model_decoder_layer_cls(monkeypatch, variant):
    from types import SimpleNamespace

    from vllm.config import set_current_vllm_config
    from vllm.model_executor.models.lilicorr import LiLiCorr
    from vllm.model_executor.models.qwen3_dflash import DFlashQwen3DecoderLayer
    from vllm.model_executor.models.qwen3_dflash2 import (
        DFlash2Qwen3DecoderLayer,
        DFlash2Qwen3Model,
    )

    # 1. Mock get_current_vllm_config and TP groups
    mock_current_vllm_config = SimpleNamespace(
        cache_config=SimpleNamespace(
            block_size=16,
            user_specified_block_size=False,
            kv_cache_dtype_skip_layers=[],
            cache_dtype="auto",
            sliding_window=None,
            enable_prefix_caching=False,
        ),
        kv_transfer_config=None,
        speculative_config=None,
        attention_config=SimpleNamespace(
            use_non_causal=False,
            backend=None,
            backend_per_kind={},
        ),
        parallel_config=SimpleNamespace(
            prefill_context_parallel_size=1,
            decode_context_parallel_size=1,
        ),
        compilation_config=SimpleNamespace(
            compile_custom_ops=False,
            custom_ops="all",
            enabled_custom_ops=set(),
            static_forward_context={},
            mode=0,  # CompilationMode.NONE is 0
        ),
        model_config=SimpleNamespace(
            dtype=torch.float32,
            is_mm_prefix_lm=False,
            rswa_window=None,
        ),
        kernel_config=SimpleNamespace(
            linear_backend="auto",
        ),
    )
    from vllm.platforms import current_platform

    monkeypatch.setattr(
        current_platform,
        "get_attn_backend_cls",
        lambda *args, **kwargs: (
            "vllm.v1.attention.backends.cpu_attn.CPUAttentionBackend"
        ),
    )

    class MockGroup:
        rank_in_group = 0
        world_size = 1

    monkeypatch.setattr(
        "vllm.distributed.parallel_state._TP",
        MockGroup(),
    )

    # 2. Mock vllm_config
    hf_config = SimpleNamespace(
        vocab_size=1000,
        hidden_size=256,
        num_hidden_layers=2,
        num_attention_heads=8,
        num_key_value_heads=2,
        max_position_embeddings=2048,
        rms_norm_eps=1e-6,
        rope_parameters={},
        intermediate_size=512,
        hidden_act="silu",
        dflash_config={
            "selector_rank": 4,
            "selector_top_k": 3,
            "conv_kernel_size": 0 if variant == "lilicorr_plain" else 3,
            "conv_group_size": 0 if variant == "lilicorr_plain" else 2,
            "block_size": 5,
            "lilicorr_candidate_topk": 4,
            "lilicorr_hidden_size": 8,
            "lilicorr_num_layers": 2,
            "lilicorr_num_heads": 2,
            "lilicorr_mlp_ratio": 2.0,
            "lilicorr_factor_dim": 4,
            "lilicorr_vector_eps": 1e-6,
            "lilicorr_logit_scale": 3.0,
            "use_aux_hidden_state": False,
        },
    )
    vllm_config = SimpleNamespace(
        speculative_config=SimpleNamespace(
            draft_model_config=SimpleNamespace(
                hf_config=hf_config,
                quantization=None,
            ),
            num_speculative_tokens=4,
            enable_adaptive_verification=False,
        ),
        model_config=SimpleNamespace(
            dtype=torch.float32,
            is_mm_prefix_lm=False,
        ),
        load_config=SimpleNamespace(
            quantization=None,
            quantization_param_path=None,
        ),
    )
    mock_current_vllm_config.speculative_config = vllm_config.speculative_config
    vllm_config.compilation_config = mock_current_vllm_config.compilation_config

    # 3. Instantiate the model under meta device to avoid parameter allocation issues
    with set_current_vllm_config(mock_current_vllm_config), torch.device("meta"):
        model_cls = DFlash2Qwen3Model if variant == "dflash2" else LiLiCorr
        model = model_cls(vllm_config=vllm_config)

    # 4. Assert that the layers are DFlash2Qwen3DecoderLayer (the subclass)
    assert len(model.layers) == 2
    expected = (
        DFlashQwen3DecoderLayer
        if variant == "lilicorr_plain"
        else DFlash2Qwen3DecoderLayer
    )
    assert type(model.layers[0]) is expected


def test_conv_projections_use_draft_quant_config(monkeypatch):
    from torch import nn

    from vllm.distributed import parallel_state
    from vllm.model_executor.layers.quantization import modelopt
    from vllm.model_executor.models.qwen3_dflash import DFlashQwen3DecoderLayer
    from vllm.model_executor.models.qwen3_dflash2 import DFlash2Qwen3DecoderLayer
    from vllm.model_executor.models.utils import AutoWeightsLoader

    monkeypatch.setattr(
        parallel_state, "_TP", SimpleNamespace(rank_in_group=0, world_size=1)
    )
    monkeypatch.setattr(
        DFlashQwen3DecoderLayer,
        "__init__",
        lambda self, *a, **kw: nn.Module.__init__(self),
    )
    # Exercise the real quantized parameter allocation without selecting a GPU kernel.
    monkeypatch.setattr(
        modelopt,
        "select_linear_kernel",
        lambda *a, **kw: SimpleNamespace(input_quant_key=lambda: None),
    )
    quant_config = modelopt.ModelOptNvFp4Config(
        quant_method="W4A16_NVFP4", is_checkpoint_nvfp4_serialized=True
    )
    layer = DFlash2Qwen3DecoderLayer(
        SimpleNamespace(
            speculative_config=SimpleNamespace(num_speculative_tokens=7),
            model_config=SimpleNamespace(dtype=torch.bfloat16),
        ),
        config=SimpleNamespace(
            hidden_size=16,
            dflash_config={"conv_kernel_size": 2, "conv_group_size": 2},
        ),
        layer_idx=0,
        prefix="model.layers.0",
        quant_config=quant_config,
    )
    for name in ("attention_conv", "mlp_conv"):
        module = getattr(layer, name)
        projection = module.kernel_projection
        assert projection.weight.dtype == torch.uint8
        assert projection.weight.shape == (32, 8)
        weight = torch.ones_like(projection.weight)
        AutoWeightsLoader(module).load_weights([("kernel_projection.weight", weight)])
        torch.testing.assert_close(projection.weight, weight)
        assert module.base_kernel.dtype == torch.bfloat16


def test_context_kv_uses_quantized_projection_fallback(monkeypatch):
    from torch import nn

    from vllm.model_executor.models import qwen3_dflash

    class Projection(nn.Module):
        def __init__(self, packed_weight):
            super().__init__()
            self.register_buffer("packed_weight", packed_weight)
            self.quant_method = object()
            self.calls = 0

        def forward(self, hidden_states):
            self.calls += 1
            return torch.nn.functional.linear(hidden_states, self.packed_weight), None

    monkeypatch.setattr(
        qwen3_dflash.ops,
        "rms_norm",
        lambda output, hidden_states, weight, eps: output.copy_(hidden_states),
    )
    context_states = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    projections = [
        Projection(
            torch.tensor(
                [
                    [0.0, 0.0],
                    [0.0, 0.0],
                    [1.0, 0.0],
                    [0.0, 1.0],
                    [1.0, 1.0],
                    [1.0, -1.0],
                ]
            )
        ),
        Projection(
            torch.tensor(
                [
                    [0.0, 0.0],
                    [0.0, 0.0],
                    [2.0, 0.0],
                    [0.0, 2.0],
                    [-1.0, 0.0],
                    [0.0, -1.0],
                ]
            )
        ),
    ]
    model = SimpleNamespace(
        hidden_norm=SimpleNamespace(weight=nn.Parameter(torch.ones(2))),
        _rms_norm_eps=1e-6,
    )
    layers_attn = [
        SimpleNamespace(
            qkv_proj=projection,
            q_size=2,
            k_norm=SimpleNamespace(weight=nn.Parameter(torch.ones(2))),
        )
        for projection in projections
    ]
    qwen3_dflash.DFlashQwen3Model._build_context_kv_buffers(
        model, layers_attn, has_bias=False
    )
    assert model._fused_kv_weight is None
    assert all(not hasattr(projection, "weight") for projection in projections)

    actual_k, actual_v = qwen3_dflash.DFlashQwen3Model._project_context_kv(
        model,
        context_states,
        num_ctx=2,
        num_layers=2,
        num_kv_heads=1,
        head_dim=2,
    )

    expected_k = torch.tensor(
        [[[1.0, 2.0], [3.0, 4.0]], [[2.0, 4.0], [6.0, 8.0]]]
    ).unsqueeze(2)
    expected_v = torch.tensor(
        [[[3.0, -1.0], [7.0, -1.0]], [[-1.0, -2.0], [-3.0, -4.0]]]
    ).unsqueeze(2)
    torch.testing.assert_close(actual_k, expected_k)
    torch.testing.assert_close(actual_v, expected_v)
    assert [projection.calls for projection in projections] == [1, 1]
