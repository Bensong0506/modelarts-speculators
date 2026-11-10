# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in configuration for the private, seven-candidate D-MTP experiment."""

import copy


def validate_checkpoint(config):
    expected = {
        "schema_version": 1,
        "draft_input_slots": 7,
        "proposed_tokens": 7,
        "dflash_predicted_tokens": 6,
        "mtp_hidden": "single_stream_before_first_candidate",
        "target_context": "strictly_before_pending_A",
        "requires_hybrid_vllm_proposer": True,
    }
    metadata = getattr(config, "dmtp", {})
    if any(metadata.get(k) != v for k, v in expected.items()):
        raise ValueError("D-MTP checkpoint has an incompatible alignment contract")
    if metadata.get("mode") not in ("token", "fusion"):
        raise ValueError("D-MTP mode must be token or fusion")
    residual = metadata.get("fusion_residual", False)
    if not isinstance(residual, bool) or (residual and metadata["mode"] != "fusion"):
        raise ValueError("fusion_residual must be boolean and requires fusion mode")
    if getattr(config, "hc_mult", 1) != 1:
        raise ValueError("DFlash consumes collapsed target features, not HC residuals")
    if config.dflash_config.get("moe"):
        raise ValueError("This matched D-MTP control requires a dense DFlash2 tail")
    if config.dflash_config.get("block_size", 7) != 7:
        raise ValueError("D-MTP exports must declare seven query slots")


def compress_context_gap(config):
    """Preserve the absolute attention mask while removing the pending-A KV hole.

    Query RoPE advances by one; query cache slots remain next to target context.
    Reducing SWA by one preserves its lower context bound. The seven query
    tokens must still attend to one another without restriction.
    """
    config = copy.deepcopy(config)
    window = config.dflash_config.get(
        "swa_window_size", getattr(config, "sliding_window", None)
    )
    if window:
        if window <= 7:
            raise ValueError("D-MTP needs an SWA window larger than its query block")
        config.dflash_config["swa_window_size"] = window - 1
    return config


def make_child_configs(config):
    from vllm.config import SpeculativeConfig
    from vllm.transformers_utils.config import get_hf_text_config

    spec = config.speculative_config
    options = config.additional_config.get("swx_dmtp")
    if not isinstance(options, dict) or not options.get("model"):
        raise ValueError("additional_config.swx_dmtp.model must name a hybrid export")
    if set(options) != {"model"}:
        raise ValueError("Unknown swx_dmtp option")
    if spec.method != "mtp" or spec.num_speculative_tokens != 7:
        raise ValueError("D-MTP scheduler requires method=mtp and seven candidates")
    if "Qwen4ExpMTP" not in spec.draft_model_config.architectures:
        raise ValueError("This D-MTP adapter supports the Qwen4Exp native MTP only")
    if not config.use_v2_model_runner:
        raise ValueError("D-MTP requires VLLM_USE_V2_MODEL_RUNNER=1")
    if (
        config.scheduler_config.max_num_seqs * 7
        > config.scheduler_config.max_num_batched_tokens
    ):
        raise ValueError("D-MTP requires max_num_batched_tokens >= 7 * max_num_seqs")
    if spec.draft_sample_method != "greedy" or spec.enable_adaptive_verification:
        raise ValueError("D-MTP supports fixed-width greedy drafts only")
    parallel = config.parallel_config
    sizes = (
        parallel.tensor_parallel_size,
        parallel.pipeline_parallel_size,
        parallel.data_parallel_size,
        parallel.decode_context_parallel_size,
        parallel.prefill_context_parallel_size,
    )
    if any(size != 1 for size in sizes) or parallel.enable_eplb:
        raise ValueError("D-MTP requires TP=PP=DP=DCP=PCP=1 and no EPLB")
    if config.lora_config or getattr(config, "watermark_config", None):
        raise ValueError("D-MTP LoRA and watermarking are not implemented")
    if any(config.additional_config.get(k) for k in ("swx_ngram", "swx_long_ngram")):
        raise ValueError("D-MTP cannot be combined with the n-gram experiment")

    native = copy.copy(config)
    native.speculative_config = copy.copy(spec)
    native.speculative_config.num_speculative_tokens = 1
    tail = copy.copy(config)
    tail.speculative_config = SpeculativeConfig(
        target_model_config=config.model_config,
        target_parallel_config=parallel,
        model=options["model"],
        method="dflash",
        num_speculative_tokens=6,
        draft_sample_method="greedy",
    )
    draft = tail.speculative_config.draft_model_config
    if draft.architectures != ["DMTPDFlash2DraftModel"]:
        raise ValueError("swx_dmtp.model must be a DMTPDFlash2DraftModel export")
    validate_checkpoint(draft.hf_config)
    draft.hf_config = compress_context_gap(draft.hf_config)
    draft.hf_text_config = get_hf_text_config(draft.hf_config)
    return native, tail
