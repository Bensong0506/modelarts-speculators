# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Native MTP T1 followed by six parallel DFlash2 candidates.

MTP advances only with the target-verified prefix, using vLLM's native shifted
inputs and rejection handling. DFlash keeps its own target-context/scratch KV.
The target runner verifies the returned seven candidates in the ordinary way.
"""

import torch
from torch import nn

from vllm.v1.worker.gpu.spec_decode.autoregressive.cudagraph_utils import (
    SpeculatorCudaGraphManager,
)
from vllm.v1.worker.gpu.spec_decode.dflash.speculator import prepare_dflash_inputs
from vllm.v1.worker.gpu.spec_decode.dflash2.speculator import DFlash2Speculator
from vllm.v1.worker.gpu.spec_decode.dmtp.config import make_child_configs
from vllm.v1.worker.gpu.spec_decode.mtp.speculator import MTPSpeculator
from vllm.v1.worker.gpu.spec_decode.speculator import DraftModelSpeculator


class FirstTokenMTP(MTPSpeculator):
    def __init__(self, config, device):
        super().__init__(config, device)
        self.first_hidden = torch.zeros(
            self.max_num_reqs,
            self.draft_model_config.get_hidden_size(),
            device=device,
            dtype=self.dtype,
        )

    def init_cudagraph_manager(self, cudagraph_mode):
        super().init_cudagraph_manager(cudagraph_mode)
        # One native output, but the target verifies seven drafts plus its bonus.
        self.prefill_cudagraph_manager = SpeculatorCudaGraphManager(
            self.vllm_config, self.device, cudagraph_mode, decode_query_len=8
        )

    def _prefill(self, num_reqs, *args, **kwargs):
        self._sample_hidden_rows = num_reqs
        return super()._prefill(num_reqs, *args, **kwargs)

    def _run_model(self, *args, **kwargs):
        sample, feedback = super()._run_model(*args, **kwargs)
        n = self._sample_hidden_rows
        if sample.shape[-1] != self.first_hidden.shape[-1]:
            raise ValueError("MTP fusion requires its single-stream LM-head input")
        self.first_hidden[:n].copy_(sample[self.last_token_indices[:n]])
        return sample, feedback


class DMTPTail(DFlash2Speculator):
    def __init__(self, config, device):
        super().__init__(config, device)
        self.first_tokens = torch.zeros(
            self.max_num_reqs, dtype=torch.int64, device=device
        )

    def _prepare_draft_inputs(self, *args):
        prepare_dflash_inputs(*args)
        batch = args[9]
        n = batch.num_reqs
        q = n * 7
        self.input_buffers.input_ids[:q:7].copy_(self.first_tokens[:n])
        # Keep physical slots contiguous; advance absolute query RoPE only.
        # compress_context_gap adjusts SWA so context masking stays identical.
        self.input_buffers.positions[:q].add_(1).clamp_(max=self.max_model_len - 1)
        self.sample_pos[: n * 6].add_(1)

    def _num_graph_context_tokens(self, num_reqs):
        return min(num_reqs * 8, self.max_num_tokens)


class DMTPSpeculator(DraftModelSpeculator):
    num_query_per_req = 7

    def __init__(self, vllm_config, device):
        native_config, tail_config = make_child_configs(vllm_config)
        super().__init__(vllm_config, device)
        self.native = FirstTokenMTP(native_config, device)
        self.tail = DMTPTail(tail_config, device)
        self.aux_speculative_config = self.tail.speculative_config

    def load_draft_model(self, target_model, target_attn_layer_names):
        raise NotImplementedError("The hybrid owns two separately loaded draft models")

    def load_model(self, target_model):
        self.native.load_model(target_model)
        self.tail.load_model(target_model)
        self.draft_attn_layer_names = (
            self.native.draft_attn_layer_names | self.tail.draft_attn_layer_names
        )
        # Expose both models to memory/offload traversal. EPLB is disabled.
        self.model = nn.ModuleDict(
            {"native": self.native.model, "tail": self.tail.model}
        )
        self.supports_mm_inputs = self.native.supports_mm_inputs

    def set_attn(self, *args, **kwargs):
        self.native.set_attn(*args, **kwargs)
        self.tail.set_attn(*args, **kwargs)

    def init_cudagraph_manager(self, cudagraph_mode):
        self.native.init_cudagraph_manager(cudagraph_mode)
        self.tail.init_cudagraph_manager(cudagraph_mode)

    def capture(self):
        self.native.capture()
        self.tail.capture()

    @torch.inference_mode()
    def propose(
        self,
        input_batch,
        attn_metadata,
        slot_mappings,
        last_hidden_states,
        aux_hidden_states,
        num_sampled,
        num_rejected,
        last_sampled,
        next_prefill_tokens,
        temperature,
        seeds,
        dp_sync=None,
        dummy_run=False,
        skip_attn_for_dummy_run=False,
        mm_inputs=None,
        is_profile=False,
    ):
        if not aux_hidden_states:
            raise ValueError("D-MTP needs the configured DFlash target layer features")
        common = dict(
            input_batch=input_batch,
            attn_metadata=attn_metadata,
            slot_mappings=slot_mappings,
            last_hidden_states=last_hidden_states,
            num_sampled=num_sampled,
            num_rejected=num_rejected,
            last_sampled=last_sampled,
            next_prefill_tokens=next_prefill_tokens,
            temperature=temperature,
            seeds=seeds,
            dp_sync=dp_sync,
            dummy_run=dummy_run,
            skip_attn_for_dummy_run=skip_attn_for_dummy_run,
            is_profile=is_profile,
        )
        first = self.native.propose(
            **common, aux_hidden_states=None, mm_inputs=mm_inputs
        )
        n = input_batch.num_reqs
        self.tail.first_tokens[:n].copy_(first[:, 0])
        self.tail.model.model.mtp_sample_hidden[:n].copy_(self.native.first_hidden[:n])
        suffix = self.tail.propose(**common, aux_hidden_states=aux_hidden_states)
        self.draft_tokens[:n, :1].copy_(first)
        self.draft_tokens[:n, 1:].copy_(suffix)
        return self.draft_tokens[:n]
