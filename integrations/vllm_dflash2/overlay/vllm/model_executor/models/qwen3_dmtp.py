# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Checkpoint-compatible D-MTP tail: [native T1, MASK x 6]."""

import torch

from vllm.model_executor.models.qwen3_dflash2 import (
    DFlash2Qwen3ForCausalLM,
    DFlash2Qwen3Model,
)
from vllm.v1.worker.gpu.spec_decode.dmtp.config import validate_checkpoint
from vllm.v1.worker.gpu.spec_decode.dmtp.fusion import AnchorFusion


class DMTPDFlash2Model(DFlash2Qwen3Model):
    def __init__(self, *, vllm_config, start_layer_id=0, prefix=""):
        super().__init__(
            vllm_config=vllm_config, start_layer_id=start_layer_id, prefix=prefix
        )
        validate_checkpoint(self.config)
        spec = vllm_config.speculative_config
        if spec.num_speculative_tokens != 6 or not vllm_config.additional_config.get(
            "swx_dmtp"
        ):
            raise ValueError("A D-MTP export requires the hybrid proposer")
        self.dmtp_mode = self.config.dmtp["mode"]
        width = self.config.hidden_size
        dtype = vllm_config.model_config.dtype
        self.register_buffer(
            "mtp_sample_hidden",
            torch.zeros(vllm_config.scheduler_config.max_num_seqs, width, dtype=dtype),
            persistent=False,
        )
        if self.dmtp_mode == "fusion":
            self.anchor_fusion = AnchorFusion(
                width, dtype, residual=self.config.dmtp.get("fusion_residual", False)
            )

    def embed_input_ids(self, input_ids):
        embedding = super().embed_input_ids(input_ids)
        if self.dmtp_mode == "token":
            return embedding
        blocks = embedding.reshape(-1, 7, self.config.hidden_size)
        first = self.anchor_fusion(
            blocks[:, 0], self.mtp_sample_hidden[: blocks.shape[0]]
        )
        return torch.cat([first[:, None], blocks[:, 1:]], dim=1).flatten(0, 1)


class DMTPDFlash2ForCausalLM(DFlash2Qwen3ForCausalLM):
    model_cls = DMTPDFlash2Model


EntryClass = DMTPDFlash2ForCausalLM
