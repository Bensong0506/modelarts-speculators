# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
from torch import nn


class FusionNorm(nn.Module):
    """Match the trainer's float32 RMS/weight accumulation before casting."""

    def __init__(self, width, dtype):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width, dtype=dtype), requires_grad=False)

    def forward(self, x):
        y = x.float()
        return (
            y
            * torch.rsqrt(y.square().mean(-1, keepdim=True) + 1e-6)
            * self.weight.to(x.dtype).float()
        ).to(x.dtype)


class AnchorFusion(nn.Module):
    def __init__(self, width, dtype, residual=False):
        super().__init__()
        self.residual = residual
        self.embedding_norm = FusionNorm(width, dtype)
        self.mtp_norm = FusionNorm(width, dtype)
        self.projection = nn.Linear(2 * width, width, bias=False, dtype=dtype)
        self.requires_grad_(False)

    def forward(self, embedding, mtp_hidden):
        fused = self.projection(
            torch.cat(
                [self.embedding_norm(embedding), self.mtp_norm(mtp_hidden)], dim=-1
            )
        )
        return embedding + fused if self.residual else fused
