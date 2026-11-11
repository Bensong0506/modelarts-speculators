"""DSpine mechanisms on the pretrained cached DFlash2 backbone.

Config, adjacent injection, PCA transfer initialization and candidate transitions
adapted from A-Gentle-Cat/SpecForge, commit 2dadea10e317842af21c79e2097dc05abe1e2b5a
(MIT; see THIRD_PARTY_NOTICES.md). The convolutional backbone, five cached target
layers and seven proposal slots are retained. This is a feature transplant.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import NamedTuple

import torch
from safetensors.torch import load_file
from torch import nn
from torch.nn import functional

from .model import Draft


@dataclass(frozen=True)
class DSpineConfig:
    transfer_rank: int = 1024
    message_size: int = 512
    top_k: int = 16
    ce_weight: float = 0.1
    alignment_weight: float = 0.5
    alignment_warmup_steps: int | None = None
    replacement_layers: int = 3
    replacement_probability: float = 0.5
    replacement_hold_ratio: float = 1 / 6
    replacement_end_ratio: float = 1 / 3
    candidate_mass_threshold: float = 1e-4

    def validate(self, hidden_size: int, vocab_size: int) -> None:  # noqa: C901
        for name in (
            "transfer_rank",
            "message_size",
            "top_k",
            "alignment_warmup_steps",
            "replacement_layers",
        ):
            value = getattr(self, name)
            if name == "alignment_warmup_steps" and value is None:
                continue
            if not isinstance(value, int) or isinstance(value, bool):
                raise ValueError(f"DSpine {name} must be an integer")
        for name, value in asdict(self).items():
            if value is not None and not math.isfinite(value):
                raise ValueError(f"DSpine {name} must be finite")
        if not 0 < self.transfer_rank <= min(hidden_size, vocab_size - 1):
            raise ValueError(
                "DSpine transfer_rank must be in [1, min(hidden_size, vocab_size - 1)]"
            )
        if self.message_size < 1 or not 1 <= self.top_k <= vocab_size:
            raise ValueError(
                "DSpine requires message_size > 0 and top_k in [1, vocab_size]"
            )
        if not 0 <= self.ce_weight <= 1 or self.alignment_weight < 0:
            raise ValueError(
                "DSpine requires ce_weight in [0, 1] and alignment_weight >= 0"
            )
        if (self.alignment_warmup_steps or 0) < 0 or self.replacement_layers < 0:
            raise ValueError(
                "DSpine warmup steps and replacement layers must be nonnegative"
            )
        if not 0 <= self.replacement_probability <= 1:
            raise ValueError("DSpine replacement_probability must be in [0, 1]")
        if not 0 <= self.replacement_hold_ratio < self.replacement_end_ratio <= 1:
            raise ValueError(
                "DSpine requires 0 <= replacement_hold_ratio "
                "< replacement_end_ratio <= 1"
            )
        if not 0 <= self.candidate_mass_threshold < 1:
            raise ValueError("DSpine candidate_mass_threshold must be in [0, 1)")

    def schedule(
        self, step: int, total_steps: int | None, warmup_ratio: float | None = None
    ) -> tuple[float, float]:
        warmup = self.alignment_warmup_steps
        if warmup is None:
            if warmup_ratio is None or not total_steps or total_steps <= 0:
                raise ValueError(
                    "DSpine alignment warmup follows the LR warmup and needs "
                    "warmup_ratio and a positive total_steps"
                )
            warmup = int(warmup_ratio * total_steps)
        beta = self.alignment_weight * (min(max(step, 0) / warmup, 1) if warmup else 1)
        if not total_steps or total_steps <= 0:
            return beta, 0.0
        progress = max(step, 0) / total_steps
        fraction = (self.replacement_end_ratio - progress) / (
            self.replacement_end_ratio - self.replacement_hold_ratio
        )
        return beta, self.replacement_probability * min(max(fraction, 0), 1)


class DSpineOutput(NamedTuple):
    hidden_states: torch.Tensor
    pre_injection: torch.Tensor
    layer_features: torch.Tensor


class AdjacentInjection(nn.Module):
    def __init__(self, hidden_size: int, num_layers: int, config: DSpineConfig):
        super().__init__()
        self.rank = config.transfer_rank
        self.replacement_layers = config.replacement_layers
        self.readouts = nn.ModuleList(
            nn.Linear(hidden_size, self.rank, bias=False) for _ in range(num_layers)
        )
        self.writes = nn.ModuleList(
            nn.Linear(config.message_size, hidden_size, bias=False)
            for _ in range(num_layers)
        )
        self.message = nn.Linear(self.rank, config.message_size, bias=False)
        self.receiver_gate = nn.Linear(self.rank, config.message_size, bias=False)
        self.predecessor_gate = nn.Linear(self.rank, config.message_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(config.message_size))
        for write in self.writes:
            nn.init.zeros_(write.weight)
        nn.init.zeros_(self.predecessor_gate.weight)

    def feature(self, hidden: torch.Tensor, layer: int) -> torch.Tensor:
        projected = self.readouts[layer](hidden)
        return (
            math.sqrt(self.rank) * functional.normalize(projected.float(), dim=-1)
        ).to(hidden.dtype)

    def write(
        self,
        hidden: torch.Tensor,
        receiver: torch.Tensor,
        predecessor: torch.Tensor,
        layer: int,
    ) -> torch.Tensor:
        # FSDP may keep the token-code buffer in FP32 while projections use BF16.
        predecessor = predecessor.to(receiver.dtype)
        gate = torch.sigmoid(
            self.receiver_gate(receiver)
            + self.predecessor_gate(predecessor)
            + self.bias
        )
        delta = self.writes[layer](gate * self.message(predecessor))
        scale = (
            torch.linalg.vector_norm(hidden.float(), dim=-1, keepdim=True)
            / math.sqrt(hidden.shape[-1])
        ).to(hidden.dtype)
        return hidden + scale * delta

    def forward(
        self,
        hidden: torch.Tensor,
        layer: int,
        token_ids: torch.Tensor,
        codes: torch.Tensor,
        replace: torch.Tensor | None = None,
    ) -> torch.Tensor:
        features = self.feature(hidden, layer)
        anchor = math.sqrt(self.rank) * functional.embedding(token_ids[..., :1], codes)
        predecessors = torch.cat((anchor, features[..., 1:-1, :]), dim=-2)
        if replace is not None and layer < self.replacement_layers:
            correct = math.sqrt(self.rank) * functional.embedding(
                token_ids[..., :-1], codes
            )
            predecessors = torch.where(replace[..., None, None], correct, predecessors)
        successors = self.write(
            hidden[..., 1:, :], features[..., 1:, :], predecessors, layer
        )
        return torch.cat((hidden[..., :1, :], successors), dim=-2)


class DSpineDraft(Draft):
    @classmethod
    def load_feature(
        cls, base, shared_path, device, enabled=True, rank=1024, message_size=512
    ):
        model = Draft.load(base, shared_path, device)
        return cls.adapt(model, device, enabled, rank, message_size)

    @classmethod
    def adapt(cls, model, device, enabled, rank, message_size, initialize=True):
        model.__class__ = cls
        model.feature_enabled = enabled
        model.dspine_config = DSpineConfig(
            transfer_rank=rank, message_size=message_size
        )
        model.dspine_config.validate(
            model.config["hidden_size"], model.config["vocab_size"]
        )
        model.injection = AdjacentInjection(
            model.config["hidden_size"], len(model.layers), model.dspine_config
        ).to(device)
        model.register_buffer(
            "transfer_codes",
            torch.zeros(model.config["vocab_size"], rank, device=device),
        )
        model.register_buffer("transfer_ready", torch.tensor(False, device=device))
        for layer in model.layers:
            layer.self_attn.dspine_causal = True
        model.step = 0
        model.total_steps = 1
        model.warmup_ratio = 0.2
        model.config["architectures"] = ["CachedDSpineFeatureDraft"]
        model.config["dspine_feature"] = dict(
            enabled=enabled,
            **asdict(model.dspine_config),
            convolution_retained=True,
            causal_noise=True,
            serving_verified=False,
        )
        model.config["is_causal"] = True
        if initialize:
            model.initialize_transfer_space(model.lm_head.weight)
        return model

    @classmethod
    def load_export(cls, checkpoint, shared_path, device="cuda"):
        checkpoint = Path(checkpoint)
        config = json.loads((checkpoint / "config.json").read_text())
        if config.get("architectures") != ["CachedDSpineFeatureDraft"]:
            raise ValueError("Expected a cached DSpine feature export")
        metadata = config["dspine_feature"]
        shared = torch.load(shared_path, map_location="cpu", weights_only=True)
        model = cls.adapt(
            Draft(config, shared).to(device=device, dtype=torch.bfloat16),
            device,
            metadata["enabled"],
            metadata["transfer_rank"],
            metadata["message_size"],
            initialize=False,
        )
        state = load_file(str(checkpoint / "model.safetensors"))
        missing, extra = model.load_state_dict(state, strict=False)
        if set(missing) != {"embed_tokens.weight", "lm_head.weight"} or extra:
            raise ValueError((missing, extra))
        if not bool(model.transfer_ready):
            raise ValueError("Export has no initialized transfer space")
        return model.eval()

    def backbone(self, features, tokens, anchors):
        if features.ndim == 2:  # noqa: PLR2004
            features, tokens, anchors = features[None], tokens[None], anchors[None]
        context = self.hidden_norm(self.fc(features))
        flat = anchors.reshape(-1)
        row = torch.arange(len(tokens), device=tokens.device)[:, None, None]
        block = tokens[
            row, anchors[:, :, None] + torch.arange(8, device=tokens.device)
        ].reshape(-1, 8)
        noisy = torch.full_like(block, self.config["dflash_config"]["mask_token_id"])
        noisy[:, 0] = block[:, 0]
        hidden = self.embed_tokens(noisy)
        _, probability = self.dspine_config.schedule(
            self.step, self.total_steps, self.warmup_ratio
        )
        replacement = (
            (torch.rand(len(block), device=tokens.device) < probability)
            if self.training and self.feature_enabled
            else None
        )
        layer_features = []
        for index, layer in enumerate(self.layers):
            pre = layer(hidden, context, flat)
            hidden = (
                self.injection(pre, index, block, self.transfer_codes, replacement)
                if self.feature_enabled
                else pre
            )
            if self.feature_enabled:
                layer_features.append(self.injection.feature(hidden[:, 1:], index))
        self.pre_injection = pre[:, 1:]
        self.layer_features = torch.stack(layer_features, 1) if layer_features else None
        return self.norm(hidden)[:, 1:]

    @torch.no_grad()
    def initialize_transfer_space(
        self, head_weight: torch.Tensor, chunk_size: int = 4096
    ) -> None:
        """Initialize fixed whitened token codes and readouts from the target head."""
        if tuple(head_weight.shape) != (
            self.config["vocab_size"],
            self.config["hidden_size"],
        ):
            raise ValueError(
                "DSpine transfer space requires the target head's "
                "full vocabulary and hidden width"
            )
        if chunk_size < 1:
            raise ValueError("chunk_size must be positive")
        width = head_weight.shape[1]
        covariance = torch.zeros(
            width, width, device=head_weight.device, dtype=torch.float32
        )
        total = covariance.new_zeros(width)
        for rows in head_weight.split(chunk_size):
            normalized = functional.normalize(rows.float(), dim=-1)
            total.add_(normalized.sum(dim=0))
            covariance.add_(normalized.T @ normalized)
        mean = total / head_weight.shape[0]
        covariance.sub_(head_weight.shape[0] * torch.outer(mean, mean))
        values, directions = torch.linalg.eigh(covariance)
        values = values[-self.dspine_config.transfer_rank :]
        directions = directions[:, -self.dspine_config.transfer_rank :]
        if not torch.isfinite(values).all() or (values <= 0).any():
            raise ValueError(
                "Target head has insufficient positive principal directions "
                "for transfer_rank"
            )
        for start in range(0, head_weight.shape[0], chunk_size):
            rows = head_weight[start : start + chunk_size]
            codes = functional.normalize(
                (functional.normalize(rows.float(), dim=-1) - mean)
                @ directions
                / values.sqrt(),
                dim=-1,
            )
            self.transfer_codes[start : start + len(rows)].copy_(codes)
        for readout in self.injection.readouts:
            readout.weight.copy_(directions.T)
        self.transfer_ready.fill_(True)

    def refine(
        self, hidden: torch.Tensor, predecessor_ids: torch.Tensor
    ) -> torch.Tensor:
        torch._assert_async(  # noqa: SLF001
            self.transfer_ready,
            "Initialize or load the DSpine transfer space before use",
        )
        layer = len(self.layers) - 1
        receiver = self.injection.feature(hidden, layer)
        predecessor = math.sqrt(self.injection.rank) * functional.embedding(
            predecessor_ids, self.transfer_codes
        )
        return self.norm(self.injection.write(hidden, receiver, predecessor, layer))

    def transition_scores(
        self,
        pre_injection: torch.Tensor,
        candidate_ids: torch.Tensor,
        anchor_ids: torch.Tensor,
        head_weight: torch.Tensor,
    ) -> torch.Tensor:
        """Score all adjacent candidate pairs from cached pre-injection states."""
        count = candidate_ids.shape[-1]
        anchor = anchor_ids[..., None, None].expand(*anchor_ids.shape, 1, count)
        predecessors = torch.cat((anchor, candidate_ids[..., :-1, :]), dim=-2)
        hidden = pre_injection.unsqueeze(-2).expand(
            *pre_injection.shape[:-1], count, pre_injection.shape[-1]
        )
        refined = self.refine(hidden, predecessors)
        weights = functional.embedding(candidate_ids, head_weight)
        return torch.einsum("...ikh,...ijh->...ikj", refined, weights).float()

    @staticmethod
    def select_cached(
        scores: torch.Tensor, candidate_ids: torch.Tensor
    ) -> torch.Tensor:
        index = torch.zeros(
            candidate_ids.shape[:-2], device=candidate_ids.device, dtype=torch.long
        )
        selected = []
        for position in range(candidate_ids.shape[-2]):
            row = (
                scores[..., position, :, :]
                .gather(
                    -2, index[..., None, None].expand(*index.shape, 1, scores.shape[-1])
                )
                .squeeze(-2)
            )
            index = row.argmax(dim=-1)
            selected.append(
                candidate_ids[..., position, :].gather(-1, index[..., None]).squeeze(-1)
            )
        return torch.stack(selected, dim=-1)
