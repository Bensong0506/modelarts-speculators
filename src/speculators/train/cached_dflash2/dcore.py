"""DCore mechanisms adapted to the audited DFlash2 cached continuation.

D-PACE weights follow ernie-research/DCore (MIT), commit
5eda323c6c0553984d326652156ba350de7c7ddc, core/dflash.py. Context fusion keeps
the pretrained feature projection and convolutional backbone: this is a
five-target-layer feature transplant, not the paper's nine-layer DFlare model.
"""

import copy
import json
from pathlib import Path

import torch
from safetensors.torch import load_file
from torch import nn
from torch.nn import functional

from .model import Draft


def continuation_weights(probability, alpha=0.5):
    """Detached cumulative confidence times continuation value (suffix sum)."""
    with torch.no_grad():
        prefix = (alpha + (1 - alpha) * probability).cumprod(-1)
        return prefix.flip(-1).cumsum(-1).flip(-1)


class DCoreHeadLoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden, labels, head, alpha, chunk_size):
        if head.requires_grad or labels.shape != hidden.shape[:-1]:
            raise ValueError("Expected frozen head and aligned transcript labels")
        h = hidden.reshape(-1, hidden.shape[-1])
        labels = labels.reshape(-1)
        ce = torch.empty(len(h), device=h.device, dtype=torch.float32)
        grad = torch.empty_like(h)
        with torch.autocast(h.device.type, enabled=False):
            for start in range(0, len(h), chunk_size):
                end = min(start + chunk_size, len(h))
                logits = functional.linear(h[start:end].to(head.dtype), head).float()
                gold = labels[start:end]
                ce[start:end] = functional.cross_entropy(logits, gold, reduction="none")
        weights = continuation_weights((-ce).exp().reshape(hidden.shape[:-1]), alpha)
        scale = weights.reshape(-1) / weights.sum().clamp_min(1e-6)
        # A second bounded pass applies the global detached scale before casting
        # the logit gradient to BF16, as autograd through BF16 logits would.
        with torch.autocast(h.device.type, enabled=False):
            for start in range(0, len(h), chunk_size):
                end = min(start + chunk_size, len(h))
                logits = functional.linear(h[start:end].to(head.dtype), head).float()
                gold = labels[start:end]
                delta = logits.softmax(-1)
                delta.scatter_add_(
                    1, gold[:, None], -torch.ones_like(gold[:, None], dtype=delta.dtype)
                )
                grad[start:end] = (delta * scale[start:end, None]).to(head.dtype) @ head
        ctx.save_for_backward(grad.reshape_as(hidden))
        return (ce * scale).sum()

    @staticmethod
    def backward(ctx, upstream):
        (gradient,) = ctx.saved_tensors
        return upstream * gradient, None, None, None, None


def dcore_loss(hidden, labels, head, alpha=0.5, chunk_size=128):
    return DCoreHeadLoss.apply(hidden, labels, head, alpha, chunk_size)


class DCoreDraft(Draft):
    @classmethod
    def load_feature(cls, base, shared_path, device, enabled=True):
        model = Draft.load(base, shared_path, device)
        if not enabled:
            return model
        return cls.adapt(model, device)

    @classmethod
    def adapt(cls, model, device):
        enabled = True
        model.__class__ = cls
        count = len(model.config["dflash_config"]["target_layer_ids"])
        model.layer_fusion_weights = nn.Parameter(
            torch.zeros(len(model.layers), count, device=device)
        )
        for layer in model.layers:
            attn = layer.self_attn
            attn.k_proj_target = copy.deepcopy(attn.k_proj)
            attn.v_proj_target = copy.deepcopy(attn.v_proj)
        model.feature_enabled = enabled
        model.config["architectures"] = ["CachedDCoreFeatureDraft"]
        model.config["dcore_feature"] = {
            "enabled": enabled,
            "feature_count": count,
            "pretrained_projection_retained": True,
            "convolution_retained": True,
            "selector_retained_frozen": True,
            "official_nine_layer_recipe": False,
            "serving_verified": False,
        }
        return model

    @classmethod
    def load_export(cls, checkpoint, shared_path, device="cuda"):
        checkpoint = Path(checkpoint)
        config = json.loads((checkpoint / "config.json").read_text())
        if config.get("architectures") != ["CachedDCoreFeatureDraft"]:
            raise ValueError("Expected a cached DCore feature export")
        shared = torch.load(shared_path, map_location="cpu", weights_only=True)
        model = cls.adapt(
            Draft(config, shared).to(device=device, dtype=torch.bfloat16), device
        )
        state = load_file(str(checkpoint / "model.safetensors"))
        missing, extra = model.load_state_dict(state, strict=False)
        if set(missing) != {"embed_tokens.weight", "lm_head.weight"} or extra:
            raise ValueError((missing, extra))
        return model.eval()

    def backbone(self, features, tokens, anchors):
        if not self.feature_enabled:
            return super().backbone(features, tokens, anchors)
        if features.ndim == 2:  # noqa: PLR2004
            features, tokens, anchors = features[None], tokens[None], anchors[None]
        count = self.layer_fusion_weights.shape[-1]
        source = features.reshape(*features.shape[:2], count, -1)
        flat = anchors.reshape(-1)
        ids = torch.full(
            (len(flat), 8),
            self.config["dflash_config"]["mask_token_id"],
            device=tokens.device,
            dtype=torch.long,
        )
        ids[:, 0] = tokens.gather(1, anchors).reshape(-1)
        hidden = self.embed_tokens(ids)
        for index, layer in enumerate(self.layers):
            # Multiplication by one at initialization exactly retains the input to fc.
            mixing = self.layer_fusion_weights[index].softmax(-1) * count
            weighted = (source * mixing.to(source.dtype)[None, None, :, None]).flatten(
                -2
            )
            context = self.hidden_norm(self.fc(weighted))
            hidden = layer(hidden, context, flat)
        return self.norm(hidden)[:, 1:]
