"""D-MTP training components; native MTP and its cache are supplied separately.

Contract: anchor a is the emitted, pending token A. Target context ends at a-1.
mtp_hidden[a] is the single-stream LM-head input that predicts token a+1,
computed from A and the target HC state at a-1 with native causal MTP history.
It must never be the MTP state after consuming token a+1.
"""

import json
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file
from torch import nn

from .model import Draft, Layer, Norm, rope
from .sdpa import sdpa


def block_attention(module, x, context, query_anchors, context_ends, window):
    """Separate query RoPE positions from the exclusive target-context boundary."""
    blocks, slots, _ = x.shape
    contexts, length, _ = context.shape
    if blocks % contexts != 0:
        raise ValueError("Each context needs the same anchor count")
    per = blocks // contexts
    qpos = query_anchors[:, None] + torch.arange(slots, device=x.device)
    cpos = torch.arange(length, device=x.device)[None, :]
    q = rope(
        module.q_norm(module.q_proj(x).reshape(blocks, slots, module.nq, module.hd)),
        qpos,
    )
    kc = rope(
        module.k_norm(
            module.k_proj(context).reshape(contexts, length, module.nk, module.hd)
        ),
        cpos,
    )
    kn = rope(
        module.k_norm(module.k_proj(x).reshape(blocks, slots, module.nk, module.hd)),
        qpos,
    )
    vc = module.v_proj(context).reshape(contexts, length, module.nk, module.hd)
    vn = module.v_proj(x).reshape(blocks, slots, module.nk, module.hd)
    q = q.reshape(contexts, per * slots, module.nq, module.hd)
    k = torch.cat([kc, kn.reshape(contexts, per * slots, module.nk, module.hd)], 1)
    v = torch.cat([vc, vn.reshape(contexts, per * slots, module.nk, module.hd)], 1)
    limits = context_ends.reshape(contexts, per).repeat_interleave(slots, 1)
    mask = cpos[:, None, :] < limits[:, :, None]
    if window:
        mask &= cpos[:, None, :] >= qpos.reshape(contexts, per * slots, 1) - window + 1
    ids = torch.arange(per * slots, device=x.device) // slots
    noise = (ids[:, None] == ids[None, :])[None].expand(contexts, -1, -1)
    mask = torch.cat([mask, noise], -1)[:, None]
    y = sdpa(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), attn_mask=mask)
    return module.o_proj(y.transpose(1, 2).reshape(blocks, slots, -1))


class DMTPLayer(Layer):
    def forward(self, x, ctx, anchors, context_ends=None):
        if context_ends is None:
            raise ValueError("D-MTP requires an explicit context boundary")
        query_anchors = anchors
        z, kernel = self.attention_conv.prepare(self.input_layernorm(x))
        attended = block_attention(
            self.self_attn, z, ctx, query_anchors, context_ends, self.window
        )
        x = x + self.attention_conv.finish(attended, kernel)
        z, kernel = self.mlp_conv.prepare(self.post_attention_layernorm(x))
        return x + self.mlp_conv.finish(self.mlp(z), kernel)


class AnchorFusion(nn.Module):
    def __init__(self, width, residual=False):
        super().__init__()
        self.residual = residual
        self.embedding_norm = Norm(width)
        self.mtp_norm = Norm(width)
        self.projection = nn.Linear(2 * width, width, bias=False)
        if residual:
            nn.init.zeros_(self.projection.weight)

    def forward(self, embedding, mtp_hidden):
        if embedding.shape != mtp_hidden.shape:
            raise ValueError(
                "Fusion needs the single-stream MTP LM-head input, not HC4 feedback"
            )
        fused = self.projection(
            torch.cat([self.embedding_norm(embedding), self.mtp_norm(mtp_hidden)], -1)
        )
        return embedding + fused if self.residual else fused


class DMTPDraft(Draft):
    def __init__(self, config, shared, mode="fusion", fusion_residual=False):
        if mode not in ("fusion", "token"):
            raise ValueError(mode)
        super().__init__(config, shared)
        self.mode = mode
        self.fusion_residual = fusion_residual
        self.layers = nn.ModuleList([DMTPLayer(config) for _ in self.layers])
        if mode == "fusion":
            self.anchor_fusion = AnchorFusion(config["hidden_size"], fusion_residual)

    @classmethod
    def from_dense(
        cls, base, shared_path, device="cuda", mode="fusion", fusion_residual=False
    ):
        base = Path(base)
        config = json.loads((base / "config.json").read_text())
        if config["dflash_config"].get("moe") or config.get("dmtp"):
            raise ValueError(
                "Continuation must start from the matched dense checkpoint"
            )
        shared = torch.load(shared_path, map_location="cpu", weights_only=True)
        model = cls(config, shared, mode, fusion_residual).to(dtype=torch.bfloat16)
        missing, unexpected = model.load_state_dict(
            load_file(str(base / "model.safetensors")), strict=False
        )
        expected = {"embed_tokens.weight", "lm_head.weight"}
        expected.update(k for k in model.state_dict() if k.startswith("anchor_fusion."))
        if set(missing) != expected or unexpected:
            raise ValueError((missing, unexpected))
        return model.to(device)

    def backbone(self, features, tokens, anchors, mtp_hidden=None, first_tokens=None):
        if features.ndim == 2:  # noqa: PLR2004
            features, tokens, anchors = features[None], tokens[None], anchors[None]
            if mtp_hidden is not None:
                mtp_hidden = mtp_hidden.reshape(-1, self.config["hidden_size"])
        a = anchors.reshape(-1)
        if first_tokens is None:
            raise ValueError(
                "Explicit first_tokens required: gold for training, "
                "native MTP for evaluation"
            )
        first = first_tokens.reshape(-1)
        if first.shape != a.shape:
            raise ValueError("One first candidate per block required")
        ids = torch.full(
            (a.numel(), 7),
            self.config["dflash_config"]["mask_token_id"],
            device=tokens.device,
            dtype=torch.long,
        )
        ids[:, 0] = first
        x = self.embed_tokens(ids)
        if self.mode == "fusion":
            if mtp_hidden is None:
                raise ValueError("Native MTP sample hidden is required")
            x = torch.cat(
                [
                    self.anchor_fusion(x[:, 0], mtp_hidden.reshape(a.numel(), -1))[
                        :, None
                    ],
                    x[:, 1:],
                ],
                1,
            )
        ctx = self.hidden_norm(self.fc(features))
        for layer in self.layers:
            x = layer(x, ctx, anchors=a + 1, context_ends=a)
        return self.norm(x)[:, 1:]

    def forward(self, features, tokens, anchors, mtp_hidden=None, first_tokens=None):
        h = self.backbone(features, tokens, anchors, mtp_hidden, first_tokens)
        return h, self.lm_head(h)

    @torch.no_grad()
    def propose(self, features, tokens, anchors, mtp_hidden, first_tokens):
        h, logits = self(features, tokens, anchors, mtp_hidden, first_tokens)
        suffix = self.candidate_selector.path(h, logits, first_tokens.reshape(-1))
        return torch.cat([first_tokens.reshape(-1, 1), suffix], 1)

    def export_hf(self, dest):
        """Deliberately identify the hybrid architecture; vanilla vLLM cannot run it."""
        dest = Path(dest)
        dest.mkdir(parents=True, exist_ok=False)
        state = {
            k: v.detach().cpu().to(torch.bfloat16).contiguous()
            for k, v in self.state_dict().items()
            if not k.startswith(("embed_tokens.", "lm_head."))
        }
        save_file(state, str(dest / "model.safetensors"), metadata={"format": "pt"})
        config = json.loads(json.dumps(self.config))
        config["architectures"] = ["DMTPDFlash2DraftModel"]
        config["dflash_config"]["block_size"] = 7
        config["dmtp"] = {
            "schema_version": 1,
            "mode": self.mode,
            "draft_input_slots": 7,
            "fusion_residual": self.fusion_residual,
            "proposed_tokens": 7,
            "dflash_predicted_tokens": 6,
            "mtp_hidden": "single_stream_before_first_candidate",
            "target_context": "strictly_before_pending_A",
            "requires_hybrid_vllm_proposer": True,
            "vllm_runtime_verified": False,
        }
        (dest / "config.json").write_text(json.dumps(config, indent=2) + "\n")
