"""Checkpoint-compatible dense DFlash2 backbone for the audited cached recipe.

The target features, RoPE, GR mean convention, and candidate selector follow
the existing checkpoint. No target parameters or verification rule are changed.
"""

import json
from pathlib import Path

import torch
from safetensors.torch import load_file
from torch import nn
from torch.nn import functional

from .sdpa import sdpa


class Norm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        y = x.float()
        return (
            y
            * torch.rsqrt(y.square().mean(-1, keepdim=True) + self.eps)
            * self.weight.to(x.dtype).float()
        ).to(x.dtype)


class Conv(nn.Module):
    def __init__(self, d, group=16, taps=2):
        super().__init__()
        self.group = group
        self.taps = taps
        self.ng = d // group
        self.base_kernel = nn.Parameter(torch.zeros(2, taps, d))
        self.kernel_projection = nn.Linear(d, 2 * taps * self.ng, bias=False)

    def convolve(self, x, delta, side):
        out = torch.zeros_like(x, dtype=torch.float32)
        for tap in range(self.taps):
            coef = self.base_kernel[side, tap].to(x.dtype).float() + delta[
                :, :, tap
            ].float().repeat_interleave(self.group, -1)
            if tap == 0:
                out = out + x.float() * coef
            else:
                out[:, tap:] = out[:, tap:] + x[:, :-tap].float() * coef[:, tap:]
        return out.to(x.dtype)

    def prepare(self, x):
        c = self.kernel_projection(x).reshape(*x.shape[:2], 2, self.taps, self.ng)
        return self.convolve(x, c[:, :, 0], 0), c[:, :, 1]

    def finish(self, x, c):
        return self.convolve(x, c, 1)


class MLP(nn.Module):
    def __init__(self, d, m):
        super().__init__()
        self.gate_proj = nn.Linear(d, m, bias=False)
        self.up_proj = nn.Linear(d, m, bias=False)
        self.down_proj = nn.Linear(m, d, bias=False)

    def forward(self, x):
        return self.down_proj(
            (functional.silu(self.gate_proj(x).float()) * self.up_proj(x).float()).to(
                x.dtype
            )
        )


def rope(x, pos, theta=10000):
    d = x.shape[-1]
    inv = theta ** (-torch.arange(0, d, 2, device=x.device, dtype=torch.float32) / d)
    f = pos.float()[..., None] * inv
    f = torch.cat([f, f], -1)[:, :, None, :]
    a, b = x.chunk(2, -1)
    rot = torch.cat([-b, a], -1)
    return (
        x.float() * f.cos().to(x.dtype).float()
        + rot.float() * f.sin().to(x.dtype).float()
    ).to(x.dtype)


class Attention(nn.Module):
    def __init__(self, c):
        super().__init__()
        d = c["hidden_size"]
        self.nq = c["num_attention_heads"]
        self.nk = c["num_key_value_heads"]
        self.hd = c["head_dim"]
        self.q_proj = nn.Linear(d, self.nq * self.hd, bias=False)
        self.k_proj = nn.Linear(d, self.nk * self.hd, bias=False)
        self.v_proj = nn.Linear(d, self.nk * self.hd, bias=False)
        self.o_proj = nn.Linear(self.nq * self.hd, d, bias=False)
        self.q_norm = Norm(self.hd)
        self.k_norm = Norm(self.hd)

    def forward(self, x, ctx, anchors, window):
        from .attention import forward  # noqa: PLC0415

        return forward(self, x, ctx, anchors, window)

    def reference_forward(self, x, ctx, anchors, window):
        b, s, _ = x.shape
        nctx, length, _ = ctx.shape
        if b % nctx != 0:
            raise ValueError("Each context needs the same anchor count")
        per_context = b // nctx
        qpos = anchors[:, None] + torch.arange(s, device=x.device)
        cpos = torch.arange(length, device=x.device)[None, :]
        q = rope(self.q_norm(self.q_proj(x).reshape(b, s, self.nq, self.hd)), qpos)
        kc = rope(
            self.k_norm(self.k_proj(ctx).reshape(nctx, length, self.nk, self.hd)), cpos
        )
        kc = (
            kc[:, None]
            .expand(-1, per_context, -1, -1, -1)
            .reshape(b, length, self.nk, self.hd)
        )
        kn = rope(self.k_norm(self.k_proj(x).reshape(b, s, self.nk, self.hd)), qpos)
        vc = self.v_proj(ctx).reshape(nctx, length, self.nk, self.hd)
        vc = (
            vc[:, None]
            .expand(-1, per_context, -1, -1, -1)
            .reshape(b, length, self.nk, self.hd)
        )
        vn = self.v_proj(x).reshape(b, s, self.nk, self.hd)
        k = torch.cat([kc, kn], 1)
        v = torch.cat([vc, vn], 1)
        mask = (cpos[:, None, :] < anchors[:, None, None]).expand(b, s, length)
        if window:
            mask = mask & (cpos[:, None, :] >= qpos[:, :, None] - window + 1)
        mask = torch.cat(
            [mask, torch.ones(b, s, s, device=x.device, dtype=torch.bool)], -1
        )[:, None]
        y = sdpa(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), attn_mask=mask
        )
        return self.o_proj(y.transpose(1, 2).reshape(b, s, -1))


class Layer(nn.Module):
    def __init__(self, c):
        super().__init__()
        d = c["hidden_size"]
        dc = c["dflash_config"]
        self.input_layernorm = Norm(d)
        self.post_attention_layernorm = Norm(d)
        self.self_attn = Attention(c)
        self.mlp = MLP(d, c["intermediate_size"])
        self.attention_conv = Conv(d, dc["conv_group_size"], dc["conv_kernel_size"])
        self.mlp_conv = Conv(d, dc["conv_group_size"], dc["conv_kernel_size"])
        self.window = c.get("sliding_window")

    def forward(self, x, ctx, anchors):
        z, k = self.attention_conv.prepare(self.input_layernorm(x))
        x = x + self.attention_conv.finish(
            self.self_attn(z, ctx, anchors, self.window), k
        )
        z, k = self.mlp_conv.prepare(self.post_attention_layernorm(x))
        return x + self.mlp_conv.finish(self.mlp(z), k)


class Selector(nn.Module):
    def __init__(self, c):
        super().__init__()
        d = c["hidden_size"]
        v = c["vocab_size"]
        r = c["dflash_config"]["selector_rank"]
        self.predecessor_codebook = nn.Parameter(torch.empty(v, r))
        self.successor_codebook = nn.Parameter(torch.empty(v, r))
        self.hidden_projection = nn.Linear(d, r, bias=False)

    def scores(self, h, unary, ids, prev):
        gate = self.predecessor_codebook[prev] * self.hidden_projection(h)
        return unary.gather(-1, ids) + (
            gate[..., None, :] * self.successor_codebook[ids]
        ).sum(-1)

    def path(self, h, logits, anchors):
        ids = logits.topk(16, -1).indices
        prev = anchors
        selected = []
        for j in range(logits.shape[1]):
            score = self.scores(h[:, j], logits[:, j], ids[:, j], prev)
            prev = ids[:, j].gather(-1, score.argmax(-1, keepdim=True)).squeeze(-1)
            selected.append(prev)
        return torch.stack(selected, 1)


class Draft(nn.Module):
    def __init__(self, c, shared):
        super().__init__()
        rope_config = c.get("rope_parameters", {})
        if (
            c.get("attention_bias", False)
            or c.get("hidden_act", "silu") != "silu"
            or c.get("rms_norm_eps", 1e-6) != 1e-6  # noqa: PLR2004
            or rope_config.get("rope_theta", c.get("rope_theta", 10000)) != 10000  # noqa: PLR2004
            or rope_config.get("rope_type", "default") != "default"
            or c.get("rope_scaling")
            or c["dflash_config"].get("selector_top_k", 16) != 16  # noqa: PLR2004
        ):
            raise ValueError(
                "Unsupported checkpoint math; use the audited DFlash2 config"
            )
        expected = (c["vocab_size"], c["hidden_size"])
        if any(
            shared[key].shape != expected
            for key in ("embed_tokens.weight", "lm_head.weight")
        ):
            raise ValueError(
                "Shared embedding/head shape differs from draft checkpoint"
            )
        self.config = c
        d = c["hidden_size"]
        dc = c["dflash_config"]
        self.fc = nn.Linear(len(dc["target_layer_ids"]) * d, d, bias=False)
        self.hidden_norm = Norm(d)
        self.norm = Norm(d)
        self.layers = nn.ModuleList([Layer(c) for _ in range(c["num_hidden_layers"])])
        self.candidate_selector = Selector(c)
        self.embed_tokens = nn.Embedding.from_pretrained(
            shared["embed_tokens.weight"], freeze=True
        )
        self.lm_head = nn.Linear(d, c["vocab_size"], bias=False)
        self.lm_head.weight = nn.Parameter(
            shared["lm_head.weight"], requires_grad=False
        )

    @classmethod
    def load(cls, base, shared_path, device="cuda"):
        c = json.loads((Path(base) / "config.json").read_text())
        if c.get("dmtp") or c["dflash_config"].get("moe"):
            raise ValueError("Expected an unfused dense initial checkpoint")
        shared = torch.load(shared_path, map_location="cpu", weights_only=True)
        model = cls(c, shared).to(dtype=torch.bfloat16)
        state = load_file(str(Path(base) / "model.safetensors"))
        missing, unexpected = model.load_state_dict(state, strict=False)
        if set(missing) != {"embed_tokens.weight", "lm_head.weight"} or unexpected:
            raise ValueError((missing, unexpected))
        return model.to(device=device, dtype=torch.bfloat16)

    def backbone(self, features, tokens, anchors):
        if features.ndim == 2:  # noqa: PLR2004
            features = features[None]
            tokens = tokens[None]
            anchors = anchors[None]
        ctx = self.hidden_norm(self.fc(features))
        flat_anchors = anchors.reshape(-1)
        ids = torch.full(
            (len(flat_anchors), 8),
            self.config["dflash_config"]["mask_token_id"],
            device=tokens.device,
            dtype=torch.long,
        )
        ids[:, 0] = tokens.gather(1, anchors).reshape(-1)
        x = self.embed_tokens(ids)
        for layer in self.layers:
            x = layer(x, ctx, flat_anchors)
        return self.norm(x)[:, 1:]

    def forward(self, features, tokens, anchors):
        h = self.backbone(features, tokens, anchors)
        return h, self.lm_head(h)

    def export(self, path):
        state = {
            k: v.detach().to(device="cpu", dtype=torch.bfloat16)
            for k, v in self.state_dict().items()
            if not k.startswith(("embed_tokens.", "lm_head."))
        }
        torch.save(state, path)
