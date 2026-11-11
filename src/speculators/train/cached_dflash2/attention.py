"""Equivalent block attention with shared prefix K/V across one image's anchors."""

import torch

from .model import rope
from .sdpa import sdpa


def forward(module, x, context, anchors, window):
    blocks, slots, _ = x.shape
    contexts, length, _ = context.shape
    per = blocks // contexts
    if blocks % contexts != 0:
        raise ValueError("Each context needs the same anchor count")
    device = x.device
    qpos = anchors[:, None] + torch.arange(slots, device=device)
    cpos = torch.arange(length, device=device)[None, :]
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
    limits = anchors.reshape(contexts, per).repeat_interleave(slots, dim=1)
    mask = cpos[:, None, :] < limits[:, :, None]
    if window:
        mask = mask & (
            cpos[:, None, :] >= qpos.reshape(contexts, per * slots, 1) - window + 1
        )
    block_ids = torch.arange(per * slots, device=device) // slots
    noise = (block_ids[:, None] == block_ids[None, :])[None].expand(contexts, -1, -1)
    if getattr(module, "dspine_causal", False):
        positions = torch.arange(per * slots, device=device)
        noise = noise & (positions[:, None] >= positions[None, :])[None]
    mask = torch.cat([mask, noise], -1)[:, None]
    y = sdpa(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), attn_mask=mask)
    return module.o_proj(y.transpose(1, 2).reshape(blocks, slots, -1))
