"""Bounded-memory DSpine CE/L1, candidate refinement and layer alignment.

Follows SpecForge 2dadea10's objective terms. This cached continuation normalizes
by weighted proposal tokens, matching its existing trainer, rather than scaling
the final objective by sequences as the upstream online trainer does.
"""

import torch
from torch.nn import functional
from torch.utils.checkpoint import checkpoint


class SpineHeadLoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden, teacher, labels, head, weights, chunk_size):
        if (
            head.requires_grad
            or hidden.shape != teacher.shape
            or labels.shape != hidden.shape[:-1]
        ):
            raise ValueError("Expected frozen head and aligned teacher/labels")
        shape = hidden.shape[:-1]
        h, t = hidden.flatten(0, -2), teacher.flatten(0, -2)
        gold = labels.reshape(-1)
        scale = weights.expand(shape).reshape(-1)
        scale = scale / scale.sum()
        grad = torch.empty_like(h)
        count = min(16, head.shape[0])
        candidates = torch.empty(len(h), count, dtype=torch.long, device=h.device)
        teacher_scores = torch.empty(len(h), count, device=h.device)
        mass = torch.empty(len(h), device=h.device)
        ce_total = torch.zeros((), device=h.device)
        l1_total = torch.zeros_like(ce_total)
        with torch.autocast(h.device.type, enabled=False):
            for start in range(0, len(h), chunk_size):
                end = min(start + chunk_size, len(h))
                logits = functional.linear(h[start:end].to(head.dtype), head).float()
                target = functional.linear(t[start:end].to(head.dtype), head).float()
                q, p = logits.softmax(-1), target.softmax(-1)
                ids = logits.topk(count, -1).indices
                candidates[start:end] = ids
                teacher_scores[start:end] = target.gather(-1, ids)
                mass[start:end] = p.gather(-1, ids).sum(-1)
                ce = functional.cross_entropy(logits, gold[start:end], reduction="none")
                l1 = (q - p).abs().sum(-1)
                sign = (q - p).sign()
                delta = 0.9 * q * (sign - (sign * q).sum(-1, keepdim=True)) + 0.1 * q
                delta.scatter_add_(
                    1,
                    gold[start:end, None],
                    -0.1 * torch.ones_like(gold[start:end, None], dtype=delta.dtype),
                )
                grad[start:end] = (delta * scale[start:end, None]).to(head.dtype) @ head
                ce_total += (ce * scale[start:end]).sum()
                l1_total += (l1 * scale[start:end]).sum()
        ctx.save_for_backward(grad.reshape_as(hidden))
        candidates = candidates.reshape(*shape, count)
        teacher_scores = teacher_scores.reshape(*shape, count)
        mass = mass.reshape(shape)
        ctx.mark_non_differentiable(
            ce_total, l1_total, candidates, teacher_scores, mass
        )
        return (
            0.1 * ce_total + 0.9 * l1_total,
            ce_total,
            l1_total,
            candidates,
            teacher_scores,
            mass,
        )

    @staticmethod
    def backward(ctx, upstream, *_unused):
        (grad,) = ctx.saved_tensors
        return upstream * grad, None, None, None, None, None


def dspine_loss(model, hidden, teacher, labels, predecessors, weights, chunk_size=128):
    base, ce, l1, ids, teacher_scores, mass = SpineHeadLoss.apply(
        hidden, teacher, labels, model.lm_head.weight, weights, chunk_size
    )
    if not model.feature_enabled:
        return base, {"backbone_ce": ce, "backbone_l1": l1}
    covered = (ids == labels[..., None]).any(-1)
    eligible = mass > model.dspine_config.candidate_mass_threshold
    weighted = weights.expand_as(labels)
    ce_den = (weighted * covered).sum().clamp_min(1e-30)
    l1_den = (weighted * eligible).sum().clamp_min(1e-30)
    pre = model.pre_injection.flatten(0, -2)
    flat_ids = ids.flatten(0, -2)
    flat_teacher = teacher_scores.flatten(0, -2)
    flat_prev = predecessors.reshape(-1)
    flat_labels = labels.reshape(-1)
    flat_weights = weighted.reshape(-1)
    flat_eligible = eligible.reshape(-1)

    def refinement_chunk(h, candidate, target, previous, gold, w, valid):
        refined = model.refine(h, previous)
        scores = (
            (refined[:, None] * functional.embedding(candidate, model.lm_head.weight))
            .sum(-1)
            .float()
        )
        matches = candidate == gold[:, None]
        candidate_ce = functional.cross_entropy(
            scores, matches.long().argmax(-1), reduction="none"
        )
        candidate_l1 = (scores.softmax(-1) - target.softmax(-1)).abs().sum(-1)
        return (candidate_ce * matches.any(-1) * w).sum(), (
            candidate_l1 * valid * w
        ).sum()

    refine_ce = base.new_zeros(())
    refine_l1 = base.new_zeros(())
    for start in range(0, len(pre), chunk_size):
        sl = slice(start, start + chunk_size)
        values = (
            pre[sl],
            flat_ids[sl],
            flat_teacher[sl],
            flat_prev[sl],
            flat_labels[sl],
            flat_weights[sl],
            flat_eligible[sl],
        )
        a, b = checkpoint(refinement_chunk, *values, use_reentrant=False)
        refine_ce = refine_ce + a
        refine_l1 = refine_l1 + b
    refinement = 0.1 * refine_ce / ce_den + 0.9 * refine_l1 / l1_den
    reference = functional.embedding(labels, model.transfer_codes).float()
    alignment = 1 - functional.cosine_similarity(
        model.layer_features.float(), reference[:, None], dim=-1
    )
    alignment = (alignment * weighted[:, None]).sum() / (
        weighted.sum() * model.layer_features.shape[1]
    )
    beta, _ = model.dspine_config.schedule(
        model.step, model.total_steps, model.warmup_ratio
    )
    loss = base + refinement + beta * ce.detach() * alignment
    return loss, {
        "backbone_ce": ce,
        "backbone_l1": l1,
        "refinement": refinement.detach(),
        "alignment": alignment.detach(),
        "alignment_beta": beta,
        "candidate_coverage": covered.float().mean(),
    }
