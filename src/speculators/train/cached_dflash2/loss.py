"""Frozen full-vocabulary teacher losses with bounded temporary memory.

CE targets are the teacher argmax at the supplied prefix, never transcript IDs.
Only draft hidden states receive gradients. The head and teacher stay frozen.
"""

import torch
from torch.nn import functional


class FrozenHeadLoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden, teacher, head, weights, chunk_size, objective):
        if objective not in ("kl", "ce") or head.requires_grad:
            raise ValueError("Expected kl/ce and a frozen output head")
        if hidden.shape != teacher.shape or weights.numel() != hidden.shape[-2]:
            raise ValueError("Teacher/slot alignment mismatch")
        shape = hidden.shape[:-1]
        h = hidden.reshape(-1, hidden.shape[-1])
        t = teacher.reshape_as(h)
        scale = weights.expand(shape).reshape(-1) / len(h)
        grad = torch.empty_like(h)
        candidates = torch.empty(
            (len(h), min(16, head.shape[0])), device=h.device, dtype=torch.long
        )
        labels = torch.empty(len(h), device=h.device, dtype=torch.long)
        loss = torch.zeros((), device=h.device, dtype=torch.float32)
        with torch.autocast(h.device.type, enabled=False):
            for start in range(0, len(h), chunk_size):
                end = min(start + chunk_size, len(h))
                q = functional.linear(h[start:end].to(head.dtype), head).float()
                target = functional.linear(t[start:end].to(head.dtype), head).float()
                gold = target.argmax(-1)
                labels[start:end] = gold
                candidates[start:end] = q.topk(candidates.shape[-1], -1).indices
                delta = q.softmax(-1)
                if objective == "kl":
                    p = target.softmax(-1)
                    per_slot = functional.kl_div(
                        q.log_softmax(-1), p, reduction="none"
                    ).sum(-1)
                    delta -= p
                else:
                    per_slot = functional.cross_entropy(q, gold, reduction="none")
                    delta.scatter_add_(
                        1,
                        gold[:, None],
                        -torch.ones_like(gold[:, None], dtype=delta.dtype),
                    )
                loss += (per_slot * scale[start:end]).sum()
                delta = (delta * scale[start:end, None]).to(head.dtype)
                grad[start:end] = delta @ head
        ctx.save_for_backward(grad.reshape_as(hidden))
        ctx.mark_non_differentiable(candidates, labels)
        return loss, candidates.reshape(*shape, -1), labels.reshape(shape)

    @staticmethod
    def backward(ctx, upstream, _unused_candidates, _unused_labels):
        (gradient,) = ctx.saved_tensors
        return gradient * upstream, None, None, None, None, None


def frozen_head_loss(hidden, teacher, head, weights, objective, chunk_size=128):
    return FrozenHeadLoss.apply(hidden, teacher, head, weights, chunk_size, objective)


@torch.no_grad()
def head_candidates(hidden, teacher, head, chunk_size=128):
    """Evaluation uses the same head arithmetic as the training objective."""
    if hidden.shape != teacher.shape:
        raise ValueError("Teacher/slot alignment mismatch")
    shape = hidden.shape[:-1]
    h, t = hidden.reshape(-1, hidden.shape[-1]), teacher.reshape(-1, teacher.shape[-1])
    ids, values, labels = [], [], []
    with torch.autocast(h.device.type, enabled=False):
        for start in range(0, len(h), chunk_size):
            q = functional.linear(
                h[start : start + chunk_size].to(head.dtype), head
            ).float()
            target = functional.linear(
                t[start : start + chunk_size].to(head.dtype), head
            ).float()
            top = q.topk(min(16, head.shape[0]), -1)
            ids.append(top.indices)
            values.append(top.values)
            labels.append(target.argmax(-1))
    return (
        torch.cat(ids).reshape(*shape, -1),
        torch.cat(values).reshape(*shape, -1),
        torch.cat(labels).reshape(shape),
    )
