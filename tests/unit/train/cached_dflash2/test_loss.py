# Standalone unittest runner also works without the full repository test environment.
# ruff: noqa: PT009
"""Guard loss gradients/labels, true prefix denominators and epoch coverage."""

import unittest

import torch
from torch.nn import functional

from speculators.train.cached_dflash2.common import (
    epoch_plan,
    linear_lr,
    prefix_summary,
)
from speculators.train.cached_dflash2.loss import frozen_head_loss, head_candidates


class RecipeTests(unittest.TestCase):
    def test_chunked_losses_match_full_reference_including_backward(self):
        # The training image defaults to TF32. Use IEEE FP32 for the tight
        # gradient reference; BF16 training arithmetic is checked separately.
        torch.backends.cuda.matmul.allow_tf32 = False
        device = "cuda" if torch.cuda.is_available() else "cpu"
        for dtype in (torch.float32, torch.bfloat16):
            for objective in ("kl", "ce"):
                with self.subTest(dtype=dtype, objective=objective):
                    torch.manual_seed(280)
                    h = (
                        torch.randn(3, 7, 24, device=device, dtype=dtype) / 3
                    ).requires_grad_()
                    target = torch.randn_like(h, requires_grad=True)
                    head = torch.randn(73, 24, device=device, dtype=dtype) / 4
                    weights = torch.exp(-torch.arange(7, device=device) / 4)
                    weights /= weights.mean()
                    actual, candidates, labels = frozen_head_loss(
                        h, target, head, weights, objective, chunk_size=5
                    )
                    q, t = (
                        functional.linear(h, head).float(),
                        functional.linear(target.detach(), head).float(),
                    )
                    gold = t.argmax(-1)
                    ref = (
                        functional.kl_div(
                            q.log_softmax(-1), t.softmax(-1), reduction="none"
                        ).sum(-1)
                        if objective == "kl"
                        else functional.cross_entropy(
                            q.reshape(-1, 73), gold.flatten(), reduction="none"
                        ).reshape(3, 7)
                    )
                    ref = (ref * weights).mean()
                    (grad,) = torch.autograd.grad(actual, h, retain_graph=True)
                    (expected,) = torch.autograd.grad(ref, h)
                    torch.testing.assert_close(actual, ref, rtol=1e-5, atol=2e-6)
                    torch.testing.assert_close(
                        grad,
                        expected,
                        rtol=0.02 if dtype == torch.bfloat16 else 1e-5,
                        atol=3e-5 if dtype == torch.bfloat16 else 1e-7,
                    )
                    torch.testing.assert_close(labels, gold)
                    ids, logits, eval_labels = head_candidates(h, target, head, 5)
                    torch.testing.assert_close(ids, candidates)
                    torch.testing.assert_close(logits, q.topk(16, -1).values)
                    torch.testing.assert_close(labels, eval_labels)
                    self.assertIsNone(target.grad)

    def test_ce_uses_teacher_argmax_and_stops_teacher_gradient(self):
        h = torch.tensor([[[0.1, 0.9]]], requires_grad=True)
        teacher = torch.tensor([[[2.0, -1.0]]], requires_grad=True)
        head = torch.eye(2)
        loss, _, labels = frozen_head_loss(h, teacher, head, torch.ones(1), "ce")
        loss.backward()
        self.assertEqual(labels.item(), 0)
        assert h.grad is not None
        self.assertLess(h.grad[0, 0, 0].item(), 0)
        self.assertGreater(h.grad[0, 0, 1].item(), 0)
        self.assertIsNone(teacher.grad)

    def test_epoch_plan_never_drops_or_repeats_images_within_epoch(self):
        records = [{"id": n, "tokens": 10 + n % 40} for n in range(521)]
        a = epoch_plan(records, 5, 32, 600, 280)
        self.assertEqual(a, epoch_plan(records, 5, 32, 600, 280))
        for epoch in range(1, 6):
            batches = [b for e, b in a if e == epoch]
            self.assertEqual(
                sorted(r["id"] for b in batches for r in b), list(range(521))
            )
            for b in batches:
                self.assertLessEqual(len(b), 32)
                self.assertLessEqual(max(r["tokens"] for r in b) * len(b), 600)

    def test_prefix_metrics_count_only_unbroken_prefix_and_include_bonus(self):
        matched = torch.tensor(
            [[1, 0, 1, 1, 1, 1, 1], [1, 1, 0, 1, 1, 1, 1], [0, 1, 1, 1, 1, 1, 1]]
        )
        out = prefix_summary(
            matched.cumprod(-1).sum(0).tolist(), 3, matched.sum(0).tolist()
        )
        self.assertEqual(out["accepted_draft_tokens"], 3)
        self.assertEqual(out["accepted_length"], 2)
        self.assertEqual(out["acceptance_rate"], 1 / 7)
        self.assertEqual(out["conditional_acceptance"][:3], [2 / 3, 0.5, 0])
        self.assertIsNone(out["conditional_acceptance"][3])

    def test_schedule_has_twenty_percent_warmup_then_decreases(self):
        values = [linear_lr(s, 100, 20, 3e-5) for s in range(1, 101)]
        self.assertAlmostEqual(values[19], 3e-5)
        self.assertTrue(
            all(a < b for a, b in zip(values[:19], values[1:20], strict=True))
        )
        self.assertTrue(
            all(a > b for a, b in zip(values[20:-1], values[21:], strict=True))
        )


if __name__ == "__main__":
    unittest.main()
