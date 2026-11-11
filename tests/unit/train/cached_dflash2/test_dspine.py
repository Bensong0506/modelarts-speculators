# ruff: noqa: PT009
"""Full-vocabulary gradients, gold isolation, causal attention and transition cache."""

import tempfile
import unittest
from pathlib import Path

import torch
from torch.nn import functional

from speculators.train.cached_dflash2.dspine import DSpineDraft
from speculators.train.cached_dflash2.dspine_loss import SpineHeadLoss, dspine_loss

from .test_alignment import make_fixture


class DSpineTests(unittest.TestCase):
    def test_full_vocab_ce_l1_gradient_matches_autograd(self):
        torch.manual_seed(35)
        hidden = torch.randn(3, 7, 12, requires_grad=True)
        teacher = torch.randn_like(hidden, requires_grad=True)
        labels = torch.randint(0, 31, (3, 7))
        head = torch.randn(31, 12) / 3
        weights = torch.exp(-torch.arange(7) / 4)
        actual, _, _, candidates, scores, mass = SpineHeadLoss.apply(
            hidden, teacher, labels, head, weights, 5
        )
        logits = functional.linear(hidden, head)
        target = functional.linear(teacher.detach(), head)
        ce = functional.cross_entropy(
            logits.flatten(0, 1), labels.flatten(), reduction="none"
        ).reshape(3, 7)
        l1 = (logits.softmax(-1) - target.softmax(-1)).abs().sum(-1)
        expected = ((0.1 * ce + 0.9 * l1) * weights).sum() / (3 * weights.sum())
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(
            torch.autograd.grad(actual, hidden)[0],
            torch.autograd.grad(expected, hidden)[0],
        )
        self.assertFalse(scores.requires_grad)
        self.assertFalse(mass.requires_grad)
        self.assertEqual(tuple(candidates.shape), (3, 7, 16))
        self.assertIsNone(teacher.grad)

    def test_bfloat16_head_gradient_matches_reference(self):
        torch.manual_seed(37)
        hidden = (torch.randn(2, 7, 12).bfloat16() / 3).requires_grad_()
        labels = torch.randint(0, 31, (2, 7))
        head = (torch.randn(31, 12) / 3).bfloat16()
        logits = functional.linear(hidden, head).float()
        ce = functional.cross_entropy(
            logits.flatten(0, 1), labels.flatten(), reduction="none"
        ).reshape(2, 7)
        teacher = torch.randn_like(hidden)
        weights = torch.exp(-torch.arange(7) / 4)
        actual = SpineHeadLoss.apply(hidden, teacher, labels, head, weights, 4)[0]
        target = functional.linear(teacher, head).float()
        l1 = (logits.softmax(-1) - target.softmax(-1)).abs().sum(-1)
        expected = ((0.1 * ce + 0.9 * l1) * weights).sum() / (2 * weights.sum())
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(
            torch.autograd.grad(actual, hidden)[0],
            torch.autograd.grad(expected, hidden)[0],
            rtol=0.02,
            atol=3e-5,
        )

    def test_eval_never_consumes_gold_successors_and_transition_cache_matches_loop(
        self,
    ):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "fixture"
            make_fixture(root)
            model = (
                DSpineDraft.load_feature(
                    root / "dense", root / "shared.pt", "cpu", rank=8, message_size=4
                )
                .float()
                .eval()
            )
            with torch.no_grad():
                for write in model.injection.writes:
                    write.weight.normal_(std=0.05)
            features = torch.randn(1, 32, 64)
            tokens = torch.randint(0, 63, (1, 32))
            anchors = torch.tensor([[10]])
            hidden = model.backbone(features, tokens, anchors)
            pre = model.pre_injection.clone()
            changed = tokens.clone()
            changed[:, 11:] = torch.randint(0, 63, changed[:, 11:].shape)
            altered = features.clone()
            altered[:, 10:] = torch.randn_like(altered[:, 10:]) * 100
            torch.testing.assert_close(
                hidden, model.backbone(altered, changed, anchors), rtol=0, atol=0
            )
            ids = model.lm_head(hidden).topk(16, -1).indices
            scores = model.transition_scores(
                pre, ids, tokens[:, 10], model.lm_head.weight
            )
            cached = model.select_cached(scores, ids)
            prev, expected = tokens[:, 10], []
            for slot in range(7):
                refined = model.refine(pre[:, slot], prev)
                logits = (
                    refined[:, None]
                    * functional.embedding(ids[:, slot], model.lm_head.weight)
                ).sum(-1)
                prev = (
                    ids[:, slot].gather(-1, logits.argmax(-1, keepdim=True)).squeeze(-1)
                )
                expected.append(prev)
            torch.testing.assert_close(cached, torch.stack(expected, 1))

    def test_complete_objective_has_finite_new_module_gradients(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "fixture"
            make_fixture(root)
            model = (
                DSpineDraft.load_feature(
                    root / "dense", root / "shared.pt", "cpu", rank=8, message_size=4
                )
                .float()
                .train()
            )
            features = torch.randn(1, 32, 64)
            tokens = torch.randint(0, 63, (1, 32))
            anchors = torch.tensor([[8, 12]])
            hidden = model.backbone(features, tokens, anchors)
            indices = anchors.flatten()[:, None] + torch.arange(7)
            labels, predecessors = tokens[0, indices + 1], tokens[0, indices]
            loss, _ = dspine_loss(
                model,
                hidden,
                torch.randn_like(hidden),
                labels,
                predecessors,
                torch.ones(7),
                chunk_size=5,
            )
            loss.backward()
            self.assertTrue(bool(torch.isfinite(loss)))
            self.assertGreater(float(model.injection.writes[-1].weight.grad.norm()), 0)
            self.assertFalse(model.transfer_codes.requires_grad)


from . import test_pipeline as pipeline  # noqa: E402


class FeaturePipelineTests(pipeline.PipelineTests):
    def config(self, arm="dense", name="run"):
        result = super().config(arm, name)
        result.dspine_enabled = True
        result.objective = "dspine"
        result.transfer_rank = 8
        result.message_size = 4
        return result

    def test_feature_export_restores_all_tensors(self):
        from safetensors.torch import load_file  # noqa: PLC0415

        from speculators.train.cached_dflash2 import trainer  # noqa: PLC0415

        config = self.config(name="export-roundtrip")
        trainer.train(config)
        checkpoint = config.output / "epochs/epoch-02"
        model = DSpineDraft.load_export(checkpoint, config.shared, "cpu")
        state = load_file(str(checkpoint / "model.safetensors"))
        for name, value in state.items():
            actual = model.state_dict()[name]
            torch.testing.assert_close(actual, value.to(actual.dtype), rtol=0, atol=0)
