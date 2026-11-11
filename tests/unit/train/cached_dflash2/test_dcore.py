# ruff: noqa: PT009
"""Reference gradients, warm start equivalence and causal target boundaries."""

import tempfile
import unittest
from pathlib import Path

import torch
from torch.nn import functional

from speculators.train.cached_dflash2.dcore import (
    DCoreDraft,
    continuation_weights,
    dcore_loss,
)
from speculators.train.cached_dflash2.model import Draft

from .test_alignment import make_fixture


class DCoreTests(unittest.TestCase):
    def test_full_vocabulary_gradient_matches_detached_reference(self):
        torch.manual_seed(33)
        hidden = torch.randn(3, 7, 12, requires_grad=True)
        head = torch.randn(31, 12) / 3
        labels = torch.randint(0, 31, (3, 7))
        actual = dcore_loss(hidden, labels, head, chunk_size=5)
        logits = functional.linear(hidden, head)
        ce = functional.cross_entropy(
            logits.flatten(0, 1), labels.flatten(), reduction="none"
        ).reshape(3, 7)
        weights = continuation_weights((-ce).exp())
        expected = (ce * weights).sum() / weights.sum()
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(
            torch.autograd.grad(actual, hidden)[0],
            torch.autograd.grad(expected, hidden)[0],
        )
        self.assertFalse(weights.requires_grad)

    def test_bfloat16_head_gradient_matches_reference(self):
        torch.manual_seed(37)
        hidden = (torch.randn(2, 7, 12).bfloat16() / 3).requires_grad_()
        labels = torch.randint(0, 31, (2, 7))
        head = (torch.randn(31, 12) / 3).bfloat16()
        logits = functional.linear(hidden, head).float()
        ce = functional.cross_entropy(
            logits.flatten(0, 1), labels.flatten(), reduction="none"
        ).reshape(2, 7)
        actual = dcore_loss(hidden, labels, head, chunk_size=4)
        weights = continuation_weights((-ce).exp())
        expected = (ce * weights).sum() / weights.sum()
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(
            torch.autograd.grad(actual, hidden)[0],
            torch.autograd.grad(expected, hidden)[0],
            rtol=0.02,
            atol=3e-5,
        )

    def test_zero_fusion_and_copied_context_projections_preserve_initial_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "fixture"
            make_fixture(root)
            dense = Draft.load(root / "dense", root / "shared.pt", "cpu").float()
            model = DCoreDraft.load_feature(
                root / "dense", root / "shared.pt", "cpu"
            ).float()
            features = torch.randn(2, 32, 64, requires_grad=True)
            tokens = torch.randint(0, 63, (2, 32))
            anchors = torch.tensor([[9, 12], [10, 15]])
            actual = model.backbone(features, tokens, anchors)
            expected = dense.backbone(features, tokens, anchors)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            actual.square().sum().backward()
            self.assertGreater(float(model.layer_fusion_weights.grad.norm()), 0)
            assert features.grad is not None
            self.assertEqual(int(torch.count_nonzero(features.grad[0, 12:])), 0)
            self.assertEqual(int(torch.count_nonzero(features.grad[1, 15:])), 0)


from . import test_pipeline as pipeline  # noqa: E402


class FeaturePipelineTests(pipeline.PipelineTests):
    def config(self, arm="dense", name="run"):
        result = super().config(arm, name)
        result.dcore_context = True
        result.objective = "dpace"
        return result

    def test_feature_export_restores_all_tensors(self):
        from safetensors.torch import load_file  # noqa: PLC0415

        from speculators.train.cached_dflash2 import trainer  # noqa: PLC0415

        config = self.config(name="export-roundtrip")
        trainer.train(config)
        checkpoint = config.output / "epochs/epoch-02"
        model = DCoreDraft.load_export(checkpoint, config.shared, "cpu")
        state = load_file(str(checkpoint / "model.safetensors"))
        for name, value in state.items():
            actual = model.state_dict()[name]
            torch.testing.assert_close(actual, value.to(actual.dtype), rtol=0, atol=0)
