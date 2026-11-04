# Standalone unittest runner also works without the full repository test environment.
# ruff: noqa: PT009, PT027
"""CPU checks of the actual runtime helpers; full vLLM/GPU tests are separate."""

import importlib.util
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

SOURCE = (
    Path(__file__).resolve().parents[4]
    / "integrations/vllm_dmtp/overlay/vllm/v1/worker/gpu/spec_decode/dmtp"
)


def load(name):
    spec = importlib.util.spec_from_file_location(
        "dmtp_" + name, SOURCE / (name + ".py")
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


config = load("config")
fusion = load("fusion")


class RuntimeMathTests(unittest.TestCase):
    def test_compressed_cache_matches_absolute_window_attention(self):
        """Check outputs and gradients at block/window boundaries, not just indices."""
        torch.manual_seed(1102)
        for window in [None, 8, 16, 2048]:
            original = SimpleNamespace(dflash_config={"swa_window_size": window})
            runtime_window = config.compress_context_gap(original).dflash_config[
                "swa_window_size"
            ]
            self.assertEqual(original.dflash_config["swa_window_size"], window)
            for a in [1, 7, 15, 16, 17, 2048, 2051]:
                q = torch.randn(7, 8, requires_grad=True)
                k = torch.randn(a + 7, 8, requires_grad=True)
                v = torch.randn(a + 7, 8, requires_grad=True)
                # Context excludes target feature at a; queries use positions a+1..a+7.
                absolute_q = torch.arange(a + 1, a + 8)
                absolute_k = torch.cat([torch.arange(a), absolute_q])
                reference_mask = torch.ones(7, a + 7, dtype=torch.bool)
                if window:
                    reference_mask[:, :a] = (
                        absolute_k[:a] >= absolute_q[:, None] - window + 1
                    )
                # The paged attention backend sees adjacent physical cache slots.
                cache_q = torch.arange(a, a + 7)
                cache_k = torch.arange(a + 7)
                runtime_mask = torch.ones_like(reference_mask)
                if runtime_window:
                    runtime_mask = cache_k >= cache_q[:, None] - runtime_window + 1
                self.assertTrue(torch.equal(reference_mask, runtime_mask))

                def attend(mask, q=q, k=k, v=v):
                    scores = (q @ k.T / 8**0.5).masked_fill(~mask, float("-inf"))
                    return scores.softmax(-1) @ v

                expected, actual = attend(reference_mask), attend(runtime_mask)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                g1 = torch.autograd.grad(
                    expected.square().sum(), (q, k, v), retain_graph=True
                )
                g2 = torch.autograd.grad(actual.square().sum(), (q, k, v))
                for x, y in zip(g1, g2, strict=True):
                    torch.testing.assert_close(x, y, rtol=0, atol=0)

    def test_runtime_fusion_matches_training_formula_and_weight_names(self):
        torch.manual_seed(1102)
        m = fusion.AnchorFusion(32, torch.float32)
        self.assertEqual(
            set(m.state_dict()),
            {"embedding_norm.weight", "mtp_norm.weight", "projection.weight"},
        )
        with torch.no_grad():
            m.embedding_norm.weight.uniform_(0.5, 1.5)
            m.mtp_norm.weight.uniform_(0.5, 1.5)
        x, h = torch.randn(3, 32), torch.randn(3, 32)

        def norm(t, weight):
            t32 = t.float()
            return (
                t32
                * torch.rsqrt(t32.square().mean(-1, keepdim=True) + 1e-6)
                * weight.to(t.dtype).float()
            ).to(t.dtype)

        expected = torch.nn.functional.linear(
            torch.cat(
                [norm(x, m.embedding_norm.weight), norm(h, m.mtp_norm.weight)], -1
            ),
            m.projection.weight,
        )
        torch.testing.assert_close(m(x, h), expected, rtol=0, atol=0)
        self.assertFalse(torch.equal(m(x, h), m(x, h.flip(0))))

    def test_reject_window_that_cannot_hold_unrestricted_query_block(self):
        for window in [1, 6, 7]:
            with self.assertRaises(ValueError):
                config.compress_context_gap(
                    SimpleNamespace(dflash_config={"swa_window_size": window})
                )

    def test_residual_runtime_matches_trainer_with_trained_weights(self):
        from speculators.train.cached_dflash2.dmtp import AnchorFusion  # noqa: PLC0415

        torch.manual_seed(1103)
        trainer = AnchorFusion(32, residual=True)
        with torch.no_grad():
            trainer.projection.weight.normal_(std=0.03)
        runtime = fusion.AnchorFusion(32, torch.float32, residual=True)
        runtime.load_state_dict(trainer.state_dict())
        x, h = torch.randn(3, 32), torch.randn(3, 32)
        torch.testing.assert_close(runtime(x, h), trainer(x, h), rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
