# Standalone unittest runner also works without the full repository test environment.
# ruff: noqa: PT009, PT027
"""Small behavioral checks for temporal alignment and the training interface."""

import copy
import json
import tempfile
import unittest
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from speculators.train.cached_dflash2.data import ALIGNMENT, predictions
from speculators.train.cached_dflash2.dmtp import (
    AnchorFusion,
    DMTPDraft,
    block_attention,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def tiny_model(mode="fusion"):
    torch.manual_seed(1102)
    config = {
        "hidden_size": 32,
        "vocab_size": 64,
        "intermediate_size": 64,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 8,
        "num_hidden_layers": 2,
        "architectures": ["DFlash2DraftModel"],
        "dflash_config": {
            "target_layer_ids": [0, 1],
            "conv_group_size": 8,
            "conv_kernel_size": 2,
            "selector_rank": 8,
            "mask_token_id": 63,
        },
    }
    shared = {
        "embed_tokens.weight": torch.randn(64, 32) / 4,
        "lm_head.weight": torch.randn(64, 32) / 4,
    }
    model = DMTPDraft(config, shared, mode)
    with torch.no_grad():
        for layer in model.layers:
            for conv in (layer.attention_conv, layer.mlp_conv):
                conv.base_kernel[:, 0].fill_(1)
                conv.kernel_projection.weight.zero_()
        for p in model.candidate_selector.parameters():
            p.normal_(std=0.02)
    return model.to(DEVICE), shared


class TimingTests(unittest.TestCase):
    def setUp(self):
        self.model, _ = tiny_model()
        self.features = torch.randn(1, 20, 64, device=DEVICE)
        self.tokens = torch.randint(0, 63, (1, 20), device=DEVICE)
        self.a = torch.tensor([[7]], device=DEVICE)
        self.first = self.tokens[:, 8:9]
        self.mtp = torch.randn(1, 1, 32, device=DEVICE)

    def test_future_target_features_have_zero_influence_and_gradient(self):
        self.features.requires_grad_()
        out = self.model.backbone(
            self.features, self.tokens, self.a, self.mtp, self.first
        )
        changed = self.features.detach().clone()
        changed[:, 7:] = torch.randn_like(changed[:, 7:]) * 20
        other = self.model.backbone(changed, self.tokens, self.a, self.mtp, self.first)
        torch.testing.assert_close(out, other, rtol=0, atol=0)
        (grad,) = torch.autograd.grad(out.square().sum(), self.features)
        self.assertEqual(int(torch.count_nonzero(grad[:, 7:])), 0)
        self.assertGreater(float(grad[:, :7].norm()), 0)

    def test_zero_residual_preserves_token_tail_then_learns(self):
        self.model.anchor_fusion = AnchorFusion(32, residual=True).to(DEVICE)
        token = copy.deepcopy(self.model)
        token.mode = "token"
        expected = token.backbone(
            self.features, self.tokens, self.a, self.mtp, self.first
        )
        actual = self.model.backbone(
            self.features, self.tokens, self.a, self.mtp, self.first
        )
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        (actual * torch.randn_like(actual)).sum().backward()
        weight = self.model.anchor_fusion.projection.weight
        self.assertGreater(float(weight.grad.norm()), 0)
        with torch.no_grad():
            weight.add_(weight.grad, alpha=-0.01)
        changed = self.model.backbone(
            self.features, self.tokens, self.a, self.mtp, self.first
        )
        self.assertFalse(torch.equal(changed, expected))

    def test_shifted_query_attention_matches_explicit_block_reference(self):
        attn = self.model.layers[0].self_attn
        x = torch.randn(2, 7, 32, device=DEVICE)
        context = torch.randn(1, 20, 32, device=DEVICE)
        # Reference uses original attention after moving forbidden target slots
        # out of the visible context. At a+1 only the extra slot a needs masking.
        qanchors = torch.tensor([8, 11], device=DEVICE)
        ends = qanchors - 1
        for window in [None, 6]:
            actual = block_attention(attn, x, context, qanchors, ends, window)
            expected = []
            for i in range(2):
                query = x[i : i + 1]
                ctx = context[:, : ends[i]]
                # Truncated context cannot reveal target feature at the pending A.
                expected.append(
                    attn.reference_forward(query, ctx, qanchors[i : i + 1], window)
                )
            torch.testing.assert_close(
                actual, torch.cat(expected), rtol=2e-4, atol=2e-5
            )

    def test_inference_keeps_native_first_candidate_and_fusion_receives_gradients(self):
        wrong_first = (self.first + 1) % 63
        proposed = self.model.propose(
            self.features, self.tokens, self.a, self.mtp, wrong_first
        )
        self.assertEqual(tuple(proposed.shape), (1, 7))
        torch.testing.assert_close(proposed[:, 0], wrong_first.reshape(-1))
        h = self.model.backbone(
            self.features, self.tokens, self.a, self.mtp, self.first
        )
        h.square().mean().backward()
        self.assertGreater(
            float(self.model.anchor_fusion.projection.weight.grad.norm()), 0
        )
        self.assertIsNone(self.model.embed_tokens.weight.grad)
        self.assertIsNone(self.model.lm_head.weight.grad)
        with self.assertRaises(ValueError):
            self.model.anchor_fusion(
                torch.zeros(1, 32, device=DEVICE), torch.zeros(1, 128, device=DEVICE)
            )

    def test_training_and_evaluation_use_distinct_first_token_sources(self):
        data = {
            "ids": self.tokens,
            "features": self.features,
            "mtp_sample_hidden": torch.randn(1, 20, 32, device=DEVICE),
            "mtp_token": (self.tokens + 2) % 63,
        }
        _, index, first_train = predictions(self.model, data, self.a, True)
        _, _, first_eval = predictions(self.model, data, self.a, False)
        torch.testing.assert_close(first_train, self.tokens[:, 8])
        torch.testing.assert_close(first_eval, data["mtp_token"][:, 7])
        self.assertEqual(index.flatten().tolist(), [8, 9, 10, 11, 12, 13])

    def test_export_identifies_hybrid_and_roundtrips_all_fusion_weights(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "export"
            self.model.export_hf(out)
            config = json.loads((out / "config.json").read_text())
            self.assertEqual(config["architectures"], ["DMTPDFlash2DraftModel"])
            self.assertFalse(config["dmtp"]["vllm_runtime_verified"])
            restored, _ = tiny_model()
            missing, extra = restored.load_state_dict(
                load_file(str(out / "model.safetensors")), strict=False
            )
            self.assertEqual(set(missing), {"lm_head.weight", "embed_tokens.weight"})
            self.assertFalse(extra)


def make_fixture(dest):
    """Synthetic CPU-sized weights and caches, solely for pipeline tests."""
    dest.mkdir(parents=True, exist_ok=False)
    model, shared = tiny_model()
    base = dest / "dense"
    base.mkdir()
    (base / "config.json").write_text(json.dumps(model.config))
    state = {
        k: v.detach().cpu().to(torch.bfloat16).contiguous()
        for k, v in model.state_dict().items()
        if not k.startswith(("anchor_fusion.", "embed_tokens.", "lm_head."))
    }
    save_file(state, str(base / "model.safetensors"))
    torch.save({k: v.bfloat16() for k, v in shared.items()}, dest / "shared.pt")
    records = []
    for i in range(8):
        ids = torch.randint(0, 63, (32,))
        path = dest / f"target-{i}.pt"
        mtp = dest / f"mtp-{i}.pt"
        torch.save(
            {
                "ids": ids,
                "features": torch.randn(32, 64).bfloat16(),
                "teacher_hidden": torch.randn(32, 32).bfloat16(),
                "prompt_length": 8,
                "image_sha256": f"synthetic-{i}",
            },
            path,
        )
        torch.save(
            {
                "ids": ids,
                "mtp_sample_hidden": torch.randn(32, 32).bfloat16(),
                "mtp_token": torch.randint(0, 63, (32,)),
                "mtp_valid": torch.ones(32, dtype=torch.bool),
                "alignment": ALIGNMENT,
                "history": "native_causal_target_aligned",
                "image_sha256": f"synthetic-{i}",
                "synthetic_test_only": True,
            },
            mtp,
        )
        records.append(
            {
                "id": f"synthetic-{i}",
                "path": str(path),
                "mtp_path": str(mtp),
                "tokens": 32,
                "prompt_length": 8,
                "split": "train" if i < 6 else "val",
                "image_sha256": f"synthetic-{i}",
            }
        )
    (dest / "manifest.json").write_text(json.dumps(records))
