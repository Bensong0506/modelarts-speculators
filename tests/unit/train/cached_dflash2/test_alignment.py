# ruff: noqa: PT009
"""Dense temporal alignment, seven-token supervision and checkpoint compatibility."""

import json
import tempfile
import unittest
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from speculators.train.cached_dflash2.data import predictions
from speculators.train.cached_dflash2.model import Draft
from speculators.train.cached_dflash2.trainer import export_epoch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def tiny_model():
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
    model = Draft(config, shared)
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
        self.model, self.shared = tiny_model()
        self.features = torch.randn(1, 20, 64, device=DEVICE)
        self.tokens = torch.randint(0, 63, (1, 20), device=DEVICE)
        self.a = torch.tensor([[7]], device=DEVICE)

    def test_future_target_features_have_zero_influence_and_gradient(self):
        self.features.requires_grad_()
        out = self.model.backbone(self.features, self.tokens, self.a)
        changed = self.features.detach().clone()
        changed[:, 7:] = torch.randn_like(changed[:, 7:]) * 20
        other = self.model.backbone(changed, self.tokens, self.a)
        torch.testing.assert_close(out, other, rtol=0, atol=0)
        (grad,) = torch.autograd.grad(out.square().sum(), self.features)
        self.assertEqual(int(torch.count_nonzero(grad[:, 7:])), 0)
        self.assertGreater(float(grad[:, :7].norm()), 0)

    def test_supervision_uses_pending_token_and_seven_teacher_positions(self):
        data = {"ids": self.tokens, "features": self.features}
        hidden, index, first = predictions(self.model, data, self.a)
        self.assertEqual(tuple(hidden.shape), (1, 7, 32))
        torch.testing.assert_close(first, self.tokens[:, 7])
        self.assertEqual(index.flatten().tolist(), [7, 8, 9, 10, 11, 12, 13])

    def test_export_is_dense_and_roundtrips_without_shared_weights(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            out = export_epoch(self.model, root, 1)
            config = json.loads((out / "config.json").read_text())
            self.assertEqual(config["architectures"], ["DFlash2DraftModel"])
            self.assertNotIn("dmtp", config)
            state = load_file(str(out / "model.safetensors"))
            self.assertFalse(
                any(
                    k.startswith(("lm_head.", "embed_tokens.", "anchor_fusion."))
                    for k in state
                )
            )
            shared = root / "shared.pt"
            torch.save(self.shared, shared)
            restored = Draft.load(out, shared, DEVICE)
            for name, expected in state.items():
                torch.testing.assert_close(
                    restored.state_dict()[name], expected.to(DEVICE), rtol=0, atol=0
                )


def make_fixture(dest):
    """Synthetic CPU-sized dense weights and target caches; no MTP sidecars."""
    dest.mkdir(parents=True, exist_ok=False)
    model, shared = tiny_model()
    base = dest / "dense"
    base.mkdir()
    (base / "config.json").write_text(json.dumps(model.config))
    state = {
        k: v.detach().cpu().to(torch.bfloat16).contiguous()
        for k, v in model.state_dict().items()
        if not k.startswith(("embed_tokens.", "lm_head."))
    }
    save_file(state, str(base / "model.safetensors"))
    torch.save({k: v.bfloat16() for k, v in shared.items()}, dest / "shared.pt")
    records = []
    for i in range(8):
        ids = torch.randint(0, 63, (32,))
        path = dest / f"target-{i}.pt"
        torch.save(
            {
                "ids": ids.int(),
                "features": torch.randn(32, 64).bfloat16(),
                "teacher_hidden": torch.randn(32, 32).bfloat16(),
                "prompt_length": 8,
                "image_sha256": f"synthetic-{i}",
            },
            path,
        )
        records.append(
            {
                "id": f"synthetic-{i}",
                "path": str(path),
                "tokens": 32,
                "prompt_length": 8,
                "split": "train" if i < 6 else "val",
                "image_sha256": f"synthetic-{i}",
            }
        )
    (dest / "manifest.json").write_text(json.dumps(records))
