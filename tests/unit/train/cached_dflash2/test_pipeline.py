# Standalone unittest runner also works without the full repository test environment.
# ruff: noqa: PT009, PT027
"""Integration tests: durable restart, exported weights and corrupt cache rejection."""

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from safetensors.torch import load_file

from speculators.train.cached_dflash2.checkpoint import reconcile_logs
from speculators.train.cached_dflash2.config import CachedDFlash2Config, resolve
from speculators.train.cached_dflash2.data import load_row, preflight, read_manifest
from speculators.train.cached_dflash2.trainer import train

from .test_alignment import make_fixture


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.data = self.root / "data"
        make_fixture(self.data)

    def config(self, arm="dense", name="run"):
        return CachedDFlash2Config(
            dense_checkpoint=self.data / "dense",
            shared=self.data / "shared.pt",
            manifest=self.data / "manifest.json",
            output=self.root / name,
            arm=arm,
            device="cpu",
            synthetic_smoke_test=True,
            epochs=2,
            batch_sequences=3,
            anchors=2,
            eval_anchors=2,
        )

    def test_all_arms_export_with_frozen_selector_and_valid_runtime_metadata(self):
        for arm in ("dense", "token", "fusion"):
            config = self.config(arm, arm)
            self.assertEqual(preflight(config)["planned_updates"], 4)
            with contextlib.redirect_stdout(io.StringIO()):
                summary = train(config)
            self.assertTrue(summary["completed"])
            self.assertEqual(summary["total_steps"], 4)
            self.assertEqual(summary["images_seen"], 12)
            self.assertTrue(summary["selector_unchanged"])
            self.assertFalse(summary["online_validated"])
            logs = [
                json.loads(line)
                for line in (config.output / "train.jsonl").read_text().splitlines()
            ]
            self.assertEqual([row["step"] for row in logs], [1, 2, 3, 4])
            self.assertTrue(
                all(torch.isfinite(torch.tensor(row["loss"])) for row in logs)
            )
            exported = json.loads(
                (config.output / "epochs/epoch-02/config.json").read_text()
            )
            self.assertEqual("dmtp" in exported, arm != "dense")
            if arm != "dense":
                self.assertEqual(exported["dmtp"]["mode"], arm)
                self.assertFalse(exported["dmtp"]["vllm_runtime_verified"])
            for artifact in (
                "run.yaml",
                "train_command.txt",
                "provenance.json",
                "resume.pt",
            ):
                self.assertTrue((config.output / artifact).is_file())

    def test_interrupt_mid_epoch_then_resume_matches_uninterrupted_weights(self):
        config = self.config("fusion", "interrupted")
        from speculators.train.cached_dflash2 import trainer  # noqa: PLC0415

        original = trainer.cache.predictions
        training_calls = 0

        def fail_mid_epoch(*args, **kwargs):
            nonlocal training_calls
            if args[-1] is True:
                training_calls += 1
                if training_calls == 4:
                    raise RuntimeError("simulated interruption")
            return original(*args, **kwargs)

        with (  # noqa: SIM117
            contextlib.redirect_stdout(io.StringIO()),
            patch.object(trainer.cache, "predictions", fail_mid_epoch),
        ):
            with self.assertRaisesRegex(RuntimeError, "simulated interruption"):
                train(config)
        config.resume = True
        with contextlib.redirect_stdout(io.StringIO()):
            train(config)
            reference = self.config("fusion", "reference")
            train(reference)
        actual_weights = load_file(
            str(config.output / "epochs/epoch-02/model.safetensors")
        )
        expected_weights = load_file(
            str(reference.output / "epochs/epoch-02/model.safetensors")
        )
        for name, expected in expected_weights.items():
            torch.testing.assert_close(actual_weights[name], expected, rtol=0, atol=0)
        logs = [
            json.loads(line)
            for line in (config.output / "train.jsonl").read_text().splitlines()
        ]
        self.assertEqual([row["step"] for row in logs], [1, 2, 3, 4])
        self.assertTrue((config.output / "rolled-back-train.jsonl").is_file())

    def test_manifest_rejects_image_leakage_and_cache_rejects_wrong_prompt(self):
        manifest = self.data / "manifest.json"
        records = json.loads(manifest.read_text())
        row = records[0].copy()
        row["prompt_length"] += 1
        with self.assertRaisesRegex(ValueError, "Prompt boundary"):
            load_row(row, False, 32)
        records[-1]["image_sha256"] = records[0]["image_sha256"]
        manifest.write_text(json.dumps(records))
        with self.assertRaisesRegex(ValueError, "leakage"):
            read_manifest(manifest, True, True)

    def test_yaml_cli_override_and_unknown_field_rejection(self):
        config = self.config()
        path = self.root / "config.yaml"
        config.save(path)
        actual = resolve(["--config", str(path), "--lr", "1e-5", "--arm", "token"])
        self.assertEqual(actual.lr, 1e-5)
        self.assertEqual(actual.arm, "token")
        with path.open("a") as file:
            file.write("typo_lr: 0.01\n")
        with self.assertRaises(ValueError):
            resolve(["--config", str(path)])

    def test_corrupt_durable_log_fails_instead_of_silently_losing_history(self):
        (self.root / "train.jsonl").write_text('{"step": 1}\n{"step": 1}\n')
        with self.assertRaisesRegex(ValueError, "duplicate"):
            reconcile_logs(self.root, 2, [])
