"""Test deployment guards and actual acceptance/output denominators without vLLM."""

# ruff: noqa: PT009, PT027
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

SCRIPTS = Path(__file__).resolve().parents[4] / "scripts"


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


load_script("_provenance")
runtime = load_script("vllm_dmtp_runtime")
serve = load_script("serve_cached_dflash2")
evaluate = load_script("evaluate_cached_dflash2")
comparison = load_script("compare_cached_dflash2_outputs")


class ToolTests(unittest.TestCase):
    def test_target_copy_requires_every_indexed_shard(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "config.json").write_text("{}")
            index = root / "model.safetensors.index.json"
            index.write_text(
                json.dumps(
                    {"weight_map": {"a": "part1.safetensors", "b": "part2.safetensors"}}
                )
            )
            (root / "part1.safetensors").write_bytes(b"first")
            with self.assertRaisesRegex(ValueError, "missing=.*part2"):
                serve.target_checkpoint_files(root)
            (root / "part2.safetensors").write_bytes(b"second")
            self.assertEqual(len(serve.target_checkpoint_files(root)), 4)
            self.assertIn(index, serve.target_checkpoint_files(root))

    def test_bundle_matches_all_original_runtime_hashes(self):
        self.assertEqual(len(runtime.manifest()["files"]), 22)

    def test_overlay_requires_base_and_clean_tree_and_detects_tampering(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            checkout, bundle = root / "checkout", root / "bundle"
            checkout.mkdir()
            subprocess.run(  # noqa: S603
                ["git", "init", str(checkout)],  # noqa: S607
                check=True,
                capture_output=True,
            )
            runtime.git(checkout, "config", "user.email", "test@example.invalid")
            runtime.git(checkout, "config", "user.name", "Component Test")
            (checkout / "base.txt").write_text("base\n")
            runtime.git(checkout, "add", "base.txt")
            runtime.git(checkout, "commit", "-m", "base")
            source = bundle / "overlay/vllm/adapter.py"
            source.parent.mkdir(parents=True)
            source.write_text("# adapter\n")
            (bundle / "source-provenance.json").write_text(
                json.dumps(
                    {
                        "base_commit": runtime.git(checkout, "rev-parse", "HEAD"),
                        "files": [
                            {
                                "path": "vllm/adapter.py",
                                "sha256": runtime.sha256(source),
                            }
                        ],
                    }
                )
            )
            with patch.object(runtime, "INTEGRATION", bundle):
                self.assertTrue(runtime.apply_overlay(checkout)["dry_run"])
                (checkout / "untracked.py").write_text("# local\n")
                with self.assertRaisesRegex(ValueError, "clean"):
                    runtime.apply_overlay(checkout, True)
                (checkout / "untracked.py").unlink()
                self.assertEqual(
                    runtime.apply_overlay(checkout, True)["verified_overlay_files"], 1
                )
                (checkout / "vllm/adapter.py").write_text("# tampered\n")
                with self.assertRaisesRegex(ValueError, "changed runtime"):
                    runtime.check_runtime(checkout)

    def test_serving_uses_mtp_scheduler_and_custom_tail_only_for_hybrid(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)
            metadata = {
                "schema_version": 1,
                "mode": "fusion",
                "draft_input_slots": 7,
                "proposed_tokens": 7,
                "dflash_predicted_tokens": 6,
                "mtp_hidden": "single_stream_before_first_candidate",
                "target_context": "strictly_before_pending_A",
                "requires_hybrid_vllm_proposer": True,
            }
            (path / "config.json").write_text(
                json.dumps(
                    {"architectures": ["DMTPDFlash2DraftModel"], "dmtp": metadata}
                )
            )
            spec, additional = serve.draft_options("fusion", path)
            self.assertEqual(spec, {"method": "mtp", "num_speculative_tokens": 7})
            self.assertEqual(additional["swx_dmtp"]["model"], str(path.resolve()))
            with self.assertRaises(ValueError):
                serve.draft_options("dense", path)
            with self.assertRaises(ValueError):
                serve.draft_options("token", path)
            args = SimpleNamespace(
                mode="fusion",
                draft=path,
                target=path,
                batched_tokens=2048,
                concurrency=1,
                kv_cache_gib=4,
                graph=False,
                host="127.0.0.1",
                port=8000,
            )
            argv = serve.command(args)
            self.assertIn("--additional-config", argv)
            self.assertIn("--enforce-eager", argv)

    def test_online_counts_use_actual_proposals_and_verification_steps(self):
        records = [
            {
                "token_ids": [1, 2, 3],
                "spec": {
                    "num_accepted_draft_tokens": 2,
                    "num_draft_tokens": 14,
                    "num_spec_steps": 2,
                },
            }
        ]
        result = evaluate.summarize(records, 2)
        self.assertEqual(result["acceptance_rate"], 1 / 7)
        self.assertEqual(result["accepted_length"], 2)

    def test_output_comparison_rejects_different_images_and_finds_first_difference(
        self,
    ):
        a = {
            "x": {"image_sha256": "a", "token_ids": [1, 2, 3], "finish_reason": "stop"}
        }
        b = {
            "x": {"image_sha256": "a", "token_ids": [1, 4, 3], "finish_reason": "stop"}
        }
        result = comparison.compare(a, b)
        self.assertEqual(result["changed"], 1)
        self.assertEqual(result["differences"][0]["first_difference"], 1)
        b["x"]["image_sha256"] = "b"
        with self.assertRaisesRegex(ValueError, "identities"):
            comparison.compare(a, b)
