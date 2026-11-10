"""Build and optionally launch the pinned dense DFlash2 vLLM OpenAI server."""

import argparse
import concurrent.futures
import datetime
import importlib.util
import json
import os
import shlex
import sys
from pathlib import Path

from _provenance import git_diff, git_sha, package_versions
from vllm_dflash2_runtime import check_runtime, sha256


def draft_options(mode, draft):
    if mode not in ("target", "dense"):
        raise ValueError("This branch supports only target or dense inference")
    if mode == "target":
        if draft:
            raise ValueError("Target-only mode does not take --draft")
        return None, None
    if not draft:
        raise ValueError("--draft must name an exported epoch directory")
    draft = Path(draft).resolve()
    config = json.loads((draft / "config.json").read_text())
    if config.get("dmtp") or config.get("architectures") != ["DFlash2DraftModel"]:
        raise ValueError("Dense serving needs a DFlash2DraftModel export")
    return {
        "method": "dflash",
        "num_speculative_tokens": 7,
        "model": str(draft),
    }, None


def command(args):
    if args.batched_tokens < 7 * args.concurrency:
        raise ValueError("batched-tokens must be at least 7 * concurrency")
    speculative, additional = draft_options(args.mode, args.draft)
    result = [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        str(args.target.resolve()),
        "--served-model-name",
        "allava-target",
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--dtype",
        "bfloat16",
        "--tensor-parallel-size",
        "1",
        "--max-model-len",
        "8192",
        "--max-num-seqs",
        str(args.concurrency),
        "--max-num-batched-tokens",
        str(args.batched_tokens),
        "--kv-cache-memory-bytes",
        str(args.kv_cache_gib << 30),
        "--no-enable-prefix-caching",
        "--block-size",
        "384",
        "--attention-config",
        json.dumps({"flash_attn_version": 2}),
        "--engram-config",
        json.dumps({"cpu_offload": False}),
        "--limit-mm-per-prompt",
        json.dumps({"image": 1, "video": 0}),
        "--mm-processor-kwargs",
        json.dumps({"size": {"shortest_edge": 65536, "longest_edge": 262144}}),
        "--kernel-config",
        json.dumps({"moe_backend": "triton"}),
    ]
    if args.graph:
        result += [
            "--compilation-config",
            json.dumps({"mode": 0, "cudagraph_mode": "FULL_DECODE_ONLY"}),
        ]
    else:
        result.append("--enforce-eager")
    if speculative:
        result += ["--speculative-config", json.dumps(speculative)]
    if additional:
        result += ["--additional-config", json.dumps(additional)]
    return result


def target_checkpoint_files(target):
    """Reject an incomplete sharded copy before recording or loading weights."""
    target = Path(target)
    files = [target / "config.json", *sorted(target.glob("*.safetensors"))]
    if not files[0].is_file() or len(files) == 1:
        raise ValueError("Target must be a local config + safetensors checkpoint")
    index = target / "model.safetensors.index.json"
    if index.is_file():
        shards = set(json.loads(index.read_text())["weight_map"].values())
        available = {p.name for p in files[1:]}
        if shards != available:
            raise ValueError(
                f"Target shard index mismatch: missing={sorted(shards - available)}, "
                f"extra={sorted(available - shards)}"
            )
        files.insert(1, index)
    return files


def prepare_runtime(args, argv):
    """Check the installed fork and save provenance before engine creation."""
    audit = check_runtime(args.runtime_root)
    installed = importlib.util.find_spec("vllm")
    if (
        installed is None
        or installed.origin is None
        or Path(installed.origin).resolve().parent
        != args.runtime_root.resolve() / "vllm"
    ):
        raise ValueError(
            "This Python does not import the checked private vLLM checkout"
        )
    checkpoint_files = target_checkpoint_files(args.target)
    args.provenance_dir.mkdir(parents=True, exist_ok=False)
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        hashes = list(pool.map(sha256, checkpoint_files))
    (args.provenance_dir / "checkpoint_sha256.txt").write_text(
        "".join(
            f"{digest}  {path.name}\n"
            for path, digest in zip(checkpoint_files, hashes, strict=True)
        )
    )
    if args.draft:
        (args.provenance_dir / "drafter_checkpoint_sha256.txt").write_text(
            "".join(
                f"{sha256(args.draft / name)}  {name}\n"
                for name in ("config.json", "model.safetensors")
            )
        )
    repo = Path(__file__).resolve().parents[1]
    header = [
        f"# Timestamp: {datetime.datetime.now(datetime.UTC).isoformat()}",
        f"# Speculators: {git_sha(repo)}",
        *package_versions(),
    ]
    (args.provenance_dir / "vllm_command.txt").write_text(
        "\n".join(header) + "\nVLLM_USE_V2_MODEL_RUNNER=1 " + shlex.join(argv) + "\n"
    )
    (args.provenance_dir / "vllm.patch").write_text(git_diff(args.runtime_root))
    (args.provenance_dir / "speculators.patch").write_text(git_diff(repo))
    (args.provenance_dir / "runtime.json").write_text(
        json.dumps(audit, indent=2) + "\n"
    )
    os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "1"
    os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=["dense", "target"],
        required=True,
    )
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--draft", type=Path)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--provenance-dir", type=Path, required=True)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--kv-cache-gib", type=int, default=4)
    parser.add_argument("--batched-tokens", type=int, default=2048)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--graph", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if min(args.concurrency, args.kv_cache_gib, args.batched_tokens) <= 0:
        parser.error("Resource limits must be positive")
    argv = command(args)
    if args.dry_run:
        print("VLLM_USE_V2_MODEL_RUNNER=1 " + shlex.join(argv))
        return
    prepare_runtime(args, argv)
    os.execv(sys.executable, argv)  # noqa: S606


if __name__ == "__main__":
    main()
