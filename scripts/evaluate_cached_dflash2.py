"""Matched image-conditioned vLLM acceptance evaluation for the pinned runtime."""

import argparse
import hashlib
import io
import json
import sys
import time
from pathlib import Path

from PIL import Image
from serve_cached_dflash2 import draft_options, prepare_runtime
from vllm_dmtp_runtime import sha256


def image_for(row, base):
    path = Path(row.get("image_path", row.get("image_tar", "")))
    if not path.is_absolute():
        path = base / path
    if "image_path" in row:
        payload = path.read_bytes()
    else:
        with path.open("rb") as stream:
            stream.seek(row["image_offset"])
            payload = stream.read(row["image_size"])
    if hashlib.sha256(payload).hexdigest() != row["image_sha256"]:
        raise ValueError(f"Image identity mismatch: {row['id']}")
    return Image.open(io.BytesIO(payload)).convert("RGB")


def request(tokenizer, row, base):
    prompt = tokenizer.apply_chat_template(
        [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": row["question"]},
                ],
            }
        ],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    return {"prompt": prompt, "multi_modal_data": {"image": image_for(row, base)}}


def summarize(records, elapsed):
    counters = [r["spec"] for r in records if r["spec"] is not None]
    accepted = sum(r["num_accepted_draft_tokens"] for r in counters)
    drafted = sum(r["num_draft_tokens"] for r in counters)
    steps = sum(r["num_spec_steps"] for r in counters)
    return {
        "images": len(records),
        "seconds": elapsed,
        "acceptance_rate": accepted / drafted if drafted else None,
        "accepted_length": 1 + accepted / steps if steps else None,
        "accepted_draft_tokens": accepted,
        "proposed_tokens": drafted,
        "verification_steps": steps,
        "tokens_per_second": sum(len(r["token_ids"]) for r in records) / elapsed,
    }


def main():  # noqa: C901
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=["dense", "token", "fusion", "target", "native-mtp"],
        required=True,
    )
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--target-precision", choices=["fp8", "bf16"], required=True)
    parser.add_argument("--draft", type=Path)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--provenance-dir", type=Path, required=True)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--kv-cache-gib", type=int, default=4)
    parser.add_argument("--batched-tokens", type=int, default=2048)
    parser.add_argument("--limit", type=int, default=16)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--graph", action="store_true")
    parser.add_argument(
        "--progress", action="store_true", help="Log completed requests"
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if (
        min(
            args.concurrency,
            args.kv_cache_gib,
            args.limit,
            args.max_tokens,
            args.repeats,
        )
        <= 0
        or args.batched_tokens < 7 * args.concurrency
    ):
        parser.error(
            "Positive resource/count limits and "
            "batched-tokens >= 7 * concurrency required"
        )
    speculative, additional = draft_options(args.mode, args.draft)
    kwargs = {
        "model": str(args.target.resolve()),
        "dtype": "bfloat16",
        "tensor_parallel_size": 1,
        "max_model_len": 8192,
        "max_num_seqs": args.concurrency,
        "max_num_batched_tokens": args.batched_tokens,
        "kv_cache_memory_bytes": args.kv_cache_gib << 30,
        "enable_prefix_caching": False,
        "enforce_eager": not args.graph,
        "block_size": 384,
        "attention_config": {"flash_attn_version": 2},
        "engram_config": {"cpu_offload": False},
        "kernel_config": {"moe_backend": "triton"},
        "limit_mm_per_prompt": {"image": 1, "video": 0},
        "mm_processor_kwargs": {
            "size": {"shortest_edge": 65536, "longest_edge": 262144}
        },
        "per_request_spec_decode_metrics": "none"
        if args.mode == "target"
        else "summary",
    }
    if speculative:
        kwargs["speculative_config"] = speculative
    if additional:
        kwargs["additional_config"] = additional
    if args.graph:
        kwargs["compilation_config"] = {"mode": 0, "cudagraph_mode": "FULL_DECODE_ONLY"}
    rows = [
        json.loads(line)
        for line in args.prompts.read_text().splitlines()
        if line.strip()
    ][: args.limit]
    if (
        len(rows) != args.limit
        or len({r["image_sha256"] for r in rows}) != len(rows)
        or len({r["id"] for r in rows}) != len(rows)
    ):
        raise ValueError(
            "Requested count must match a manifest subset of unique IDs and images"
        )
    report = {
        "completed": False,
        "mode": args.mode,
        "engine_kwargs": kwargs,
        "target_precision_operator_declared": args.target_precision,
        "target_config": json.loads((args.target / "config.json").read_text()),
        "prompt_manifest_sha256": sha256(args.prompts),
        "repeats": [],
        "scope": (
            "Online image-conditioned greedy generation; AL includes bonus; "
            "output equivalence is a separate check."
        ),
    }
    if args.dry_run:
        print(json.dumps(report, indent=2))
        return
    prepare_runtime(args, [sys.executable, *sys.argv])
    out = args.provenance_dir
    (out / "engine_kwargs.json").write_text(json.dumps(kwargs, indent=2) + "\n")
    from vllm import LLM, SamplingParams  # noqa: PLC0415

    llm = LLM(**kwargs)
    inputs = [
        request(llm.get_tokenizer(), row, args.prompts.resolve().parent) for row in rows
    ]
    llm.generate(
        inputs[: args.concurrency],
        SamplingParams(temperature=0, max_tokens=16, seed=280),
        use_tqdm=False,
    )
    reference = None
    for repeat in range(args.repeats):
        start = time.monotonic()
        outputs = llm.generate(
            inputs,
            SamplingParams(temperature=0, max_tokens=args.max_tokens, seed=280),
            use_tqdm=args.progress,
        )
        elapsed = time.monotonic() - start
        records = []
        for row, output in zip(rows, outputs, strict=True):
            answer = output.outputs[0]
            metrics = answer.spec_decode_metrics
            if args.mode != "target" and metrics is None:
                raise RuntimeError(
                    "Missing acceptance counters; do not substitute token accuracy"
                )
            records.append(
                {
                    "id": row["id"],
                    "image_sha256": row["image_sha256"],
                    "repeat": repeat,
                    "token_ids": list(answer.token_ids),
                    "finish_reason": answer.finish_reason,
                    "spec": metrics.to_dict() if metrics is not None else None,
                }
            )
        result = summarize(records, elapsed)
        result["repeat"] = repeat
        result["changed_vs_repeat0"] = (
            None
            if reference is None
            else sum(
                x["token_ids"] != y["token_ids"]
                for x, y in zip(reference, records, strict=True)
            )
        )
        if reference is None:
            reference = records
        report["repeats"].append(result)
        with (out / "requests.jsonl").open("a") as stream:
            for record in records:
                stream.write(json.dumps(record) + "\n")
        (out / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(result), flush=True)
    report["completed"] = True
    (out / "summary.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
