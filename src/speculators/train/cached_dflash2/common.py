"""Pure planning and prefix-metric helpers shared by all recipe arms."""

import hashlib
import json
import random
from pathlib import Path
from typing import Any

MIN_RESPONSE_TOKENS = 16


def atomic_json(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temp.replace(path)


def digest_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def epoch_plan(records, epochs, batch_size, token_budget, seed):
    if min(epochs, batch_size, token_budget) <= 0 or not records:
        raise ValueError("Positive batch limits and nonempty records required")
    if any(r["tokens"] > token_budget for r in records):
        raise ValueError("A sequence exceeds the padded context token budget")
    rng = random.Random(seed)
    plan: list[tuple[int, list[dict[str, Any]]]] = []
    for epoch in range(1, epochs + 1):
        order = list(records)
        rng.shuffle(order)
        current = []
        for offset in range(0, len(order), 256):
            batch: list[dict[str, Any]] = []
            longest = 0
            for record in sorted(
                order[offset : offset + 256], key=lambda r: r["tokens"]
            ):
                length = max(longest, record["tokens"])
                if batch and (
                    len(batch) >= batch_size or length * (len(batch) + 1) > token_budget
                ):
                    current.append(batch)
                    batch = []
                    longest = 0
                batch.append(record)
                longest = max(longest, record["tokens"])
            if batch:
                current.append(batch)
        rng.shuffle(current)
        plan.extend((epoch, batch) for batch in current)
    return plan


def linear_lr(step, total, warmup, peak):
    if not 1 <= step <= total or not 0 <= warmup < total:
        raise ValueError("Invalid schedule position")
    if step <= warmup:
        return peak * step / max(1, warmup)
    return peak * (total - step + 1) / (total - warmup)


def prefix_summary(prefix_counts, blocks, independent_counts=None):
    if blocks <= 0 or len(prefix_counts) != 7:  # noqa: PLR2004
        raise ValueError("Expected seven proposed positions and nonempty blocks")
    if any(not 0 <= n <= blocks for n in prefix_counts) or any(
        b > a for a, b in zip(prefix_counts, prefix_counts[1:], strict=False)
    ):
        raise ValueError("Cumulative prefix counts must be nonincreasing")
    previous = [blocks] + prefix_counts[:-1]
    accepted = sum(prefix_counts)
    return {
        "blocks": blocks,
        "accepted_draft_tokens": accepted,
        "proposed_tokens": 7 * blocks,
        "accepted_length": 1 + accepted / blocks,
        "acceptance_rate": accepted / (7 * blocks),
        "t1_acceptance": prefix_counts[0] / blocks,
        "prefix_acceptance": [n / blocks for n in prefix_counts],
        "conditional_acceptance": [
            n / d if d else None for n, d in zip(prefix_counts, previous, strict=True)
        ],
        "independent_position_accuracy": None
        if independent_counts is None
        else [n / blocks for n in independent_counts],
    }
