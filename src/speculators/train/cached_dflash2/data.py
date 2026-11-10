"""Read aligned target caches for seven-token dense DFlash2 training."""

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import torch

from .common import MIN_RESPONSE_TOKENS, epoch_plan, file_sha256


def read_manifest(path, synthetic_smoke_test=False):  # noqa: C901
    path = Path(path)
    records = json.loads(path.read_text())
    if not isinstance(records, list) or not records:
        raise ValueError("Expected a nonempty manifest array")
    if len({r["id"] for r in records}) != len(records):
        raise ValueError("Duplicate sample IDs")
    image_splits: dict[str, str] = {}
    for record in records:
        if record["split"] not in ("train", "val", "test"):
            raise ValueError("Unknown split")
        identity = record["image_sha256"]
        previous = image_splits.setdefault(identity, record["split"])
        if previous != record["split"]:
            raise ValueError("Image leakage across train/val/test")
        if not 0 < record["prompt_length"] < record["tokens"]:
            raise ValueError("Invalid transcript or prompt length")
        cache_path = Path(record["path"]).expanduser()
        if not cache_path.is_absolute():
            cache_path = path.resolve().parent / cache_path
        if not cache_path.is_file():
            raise FileNotFoundError(cache_path)
        record["path"] = str(cache_path)
    synthetic = [r["image_sha256"].startswith("synthetic-") for r in records]
    if any(synthetic) and (not synthetic_smoke_test or not all(synthetic)):
        raise ValueError(
            "Synthetic fixtures need explicit mode and cannot mix with real data"
        )
    for split in ("train", "val"):
        if not any(
            r["split"] == split
            and r["tokens"] - r["prompt_length"] >= MIN_RESPONSE_TOKENS
            for r in records
        ):
            raise ValueError(
                "Nonempty train and val splits with >=16 response tokens required"
            )
    return records


def preflight(config):
    """Read all caches on CPU, checking alignment, optional hashes and dimensions."""
    model_config = json.loads((config.dense_checkpoint / "config.json").read_text())
    width = model_config["hidden_size"]
    layers = len(model_config["dflash_config"]["target_layer_ids"])
    records = read_manifest(config.manifest, config.synthetic_smoke_test)
    for record in records:
        data = load_row(record, width)
        keys = ["features", "teacher_hidden"]
        if any(not torch.isfinite(data[key]).all() for key in keys):
            raise ValueError("Cache contains nonfinite hidden states")
        if data["features"].shape[1] != layers * width:
            raise ValueError(
                "Target feature width differs from checkpoint target layers"
            )
        if "sha256" in record and file_sha256(record["path"]) != record["sha256"]:
            raise ValueError(f"Cache checksum mismatch: {record['id']}")
    train = [
        r
        for r in records
        if r["split"] == "train"
        and r["tokens"] - r["prompt_length"] >= MIN_RESPONSE_TOKENS
    ]
    plan = epoch_plan(
        train,
        config.epochs,
        config.batch_sequences,
        config.context_token_budget,
        config.seed,
    )
    return {
        "checked_records": len(records),
        "train_images": len(train),
        "planned_updates": len(plan),
        "cache_target_precision": config.cache_target_precision,
        "note": "Precision is operator-declared; verify capture provenance separately.",
    }


def load_row(record, width):
    data = torch.load(record["path"], map_location="cpu", weights_only=True)
    if (
        len(data["ids"]) != record["tokens"]
        or data["image_sha256"] != record["image_sha256"]
    ):
        raise ValueError("Target cache and manifest mismatch")
    length = len(data["ids"])
    if data["ids"].ndim != 1 or data["ids"].dtype not in (torch.int32, torch.int64):
        raise ValueError("Cache IDs must be a 1D int32/int64 tensor")
    # vLLM target captures may use int32.
    # Canonicalize without changing token identity or accepting floating IDs.
    data["ids"] = data["ids"].long()
    if data["prompt_length"] != record["prompt_length"]:
        raise ValueError("Prompt boundary differs from manifest")
    if data["teacher_hidden"].shape != (length, width):
        raise ValueError("Invalid teacher hidden shape")
    if (
        data["features"].ndim != 2  # noqa: PLR2004
        or data["features"].shape[0] != length
        or data["features"].shape[1] % width
    ):
        raise ValueError("Invalid concatenated target feature shape")
    for key in ("features", "teacher_hidden"):
        if data[key].dtype != torch.bfloat16:
            raise ValueError("Cache features must be finite BF16 tensors")
    return data


def load_batch(records, width, device):
    with ThreadPoolExecutor(max_workers=min(4, len(records))) as pool:
        rows = list(pool.map(lambda r: load_row(r, width), records))
    length = max(len(x["ids"]) for x in rows)
    keys = ["ids", "features", "teacher_hidden"]
    batch: dict[str, Any] = {}
    for key in keys:
        value = torch.zeros(
            (len(rows), length, *rows[0][key].shape[1:]), dtype=rows[0][key].dtype
        )
        for i, row in enumerate(rows):
            value[i, : len(row[key])] = row[key]
        batch[key] = value.to(device)
    batch["lengths"] = [len(x["ids"]) for x in rows]
    batch["prompt_lengths"] = [x["prompt_length"] for x in rows]
    return batch


def choose_anchors(data, count, rng=None):
    rows = []
    for start, length in zip(data["prompt_lengths"], data["lengths"], strict=True):
        # Keep the seven-token evaluation window used by the original dense run.
        end = length - 8
        if end < start:
            raise ValueError("Transcript too short")
        if rng is None:
            row = torch.linspace(start, end, steps=count).long().tolist()
        else:
            row = [rng.randrange(start, end + 1) for _ in range(count)]
        rows.append(row)
    return torch.tensor(rows, device=data["ids"].device)


def predictions(model, data, anchors):
    rows = torch.arange(anchors.shape[0], device=anchors.device)[:, None]
    first = data["ids"][rows, anchors]
    h = model.backbone(data["features"], data["ids"], anchors)
    index = anchors[:, :, None] + torch.arange(7, device=anchors.device)
    return h, index, first.reshape(-1)
