"""Read aligned target/MTP caches; keep teacher-forced and native T1 separate."""

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import torch

from .common import MIN_RESPONSE_TOKENS, epoch_plan, file_sha256
from .dmtp import DMTPDraft

ALIGNMENT = "pending_A_at_a__target_HC_at_a-1__predict_T1_at_a+1"


def read_manifest(path, needs_mtp, synthetic_smoke_test=False):  # noqa: C901
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
        for key in ("path", "mtp_path") if needs_mtp else ("path",):
            if key not in record:
                raise ValueError(
                    f"Missing {key}; native MTP sidecars must be captured first"
                )
            cache_path = Path(record[key]).expanduser()
            if not cache_path.is_absolute():
                cache_path = path.resolve().parent / cache_path
            if not cache_path.is_file():
                raise FileNotFoundError(cache_path)
            record[key] = str(cache_path)
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
    records = read_manifest(
        config.manifest, config.arm != "dense", config.synthetic_smoke_test
    )
    for record in records:
        data = load_row(record, config.arm != "dense", width)
        keys = ["features", "teacher_hidden"]
        if config.arm != "dense":
            keys.append("mtp_sample_hidden")
        if any(not torch.isfinite(data[key]).all() for key in keys):
            raise ValueError("Cache contains nonfinite hidden states")
        if data["features"].shape[1] != layers * width:
            raise ValueError(
                "Target feature width differs from checkpoint target layers"
            )
        if (
            record["tokens"] - record["prompt_length"] >= MIN_RESPONSE_TOKENS
            and config.arm != "dense"
            and not data["mtp_valid"][
                record["prompt_length"] : record["tokens"] - 7
            ].all()
        ):
            raise ValueError("Incomplete native MTP anchors")
        for key, checksum in (("path", "sha256"), ("mtp_path", "mtp_sha256")):
            if (
                checksum in record
                and (key != "mtp_path" or config.arm != "dense")
                and file_sha256(record[key]) != record[checksum]
            ):
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


def load_row(record, needs_mtp, width):  # noqa: C901
    data = torch.load(record["path"], map_location="cpu", weights_only=True)
    if (
        len(data["ids"]) != record["tokens"]
        or data["image_sha256"] != record["image_sha256"]
    ):
        raise ValueError("Target cache and manifest mismatch")
    length = len(data["ids"])
    if data["ids"].ndim != 1 or data["ids"].dtype != torch.long:
        raise ValueError("Cache IDs must be a 1D int64 tensor")
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
    if needs_mtp:
        side = torch.load(record["mtp_path"], map_location="cpu", weights_only=True)
        if (
            side.get("alignment") != ALIGNMENT
            or side.get("history") != "native_causal_target_aligned"
        ):
            raise ValueError("Missing audited native MTP timing/history contract")
        if not torch.equal(side["ids"], data["ids"]):
            raise ValueError("MTP and target transcripts differ")
        if side["image_sha256"] != record["image_sha256"]:
            raise ValueError("MTP and target image identities differ")
        if side["mtp_sample_hidden"].shape != (len(data["ids"]), width):
            raise ValueError(
                "MTP must supply the single-stream LM-head input, not HC4 feedback"
            )
        if (
            side["mtp_token"].shape != (length,)
            or side["mtp_token"].dtype != torch.long
        ):
            raise ValueError("Native MTP tokens must be a 1D int64 tensor")
        if (
            side["mtp_valid"].shape != (length,)
            or side["mtp_valid"].dtype != torch.bool
        ):
            raise ValueError("Native MTP validity mask must be a 1D bool tensor")
        if side["mtp_sample_hidden"].dtype != torch.bfloat16:
            raise ValueError("Native MTP hidden must be finite BF16")
        for key in ("mtp_sample_hidden", "mtp_token", "mtp_valid"):
            data[key] = side[key]
    return data


def load_batch(records, needs_mtp, width, device):
    with ThreadPoolExecutor(max_workers=min(4, len(records))) as pool:
        rows = list(pool.map(lambda r: load_row(r, needs_mtp, width), records))
    length = max(len(x["ids"]) for x in rows)
    keys = ["ids", "features", "teacher_hidden"]
    if needs_mtp:
        keys += ["mtp_sample_hidden", "mtp_token", "mtp_valid"]
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
        # Keep exactly the prior seven-token evaluation window for every arm.
        end = length - 8
        if end < start:
            raise ValueError("Transcript too short")
        if rng is None:
            row = torch.linspace(start, end, steps=count).long().tolist()
        else:
            row = [rng.randrange(start, end + 1) for _ in range(count)]
        rows.append(row)
    a = torch.tensor(rows, device=data["ids"].device)
    if "mtp_valid" in data and not data["mtp_valid"].gather(1, a).all():
        raise ValueError(
            "Native MTP capture incomplete at selected anchors; "
            "do not silently change the sample"
        )
    return a


def predictions(model, data, anchors, training):
    rows = torch.arange(anchors.shape[0], device=anchors.device)[:, None]
    if isinstance(model, DMTPDraft):
        first = (
            data["ids"][rows, anchors + 1]
            if training
            else data["mtp_token"][rows, anchors]
        )
        mtp = data["mtp_sample_hidden"][rows, anchors]
        h = model.backbone(data["features"], data["ids"], anchors, mtp, first)
        index = anchors[:, :, None] + 1 + torch.arange(6, device=anchors.device)
    else:
        first = data["ids"][rows, anchors]
        h = model.backbone(data["features"], data["ids"], anchors)
        index = anchors[:, :, None] + torch.arange(7, device=anchors.device)
    return h, index, first.reshape(-1)
