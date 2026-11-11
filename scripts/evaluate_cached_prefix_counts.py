"""Evaluate an exported draft on fixed cached prefixes with per-image counts.

No online generation. Images are represented only by their SHA-256 identity.
"""

import argparse
import json
import shlex
import sys
from datetime import UTC, datetime
from pathlib import Path

import torch
import yaml

from speculators.provenance import (
    find_package_repo,
    git_diff,
    git_sha,
    package_versions,
)
from speculators.train.cached_dflash2 import data as cache
from speculators.train.cached_dflash2.common import (
    MIN_RESPONSE_TOKENS,
    atomic_json,
    epoch_plan,
    file_sha256,
    prefix_summary,
)
from speculators.train.cached_dflash2.config import CachedDFlash2Config
from speculators.train.cached_dflash2.dspine import DSpineDraft
from speculators.train.cached_dflash2.loss import head_candidates
from speculators.train.cached_dflash2.model import Draft
from speculators.train.cached_dflash2.trainer import selected_path


def read_config(args):
    values = yaml.safe_load((args.run / "run.yaml").read_text())
    if (
        args.kind == "dense"
        and "fusion_init" in values
        and values.pop("fusion_init") != "residual-zero"
    ):
        raise ValueError("Unsupported historical dense configuration")
    if args.manifest:
        old = json.loads(Path(values["manifest"]).read_text())
        new = json.loads(args.manifest.read_text())
        keys = ["id", "split", "image_sha256", "tokens", "prompt_length", "sha256"]

        def identity(rows):
            return [
                {k: r[k] for k in keys if k in r} for r in rows if r["split"] == "val"
            ]

        if identity(old) != identity(new):
            raise ValueError("Cache remap changed validation identity/content")
        values["manifest"] = args.manifest
    return CachedDFlash2Config.model_validate(values)


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=["dense", "dspine"], required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    config = read_config(args)
    torch.set_num_threads(8)
    torch.manual_seed(config.seed)
    if args.kind == "dspine":
        model = DSpineDraft.load_export(args.checkpoint, config.shared, config.device)
    else:
        model = Draft.load(args.checkpoint, config.shared, config.device)
    model.eval()
    model.requires_grad_(False)
    records = [
        r
        for r in cache.read_manifest(config.manifest, config.synthetic_smoke_test)
        if r["split"] == "val"
        and r["tokens"] - r["prompt_length"] >= MIN_RESPONSE_TOKENS
    ]
    if not records:
        raise ValueError("No eligible validation records")
    rows_out = []
    for _, batch in epoch_plan(
        records, 1, config.eval_batch_sequences, config.context_token_budget, 0
    ):
        data = cache.load_batch(batch, model.config["hidden_size"], config.device)
        anchors = cache.choose_anchors(data, config.eval_anchors)
        row = torch.arange(len(batch), device=config.device)[:, None, None]
        with torch.autocast(config.device.split(":")[0], dtype=torch.bfloat16):
            hidden, index, first = cache.predictions(model, data, anchors)
            teacher = data["teacher_hidden"][row, index].reshape_as(hidden)
            candidates, logits, teacher_ids = head_candidates(
                hidden, teacher, model.lm_head.weight
            )
            selected = selected_path(model, hidden, candidates, logits, first)
            unary = candidates[:, :, 0]
        transcript = data["ids"][
            row, anchors[:, :, None] + torch.arange(1, 8, device=config.device)
        ].reshape(-1, 7)
        metrics = {}
        for source, gold in [("teacher", teacher_ids), ("transcript", transcript)]:
            for path, chosen in [("selected", selected), ("unary", unary)]:
                matched = (chosen == gold).long()
                metrics[source + "_" + path] = (
                    matched.cumprod(-1)
                    .reshape(len(batch), config.eval_anchors, 7)
                    .sum(1)
                    .cpu()
                    .tolist(),
                    matched.reshape(len(batch), config.eval_anchors, 7)
                    .sum(1)
                    .cpu()
                    .tolist(),
                )
        agreement = (
            (teacher_ids == transcript).reshape(len(batch), -1).sum(1).cpu().tolist()
        )
        for i, record in enumerate(batch):
            rows_out.append(
                {
                    "image_sha256": record["image_sha256"],
                    "blocks": config.eval_anchors,
                    "teacher_transcript_matches": agreement[i],
                    **{
                        name: {
                            "prefix_counts": value[0][i],
                            "independent_counts": value[1][i],
                        }
                        for name, value in metrics.items()
                    },
                }
            )
        if len(rows_out) % 100 == 0:
            print(json.dumps({"images": len(rows_out)}), flush=True)
    blocks = sum(r["blocks"] for r in rows_out)
    result = {
        name: prefix_summary(
            [sum(r[name]["prefix_counts"][i] for r in rows_out) for i in range(7)],
            blocks,
            [sum(r[name]["independent_counts"][i] for r in rows_out) for i in range(7)],
        )
        for name in metrics
    }
    result.update(
        metric="offline_cached_prefix_proxy",
        validation_images=len(records),
        anchors_per_image=config.eval_anchors,
        teacher_transcript_agreement=sum(
            r["teacher_transcript_matches"] for r in rows_out
        )
        / (7 * blocks),
        online_validated=False,
        checkpoint_sha256=file_sha256(args.checkpoint / "model.safetensors"),
        kind=args.kind,
        precision="reloaded BF16 export",
        rows=rows_out,
    )
    repo = find_package_repo("speculators")
    provenance = {
        "git_sha": git_sha(repo),
        "patch": git_diff(repo),
        "argv": sys.argv,
        "time_utc": datetime.now(UTC).isoformat(),
        "packages": package_versions(),
        "torch_version": str(torch.__version__),
        "cuda_version": torch.version.cuda,
        "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
        "script_sha256": file_sha256(Path(__file__)),
        "manifest_sha256": file_sha256(config.manifest),
    }
    result["provenance"] = provenance
    args.output.with_suffix(".eval_command.txt").write_text(
        "# "
        + provenance["time_utc"]
        + "\n# git_sha: "
        + provenance["git_sha"]
        + "\n"
        + "\n".join(provenance["packages"])
        + "\n"
        + shlex.join([sys.executable, *sys.argv])
        + "\n"
    )
    args.output.with_suffix(".patch").write_text(provenance["patch"])
    atomic_json(args.output, result)
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}), flush=True)


if __name__ == "__main__":
    main()
