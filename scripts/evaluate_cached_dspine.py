"""Evaluate an exported cached dspine model on saved prefixes; no serving claim."""

import argparse
import json
from pathlib import Path

import yaml

from speculators.train.cached_dflash2.common import atomic_json, file_sha256
from speculators.train.cached_dflash2.config import CachedDFlash2Config
from speculators.train.cached_dflash2.data import read_manifest
from speculators.train.cached_dflash2.dspine import DSpineDraft
from speculators.train.cached_dflash2.trainer import evaluate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    config = CachedDFlash2Config.model_validate(
        yaml.safe_load((args.run / "run.yaml").read_text())
    )
    records = [
        r
        for r in read_manifest(config.manifest)
        if r["split"] == args.split and r["tokens"] - r["prompt_length"] >= 16  # noqa: PLR2004
    ]
    if not records:
        raise ValueError("No held-out rows for requested split")
    model = DSpineDraft.load_export(args.checkpoint, config.shared, config.device)
    result = evaluate(model, records, config)
    result.update(
        split=args.split,
        checkpoint_sha256=file_sha256(args.checkpoint / "model.safetensors"),
    )
    atomic_json(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
