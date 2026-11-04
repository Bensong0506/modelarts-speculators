"""Restore logs to the same durable epoch boundary as the optimizer state."""

import json
from pathlib import Path

from .common import atomic_json


def reconcile_logs(output: Path, durable_step: int, history: list) -> None:
    """Keep rolled-back records for diagnosis, never append duplicate step numbers."""
    log = output / "train.jsonl"
    if log.exists():
        lines = log.read_text().splitlines(keepends=True)
        retained, rolled_back = [], []
        expected = 1
        for line in lines:
            try:
                record = json.loads(line)
                step = record["step"]
            except (ValueError, KeyError):
                if expected <= durable_step:
                    raise ValueError(
                        "Corrupt log before durable optimizer checkpoint"
                    ) from None
                rolled_back.append(line)
                continue
            if step <= durable_step:
                if step != expected:
                    raise ValueError("Missing or duplicate durable training step")
                retained.append(line)
                expected += 1
            else:
                rolled_back.append(line)
        if expected - 1 != durable_step:
            raise ValueError("Training log does not reach the durable optimizer step")
        if rolled_back:
            with (output / "rolled-back-train.jsonl").open("a") as backup:
                backup.writelines(rolled_back)
            temporary = log.with_suffix(".jsonl.tmp")
            temporary.write_text("".join(retained))
            temporary.replace(log)
    elif durable_step:
        raise ValueError("Missing training log for resumed optimizer")
    atomic_json(output / "validation.json", history)
