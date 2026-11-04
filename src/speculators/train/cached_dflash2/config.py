"""Strict YAML/CLI configuration for one cached DFlash2 training arm."""

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class CachedDFlash2Config(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dense_checkpoint: Path
    shared: Path
    manifest: Path
    output: Path
    arm: Literal["dense", "token", "fusion"] = "dense"
    fusion_init: Literal["replace", "residual-zero"] = "residual-zero"
    objective: Literal["kl", "ce"] = "kl"
    lr: float = Field(default=3e-5, gt=0)
    device: str = "cuda"
    epochs: int = Field(default=5, gt=0)
    warmup_fraction: float = Field(default=0.2, ge=0, lt=1)
    batch_sequences: int = Field(default=32, gt=0)
    context_token_budget: int = Field(default=48000, gt=0)
    anchors: int = Field(default=64, gt=0)
    eval_batch_sequences: int = Field(default=4, gt=0)
    eval_anchors: int = Field(default=8, gt=0)
    seed: int = 280
    memory_limit_gib: float = Field(default=96, gt=0)
    cache_target_precision: Literal["fp8", "bf16"] = "fp8"
    resume: bool = False
    check_data: bool = False
    synthetic_smoke_test: bool = False
    pilot_steps: int = Field(default=0, ge=0)
    pilot_val_images: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def validate_run(self):
        if self.pilot_val_images and not self.pilot_steps:
            raise ValueError("Reduced validation is allowed only in an explicit pilot")
        if self.pilot_steps and self.resume:
            raise ValueError("Pilots cannot resume; use a new pilot output directory")
        if not re.fullmatch(r"cpu|cuda(?::\d+)?", self.device):
            raise ValueError("device must be cpu or cuda[:index]")
        if self.device == "cpu" and not self.synthetic_smoke_test:
            raise ValueError("CPU execution is only for explicit synthetic smoke tests")
        return self

    def save(self, path: Path) -> None:
        path.write_text(yaml.safe_dump(self.model_dump(mode="json"), sort_keys=True))


def resolve(argv: list[str] | None = None) -> CachedDFlash2Config:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="YAML file; CLI values override it")
    for name, field in CachedDFlash2Config.model_fields.items():
        annotation = field.annotation
        kwargs: dict[str, Any] = {"default": argparse.SUPPRESS}
        if annotation is bool:
            kwargs["action"] = argparse.BooleanOptionalAction
        else:
            kwargs["type"] = annotation if annotation in (int, float, Path) else str
        parser.add_argument("--" + name.replace("_", "-"), **kwargs)
    supplied = vars(parser.parse_args(argv))
    config = supplied.pop("config")
    values = {}
    if config:
        expanded = os.path.expandvars(config.read_text())
        if re.search(r"\$\{[A-Za-z_][A-Za-z0-9_]*\}", expanded):
            raise ValueError("Set all environment variables referenced by the YAML")
        values = yaml.safe_load(expanded)
        if not isinstance(values, dict):
            raise ValueError("Expected a YAML mapping")
        for name in ("dense_checkpoint", "shared", "manifest", "output"):
            if name in values:
                path = Path(values[name]).expanduser()
                values[name] = (
                    path if path.is_absolute() else config.resolve().parent / path
                )
    values.update(supplied)
    result = CachedDFlash2Config.model_validate(values)
    for name in ("dense_checkpoint", "shared", "manifest", "output"):
        setattr(result, name, getattr(result, name).expanduser().resolve())
    return result


def main(argv: list[str] | None = None) -> None:
    # Keep --help and configuration validation independent of CUDA/model loading.
    config = resolve(argv)
    if config.check_data:
        from .data import preflight  # noqa: PLC0415

        print(json.dumps(preflight(config), indent=2))  # noqa: T201
    else:
        from .trainer import train  # noqa: PLC0415

        train(config)
