"""CLI adapter for image-identity-checked target/MTP caches."""

import typer


def train_cached_dflash2(ctx: typer.Context):
    from speculators.train.cached_dflash2.config import main  # noqa: PLC0415

    main(ctx.args)
