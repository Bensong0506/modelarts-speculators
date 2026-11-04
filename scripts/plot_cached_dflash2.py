"""Plot real train.jsonl losses, keeping CE and KL objectives in separate panels."""

import argparse
import json
import math
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--window", type=int, default=50)
    args = parser.parse_args()
    if args.window < 1:
        parser.error("window must be positive")
    import matplotlib as mpl  # noqa: PLC0415

    mpl.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    runs = []
    for path in args.runs:
        provenance = json.loads((path / "provenance.json").read_text())
        rows = [
            json.loads(line) for line in (path / "train.jsonl").read_text().splitlines()
        ]
        if (
            not rows
            or [row["step"] for row in rows] != list(range(1, len(rows) + 1))
            or any(not math.isfinite(row["loss"]) for row in rows)
        ):
            raise ValueError(f"Missing/duplicate/nonfinite training updates: {path}")
        runs.append((path.name, provenance["args"]["objective"], rows))
    objectives = sorted({objective for _, objective, _ in runs})
    fig, axes = plt.subplots(
        1, len(objectives), figsize=(7 * len(objectives), 4), squeeze=False
    )
    for objective, axis in zip(objectives, axes[0], strict=True):
        for name, kind, rows in runs:
            if kind != objective:
                continue
            steps = [row["step"] for row in rows]
            loss = [row["loss"] for row in rows]
            smooth = [
                sum(loss[max(0, i - args.window + 1) : i + 1]) / min(i + 1, args.window)
                for i in range(len(loss))
            ]
            (line,) = axis.plot(steps, smooth, label=name)
            axis.plot(steps, loss, alpha=0.12, color=line.get_color(), linewidth=0.5)
        axis.set(
            xlabel="Optimizer updates",
            ylabel=f"{objective.upper()} loss",
            title=f"Actual losses; trailing mean ({args.window} updates)",
        )
        axis.legend()
        axis.grid(alpha=0.2)
    fig.tight_layout()
    args.output.mkdir(parents=True, exist_ok=False)
    for suffix in ("png", "pdf"):
        fig.savefig(args.output / f"loss_curves.{suffix}", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
