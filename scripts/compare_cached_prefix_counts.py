"""Compare exported-model cached-prefix counts using paired image bootstraps."""

import argparse
import json
from pathlib import Path

import numpy as np


def image_values(result, metric):
    values = {}
    blocks = set()
    accepted = 0
    for row in result["rows"]:
        identity = row["image_sha256"]
        prefix = row[metric]["prefix_counts"]
        count = row["blocks"]
        if identity in values or count <= 0 or len(prefix) != 7:  # noqa: PLR2004
            raise ValueError("Expected unique images, positive blocks, seven slots")
        if any(not 0 <= x <= count for x in prefix) or any(
            b > a for a, b in zip(prefix, prefix[1:], strict=False)
        ):
            raise ValueError("Invalid cumulative prefix counts")
        values[identity] = sum(prefix) / count
        blocks.add(count)
        accepted += sum(prefix)
    if not values or len(blocks) != 1:
        raise ValueError("This image bootstrap requires equal anchors per image")
    if accepted != result[metric]["accepted_draft_tokens"]:
        raise ValueError("Per-image counts disagree with aggregate")
    return values


def compare(current, reference, *, resamples, seed):
    if current.keys() != reference.keys():
        raise ValueError("Paired comparison requires identical image identities")
    delta = np.array([current[k] - reference[k] for k in sorted(current)])
    rng = np.random.default_rng(seed)
    samples = []
    remaining = resamples
    while remaining:
        count = min(remaining, 500)
        indices = rng.integers(0, len(delta), size=(count, len(delta)))
        samples.extend(delta[indices].mean(1).tolist())
        remaining -= count
    low, high = np.quantile(samples, [0.025, 0.975])
    return {
        "AL_difference": float(delta.mean()),
        "AL_95pct_image_bootstrap_CI": [float(low), float(high)],
        "AR_difference_percentage_points": float(delta.mean() / 7 * 100),
        "AR_95pct_CI_percentage_points": [
            float(low / 7 * 100),
            float(high / 7 * 100),
        ],
        "images": len(delta),
        "resamples": resamples,
        "seed": seed,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", nargs=2, action="append", required=True, metavar=("LABEL", "JSON")
    )
    parser.add_argument("--reference", required=True)
    parser.add_argument(
        "--metric",
        default="teacher_selected",
        choices=[
            "teacher_selected",
            "teacher_unary",
            "transcript_selected",
            "transcript_unary",
        ],
    )
    parser.add_argument("--resamples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20261111)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.resamples < 1 or len({name for name, _ in args.input}) != len(args.input):
        raise ValueError("Positive resamples and unique labels required")
    data = {name: json.loads(Path(path).read_text()) for name, path in args.input}
    values = {name: image_values(result, args.metric) for name, result in data.items()}
    if args.reference not in values:
        raise ValueError("Reference must name an input")
    result = {
        "metric": "offline_cached_prefix_proxy",
        "path": args.metric,
        "reference": args.reference,
        "online_validated": False,
        "comparisons": {
            name: compare(
                counts, values[args.reference], resamples=args.resamples, seed=args.seed
            )
            for name, counts in values.items()
            if name != args.reference
        },
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
