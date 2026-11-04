"""Compare full generated token sequences on identical held-out images."""

import argparse
import json
from pathlib import Path


def load(path, repeat):
    records = [json.loads(line) for line in path.read_text().splitlines()]
    selected = [r for r in records if r["repeat"] == repeat]
    result = {r["id"]: r for r in selected}
    if not result or len(result) != len(selected):
        raise ValueError("Empty repeat or duplicate request IDs")
    return result


def compare(reference, candidate):
    if reference.keys() != candidate.keys():
        raise ValueError("Evaluation request IDs differ")
    changed = []
    for identity, first in reference.items():
        second = candidate[identity]
        if first["image_sha256"] != second["image_sha256"]:
            raise ValueError("Evaluation image identities differ")
        a, b = first["token_ids"], second["token_ids"]
        if a != b or first["finish_reason"] != second["finish_reason"]:
            first_difference = next(
                (i for i, (x, y) in enumerate(zip(a, b, strict=False)) if x != y),
                min(len(a), len(b)),
            )
            changed.append(
                {
                    "id": identity,
                    "first_difference": first_difference,
                    "reference_length": len(a),
                    "candidate_length": len(b),
                }
            )
    return {
        "requests": len(reference),
        "changed": len(changed),
        "identical_on_these_requests": not changed,
        "differences": changed,
        "scope": (
            "Exact token IDs and finish reasons on supplied requests; "
            "not a general losslessness proof."
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--reference-repeat", type=int, default=0)
    parser.add_argument("--candidate-repeat", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = compare(
        load(args.reference, args.reference_repeat),
        load(args.candidate, args.candidate_repeat),
    )
    with args.output.open("x") as stream:
        stream.write(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "differences"}))
    raise SystemExit(bool(report["changed"]))


if __name__ == "__main__":
    main()
