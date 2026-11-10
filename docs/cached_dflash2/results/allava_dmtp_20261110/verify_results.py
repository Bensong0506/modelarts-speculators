# Historical-result audits intentionally use asserts and fixed experiment sizes.
# ruff: noqa: INP001, S101, PT018, PLR2004, T201
"""Recompute this archive's pooled metrics and paired bootstrap from public counters."""

import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
COUNTERS = ("num_accepted_draft_tokens", "num_draft_tokens", "num_spec_steps")


def read_json(path):
    return json.loads(path.read_text())


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def measures(counts):
    assert np.all(counts[..., 1:] > 0)
    return np.stack(
        (100 * counts[..., 0] / counts[..., 1], 1 + counts[..., 0] / counts[..., 2]),
        axis=-1,
    )


def counters(rows):
    return np.array([[r["spec"][k] for k in COUNTERS] for r in rows], dtype=float)


def check_run(epoch, arm, expected):
    run = ROOT / "online" / f"epoch-{epoch:02d}-{arm}"
    summary = read_json(run / "summary.json")
    assert summary["completed"] and summary["mode"] == arm
    assert summary["target_precision_operator_declared"] == "fp8"
    rows = read_rows(run / "request-counters.jsonl")
    assert len(rows) == len({r["id"] for r in rows}) == 100
    assert {r["repeat"] for r in rows} == {0}
    pooled = counters(rows).sum(0)
    np.testing.assert_allclose(
        measures(pooled),
        [100 * expected["acceptance_rate"], expected["accepted_length"]],
        rtol=0,
        atol=1e-10,
    )
    recorded = summary["repeats"][0]
    for key in (
        "acceptance_rate",
        "accepted_length",
        "accepted_draft_tokens",
        "proposed_tokens",
        "verification_steps",
    ):
        assert recorded[key] == expected[key]
    hist = np.array([r["spec"]["acceptance_histogram"] for r in rows]).sum(0)
    assert hist.shape == (8,) and hist.sum() == pooled[2]
    assert hist @ np.arange(8) == pooled[0] and pooled[1] == 7 * pooled[2]
    np.testing.assert_array_equal(hist, expected["acceptance_histogram"])
    kwargs = read_json(run / "engine_kwargs.json")
    assert kwargs == summary["engine_kwargs"]
    assert kwargs.pop("speculative_config")["num_speculative_tokens"] == 7
    kwargs.pop("additional_config", None)
    return rows, summary, kwargs, (run / "checkpoint_sha256.txt").read_text()


def compare(base, candidate, expected):
    assert {r["id"] for r in base} == {r["id"] for r in candidate}
    indexed = {r["id"]: r for r in candidate}
    paired = [indexed[r["id"]] for r in base]
    x, y = counters(base), counters(paired)
    delta = measures(y.sum(0)) - measures(x.sum(0))
    np.testing.assert_allclose(
        delta, [expected["ar_delta_pp"], expected["al_delta"]], rtol=0, atol=1e-10
    )
    rng = np.random.default_rng(20261110)
    draws = []
    for _ in range(20):
        indices = rng.integers(0, 100, (100, 100))
        draws.append(measures(y[indices].sum(1)) - measures(x[indices].sum(1)))
    ci = np.quantile(np.concatenate(draws), [0.025, 0.975], axis=0)
    np.testing.assert_allclose(
        ci[:, 0], expected["paired_image_ar_delta_pp_ci95"], rtol=0, atol=1e-10
    )
    np.testing.assert_allclose(
        ci[:, 1], expected["paired_image_al_delta_ci95"], rtol=0, atol=1e-10
    )
    assert (
        sum(
            a["output_sha256"] != b["output_sha256"]
            for a, b in zip(base, paired, strict=True)
        )
        == expected["changed_outputs"]
    )


def main():
    report = read_json(ROOT / "results/online-comparison.json")
    assert report["completed"] and report["completed_online_runs"] == 18
    assert report["lossless_acceleration_validated"] is False
    expected = {(r["epoch"], r["arm"]): r for r in report["rows"]}
    targets, configs, prompts = set(), set(), set()
    for epoch in range(6):
        base, _, base_kwargs, _ = check_run(epoch, "dense", expected[epoch, "dense"])
        for arm in ("dense", "token", "fusion"):
            rows, summary, kwargs, target = check_run(epoch, arm, expected[epoch, arm])
            assert kwargs == base_kwargs
            targets.add(target)
            configs.add(summary["target_config_canonical_sha256"])
            prompts.add(summary["prompt_manifest_sha256"])
            if arm != "dense":
                compare(base, rows, expected[epoch, arm]["vs_same_epoch_dense"])
    assert len(targets) == len(configs) == len(prompts) == 1
    origins = read_json(ROOT / "provenance/original-artifact-sha256.json")
    cached = {
        (r["arm"], r["epoch"]): r
        for r in read_json(ROOT / "results/cached-comparison.json")["rows"]
    }
    native_t1 = set()
    for arm in ("dense", "token", "fusion"):
        run = ROOT / "training" / arm
        rows = read_rows(run / "train.jsonl")
        assert len(rows) == 2815 and [r["step"] for r in rows] == list(range(1, 2816))
        assert np.isfinite([r["loss"] for r in rows]).all()
        assert rows[-1]["images_seen"] == 89990
        assert (
            hashlib.sha256((run / "train.jsonl").read_bytes()).hexdigest()
            == origins[f"training/{arm}/train.jsonl"]
        )
        summary = read_json(run / "summary.json")
        assert (
            summary["completed"]
            and summary["epochs"] == 5
            and summary["selector_unchanged"]
        )
        values = [read_json(run / "initial.json"), *read_json(run / "validation.json")]
        for epoch, value in enumerate(values):
            metric = value["teacher_selected"]
            assert (
                value["validation_images"] == 1000 and value["anchors_per_image"] == 8
            )
            np.testing.assert_allclose(
                [
                    metric["acceptance_rate"],
                    metric["accepted_length"],
                    metric["t1_acceptance"],
                ],
                [cached[arm, epoch][k] for k in ("ar", "al", "t1")],
                rtol=0,
                atol=1e-10,
            )
            if arm != "dense":
                native_t1.add(metric["t1_acceptance"])
    assert len(native_t1) == 1
    print(
        "Verified: 18 online runs, paired intervals/output digests, "
        "three 2815-update logs, and separate cached metrics."
    )
    print(
        "This rechecks archived counters; it does not rerun generation "
        "or establish lossless output equivalence."
    )


if __name__ == "__main__":
    main()
