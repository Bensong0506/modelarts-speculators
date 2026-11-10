# Historical-result audits intentionally use asserts and fixed experiment sizes.
# ruff: noqa: INP001, S101, PT018, PLR2004, T201
"""Render completed training logs and matched online comparisons as local artifacts."""

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import numpy as np

mpl.use("Agg")
import matplotlib.pyplot as plt

EXP = Path(__file__).resolve().parent
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--training-only", action="store_true")
args = parser.parse_args()
out = EXP / "plots"
out.mkdir(exist_ok=True)
colors = {"dense": "#475569", "token": "#2563eb", "fusion": "#e47d17"}
labels = {
    "dense": "Dense",
    "token": "D-MTP token-only",
    "fusion": "D-MTP hidden fusion",
}
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
for arm in ("dense", "token", "fusion"):
    path = EXP / "training" / arm
    rows = [json.loads(s) for s in (path / "train.jsonl").read_text().splitlines()]
    assert len(rows) == 2815 and [r["step"] for r in rows] == list(range(1, 2816))
    loss = np.array([r["loss"] for r in rows])
    assert np.isfinite(loss).all()
    smooth = np.array([loss[max(0, i - 49) : i + 1].mean() for i in range(len(loss))])
    ax = axes[0 if arm == "dense" else 1]
    ax.plot(np.arange(1, 2816), loss, alpha=0.1, lw=0.5, color=colors[arm])
    ax.plot(np.arange(1, 2816), smooth, color=colors[arm], label=labels[arm])
for i, ax in enumerate(axes):
    ax.set(
        xlabel="Optimizer updates",
        ylabel="KL training loss",
        title=[
            "Dense reference: seven predicted tokens",
            "D-MTP: six suffix tokens, gold first-token input",
        ][i],
    )
    ax.legend()
    ax.grid(alpha=0.2)
fig.suptitle("Matched data and training budget; trailing mean over 50 updates")
fig.text(
    0.5,
    0.01,
    "Prediction contracts differ between panels; "
    "lower training loss alone does not establish better online acceptance.",
    ha="center",
    fontsize=9,
)
fig.tight_layout(rect=(0, 0.045, 1, 0.96))
for ext in ("png", "pdf"):
    fig.savefig(out / f"loss_curves.{ext}", dpi=180)
plt.close(fig)
if args.training_only:
    print(out / "loss_curves.png")
    raise SystemExit(0)

report = json.loads((EXP / "results/online-comparison.json").read_text())
assert report["completed"]
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
for arm in ("dense", "token", "fusion"):
    rows = sorted(
        (r for r in report["rows"] if r["arm"] == arm), key=lambda r: r["epoch"]
    )
    epochs = np.array([r["epoch"] for r in rows])
    al = np.array([r["accepted_length"] for r in rows])
    axes[0].plot(epochs, al, "o-", label=labels[arm], color=colors[arm])
    if arm != "dense":
        delta = np.array([r["vs_same_epoch_dense"]["al_delta"] for r in rows])
        intervals = np.array(
            [r["vs_same_epoch_dense"]["paired_image_al_delta_ci95"] for r in rows]
        )
        trained = epochs > 0
        axes[1].plot(
            epochs[trained], delta[trained], "o-", label=labels[arm], color=colors[arm]
        )
        axes[1].fill_between(
            epochs[trained],
            intervals[trained, 0],
            intervals[trained, 1],
            alpha=0.13,
            color=colors[arm],
        )
axes[0].set(
    ylabel="Accepted length, including bonus", title="100-image online validation"
)
axes[1].axhline(0, color=colors["dense"], ls="--", lw=1)
axes[1].set(
    ylabel="AL difference versus same-epoch dense",
    title="Trained checkpoints: paired-image 95% intervals",
)
for ax in axes:
    ax.set(xlabel="Completed training epochs", xticks=range(6))
    ax.grid(alpha=0.2)
    ax.legend()
axes[1].set_xticks(range(1, 6))
fig.suptitle(
    "FP8 target / BF16 draft | C64 max | K7 | max1024 | same GPU within each epoch"
)
fig.text(
    0.5,
    0.01,
    "Single training seed, validation set only. "
    "Target-only output equivalence remains unresolved.",
    ha="center",
    fontsize=9,
)
fig.tight_layout(rect=(0, 0.045, 1, 0.96))
for ext in ("png", "pdf"):
    fig.savefig(out / f"online_acceptance.{ext}", dpi=180)
plt.close(fig)
print(out)
