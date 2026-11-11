"""Controlled warm-start recipes; validation numbers are cached-prefix proxies.

Seven proposals, one verifier bonus token. Online vLLM confirmation is separate.
This trainer never updates the target, shared embedding/head, or selector.
"""

import fcntl
import hashlib
import json
import random
import sys
import time
from pathlib import Path

import torch
from safetensors.torch import save_file

from . import data as cache
from .common import (
    MIN_RESPONSE_TOKENS,
    atomic_json,
    digest_json,
    epoch_plan,
    file_sha256,
    linear_lr,
    prefix_summary,
)
from .config import CachedDFlash2Config
from .dcore import DCoreDraft, dcore_loss
from .loss import frozen_head_loss, head_candidates

FROZEN = ("embed_tokens.", "lm_head.", "candidate_selector.")


def selector_hash(model):
    h = hashlib.sha256()
    for name, value in sorted(model.candidate_selector.state_dict().items()):
        h.update(name.encode())
        h.update(value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def selected_path(model, hidden, candidates, unary, first):
    prev, chosen = first, []
    for j in range(hidden.shape[1]):
        gate = model.candidate_selector.predecessor_codebook[
            prev
        ] * model.candidate_selector.hidden_projection(hidden[:, j])
        scores = unary[:, j] + (
            gate[:, None]
            * model.candidate_selector.successor_codebook[candidates[:, j]]
        ).sum(-1)
        prev = candidates[:, j].gather(-1, scores.argmax(-1, keepdim=True)).squeeze(-1)
        chosen.append(prev)
    return torch.stack(chosen, 1)


@torch.no_grad()
def evaluate(model, records, args):
    model.eval()
    counters = {
        f"{source}_{path}": [
            torch.zeros(7, dtype=torch.long, device=args.device),
            torch.zeros(7, dtype=torch.long, device=args.device),
        ]
        for source in ("teacher", "transcript")
        for path in ("selected", "unary")
    }
    blocks, agreement, slots = 0, 0, 0
    for _, batch in epoch_plan(
        records, 1, args.eval_batch_sequences, args.context_token_budget, 0
    ):
        data = cache.load_batch(batch, model.config["hidden_size"], args.device)
        anchors = cache.choose_anchors(data, args.eval_anchors)
        row = torch.arange(len(batch), device=args.device)[:, None, None]
        with torch.autocast(args.device.split(":")[0], dtype=torch.bfloat16):
            hidden, index, first = cache.predictions(model, data, anchors)
            teacher = data["teacher_hidden"][row, index].reshape_as(hidden)
            candidates, logits, teacher_ids = head_candidates(
                hidden, teacher, model.lm_head.weight
            )
            selected = selected_path(model, hidden, candidates, logits, first)
            unary = candidates[:, :, 0]
        transcript = data["ids"][
            row, anchors[:, :, None] + torch.arange(1, 8, device=args.device)
        ].reshape(-1, 7)
        for source, gold in (("teacher", teacher_ids), ("transcript", transcript)):
            for path, chosen in (("selected", selected), ("unary", unary)):
                matched = chosen == gold
                counters[f"{source}_{path}"][0] += matched.long().cumprod(-1).sum(0)
                counters[f"{source}_{path}"][1] += matched.sum(0)
        blocks += len(selected)
        agreement += int((teacher_ids == transcript).sum())
        slots += transcript.numel()
    result = {
        name: prefix_summary(prefix.tolist(), blocks, independent.tolist())
        for name, (prefix, independent) in counters.items()
    }
    result.update(
        metric="offline_cached_prefix_proxy",
        validation_images=len(records),
        anchors_per_image=args.eval_anchors,
        teacher_transcript_agreement=agreement / slots,
        online_validated=False,
        limitation=(
            "Later cached teacher labels follow the saved transcript, "
            "not a regenerated divergent teacher prefix."
        ),
    )
    return result


def export_epoch(model, output, epoch):
    dest = output / "epochs" / f"epoch-{epoch:02d}"
    temporary = dest.with_name(dest.name + ".tmp")
    temporary.mkdir(parents=True, exist_ok=True)
    state = {
        k: v.detach().to(device="cpu", dtype=torch.bfloat16).contiguous()
        for k, v in model.state_dict().items()
        if not k.startswith(("embed_tokens.", "lm_head."))
    }
    config = json.loads(json.dumps(model.config))
    save_file(state, str(temporary / "model.safetensors"), metadata={"format": "pt"})
    atomic_json(temporary / "config.json", config)
    atomic_json(
        temporary / "export.json",
        {
            "epoch": epoch,
            "source_run": str(output),
            "safetensors_sha256": file_sha256(temporary / "model.safetensors"),
            "online_validated": False,
        },
    )
    if dest.exists():
        # Resume may repeat validation/export after the last durable optimizer state.
        for p in temporary.iterdir():
            p.replace(dest / p.name)
        temporary.rmdir()
    else:
        temporary.rename(dest)
    return dest


def save_resume(output, model, optimizer, step, epoch, seen, rng, history):
    state = {
        k: v.detach().cpu()
        for k, v in model.state_dict().items()
        if not k.startswith(FROZEN)
    }
    opt = optimizer.state_dict()
    opt["state"] = {
        k: {n: v.cpu() if isinstance(v, torch.Tensor) else v for n, v in s.items()}
        for k, s in opt["state"].items()
    }
    value = {
        "model": state,
        "optimizer": opt,
        "step": step,
        "epoch": epoch,
        "seen": seen,
        "rng": rng.getstate(),
        "torch_rng": torch.get_rng_state(),
        "history": history,
        "cuda_rng": torch.cuda.get_rng_state(next(model.parameters()).device)
        if next(model.parameters()).is_cuda
        else None,
    }
    torch.save(value, output / "resume.tmp.pt")
    (output / "resume.tmp.pt").replace(output / "resume.pt")


def train(a: CachedDFlash2Config):  # noqa: C901
    """Train one arm; exports are BF16, durable optimizer state is FP32."""
    a.output.mkdir(parents=True, exist_ok=a.resume)
    lock = (a.output / ".lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if a.device.startswith("cuda"):
        torch.cuda.set_device(torch.device(a.device).index or 0)
        free, total_memory = torch.cuda.mem_get_info()
        if free < (a.memory_limit_gib + 16) * 2**30:
            raise RuntimeError(f"Insufficient free memory: {free / 2**30:.1f} GiB")
        torch.cuda.set_per_process_memory_fraction(
            a.memory_limit_gib * 2**30 / total_memory
        )
    torch.set_num_threads(8)
    torch.manual_seed(a.seed)
    rng = random.Random(a.seed)
    records = cache.read_manifest(a.manifest, a.synthetic_smoke_test)
    synthetic = [r["image_sha256"].startswith("synthetic-") for r in records]
    if any(synthetic) and (not a.synthetic_smoke_test or not all(synthetic)):
        raise ValueError(
            "Synthetic fixtures require explicit mode; cannot mix with real data"
        )
    train = [
        r
        for r in records
        if r["split"] == "train"
        and r["tokens"] - r["prompt_length"] >= MIN_RESPONSE_TOKENS
    ]
    val = [
        r
        for r in records
        if r["split"] == "val"
        and r["tokens"] - r["prompt_length"] >= MIN_RESPONSE_TOKENS
    ]
    if (
        not train
        or not val
        or {r["image_sha256"] for r in train} & {r["image_sha256"] for r in val}
    ):
        raise ValueError("Nonempty disjoint train and validation splits required")
    if a.pilot_val_images:
        if not a.pilot_steps:
            raise ValueError("Small validation is only allowed in a marked pilot")
        val = val[: a.pilot_val_images]
    plan = epoch_plan(
        train, a.epochs, a.batch_sequences, a.context_token_budget, a.seed
    )
    epoch_steps = [sum(ep == e for ep, _ in plan) for e in range(1, a.epochs + 1)]
    total = len(plan)
    warmup = int(total * a.warmup_fraction)
    model = DCoreDraft.load_feature(
        a.dense_checkpoint, a.shared, a.device, a.dcore_context
    )
    params = []
    for name, parameter in model.named_parameters():
        trainable = not name.startswith(FROZEN)
        parameter.requires_grad_(trainable)
        if trainable:
            parameter.data = parameter.data.float()
            params.append(parameter)
    frozen_hash = selector_hash(model)
    optimizer = torch.optim.AdamW(params, lr=a.lr, betas=(0.9, 0.95), weight_decay=0.01)
    args_record = {
        k: str(v) if isinstance(v, Path) else v
        for k, v in vars(a).items()
        if k not in ("resume", "check_data")
    }
    provenance = {
        "args": args_record,
        "manifest_sha256": file_sha256(a.manifest),
        "initial_weights_sha256": file_sha256(a.dense_checkpoint / "model.safetensors"),
        "initial_config_sha256": file_sha256(a.dense_checkpoint / "config.json"),
        "shared_weights_sha256": file_sha256(a.shared),
        "selector_sha256": frozen_hash,
        "train_images": len(train),
        "validation_images": len(val),
        "excluded_short": [
            r["id"]
            for r in records
            if r["split"] in ("train", "val")
            and r["tokens"] - r["prompt_length"] < MIN_RESPONSE_TOKENS
        ],
        "epoch_steps": epoch_steps,
        "total_steps": total,
        "warmup_steps": warmup,
        "examples_sha256": digest_json(
            [
                {
                    k: r[k]
                    for k in ("id", "split", "image_sha256", "tokens", "prompt_length")
                }
                for r in train + val
            ]
        ),
        "batch_plan_sha256": digest_json(
            [(ep, [r["id"] for r in batch]) for ep, batch in plan]
        ),
        "source_sha256": {
            path.name: file_sha256(path)
            for path in sorted(Path(__file__).parent.glob("*.py"))
        },
        "torch_version": str(torch.__version__),
        "cuda_version": torch.version.cuda,
        "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
        "trainable_parameters": sum(x.numel() for x in params),
        "frozen_components": list(FROZEN),
        "ce_target": ("transcript IDs for D-PACE; cached teacher argmax for CE"),
        "precision": (
            "FP32 master/Adam; BF16 autocast, frozen head and cached teacher features"
        ),
        "loss_weights": "detached D-PACE suffix-confidence, sum normalized"
        if a.objective == "dpace"
        else "exp(-slot/4), mean one",
        "selection": (
            "validation teacher_selected prefix AL; vLLM confirmation required"
        ),
        "online_validated": False,
        "pilot": bool(a.pilot_steps),
    }
    if a.resume:
        previous = json.loads((a.output / "provenance.json").read_text())
        if previous != provenance:
            raise ValueError("Resume provenance changed")
    else:
        atomic_json(a.output / "provenance.json", provenance)
        from speculators.train.utils import save_train_command  # noqa: PLC0415

        save_train_command(str(a.output), argv=sys.argv)
        a.save(a.output / "run.yaml")
    start_step, seen, history = 0, 0, []
    if a.resume and (a.output / "resume.pt").exists():
        saved = torch.load(
            a.output / "resume.pt", map_location=a.device, weights_only=False
        )
        missing, extra = model.load_state_dict(saved["model"], strict=False)
        if extra or any(not k.startswith(FROZEN) for k in missing):
            raise ValueError("Resume model incomplete")
        optimizer.load_state_dict(saved["optimizer"])
        rng.setstate(saved["rng"])
        torch.set_rng_state(saved["torch_rng"].cpu())
        if a.device.startswith("cuda") and saved["cuda_rng"] is not None:
            torch.cuda.set_rng_state(saved["cuda_rng"].cpu(), a.device)
        start_step, seen, history = saved["step"], saved["seen"], saved["history"]
        del saved
    else:
        initial = evaluate(model, val, a)
        atomic_json(a.output / "initial.json", initial)
        print(json.dumps(dict(event="initial", **initial)), flush=True)  # noqa: T201
        if not a.pilot_steps:
            export_epoch(model, a.output, 0)
    from .checkpoint import reconcile_logs  # noqa: PLC0415

    reconcile_logs(a.output, start_step, history)
    started = time.monotonic()
    with (a.output / "train.jsonl").open("a", buffering=1) as log:
        for index in range(start_step, min(total, a.pilot_steps or total)):
            step = index + 1
            epoch, batch = plan[index]
            model.train()
            data = cache.load_batch(batch, model.config["hidden_size"], a.device)
            anchors = cache.choose_anchors(data, a.anchors, rng)
            rows = torch.arange(len(batch), device=a.device)[:, None, None]
            lr = linear_lr(step, total, warmup, a.lr)
            for group in optimizer.param_groups:
                group["lr"] = lr
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(a.device.split(":")[0], dtype=torch.bfloat16):
                hidden, teacher_index, _ = cache.predictions(model, data, anchors)
                weights = torch.exp(-torch.arange(hidden.shape[1], device=a.device) / 4)
                weights /= weights.mean()
                teacher = data["teacher_hidden"][rows, teacher_index].reshape_as(hidden)
                if a.objective == "dpace":
                    labels = data["ids"][rows, teacher_index + 1].reshape(
                        hidden.shape[:-1]
                    )
                    loss = dcore_loss(
                        hidden, labels, model.lm_head.weight, a.dpace_alpha
                    )
                else:
                    loss, _, _ = frozen_head_loss(
                        hidden, teacher, model.lm_head.weight, weights, a.objective
                    )
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite training loss")
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(params, 1.0, error_if_nonfinite=True)
            optimizer.step()
            seen += len(batch)
            metrics = {
                "step": step,
                "epoch": epoch,
                "loss": float(loss),
                "lr": lr,
                "grad_norm": float(norm),
                "images_seen": seen,
                "anchors_seen": seen * a.anchors,
                "seconds": time.monotonic() - started,
                "supervised_slots": seen * a.anchors * hidden.shape[1],
                "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30
                if a.device.startswith("cuda")
                else None,
            }
            log.write(json.dumps(metrics) + "\n")
            if step == 1 or step % 10 == 0:
                print(json.dumps(metrics), flush=True)  # noqa: T201
                atomic_json(
                    a.output / "progress.json", dict(state="training", **metrics)
                )
            # Free activations before full validation/checkpoint CPU copies.
            del hidden, teacher, data, loss
            at_epoch_end = step == total or plan[step][0] != epoch
            if at_epoch_end or (a.pilot_steps and step == a.pilot_steps):
                measured = evaluate(model, val, a)
                measured.update(
                    epoch=epoch,
                    step=step,
                    images_seen=seen,
                    full_epoch_completed=at_epoch_end,
                    pilot=bool(a.pilot_steps),
                )
                if selector_hash(model) != frozen_hash:
                    raise RuntimeError("Frozen selector changed")
                history.append(measured)
                atomic_json(a.output / "validation.json", history)
                print(json.dumps(dict(event="validation", **measured)), flush=True)  # noqa: T201
                if not a.pilot_steps:
                    export_epoch(model, a.output, epoch)
                    save_resume(
                        a.output, model, optimizer, step, epoch, seen, rng, history
                    )
                atomic_json(
                    a.output / "progress.json",
                    dict(
                        state="epoch_completed" if at_epoch_end else "pilot_completed",
                        **metrics,
                    ),
                )
    initial = json.loads((a.output / "initial.json").read_text())
    choices = [dict(epoch=0, step=0, **initial)] + history
    best = max(choices, key=lambda v: v["teacher_selected"]["accepted_length"])
    summary = {
        "completed": not a.pilot_steps,
        "pilot_completed": bool(a.pilot_steps),
        "arm": a.arm,
        "epochs": a.epochs,
        "total_steps": total,
        "images_seen": seen,
        "best_epoch": best["epoch"],
        "best_validation": best,
        "selector_unchanged": True,
        "online_validated": False,
        "selection": (
            "offline validation teacher_selected AL; "
            "confirm with real vLLM before recipe winner"
        ),
    }
    atomic_json(a.output / "summary.json", summary)
    print(json.dumps(dict(event="finished", **summary)), flush=True)  # noqa: T201
    fcntl.flock(lock, fcntl.LOCK_UN)
    lock.close()
    return summary
