#!/usr/bin/env python3
"""research: exact compact 80->92 continuation control.

research's compact_equal_epoch and compact_wordmatched arms shared deterministic order and
masking keys, but they were separate fresh runs and the script did not reset Torch/CUDA
randomness before each arm.  If dropout is active, the first 80 epochs are not guaranteed
to be identical, so the 80-to-92 difference should not be attributed cleanly to extra
recurrence.

This script trains the compact arm once from the coherent86 parent for 80 epochs, saves
model+optimizer+RNG state, then continues the same live state to 92 epochs.  It scores
both checkpoints on the research surface readout so the extra 12 epochs can be interpreted
as a true continuation of the same optimization path.  The saved checkpoints can then be
scored by research's common-target probe.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse
import json
import os
import pathlib
import random
import sys
import time
from typing import Any, Dict, List

import torch
import torch.nn.functional as F

ROOT = _public_path('.')
STUDY = _public_path('experiments/archive/functional_learning')
SCRIPTS = _public_path('experiments/archive/functional_learning/scripts')
sys.path.insert(0, str(SCRIPTS))
import coherent86_continuation_trainer as base_loader  # noqa: E402
import corrected_bridge_trainer as bridge  # noqa: E402
import concentrated_compact_learning_test as research  # noqa: E402

DEFAULT_OUT = _public_path('experiments/archive/functional_learning/data/compact_continuation_control')
DEFAULT_LABELS = _public_path('experiments/archive/functional_learning/data/selective_reviewed_labels/selective_semantic_labels.jsonl')
DEFAULT_STEP57 = _public_path('experiments/archive/functional_learning/data/concentrated_compact_learning_schedulematched')


def rel(p: pathlib.Path | str) -> str:
    try:
        return str(pathlib.Path(p).resolve().relative_to(ROOT))
    except Exception:
        return str(p)


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def reset_all(seed: int) -> None:
    random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def serializable_rng_state(order_rng: random.Random) -> Dict[str, Any]:
    state = {
        "python_random_state_repr": repr(random.getstate()),
        "order_random_state_repr": repr(order_rng.getstate()),
        "torch_rng_state_hex_prefix": torch.get_rng_state().numpy().tobytes()[:64].hex(),
    }
    if torch.cuda.is_available():
        try:
            state["cuda_rng_state_count"] = len(torch.cuda.get_rng_state_all())
            state["cuda_rng_state_hex_prefixes"] = [x.cpu().numpy().tobytes()[:64].hex() for x in torch.cuda.get_rng_state_all()]
        except Exception as e:
            state["cuda_rng_state_error"] = repr(e)
    return state


def save_training_state(path: pathlib.Path, model, optimizer, order_rng: random.Random, epoch: int, args: argparse.Namespace) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "epoch": int(epoch),
        "optimizer_state_dict": optimizer.state_dict(),
        "python_random_state": random.getstate(),
        "order_random_state": order_rng.getstate(),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "args": vars(args),
        "created_utc": now(),
    }
    torch.save(payload, str(path))


def train_to_checkpoints(args: argparse.Namespace) -> None:
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ["HF_HOME"] = str((out_dir / "hf_cache/hf_home").resolve())
    os.environ["TRANSFORMERS_CACHE"] = str((out_dir / "hf_cache/transformers").resolve())
    os.environ["HF_MODULES_CACHE"] = str((out_dir / "hf_cache/modules").resolve())
    (out_dir / "hf_cache/modules").mkdir(parents=True, exist_ok=True)

    tokenizer = bridge.AutoTokenizer.from_pretrained(str(bridge.PARENT_PATH), local_files_only=True, use_fast=True)
    records = research.load_jsonl(pathlib.Path(args.labels))
    plan_args = argparse.Namespace(
        max_pairs=int(args.max_pairs),
        max_length=int(args.max_length),
        max_train_targets_per_row=int(args.max_train_targets_per_row),
        max_eval_targets_per_view=int(args.max_eval_targets_per_view),
        base_epochs=int(args.epoch80),
    )
    plan, _current_ex, compact_ex, eval_tasks = research.build_plan(records, tokenizer, plan_args)
    compact_arm = [a for a in plan["arms"] if a["name"] == "compact_equal_epoch"][0]
    compact_arm = dict(compact_arm)
    compact_arm["epochs"] = int(args.epoch80)
    full_arm = dict(compact_arm)
    full_arm["name"] = "compact_exact_continuation_epoch92"
    full_arm["epochs"] = int(args.epoch92)
    parent_scores_path = _public_path('experiments/archive/functional_learning/data/concentrated_compact_learning_schedulematched/parent_eval_scores.jsonl')
    parent_scores = research.load_jsonl(parent_scores_path)

    plan_out = {
        "status": "COMPACT_CONTINUATION_PLAN",
        "created_utc": now(),
        "scientific_purpose": "Train compact once to 80 epochs and continue the identical state to 92 epochs, saving model, optimizer, and RNG state at epoch80.",
        "input_step57_plan": plan,
        "compact_examples": len(compact_ex),
        "eval_tasks": len(eval_tasks),
        "parent_scores": rel(parent_scores_path),
        "epoch80": int(args.epoch80),
        "epoch92": int(args.epoch92),
        "schedule_key": compact_arm.get("schedule_key", "compact"),
        "interpretation": "Only epoch92_continued minus epoch80 in this run is a clean extra-recurrence contrast; research compact_wordmatched was a descriptive independent 92-epoch run.",
    }
    (out_dir / "plan.json").write_text(json.dumps(plan_out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    research.write_jsonl(out_dir / "compact_training_examples.jsonl", [{k: v for k, v in x.items() if k not in {"input_ids", "attention_mask"}} for x in compact_ex])
    print(json.dumps(plan_out, indent=2, ensure_ascii=False), flush=True)
    if args.dry_run:
        return

    if args.device == "auto":
        device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    elif args.device == "cuda":
        device = torch.device(f"cuda:{args.gpu}")
    else:
        device = torch.device("cpu")
    reset_all(int(args.seed))
    print(json.dumps({"event": "device_selected", "device": str(device), "seed": int(args.seed)}), flush=True)
    model, missing, unexpected = bridge.load_model(device, private_scale=float(args.private_scale))
    ident = bridge.model_identity(model)
    if ident.get("class") != "FrozenSlowPrivateDebertaV2ForMaskedLM" or int(ident.get("private_adapter_params", 0)) != 995584:
        raise RuntimeError(f"bad model identity: {ident}")
    opt, opt_info = bridge.freeze_to_private_optimizer(model, float(args.lr), float(args.weight_decay))
    model.train()
    order_rng = random.Random(research.stable_seed("arm-order", int(args.seed), str(compact_arm.get("schedule_key", "compact"))))

    logs: List[Dict[str, Any]] = []
    checkpoint_summaries: Dict[str, Any] = {}
    t0 = time.time()
    for epoch in range(int(args.epoch92)):
        order = list(range(len(compact_ex)))
        order_rng.shuffle(order)
        epoch_loss_sum = 0.0
        epoch_batches = 0
        epoch_targets = 0
        for start in range(0, len(order), int(args.batch_size)):
            batch = [compact_ex[i] for i in order[start:start + int(args.batch_size)]]
            ids, att, labels, st = research.collate_train(batch, tokenizer, epoch, str(compact_arm.get("schedule_key", "compact")), int(args.seed), float(args.mask_prob), device)
            if int((labels != -100).sum().item()) == 0:
                continue
            out = model(input_ids=ids, attention_mask=att)
            vocab = out.logits.shape[-1]
            loss = F.cross_entropy(out.logits.reshape(-1, vocab), labels.reshape(-1), ignore_index=-100, reduction="mean")
            if torch.isnan(loss) or torch.isinf(loss):
                raise RuntimeError(f"bad loss at epoch {epoch + 1}")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = float(torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], float(args.max_grad_norm)).detach().cpu())
            opt.step()
            epoch_loss_sum += float(loss.detach().cpu())
            epoch_batches += 1
            epoch_targets += int(st["target_tokens"])
            del ids, att, labels, out, loss
        log = {
            "epoch": epoch + 1,
            "mean_loss": epoch_loss_sum / max(1, epoch_batches),
            "batches": epoch_batches,
            "target_tokens": epoch_targets,
            "charged_row_words_cum": (epoch + 1) * int(compact_arm["row_words_per_epoch"]),
            "view_words_cum": (epoch + 1) * int(compact_arm["view_words_per_epoch"]),
            "elapsed_sec": round(time.time() - t0, 1),
        }
        logs.append(log)
        if epoch == 0 or (epoch + 1) % int(args.log_every) == 0 or epoch + 1 in {int(args.epoch80), int(args.epoch92)}:
            print(json.dumps({"event": "train_epoch", **log}, ensure_ascii=False), flush=True)
        if epoch + 1 in {int(args.epoch80), int(args.epoch92)}:
            tag = f"epoch{epoch + 1:04d}"
            model.eval()
            scores = research.score_mask_tasks(model, tokenizer, eval_tasks, device, int(args.eval_batch_size))
            eval_summary = research.summarize_eval(scores, parent_scores)
            ckpt_dir = out_dir / tag / "checkpoint"
            bridge.save_checkpoint(model, tokenizer, ckpt_dir, {"step": "compact_continuation_control", "tag": tag, "arm": full_arm, "model_identity": ident}, float(args.private_scale))
            research.write_jsonl(out_dir / tag / "eval_scores.jsonl", scores)
            (out_dir / tag / "eval_summary.json").write_text(json.dumps(eval_summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            save_training_state(out_dir / tag / "training_state.pt", model, opt, order_rng, epoch + 1, args)
            rng_summary = serializable_rng_state(order_rng)
            checkpoint_summaries[tag] = {
                "epoch": epoch + 1,
                "train_log": log,
                "checkpoint": rel(ckpt_dir),
                "training_state": rel(out_dir / tag / "training_state.pt"),
                "eval_summary": eval_summary,
                "rng_summary": rng_summary,
            }
            model.train()

    research.write_jsonl(out_dir / "train_log.jsonl", logs)
    final = {
        "status": "COMPACT_CONTINUATION_CONTROL_DONE",
        "created_utc": now(),
        "plan": rel(out_dir / "plan.json"),
        "model_identity": ident,
        "optimizer": opt_info,
        "completed_epochs": int(args.epoch92),
        "checkpoint_summaries": checkpoint_summaries,
        "scientific_status": "true compact continuation control; compare epoch0092 against epoch0080 from the same live trajectory before attributing extra recurrence",
    }
    (out_dir / "summary.json").write_text(json.dumps(final, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(final, indent=2, ensure_ascii=False), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--labels", type=pathlib.Path, default=DEFAULT_LABELS)
    ap.add_argument("--out-dir", type=pathlib.Path, default=DEFAULT_OUT)
    ap.add_argument("--epoch80", type=int, default=80)
    ap.add_argument("--epoch92", type=int, default=92)
    ap.add_argument("--max-pairs", type=int, default=0)
    ap.add_argument("--max-length", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--eval-batch-size", type=int, default=64)
    ap.add_argument("--mask-prob", type=float, default=0.35)
    ap.add_argument("--max-train-targets-per-row", type=int, default=16)
    ap.add_argument("--max-eval-targets-per-view", type=int, default=8)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--max-grad-norm", type=float, default=1.0)
    ap.add_argument("--private-scale", type=float, default=0.75)
    ap.add_argument("--seed", type=int, default=57057)
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda", "auto"])
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    if int(args.epoch92) <= int(args.epoch80):
        raise ValueError("epoch92 must exceed epoch80")
    train_to_checkpoints(args)


if __name__ == "__main__":
    main()
