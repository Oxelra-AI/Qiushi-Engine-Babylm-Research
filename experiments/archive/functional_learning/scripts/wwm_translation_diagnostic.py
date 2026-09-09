#!/usr/bin/env python3
"""research: WWM translation diagnostic.

Tests whether the research compact+support allocation mechanism survives translation
from source-visible answer-only training to ordinary whole-word masking (WWM).

The critical question: research showed compact+aux preserves
base common-target following while learning auxiliary support. But that training
used concentrated credit on content words in the second view, with the source text
always fully visible. A real BabyLM legal stream uses standard WWM where any position
can be masked at ~15%. The relation bridge already showed that supplying the same
experience under ordinary WWM does not preserve the successful learning process.

This diagnostic directly tests whether the research mechanism is an artifact of the
concentrated credit scheme or a real property of the text correspondence.

Arms tested (same rows and schedules as research):
  compact_aux_wwm: compact views + aux rows, full WWM on ALL positions
  current_aux_wwm: current views + aux substitution, full WWM on ALL positions  
  compact_base_wwm: compact views only, full WWM (no aux)

If compact_aux_wwm produces similar common-target g as research's compact_aux_support,
the mechanism transfers and the legal-stream candidate is justified.
If WWM loses the signal, we need focused credit in the legal stream.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse
import csv
import hashlib
import json
import math
import os
import pathlib
import random
import statistics
import sys
import time
from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

ROOT = _public_path('.')
STUDY = _public_path('experiments/archive/functional_learning')
SCRIPTS = _public_path('experiments/archive/functional_learning/scripts')
sys.path.insert(0, str(SCRIPTS))
import corrected_bridge_trainer as bridge  # noqa: E402
import concentrated_compact_learning_test as s57  # noqa: E402
import common_target_probe as s58  # noqa: E402
import allocation_learner_comparison as s60  # noqa: E402

DEFAULT_OUT = _public_path('experiments/archive/functional_learning/data/wwm_translation_diagnostic')


def rel(p: pathlib.Path | str) -> str:
    try:
        return str(pathlib.Path(p).resolve().relative_to(ROOT))
    except Exception:
        return str(p)


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


# ── WWM collation: replaces research's answer-only collation ──────────────────

def build_wwm_word_groups(input_ids: List[int], tokenizer) -> List[List[int]]:
    """Build word groups over ALL tokens (source + view), like the bridge trainer."""
    special_ids = set(int(x) for x in tokenizer.all_special_ids)
    groups: List[List[int]] = []
    current_group: List[int] = []

    for i, tid in enumerate(input_ids):
        if int(tid) in special_ids:
            if current_group:
                groups.append(current_group)
                current_group = []
            continue
        tok = tokenizer.convert_ids_to_tokens(int(tid))
        if tok is None:
            if current_group:
                groups.append(current_group)
                current_group = []
            continue
        is_start = bridge.is_word_start(str(tok))
        if is_start and current_group:
            groups.append(current_group)
            current_group = []
        current_group.append(i)

    if current_group:
        groups.append(current_group)
    return groups


def collate_presentations_wwm(
    batch: List[Dict[str, Any]], tokenizer, epoch: int, seed: int,
    mask_prob: float, device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
    """Full whole-word masking on ALL positions (source + view), not just view content words."""
    max_len = max(len(x["input_ids"]) for x in batch)
    pad_id = int(tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0)
    mask_id = int(tokenizer.mask_token_id)
    ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
    att = torch.zeros((len(batch), max_len), dtype=torch.long)
    labels = torch.full((len(batch), max_len), -100, dtype=torch.long)
    target_groups = 0
    target_tokens = 0

    for i, ex in enumerate(batch):
        L = len(ex["input_ids"])
        ids[i, :L] = torch.tensor(ex["input_ids"], dtype=torch.long)
        att[i, :L] = 1

        # Build word groups over ALL tokens (not just view content words)
        word_groups = build_wwm_word_groups(ex["input_ids"], tokenizer)

        # Standard 15% whole-word masking with deterministic row-keyed seed
        mseed = s60.stable_seed(
            "wwm-mask", seed, epoch, ex["pair_id"],
            ex["view_kind"], ex.get("presentation_kind"),
            ex.get("presentation_index", 0),
        )
        rng = random.Random(mseed)

        chosen = [g for g in word_groups if rng.random() < float(mask_prob)]
        # Ensure at least one group is masked per row
        if not chosen and word_groups:
            chosen = [rng.choice(word_groups)]

        for g in chosen:
            target_groups += 1
            for p in g:
                if p < L:
                    labels[i, p] = ids[i, p]
                    # Standard BERT masking: 80% MASK, 10% random, 10% keep
                    r = rng.random()
                    if r < 0.8:
                        ids[i, p] = mask_id
                    elif r < 0.9:
                        ids[i, p] = rng.randrange(int(tokenizer.vocab_size))
                    # else keep original
                    target_tokens += 1

    return ids.to(device), att.to(device), labels.to(device), {
        "target_groups": target_groups,
        "target_tokens": target_tokens,
    }


# ── Training loop with WWM ──────────────────────────────────────────────────

def train_arm_wwm(
    arm_name: str, rep_seed: int, tokenizer, device: torch.device, args: argparse.Namespace,
    out_dir: pathlib.Path, base_current, base_compact, aux_compact,
    aux_schedule_by_epoch, rec_schedule_by_epoch, subst_schedule_by_epoch,
    surface_tasks, parent_surface_rows, common_records, parent_common_rows,
) -> Dict[str, Any]:
    """Same as research train_arm but uses WWM instead of answer-only masking."""
    arm_dir = out_dir / f"seed_{rep_seed}" / arm_name
    arm_dir.mkdir(parents=True, exist_ok=True)
    print(json.dumps({"event": "arm_start_wwm", "seed": rep_seed, "arm": arm_name}), flush=True)

    s60.set_global_seed(s60.stable_seed("arm-global-wwm", rep_seed, arm_name))
    model, missing, unexpected = bridge.load_model(device, private_scale=float(args.private_scale))
    ident = bridge.model_identity(model)
    if ident.get("class") != "FrozenSlowPrivateDebertaV2ForMaskedLM" or int(ident.get("private_adapter_params", 0)) != 995584:
        raise RuntimeError(f"bad model identity: {ident}")
    opt, opt_info = bridge.freeze_to_private_optimizer(model, float(args.lr), float(args.weight_decay))
    model.train()
    logs: List[Dict[str, Any]] = []
    t0 = time.time()
    epochs = int(args.epochs)

    for epoch in range(1, epochs + 1):
        presentations, pst = s60.make_presentations_for_epoch(
            arm_name, epoch, base_current, base_compact, aux_compact,
            aux_schedule_by_epoch, rec_schedule_by_epoch, subst_schedule_by_epoch,
        )
        rng = random.Random(s60.stable_seed("order-wwm", rep_seed, arm_name, epoch))
        order = list(range(len(presentations)))
        rng.shuffle(order)
        epoch_loss_sum = 0.0
        epoch_batches = 0
        epoch_targets = 0
        epoch_groups = 0

        for start in range(0, len(order), int(args.batch_size)):
            batch = [presentations[j] for j in order[start:start + int(args.batch_size)]]
            # WWM collation instead of answer-only
            ids, att, labels, st = collate_presentations_wwm(
                batch, tokenizer, epoch, rep_seed, float(args.mask_prob), device,
            )
            if int((labels != -100).sum().item()) == 0:
                continue
            out = model(input_ids=ids, attention_mask=att)
            vocab = out.logits.shape[-1]
            loss = F.cross_entropy(
                out.logits.reshape(-1, vocab), labels.reshape(-1),
                ignore_index=-100, reduction="mean",
            )
            if torch.isnan(loss) or torch.isinf(loss):
                raise RuntimeError(f"bad loss {arm_name} seed {rep_seed} epoch {epoch}")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = float(
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    float(args.max_grad_norm),
                ).detach().cpu()
            )
            opt.step()
            epoch_loss_sum += float(loss.detach().cpu())
            epoch_batches += 1
            epoch_targets += int(st["target_tokens"])
            epoch_groups += int(st["target_groups"])
            del ids, att, labels, out, loss

        log = {
            "epoch": epoch,
            "mean_loss": epoch_loss_sum / max(1, epoch_batches),
            "batches": epoch_batches,
            "target_groups": epoch_groups,
            "target_tokens": epoch_targets,
            "presentations": len(presentations),
            "row_words_epoch": int(pst.get("row_words", 0)),
            "view_words_epoch": int(pst.get("view_words", 0)),
            "row_words_cum": sum(int(x.get("row_words_epoch", 0)) for x in logs) + int(pst.get("row_words", 0)),
            "elapsed_sec": round(time.time() - t0, 1),
            **{k: int(v) for k, v in pst.items() if k.endswith("_presentations")},
        }
        logs.append(log)
        if epoch == 1 or epoch % int(args.log_every) == 0 or epoch == epochs:
            print(json.dumps({"event": "train_epoch_wwm", "seed": rep_seed, "arm": arm_name, **log}), flush=True)

    model.eval()
    surface_rows = s60.score_surface_tasks(model, tokenizer, surface_tasks, device, int(args.eval_batch_size))
    common_rows = s60.score_common_records(model, tokenizer, common_records, device, int(args.eval_batch_size))
    s60.write_jsonl(arm_dir / "surface_scores.jsonl", surface_rows)
    s60.write_jsonl(arm_dir / "common_scores.jsonl", common_rows)
    surf_summary = s60.summarize_surface(surface_rows, parent_surface_rows)
    common_summary = s60.summarize_common(common_rows, parent_common_rows)
    slim_common = {k: v for k, v in common_summary.items() if k not in {"margins", "source_follow_rows", "delta_rows"}}
    s60.write_jsonl(arm_dir / "common_delta_rows.jsonl", common_summary["delta_rows"])
    s60.write_jsonl(arm_dir / "common_source_follow_rows.jsonl", common_summary["source_follow_rows"])

    summary = {
        "seed": int(rep_seed),
        "arm": arm_name,
        "objective": "wwm",
        "mask_prob": float(args.mask_prob),
        "model_identity": ident,
        "optimizer": opt_info,
        "completed_epochs": epochs,
        "final_train_log": logs[-1] if logs else None,
        "train_log": rel(arm_dir / "train_log.jsonl"),
        "surface_summary": surf_summary,
        "common_summary": slim_common,
        "elapsed_sec": round(time.time() - t0, 1),
    }
    s60.write_jsonl(arm_dir / "train_log.jsonl", logs)
    (arm_dir / "arm_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8",
    )
    print(json.dumps({
        "event": "arm_done_wwm", "seed": rep_seed, "arm": arm_name,
        "elapsed_sec": summary["elapsed_sec"],
    }), flush=True)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return summary


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=str, default=str(DEFAULT_OUT))
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--seeds", type=str, default="61001,61002,61003")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--mask-prob", type=float, default=0.15,
                    help="Standard WWM masking rate (0.15), not research's 0.35 content-word selection rate")
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--max-grad-norm", type=float, default=1.0)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--eval-batch-size", type=int, default=64)
    ap.add_argument("--private-scale", type=float, default=0.75)
    ap.add_argument("--log-every", type=int, default=20)
    ap.add_argument("--max-length", type=int, default=320)
    ap.add_argument("--max-train-targets", type=int, default=16)
    ap.add_argument("--dry-run", action="store_true")
    # Arms to run: subset of the research arms
    ap.add_argument("--arms", type=str, default="compact_aux_support,current_aux_substitution,compact_base80_unspent")
    args = ap.parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(",")]
    arm_names = [a.strip() for a in args.arms.split(",")]
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    os.environ["HF_HOME"] = str((out_dir / "hf_cache/hf_home").resolve())
    os.environ["TRANSFORMERS_CACHE"] = str((out_dir / "hf_cache/transformers").resolve())
    os.environ["HF_MODULES_CACHE"] = str((out_dir / "hf_cache/modules").resolve())
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    (out_dir / "hf_cache/modules").mkdir(parents=True, exist_ok=True)

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    # ── Load the same data as research ────────────────────────────────────────
    design_dir = _public_path('experiments/archive/functional_learning/data/allocation_design_v2')
    design = s60.load_json(design_dir / "allocation_design_v2.json")

    base_current_rows = s60.load_jsonl(design_dir / "base_current_rows.jsonl")
    base_compact_rows = s60.load_jsonl(design_dir / "base_compact_rows.jsonl")
    aux_compact_rows = s60.load_jsonl(design_dir / "aux_compact_rows.jsonl")
    aux_schedule = s60.load_jsonl(design_dir / "aux_support_schedule_epoch80.jsonl")
    aux_tasks_path = design_dir / "aux_support_tasks_repaired.jsonl"
    label_path = pathlib.Path(design["inputs"]["semantic_labels"])
    if not label_path.is_absolute():
        label_path = ROOT / label_path

    tokenizer = bridge.AutoTokenizer.from_pretrained(str(bridge.PARENT_PATH), local_files_only=True, use_fast=True)
    if tokenizer.mask_token_id is None:
        raise RuntimeError("tokenizer has no mask token")

    # Build train examples - train_groups needed for surface eval; WWM masking overrides during training
    base_current, bc_stats = s60.build_train_examples(
        [s60.row_to_record(r) for r in base_current_rows], tokenizer, int(args.max_length), "current", int(args.max_train_targets),
    )
    base_compact, bk_stats = s60.build_train_examples(
        [s60.row_to_record(r) for r in base_compact_rows], tokenizer, int(args.max_length), "compact", int(args.max_train_targets),
    )
    aux_compact, ak_stats = s60.build_train_examples(
        [s60.row_to_record(r) for r in aux_compact_rows], tokenizer, int(args.max_length), "compact", int(args.max_train_targets),
    )
    if set(base_current) != set(base_compact):
        raise RuntimeError("base current/compact IDs differ")
    if set(aux_compact) != set(r["pair_id"] for r in aux_compact_rows):
        raise RuntimeError("aux compact IDs lost during tokenization")

    # Build evaluation tasks (same as research)
    base_surf, bsurf_stats = s60.build_surface_tasks(base_compact_rows, tokenizer, int(args.max_length), int(args.max_train_targets), "base_surface")
    aux_surf, asurf_stats = s60.build_surface_tasks(aux_compact_rows, tokenizer, int(args.max_length), int(args.max_train_targets), "aux_surface")
    surface_tasks = base_surf + aux_surf
    common_records, common_stats = s60.build_common_candidate_records(
        tokenizer, label_path, aux_tasks_path, int(args.max_length),
    )

    # Build schedules per seed (same logic as research)
    aux_schedule_by_epoch = {int(e["epoch"]): e for e in aux_schedule}

    plan = {
        "status": "WWM_TRANSLATION_PLAN",
        "created_utc": now(),
        "purpose": "Test whether research allocation signal survives translation to standard whole-word masking",
        "objective": "wwm",
        "mask_prob": float(args.mask_prob),
        "answer_only_mask_prob": 0.35,
        "comparison_baseline": "research answer-only (source-visible, content-word-only masking at 0.35 selection rate)",
        "wwm_difference": "Standard 15% masking on ALL positions (source + view), not just view content words",
        "arms": arm_names,
        "seeds": seeds,
        "epochs": int(args.epochs),
        "base_faithful_rows": len(base_compact),
        "aux_reviewed_rows": len(aux_compact),
        "base_surface_tasks": len(base_surf),
        "aux_surface_tasks": len(aux_surf),
        "common_target_records": len(common_records),
        "train_example_stats": {
            "base_current": bc_stats,
            "base_compact": bk_stats,
            "aux_compact": ak_stats,
        },
    }

    if args.dry_run:
        plan["dry_run"] = True
        print(json.dumps(plan, indent=2, ensure_ascii=False), flush=True)
        return

    (out_dir / "plan.json").write_text(json.dumps(plan, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    # ── Score parent for baseline ────────────────────────────────────────────
    print(json.dumps({"event": "scoring_parent"}), flush=True)
    s60.set_global_seed(42)
    model, _, _ = bridge.load_model(device, private_scale=float(args.private_scale))
    model.eval()
    parent_surface_rows = s60.score_surface_tasks(model, tokenizer, surface_tasks, device, int(args.eval_batch_size))
    parent_common_rows = s60.score_common_records(model, tokenizer, common_records, device, int(args.eval_batch_size))
    parent_surface_summary = s60.summarize_surface(parent_surface_rows)
    parent_common_summary = s60.summarize_common(parent_common_rows)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    s60.write_jsonl(out_dir / "parent_surface_scores.jsonl", parent_surface_rows)
    s60.write_jsonl(out_dir / "parent_common_scores.jsonl", parent_common_rows)
    print(json.dumps({"event": "parent_scored"}), flush=True)

    # ── Run arms ─────────────────────────────────────────────────────────────
    all_summaries: List[Dict[str, Any]] = []
    for seed in seeds:
        # Build per-seed balanced schedules (same as research)
        rec_schedule, rec_summary = s60.build_balanced_recurrence_schedule(base_compact_rows, aux_schedule, seed)
        rec_by_epoch = {int(e["epoch"]): e for e in rec_schedule}
        subst_schedule, subst_summary = s60.build_balanced_current_substitution_schedule(
            base_current_rows, aux_schedule,
            int(design["word_accounting_per_base_epoch"]["current_base_row_words"]), seed,
        )
        subst_by_epoch = {int(e["epoch"]): e for e in subst_schedule}

        for arm_name in arm_names:
            summary = train_arm_wwm(
                arm_name, seed, tokenizer, device, args, out_dir,
                base_current, base_compact, aux_compact,
                aux_schedule_by_epoch, rec_by_epoch, subst_by_epoch,
                surface_tasks, parent_surface_rows, common_records, parent_common_rows,
            )
            all_summaries.append(summary)

    # ── Aggregate ────────────────────────────────────────────────────────────
    agg = s60.aggregate_run_results(parent_common_summary, parent_surface_summary, all_summaries)
    agg["objective"] = "wwm"
    agg["comparison_note"] = (
        "Compare these WWM results against research answer-only results. "
        "If compact_aux_wwm preserves similar base_common_g and aux_common_g as "
        "research's compact_aux_support, the mechanism transfers to realistic conditions."
    )
    (out_dir / "summary.json").write_text(json.dumps(agg, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    # Per-seed CSV
    metrics_csv = out_dir / "per_seed_arm_metrics.csv"
    if agg["per_seed_arm_metrics"]:
        keys = list(agg["per_seed_arm_metrics"][0].keys())
        with metrics_csv.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, keys)
            w.writeheader()
            w.writerows(agg["per_seed_arm_metrics"])

    # ── Load research reference for direct comparison ─────────────────────────
    ref_path = _public_path('experiments/archive/functional_learning/data/allocation_balanced_synthesis/summary.json')
    comparison = {"wwm_by_arm": agg.get("by_arm_metric", {})}
    if ref_path.exists():
        ref = json.loads(ref_path.read_text(encoding="utf-8"))
        comparison["answer_only_by_arm"] = ref.get("by_arm_metric", {})
        # Direct comparison for each shared arm
        for arm in arm_names:
            wwm = comparison["wwm_by_arm"].get(arm, {})
            ao = comparison["answer_only_by_arm"].get(arm, {})
            if wwm and ao:
                comparison[f"delta_{arm}"] = {}
                for k in ["base_common_g", "aux_common_g", "base_compact_with_source_delta_nll", "aux_compact_with_source_delta_nll"]:
                    wm = wwm.get(k, {}).get("mean")
                    am = ao.get(k, {}).get("mean")
                    if wm is not None and am is not None:
                        comparison[f"delta_{arm}"][k] = {
                            "wwm_mean": wm,
                            "answer_only_mean": am,
                            "delta_wwm_minus_ao": wm - am,
                        }
    (out_dir / "wwm_vs_answer_only_comparison.json").write_text(
        json.dumps(comparison, indent=2, ensure_ascii=False) + "\n", encoding="utf-8",
    )

    final = {
        "status": "WWM_TRANSLATION_DONE",
        "created_utc": now(),
        "out_dir": rel(out_dir),
        "seeds": seeds,
        "arms": arm_names,
        "objective": "wwm",
        "mask_prob": float(args.mask_prob),
        "n_summaries": len(all_summaries),
        "summary": rel(out_dir / "summary.json"),
        "comparison": rel(out_dir / "wwm_vs_answer_only_comparison.json"),
        "metrics_csv": rel(metrics_csv),
        "interpretation": (
            "If WWM compact_aux shows base_common_g > 0.10 and aux_common_g > 0.05, "
            "the allocation mechanism transfers to realistic conditions and a legal-stream "
            "candidate is justified. If WWM loses the signal, focused credit or a "
            "different objective design is needed for the legal stream."
        ),
    }
    print(json.dumps(final, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
