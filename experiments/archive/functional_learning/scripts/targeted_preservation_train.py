#!/usr/bin/env python3
"""research: Targeted preservation training — (M,S) acquisition + KL preservation.

Scientific purpose
------------------
research established that dense second-view input masking alone (the (M,S) arm)
reproduces the full dense source-responsive shift without broad target coverage.
Both (M,S) and (M,M) degrade context-independent prediction, CDI endpoint
likelihood/rank, and BLiMP/Supplement/EWoK performance.

This trainer adds a targeted KL preservation branch: for each Qwen pair row,
alongside the (M,S) focused acquisition rendering, a SEPARATE standard 15% WWM
rendering of the same text is created and the student's output distribution is
penalised for diverging from the frozen parent (coherent86) distribution.

Architecture
-----------
For each Qwen pair row in each macro update:
  Acquisition: (M,S) dense masks, sparse labels → focused CE loss  [via research]
  Preservation: standard 15% WWM on same full text → KL(student || teacher)

For non-Qwen rows: standard 15% WWM → CE loss (unchanged)

Macro-update optimized loss:
  L = λ_focus * mean(CE_focus) + (1-λ_focus) * mean(CE_ordinary) + λ_pres * mean(KL_pres)

Exposure accounting: each Qwen pair row is presented twice per macro update
(once for acquisition, once for preservation).  Total word exposure includes
both presentations; the endpoint label uses conservative cumulative count.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse
import hashlib
import json
import math
import os
import pathlib
import random
import sys
import time
from collections import Counter
from typing import Any, Dict, List, Tuple

import torch
import torch.nn.functional as F

ROOT = _public_path('.')
STUDY = _public_path('experiments/archive/functional_learning')
SCRIPTS = _public_path('experiments/archive/functional_learning/scripts')
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

# --- Import base training utilities BEFORE research monkeypatch ---
import real_stream_train_weighted as s64  # noqa: E402
import corrected_bridge_trainer as bridge  # noqa: E402

# Apply research monkeypatch: acquisition uses dense masks + sparse labels
import densemask_sparselabel_train as s75  # noqa: E402

DEFAULT_TAIL = s64.DEFAULT_TAIL
DEFAULT_OUT = _public_path('experiments/archive/functional_learning/data/targeted_preservation_train')

PARENT_EXPOSURE_WORDS = 86_005_295  # coherent86 total word exposure before continuation


def rel(p: pathlib.Path | str) -> str:
    try:
        return str(pathlib.Path(p).resolve().relative_to(ROOT))
    except Exception:
        return str(p)


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def load_teacher(device: torch.device, private_scale: float) -> Tuple[Any, Dict[str, Any]]:
    """Load coherent86 as frozen teacher for KL preservation.

    Uses the same bridge loader as the student to avoid read-only HF module
    cache issues.  Then freezes all parameters and sets eval mode.
    """
    teacher, _missing, _unexpected = bridge.load_model(device, private_scale)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    # Disable gradient checkpointing for teacher (pure inference)
    try:
        teacher.gradient_checkpointing_disable()
    except Exception:
        pass
    ident = bridge.model_identity(teacher)
    print(json.dumps({"event": "teacher_loaded", "identity": ident, "device": str(device),
                       "all_frozen": all(not p.requires_grad for p in teacher.parameters())}), flush=True)
    return teacher, ident


def prepare_preservation(macro_rows: List[Dict[str, Any]], tokenizer, seq_length: int,
                         wgb, train_seed: int, mask_prob: float
                         ) -> Tuple[List[Dict[str, Any]], int, int]:
    """Create standard 15% WWM examples for Qwen pair rows only (preservation)."""
    pres_examples: List[Dict[str, Any]] = []
    total_targets = 0
    total_words = 0
    for row in macro_rows:
        is_qwen = (row.get("source") == "qwen_pair_packed"
                    and bool(row.get("qwen_pair_segments")))
        if not is_qwen:
            continue
        tok = bridge.tokenize_row(row, tokenizer, seq_length, wgb)
        # Use a distinct seed family so preservation masks are independent of acquisition
        seed = s64.stable_seed("preservation-wwm-std", int(train_seed), bridge.row_key(row))
        masked, labels, _mstats = bridge.apply_wwm_row(
            tok["input_ids"], tok["attention_mask"], tok["word_group"],
            tokenizer, seed, mask_prob
        )
        n_tgt = int((labels != -100).sum().item())
        if n_tgt > 0:
            pres_examples.append({
                "input_ids": masked,
                "attention_mask": tok["attention_mask"],
                "labels": labels,
            })
            total_targets += n_tgt
        total_words += int(row.get("words", s64.wc(row.get("text", ""))))
    return pres_examples, total_targets, total_words


def train_preserved(rows: List[Dict[str, Any]], prefix_info: Dict[str, Any],
                    args: argparse.Namespace, device: torch.device, tokenizer) -> Dict[str, Any]:
    """Full training loop with (M,S) acquisition and KL preservation."""
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    s64.set_global_seed(s64.stable_seed(
        bytes((115, 116, 101, 112, 48, 56, 56, 45, 112, 114, 101, 115, 101, 114, 118, 97, 116, 105, 111, 110)).decode('utf-8'), int(args.train_seed),
        float(args.focus_lambda), float(args.lambda_pres)))

    # ---- Load student ----
    model, missing, unexpected = bridge.load_model(device, args.private_scale)
    ident = bridge.model_identity(model)
    if (ident.get("class") != "FrozenSlowPrivateDebertaV2ForMaskedLM"
            or int(ident.get("private_adapter_params", 0)) != 995584):
        raise RuntimeError(f"Bad student identity: {ident}")

    # ---- Load frozen teacher ----
    teacher, teacher_ident = load_teacher(device, float(args.private_scale))
    if teacher_ident.get("class") != "FrozenSlowPrivateDebertaV2ForMaskedLM":
        raise RuntimeError(f"Bad teacher identity: {teacher_ident}")

    # ---- Optimizer (private adapters only) ----
    opt, opt_info = bridge.freeze_to_private_optimizer(
        model, float(args.lr), float(args.weight_decay))
    try:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False})
    except TypeError:
        try:
            model.gradient_checkpointing_enable()
        except Exception:
            pass

    wgb = bridge.WordGroupBuilder(tokenizer)

    # ---- Config record ----
    config = {
        "status": "TARGETED_PRESERVATION_CONFIG",
        "created_utc": now(),
        "method": "dense_mask_sparse_label_acquisition_with_kl_preservation",
        "acquisition": "(M,S) dense second-view masks, sparse focus labels via research",
        "preservation": "standard 15% WWM on same Qwen pair text, KL to frozen parent",
        "lambda_pres": float(args.lambda_pres),
        "kl_temperature": float(args.kl_temperature),
        "focus_lambda": float(args.focus_lambda),
        "focus_prob": float(args.focus_prob),
        "max_focus_groups_per_row": int(args.max_focus_groups_per_row),
        "mask_prob": float(args.mask_prob),
        "train_seed": int(args.train_seed),
        "max_updates": int(args.max_updates),
        "parent_path": rel(bridge.PARENT_PATH),
        "student_identity": ident,
        "teacher_identity": teacher_ident,
        "optimizer": {"trainable_tensors": opt_info["trainable_tensors"],
                      "trainable_params": opt_info["trainable_params"]},
        "exposure_accounting": "prefix_words + preservation_words (conservative)",
    }
    (out_dir / "train_config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"event": "train_start", "method": "targeted_preservation",
                       "rows": len(rows), "prefix_words": prefix_info.get("prefix_words"),
                       "device": str(device)}, ensure_ascii=False), flush=True)

    # ---- Training loop ----
    objective = "correspondence_focus_weighted"
    cursor = 0
    cum_words = 0
    cum_pres_words = 0
    logs: List[Dict[str, Any]] = []
    checkpoints: List[Dict[str, Any]] = []
    t0 = time.time()
    model.train()
    micro_bs = int(args.micro_batch)
    T = float(args.kl_temperature)
    lambda_pres = float(args.lambda_pres)

    for update_i in range(int(args.max_updates)):
        # ---- Gather rows for this macro update ----
        macro_rows: List[Dict[str, Any]] = []
        words = 0
        while cursor < len(rows) and words < int(args.words_per_update):
            r = rows[cursor]
            macro_rows.append(r)
            words += int(r.get("words", s64.wc(r.get("text", ""))))
            cursor += 1
        if not macro_rows:
            print(json.dumps({"event": "data_exhausted", "update": update_i}), flush=True)
            break

        schedule_idx = int(args.schedule_offset) + update_i
        lr = s64.lr_at_update(schedule_idx, int(args.schedule_total),
                              int(args.warmup), float(args.lr))
        for pg in opt.param_groups:
            pg["lr"] = lr

        # ---- Acquisition examples (M,S) ----
        acq_examples, acq_prep = s64.prepare_macro(
            macro_rows, tokenizer, int(args.seq_length), wgb, objective,
            int(args.train_seed), float(args.mask_prob),
            float(args.focus_prob), int(args.max_focus_groups_per_row))
        focus_n = int(acq_prep["focus_target_tokens"])
        ord_n = int(acq_prep["ordinary_target_tokens"])
        n_acq = focus_n + ord_n
        if n_acq <= 0 or focus_n <= 0 or ord_n <= 0:
            raise RuntimeError(f"No acq targets at update {update_i}: {acq_prep}")
        lam_f = float(args.focus_lambda)
        lam_o = 1.0 - lam_f

        # ---- Preservation examples (standard WWM on Qwen rows) ----
        pres_examples, pres_n, pres_words = prepare_preservation(
            macro_rows, tokenizer, int(args.seq_length), wgb,
            int(args.train_seed), float(args.mask_prob))

        # ---- Forward + Backward ----
        opt.zero_grad(set_to_none=True)
        focus_sum = 0.0
        ord_sum = 0.0
        focus_seen = 0
        ord_seen = 0
        pres_kl_sum = 0.0
        pres_seen = 0

        # -- Acquisition microbatches --
        for start in range(0, len(acq_examples), micro_bs):
            mb = acq_examples[start:start + micro_bs]
            inp = torch.stack([x["input_ids"] for x in mb]).to(device)
            att = torch.stack([x["attention_mask"] for x in mb]).to(device)
            lab = torch.stack([x["labels"] for x in mb]).to(device)
            cf = torch.stack([
                (x["labels"] != -100) if x["component"] == "focus"
                else torch.zeros_like(x["labels"], dtype=torch.bool) for x in mb
            ]).to(device)
            co = torch.stack([
                (x["labels"] != -100) if x["component"] == "ordinary"
                else torch.zeros_like(x["labels"], dtype=torch.bool) for x in mb
            ]).to(device)
            out = model(input_ids=inp, attention_mask=att)
            V = out.logits.shape[-1]
            ce = F.cross_entropy(out.logits.reshape(-1, V), lab.reshape(-1),
                                 ignore_index=-100, reduction="none").view_as(lab)
            fs = ce[cf].sum() if cf.any() else torch.tensor(0.0, device=device)
            os_ = ce[co].sum() if co.any() else torch.tensor(0.0, device=device)
            loss = torch.tensor(0.0, device=device)
            if focus_n > 0 and lam_f != 0.0:
                loss = loss + lam_f * (fs / float(focus_n))
            if ord_n > 0 and lam_o != 0.0:
                loss = loss + lam_o * (os_ / float(ord_n))
            if torch.isnan(loss) or torch.isinf(loss):
                raise RuntimeError(f"bad acq loss at update {update_i}")
            loss.backward()
            focus_sum += float(fs.detach().cpu())
            ord_sum += float(os_.detach().cpu())
            focus_seen += int(cf.sum().detach().cpu())
            ord_seen += int(co.sum().detach().cpu())
            del inp, att, lab, cf, co, out, ce, fs, os_, loss

        # -- Preservation microbatches (KL to teacher) --
        if pres_n > 0 and lambda_pres > 0.0 and pres_examples:
            for start in range(0, len(pres_examples), micro_bs):
                mb = pres_examples[start:start + micro_bs]
                p_inp = torch.stack([x["input_ids"] for x in mb]).to(device)
                p_att = torch.stack([x["attention_mask"] for x in mb]).to(device)
                p_lab = torch.stack([x["labels"] for x in mb]).to(device)

                with torch.no_grad():
                    t_out = teacher(input_ids=p_inp, attention_mask=p_att)

                s_out = model(input_ids=p_inp, attention_mask=p_att)

                mask = p_lab != -100
                if mask.any():
                    s_logits = s_out.logits[mask] / T
                    t_logits = t_out.logits[mask].detach() / T
                    s_lp = F.log_softmax(s_logits, dim=-1)
                    t_p = F.softmax(t_logits, dim=-1)
                    kl = F.kl_div(s_lp, t_p, reduction="sum")
                    if T != 1.0:
                        kl = kl * (T * T)
                    p_loss = lambda_pres * (kl / float(pres_n))
                    if torch.isnan(p_loss) or torch.isinf(p_loss):
                        raise RuntimeError(f"bad pres loss at update {update_i}")
                    p_loss.backward()
                    pres_kl_sum += float(kl.detach().cpu())
                    pres_seen += int(mask.sum().detach().cpu())

                del p_inp, p_att, p_lab, t_out, s_out
                try:
                    del mask, s_logits, t_logits, s_lp, t_p, kl, p_loss
                except NameError:
                    pass

        # ---- Verify targets ----
        if focus_seen != focus_n or ord_seen != ord_n:
            raise RuntimeError(f"target mismatch update {update_i}: "
                               f"prep=({focus_n},{ord_n}) seen=({focus_seen},{ord_seen})")

        # ---- Step ----
        grad_norm = float(torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad],
            float(args.max_grad_norm)).detach().cpu())
        opt.step()

        cum_words += words
        cum_pres_words += pres_words
        focus_loss = float(focus_sum / focus_n) if focus_n else 0.0
        ordinary_loss = float(ord_sum / ord_n) if ord_n else 0.0
        pres_kl_mean = float(pres_kl_sum / pres_n) if pres_n else 0.0
        pooled_ce = float((focus_sum + ord_sum) / max(1, n_acq))
        optimized_acq = float(lam_f * focus_loss + lam_o * ordinary_loss)
        optimized_total = optimized_acq + lambda_pres * pres_kl_mean

        log = {
            "update": update_i + 1,
            "schedule_idx": schedule_idx,
            "lr": lr,
            "rows": len(macro_rows),
            "words": words,
            "cum_words": cum_words,
            "cum_pres_words": cum_pres_words,
            "cum_words_total": cum_words + cum_pres_words,
            "pres_words": pres_words,
            "pres_targets": pres_n,
            "pres_kl_mean": pres_kl_mean,
            "focus_loss": focus_loss,
            "ordinary_loss": ordinary_loss,
            "pooled_ce": pooled_ce,
            "optimized_acq": optimized_acq,
            "optimized_total": optimized_total,
            "lambda_pres": lambda_pres,
            "kl_temperature": T,
            "focus_target_tokens": focus_n,
            "ordinary_target_tokens": ord_n,
            "qwen_rows": int(acq_prep["qwen_rows"]),
            "grad_norm_preclip": grad_norm,
            "elapsed_sec": round(time.time() - t0, 1),
        }
        logs.append(log)
        if update_i == 0 or (update_i + 1) % int(args.log_every) == 0:
            print(json.dumps({"event": "update", **log}, ensure_ascii=False), flush=True)

        if ((update_i + 1) % int(args.checkpoint_every) == 0
                or update_i + 1 == int(args.max_updates)
                or cursor >= len(rows)):
            ckpt_name = f"update_{update_i + 1:04d}"
            ckpt_dir = out_dir / "checkpoints" / ckpt_name
            metadata = {
                "method": "targeted_preservation",
                "objective": objective,
                "update": update_i + 1,
                "cum_words": cum_words,
                "cum_words_total": cum_words + cum_pres_words,
                "train_seed": int(args.train_seed),
                "prefix_info": prefix_info,
                "final_update": log,
                "model_identity": ident,
                "teacher_identity": teacher_ident,
                "private_scale": float(args.private_scale),
                "lambda_pres": lambda_pres,
                "kl_temperature": T,
            }
            s64.save_checkpoint(model, tokenizer, ckpt_dir, metadata,
                                float(args.private_scale))
            checkpoints.append({"update": update_i + 1, "path": rel(ckpt_dir),
                                "optimized_total": optimized_total})
            print(json.dumps({"event": "checkpoint", "update": update_i + 1,
                               "path": rel(ckpt_dir)}, ensure_ascii=False), flush=True)

        if cursor >= len(rows):
            print(json.dumps({"event": "prefix_exhausted",
                               "update": update_i + 1}, ensure_ascii=False), flush=True)
            break

    # ---- Write summary ----
    total_focus = sum(int(x["focus_target_tokens"]) for x in logs)
    total_ord = sum(int(x["ordinary_target_tokens"]) for x in logs)
    total_pres = sum(int(x["pres_targets"]) for x in logs)
    total_pres_w = sum(int(x["pres_words"]) for x in logs)
    endpoint_exposure = PARENT_EXPOSURE_WORDS + cum_words + total_pres_w

    summary = {
        "status": "TARGETED_PRESERVATION_TRAIN_DONE",
        "method": "dense_mask_sparse_label_acquisition_with_kl_preservation",
        "out_dir": rel(out_dir),
        "completed_updates": len(logs),
        "total_words_consumed": cum_words,
        "total_preservation_words": total_pres_w,
        "total_words_combined": cum_words + total_pres_w,
        "total_focus_targets": total_focus,
        "total_ordinary_targets": total_ord,
        "total_preservation_targets": total_pres,
        "lambda_pres": float(args.lambda_pres),
        "kl_temperature": float(args.kl_temperature),
        "focus_lambda": float(args.focus_lambda),
        "final_update": logs[-1] if logs else None,
        "checkpoints": checkpoints,
        "elapsed_sec": round(time.time() - t0, 1),
        "model_identity": ident,
        "teacher_identity": teacher_ident,
        "exposure_accounting": {
            "parent_exposure": PARENT_EXPOSURE_WORDS,
            "prefix_words": cum_words,
            "preservation_additional_words": total_pres_w,
            "endpoint_exposure_conservative": endpoint_exposure,
            "endpoint_label": f"endpoint_{endpoint_exposure / 1e6:.6f}M",
            "note": "Conservative: counts each Qwen pair twice (acquisition + preservation)",
        },
    }
    (out_dir / "train_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    s64.write_jsonl(out_dir / "update_log.jsonl", logs)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)

    del model, teacher
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tail-jsonl", type=pathlib.Path, default=DEFAULT_TAIL)
    ap.add_argument("--out-dir", type=pathlib.Path, default=DEFAULT_OUT)
    # Training schedule (identical to research/research defaults)
    ap.add_argument("--max-updates", type=int, default=80)
    ap.add_argument("--words-per-update", type=int, default=39533)
    ap.add_argument("--schedule-total", type=int, default=455)
    ap.add_argument("--schedule-offset", type=int, default=101)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--max-grad-norm", type=float, default=1.0)
    ap.add_argument("--private-scale", type=float, default=0.75)
    ap.add_argument("--seq-length", type=int, default=512)
    ap.add_argument("--micro-batch", type=int, default=8)
    # Acquisition (M,S) parameters
    ap.add_argument("--mask-prob", type=float, default=0.15)
    ap.add_argument("--focus-prob", type=float, default=0.35)
    ap.add_argument("--max-focus-groups-per-row", type=int, default=16)
    ap.add_argument("--focus-lambda", type=float, default=0.15)
    # Preservation parameters
    ap.add_argument("--lambda-pres", type=float, default=0.5,
                    help="Weight for KL preservation loss")
    ap.add_argument("--kl-temperature", type=float, default=1.0,
                    help="Temperature for KL distillation")
    # Bookkeeping
    ap.add_argument("--checkpoint-every", type=int, default=40)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--train-seed", type=int, default=62064)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if not (0.0 <= float(args.focus_lambda) <= 1.0):
        raise SystemExit("--focus-lambda must be in [0,1]")
    if float(args.lambda_pres) < 0:
        raise SystemExit("--lambda-pres must be >= 0")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str((args.out_dir / "hf_cache").resolve())
    os.environ["TRANSFORMERS_CACHE"] = str((args.out_dir / "hf_cache/transformers").resolve())
    os.environ["HF_MODULES_CACHE"] = str((args.out_dir / "hf_cache/modules").resolve())
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    (args.out_dir / "hf_cache/modules").mkdir(parents=True, exist_ok=True)

    rows, prefix_info = s64.load_prefix(
        args.tail_jsonl, int(args.max_updates), int(args.words_per_update))
    tokenizer = bridge.AutoTokenizer.from_pretrained(
        str(bridge.PARENT_PATH), local_files_only=True, use_fast=True)

    if args.dry_run:
        wgb = bridge.WordGroupBuilder(tokenizer)
        macro_rows: List[Dict[str, Any]] = []
        w = 0
        for r in rows:
            if w >= int(args.words_per_update):
                break
            macro_rows.append(r)
            w += int(r.get("words", s64.wc(r.get("text", ""))))
        acq_ex, acq_prep = s64.prepare_macro(
            macro_rows, tokenizer, int(args.seq_length), wgb,
            "correspondence_focus_weighted", int(args.train_seed),
            float(args.mask_prob), float(args.focus_prob),
            int(args.max_focus_groups_per_row))
        pres_ex, pres_n, pres_w = prepare_preservation(
            macro_rows, tokenizer, int(args.seq_length), wgb,
            int(args.train_seed), float(args.mask_prob))
        dry_result = {
            "status": "DRY_RUN_DONE",
            "created_utc": now(),
            "macro_rows": len(macro_rows),
            "acq_examples": len(acq_ex),
            "acq_focus_targets": int(acq_prep["focus_target_tokens"]),
            "acq_ordinary_targets": int(acq_prep["ordinary_target_tokens"]),
            "acq_qwen_rows": int(acq_prep["qwen_rows"]),
            "pres_examples": len(pres_ex),
            "pres_targets": pres_n,
            "pres_words": pres_w,
            "lambda_pres": float(args.lambda_pres),
            "kl_temperature": float(args.kl_temperature),
        }
        (args.out_dir / "dry_run.json").write_text(
            json.dumps(dry_result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(json.dumps(dry_result, indent=2, ensure_ascii=False), flush=True)
        return

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    train_preserved(rows, prefix_info, args, device, tokenizer)


if __name__ == "__main__":
    main()
