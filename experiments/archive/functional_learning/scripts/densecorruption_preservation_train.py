#!/usr/bin/env python3
"""research: dense-corruption non-label preservation arm.

This is a carefully matched explanatory arm, not a v5 prerequisite. It keeps the
research exact `(M,S)` acquisition pathway fixed and changes only the parent-KL
preservation branch relative to research clean preservation:

  * clean preservation: ordinary 15% WWM on full Qwen pair rows, KL on ordinary
    WWM targets;
  * this arm: dense second-view corruption on full Qwen pair rows, KL on dense
    masked positions that are *not* sparse acquisition labels.

The full arm changes both input rendering and preservation target support. research
therefore also produced a common-support gradient/uncertainty measurement before
this arm is interpreted. Equal lambda is not assumed to mean equal constraint
strength.
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
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

ROOT = _public_path('.')
STUDY = _public_path('experiments/archive/functional_learning')
SCRIPTS = _public_path('experiments/archive/functional_learning/scripts')
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

# Import the clean trainer because it already applies the research monkeypatch for
# acquisition and contains the repaired RNG/teacher helpers.
import clean_preservation_train as clean  # noqa: E402
import real_stream_train_weighted as s64  # noqa: E402
import corrected_bridge_trainer as bridge  # noqa: E402
import preservation_geometry_matched_diagnostic as geom  # noqa: E402

DEFAULT_OUT = _public_path('experiments/archive/functional_learning/data/densecorruption_preservation_lambda1_full80')
PARENT_EXPOSURE_WORDS = clean.PARENT_EXPOSURE_WORDS


def rel(p: pathlib.Path | str | None) -> str | None:
    if p is None:
        return None
    try:
        return str(pathlib.Path(p).resolve().relative_to(ROOT))
    except Exception:
        return str(p)


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def setup_cache(out_dir: pathlib.Path) -> None:
    clean.setup_cache(out_dir)


def labels_at_positions(input_ids: torch.Tensor, positions: List[int]) -> torch.Tensor:
    labels = torch.full_like(input_ids, -100)
    for p in positions:
        p = int(p)
        if 0 <= p < int(input_ids.numel()):
            labels[p] = input_ids[p]
    return labels


def prepare_densecorruption_nonlabel_preservation(
    macro_rows: List[Dict[str, Any]], tokenizer, seq_length: int, wgb, train_seed: int,
    mask_prob: float, focus_prob: float, max_focus_groups_per_row: int, max_dense_mask_groups: int,
) -> Tuple[List[Dict[str, Any]], int, int, Dict[str, Any]]:
    examples: List[Dict[str, Any]] = []
    total_targets = 0
    total_words = 0
    c = Counter()
    kind_counts = Counter()
    row_records: List[Dict[str, Any]] = []
    mask_id = int(tokenizer.mask_token_id)
    for row in macro_rows:
        is_qwen = row.get("source") == "qwen_pair_packed" and bool(row.get("qwen_pair_segments"))
        if not is_qwen:
            continue
        c["qwen_rows"] += 1
        tok = bridge.tokenize_row(row, tokenizer, seq_length, wgb)
        row_key = bridge.row_key(row)
        seed = s64.stable_seed("view-focus", int(train_seed), row_key)
        masked, sparse_labels, mstats = geom.densemask_sparse_label_row(
            row, tok, tokenizer, seed, float(focus_prob), int(max_focus_groups_per_row), int(max_dense_mask_groups)
        )
        orig = tok["input_ids"]
        att = tok["attention_mask"]
        dense_mask_pos = {int(i) for i in torch.nonzero((masked == mask_id) & (orig != mask_id) & (att == 1), as_tuple=False).view(-1).tolist()}
        sparse_label_pos = {int(i) for i in torch.nonzero(sparse_labels != -100, as_tuple=False).view(-1).tolist()}
        nonlabel_pos = sorted(dense_mask_pos - sparse_label_pos)
        n_tgt = len(nonlabel_pos)
        words = int(row.get("words", s64.wc(row.get("text", ""))))
        total_words += words
        c["dense_masked_tokens"] += len(dense_mask_pos)
        c["sparse_label_targets"] += len(sparse_label_pos)
        c["dense_nonlabel_targets"] += n_tgt
        c["candidate_groups"] += int(mstats.get("n_candidate_groups", 0))
        c["dense_masked_groups"] += int(mstats.get("n_masked_groups", 0))
        c["sparse_label_groups"] += int(mstats.get("n_selected_groups", 0))
        c["zero_dense_nonlabel_rows"] += int(n_tgt == 0)
        kind_counts.update(mstats.get("selected_candidate_kind_counts") or {})
        row_records.append({
            "row_key": row_key,
            "words": words,
            "dense_masked_tokens": len(dense_mask_pos),
            "sparse_label_targets": len(sparse_label_pos),
            "dense_nonlabel_targets": n_tgt,
        })
        if n_tgt > 0:
            examples.append({
                "input_ids": masked,
                "attention_mask": att,
                "labels": labels_at_positions(orig, nonlabel_pos),
                "row_key": row_key,
                "words": words,
                "source": row.get("source", ""),
                "preservation_component": "dense_corruption_nonlabel",
            })
            total_targets += n_tgt
    c["examples"] = len(examples)
    stats = {
        "counts": dict(c),
        "selected_sparse_label_kind_counts": dict(kind_counts),
        "label_to_mask_token_ratio": float(c["sparse_label_targets"] / c["dense_masked_tokens"]) if c["dense_masked_tokens"] else None,
        "nonlabel_to_mask_token_ratio": float(c["dense_nonlabel_targets"] / c["dense_masked_tokens"]) if c["dense_masked_tokens"] else None,
        "row_records_head": row_records[:12],
    }
    return examples, total_targets, total_words, stats


def train_dense_preservation(rows: List[Dict[str, Any]], prefix_info: Dict[str, Any], args: argparse.Namespace,
                             device: torch.device, tokenizer) -> Dict[str, Any]:
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    objective = "correspondence_focus_weighted"
    acq_seed = clean.seed_original_acquisition(objective, int(args.train_seed), float(args.focus_lambda))
    s64.set_global_seed(acq_seed)

    model, missing, unexpected = bridge.load_model(device, float(args.private_scale))
    ident = bridge.model_identity(model)
    if ident.get("class") != "FrozenSlowPrivateDebertaV2ForMaskedLM" or int(ident.get("private_adapter_params", 0)) != 995584:
        raise RuntimeError(f"Bad student identity: {ident}")
    opt, opt_info = bridge.freeze_to_private_optimizer(model, float(args.lr), float(args.weight_decay))
    try:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    except TypeError:
        try:
            model.gradient_checkpointing_enable()
        except Exception:
            pass

    post_student_rng = clean.rng_state()
    post_student_rng_digest = clean.rng_digest()
    teacher = None
    teacher_ident: Optional[Dict[str, Any]] = None
    if float(args.lambda_pres) > 0.0 or not bool(args.skip_teacher_when_zero):
        teacher, teacher_ident = clean.load_teacher(device, float(args.private_scale))
        if bool(args.restore_rng_after_teacher_load):
            clean.set_rng_state(post_student_rng)

    wgb = bridge.WordGroupBuilder(tokenizer)
    config = {
        "status": "DENSECORRUPTION_PRESERVATION_CONFIG",
        "created_utc": now(),
        "method": "ms_acquisition_plus_densecorruption_nonlabel_parent_distillation_clean_rng",
        "acquisition": "research (M,S): dense second-view masks with sparse labels",
        "preservation_rendering": "dense second-view corruption on full packed Qwen pair rows",
        "preservation_target_support": "KL at dense-masked positions that are not sparse acquisition labels",
        "preservation_scientific_scope": "joint input-rendering and target-support intervention; not a pure geometry isolation",
        "matched_geometry_diagnostic": "experiments/archive/functional_learning/data/preservation_geometry_matched_diagnostic_full_v2/preservation_geometry_matched_diagnostic.json",
        "support_profile": "experiments/archive/functional_learning/data/geometry_arm_support_profile/geometry_arm_support_profile.json",
        "implemented_kl_direction": "KL(teacher || student)",
        "pres_student_mode": str(args.pres_student_mode),
        "lambda_pres": float(args.lambda_pres),
        "kl_temperature": float(args.kl_temperature),
        "focus_lambda": float(args.focus_lambda),
        "focus_prob": float(args.focus_prob),
        "max_focus_groups_per_row": int(args.max_focus_groups_per_row),
        "max_dense_mask_groups": int(args.max_dense_mask_groups),
        "mask_prob": float(args.mask_prob),
        "train_seed": int(args.train_seed),
        "acquisition_global_seed": acq_seed,
        "rng_controls": {
            "global_seed_independent_of_lambda_pres": True,
            "restore_rng_after_teacher_load": bool(args.restore_rng_after_teacher_load),
            "restore_rng_after_preservation_forward": bool(args.restore_rng_after_preservation_forward),
            "fork_preservation_forward_rng": bool(args.fork_preservation_forward_rng),
            "preservation_forward_seed_family": str(args.preservation_forward_seed_family),
            "post_student_setup_rng_digest": post_student_rng_digest,
        },
        "max_updates": int(args.max_updates),
        "parent_path": rel(bridge.PARENT_PATH),
        "student_identity": ident,
        "teacher_identity": teacher_ident,
        "optimizer": {"trainable_tensors": opt_info["trainable_tensors"], "trainable_params": opt_info["trainable_params"]},
        "exposure_accounting": "prefix words plus auxiliary preservation student presentations for Qwen rows; teacher inference is compute not exposure",
    }
    (out_dir / "train_config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"event": "train_start", "method": config["method"], "rows": len(rows), "prefix_words": prefix_info.get("prefix_words"), "device": str(device), "acquisition_seed": acq_seed}, ensure_ascii=False), flush=True)

    cursor = 0
    cum_words = 0
    cum_pres_words = 0
    logs: List[Dict[str, Any]] = []
    checkpoints: List[Dict[str, Any]] = []
    rng_records: List[Dict[str, Any]] = []
    t0 = time.time()
    micro_bs = int(args.micro_batch)
    T = float(args.kl_temperature)
    lambda_pres = float(args.lambda_pres)
    lam_f = float(args.focus_lambda)
    lam_o = 1.0 - lam_f
    model.train()

    for update_i in range(int(args.max_updates)):
        macro_rows: List[Dict[str, Any]] = []
        words = 0
        while cursor < len(rows) and words < int(args.words_per_update):
            r = rows[cursor]
            macro_rows.append(r)
            words += int(r.get("words", s64.wc(r.get("text", ""))))
            cursor += 1
        if not macro_rows:
            print(json.dumps({"event": "data_exhausted", "update": update_i}, ensure_ascii=False), flush=True)
            break

        schedule_idx = int(args.schedule_offset) + update_i
        lr = s64.lr_at_update(schedule_idx, int(args.schedule_total), int(args.warmup), float(args.lr))
        for pg in opt.param_groups:
            pg["lr"] = lr

        rng_before_acq = clean.rng_digest() if update_i < int(args.rng_record_updates) else None
        examples, prep = s64.prepare_macro(
            macro_rows, tokenizer, int(args.seq_length), wgb, objective,
            int(args.train_seed), float(args.mask_prob), float(args.focus_prob), int(args.max_focus_groups_per_row)
        )
        focus_n = int(prep["focus_target_tokens"])
        ord_n = int(prep["ordinary_target_tokens"])
        n_acq = focus_n + ord_n
        if n_acq <= 0 or focus_n <= 0 or ord_n <= 0:
            raise RuntimeError(f"No acquisition targets at update {update_i}: {prep}")

        pres_examples, pres_n, pres_words, pres_stats = prepare_densecorruption_nonlabel_preservation(
            macro_rows, tokenizer, int(args.seq_length), wgb, int(args.train_seed), float(args.mask_prob),
            float(args.focus_prob), int(args.max_focus_groups_per_row), int(args.max_dense_mask_groups)
        )

        opt.zero_grad(set_to_none=True)
        focus_sum = 0.0
        ord_sum = 0.0
        focus_seen = 0
        ord_seen = 0
        pres_kl_sum = 0.0
        pres_seen = 0
        model.train()

        for start in range(0, len(examples), micro_bs):
            mb = examples[start:start + micro_bs]
            inp = torch.stack([x["input_ids"] for x in mb]).to(device)
            att = torch.stack([x["attention_mask"] for x in mb]).to(device)
            lab = torch.stack([x["labels"] for x in mb]).to(device)
            cf = torch.stack([(x["labels"] != -100) if x["component"] == "focus" else torch.zeros_like(x["labels"], dtype=torch.bool) for x in mb]).to(device)
            co = torch.stack([(x["labels"] != -100) if x["component"] == "ordinary" else torch.zeros_like(x["labels"], dtype=torch.bool) for x in mb]).to(device)
            out = model(input_ids=inp, attention_mask=att)
            V = out.logits.shape[-1]
            ce = F.cross_entropy(out.logits.reshape(-1, V), lab.reshape(-1), ignore_index=-100, reduction="none").view_as(lab)
            fs = ce[cf].sum() if cf.any() else torch.tensor(0.0, device=device)
            os_ = ce[co].sum() if co.any() else torch.tensor(0.0, device=device)
            loss = torch.tensor(0.0, device=device)
            if lam_f != 0.0:
                loss = loss + lam_f * (fs / float(focus_n))
            if lam_o != 0.0:
                loss = loss + lam_o * (os_ / float(ord_n))
            if torch.isnan(loss) or torch.isinf(loss):
                raise RuntimeError(f"bad acquisition loss at update {update_i}")
            loss.backward()
            focus_sum += float(fs.detach().cpu())
            ord_sum += float(os_.detach().cpu())
            focus_seen += int(cf.sum().detach().cpu())
            ord_seen += int(co.sum().detach().cpu())
            del inp, att, lab, cf, co, out, ce, fs, os_, loss

        rng_after_acq = clean.rng_digest() if update_i < int(args.rng_record_updates) else None
        rng_saved_for_pres = clean.rng_state() if bool(args.restore_rng_after_preservation_forward) else None
        pres_forward_seed = None
        if bool(args.fork_preservation_forward_rng):
            pres_forward_seed = int(s64.stable_seed(str(args.preservation_forward_seed_family), int(args.train_seed), update_i))
            random.seed(pres_forward_seed)
            torch.manual_seed(pres_forward_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(pres_forward_seed)
        rng_before_pres = clean.rng_digest() if update_i < int(args.rng_record_updates) else None

        if pres_n > 0 and lambda_pres > 0.0 and pres_examples:
            if teacher is None:
                raise RuntimeError("lambda_pres > 0 requires teacher")
            clean.set_student_pres_mode(model, str(args.pres_student_mode))
            teacher.eval()
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
                        raise RuntimeError(f"bad preservation loss at update {update_i}")
                    p_loss.backward()
                    pres_kl_sum += float(kl.detach().cpu())
                    pres_seen += int(mask.sum().detach().cpu())
                del p_inp, p_att, p_lab, t_out, s_out
                try:
                    del mask, s_logits, t_logits, s_lp, t_p, kl, p_loss
                except NameError:
                    pass
            model.train()

        rng_after_pres_before_restore = clean.rng_digest() if update_i < int(args.rng_record_updates) else None
        if rng_saved_for_pres is not None:
            clean.set_rng_state(rng_saved_for_pres)
        rng_after_pres_restore = clean.rng_digest() if update_i < int(args.rng_record_updates) else None

        if focus_seen != focus_n or ord_seen != ord_n:
            raise RuntimeError(f"target mismatch update {update_i}: prep=({focus_n},{ord_n}) seen=({focus_seen},{ord_seen})")
        if lambda_pres > 0.0 and pres_seen != pres_n:
            raise RuntimeError(f"preservation target mismatch update {update_i}: prep={pres_n} seen={pres_seen}")

        grad_norm = float(torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], float(args.max_grad_norm)).detach().cpu())
        opt.step()
        cum_words += words
        cum_pres_words += pres_words if lambda_pres > 0.0 else 0
        focus_loss = float(focus_sum / focus_n) if focus_n else 0.0
        ordinary_loss = float(ord_sum / ord_n) if ord_n else 0.0
        pres_kl_mean = float(pres_kl_sum / pres_n) if pres_n else 0.0
        pooled_ce = float((focus_sum + ord_sum) / max(1, n_acq))
        optimized_acq = float(lam_f * focus_loss + lam_o * ordinary_loss)
        optimized_total = float(optimized_acq + lambda_pres * pres_kl_mean)

        log = {
            "update": update_i + 1,
            "schedule_idx": schedule_idx,
            "lr": lr,
            "rows": len(macro_rows),
            "words": words,
            "cum_words": cum_words,
            "pres_words": pres_words,
            "cum_pres_words_counted": cum_pres_words,
            "cum_words_total_counted": cum_words + cum_pres_words,
            "pres_targets": pres_n,
            "pres_targets_seen": pres_seen,
            "pres_kl_mean": pres_kl_mean,
            "pres_support_counts": pres_stats.get("counts", {}),
            "focus_loss": focus_loss,
            "ordinary_loss": ordinary_loss,
            "pooled_ce": pooled_ce,
            "optimized_acq": optimized_acq,
            "optimized_total": optimized_total,
            "lambda_pres": lambda_pres,
            "kl_temperature": T,
            "pres_student_mode": str(args.pres_student_mode),
            "focus_target_tokens": focus_n,
            "ordinary_target_tokens": ord_n,
            "qwen_rows": int(prep["qwen_rows"]),
            "qwen_focus_rows": int(prep["qwen_focus_rows"]),
            "ordinary_wwm_rows": int(prep["ordinary_wwm_rows"]),
            "focus_selected_groups": int(prep["focus_selected_groups"]),
            "focus_candidate_groups": int(prep["focus_candidate_groups"]),
            "grad_norm_preclip": grad_norm,
            "elapsed_sec": round(time.time() - t0, 1),
        }
        logs.append(log)
        if update_i < int(args.rng_record_updates):
            rng_records.append({
                "update": update_i + 1,
                "before_acquisition": rng_before_acq,
                "after_acquisition": rng_after_acq,
                "before_preservation": rng_before_pres,
                "after_preservation_before_restore": rng_after_pres_before_restore,
                "after_preservation_restore": rng_after_pres_restore,
                "restored_after_preservation": bool(args.restore_rng_after_preservation_forward),
                "forked_preservation_forward_rng": bool(args.fork_preservation_forward_rng),
                "preservation_forward_seed": pres_forward_seed,
            })
        if update_i == 0 or (update_i + 1) % int(args.log_every) == 0:
            print(json.dumps({"event": "update", **log}, ensure_ascii=False), flush=True)

        if ((update_i + 1) % int(args.checkpoint_every) == 0 or update_i + 1 == int(args.max_updates) or cursor >= len(rows)):
            ckpt_name = f"update_{update_i + 1:04d}"
            ckpt_dir = out_dir / "checkpoints" / ckpt_name
            metadata = {
                "method": config["method"],
                "objective": objective,
                "update": update_i + 1,
                "cum_words": cum_words,
                "cum_pres_words_counted": cum_pres_words,
                "cum_words_total_counted": cum_words + cum_pres_words,
                "train_seed": int(args.train_seed),
                "acquisition_global_seed": acq_seed,
                "prefix_info": prefix_info,
                "final_update": log,
                "model_identity": ident,
                "teacher_identity": teacher_ident,
                "private_scale": float(args.private_scale),
                "lambda_pres": lambda_pres,
                "kl_temperature": T,
                "pres_student_mode": str(args.pres_student_mode),
                "preservation_rendering_scope": config["preservation_scientific_scope"],
                "preservation_target_support": config["preservation_target_support"],
                "implemented_kl_direction": config["implemented_kl_direction"],
            }
            s64.save_checkpoint(model, tokenizer, ckpt_dir, metadata, float(args.private_scale))
            checkpoints.append({"update": update_i + 1, "path": rel(ckpt_dir), "optimized_total": optimized_total, "optimized_acq": optimized_acq})
            print(json.dumps({"event": "checkpoint", "update": update_i + 1, "path": rel(ckpt_dir)}, ensure_ascii=False), flush=True)
        if cursor >= len(rows):
            print(json.dumps({"event": "prefix_exhausted", "update": update_i + 1}, ensure_ascii=False), flush=True)
            break

    total_focus = sum(int(x["focus_target_tokens"]) for x in logs)
    total_ord = sum(int(x["ordinary_target_tokens"]) for x in logs)
    total_pres = sum(int(x["pres_targets"]) for x in logs)
    total_pres_w_counted = sum(int(x["pres_words"]) for x in logs) if lambda_pres > 0.0 else 0
    endpoint_exposure = PARENT_EXPOSURE_WORDS + cum_words + total_pres_w_counted
    pres_support_total = Counter()
    for x in logs:
        pres_support_total.update(x.get("pres_support_counts") or {})
    summary = {
        "status": "DENSECORRUPTION_PRESERVATION_TRAIN_DONE",
        "method": config["method"],
        "out_dir": rel(out_dir),
        "completed_updates": len(logs),
        "total_words_consumed": cum_words,
        "total_preservation_words_counted": total_pres_w_counted,
        "total_words_combined_counted": cum_words + total_pres_w_counted,
        "total_focus_targets": total_focus,
        "total_ordinary_targets": total_ord,
        "total_preservation_targets_prepared": total_pres,
        "total_preservation_support_counts": dict(pres_support_total),
        "lambda_pres": lambda_pres,
        "kl_temperature": T,
        "focus_lambda": lam_f,
        "pres_student_mode": str(args.pres_student_mode),
        "implemented_kl_direction": config["implemented_kl_direction"],
        "preservation_rendering_scope": config["preservation_scientific_scope"],
        "preservation_target_support": config["preservation_target_support"],
        "final_update": logs[-1] if logs else None,
        "checkpoints": checkpoints,
        "rng_records": rng_records,
        "elapsed_sec": round(time.time() - t0, 1),
        "model_identity": ident,
        "teacher_identity": teacher_ident,
        "exposure_accounting": {
            "parent_exposure": PARENT_EXPOSURE_WORDS,
            "prefix_words": cum_words,
            "preservation_additional_words_counted": total_pres_w_counted,
            "endpoint_exposure_conservative": endpoint_exposure,
            "endpoint_label": f"endpoint_{endpoint_exposure / 1e6:.6f}M",
            "note": "Counts auxiliary student preservation presentations for Qwen rows when lambda_pres > 0; teacher inference is compute, not student exposure.",
        },
    }
    (out_dir / "train_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    s64.write_jsonl(out_dir / "update_log.jsonl", logs)
    if rng_records:
        (out_dir / "rng_records.json").write_text(json.dumps(rng_records, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)

    del model
    if teacher is not None:
        del teacher
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tail-jsonl", type=pathlib.Path, default=s64.DEFAULT_TAIL)
    ap.add_argument("--out-dir", type=pathlib.Path, default=DEFAULT_OUT)
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
    ap.add_argument("--mask-prob", type=float, default=0.15)
    ap.add_argument("--focus-prob", type=float, default=0.35)
    ap.add_argument("--max-focus-groups-per-row", type=int, default=16)
    ap.add_argument("--max-dense-mask-groups", type=int, default=128)
    ap.add_argument("--focus-lambda", type=float, default=0.15)
    ap.add_argument("--lambda-pres", type=float, default=1.0)
    ap.add_argument("--kl-temperature", type=float, default=1.0)
    ap.add_argument("--pres-student-mode", choices=["eval", "train"], default="eval")
    ap.add_argument("--restore-rng-after-teacher-load", action="store_true", default=True)
    ap.add_argument("--no-restore-rng-after-teacher-load", dest="restore_rng_after_teacher_load", action="store_false")
    ap.add_argument("--restore-rng-after-preservation-forward", action="store_true", default=True)
    ap.add_argument("--no-restore-rng-after-preservation-forward", dest="restore_rng_after_preservation_forward", action="store_false")
    ap.add_argument("--fork-preservation-forward-rng", action="store_true", default=True)
    ap.add_argument("--no-fork-preservation-forward-rng", dest="fork_preservation_forward_rng", action="store_false")
    ap.add_argument("--preservation-forward-seed-family", type=str, default=bytes((115, 116, 101, 112, 49, 48, 54, 45, 100, 101, 110, 115, 101, 99, 111, 114, 114, 117, 112, 116, 105, 111, 110, 45, 112, 114, 101, 115, 101, 114, 118, 97, 116, 105, 111, 110, 45, 102, 111, 114, 119, 97, 114, 100)).decode('utf-8'))
    ap.add_argument("--skip-teacher-when-zero", action="store_true", default=True)
    ap.add_argument("--load-teacher-when-zero", dest="skip_teacher_when_zero", action="store_false")
    ap.add_argument("--checkpoint-every", type=int, default=40)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--rng-record-updates", type=int, default=3)
    ap.add_argument("--train-seed", type=int, default=62064)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if not (0.0 <= float(args.focus_lambda) <= 1.0):
        raise SystemExit("--focus-lambda must be in [0,1]")
    if float(args.lambda_pres) < 0:
        raise SystemExit("--lambda-pres must be >= 0")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    setup_cache(args.out_dir)
    rows, prefix_info = s64.load_prefix(args.tail_jsonl, int(args.max_updates), int(args.words_per_update))
    tokenizer = bridge.AutoTokenizer.from_pretrained(str(bridge.PARENT_PATH), local_files_only=True, use_fast=True)

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
            "correspondence_focus_weighted", int(args.train_seed), float(args.mask_prob),
            float(args.focus_prob), int(args.max_focus_groups_per_row)
        )
        pres_ex, pres_n, pres_w, pres_stats = prepare_densecorruption_nonlabel_preservation(
            macro_rows, tokenizer, int(args.seq_length), wgb, int(args.train_seed), float(args.mask_prob),
            float(args.focus_prob), int(args.max_focus_groups_per_row), int(args.max_dense_mask_groups)
        )
        dry = {
            "status": "DENSECORRUPTION_PRESERVATION_DRY_RUN_DONE",
            "created_utc": now(),
            "macro_rows": len(macro_rows),
            "acq_examples": len(acq_ex),
            "acq_focus_targets": int(acq_prep["focus_target_tokens"]),
            "acq_ordinary_targets": int(acq_prep["ordinary_target_tokens"]),
            "acq_qwen_rows": int(acq_prep["qwen_rows"]),
            "pres_examples": len(pres_ex),
            "pres_targets": pres_n,
            "pres_words": pres_w,
            "pres_support_counts": pres_stats.get("counts", {}),
            "lambda_pres": float(args.lambda_pres),
            "pres_student_mode": str(args.pres_student_mode),
            "acquisition_global_seed": clean.seed_original_acquisition("correspondence_focus_weighted", int(args.train_seed), float(args.focus_lambda)),
            "preservation_scope": "dense-corrupted full Qwen row; KL at dense masked non-label positions; joint rendering/support intervention",
        }
        (args.out_dir / "dry_run.json").write_text(json.dumps(dry, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(json.dumps(dry, indent=2, ensure_ascii=False), flush=True)
        return

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    train_dense_preservation(rows, prefix_info, args, device, tokenizer)


if __name__ == "__main__":
    main()
