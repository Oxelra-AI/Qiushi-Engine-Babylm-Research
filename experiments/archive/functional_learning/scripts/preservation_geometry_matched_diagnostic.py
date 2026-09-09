#!/usr/bin/env python3
"""research: matched preservation-geometry diagnostic before a full arm.

Scientific purpose
------------------
Clean preservation applies KL(teacher || student) on an ordinary 15% WWM
rendering of full Qwen pair rows.  A proposed corruption-geometry arm applies
parent KL on dense-corrupted Qwen second-view states, especially masked positions
that were *not* sparse acquisition labels.  A full training arm changes both the
input rendering and target support, so this diagnostic first holds target
positions fixed where possible.

At the acquisition-shifted exact (M,S) checkpoint, for selected real macro
updates from the legal unchanged-Qwen prefix, compare:

  1. ordinary-WWM input rendering, KL at positions that are also dense non-label
     masked positions;
  2. dense-corrupted input rendering, KL at exactly the same positions;
  3. full ordinary-preservation support (clean-like);
  4. full dense-nonlabel support (proposed policy-like).

For each preservation branch the script records teacher uncertainty, KL value,
private-adapter gradient norm, and cosine/dot alignment with the fixed
acquisition gradient for the same selected macros.  Gradients are computed in
student eval mode to isolate prediction-state geometry from dropout.  This is a
local diagnostic, not a training result and not BabyLM leaderboard scoring.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse
import json
import math
import os
import pathlib
import random
import sys
import time
from collections import Counter
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn.functional as F

ROOT = _public_path('.')
STUDY = _public_path('experiments/archive/functional_learning')
SCRIPTS = _public_path('experiments/archive/functional_learning/scripts')
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import real_stream_train_weighted as s64  # noqa: E402
import corrected_bridge_trainer as bridge  # noqa: E402

DEFAULT_OUT = _public_path('experiments/archive/functional_learning/data/preservation_geometry_matched_diagnostic')
DEFAULT_STUDENT = _public_path('experiments/archive/functional_learning/data/densemask_sparselabel_train_seed62064/correspondence_focus_weighted/checkpoints/update_0080')


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
    hf = out_dir / "hf_cache"
    os.environ["HF_HOME"] = str(hf.resolve())
    os.environ["HF_HUB_CACHE"] = str((hf / "hub").resolve())
    os.environ["HF_DATASETS_CACHE"] = str((hf / "datasets").resolve())
    os.environ["TRANSFORMERS_CACHE"] = str((hf / "transformers").resolve())
    os.environ["HF_MODULES_CACHE"] = str((hf / "modules").resolve())
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    for sub in ["hub", "datasets", "transformers", "modules"]:
        (hf / sub).mkdir(parents=True, exist_ok=True)


def parse_macro_indices(s: str) -> List[int]:
    out: List[int] = []
    for part in str(s).replace(",", " ").split():
        if part.strip():
            out.append(int(part))
    return sorted(dict.fromkeys(out))


def split_macros(rows: List[Dict[str, Any]], max_updates: int, words_per_update: int) -> List[Dict[str, Any]]:
    macros: List[Dict[str, Any]] = []
    cursor = 0
    for update_i in range(int(max_updates)):
        macro_rows: List[Dict[str, Any]] = []
        words = 0
        while cursor < len(rows) and words < int(words_per_update):
            r = rows[cursor]
            macro_rows.append(r)
            words += int(r.get("words", s64.wc(r.get("text", ""))))
            cursor += 1
        if not macro_rows:
            break
        macros.append({
            "update_index0": update_i,
            "rows": macro_rows,
            "words": words,
            "qwen_rows": sum(1 for r in macro_rows if r.get("source") == "qwen_pair_packed" and bool(r.get("qwen_pair_segments"))),
        })
    return macros


def densemask_sparse_label_row(row: Dict[str, Any], tok: Dict[str, Any], tokenizer, seed: int,
                               focus_prob: float, max_focus_groups_per_row: int,
                               max_mask_groups: int) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
    """Local copy of research dense-mask/sparse-label constructor, without atexit side effects."""
    input_ids: torch.Tensor = tok["input_ids"]
    offsets: torch.Tensor = tok["offsets"]
    attention_mask: torch.Tensor = tok["attention_mask"]
    masked = input_ids.clone()
    labels = torch.full_like(input_ids, -100)
    all_groups: List[Dict[str, Any]] = []
    text = str(row.get("text", ""))
    for seg_i, seg in enumerate(row.get("qwen_pair_segments") or []):
        view_text = str(seg.get("view_text", ""))
        vstart = int(seg.get("view_start", -1))
        vend = int(seg.get("view_end", -1))
        if vstart < 0 or vend <= vstart or vend > len(text):
            continue
        if text[vstart:vend] != view_text:
            continue
        for a, b, word in s64.content_word_spans(view_text):
            pos = s64.locate_positions(offsets, vstart + a, vstart + b)
            pos = [p for p in pos if int(attention_mask[p]) == 1]
            if pos:
                all_groups.append({
                    "segment_index": seg_i,
                    "pair_id": seg.get("pair_id"),
                    "view_kind": seg.get("view_kind"),
                    "candidate_kind": seg.get("candidate_kind"),
                    "word": word,
                    "positions": pos,
                })
    rng = random.Random(int(seed))
    label_candidates = list(all_groups)
    if len(label_candidates) > int(max_focus_groups_per_row):
        label_candidates = rng.sample(label_candidates, int(max_focus_groups_per_row))
    label_groups = [g for g in label_candidates if rng.random() < float(focus_prob)]
    if not label_groups and label_candidates:
        label_groups = [rng.choice(label_candidates)]

    mask_groups = list(all_groups)
    if len(mask_groups) > int(max_mask_groups):
        mrng = random.Random(s64.stable_seed("densemask-mask-cap", seed, bridge.row_key(row), int(max_mask_groups)))
        sampled = mrng.sample(mask_groups, int(max_mask_groups))
        for g in label_groups:
            if g not in sampled:
                sampled.append(g)
        mask_groups = sampled

    label_pos = set()
    mask_pos = set()
    kind_counts = Counter()
    for g in mask_groups:
        for p in g["positions"]:
            if int(attention_mask[p]) == 1:
                masked[p] = int(tokenizer.mask_token_id)
                mask_pos.add(int(p))
    for g in label_groups:
        kind_counts[str(g.get("candidate_kind", "unknown"))] += 1
        for p in g["positions"]:
            if int(attention_mask[p]) == 1:
                labels[p] = input_ids[p]
                masked[p] = int(tokenizer.mask_token_id)
                mask_pos.add(int(p))
                label_pos.add(int(p))
    return masked, labels, {
        "n_candidate_groups": len(all_groups),
        "n_label_candidate_groups_after_cap": len(label_candidates),
        "n_selected_groups": len(label_groups),
        "n_target_tokens": len(label_pos),
        "n_masked_groups": len(mask_groups),
        "n_masked_tokens": len(mask_pos),
        "selected_candidate_kind_counts": dict(kind_counts),
        "mask_positions": sorted(mask_pos),
        "label_positions": sorted(label_pos),
    }


def install_densemask_patch(max_mask_groups: int) -> None:
    def patched(row: Dict[str, Any], tok: Dict[str, Any], tokenizer, seed: int,
                focus_prob: float, max_focus_groups_per_row: int):
        return densemask_sparse_label_row(row, tok, tokenizer, seed, focus_prob, max_focus_groups_per_row, max_mask_groups)
    s64.apply_view_focus_row = patched


def labels_at_positions(input_ids: torch.Tensor, positions: Iterable[int]) -> torch.Tensor:
    labels = torch.full_like(input_ids, -100)
    for p in positions:
        p = int(p)
        if 0 <= p < int(input_ids.numel()):
            labels[p] = input_ids[p]
    return labels


def build_preservation_renderings(macro_rows: List[Dict[str, Any]], tokenizer, wgb, args: argparse.Namespace) -> Tuple[Dict[str, List[Dict[str, Any]]], Dict[str, Any]]:
    ordinary_common: List[Dict[str, Any]] = []
    dense_common: List[Dict[str, Any]] = []
    ordinary_full: List[Dict[str, Any]] = []
    dense_nonlabel_full: List[Dict[str, Any]] = []
    stats = Counter()
    kind_counts = Counter()
    row_records: List[Dict[str, Any]] = []
    mask_id = int(tokenizer.mask_token_id)
    for row in macro_rows:
        is_qwen = row.get("source") == "qwen_pair_packed" and bool(row.get("qwen_pair_segments"))
        if not is_qwen:
            continue
        stats["qwen_rows"] += 1
        tok = bridge.tokenize_row(row, tokenizer, int(args.seq_length), wgb)
        row_key = bridge.row_key(row)
        pres_seed = s64.stable_seed("preservation-wwm-std", int(args.train_seed), row_key)
        ordinary_masked, ordinary_labels, ordinary_stats = bridge.apply_wwm_row(
            tok["input_ids"], tok["attention_mask"], tok["word_group"], tokenizer, pres_seed, float(args.mask_prob)
        )
        acq_seed = s64.stable_seed("view-focus", int(args.train_seed), row_key)
        dense_masked, sparse_labels, dense_stats = densemask_sparse_label_row(
            row, tok, tokenizer, acq_seed, float(args.focus_prob), int(args.max_focus_groups_per_row), int(args.max_dense_mask_groups)
        )
        att = tok["attention_mask"]
        orig = tok["input_ids"]
        ordinary_pos = {int(i) for i in torch.nonzero(ordinary_labels != -100, as_tuple=False).view(-1).tolist()}
        dense_label_pos = {int(i) for i in torch.nonzero(sparse_labels != -100, as_tuple=False).view(-1).tolist()}
        dense_mask_pos = {int(i) for i in torch.nonzero((dense_masked == mask_id) & (orig != mask_id) & (att == 1), as_tuple=False).view(-1).tolist()}
        dense_nonlabel_pos = dense_mask_pos - dense_label_pos
        common_pos = sorted(ordinary_pos & dense_nonlabel_pos)
        stats["ordinary_full_targets"] += len(ordinary_pos)
        stats["dense_masked_tokens"] += len(dense_mask_pos)
        stats["dense_sparse_label_targets"] += len(dense_label_pos)
        stats["dense_nonlabel_targets"] += len(dense_nonlabel_pos)
        stats["common_targets"] += len(common_pos)
        stats["candidate_groups"] += int(dense_stats.get("n_candidate_groups", 0))
        stats["dense_masked_groups"] += int(dense_stats.get("n_masked_groups", 0))
        stats["dense_label_groups"] += int(dense_stats.get("n_selected_groups", 0))
        kind_counts.update(dense_stats.get("selected_candidate_kind_counts") or {})
        words = int(row.get("words", s64.wc(row.get("text", ""))))
        if ordinary_pos:
            ordinary_full.append({
                "input_ids": ordinary_masked,
                "attention_mask": att,
                "labels": ordinary_labels,
                "row_key": row_key,
                "words": words,
            })
        if dense_nonlabel_pos:
            dense_nonlabel_full.append({
                "input_ids": dense_masked,
                "attention_mask": att,
                "labels": labels_at_positions(orig, sorted(dense_nonlabel_pos)),
                "row_key": row_key,
                "words": words,
            })
        if common_pos:
            common_labels = labels_at_positions(orig, common_pos)
            ordinary_common.append({
                "input_ids": ordinary_masked,
                "attention_mask": att,
                "labels": common_labels,
                "row_key": row_key,
                "words": words,
            })
            dense_common.append({
                "input_ids": dense_masked,
                "attention_mask": att,
                "labels": common_labels,
                "row_key": row_key,
                "words": words,
            })
        row_records.append({
            "row_key": row_key,
            "words": words,
            "ordinary_targets": len(ordinary_pos),
            "dense_masked_tokens": len(dense_mask_pos),
            "dense_sparse_label_targets": len(dense_label_pos),
            "dense_nonlabel_targets": len(dense_nonlabel_pos),
            "common_targets": len(common_pos),
        })
    stats["ordinary_common_examples"] = len(ordinary_common)
    stats["dense_common_examples"] = len(dense_common)
    stats["ordinary_full_examples"] = len(ordinary_full)
    stats["dense_nonlabel_full_examples"] = len(dense_nonlabel_full)
    return {
        "ordinary_common": ordinary_common,
        "dense_common": dense_common,
        "ordinary_full": ordinary_full,
        "dense_nonlabel_full": dense_nonlabel_full,
    }, {
        "counts": dict(stats),
        "selected_label_kind_counts": dict(kind_counts),
        "row_records_head": row_records[:12],
    }


def load_endpoint(model_path: pathlib.Path, device: torch.device, private_scale: float):
    endpoint = pathlib.Path(model_path)
    model, missing, unexpected = bridge.base_loader.load_model(endpoint, device, 128, float(private_scale))
    if hasattr(model, "set_private_enabled"):
        model.set_private_enabled(True)
    ident = bridge.model_identity(model)
    return model, {"path": rel(endpoint), "identity": ident, "missing": len(missing), "unexpected": len(unexpected)}


def set_private_trainable(model) -> Tuple[List[Tuple[str, torch.nn.Parameter]], Dict[str, Any]]:
    for _, p in model.named_parameters():
        p.requires_grad_(False)
    private: List[Tuple[str, torch.nn.Parameter]] = []
    for n, p in model.named_parameters():
        if ".private_adapter." in n:
            p.requires_grad_(True)
            private.append((n, p))
    if not private:
        raise RuntimeError("No private adapter params")
    return private, {
        "private_tensors": len(private),
        "private_params": int(sum(p.numel() for _, p in private)),
        "names_head": [n for n, _ in private[:8]],
    }


def zero_grads(params: List[Tuple[str, torch.nn.Parameter]]) -> None:
    for _, p in params:
        p.grad = None


def grad_vector(params: List[Tuple[str, torch.nn.Parameter]]) -> torch.Tensor:
    pieces = []
    for _, p in params:
        if p.grad is None:
            pieces.append(torch.zeros(p.numel(), dtype=torch.float32))
        else:
            pieces.append(p.grad.detach().float().cpu().reshape(-1).clone())
    if not pieces:
        return torch.zeros(0, dtype=torch.float32)
    return torch.cat(pieces)


def vec_stats(v: torch.Tensor) -> Dict[str, Any]:
    if v.numel() == 0:
        return {"l2": 0.0, "linf": 0.0, "nonzero": 0}
    return {
        "l2": float(torch.linalg.norm(v).item()),
        "linf": float(v.abs().max().item()),
        "mean_abs": float(v.abs().mean().item()),
        "nonzero": int((v != 0).sum().item()),
        "numel": int(v.numel()),
    }


def cosine(a: torch.Tensor, b: torch.Tensor) -> Optional[float]:
    if a.numel() == 0 or b.numel() == 0:
        return None
    na = float(torch.linalg.norm(a).item())
    nb = float(torch.linalg.norm(b).item())
    if na <= 0.0 or nb <= 0.0:
        return None
    return float(torch.dot(a, b).item() / (na * nb))


def dot(a: torch.Tensor, b: torch.Tensor) -> Optional[float]:
    if a.numel() == 0 or b.numel() == 0:
        return None
    return float(torch.dot(a, b).item())


def count_targets(examples: List[Dict[str, Any]]) -> int:
    return int(sum(int((x["labels"] != -100).sum().item()) for x in examples))


def batches(examples: List[Dict[str, Any]], batch_size: int):
    for start in range(0, len(examples), int(batch_size)):
        yield examples[start:start + int(batch_size)]


def grad_acquisition(student, private_params, macros: List[Dict[str, Any]], tokenizer, wgb, args: argparse.Namespace,
                     mode: str) -> Tuple[torch.Tensor, Dict[str, Any]]:
    assert mode in {"full_weighted", "focus_mean", "ordinary_mean"}
    zero_grads(private_params)
    student.eval()
    records = []
    total_focus = 0
    total_ord = 0
    total_loss_value = 0.0
    nonempty = 0
    prepared_stats = []
    for macro in macros:
        examples, prep = s64.prepare_macro(
            macro["rows"], tokenizer, int(args.seq_length), wgb, "correspondence_focus_weighted",
            int(args.train_seed), float(args.mask_prob), float(args.focus_prob), int(args.max_focus_groups_per_row)
        )
        focus_n = int(prep["focus_target_tokens"])
        ord_n = int(prep["ordinary_target_tokens"])
        if mode == "full_weighted" and (focus_n <= 0 or ord_n <= 0):
            continue
        if mode == "focus_mean" and focus_n <= 0:
            continue
        if mode == "ordinary_mean" and ord_n <= 0:
            continue
        nonempty += 1
        total_focus += focus_n
        total_ord += ord_n
        focus_sum = 0.0
        ord_sum = 0.0
        for mb in batches(examples, int(args.micro_batch)):
            inp = torch.stack([x["input_ids"] for x in mb]).to(args.device_obj)
            att = torch.stack([x["attention_mask"] for x in mb]).to(args.device_obj)
            lab = torch.stack([x["labels"] for x in mb]).to(args.device_obj)
            cf = torch.stack([(x["labels"] != -100) if x.get("component") == "focus" else torch.zeros_like(x["labels"], dtype=torch.bool) for x in mb]).to(args.device_obj)
            co = torch.stack([(x["labels"] != -100) if x.get("component") == "ordinary" else torch.zeros_like(x["labels"], dtype=torch.bool) for x in mb]).to(args.device_obj)
            out = student(input_ids=inp, attention_mask=att)
            V = out.logits.shape[-1]
            ce = F.cross_entropy(out.logits.reshape(-1, V), lab.reshape(-1), ignore_index=-100, reduction="none").view_as(lab)
            fs = ce[cf].sum() if cf.any() else torch.tensor(0.0, device=args.device_obj)
            os_ = ce[co].sum() if co.any() else torch.tensor(0.0, device=args.device_obj)
            if mode == "full_weighted":
                loss = float(args.focus_lambda) * (fs / float(focus_n)) + (1.0 - float(args.focus_lambda)) * (os_ / float(ord_n))
            elif mode == "focus_mean":
                # In focus-only mode, ordinary microbatches contain no selected focus
                # positions. They still contribute to the bookkeeping sums but should
                # not call backward on a constant zero tensor.
                loss = (fs / float(focus_n)) if cf.any() else None
            else:
                loss = (os_ / float(ord_n)) if co.any() else None
            if loss is not None:
                (loss / float(max(1, len(macros)))).backward()
            focus_sum += float(fs.detach().cpu())
            ord_sum += float(os_.detach().cpu())
            del inp, att, lab, cf, co, out, ce, fs, os_, loss
        focus_loss = focus_sum / max(1, focus_n)
        ord_loss = ord_sum / max(1, ord_n)
        if mode == "full_weighted":
            macro_loss = float(args.focus_lambda) * focus_loss + (1.0 - float(args.focus_lambda)) * ord_loss
        elif mode == "focus_mean":
            macro_loss = focus_loss
        else:
            macro_loss = ord_loss
        total_loss_value += macro_loss
        records.append({
            "update_index0": int(macro["update_index0"]),
            "rows": len(macro["rows"]),
            "qwen_rows": int(macro["qwen_rows"]),
            "focus_targets": focus_n,
            "ordinary_targets": ord_n,
            "focus_loss": focus_loss,
            "ordinary_loss": ord_loss,
            "loss_used": macro_loss,
        })
        prepared_stats.append(prep)
    v = grad_vector(private_params)
    zero_grads(private_params)
    return v, {
        "mode": mode,
        "nonempty_macros": nonempty,
        "loss_mean_over_selected_macros": float(total_loss_value / max(1, nonempty)),
        "focus_targets_total": total_focus,
        "ordinary_targets_total": total_ord,
        "macro_records": records,
        "grad": vec_stats(v),
    }


def grad_kl(student, teacher, private_params, examples_by_macro: List[List[Dict[str, Any]]], args: argparse.Namespace) -> Tuple[torch.Tensor, Dict[str, Any]]:
    zero_grads(private_params)
    student.eval()
    teacher.eval()
    nonempty_macros = [ex for ex in examples_by_macro if count_targets(ex) > 0]
    macro_records = []
    totals = Counter()
    sums = Counter()
    T = float(args.kl_temperature)
    for macro_i, examples in enumerate(examples_by_macro):
        n_macro = count_targets(examples)
        if n_macro <= 0:
            macro_records.append({"macro_slot": macro_i, "targets": 0})
            continue
        kl_sum = 0.0
        kl_st_sum = 0.0
        teacher_entropy_sum = 0.0
        teacher_top1_sum = 0.0
        teacher_target_ce_sum = 0.0
        student_target_ce_sum = 0.0
        teacher_rank_sum = 0.0
        student_rank_sum = 0.0
        for mb in batches(examples, int(args.micro_batch)):
            inp = torch.stack([x["input_ids"] for x in mb]).to(args.device_obj)
            att = torch.stack([x["attention_mask"] for x in mb]).to(args.device_obj)
            lab = torch.stack([x["labels"] for x in mb]).to(args.device_obj)
            mask = lab != -100
            if not mask.any():
                del inp, att, lab, mask
                continue
            labels = lab[mask]
            with torch.no_grad():
                t_out = teacher(input_ids=inp, attention_mask=att)
                t_logits = t_out.logits[mask]
                t_lp_uncal = F.log_softmax(t_logits, dim=-1)
                t_p_uncal = torch.exp(t_lp_uncal)
                ent = -(t_p_uncal * t_lp_uncal).sum(dim=-1)
                top1 = t_p_uncal.max(dim=-1).values
                V = t_logits.shape[-1]
                t_ce = F.cross_entropy(t_logits.reshape(-1, V), labels.reshape(-1), reduction="none")
                t_lab_logits = t_logits.gather(1, labels.view(-1, 1)).squeeze(1)
                t_rank = (t_logits > t_lab_logits.view(-1, 1)).sum(dim=1).float() + 1.0
            s_out = student(input_ids=inp, attention_mask=att)
            s_logits = s_out.logits[mask]
            s_lp = F.log_softmax(s_logits / T, dim=-1)
            t_lp = F.log_softmax(t_logits.detach() / T, dim=-1)
            s_p = torch.exp(s_lp)
            t_p = torch.exp(t_lp)
            kts = F.kl_div(s_lp, t_p, reduction="sum")
            kst = F.kl_div(t_lp, s_p, reduction="sum")
            if T != 1.0:
                kts = kts * (T * T)
                kst = kst * (T * T)
            V = s_logits.shape[-1]
            s_ce = F.cross_entropy(s_logits.reshape(-1, V), labels.reshape(-1), reduction="none")
            s_lab_logits = s_logits.gather(1, labels.view(-1, 1)).squeeze(1)
            s_rank = (s_logits > s_lab_logits.view(-1, 1)).sum(dim=1).float() + 1.0
            # Average per macro, then average selected nonempty macros. This mirrors
            # per-update mean-loss normalization while keeping selected macros equally weighted.
            (kts / float(n_macro) / float(max(1, len(nonempty_macros)))).backward()
            nt = int(labels.numel())
            totals["tokens"] += nt
            kl_sum += float(kts.detach().cpu())
            kl_st_sum += float(kst.detach().cpu())
            teacher_entropy_sum += float(ent.sum().detach().cpu())
            teacher_top1_sum += float(top1.sum().detach().cpu())
            teacher_target_ce_sum += float(t_ce.sum().detach().cpu())
            student_target_ce_sum += float(s_ce.sum().detach().cpu())
            teacher_rank_sum += float(t_rank.sum().detach().cpu())
            student_rank_sum += float(s_rank.sum().detach().cpu())
            del inp, att, lab, mask, labels, t_out, t_logits, t_lp_uncal, t_p_uncal, ent, top1, t_ce, t_lab_logits, t_rank
            del s_out, s_logits, s_lp, t_lp, s_p, t_p, kts, kst, s_ce, s_lab_logits, s_rank
        macro_records.append({
            "macro_slot": macro_i,
            "targets": n_macro,
            "kl_teacher_to_student_mean": kl_sum / max(1, n_macro),
            "kl_student_to_teacher_mean": kl_st_sum / max(1, n_macro),
            "teacher_entropy_mean": teacher_entropy_sum / max(1, n_macro),
            "teacher_top1_prob_mean": teacher_top1_sum / max(1, n_macro),
            "teacher_target_ce_mean": teacher_target_ce_sum / max(1, n_macro),
            "student_target_ce_mean": student_target_ce_sum / max(1, n_macro),
            "teacher_rank_mean": teacher_rank_sum / max(1, n_macro),
            "student_rank_mean": student_rank_sum / max(1, n_macro),
        })
        sums["kl_ts"] += kl_sum
        sums["kl_st"] += kl_st_sum
        sums["teacher_entropy"] += teacher_entropy_sum
        sums["teacher_top1"] += teacher_top1_sum
        sums["teacher_target_ce"] += teacher_target_ce_sum
        sums["student_target_ce"] += student_target_ce_sum
        sums["teacher_rank"] += teacher_rank_sum
        sums["student_rank"] += student_rank_sum
    v = grad_vector(private_params)
    zero_grads(private_params)
    n_tok = int(totals["tokens"])
    return v, {
        "nonempty_macros": len(nonempty_macros),
        "tokens_total": n_tok,
        "kl_temperature": T,
        "kl_teacher_to_student_token_mean": float(sums["kl_ts"] / max(1, n_tok)),
        "kl_student_to_teacher_token_mean": float(sums["kl_st"] / max(1, n_tok)),
        "teacher_entropy_token_mean": float(sums["teacher_entropy"] / max(1, n_tok)),
        "teacher_top1_prob_token_mean": float(sums["teacher_top1"] / max(1, n_tok)),
        "teacher_target_ce_token_mean": float(sums["teacher_target_ce"] / max(1, n_tok)),
        "student_target_ce_token_mean": float(sums["student_target_ce"] / max(1, n_tok)),
        "teacher_rank_token_mean": float(sums["teacher_rank"] / max(1, n_tok)),
        "student_rank_token_mean": float(sums["student_rank"] / max(1, n_tok)),
        "macro_records": macro_records,
        "grad": vec_stats(v),
    }


def alignment_block(name: str, v: torch.Tensor, refs: Dict[str, torch.Tensor]) -> Dict[str, Any]:
    out = {"grad": vec_stats(v), "alignment": {}}
    for rname, rv in refs.items():
        out["alignment"][rname] = {
            "cosine": cosine(v, rv),
            "dot": dot(v, rv),
        }
    return out


def fmt(x: Any, nd: int = 6) -> str:
    if x is None:
        return ""
    if isinstance(x, float):
        if math.isnan(x) or math.isinf(x):
            return str(x)
        return f"{x:.{nd}g}"
    return str(x)


def write_outputs(out_dir: pathlib.Path, result: Dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / "preservation_geometry_matched_diagnostic.json"
    out_md = out_dir / "preservation_geometry_matched_diagnostic.md"
    out_json.write_text(json.dumps(result, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")

    support = result["support_summary"]
    branches = result["branches"]
    lines = [
        "# research preservation-geometry matched diagnostic",
        "",
        f"Created: `{result['created_utc']}`",
        "",
        "This diagnostic is run at the exact acquisition-only `(M,S)` checkpoint. It compares clean's ordinary-WWM preservation rendering with a dense-corrupted rendering before launching any full preservation-geometry training arm.",
        "",
        "## Selected acquisition macros",
        "",
        f"- Macro indices (0-based): `{result['selected_macro_indices0']}`",
        f"- Selected rows: `{support['selected_rows']}`; Qwen rows: `{support['qwen_rows']}`; selected words: `{support['selected_words']}`",
        "",
        "## Target support",
        "",
        "| quantity | value |",
        "|---|---:|",
    ]
    for k in ["ordinary_full_targets", "dense_masked_tokens", "dense_sparse_label_targets", "dense_nonlabel_targets", "common_targets", "ordinary_common_examples", "ordinary_full_examples", "dense_nonlabel_full_examples"]:
        lines.append(f"| {k} | {support.get(k, '')} |")
    lines += [
        "",
        "The common-support comparison uses `common_targets = ordinary-WWM target positions ∩ dense-corruption masked non-label positions`; KL is evaluated at the same token positions under the two input renderings.",
        "",
        "## Acquisition gradients",
        "",
        "| gradient | loss mean | L2 norm | focus targets | ordinary targets |",
        "|---|---:|---:|---:|---:|",
    ]
    for name in ["acquisition_full_weighted", "acquisition_focus_mean", "acquisition_ordinary_mean"]:
        rec = result["acquisition"][name]
        lines.append(f"| {name} | {fmt(rec.get('loss_mean_over_selected_macros'))} | {fmt(rec['grad'].get('l2'))} | {rec.get('focus_targets_total')} | {rec.get('ordinary_targets_total')} |")
    lines += ["", "## Preservation branches", "", "| branch | tokens | KL(t||s) | teacher entropy | teacher top1 | teacher target CE | student target CE | grad L2 | cos vs full acquisition | cos vs focus acquisition | cos vs ordinary acquisition |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name in ["ordinary_common", "dense_common", "ordinary_full", "dense_nonlabel_full"]:
        rec = branches[name]
        aln = rec.get("alignment", {})
        lines.append(
            f"| {name} | {rec.get('tokens_total')} | {fmt(rec.get('kl_teacher_to_student_token_mean'))} | {fmt(rec.get('teacher_entropy_token_mean'))} | {fmt(rec.get('teacher_top1_prob_token_mean'))} | {fmt(rec.get('teacher_target_ce_token_mean'))} | {fmt(rec.get('student_target_ce_token_mean'))} | {fmt(rec['grad'].get('l2'))} | {fmt(aln.get('acquisition_full_weighted', {}).get('cosine'))} | {fmt(aln.get('acquisition_focus_mean', {}).get('cosine'))} | {fmt(aln.get('acquisition_ordinary_mean', {}).get('cosine'))} |"
        )
    lines += [
        "",
        "## Direct rendering contrast on common support",
        "",
    ]
    lines.append(json.dumps(result.get("common_rendering_contrast", {}), indent=2, ensure_ascii=False))
    lines += [
        "",
        "## Interpretation",
        "",
        result.get("interpretation", ""),
        "",
        "This file is a local gradient/uncertainty diagnostic. It does not change the repaired BabyLM coordinate and should not be used as a benchmark score.",
    ]
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "out_json": rel(out_json), "out_md": rel(out_md)}, indent=2), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tail-jsonl", type=pathlib.Path, default=s64.DEFAULT_TAIL)
    ap.add_argument("--student-path", type=pathlib.Path, default=DEFAULT_STUDENT)
    ap.add_argument("--student-name", default="densemask_sparselabel_seed62064_u0080")
    ap.add_argument("--out-dir", type=pathlib.Path, default=DEFAULT_OUT)
    ap.add_argument("--macro-indices", default="0,20,40,60")
    ap.add_argument("--max-updates", type=int, default=80)
    ap.add_argument("--words-per-update", type=int, default=39533)
    ap.add_argument("--seq-length", type=int, default=512)
    ap.add_argument("--train-seed", type=int, default=62064)
    ap.add_argument("--mask-prob", type=float, default=0.15)
    ap.add_argument("--focus-prob", type=float, default=0.35)
    ap.add_argument("--max-focus-groups-per-row", type=int, default=16)
    ap.add_argument("--max-dense-mask-groups", type=int, default=128)
    ap.add_argument("--focus-lambda", type=float, default=0.15)
    ap.add_argument("--private-scale", type=float, default=0.75)
    ap.add_argument("--micro-batch", type=int, default=8)
    ap.add_argument("--kl-temperature", type=float, default=1.0)
    ap.add_argument("--device", choices=["cuda", "cpu", "auto"], default="cuda")
    ap.add_argument("--gpu", type=int, default=0)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    setup_cache(args.out_dir)
    torch.set_num_threads(max(1, min(8, int(os.environ.get("QIUSHI_TORCH_THREADS", "8")))))
    if args.device == "cuda" and torch.cuda.is_available():
        device = torch.device(f"cuda:{int(args.gpu)}")
    elif args.device == "auto" and torch.cuda.is_available():
        device = torch.device(f"cuda:{int(args.gpu)}")
    else:
        device = torch.device("cpu")
    args.device_obj = device

    install_densemask_patch(int(args.max_dense_mask_groups))
    rows, prefix_info = s64.load_prefix(pathlib.Path(args.tail_jsonl), int(args.max_updates), int(args.words_per_update))
    macros_all = split_macros(rows, int(args.max_updates), int(args.words_per_update))
    wanted = parse_macro_indices(args.macro_indices)
    selected = [macros_all[i] for i in wanted if 0 <= i < len(macros_all)]
    if not selected:
        raise RuntimeError(f"No selected macros from {wanted}")

    tokenizer = bridge.AutoTokenizer.from_pretrained(str(bridge.PARENT_PATH), local_files_only=True, use_fast=True)
    wgb = bridge.WordGroupBuilder(tokenizer)

    per_macro_renderings: Dict[str, List[List[Dict[str, Any]]]] = {k: [] for k in ["ordinary_common", "dense_common", "ordinary_full", "dense_nonlabel_full"]}
    support_counts = Counter()
    macro_support_records = []
    for macro in selected:
        rends, st = build_preservation_renderings(macro["rows"], tokenizer, wgb, args)
        for k, ex in rends.items():
            per_macro_renderings[k].append(ex)
        counts = st["counts"]
        support_counts.update(counts)
        support_counts["selected_rows"] += len(macro["rows"])
        support_counts["selected_words"] += int(macro["words"])
        macro_support_records.append({
            "update_index0": int(macro["update_index0"]),
            "rows": len(macro["rows"]),
            "words": int(macro["words"]),
            "qwen_rows": int(macro["qwen_rows"]),
            **counts,
        })

    teacher, teacher_load = load_endpoint(bridge.PARENT_PATH, device, float(args.private_scale))
    student, student_load = load_endpoint(pathlib.Path(args.student_path), device, float(args.private_scale))
    teacher.eval()
    student.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    private_params, private_info = set_private_trainable(student)

    t0 = time.time()
    g_full, acq_full = grad_acquisition(student, private_params, selected, tokenizer, wgb, args, "full_weighted")
    print(json.dumps({"event": "grad_done", "name": "acquisition_full_weighted", "l2": acq_full["grad"]["l2"], "elapsed_sec": round(time.time()-t0, 1)}), flush=True)
    g_focus, acq_focus = grad_acquisition(student, private_params, selected, tokenizer, wgb, args, "focus_mean")
    acq_focus["actual_training_scaled_l2_if_lambda_0p15"] = float(float(args.focus_lambda) * acq_focus["grad"]["l2"])
    print(json.dumps({"event": "grad_done", "name": "acquisition_focus_mean", "l2": acq_focus["grad"]["l2"], "elapsed_sec": round(time.time()-t0, 1)}), flush=True)
    g_ord, acq_ord = grad_acquisition(student, private_params, selected, tokenizer, wgb, args, "ordinary_mean")
    acq_ord["actual_training_scaled_l2_if_lambda_0p85"] = float((1.0 - float(args.focus_lambda)) * acq_ord["grad"]["l2"])
    print(json.dumps({"event": "grad_done", "name": "acquisition_ordinary_mean", "l2": acq_ord["grad"]["l2"], "elapsed_sec": round(time.time()-t0, 1)}), flush=True)

    refs = {"acquisition_full_weighted": g_full, "acquisition_focus_mean": g_focus, "acquisition_ordinary_mean": g_ord}
    branches: Dict[str, Any] = {}
    grad_vectors: Dict[str, torch.Tensor] = {}
    for name in ["ordinary_common", "dense_common", "ordinary_full", "dense_nonlabel_full"]:
        gv, rec = grad_kl(student, teacher, private_params, per_macro_renderings[name], args)
        rec.update(alignment_block(name, gv, refs))
        branches[name] = rec
        grad_vectors[name] = gv
        print(json.dumps({"event": "grad_done", "name": name, "tokens": rec["tokens_total"], "kl": rec["kl_teacher_to_student_token_mean"], "l2": rec["grad"]["l2"], "elapsed_sec": round(time.time()-t0, 1)}), flush=True)

    # Add preservation-vs-preservation cosines after all branches exist.
    for name, gv in grad_vectors.items():
        for other, ov in grad_vectors.items():
            if other == name:
                continue
            branches[name].setdefault("alignment_to_preservation_branches", {})[other] = {"cosine": cosine(gv, ov), "dot": dot(gv, ov)}

    oc = branches["ordinary_common"]
    dc = branches["dense_common"]
    common_contrast = {
        "dense_minus_ordinary_teacher_entropy": dc["teacher_entropy_token_mean"] - oc["teacher_entropy_token_mean"],
        "dense_minus_ordinary_teacher_top1_prob": dc["teacher_top1_prob_token_mean"] - oc["teacher_top1_prob_token_mean"],
        "dense_minus_ordinary_teacher_target_ce": dc["teacher_target_ce_token_mean"] - oc["teacher_target_ce_token_mean"],
        "dense_minus_ordinary_student_target_ce": dc["student_target_ce_token_mean"] - oc["student_target_ce_token_mean"],
        "dense_minus_ordinary_kl_teacher_to_student": dc["kl_teacher_to_student_token_mean"] - oc["kl_teacher_to_student_token_mean"],
        "dense_over_ordinary_grad_l2": (dc["grad"]["l2"] / oc["grad"]["l2"]) if oc["grad"]["l2"] else None,
        "cos_ordinary_dense_common_grad": cosine(grad_vectors["ordinary_common"], grad_vectors["dense_common"]),
        "cos_ordinary_common_vs_acq_full": branches["ordinary_common"]["alignment"]["acquisition_full_weighted"]["cosine"],
        "cos_dense_common_vs_acq_full": branches["dense_common"]["alignment"]["acquisition_full_weighted"]["cosine"],
        "cos_ordinary_common_vs_acq_focus": branches["ordinary_common"]["alignment"]["acquisition_focus_mean"]["cosine"],
        "cos_dense_common_vs_acq_focus": branches["dense_common"]["alignment"]["acquisition_focus_mean"]["cosine"],
        "cos_ordinary_common_vs_acq_ordinary": branches["ordinary_common"]["alignment"]["acquisition_ordinary_mean"]["cosine"],
        "cos_dense_common_vs_acq_ordinary": branches["dense_common"]["alignment"]["acquisition_ordinary_mean"]["cosine"],
    }

    interpretation_parts = []
    if support_counts.get("common_targets", 0) <= 0:
        interpretation_parts.append("No common targets were found; this diagnostic cannot compare rendering geometry at fixed support for the selected macros.")
    else:
        interpretation_parts.append(
            "The common-support branch separates input rendering from target support: the same token positions are preserved under ordinary-WWM context and dense-corrupted context. Any difference in teacher entropy, KL, or gradient alignment here is a prediction-state geometry effect rather than a support-count effect for these positions."
        )
        if common_contrast["dense_over_ordinary_grad_l2"] is not None:
            interpretation_parts.append(
                f"On the selected macros, dense-common KL has {common_contrast['dense_over_ordinary_grad_l2']:.3g}× the ordinary-common private-gradient L2. Its cosine with the full acquisition gradient is {common_contrast['cos_dense_common_vs_acq_full']}, versus {common_contrast['cos_ordinary_common_vs_acq_full']} for ordinary-common."
            )
        interpretation_parts.append(
            "The full dense-nonlabel branch remains a joint rendering-and-support intervention because its target set differs from clean ordinary WWM. It should be interpreted through both the full-support measurements and the common-support diagnostic, not as a pure geometry isolation."
        )
    interpretation_parts.append(
        "Because gradients are evaluated at the acquisition-shifted exact `(M,S)` endpoint, nonzero KL reflects actual drift from the coherent86 parent. The same test at initialization would be uninformative: teacher and student predictions coincide and the deterministic KL gradient is zero."
    )

    support_summary = dict(support_counts)
    for k in ["ordinary_full_targets", "dense_nonlabel_targets", "common_targets"]:
        support_summary[k] = int(support_summary.get(k, 0))
    support_summary["common_over_ordinary_full_targets"] = float(support_summary.get("common_targets", 0) / max(1, support_summary.get("ordinary_full_targets", 0)))
    support_summary["common_over_dense_nonlabel_targets"] = float(support_summary.get("common_targets", 0) / max(1, support_summary.get("dense_nonlabel_targets", 0)))

    result = {
        "status": "PRESERVATION_GEOMETRY_MATCHED_DIAGNOSTIC_DONE",
        "created_utc": now(),
        "script": rel(_public_path('experiments/archive/functional_learning/scripts/preservation_geometry_matched_diagnostic.py')),
        "scientific_question": "Do ordinary-WWM and dense-corrupted preservation prediction states impose different parent constraints at common target positions after acquisition?",
        "scope": "local eval-mode gradient/uncertainty diagnostic at exact acquisition-only (M,S); not a training result or leaderboard score",
        "student": {"name": args.student_name, **student_load},
        "teacher": {"name": "coherent86_private_scale_0p75", **teacher_load},
        "private_trainable_info": private_info,
        "prefix_info": prefix_info,
        "selected_macro_indices0": [int(m["update_index0"]) for m in selected],
        "macro_support_records": macro_support_records,
        "support_summary": support_summary,
        "settings": {
            "seq_length": int(args.seq_length),
            "train_seed": int(args.train_seed),
            "mask_prob": float(args.mask_prob),
            "focus_prob": float(args.focus_prob),
            "max_focus_groups_per_row": int(args.max_focus_groups_per_row),
            "max_dense_mask_groups": int(args.max_dense_mask_groups),
            "focus_lambda": float(args.focus_lambda),
            "kl_temperature": float(args.kl_temperature),
            "micro_batch": int(args.micro_batch),
            "device": str(device),
            "gradient_mode": "student eval mode for acquisition and preservation; teacher eval mode; private adapters only",
        },
        "acquisition": {
            "acquisition_full_weighted": acq_full,
            "acquisition_focus_mean": acq_focus,
            "acquisition_ordinary_mean": acq_ord,
            "acquisition_focus_mean_alignment_to_full": {"cosine": cosine(g_focus, g_full), "dot": dot(g_focus, g_full)},
            "acquisition_ordinary_mean_alignment_to_full": {"cosine": cosine(g_ord, g_full), "dot": dot(g_ord, g_full)},
            "acquisition_focus_vs_ordinary_alignment": {"cosine": cosine(g_focus, g_ord), "dot": dot(g_focus, g_ord)},
        },
        "branches": branches,
        "common_rendering_contrast": common_contrast,
        "interpretation": " ".join(interpretation_parts),
        "elapsed_sec": round(time.time() - t0, 1),
    }
    write_outputs(pathlib.Path(args.out_dir), result)


if __name__ == "__main__":
    main()
