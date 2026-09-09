#!/usr/bin/env python3
"""research rollback control for clean eval-mode preservation.

Scientific purpose
------------------
Clean eval-mode parent-function preservation could improve the dense-mask/sparse-
label trade-off in two very different ways:
  (1) a selective constraint changes the update direction, preserving ordinary and
      evidence-absent competence while allowing useful source-conditioned behavior;
  (2) it merely shrinks the original acquisition-only dense-mask update.

This script tests (2) directly.  It constructs an in-memory rollback model along
only the private-adapter update from coherent86 to acquisition-only (M,S):
    theta(alpha) = theta_parent + alpha * (theta_densemask - theta_parent)
with alpha chosen to match the clean eval endpoint's KL(parent || model) on the
same ordinary non-Qwen legal-tail masked-position substrate used by the research
temperature readout.  It then compares the matched rollback against clean eval on
fixed source-response and evidence-availability readouts.

It writes research evidence only; it does not create a candidate checkpoint.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse
import json
import math
import pathlib
import statistics
import sys
import time
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn.functional as F
from safetensors.torch import load_file

ROOT = _public_path('.')
STUDY = _public_path('experiments/archive/functional_learning')
SCRIPTS = _public_path('experiments/archive/functional_learning/scripts')
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import corrected_bridge_trainer as bridge  # noqa: E402
import qwen_view_surface_probe as qview  # noqa: E402
import temperature_source_readout as s86  # noqa: E402
import evidence_availability_readout as s90  # noqa: E402

DEFAULT_OUT = _public_path('experiments/archive/functional_learning/data/rollback_control')
PARENT = bridge.PARENT_PATH
DENSEMASK = _public_path('experiments/archive/functional_learning/data/densemask_sparselabel_train_seed62064/correspondence_focus_weighted/checkpoints/update_0080')
CLEAN_EVAL = _public_path('experiments/archive/functional_learning/data/clean_preservation_lambda1_eval_full80/checkpoints/update_0080')
SPARSE = _public_path('experiments/archive/functional_learning/data/unchanged_focus_weighted_train/correspondence_focus_weighted/checkpoints/update_0080')
EVIDENCE = _public_path('experiments/archive/functional_learning/data/evidence_availability_readout_full')
TEMP = _public_path('experiments/archive/functional_learning/data/temperature_source_readout_clean')


def rel(p: pathlib.Path | str | None) -> str | None:
    if p is None:
        return None
    try:
        return str(pathlib.Path(p).resolve().relative_to(ROOT))
    except Exception:
        return str(p)


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def finite_vals(xs: Iterable[Any]) -> List[float]:
    vals: List[float] = []
    for x in xs:
        if x is None:
            continue
        try:
            xf = float(x)
        except Exception:
            continue
        if math.isfinite(xf):
            vals.append(xf)
    return vals


def mean(xs: Iterable[Any]) -> Optional[float]:
    vals = finite_vals(xs)
    return sum(vals) / len(vals) if vals else None


def median(xs: Iterable[Any]) -> Optional[float]:
    vals = finite_vals(xs)
    return statistics.median(vals) if vals else None


def sem(xs: Iterable[Any]) -> Optional[float]:
    vals = finite_vals(xs)
    if len(vals) <= 1:
        return None
    return statistics.stdev(vals) / math.sqrt(len(vals))


def read_json(path: pathlib.Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_jsonl(path: pathlib.Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def load_jsonl(path: pathlib.Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def parse_alpha_grid(text: str) -> List[float]:
    vals: List[float] = []
    for piece in str(text).replace(",", " ").split():
        vals.append(float(piece))
    if not vals:
        raise ValueError("empty alpha grid")
    vals = sorted(set(round(float(x), 10) for x in vals))
    for x in vals:
        if x < -1e-9 or x > 1.000000001:
            raise ValueError(f"alpha outside [0,1]: {x}")
    return vals


def default_alpha_grid() -> List[float]:
    grid = [i / 20.0 for i in range(21)]
    grid += [0.425, 0.45, 0.475, 0.525, 0.55, 0.575, 0.6, 0.625]
    return sorted(set(grid))


def load_model(path: Optional[pathlib.Path], device: torch.device, private_scale: float):
    model, info = qview.load_scoring_model(path, device, private_scale)
    model.eval()
    return model, info


def private_keys_from_state(path: pathlib.Path) -> List[str]:
    sd = load_file(str(path / "model.safetensors"))
    return sorted(k for k in sd.keys() if ".private_adapter." in k)


def validate_private_update(parent_path: pathlib.Path, dense_path: pathlib.Path) -> Dict[str, Any]:
    psd = load_file(str(parent_path / "model.safetensors"))
    dsd = load_file(str(dense_path / "model.safetensors"))
    missing = [k for k in psd if k not in dsd]
    extra = [k for k in dsd if k not in psd]
    changed_private = []
    changed_nonprivate = []
    private_norm_sq = 0.0
    for k, p in psd.items():
        if k not in dsd:
            continue
        diff = (dsd[k].float() - p.float())
        if ".private_adapter." in k:
            if torch.any(diff != 0):
                changed_private.append(k)
            private_norm_sq += float((diff * diff).sum().item())
        else:
            if torch.any(diff != 0):
                changed_nonprivate.append(k)
    return {
        "parent": rel(parent_path),
        "densemask": rel(dense_path),
        "missing_in_densemask": len(missing),
        "extra_in_densemask": len(extra),
        "private_tensors_parent": sum(1 for k in psd if ".private_adapter." in k),
        "private_tensors_changed": len(changed_private),
        "nonprivate_tensors_changed": len(changed_nonprivate),
        "private_update_l2": math.sqrt(private_norm_sq),
        "changed_private_head": changed_private[:8],
        "changed_nonprivate_head": changed_nonprivate[:8],
        "valid_private_only_update": len(missing) == 0 and len(extra) == 0 and len(changed_nonprivate) == 0 and len(changed_private) > 0,
    }


def apply_private_interpolation(model, parent_sd: Dict[str, torch.Tensor], dense_sd: Dict[str, torch.Tensor], alpha: float) -> None:
    named_params = dict(model.named_parameters())
    private_keys = [k for k in parent_sd if ".private_adapter." in k]
    missing = [k for k in private_keys if k not in named_params or k not in dense_sd]
    if missing:
        raise RuntimeError(f"missing private params for interpolation: {missing[:5]}")
    with torch.no_grad():
        for k in private_keys:
            target = parent_sd[k].to(device=named_params[k].device, dtype=named_params[k].dtype)
            delta = dense_sd[k].to(device=named_params[k].device, dtype=named_params[k].dtype) - target
            named_params[k].copy_(target + float(alpha) * delta)


def position_kl_teacher_to_model(parent_logits: List[torch.Tensor], model_logits: List[torch.Tensor], temperature: float = 1.0) -> List[float]:
    vals: List[float] = []
    T = float(temperature)
    for pl, ml in zip(parent_logits, model_logits, strict=False):
        lp = F.log_softmax(pl.float() / T, dim=-1)
        lm = F.log_softmax(ml.float() / T, dim=-1)
        p = torch.exp(lp)
        kl = (p * (lp - lm)).sum()
        if T != 1.0:
            kl = kl * (T ** 2)
        vals.append(float(kl.item()))
    return vals


def summarize_calibration_logits(logits: List[torch.Tensor], labels: List[int], temperature_grid: List[float]) -> Dict[str, Any]:
    fit = s86.fit_temperature_from_position_logits(logits, labels, temperature_grid)
    nll_t1 = [s86.nll_from_logits(row, tid, 1.0) for row, tid in zip(logits, labels, strict=False)]
    ranks = [s86.rank_from_logits(row, tid) for row, tid in zip(logits, labels, strict=False)]
    return {
        "temperature_fit": fit,
        "mean_nll_T1": mean(nll_t1),
        "median_nll_T1": median(nll_t1),
        "mean_rank_T1": mean(ranks),
        "median_rank_T1": median(ranks),
        "n_positions": len(logits),
    }


def build_calibration(tokenizer, args: argparse.Namespace) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    return s86.build_calibration_records(
        pathlib.Path(args.tail_jsonl), tokenizer,
        n_records=int(args.calib_records), seed=int(args.calib_seed), max_length=int(args.max_length), rows_after_prefix=True,
    )


def build_evidence_tasks(tokenizer, args: argparse.Namespace) -> Tuple[List[Dict[str, Any]], Dict[str, Any], Dict[str, Any]]:
    ns = argparse.Namespace(
        tail_jsonl=pathlib.Path(args.tail_jsonl),
        scope=str(args.evidence_scope),
        max_updates=int(args.max_updates),
        words_per_update=int(args.words_per_update),
        max_segments=int(args.evidence_segments),
        sample_seed=int(args.evidence_seed),
        max_targets_per_segment=int(args.evidence_targets_per_segment),
        max_length=int(args.max_length),
        conditions=list(s90.DEFAULT_CONDITIONS),
    )
    records, seg_stats = s90.build_segment_records(ns)
    tasks, task_stats = s90.build_tasks(records, tokenizer, ns)
    return tasks, seg_stats, task_stats


def build_source_tasks(tokenizer, args: argparse.Namespace) -> Tuple[List[Dict[str, Any]], Dict[str, Any], List[Dict[str, Any]], Dict[str, Any]]:
    source_records, source_segment_stats = s86.source_probe.build_segment_records(
        argparse.Namespace(
            tail_jsonl=pathlib.Path(args.tail_jsonl),
            max_updates=int(args.max_updates),
            words_per_update=int(args.words_per_update),
            max_segments=int(args.source_segments),
            sample_seed=int(args.source_seed),
        )
    )
    qwen_tasks, qwen_task_stats = s86.source_probe.build_tasks(
        source_records, tokenizer,
        argparse.Namespace(sample_seed=int(args.source_seed), max_targets_per_segment=int(args.source_targets_per_segment), max_length=int(args.max_length)),
    )
    common_bank, common_bank_summary = s86.common_probe.materialize_bank(pathlib.Path(args.labels))
    common_records, common_rec_summary = s86.common_probe.build_scoring_records(tokenizer, common_bank, int(args.max_length))
    common_summary = {"bank": common_bank_summary, "records": common_rec_summary}
    return qwen_tasks, {"segment_stats": source_segment_stats, "task_stats": qwen_task_stats}, common_records, common_summary


def compact_evidence_summary(summary: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in summary.items() if k != "interaction_rows_head"}


def selected_evidence_deltas(summary: Dict[str, Any]) -> Dict[str, Any]:
    by_cond = summary.get("by_condition", {})
    interactions = summary.get("interactions", {})
    return {
        "condition_delta_nll_vs_parent": {c: by_cond.get(c, {}).get("mean_delta_nll_vs_coherent86") for c in by_cond},
        "condition_delta_rank_vs_parent": {c: by_cond.get(c, {}).get("mean_delta_rank_vs_coherent86") for c in by_cond},
        "condition_kl_parent_to_model": {c: by_cond.get(c, {}).get("mean_kl_coherent86_to_model") for c in by_cond},
        "specific_source_penalty_delta_nll_vs_parent": interactions.get("delta_vs_coherent86_specific_source_penalty_nll", {}).get("mean"),
        "view_absence_penalty_delta_nll_vs_parent": interactions.get("delta_vs_coherent86_view_absence_penalty_nll", {}).get("mean"),
        "this_source_erasure_delta_nll_vs_parent": interactions.get("delta_vs_coherent86_this_source_erasure_cost_nll", {}).get("mean"),
        "all_sources_erasure_delta_nll_vs_parent": interactions.get("delta_vs_coherent86_all_sources_erasure_cost_nll", {}).get("mean"),
    }


def summarize_between_models(rows_a: List[Dict[str, Any]], rows_b: List[Dict[str, Any]], key_fields: Tuple[str, ...], metrics: Tuple[str, ...]) -> Dict[str, Any]:
    bmap = {tuple(str(r.get(k)) for k in key_fields): r for r in rows_b}
    out: Dict[str, Any] = {"n_a": len(rows_a), "n_matched": 0, "metrics": {}}
    diffs: Dict[str, List[float]] = {m: [] for m in metrics}
    for r in rows_a:
        key = tuple(str(r.get(k)) for k in key_fields)
        b = bmap.get(key)
        if not b:
            continue
        out["n_matched"] += 1
        for m in metrics:
            if r.get(m) is not None and b.get(m) is not None:
                diffs[m].append(float(r[m]) - float(b[m]))
    for m, vals in diffs.items():
        out["metrics"][m] = {"mean_a_minus_b": mean(vals), "median_a_minus_b": median(vals), "sem": sem(vals), "n": len(vals)}
    return out


def write_markdown(path: pathlib.Path, result: Dict[str, Any]) -> None:
    lines: List[str] = []
    lines.append("# research rollback control\n\n")
    lines.append("This readout compares clean eval-mode preservation with a matched rollback along the acquisition-only `(M,S)` private-adapter update. It is an explanatory computation, not a candidate checkpoint.\n\n")
    lines.append("## Matched ordinary-text drift\n\n")
    m = result.get("match", {})
    lines.append(f"- Clean eval KL(parent||model) on ordinary non-Qwen calibration positions: `{m.get('clean_eval_kl_mean')}` over `{m.get('n_positions')}` positions.\n")
    lines.append(f"- Dense-mask alpha=1 KL on the same positions: `{m.get('densemask_kl_mean')}`.\n")
    lines.append(f"- Selected rollback alpha: `{m.get('selected_alpha')}` with KL `{m.get('selected_alpha_kl_mean')}` and absolute error `{m.get('selected_alpha_abs_error')}`.\n")
    lines.append("\n## Evidence availability: selected deltas vs coherent86\n\n")
    for name, summ in result.get("evidence_selected", {}).items():
        lines.append(f"### `{name}`\n")
        d = summ.get("condition_delta_nll_vs_parent", {})
        for c in result.get("evidence_conditions", []):
            lines.append(f"- {c}: ΔNLL `{d.get(c)}`, Δrank `{summ.get('condition_delta_rank_vs_parent', {}).get(c)}`, KL `{summ.get('condition_kl_parent_to_model', {}).get(c)}`\n")
        lines.append(f"- source penalty ΔNLL: `{summ.get('specific_source_penalty_delta_nll_vs_parent')}`; view absence ΔNLL: `{summ.get('view_absence_penalty_delta_nll_vs_parent')}`; this-source erasure ΔNLL: `{summ.get('this_source_erasure_delta_nll_vs_parent')}`; all-source erasure ΔNLL: `{summ.get('all_sources_erasure_delta_nll_vs_parent')}`\n\n")
    lines.append("## Source response\n\n")
    for name, summ in result.get("source_selected", {}).items():
        q = summ.get("qwen_source", {}).get("source_specificity", {})
        c = summ.get("common_source_reversal", {}).get("source_follow", {})
        lines.append(f"- `{name}`: Qwen Δspecific Tfit `{q.get('mean_delta_specific_advantage_Tfit_vs_parent')}`, Δrank `{q.get('mean_delta_specific_rank_advantage_vs_parent')}`; common both `{c.get('both_correct_Tfit')}/25`, Δswing Tfit `{c.get('mean_delta_source_follow_swing_Tfit_vs_parent')}`, Δrank `{c.get('mean_delta_source_follow_rank_swing_vs_parent')}`.\n")
    lines.append("\n## Interpretation\n\n")
    interp = result.get("interpretation", {})
    for k, v in interp.items():
        lines.append(f"- {k}: {v}\n")
    path.write_text("".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", type=pathlib.Path, default=DEFAULT_OUT)
    ap.add_argument("--device", choices=["cpu", "cuda", "auto"], default="auto")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--private-scale", type=float, default=0.75)
    ap.add_argument("--tail-jsonl", type=pathlib.Path, default=s86.DEFAULT_TAIL)
    ap.add_argument("--labels", type=pathlib.Path, default=s86.DEFAULT_LABELS)
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--calib-records", type=int, default=192)
    ap.add_argument("--calib-seed", type=int, default=86031)
    ap.add_argument("--temperature-grid", default="0.7 0.8 0.9 1.0 1.1 1.25 1.4 1.6 1.9")
    ap.add_argument("--alpha-grid", default="", help="Space/comma-separated alpha grid; default uses 0..1 step .05 plus midpoints near the expected match.")
    ap.add_argument("--evidence-scope", default="post_prefix", choices=["prefix", "post_prefix", "all"])
    ap.add_argument("--evidence-segments", type=int, default=96)
    ap.add_argument("--evidence-targets-per-segment", type=int, default=2)
    ap.add_argument("--evidence-seed", type=int, default=90090)
    ap.add_argument("--source-segments", type=int, default=120)
    ap.add_argument("--source-targets-per-segment", type=int, default=4)
    ap.add_argument("--source-seed", type=int, default=86067)
    ap.add_argument("--max-updates", type=int, default=s90.PREFIX_UPDATES)
    ap.add_argument("--words-per-update", type=int, default=s90.WORDS_PER_UPDATE)
    ap.add_argument("--skip-source", action="store_true")
    ap.add_argument("--skip-evidence", action="store_true")
    args = ap.parse_args()

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.device == "auto":
        device = torch.device(f"cuda:{int(args.gpu)}" if torch.cuda.is_available() else "cpu")
    elif args.device == "cuda":
        device = torch.device(f"cuda:{int(args.gpu)}")
    else:
        device = torch.device("cpu")
    alpha_grid = parse_alpha_grid(args.alpha_grid) if str(args.alpha_grid).strip() else default_alpha_grid()
    # Temperature grid can exceed 1, unlike the rollback alpha grid.
    temp_grid = [float(x) for x in str(args.temperature_grid).replace(",", " ").split()]

    update_validation = validate_private_update(PARENT, DENSEMASK)
    if not update_validation.get("valid_private_only_update"):
        raise RuntimeError(f"rollback path is not a private-only update: {update_validation}")

    tokenizer = bridge.AutoTokenizer.from_pretrained(str(PARENT), local_files_only=True, use_fast=True)
    calibration, calib_summary = build_calibration(tokenizer, args)
    write_jsonl(out_dir / "calibration_records.jsonl", (s86.strip_task_for_json(t) for t in calibration))

    print(json.dumps({"event": "calibration_built", "records": len(calibration), "positions_expected": "token-dependent", "device": str(device)}), flush=True)
    parent_model, parent_info = load_model(None, device, float(args.private_scale))
    parent_logits, labels, _metas = s86.gather_position_logits(parent_model, tokenizer, calibration, device, int(args.batch_size))
    parent_calib = summarize_calibration_logits(parent_logits, labels, temp_grid)
    del parent_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    clean_model, clean_info = load_model(CLEAN_EVAL, device, float(args.private_scale))
    clean_logits, clean_labels, _ = s86.gather_position_logits(clean_model, tokenizer, calibration, device, int(args.batch_size))
    if clean_labels != labels:
        raise RuntimeError("clean labels differ from parent labels on calibration")
    clean_kls = position_kl_teacher_to_model(parent_logits, clean_logits, 1.0)
    clean_calib = summarize_calibration_logits(clean_logits, labels, temp_grid)
    del clean_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    dense_model, dense_info = load_model(DENSEMASK, device, float(args.private_scale))
    dense_logits, dense_labels, _ = s86.gather_position_logits(dense_model, tokenizer, calibration, device, int(args.batch_size))
    if dense_labels != labels:
        raise RuntimeError("densemask labels differ from parent labels on calibration")
    dense_kls = position_kl_teacher_to_model(parent_logits, dense_logits, 1.0)
    dense_calib = summarize_calibration_logits(dense_logits, labels, temp_grid)
    del dense_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    parent_sd = load_file(str(PARENT / "model.safetensors"))
    dense_sd = load_file(str(_public_path('experiments/archive/functional_learning/data/densemask_sparselabel_train_seed62064/correspondence_focus_weighted/checkpoints/update_0080/model.safetensors')))
    rollback_model, rollback_base_info = load_model(None, device, float(args.private_scale))
    alpha_rows: List[Dict[str, Any]] = []
    selected: Optional[Dict[str, Any]] = None
    clean_kl_mean = float(mean(clean_kls) or 0.0)
    for alpha in alpha_grid:
        apply_private_interpolation(rollback_model, parent_sd, dense_sd, float(alpha))
        logits, lbls, _ = s86.gather_position_logits(rollback_model, tokenizer, calibration, device, int(args.batch_size))
        if lbls != labels:
            raise RuntimeError("rollback labels differ from parent labels on calibration")
        kls = position_kl_teacher_to_model(parent_logits, logits, 1.0)
        calib = summarize_calibration_logits(logits, labels, temp_grid)
        row = {
            "alpha": float(alpha),
            "kl_mean": mean(kls),
            "kl_median": median(kls),
            "kl_sem": sem(kls),
            "abs_error_to_clean_kl_mean": abs(float(mean(kls) or 0.0) - clean_kl_mean),
            "calibration": calib,
        }
        alpha_rows.append(row)
        if selected is None or float(row["abs_error_to_clean_kl_mean"]) < float(selected["abs_error_to_clean_kl_mean"]):
            selected = row
        print(json.dumps({"event": "alpha_scored", "alpha": alpha, "kl_mean": row["kl_mean"], "target_clean_kl": clean_kl_mean, "abs_error": row["abs_error_to_clean_kl_mean"]}), flush=True)
    if selected is None:
        raise RuntimeError("no alpha selected")
    selected_alpha = float(selected["alpha"])
    apply_private_interpolation(rollback_model, parent_sd, dense_sd, selected_alpha)
    # Fit the rollback model's temperature on the same ordinary-text calibration positions.
    rollback_logits, rollback_labels, _ = s86.gather_position_logits(rollback_model, tokenizer, calibration, device, int(args.batch_size))
    if rollback_labels != labels:
        raise RuntimeError("selected rollback labels differ from parent labels on calibration")
    rollback_calib = summarize_calibration_logits(rollback_logits, labels, temp_grid)
    rollback_temperature = float(rollback_calib["temperature_fit"].get("best_temperature", 1.0))

    evidence_result: Dict[str, Any] = {}
    evidence_selected: Dict[str, Any] = {}
    evidence_conditions = list(s90.DEFAULT_CONDITIONS)
    if not args.skip_evidence:
        ev_tasks, ev_seg_stats, ev_task_stats = build_evidence_tasks(tokenizer, args)
        write_jsonl(out_dir / "evidence_tasks.jsonl", (s90.task_for_json(t) for t in ev_tasks))
        ev_models = {
            "coherent86": load_model(None, device, float(args.private_scale))[0],
            "sparse_focus_seed62064": load_model(SPARSE, device, float(args.private_scale))[0],
            "rollback_matched_densemask_update": rollback_model,
        }
        ev_rows = s90.score_all_models(ev_models, ev_tasks, device, tokenizer, int(args.batch_size), kl_temperature=1.0)
        # Load clean/densemask existing exact research rows for direct matched-task comparison.
        clean_ev_rows = load_jsonl(_public_path('experiments/archive/functional_learning/data/evidence_availability_readout_full/scores_clean_pres_lambda1_eval_full80.jsonl')) if (_public_path('experiments/archive/functional_learning/data/evidence_availability_readout_full/scores_clean_pres_lambda1_eval_full80.jsonl')).exists() else []
        dense_ev_rows_existing = load_jsonl(_public_path('experiments/archive/functional_learning/data/evidence_availability_readout_full/scores_densemask_sparselabel_seed62064.jsonl')) if (_public_path('experiments/archive/functional_learning/data/evidence_availability_readout_full/scores_densemask_sparselabel_seed62064.jsonl')).exists() else []
        for name, rows in ev_rows.items():
            write_jsonl(out_dir / f"evidence_scores_{name}.jsonl", rows)
        if clean_ev_rows:
            write_jsonl(out_dir / "evidence_scores_clean_eval_from_step090.jsonl", clean_ev_rows)
        parent_rows = ev_rows["coherent86"]
        sparse_rows = ev_rows["sparse_focus_seed62064"]
        summaries = {name: s90.summarize_model(rows, parent_rows, sparse_rows, evidence_conditions) for name, rows in ev_rows.items()}
        if clean_ev_rows:
            summaries["clean_pres_lambda1_eval_full80_step090"] = s90.summarize_model(clean_ev_rows, parent_rows, sparse_rows, evidence_conditions)
        if dense_ev_rows_existing:
            summaries["densemask_sparselabel_seed62064_step090"] = s90.summarize_model(dense_ev_rows_existing, parent_rows, sparse_rows, evidence_conditions)
        evidence_selected = {name: selected_evidence_deltas(summ) for name, summ in summaries.items()}
        evidence_result = {
            "segment_stats": ev_seg_stats,
            "task_stats": ev_task_stats,
            "score_paths": {name: rel(out_dir / f"evidence_scores_{name}.jsonl") for name in ev_rows},
            "summaries": {name: compact_evidence_summary(summ) for name, summ in summaries.items()},
            "rollback_minus_clean_on_rows": summarize_between_models(
                ev_rows["rollback_matched_densemask_update"], clean_ev_rows,
                key_fields=("task_id",), metrics=("nll", "rank", "kl_coherent86_to_model")
            ) if clean_ev_rows else {},
            "rollback_minus_densemask_on_rows": summarize_between_models(
                ev_rows["rollback_matched_densemask_update"], dense_ev_rows_existing,
                key_fields=("task_id",), metrics=("nll", "rank", "kl_coherent86_to_model")
            ) if dense_ev_rows_existing else {},
        }
        # Keep only rollback model resident after evidence scoring.
        for name, model in ev_models.items():
            if name != "rollback_matched_densemask_update":
                del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    source_result: Dict[str, Any] = {}
    source_selected: Dict[str, Any] = {}
    if not args.skip_source:
        qwen_tasks, qwen_stats, common_tasks, common_stats = build_source_tasks(tokenizer, args)
        write_jsonl(out_dir / "qwen_source_tasks.jsonl", (s86.strip_task_for_json(t) for t in qwen_tasks))
        write_jsonl(out_dir / "common_source_reversal_records.jsonl", (s86.strip_task_for_json(t) for t in common_tasks))
        qwen_scores = s86.score_tasks_with_temperature(rollback_model, tokenizer, qwen_tasks, device, int(args.batch_size), rollback_temperature)
        common_scores = s86.score_tasks_with_temperature(rollback_model, tokenizer, common_tasks, device, int(args.batch_size), rollback_temperature)
        write_jsonl(out_dir / "qwen_scores_rollback_matched_densemask_update.jsonl", qwen_scores)
        write_jsonl(out_dir / "common_scores_rollback_matched_densemask_update.jsonl", common_scores)
        q_parent = load_jsonl(_public_path('experiments/archive/functional_learning/data/temperature_source_readout_clean/qwen_scores_coherent86.jsonl'))
        c_parent = load_jsonl(_public_path('experiments/archive/functional_learning/data/temperature_source_readout_clean/common_scores_coherent86.jsonl'))
        rollback_summary = {
            "calibration": rollback_calib,
            "qwen_source": s86.summarize_qwen_source(qwen_scores, q_parent),
            "common_source_reversal": s86.summarize_common(common_scores, c_parent),
        }
        source_selected["rollback_matched_densemask_update"] = rollback_summary
        # Add compact existing summaries for direct contextual interpretation.
        for name in ["densemask_sparselabel_seed62064", "clean_pres_lambda1_eval_full80"]:
            p = TEMP / f"summary_{name}.json"
            if p.exists():
                source_selected[f"{name}_step090"] = read_json(p)
        source_result = {
            "qwen_task_stats": qwen_stats,
            "common_source_reversal_stats": common_stats,
            "rollback_temperature": rollback_temperature,
            "summary": rollback_summary,
        }

    result = {
        "status": "ROLLBACK_CONTROL_DONE",
        "created_utc": now(),
        "script": rel(_public_path('experiments/archive/functional_learning/scripts/rollback_control.py')),
        "scientific_question": "Does clean eval-mode deterministic parent KL improve the dense-mask trade-off beyond simple rollback of the acquisition-only private-adapter update at matched ordinary-text parent KL?",
        "paths": {
            "parent": rel(PARENT),
            "densemask_acquisition_only": rel(DENSEMASK),
            "clean_eval_preservation": rel(CLEAN_EVAL),
            "sparse_reference": rel(SPARSE),
            "out_dir": rel(out_dir),
        },
        "device": str(device),
        "private_update_validation": update_validation,
        "calibration_records": calib_summary,
        "load_info": {
            "parent": parent_info,
            "clean_eval": clean_info,
            "densemask": dense_info,
            "rollback_base": rollback_base_info,
        },
        "match": {
            "n_positions": len(labels),
            "clean_eval_kl_mean": mean(clean_kls),
            "clean_eval_kl_median": median(clean_kls),
            "densemask_kl_mean": mean(dense_kls),
            "densemask_kl_median": median(dense_kls),
            "parent_calibration": parent_calib,
            "clean_eval_calibration": clean_calib,
            "densemask_calibration": dense_calib,
            "alpha_grid": alpha_rows,
            "selected_alpha": selected_alpha,
            "selected_alpha_kl_mean": selected.get("kl_mean"),
            "selected_alpha_abs_error": selected.get("abs_error_to_clean_kl_mean"),
            "rollback_calibration": rollback_calib,
            "rollback_temperature_for_source_readout": rollback_temperature,
        },
        "evidence_conditions": evidence_conditions,
        "evidence": evidence_result,
        "evidence_selected": evidence_selected,
        "source_response": source_result,
        "source_selected": source_selected,
    }
    # High-level interpretation without overclaiming.
    interp: Dict[str, Any] = {}
    ev_sel = evidence_selected.get("rollback_matched_densemask_update", {})
    clean_ev = evidence_selected.get("clean_pres_lambda1_eval_full80_step090", {})
    dense_ev = evidence_selected.get("densemask_sparselabel_seed62064_step090", {})
    if ev_sel and clean_ev:
        interp["evidence_comparison"] = "Compare rollback_matched_densemask_update to clean_pres_lambda1_eval_full80_step090 at matched ordinary calibration KL. If rollback has materially weaker source response or larger wrong/view/source-erased costs, deterministic KL is doing more than simple update shrinkage; if they closely coincide, simple attenuation remains sufficient."
    if source_selected:
        interp["source_response_comparison"] = "Qwen and common-source summaries are non-official mechanism readouts; they test retained source-conditioned behavior after drift matching, not benchmark score."
    interp["official_score_status"] = "This script does not replace the official-sized zero-shot/Reading, repaired SuperGLUE, or measured AoA comparisons required for the candidate endpoint."
    result["interpretation"] = interp

    out_json = out_dir / "rollback_control.json"
    out_json.write_text(json.dumps(result, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    write_markdown(out_dir / "rollback_control.md", result)
    print(json.dumps({"status": result["status"], "out_json": rel(out_json), "out_md": rel(out_dir / "rollback_control.md"), "selected_alpha": selected_alpha, "clean_kl": mean(clean_kls), "selected_kl": selected.get("kl_mean")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
