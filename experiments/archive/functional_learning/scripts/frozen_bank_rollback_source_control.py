#!/usr/bin/env python3
"""research repaired rollback source-response control on one frozen bank.

The research rollback control correctly matched an interpolation of the acquisition-only
(M,S) private-adapter update to clean preservation on an ordinary-text KL substrate,
but its Qwen source-response section scored the rollback on a newly generated Qwen
bank while comparing it to parent/clean/dense summaries loaded from research.  This
script repairs that comparison by building one frozen Qwen/common source bank and
scoring all relevant models on exactly the same masked inputs, targets, and wrong-
source assignments:

  * coherent86 parent
  * acquisition-only dense-mask/sparse-label endpoint
  * clean eval-mode preservation endpoint
  * rollback interpolation theta(alpha)=parent+alpha*(densemask-parent)

Alpha is selected only from ordinary non-Qwen KL(parent||model), never from behavior.
The source bank is then scored for alpha=0, alpha=1, the closest drift match, and the
calibration bracket around the clean KL.  Alpha=0 and alpha=1 must reproduce parent
and acquisition-only dense-mask scores on the same bank; otherwise the interpolation
or scoring path is invalid.

This is a mechanism/evidence readout, not official BabyLM scoring and not a candidate
checkpoint.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse
import csv
import json
import math
import pathlib
import statistics
import sys
import time
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from safetensors.torch import load_file

ROOT = _public_path('.')
STUDY = _public_path('experiments/archive/functional_learning')
SCRIPTS = _public_path('experiments/archive/functional_learning/scripts')
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import corrected_bridge_trainer as bridge  # noqa: E402
import temperature_source_readout as s86  # noqa: E402
import rollback_control as s92  # noqa: E402

DEFAULT_OUT = _public_path('experiments/archive/functional_learning/data/frozen_bank_rollback_source_control')
PARENT = s92.PARENT
DENSEMASK = s92.DENSEMASK
CLEAN_EVAL = s92.CLEAN_EVAL


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


def parse_float_grid(text: str) -> List[float]:
    vals: List[float] = []
    for piece in str(text).replace(",", " ").split():
        if piece.strip():
            vals.append(float(piece))
    if not vals:
        raise ValueError("empty float grid")
    return sorted(set(round(v, 10) for v in vals))


def default_alpha_grid() -> List[float]:
    vals = [0.0, 0.25, 0.5, 0.6, 0.65, 0.675, 0.7, 0.725, 0.75,
            0.7625, 0.775, 0.7875, 0.8, 0.825, 0.85, 0.9, 1.0]
    return sorted(set(vals))


def write_jsonl(path: pathlib.Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def load_model(path: Optional[pathlib.Path], device: torch.device, private_scale: float):
    model, info = s92.load_model(path, device, private_scale)
    model.eval()
    return model, info


def torch_clear() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def compare_score_rows(a_rows: List[Dict[str, Any]], b_rows: List[Dict[str, Any]], *,
                       key_fields: Sequence[str] = ("task_id",),
                       metrics: Sequence[str] = ("nll_T1", "nll_Tfit", "rank_mean")) -> Dict[str, Any]:
    def make_key(row: Dict[str, Any]) -> Tuple[str, ...]:
        return tuple(str(row.get(k, "")) for k in key_fields)

    bmap = {make_key(r): r for r in b_rows}
    out: Dict[str, Any] = {"n_a": len(a_rows), "n_b": len(b_rows), "key_fields": list(key_fields), "n_matched": 0, "n_missing_in_b": 0, "metrics": {}}
    diffs: Dict[str, List[float]] = {m: [] for m in metrics}
    max_abs: Dict[str, float] = {m: 0.0 for m in metrics}
    max_key: Dict[str, Optional[Tuple[str, ...]]] = {m: None for m in metrics}
    duplicate_keys_b = len(b_rows) - len(bmap)
    for r in a_rows:
        kk = make_key(r)
        b = bmap.get(kk)
        if b is None:
            out["n_missing_in_b"] += 1
            continue
        out["n_matched"] += 1
        for m in metrics:
            if r.get(m) is None or b.get(m) is None:
                continue
            d = float(r[m]) - float(b[m])
            diffs[m].append(d)
            ad = abs(d)
            if ad > max_abs[m]:
                max_abs[m] = ad
                max_key[m] = kk
    out["duplicate_keys_in_b"] = int(duplicate_keys_b)
    for m, vals in diffs.items():
        out["metrics"][m] = {
            "n": len(vals),
            "mean_a_minus_b": mean(vals),
            "median_a_minus_b": median(vals),
            "sem": sem(vals),
            "max_abs_diff": max_abs[m],
            "max_abs_key": list(max_key[m]) if max_key[m] is not None else None,
        }
    return out


def row_signature(rows: List[Dict[str, Any]], n: int = 5) -> List[Dict[str, Any]]:
    sig = []
    for r in rows[:n]:
        sig.append({
            "task_id": r.get("task_id"),
            "base_task_id": r.get("base_task_id"),
            "row_idx": r.get("row_idx"),
            "segment_index": r.get("segment_index"),
            "pair_id": r.get("pair_id"),
            "condition": r.get("condition"),
            "target_word": r.get("target_word"),
            "source_pair_id_used": r.get("source_pair_id_used"),
            "candidate": r.get("candidate"),
            "candidate_role": r.get("candidate_role"),
        })
    return sig


def find_bracket(alpha_rows: List[Dict[str, Any]], target_kl: float) -> Dict[str, Any]:
    below = [r for r in alpha_rows if r.get("kl_mean") is not None and float(r["kl_mean"]) <= target_kl]
    above = [r for r in alpha_rows if r.get("kl_mean") is not None and float(r["kl_mean"]) >= target_kl]
    lo = max(below, key=lambda r: float(r["kl_mean"])) if below else None
    hi = min(above, key=lambda r: float(r["kl_mean"])) if above else None
    return {
        "lower_alpha": None if lo is None else float(lo["alpha"]),
        "lower_kl": None if lo is None else float(lo["kl_mean"]),
        "upper_alpha": None if hi is None else float(hi["alpha"]),
        "upper_kl": None if hi is None else float(hi["kl_mean"]),
        "bracket_width_alpha": None if lo is None or hi is None else float(hi["alpha"]) - float(lo["alpha"]),
        "bracket_width_kl": None if lo is None or hi is None else float(hi["kl_mean"]) - float(lo["kl_mean"]),
    }


def compact_source_summary(summary: Dict[str, Any]) -> Dict[str, Any]:
    q = summary.get("qwen_source", {}).get("source_specificity", {})
    c = summary.get("common_source_reversal", {}).get("source_follow", {})
    return {
        "temperature": summary.get("temperature"),
        "calibration_kl_vs_parent": summary.get("calibration_kl_vs_parent"),
        "calibration_n_positions": summary.get("calibration", {}).get("n_positions"),
        "qwen_mean_delta_specific_advantage_Tfit_vs_parent": q.get("mean_delta_specific_advantage_Tfit_vs_parent"),
        "qwen_mean_delta_specific_rank_advantage_vs_parent": q.get("mean_delta_specific_rank_advantage_vs_parent"),
        "qwen_mean_delta_total_advantage_Tfit_vs_parent": q.get("mean_delta_total_advantage_Tfit_vs_parent"),
        "qwen_complete_triplets": q.get("n_complete_triplets"),
        "common_both_correct_Tfit": c.get("both_correct_Tfit"),
        "common_mean_delta_source_follow_swing_Tfit_vs_parent": c.get("mean_delta_source_follow_swing_Tfit_vs_parent"),
        "common_mean_delta_source_follow_rank_swing_vs_parent": c.get("mean_delta_source_follow_rank_swing_vs_parent"),
        "common_n_decision_items": c.get("n_decision_items"),
    }


def write_markdown(path: pathlib.Path, result: Dict[str, Any]) -> None:
    lines: List[str] = []
    lines.append("# research frozen-bank rollback source-response control\n\n")
    lines.append("All source-response rows in this file were scored on one newly materialized frozen bank. research's Qwen comparison loaded parent/clean/dense source summaries from research while regenerating rollback tasks, so its striking Qwen rollback sign is not used as evidence here.\n\n")
    m = result.get("match", {})
    lines.append("## Ordinary-text KL match\n\n")
    lines.append(f"- Clean KL(parent||model): `{m.get('clean_eval_kl_mean')}` over `{m.get('n_positions')}` calibration positions.\n")
    lines.append(f"- Dense-mask alpha=1 KL: `{m.get('densemask_kl_mean')}`.\n")
    lines.append(f"- Selected alpha: `{m.get('selected_alpha')}` with KL `{m.get('selected_alpha_kl_mean')}` and relative error `{m.get('selected_alpha_relative_error')}`.\n")
    br = m.get("selected_bracket", {})
    lines.append(f"- Calibration bracket: lower alpha `{br.get('lower_alpha')}` KL `{br.get('lower_kl')}`, upper alpha `{br.get('upper_alpha')}` KL `{br.get('upper_kl')}`.\n\n")
    lines.append("## Frozen-bank source summaries\n\n")
    for name, summ in result.get("source_selected", {}).items():
        lines.append(f"- `{name}`: Qwen Δspecific Tfit `{summ.get('qwen_mean_delta_specific_advantage_Tfit_vs_parent')}`, Δrank `{summ.get('qwen_mean_delta_specific_rank_advantage_vs_parent')}`; common Δswing Tfit `{summ.get('common_mean_delta_source_follow_swing_Tfit_vs_parent')}`, Δrank `{summ.get('common_mean_delta_source_follow_rank_swing_vs_parent')}`, both `{summ.get('common_both_correct_Tfit')}`/25.\n")
    lines.append("\n## Endpoint interpolation checks\n\n")
    checks = result.get("interpolation_endpoint_checks", {})
    for name, obj in checks.items():
        qmax = obj.get("qwen", {}).get("metrics", {}).get("nll_T1", {}).get("max_abs_diff")
        cmax = obj.get("common", {}).get("metrics", {}).get("nll_T1", {}).get("max_abs_diff")
        lines.append(f"- `{name}`: qwen max |ΔNLL_T1| `{qmax}`, common max |ΔNLL_T1| `{cmax}`, pass `{obj.get('pass')}`.\n")
    lines.append("\n## Interpretation\n\n")
    for k, v in result.get("interpretation", {}).items():
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
    ap.add_argument("--alpha-grid", default="", help="Calibration alpha grid. Default is a fine bracket near the research selected alpha plus endpoints.")
    ap.add_argument("--source-segments", type=int, default=120)
    ap.add_argument("--source-targets-per-segment", type=int, default=4)
    ap.add_argument("--source-seed", type=int, default=86067)
    ap.add_argument("--max-updates", type=int, default=s92.s90.PREFIX_UPDATES)
    ap.add_argument("--words-per-update", type=int, default=s92.s90.WORDS_PER_UPDATE)
    ap.add_argument("--include-all-alpha-scores", action="store_true", help="Score every calibration alpha on the source bank; otherwise score endpoints, closest match, and bracket only.")
    args = ap.parse_args()

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.device == "auto":
        device = torch.device(f"cuda:{int(args.gpu)}" if torch.cuda.is_available() else "cpu")
    elif args.device == "cuda":
        device = torch.device(f"cuda:{int(args.gpu)}")
    else:
        device = torch.device("cpu")

    alpha_grid = parse_float_grid(args.alpha_grid) if str(args.alpha_grid).strip() else default_alpha_grid()
    temp_grid = parse_float_grid(args.temperature_grid)
    update_validation = s92.validate_private_update(PARENT, DENSEMASK)
    if not update_validation.get("valid_private_only_update"):
        raise RuntimeError(f"private update validation failed: {update_validation}")

    tokenizer = bridge.AutoTokenizer.from_pretrained(str(PARENT), local_files_only=True, use_fast=True)
    calibration, calib_summary = s92.build_calibration(tokenizer, args)
    qwen_tasks, qwen_stats, common_tasks, common_stats = s92.build_source_tasks(tokenizer, args)
    write_jsonl(out_dir / "calibration_records.jsonl", (s86.strip_task_for_json(t) for t in calibration))
    write_jsonl(out_dir / "qwen_source_tasks.jsonl", (s86.strip_task_for_json(t) for t in qwen_tasks))
    write_jsonl(out_dir / "common_source_reversal_records.jsonl", (s86.strip_task_for_json(t) for t in common_tasks))
    print(json.dumps({"event": "banks_built", "calib_records": len(calibration), "qwen_tasks": len(qwen_tasks), "common_tasks": len(common_tasks), "device": str(device)}), flush=True)

    # Load and score true endpoints on the exact same bank.
    endpoint_specs: List[Tuple[str, Optional[pathlib.Path]]] = [
        ("coherent86", None),
        ("densemask_sparselabel_seed62064", DENSEMASK),
        ("clean_pres_lambda1_eval_full80", CLEAN_EVAL),
    ]
    endpoint_info: Dict[str, Any] = {}
    endpoint_logits: Dict[str, List[torch.Tensor]] = {}
    endpoint_labels: Dict[str, List[int]] = {}
    endpoint_calib: Dict[str, Any] = {}
    endpoint_scores: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}

    parent_logits: Optional[List[torch.Tensor]] = None
    parent_labels: Optional[List[int]] = None
    parent_qwen: Optional[List[Dict[str, Any]]] = None
    parent_common: Optional[List[Dict[str, Any]]] = None

    for name, path in endpoint_specs:
        model, info = load_model(path, device, float(args.private_scale))
        endpoint_info[name] = info
        logits, labels, _ = s86.gather_position_logits(model, tokenizer, calibration, device, int(args.batch_size))
        if parent_labels is not None and labels != parent_labels:
            raise RuntimeError(f"calibration labels differ for {name}")
        calib = s92.summarize_calibration_logits(logits, labels, temp_grid)
        if name == "coherent86":
            parent_logits = logits
            parent_labels = labels
        assert parent_logits is not None and parent_labels is not None
        if name == "coherent86":
            kl_mean = 0.0
        else:
            kl_mean = mean(s92.position_kl_teacher_to_model(parent_logits, logits, 1.0))
        temp = float(calib["temperature_fit"].get("best_temperature", 1.0))
        qrows = s86.score_tasks_with_temperature(model, tokenizer, qwen_tasks, device, int(args.batch_size), temp)
        crows = s86.score_tasks_with_temperature(model, tokenizer, common_tasks, device, int(args.batch_size), temp)
        write_jsonl(out_dir / f"qwen_scores_{name}.jsonl", qrows)
        write_jsonl(out_dir / f"common_scores_{name}.jsonl", crows)
        endpoint_logits[name] = logits
        endpoint_labels[name] = labels
        endpoint_calib[name] = {"calibration": calib, "temperature": temp, "kl_vs_parent": kl_mean}
        endpoint_scores[name] = {"qwen": qrows, "common": crows}
        if name == "coherent86":
            parent_qwen = qrows
            parent_common = crows
        print(json.dumps({"event": "endpoint_scored", "model": name, "temperature": temp, "kl_vs_parent": kl_mean}), flush=True)
        del model
        torch_clear()

    if parent_logits is None or parent_labels is None or parent_qwen is None or parent_common is None:
        raise RuntimeError("parent scoring failed")
    clean_kl_mean = float(endpoint_calib["clean_pres_lambda1_eval_full80"]["kl_vs_parent"])
    dense_kl_mean = float(endpoint_calib["densemask_sparselabel_seed62064"]["kl_vs_parent"])

    parent_sd = load_file(str(PARENT / "model.safetensors"))
    dense_sd = load_file(str(DENSEMASK / "model.safetensors"))
    rollback_model, rollback_info = load_model(None, device, float(args.private_scale))

    alpha_rows: List[Dict[str, Any]] = []
    selected: Optional[Dict[str, Any]] = None
    for alpha in alpha_grid:
        if alpha < -1e-9 or alpha > 1.000000001:
            raise ValueError(f"alpha outside [0,1]: {alpha}")
        s92.apply_private_interpolation(rollback_model, parent_sd, dense_sd, float(alpha))
        logits, labels, _ = s86.gather_position_logits(rollback_model, tokenizer, calibration, device, int(args.batch_size))
        if labels != parent_labels:
            raise RuntimeError(f"rollback labels differ at alpha={alpha}")
        kls = s92.position_kl_teacher_to_model(parent_logits, logits, 1.0)
        calib = s92.summarize_calibration_logits(logits, labels, temp_grid)
        row = {
            "alpha": float(alpha),
            "kl_mean": mean(kls),
            "kl_median": median(kls),
            "kl_sem": sem(kls),
            "abs_error_to_clean_kl_mean": abs(float(mean(kls) or 0.0) - clean_kl_mean),
            "relative_error_to_clean_kl_mean": (abs(float(mean(kls) or 0.0) - clean_kl_mean) / clean_kl_mean) if clean_kl_mean else None,
            "calibration": calib,
        }
        alpha_rows.append(row)
        if selected is None or float(row["abs_error_to_clean_kl_mean"]) < float(selected["abs_error_to_clean_kl_mean"]):
            selected = row
        print(json.dumps({"event": "alpha_calibrated", "alpha": alpha, "kl_mean": row["kl_mean"], "clean_kl": clean_kl_mean, "rel_error": row["relative_error_to_clean_kl_mean"]}), flush=True)
    if selected is None:
        raise RuntimeError("no selected alpha")
    bracket = find_bracket(alpha_rows, clean_kl_mean)
    if args.include_all_alpha_scores:
        score_alphas = [float(r["alpha"]) for r in alpha_rows]
    else:
        score_set = {0.0, 1.0, float(selected["alpha"])}
        if bracket.get("lower_alpha") is not None:
            score_set.add(float(bracket["lower_alpha"]))
        if bracket.get("upper_alpha") is not None:
            score_set.add(float(bracket["upper_alpha"]))
        score_alphas = sorted(score_set)

    rollback_scores: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
    rollback_calibs: Dict[str, Any] = {}
    for alpha in score_alphas:
        s92.apply_private_interpolation(rollback_model, parent_sd, dense_sd, float(alpha))
        # Retrieve the already computed calibration if alpha is in the grid.
        arow = min(alpha_rows, key=lambda r: abs(float(r["alpha"]) - float(alpha)))
        temp = float(arow["calibration"]["temperature_fit"].get("best_temperature", 1.0))
        qrows = s86.score_tasks_with_temperature(rollback_model, tokenizer, qwen_tasks, device, int(args.batch_size), temp)
        crows = s86.score_tasks_with_temperature(rollback_model, tokenizer, common_tasks, device, int(args.batch_size), temp)
        tag = f"rollback_alpha_{str(alpha).replace('.', 'p')}"
        write_jsonl(out_dir / f"qwen_scores_{tag}.jsonl", qrows)
        write_jsonl(out_dir / f"common_scores_{tag}.jsonl", crows)
        rollback_scores[tag] = {"qwen": qrows, "common": crows}
        rollback_calibs[tag] = {"alpha": float(alpha), "temperature": temp, "calibration": arow["calibration"], "kl_vs_parent": arow.get("kl_mean")}
        print(json.dumps({"event": "rollback_scored", "alpha": alpha, "tag": tag, "temperature": temp, "kl_vs_parent": arow.get("kl_mean")}), flush=True)

    # Summaries and endpoint reproduction checks.
    all_summaries: Dict[str, Any] = {}
    for name, rows in endpoint_scores.items():
        all_summaries[name] = {
            "temperature": endpoint_calib[name]["temperature"],
            "calibration_kl_vs_parent": endpoint_calib[name]["kl_vs_parent"],
            "calibration": endpoint_calib[name]["calibration"],
            "qwen_source": s86.summarize_qwen_source(rows["qwen"], parent_qwen),
            "common_source_reversal": s86.summarize_common(rows["common"], parent_common),
        }
    for tag, rows in rollback_scores.items():
        all_summaries[tag] = {
            "temperature": rollback_calibs[tag]["temperature"],
            "calibration_kl_vs_parent": rollback_calibs[tag]["kl_vs_parent"],
            "calibration": rollback_calibs[tag]["calibration"],
            "qwen_source": s86.summarize_qwen_source(rows["qwen"], parent_qwen),
            "common_source_reversal": s86.summarize_common(rows["common"], parent_common),
        }

    endpoint_checks: Dict[str, Any] = {}
    common_key = ("task_id", "condition", "candidate_role", "candidate")
    if "rollback_alpha_0p0" in rollback_scores:
        qcomp = compare_score_rows(rollback_scores["rollback_alpha_0p0"]["qwen"], parent_qwen, key_fields=("task_id",))
        ccomp = compare_score_rows(rollback_scores["rollback_alpha_0p0"]["common"], parent_common, key_fields=common_key)
        endpoint_checks["rollback_alpha_0_vs_parent"] = {"qwen": qcomp, "common": ccomp}
    if "rollback_alpha_1p0" in rollback_scores:
        qcomp = compare_score_rows(rollback_scores["rollback_alpha_1p0"]["qwen"], endpoint_scores["densemask_sparselabel_seed62064"]["qwen"], key_fields=("task_id",))
        ccomp = compare_score_rows(rollback_scores["rollback_alpha_1p0"]["common"], endpoint_scores["densemask_sparselabel_seed62064"]["common"], key_fields=common_key)
        endpoint_checks["rollback_alpha_1_vs_densemask"] = {"qwen": qcomp, "common": ccomp}
    # Pass when exact endpoint interpolation reproduces the real endpoint to numerical tolerance.
    for ck, obj in endpoint_checks.items():
        maxes: List[float] = []
        for part in ["qwen", "common"]:
            for metric in ["nll_T1", "rank_mean"]:
                val = obj.get(part, {}).get("metrics", {}).get(metric, {}).get("max_abs_diff")
                if val is not None:
                    maxes.append(float(val))
        # CPU/GPU and repeated forward passes can differ by a few 1e-6 in NLL even
        # when alpha=0/1 are the same mathematical checkpoint.  Rank must still
        # match exactly in these checks; the NLL tolerance is deliberately tiny
        # relative to the observed source-response effects.
        obj["pass"] = bool(maxes and max(maxes) <= 2e-5)
        obj["endpoint_reproduction_tolerance_nll"] = 2e-5
        obj["max_abs_diff_over_checked_metrics"] = max(maxes) if maxes else None

    selected_alpha = float(selected["alpha"])
    selected_tag = f"rollback_alpha_{str(selected_alpha).replace('.', 'p')}"
    source_selected = {name: compact_source_summary(summ) for name, summ in all_summaries.items()}
    compact_rows = []
    for name, row in source_selected.items():
        rr = {"model": name, **row}
        compact_rows.append(rr)
    with (out_dir / "compact_source_summary.csv").open("w", encoding="utf-8", newline="") as f:
        fieldnames = sorted(set(k for r in compact_rows for k in r.keys()))
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in compact_rows:
            w.writerow(r)

    # Record original-bank mismatch in a small, bounded way for provenance.
    prior_mismatch: Dict[str, Any] = {}
    p92 = _public_path('experiments/archive/functional_learning/data/rollback_control_full/qwen_source_tasks.jsonl')
    p90 = _public_path('experiments/archive/functional_learning/data/temperature_source_readout_clean/qwen_source_tasks.jsonl')
    if p92.exists() and p90.exists():
        def read_first(path: pathlib.Path, n: int = 6) -> List[Dict[str, Any]]:
            rows: List[Dict[str, Any]] = []
            with path.open(encoding="utf-8") as f:
                for i, line in enumerate(f):
                    if i >= n:
                        break
                    if line.strip():
                        rows.append(json.loads(line))
            return rows
        a = read_first(p92)
        b = read_first(p90)
        prior_mismatch = {
            "rollback_probe_qwen_task_path": rel(p92),
            "preservation_probe_qwen_task_path": rel(p90),
            "first_rows_equal": row_signature(a, len(a)) == row_signature(b, len(b)),
            "rollback_probe_first_signature": row_signature(a),
            "preservation_probe_first_signature": row_signature(b),
            "interpretation": "research source-response rollback should not be compared to research parent/clean/dense source summaries when these banks differ.",
        }

    interpretation: Dict[str, Any] = {
        "source_bank_repair": "Parent, acquisition-only dense-mask, clean preservation, and rollback alphas are scored on one frozen bank, so source-specific deltas now share task_id, target word, masked input, and wrong-source assignment.",
        "alpha_matching": "Alpha is selected only by ordinary non-Qwen KL(parent||model). Behavioral source-response values are reported for the selected alpha and the calibration bracket to avoid treating a 6% KL mismatch as exact.",
        "endpoint_validation": "Rollback alpha=0 must reproduce coherent86 and alpha=1 must reproduce dense-mask on this bank; failure would invalidate interpolation scoring rather than support a mechanism.",
        "mechanism_reading": "If the selected/bracket rollback resembles clean preservation on source response as well as evidence-availability costs, drift attenuation explains most of the clean benefit. If clean remains materially different on the same bank, trajectory or training-time constraints remain plausible but must be separated from official score evidence.",
    }

    result = {
        "status": "FROZEN_BANK_ROLLBACK_SOURCE_CONTROL_DONE",
        "created_utc": now(),
        "script": rel(_public_path('experiments/archive/functional_learning/scripts/frozen_bank_rollback_source_control.py')),
        "scientific_question": "Does clean eval-mode preservation retain source response beyond what a KL-matched interpolation of the acquisition-only dense-mask update shows, when all models are scored on one frozen source-response bank?",
        "paths": {
            "parent": rel(PARENT),
            "densemask_acquisition_only": rel(DENSEMASK),
            "clean_eval_preservation": rel(CLEAN_EVAL),
            "out_dir": rel(out_dir),
            "qwen_source_tasks": rel(out_dir / "qwen_source_tasks.jsonl"),
            "common_source_reversal_records": rel(out_dir / "common_source_reversal_records.jsonl"),
            "compact_source_summary_csv": rel(out_dir / "compact_source_summary.csv"),
        },
        "device": str(device),
        "private_update_validation": update_validation,
        "prior_mismatch_note": prior_mismatch,
        "bank_stats": {
            "calibration": calib_summary,
            "qwen": qwen_stats,
            "common": common_stats,
        },
        "bank_signatures": {
            "qwen_first_rows": row_signature([s86.strip_task_for_json(t) for t in qwen_tasks[:6]], 6),
            "common_first_rows": row_signature([s86.strip_task_for_json(t) for t in common_tasks[:6]], 6),
        },
        "load_info": {"endpoints": endpoint_info, "rollback_base": rollback_info},
        "match": {
            "n_positions": len(parent_labels),
            "clean_eval_kl_mean": clean_kl_mean,
            "densemask_kl_mean": dense_kl_mean,
            "alpha_grid": alpha_rows,
            "selected_alpha": selected_alpha,
            "selected_alpha_kl_mean": selected.get("kl_mean"),
            "selected_alpha_abs_error": selected.get("abs_error_to_clean_kl_mean"),
            "selected_alpha_relative_error": selected.get("relative_error_to_clean_kl_mean"),
            "selected_bracket": bracket,
            "source_scored_alphas": score_alphas,
        },
        "calibration_summaries": {**endpoint_calib, **rollback_calibs},
        "source_summaries": all_summaries,
        "source_selected": source_selected,
        "interpolation_endpoint_checks": endpoint_checks,
        "score_paths": {
            "qwen": {name: rel(out_dir / f"qwen_scores_{name}.jsonl") for name in list(endpoint_scores.keys()) + list(rollback_scores.keys())},
            "common": {name: rel(out_dir / f"common_scores_{name}.jsonl") for name in list(endpoint_scores.keys()) + list(rollback_scores.keys())},
        },
        "interpretation": interpretation,
    }
    # Remove malformed leftover key if present; retained via bank_signatures instead.
    result["bank_stats"].pop("qwen_first_signature", None)

    out_json = out_dir / "frozen_bank_rollback_source_control.json"
    out_json.write_text(json.dumps(result, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    write_markdown(out_dir / "frozen_bank_rollback_source_control.md", result)
    print(json.dumps({
        "status": result["status"],
        "out_json": rel(out_json),
        "out_md": rel(out_dir / "frozen_bank_rollback_source_control.md"),
        "selected_alpha": selected_alpha,
        "clean_kl": clean_kl_mean,
        "selected_kl": selected.get("kl_mean"),
        "bracket": bracket,
        "endpoint_checks": {k: v.get("pass") for k, v in endpoint_checks.items()},
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
