#!/usr/bin/env python3
"""research: full EWoK interaction turnover analysis for coupled sparse20 dual-view.

Scientific purpose
------------------
research/180 showed that coupled sparse20 aligned and shuffled both repair many rows in
an old fixed EWoK stable-failure subset, while broad competence collapses. The
strategist's question is whether this is real full-surface interaction improvement or
turnover/churn: repaired baseline-wrong interactions may be offset by losses among
baseline-correct interactions, and four-cell margins may merely cross zero weakly.

This script uses existing checkpoints only. It can score one target's full 7,618-row
EWoK four-cell pseudo-likelihood surface (via the validated research reader) and then
aggregate turnover across targets. It performs no training and does not define any
submission metric.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse
import csv
import importlib.util
import json
import math
import os
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

# Set writable caches before importing any module that might import transformers.
def find_user_root() -> Path:
    return _PUBLIC_ROOT

USER_ROOT = find_user_root()
os.chdir(USER_ROOT)
A01_WS = USER_ROOT / "experiments/archive" / 'representation_and_objectives'
A02_WS = USER_ROOT / "experiments/archive" / 'frontier_consolidation'
OUT_ROOT = A01_WS / "data" / "full_ewok_coupled_turnover"
CACHE_SUFFIX = os.environ.get("CACHE_SUFFIX", "default")
CACHE_ROOT = OUT_ROOT / "hf_cache" / CACHE_SUFFIX
for name, path in {
    "HOME": CACHE_ROOT / "home",
    "XDG_CACHE_HOME": CACHE_ROOT / "xdg",
    "HF_HOME": CACHE_ROOT / "hf_home",
    "HF_MODULES_CACHE": CACHE_ROOT / "hf_modules",
    "TRANSFORMERS_CACHE": CACHE_ROOT / "transformers",
    "TORCH_HOME": CACHE_ROOT / "torch",
    "TMPDIR": CACHE_ROOT / "tmp",
}.items():
    os.environ.setdefault(name, str(path))
    Path(os.environ[name]).mkdir(parents=True, exist_ok=True)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

SCRIPT_DIR = A01_WS / "scripts"
research = SCRIPT_DIR / "fw_ewok_interaction_reader.py"
NOTE_PATH = A01_WS / "notes" / "full_ewok_coupled_turnover.md"
SUMMARY_JSON = OUT_ROOT / "full_ewok_coupled_turnover_summary.json"

TARGETS: dict[str, dict[str, Any]] = {
    "mlm_only_20M": {
        "label": "A02 exact MLM-only 20M baseline for the coupled sparse20 panel",
        "model_path": A02_WS / "training/runs/dualview_mlm_only_20M_seed43022/hf_model/final",
        "official_payload": A02_WS / "data/dualview_mlm_only_20m_eval/per_target/dualview_mlm_only_20M.json",
        "role": "baseline",
    },
    "coupled_aligned_20M": {
        "label": "A02 coupled sparse20 true-correspondence run at 20M",
        "model_path": A02_WS / "training/runs/coupled_sparse20_aligned_20M_seed43022/hf_model/final",
        "official_payload": A02_WS / "data/coupled_sparse20_aligned_20m_eval/per_target/coupled_sparse20_aligned_20M.json",
        "role": "coupled_aligned",
    },
    "coupled_shuffled_20M": {
        "label": "A01 matched coupled sparse20 shuffled-correspondence control at 20M",
        "model_path": A01_WS / "training/runs/coupled_sparse20_shuffled_20M_seed43022/hf_model/final",
        "official_payload": None,
        "role": "coupled_shuffled",
    },
}

NUMERIC_FIELDS = {
    "global_index", "local_index", "context_diff_n_hunks",
    "s11_sum", "s21_sum", "s12_sum", "s22_sum", "prior_t1_sum", "prior_t2_sum",
    "del_s11_sum", "del_s21_sum", "del_s12_sum", "del_s22_sum",
    "local_c1_actual_sum", "local_c1_swapped_sum", "local_c2_actual_sum", "local_c2_swapped_sum",
    "s11_mean", "s21_mean", "s12_mean", "s22_mean", "prior_t1_mean", "prior_t2_mean",
    "del_s11_mean", "del_s21_mean", "del_s12_mean", "del_s22_mean",
    "official_margin_t1_sum", "official_margin_t2_sum", "within_context_margin_c1_sum", "within_context_margin_c2_sum",
    "interaction_sum", "deletion_interaction_sum", "interaction_minus_deletion_sum",
    "pmi_within_context_margin_c1_sum", "pmi_within_context_margin_c2_sum",
    "official_margin_t1_mean", "official_margin_t2_mean", "within_context_margin_c1_mean", "within_context_margin_c2_mean",
    "interaction_mean", "deletion_interaction_mean", "interaction_minus_deletion_mean",
    "pmi_within_context_margin_c1_mean", "pmi_within_context_margin_c2_mean",
    "local_c1_actual_over_swapped_sum", "local_c2_actual_over_swapped_sum",
}
BOOL_FIELDS = {
    "deleted_contexts_identical", "saved_model_correct_flag", "saved_model_wrong_flag",
    "both_official_sum_positive", "both_within_context_sum_positive", "both_within_context_mean_positive",
    "stable_nonpositive_interaction", "conditional_reversal_failure_stable", "local_both_actual_over_swapped_positive",
}


def now_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def rel(p: Path | str | None) -> str | None:
    if p is None:
        return None
    pp = Path(p)
    try:
        return str(pp.resolve().relative_to(USER_ROOT))
    except Exception:
        return str(pp)


def import_from(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def finite(x: Any) -> bool:
    try:
        return math.isfinite(float(x))
    except Exception:
        return False


def as_float(x: Any) -> float:
    try:
        v = float(x)
        return v if math.isfinite(v) else float("nan")
    except Exception:
        return float("nan")


def qstats(vals: Iterable[Any]) -> dict[str, Any]:
    xs = sorted(float(v) for v in vals if finite(v))
    if not xs:
        return {"n": 0}
    def q(p: float) -> float:
        if len(xs) == 1:
            return xs[0]
        idx = p * (len(xs) - 1)
        lo = math.floor(idx); hi = math.ceil(idx)
        if lo == hi:
            return xs[lo]
        return xs[lo] * (hi - idx) + xs[hi] * (idx - lo)
    return {
        "n": len(xs),
        "min": xs[0],
        "p01": q(0.01),
        "p05": q(0.05),
        "p25": q(0.25),
        "mean": statistics.fmean(xs),
        "median": statistics.median(xs),
        "p75": q(0.75),
        "p95": q(0.95),
        "p99": q(0.99),
        "max": xs[-1],
    }


def parse_bool(x: Any) -> bool:
    if isinstance(x, bool):
        return x
    if isinstance(x, (int, float)):
        return bool(x)
    s = str(x).strip().lower()
    return s in {"true", "1", "yes", "y"}


def read_json(path: Path | None) -> Any | None:
    if path is None or not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def official_ewok_score(payload_path: Path | None) -> dict[str, Any]:
    payload = read_json(payload_path)
    if not isinstance(payload, dict):
        return {"path": rel(payload_path), "exists": bool(payload_path and payload_path.exists()), "score": None, "predictions": None}
    task = payload.get("tasks", {}).get("EWoK", {}) if isinstance(payload.get("tasks"), dict) else {}
    return {
        "path": rel(payload_path),
        "exists": True,
        "score": task.get("score"),
        "predictions": task.get("predictions"),
        "report": task.get("report"),
        "returncode": task.get("returncode"),
    }


def target_dir(target: str) -> Path:
    return OUT_ROOT / "targets" / target


def records_csv(target: str) -> Path:
    return target_dir(target) / "ewok_interaction_records.csv"


def summary_path(target: str) -> Path:
    return target_dir(target) / "ewok_interaction_summary.json"


def full_target_ready(target: str) -> bool:
    sp = summary_path(target)
    rp = records_csv(target)
    if not (sp.exists() and rp.exists()):
        return False
    try:
        summary = json.loads(sp.read_text(encoding="utf-8"))
        n = summary.get("summary", {}).get("n") if isinstance(summary.get("summary"), dict) else None
        return int(n) == 7618 and int(summary.get("row_limit") or 0) == 0 and int(summary.get("row_offset") or 0) == 0
    except Exception:
        return False


def score_target(target: str, device_name: str, threads: int, row_limit: int, row_offset: int, row_batch_size: int, masked_batch_size: int, force: bool) -> dict[str, Any]:
    meta = TARGETS[target]
    out_dir = target_dir(target)
    out_dir.mkdir(parents=True, exist_ok=True)
    if full_target_ready(target) and not force:
        return {"status": "SKIPPED_EXISTING_FULL", "target": target, "records_csv": rel(records_csv(target)), "summary_json": rel(summary_path(target))}
    ew = import_from(research, f"full_ewok_step181_{target}")
    # Preserve official full EWoK root from research, but place no files through ew.OUT_ROOT.
    import torch
    if threads > 0:
        torch.set_num_threads(threads)
    device = torch.device("cuda" if device_name == "cuda" and torch.cuda.is_available() else "cpu")
    print(json.dumps({
        "event": "score_target_start", "target": target, "device": str(device),
        "model_path": rel(meta["model_path"]), "row_limit": row_limit,
        "row_offset": row_offset, "row_batch_size": row_batch_size,
        "masked_batch_size": masked_batch_size, "utc": now_utc(),
    }), flush=True)
    res = ew.run_target(target, Path(meta["model_path"]), device, threads, row_limit, row_offset, row_batch_size, masked_batch_size)
    records = res.pop("records")
    ew.write_csv(records_csv(target), records)
    ew.write_csv(out_dir / "ewok_interaction_by_domain.csv", res["by_domain"])
    ew.write_csv(out_dir / "ewok_interaction_by_context_diff.csv", res["by_context_diff"])
    full = {
        **res,
        "target": target,
        "label": meta["label"],
        "model_path": rel(meta["model_path"]),
        "official_ewok_payload": official_ewok_score(meta.get("official_payload")),
        "records_csv": rel(records_csv(target)),
        "created_utc": now_utc(),
    }
    summary_path(target).write_text(json.dumps(full, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "event": "score_target_done", "target": target, "summary": full.get("summary"),
        "summary_json": rel(summary_path(target)), "records_csv": rel(records_csv(target)),
    }, ensure_ascii=False), flush=True)
    return {"status": "SCORED", "target": target, "summary_json": rel(summary_path(target)), "records_csv": rel(records_csv(target))}


def load_records(target: str) -> list[dict[str, Any]]:
    path = records_csv(target)
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        for raw in csv.DictReader(f):
            r: dict[str, Any] = dict(raw)
            for k in list(r.keys()):
                if k in NUMERIC_FIELDS:
                    r[k] = as_float(r[k])
                elif k in BOOL_FIELDS:
                    r[k] = parse_bool(r[k])
            if finite(r.get("global_index")):
                r["global_index"] = int(r["global_index"])
            rows.append(r)
    return rows


def sign(x: Any) -> int:
    v = as_float(x)
    if not math.isfinite(v):
        return 0
    return 1 if v > 0 else (-1 if v < 0 else 0)


def row_predicate_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    return {
        "n": n,
        "accuracy": sum(1 for r in rows if r.get("saved_model_correct_flag")) / n if n else None,
        "stable_failure": sum(1 for r in rows if r.get("conditional_reversal_failure_stable")),
        "stable_failure_frac_all": sum(1 for r in rows if r.get("conditional_reversal_failure_stable")) / n if n else None,
        "positive_interaction_sum": sum(1 for r in rows if as_float(r.get("interaction_sum")) > 0),
        "positive_both_official": sum(1 for r in rows if r.get("both_official_sum_positive")),
        "positive_both_within": sum(1 for r in rows if r.get("both_within_context_sum_positive")),
        "interaction_sum": qstats(r.get("interaction_sum") for r in rows),
        "interaction_mean": qstats(r.get("interaction_mean") for r in rows),
        "official_margin_t1_sum": qstats(r.get("official_margin_t1_sum") for r in rows),
        "official_margin_t2_sum": qstats(r.get("official_margin_t2_sum") for r in rows),
        "within_context_margin_c1_sum": qstats(r.get("within_context_margin_c1_sum") for r in rows),
        "within_context_margin_c2_sum": qstats(r.get("within_context_margin_c2_sum") for r in rows),
    }


def subset_stats(indices: list[int], base: dict[int, dict[str, Any]], cand: dict[int, dict[str, Any]]) -> dict[str, Any]:
    pairs = [(base[i], cand[i]) for i in indices if i in base and i in cand]
    n = len(pairs)
    if n == 0:
        return {"n": 0}
    delta_inter = [as_float(c.get("interaction_sum")) - as_float(b.get("interaction_sum")) for b, c in pairs]
    base_inter = [as_float(b.get("interaction_sum")) for b, _ in pairs]
    cand_inter = [as_float(c.get("interaction_sum")) for _, c in pairs]
    delta_t1 = [as_float(c.get("official_margin_t1_sum")) - as_float(b.get("official_margin_t1_sum")) for b, c in pairs]
    delta_t2 = [as_float(c.get("official_margin_t2_sum")) - as_float(b.get("official_margin_t2_sum")) for b, c in pairs]
    base_correct = [bool(b.get("saved_model_correct_flag")) for b, _ in pairs]
    cand_correct = [bool(c.get("saved_model_correct_flag")) for _, c in pairs]
    return {
        "n": n,
        "base_accuracy": sum(base_correct) / n,
        "cand_accuracy": sum(cand_correct) / n,
        "net_correct_delta_count": sum(cand_correct) - sum(base_correct),
        "base_stable_failure": sum(1 for b, _ in pairs if b.get("conditional_reversal_failure_stable")),
        "cand_stable_failure": sum(1 for _, c in pairs if c.get("conditional_reversal_failure_stable")),
        "stable_failure_delta_count": sum(1 for _, c in pairs if c.get("conditional_reversal_failure_stable")) - sum(1 for b, _ in pairs if b.get("conditional_reversal_failure_stable")),
        "interaction_sum_delta": qstats(delta_inter),
        "interaction_sum_base": qstats(base_inter),
        "interaction_sum_cand": qstats(cand_inter),
        "official_margin_t1_delta": qstats(delta_t1),
        "official_margin_t2_delta": qstats(delta_t2),
        "interaction_delta_positive_frac": sum(1 for v in delta_inter if finite(v) and v > 0) / n,
        "interaction_delta_negative_frac": sum(1 for v in delta_inter if finite(v) and v < 0) / n,
        "base_interaction_positive_frac": sum(1 for v in base_inter if finite(v) and v > 0) / n,
        "cand_interaction_positive_frac": sum(1 for v in cand_inter if finite(v) and v > 0) / n,
        "cand_interaction_gt_0p25_frac": sum(1 for v in cand_inter if finite(v) and v > 0.25) / n,
        "cand_interaction_gt_0p50_frac": sum(1 for v in cand_inter if finite(v) and v > 0.50) / n,
        "cand_interaction_gt_1p00_frac": sum(1 for v in cand_inter if finite(v) and v > 1.00) / n,
    }


def compare_candidate(name: str, base_rows: list[dict[str, Any]], cand_rows: list[dict[str, Any]]) -> dict[str, Any]:
    base = {int(r["global_index"]): r for r in base_rows}
    cand = {int(r["global_index"]): r for r in cand_rows}
    common = sorted(set(base) & set(cand))
    by_transition: dict[str, list[int]] = defaultdict(list)
    by_interaction_transition: dict[str, list[int]] = defaultdict(list)
    by_stable_transition: dict[str, list[int]] = defaultdict(list)
    by_domain_delta: dict[str, dict[str, Any]] = {}
    by_context_delta: dict[str, dict[str, Any]] = {}
    per_row: list[dict[str, Any]] = []
    for i in common:
        b = base[i]; c = cand[i]
        bc = bool(b.get("saved_model_correct_flag")); cc = bool(c.get("saved_model_correct_flag"))
        if (not bc) and cc:
            t = "repaired_base_wrong"
        elif bc and (not cc):
            t = "broken_base_correct"
        elif bc and cc:
            t = "both_correct"
        else:
            t = "both_wrong"
        by_transition[t].append(i)
        bi = sign(b.get("interaction_sum")); ci = sign(c.get("interaction_sum"))
        key = ("pos" if bi > 0 else "nonpos") + "_to_" + ("pos" if ci > 0 else "nonpos")
        by_interaction_transition[key].append(i)
        bs = bool(b.get("conditional_reversal_failure_stable")); cs = bool(c.get("conditional_reversal_failure_stable"))
        sk = ("stablefail" if bs else "notstablefail") + "_to_" + ("stablefail" if cs else "notstablefail")
        by_stable_transition[sk].append(i)
        d_inter = as_float(c.get("interaction_sum")) - as_float(b.get("interaction_sum"))
        per_row.append({
            "global_index": i,
            "domain": b.get("domain"),
            "ContextDiff": b.get("ContextDiff"),
            "TargetDiff": b.get("TargetDiff"),
            "transition": t,
            "interaction_transition": key,
            "stable_failure_transition": sk,
            "base_correct": bc,
            "cand_correct": cc,
            "base_stable_failure": bs,
            "cand_stable_failure": cs,
            "base_interaction_sum": b.get("interaction_sum"),
            "cand_interaction_sum": c.get("interaction_sum"),
            "delta_interaction_sum": d_inter,
            "base_official_margin_t1_sum": b.get("official_margin_t1_sum"),
            "cand_official_margin_t1_sum": c.get("official_margin_t1_sum"),
            "delta_official_margin_t1_sum": as_float(c.get("official_margin_t1_sum")) - as_float(b.get("official_margin_t1_sum")),
            "base_official_margin_t2_sum": b.get("official_margin_t2_sum"),
            "cand_official_margin_t2_sum": c.get("official_margin_t2_sum"),
            "delta_official_margin_t2_sum": as_float(c.get("official_margin_t2_sum")) - as_float(b.get("official_margin_t2_sum")),
            "target1": b.get("target1"),
            "target2": b.get("target2"),
            "context_diff_c1_texts_joined": b.get("context_diff_c1_texts_joined"),
            "context_diff_c2_texts_joined": b.get("context_diff_c2_texts_joined"),
        })
    # domain/context subset summaries
    for field, out in [("domain", by_domain_delta), ("ContextDiff", by_context_delta)]:
        groups: dict[str, list[int]] = defaultdict(list)
        for i in common:
            groups[str(base[i].get(field) or "")].append(i)
        for k, idxs in groups.items():
            s = subset_stats(idxs, base, cand)
            out[k] = {
                "n": s.get("n"),
                "net_correct_delta_count": s.get("net_correct_delta_count"),
                "stable_failure_delta_count": s.get("stable_failure_delta_count"),
                "interaction_delta_mean": s.get("interaction_sum_delta", {}).get("mean"),
                "interaction_delta_median": s.get("interaction_sum_delta", {}).get("median"),
                "base_accuracy": s.get("base_accuracy"),
                "cand_accuracy": s.get("cand_accuracy"),
            }
    transition_counts = {k: len(v) for k, v in sorted(by_transition.items())}
    repaired = by_transition.get("repaired_base_wrong", [])
    broken = by_transition.get("broken_base_correct", [])
    common_stats = subset_stats(common, base, cand)
    repaired_stats = subset_stats(repaired, base, cand)
    broken_stats = subset_stats(broken, base, cand)
    base_wrong = [i for i in common if not base[i].get("saved_model_correct_flag")]
    base_correct = [i for i in common if base[i].get("saved_model_correct_flag")]
    base_stablefail = [i for i in common if base[i].get("conditional_reversal_failure_stable")]
    base_nonstable = [i for i in common if not base[i].get("conditional_reversal_failure_stable")]
    # Write row deltas for later inspection; keep all rows but compact fields.
    delta_csv = OUT_ROOT / "comparisons" / f"{name}_row_deltas.csv"
    delta_csv.parent.mkdir(parents=True, exist_ok=True)
    with delta_csv.open("w", encoding="utf-8", newline="") as f:
        keys = list(per_row[0].keys()) if per_row else []
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader(); w.writerows(per_row)
    # Samples: strongest repairs/breaks by interaction delta.
    repairs_sorted = sorted([r for r in per_row if r["transition"] == "repaired_base_wrong"], key=lambda r: as_float(r["delta_interaction_sum"]), reverse=True)
    breaks_sorted = sorted([r for r in per_row if r["transition"] == "broken_base_correct"], key=lambda r: as_float(r["delta_interaction_sum"]))
    return {
        "candidate": name,
        "n_common": len(common),
        "transition_counts": transition_counts,
        "net_correct_delta_count": transition_counts.get("repaired_base_wrong", 0) - transition_counts.get("broken_base_correct", 0),
        "interaction_transition_counts": {k: len(v) for k, v in sorted(by_interaction_transition.items())},
        "stable_failure_transition_counts": {k: len(v) for k, v in sorted(by_stable_transition.items())},
        "all_rows": common_stats,
        "base_wrong_rows": subset_stats(base_wrong, base, cand),
        "base_correct_rows": subset_stats(base_correct, base, cand),
        "base_stable_failure_rows": subset_stats(base_stablefail, base, cand),
        "base_not_stable_failure_rows": subset_stats(base_nonstable, base, cand),
        "repaired_base_wrong_rows": repaired_stats,
        "broken_base_correct_rows": broken_stats,
        "turnover_balance": {
            "repaired_base_wrong_count": len(repaired),
            "broken_base_correct_count": len(broken),
            "repair_minus_break": len(repaired) - len(broken),
            "breaks_per_repair": (len(broken) / len(repaired)) if repaired else None,
            "official_accuracy_delta_points": 100.0 * (len(repaired) - len(broken)) / len(common) if common else None,
            "repaired_fraction_with_cand_interaction_gt_0p50": repaired_stats.get("cand_interaction_gt_0p50_frac"),
            "broken_fraction_with_cand_interaction_nonpositive": None if not broken else sum(1 for i in broken if as_float(cand[i].get("interaction_sum")) <= 0) / len(broken),
        },
        "domain_delta_summary": dict(sorted(by_domain_delta.items(), key=lambda kv: (-(kv[1].get("stable_failure_delta_count") or 0), kv[0]))),
        "contextdiff_delta_summary": dict(sorted(by_context_delta.items(), key=lambda kv: (-(kv[1].get("stable_failure_delta_count") or 0), kv[0]))),
        "row_delta_csv": rel(delta_csv),
        "repair_samples_high_delta": repairs_sorted[:8],
        "break_samples_low_delta": breaks_sorted[:8],
    }


def compare_aligned_shuffled(aligned: list[dict[str, Any]], shuffled: list[dict[str, Any]]) -> dict[str, Any]:
    a = {int(r["global_index"]): r for r in aligned}
    s = {int(r["global_index"]): r for r in shuffled}
    common = sorted(set(a) & set(s))
    rows = []
    for i in common:
        rows.append({
            "global_index": i,
            "domain": a[i].get("domain"),
            "ContextDiff": a[i].get("ContextDiff"),
            "aligned_correct": bool(a[i].get("saved_model_correct_flag")),
            "shuffled_correct": bool(s[i].get("saved_model_correct_flag")),
            "aligned_stable_failure": bool(a[i].get("conditional_reversal_failure_stable")),
            "shuffled_stable_failure": bool(s[i].get("conditional_reversal_failure_stable")),
            "aligned_interaction_sum": a[i].get("interaction_sum"),
            "shuffled_interaction_sum": s[i].get("interaction_sum"),
            "aligned_minus_shuffled_interaction_sum": as_float(a[i].get("interaction_sum")) - as_float(s[i].get("interaction_sum")),
        })
    dints = [r["aligned_minus_shuffled_interaction_sum"] for r in rows]
    return {
        "n_common": len(common),
        "aligned_correct_shuffled_wrong": sum(1 for r in rows if r["aligned_correct"] and not r["shuffled_correct"]),
        "shuffled_correct_aligned_wrong": sum(1 for r in rows if r["shuffled_correct"] and not r["aligned_correct"]),
        "both_correct": sum(1 for r in rows if r["aligned_correct"] and r["shuffled_correct"]),
        "both_wrong": sum(1 for r in rows if not r["aligned_correct"] and not r["shuffled_correct"]),
        "aligned_stable_failure": sum(1 for r in rows if r["aligned_stable_failure"]),
        "shuffled_stable_failure": sum(1 for r in rows if r["shuffled_stable_failure"]),
        "aligned_minus_shuffled_stable_failure_count": sum(1 for r in rows if r["aligned_stable_failure"]) - sum(1 for r in rows if r["shuffled_stable_failure"]),
        "aligned_minus_shuffled_interaction_sum": qstats(dints),
        "aligned_minus_shuffled_interaction_positive_frac": sum(1 for v in dints if finite(v) and v > 0) / len(dints) if dints else None,
        "aligned_minus_shuffled_interaction_negative_frac": sum(1 for v in dints if finite(v) and v < 0) / len(dints) if dints else None,
    }


def preflight() -> dict[str, Any]:
    d: dict[str, Any] = {"status": "READY", "created_utc": now_utc(), "targets": {}}
    for name, meta in TARGETS.items():
        summary_n = None
        summary_row_limit = None
        summary_row_offset = None
        if summary_path(name).exists():
            try:
                sd = json.loads(summary_path(name).read_text(encoding="utf-8"))
                if isinstance(sd.get("summary"), dict):
                    summary_n = sd["summary"].get("n")
                summary_row_limit = sd.get("row_limit")
                summary_row_offset = sd.get("row_offset")
            except Exception:
                pass
        rec = {
            "model_path": rel(meta["model_path"]),
            "model_exists": Path(meta["model_path"]).exists(),
            "records_csv": rel(records_csv(name)),
            "records_exists": records_csv(name).exists(),
            "summary_json": rel(summary_path(name)),
            "summary_exists": summary_path(name).exists(),
            "summary_n": summary_n,
            "summary_row_limit": summary_row_limit,
            "summary_row_offset": summary_row_offset,
            "full_records_ready": full_target_ready(name),
            "official_ewok_payload": official_ewok_score(meta.get("official_payload")),
        }
        if not rec["model_exists"]:
            d["status"] = "NOT_READY"
        d["targets"][name] = rec
    d["reader"] = {"path": rel(research), "exists": research.exists()}
    if not research.exists():
        d["status"] = "NOT_READY"
    return d


def aggregate(require_all: bool = True) -> dict[str, Any]:
    pf = preflight()
    ready = [name for name in TARGETS if full_target_ready(name)]
    missing = [name for name in TARGETS if not full_target_ready(name)]
    out: dict[str, Any] = {
        "status": "PENDING" if missing else "PASS",
        "created_utc": now_utc(),
        "boundary": "Full 7,618-row EWoK four-cell turnover analysis; no training and no custom submission scoring.",
        "preflight": pf,
        "ready_targets": ready,
        "missing_targets": missing,
    }
    if missing and require_all:
        SUMMARY_JSON.parent.mkdir(parents=True, exist_ok=True)
        SUMMARY_JSON.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        return out
    records = {name: load_records(name) for name in ready}
    out["target_summaries"] = {name: row_predicate_summary(rows) for name, rows in records.items()}
    if "mlm_only_20M" in records:
        comps: dict[str, Any] = {}
        for cand in ["coupled_aligned_20M", "coupled_shuffled_20M"]:
            if cand in records:
                comps[f"{cand}_minus_mlm_only_20M"] = compare_candidate(cand, records["mlm_only_20M"], records[cand])
        if "coupled_aligned_20M" in records and "coupled_shuffled_20M" in records:
            comps["coupled_aligned_minus_coupled_shuffled"] = compare_aligned_shuffled(records["coupled_aligned_20M"], records["coupled_shuffled_20M"])
        out["comparisons"] = comps
    out["scientific_reading"] = make_reading(out)
    SUMMARY_JSON.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_JSON.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_note(out)
    return out


def make_reading(out: dict[str, Any]) -> dict[str, Any]:
    comps = out.get("comparisons", {}) if isinstance(out.get("comparisons"), dict) else {}
    reading: dict[str, Any] = {}
    for key in ["coupled_aligned_20M_minus_mlm_only_20M", "coupled_shuffled_20M_minus_mlm_only_20M"]:
        c = comps.get(key)
        if not isinstance(c, dict):
            continue
        all_rows = c.get("all_rows", {})
        turnover = c.get("turnover_balance", {})
        reading[key] = {
            "net_ewok_accuracy_delta_points": turnover.get("official_accuracy_delta_points"),
            "repaired_minus_broken_count": turnover.get("repair_minus_break"),
            "breaks_per_repair": turnover.get("breaks_per_repair"),
            "all_rows_interaction_mean_delta": all_rows.get("interaction_sum_delta", {}).get("mean") if isinstance(all_rows.get("interaction_sum_delta"), dict) else None,
            "all_rows_interaction_median_delta": all_rows.get("interaction_sum_delta", {}).get("median") if isinstance(all_rows.get("interaction_sum_delta"), dict) else None,
            "base_correct_interaction_mean_delta": c.get("base_correct_rows", {}).get("interaction_sum_delta", {}).get("mean") if isinstance(c.get("base_correct_rows", {}).get("interaction_sum_delta"), dict) else None,
            "base_wrong_interaction_mean_delta": c.get("base_wrong_rows", {}).get("interaction_sum_delta", {}).get("mean") if isinstance(c.get("base_wrong_rows", {}).get("interaction_sum_delta"), dict) else None,
            "stable_failure_delta_count": all_rows.get("stable_failure_delta_count"),
            "reading": None,
        }
        # conservative interpretation as data, not a procedural gate
        net = turnover.get("official_accuracy_delta_points")
        bpr = turnover.get("breaks_per_repair")
        med = reading[key]["all_rows_interaction_median_delta"]
        if isinstance(net, (int, float)) and isinstance(bpr, (int, float)) and isinstance(med, (int, float)):
            if abs(net) < 0.25 and bpr > 0.9 and med <= 0.05:
                reading[key]["reading"] = "turnover-dominated: many repaired baseline-wrong rows are offset by broken baseline-correct rows, with weak median interaction movement"
            elif net > 0.75 and med > 0.05:
                reading[key]["reading"] = "net full-surface interaction improvement: enough to justify a sharply isolated non-endpoint mechanism experiment"
            else:
                reading[key]["reading"] = "mixed full-surface movement: inspect row/domain deltas before any training"
    if isinstance(comps.get("coupled_aligned_minus_coupled_shuffled"), dict):
        aa = comps["coupled_aligned_minus_coupled_shuffled"]
        reading["alignment_specificity_full_surface"] = {
            "aligned_correct_shuffled_wrong": aa.get("aligned_correct_shuffled_wrong"),
            "shuffled_correct_aligned_wrong": aa.get("shuffled_correct_aligned_wrong"),
            "aligned_minus_shuffled_stable_failure_count": aa.get("aligned_minus_shuffled_stable_failure_count"),
            "interaction_mean_delta": aa.get("aligned_minus_shuffled_interaction_sum", {}).get("mean") if isinstance(aa.get("aligned_minus_shuffled_interaction_sum"), dict) else None,
            "interaction_median_delta": aa.get("aligned_minus_shuffled_interaction_sum", {}).get("median") if isinstance(aa.get("aligned_minus_shuffled_interaction_sum"), dict) else None,
        }
    return reading


def fmt(x: Any, nd: int = 4) -> str:
    if isinstance(x, (int, float)) and math.isfinite(float(x)):
        return f"{float(x):.{nd}f}"
    return "NA"


def write_note(out: dict[str, Any]) -> None:
    NOTE_PATH.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    lines.append("# research full EWoK coupled-turnover analysis\n\n")
    lines.append(f"Status: **{out.get('status')}**\n\n")
    lines.append("This is a no-training full-surface readout over the official 7,618-row EWoK data using the validated research four-cell pseudo-likelihood reader. It asks whether the coupled sparse20 movement is real interaction improvement or row turnover/churn.\n\n")
    if out.get("missing_targets"):
        lines.append(f"Missing target records: `{out.get('missing_targets')}`. Ready: `{out.get('ready_targets')}`.\n\n")
    ts = out.get("target_summaries", {}) if isinstance(out.get("target_summaries"), dict) else {}
    if ts:
        lines.append("## Target full-surface summaries\n\n")
        lines.append("| target | n | accuracy | stable failures | positive interaction | mean interaction | median interaction |\n")
        lines.append("|---|---:|---:|---:|---:|---:|---:|\n")
        for name, s in ts.items():
            lines.append(f"| {name} | {s.get('n')} | {fmt(s.get('accuracy'))} | {s.get('stable_failure')} | {s.get('positive_interaction_sum')} | {fmt(s.get('interaction_sum', {}).get('mean') if isinstance(s.get('interaction_sum'), dict) else None)} | {fmt(s.get('interaction_sum', {}).get('median') if isinstance(s.get('interaction_sum'), dict) else None)} |\n")
        lines.append("\n")
    comps = out.get("comparisons", {}) if isinstance(out.get("comparisons"), dict) else {}
    for key in ["coupled_aligned_20M_minus_mlm_only_20M", "coupled_shuffled_20M_minus_mlm_only_20M"]:
        c = comps.get(key)
        if not isinstance(c, dict):
            continue
        tb = c.get("turnover_balance", {})
        allr = c.get("all_rows", {})
        bw = c.get("base_wrong_rows", {})
        bc = c.get("base_correct_rows", {})
        bs = c.get("base_stable_failure_rows", {})
        lines.append(f"## {key}\n\n")
        lines.append(f"Official-row turnover: repaired baseline-wrong {tb.get('repaired_base_wrong_count')}, broken baseline-correct {tb.get('broken_base_correct_count')}, repair-minus-break {tb.get('repair_minus_break')}, breaks/repair {fmt(tb.get('breaks_per_repair'))}, accuracy delta {fmt(tb.get('official_accuracy_delta_points'))} points.\n\n")
        lines.append(f"All-row interaction_sum delta mean {fmt(allr.get('interaction_sum_delta', {}).get('mean') if isinstance(allr.get('interaction_sum_delta'), dict) else None)}, median {fmt(allr.get('interaction_sum_delta', {}).get('median') if isinstance(allr.get('interaction_sum_delta'), dict) else None)}, positive-delta fraction {fmt(allr.get('interaction_delta_positive_frac'))}. Stable-failure delta {allr.get('stable_failure_delta_count')}.\n\n")
        lines.append(f"Baseline-wrong rows: n {bw.get('n')}, interaction delta mean {fmt(bw.get('interaction_sum_delta', {}).get('mean') if isinstance(bw.get('interaction_sum_delta'), dict) else None)}, median {fmt(bw.get('interaction_sum_delta', {}).get('median') if isinstance(bw.get('interaction_sum_delta'), dict) else None)}, candidate interaction >0.5 frac {fmt(bw.get('cand_interaction_gt_0p50_frac'))}.\n\n")
        lines.append(f"Baseline-correct rows: n {bc.get('n')}, interaction delta mean {fmt(bc.get('interaction_sum_delta', {}).get('mean') if isinstance(bc.get('interaction_sum_delta'), dict) else None)}, median {fmt(bc.get('interaction_sum_delta', {}).get('median') if isinstance(bc.get('interaction_sum_delta'), dict) else None)}, candidate interaction >0.5 frac {fmt(bc.get('cand_interaction_gt_0p50_frac'))}.\n\n")
        lines.append(f"Baseline stable-failure rows: n {bs.get('n')}, stable-failure delta {bs.get('stable_failure_delta_count')}, interaction delta mean {fmt(bs.get('interaction_sum_delta', {}).get('mean') if isinstance(bs.get('interaction_sum_delta'), dict) else None)}, median {fmt(bs.get('interaction_sum_delta', {}).get('median') if isinstance(bs.get('interaction_sum_delta'), dict) else None)}.\n\n")
        lines.append(f"Row-delta CSV: `{c.get('row_delta_csv')}`.\n\n")
    aa = comps.get("coupled_aligned_minus_coupled_shuffled")
    if isinstance(aa, dict):
        lines.append("## Aligned vs shuffled on the full surface\n\n")
        lines.append(f"Aligned-correct/shuffled-wrong {aa.get('aligned_correct_shuffled_wrong')}; shuffled-correct/aligned-wrong {aa.get('shuffled_correct_aligned_wrong')}; both-correct {aa.get('both_correct')}; both-wrong {aa.get('both_wrong')}; aligned-minus-shuffled stable failures {aa.get('aligned_minus_shuffled_stable_failure_count')}.\n\n")
        ais = aa.get("aligned_minus_shuffled_interaction_sum", {}) if isinstance(aa.get("aligned_minus_shuffled_interaction_sum"), dict) else {}
        lines.append(f"Aligned-minus-shuffled interaction_sum mean {fmt(ais.get('mean'))}, median {fmt(ais.get('median'))}, positive fraction {fmt(aa.get('aligned_minus_shuffled_interaction_positive_frac'))}.\n\n")
    sr = out.get("scientific_reading", {}) if isinstance(out.get("scientific_reading"), dict) else {}
    if sr:
        lines.append("## Scientific reading\n\n")
        for k, v in sr.items():
            lines.append(f"- `{k}`: {json.dumps(v, ensure_ascii=False)}\n")
        lines.append("\n")
    lines.append(f"JSON: `{rel(SUMMARY_JSON)}`\n")
    NOTE_PATH.write_text("".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", choices=sorted(TARGETS), help="score one target")
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--row_limit", type=int, default=0)
    ap.add_argument("--row_offset", type=int, default=0)
    ap.add_argument("--row_batch_size", type=int, default=64)
    ap.add_argument("--masked_batch_size", type=int, default=512)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--preflight", action="store_true")
    ap.add_argument("--aggregate", action="store_true")
    ap.add_argument("--allow-partial-aggregate", action="store_true")
    args = ap.parse_args()
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    if args.preflight:
        pf = preflight()
        p = OUT_ROOT / "preflight.json"
        p.write_text(json.dumps(pf, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(json.dumps({"status": pf["status"], "preflight": rel(p)}, ensure_ascii=False), flush=True)
        return
    if args.target:
        res = score_target(args.target, args.device, args.threads, args.row_limit, args.row_offset, args.row_batch_size, args.masked_batch_size, args.force)
        print(json.dumps(res, ensure_ascii=False), flush=True)
        return
    if args.aggregate:
        res = aggregate(require_all=not args.allow_partial_aggregate)
        print(json.dumps({"status": res.get("status"), "summary_json": rel(SUMMARY_JSON), "note": rel(NOTE_PATH), "missing": res.get("missing_targets")}, ensure_ascii=False), flush=True)
        return
    ap.error("provide --target, --preflight, or --aggregate")


if __name__ == "__main__":
    main()
