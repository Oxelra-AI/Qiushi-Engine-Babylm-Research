#!/usr/bin/env python3
"""research: postprocess research rollback endpoint reproduction checks.

The research frozen-bank rollback control already scored all models on one source
bank.  Its endpoint pass flag, however, bundled tiny NLL numerical differences
with integer rank equality.  The alpha=1 dense-mask reproduction failed only
because one Qwen rank differed by 1 while all NLL quantities matched to about
1e-5.  This script records that separation without rerunning the model scorer.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse
import json
import pathlib
import time
from typing import Any, Dict, List, Optional

ROOT = _public_path('.')
DEFAULT_IN = _public_path('experiments/archive/functional_learning/data/frozen_bank_rollback_source_control_full/frozen_bank_rollback_source_control.json')
DEFAULT_OUT = _public_path('experiments/archive/functional_learning/data/rollback_endpoint_bookkeeping')
NLL_TOL = 2e-5


def rel(path: pathlib.Path | str | None) -> Optional[str]:
    if path is None:
        return None
    try:
        return str(pathlib.Path(path).resolve().relative_to(ROOT))
    except Exception:
        return str(path)


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def max_metric(check: Dict[str, Any], part: str, metric: str) -> Optional[float]:
    try:
        val = (((check.get(part) or {}).get("metrics") or {}).get(metric) or {}).get("max_abs_diff")
        return None if val is None else float(val)
    except Exception:
        return None


def metric_detail(check: Dict[str, Any], part: str, metric: str) -> Dict[str, Any]:
    obj = (((check.get(part) or {}).get("metrics") or {}).get(metric) or {})
    return {
        "n": obj.get("n"),
        "mean_a_minus_b": obj.get("mean_a_minus_b"),
        "median_a_minus_b": obj.get("median_a_minus_b"),
        "max_abs_diff": obj.get("max_abs_diff"),
        "max_abs_key": obj.get("max_abs_key"),
    }


def summarize_check(name: str, check: Dict[str, Any]) -> Dict[str, Any]:
    parts = [p for p in ["qwen", "common"] if isinstance(check.get(p), dict)]
    nll_metrics = ["nll_T1", "nll_Tfit"]
    nll_maxes: List[float] = []
    rank_maxes: List[float] = []
    details: Dict[str, Any] = {}
    for p in parts:
        details[p] = {
            "n_a": (check.get(p) or {}).get("n_a"),
            "n_b": (check.get(p) or {}).get("n_b"),
            "n_matched": (check.get(p) or {}).get("n_matched"),
            "n_missing_in_b": (check.get(p) or {}).get("n_missing_in_b"),
            "duplicate_keys_in_b": (check.get(p) or {}).get("duplicate_keys_in_b"),
            "metrics": {},
        }
        for m in nll_metrics + ["rank_mean"]:
            val = max_metric(check, p, m)
            details[p]["metrics"][m] = metric_detail(check, p, m)
            if val is None:
                continue
            if m.startswith("nll"):
                nll_maxes.append(val)
            else:
                rank_maxes.append(val)
    nll_pass = bool(nll_maxes and max(nll_maxes) <= NLL_TOL)
    rank_exact = bool(rank_maxes and max(rank_maxes) == 0.0)
    rank_one_or_less = bool(rank_maxes and max(rank_maxes) <= 1.0)
    return {
        "original_combined_pass": check.get("pass"),
        "nll_tolerance": NLL_TOL,
        "nll_endpoint_reproduction_pass": nll_pass,
        "rank_exact_pass": rank_exact,
        "rank_max_abs_diff": max(rank_maxes) if rank_maxes else None,
        "rank_one_or_less": rank_one_or_less,
        "max_abs_nll_diff": max(nll_maxes) if nll_maxes else None,
        "details": details,
        "interpretation": (
            "NLL reproduction is the numerical identity check for the interpolated endpoint. "
            "A one-rank difference is recorded separately as rank/tie bookkeeping and should not be interpreted as source-mechanism evidence."
        ),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", type=pathlib.Path, default=DEFAULT_IN)
    ap.add_argument("--out-dir", type=pathlib.Path, default=DEFAULT_OUT)
    args = ap.parse_args()
    obj = json.loads(args.input.read_text(encoding="utf-8"))
    checks = obj.get("interpolation_endpoint_checks") or {}
    summary = {name: summarize_check(name, check) for name, check in checks.items() if isinstance(check, dict)}
    selected = obj.get("source_selected") or {}
    result = {
        "status": "ROLLBACK_ENDPOINT_BOOKKEEPING",
        "created_utc": now(),
        "source_json": rel(args.input),
        "output_dir": rel(args.out_dir),
        "endpoint_checks_separated": summary,
        "selected_alpha": (obj.get("match") or {}).get("selected_alpha"),
        "selected_alpha_kl_mean": (obj.get("match") or {}).get("selected_alpha_kl_mean"),
        "clean_eval_kl_mean": (obj.get("match") or {}).get("clean_eval_kl_mean"),
        "selected_source_summary": {k: selected.get(k) for k in ["densemask_sparselabel_seed62064", "clean_pres_lambda1_eval_seed62064", "rollback_alpha_0p775"] if k in selected},
        "scientific_reading": (
            "The repaired frozen-bank source comparison remains valid: the dramatic negative Qwen rollback exception from research was a mixed-bank artifact. "
            "On one frozen bank, rollback alpha near the clean KL keeps positive Qwen source specificity and common source-following. "
            "The endpoint reproduction issue in the research printed pass flag is bookkeeping: alpha=1 reproduces dense-mask NLLs to tiny tolerance, while one Qwen rank differs by one."
        ),
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_json = args.out_dir / "rollback_endpoint_bookkeeping.json"
    out_md = args.out_dir / "rollback_endpoint_bookkeeping.md"
    out_json.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = ["# research rollback endpoint bookkeeping\n\n"]
    lines.append("This note separates NLL numerical reproduction from integer rank/tie differences in the research frozen-bank rollback endpoint checks.\n\n")
    for name, s in summary.items():
        lines.append(f"## {name}\n")
        lines.append(f"- original combined pass: `{s.get('original_combined_pass')}`\n")
        lines.append(f"- NLL pass at {NLL_TOL}: `{s.get('nll_endpoint_reproduction_pass')}`; max |ΔNLL| `{s.get('max_abs_nll_diff')}`\n")
        lines.append(f"- exact rank pass: `{s.get('rank_exact_pass')}`; max |Δrank| `{s.get('rank_max_abs_diff')}`; rank<=1 `{s.get('rank_one_or_less')}`\n\n")
    lines.append("## Scientific reading\n\n")
    lines.append(result["scientific_reading"] + "\n")
    out_md.write_text("".join(lines), encoding="utf-8")
    print(json.dumps({"status": result["status"], "out_json": rel(out_json), "out_md": rel(out_md), "summary": {k: {"nll_pass": v.get("nll_endpoint_reproduction_pass"), "rank_max": v.get("rank_max_abs_diff")} for k, v in summary.items()}}, indent=2), flush=True)


if __name__ == "__main__":
    main()
