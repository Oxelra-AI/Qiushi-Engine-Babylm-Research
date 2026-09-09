#!/usr/bin/env python3
"""Summarize research coherence-margin pilot training trace."""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import json
import pathlib
import statistics
import time
from typing import Any

ROOT = _public_path('.')
WORKSPACE = _public_path('experiments/archive/frontier_consolidation')
RUN = _public_path('experiments/archive/frontier_consolidation/training/runs/cohmargin4M_scale1p75_seed43022')
OUT = _public_path('experiments/archive/frontier_consolidation/data/cohmargin_training_trace_summary')


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def rel(p: pathlib.Path | str) -> str:
    q = pathlib.Path(p)
    try:
        return str(q.resolve().relative_to(ROOT))
    except Exception:
        return str(q)


def load_json(path: pathlib.Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def mean(xs: list[float]) -> float | None:
    return float(sum(xs) / len(xs)) if xs else None


def slope(first: float, last: float, n: int) -> float | None:
    if n <= 1:
        return None
    return float((last - first) / (n - 1))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    log = load_jsonl(_public_path('experiments/archive/frontier_consolidation/training/runs/cohmargin4M_scale1p75_seed43022/training_log.jsonl'))
    metrics = load_json(_public_path('experiments/archive/frontier_consolidation/training/runs/cohmargin4M_scale1p75_seed43022/scientific_metrics.json'))
    manifest = load_json(_public_path('experiments/archive/frontier_consolidation/training/runs/cohmargin4M_scale1p75_seed43022/coherence_margin_manifest.json'))
    gaps = [float(r["nll_bad_minus_coh_last_micro"]) for r in log]
    margins = [float(r["margin_loss"]) for r in log]
    mlm = [float(r["mlm_loss"]) for r in log]
    loss = [float(r["loss"]) for r in log]
    out = {
        "status": "COMPLETE",
        "created_utc": now(),
        "run_dir": rel(RUN),
        "manifest": rel(_public_path('experiments/archive/frontier_consolidation/training/runs/cohmargin4M_scale1p75_seed43022/coherence_margin_manifest.json')),
        "metrics": rel(_public_path('experiments/archive/frontier_consolidation/training/runs/cohmargin4M_scale1p75_seed43022/scientific_metrics.json')),
        "training_log": rel(_public_path('experiments/archive/frontier_consolidation/training/runs/cohmargin4M_scale1p75_seed43022/training_log.jsonl')),
        "n_records": len(log),
        "charged_word_exposure": metrics.get("charged_word_exposure"),
        "coherent_words": metrics.get("coherent_words"),
        "optimizer_steps": metrics.get("actual_optimizer_steps"),
        "margin_lambda": metrics.get("margin_lambda"),
        "margin": metrics.get("margin"),
        "view_charge_multiplier": manifest.get("view_charge_multiplier"),
        "loss_first_last": [loss[0], loss[-1]] if loss else None,
        "mlm_first_last": [mlm[0], mlm[-1]] if mlm else None,
        "margin_loss_first_last": [margins[0], margins[-1]] if margins else None,
        "margin_loss_mean": mean(margins),
        "margin_loss_population_std": float(statistics.pstdev(margins)) if len(margins) > 1 else 0.0,
        "nll_bad_minus_coh_first_last": [gaps[0], gaps[-1]] if gaps else None,
        "nll_bad_minus_coh_mean": mean(gaps),
        "nll_bad_minus_coh_population_std": float(statistics.pstdev(gaps)) if len(gaps) > 1 else 0.0,
        "nll_bad_minus_coh_min": min(gaps) if gaps else None,
        "nll_bad_minus_coh_max": max(gaps) if gaps else None,
        "nll_bad_minus_coh_positive_fraction_records": mean([1.0 if g > 0 else 0.0 for g in gaps]),
        "nll_gap_slope_per_record_first_to_last": slope(gaps[0], gaps[-1], len(gaps)) if gaps else None,
        "margin_loss_slope_per_record_first_to_last": slope(margins[0], margins[-1], len(margins)) if margins else None,
        "last_10_nll_gap_mean": mean(gaps[-10:]),
        "last_10_margin_loss_mean": mean(margins[-10:]),
        "scientific_reading": "If the margin objective is creating coherent-context preference during the pilot, nll_bad_minus_coh should become positive and margin_loss should fall below softplus(margin). Here values near softplus(0.20)=0.798 indicate little separation.",
    }
    js = _public_path('experiments/archive/frontier_consolidation/data/cohmargin_training_trace_summary/cohmargin_training_trace_summary.json')
    md = _public_path('research/documents/frontier_consolidation/data/cohmargin_training_trace_summary/cohmargin_training_trace_summary.md')
    js.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = [
        "# research coherence-margin training-trace summary",
        "",
        f"Status: **{out['status']}**",
        "",
        f"Run: `{out['run_dir']}`",
        f"Records: {out['n_records']}; optimizer steps: {out['optimizer_steps']}; charged/coherent words: {out['charged_word_exposure']} / {out['coherent_words']}",
        f"Loss first→last: {out['loss_first_last']}",
        f"MLM loss first→last: {out['mlm_first_last']}",
        f"Margin loss first→last: {out['margin_loss_first_last']}; mean {out['margin_loss_mean']}; last10 {out['last_10_margin_loss_mean']}",
        f"NLL bad-minus-coh first→last: {out['nll_bad_minus_coh_first_last']}; mean {out['nll_bad_minus_coh_mean']}; last10 {out['last_10_nll_gap_mean']}; min/max {out['nll_bad_minus_coh_min']} / {out['nll_bad_minus_coh_max']}",
        "",
        out["scientific_reading"],
        "",
        f"JSON: `{rel(js)}`",
    ]
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": out["status"], "out_json": rel(js), "out_md": rel(md), "nll_gap_mean": out["nll_bad_minus_coh_mean"], "margin_loss_mean": out["margin_loss_mean"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
