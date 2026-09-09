#!/usr/bin/env python3
"""research: decompose temperature-source margins into condition-wise contributions.

This postprocesses `temperature_source_readout.py` output without rescoring
models.  It asks whether dense's larger source advantages arise from better
correct-source target support, worse wrong/absent-source target support, candidate
rank/order changes, or a mixture.  This is an explanatory diagnostic for method design,
not official scoring.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import json
import math
import pathlib
import statistics
from typing import Any, Dict, Iterable, List, Optional, Tuple

ROOT = _public_path('.')
STUDY = _public_path('experiments/archive/functional_learning')
DEFAULT_IN = _public_path('experiments/archive/functional_learning/data/temperature_source_readout/temperature_source_readout.json')
DEFAULT_OUT = _public_path('experiments/archive/functional_learning/data/source_margin_decomposition/source_margin_decomposition.json')


def rel(p: pathlib.Path | str) -> str:
    try:
        return str(pathlib.Path(p).resolve().relative_to(ROOT))
    except Exception:
        return str(p)


def finite(xs: Iterable[Any]) -> List[float]:
    out = []
    for x in xs:
        if x is None:
            continue
        try:
            xf = float(x)
        except Exception:
            continue
        if math.isfinite(xf):
            out.append(xf)
    return out


def mean(xs: Iterable[Any]) -> Optional[float]:
    vals = finite(xs)
    return sum(vals) / len(vals) if vals else None


def median(xs: Iterable[Any]) -> Optional[float]:
    vals = finite(xs)
    return statistics.median(vals) if vals else None


def pearson(xs: List[float], ys: List[float]) -> Optional[float]:
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx <= 0 or vy <= 0:
        return None
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / math.sqrt(vx * vy)


def qwen_decomposition(model_summary: Dict[str, Any]) -> Dict[str, Any]:
    q = model_summary["qwen_source"]
    cond = q["by_condition"]
    spec = q["source_specificity"]
    dc = cond["correct_source"].get("mean_delta_nll_T1_vs_parent")
    dw = cond["wrong_source"].get("mean_delta_nll_T1_vs_parent")
    dv = cond["view_only"].get("mean_delta_nll_T1_vs_parent")
    dcf = cond["correct_source"].get("mean_delta_nll_Tfit_vs_parent")
    dwf = cond["wrong_source"].get("mean_delta_nll_Tfit_vs_parent")
    dvf = cond["view_only"].get("mean_delta_nll_Tfit_vs_parent")
    drc = cond["correct_source"].get("mean_delta_rank_vs_parent")
    drw = cond["wrong_source"].get("mean_delta_rank_vs_parent")
    drv = cond["view_only"].get("mean_delta_rank_vs_parent")
    delta_spec = spec.get("mean_delta_specific_advantage_T1_vs_parent")
    delta_spec_tf = spec.get("mean_delta_specific_advantage_Tfit_vs_parent")
    delta_spec_rank = spec.get("mean_delta_specific_rank_advantage_vs_parent")
    def safe_ratio(part, total):
        if part is None or total is None or abs(float(total)) < 1e-12:
            return None
        return float(part) / float(total)
    return {
        "condition_delta_nll_T1": {"correct_source": dc, "wrong_source": dw, "view_only": dv},
        "condition_delta_nll_Tfit": {"correct_source": dcf, "wrong_source": dwf, "view_only": dvf},
        "condition_delta_rank": {"correct_source": drc, "wrong_source": drw, "view_only": drv},
        "delta_specific_advantage_T1": delta_spec,
        "delta_specific_advantage_Tfit": delta_spec_tf,
        "delta_specific_rank_advantage": delta_spec_rank,
        "specific_advantage_contribution_T1": {
            "from_correct_source_improvement_minus_delta_correct": -float(dc) if dc is not None else None,
            "from_wrong_source_degradation_delta_wrong": dw,
            "wrong_degradation_fraction_of_delta_specific": safe_ratio(dw, delta_spec),
            "correct_improvement_fraction_of_delta_specific": safe_ratio(-float(dc), delta_spec) if dc is not None else None,
        },
        "specific_advantage_contribution_Tfit": {
            "from_correct_source_improvement_minus_delta_correct": -float(dcf) if dcf is not None else None,
            "from_wrong_source_degradation_delta_wrong": dwf,
            "wrong_degradation_fraction_of_delta_specific": safe_ratio(dwf, delta_spec_tf),
            "correct_improvement_fraction_of_delta_specific": safe_ratio(-float(dcf), delta_spec_tf) if dcf is not None else None,
        },
        "specific_rank_contribution": {
            "from_correct_source_rank_improvement_minus_delta_correct_rank": -float(drc) if drc is not None else None,
            "from_wrong_source_rank_worsening_delta_wrong_rank": drw,
            "wrong_rank_worsening_fraction_of_delta_specific_rank": safe_ratio(drw, delta_spec_rank),
            "correct_rank_improvement_fraction_of_delta_specific_rank": safe_ratio(-float(drc), delta_spec_rank) if drc is not None else None,
        },
        "triplet_delta_distributions": {
            "n": len(q.get("triplets", [])),
            "share_specific_advantage_delta_positive": mean(1.0 if r.get("delta_specific_advantage_T1_vs_parent", 0) > 0 else 0.0 for r in q.get("triplets", [])),
            "share_specific_rank_delta_positive": mean(1.0 if r.get("delta_specific_rank_advantage_vs_parent", 0) > 0 else 0.0 for r in q.get("triplets", [])),
            "median_delta_specific_advantage_T1": median(r.get("delta_specific_advantage_T1_vs_parent") for r in q.get("triplets", [])),
            "median_delta_specific_rank_advantage": median(r.get("delta_specific_rank_advantage_vs_parent") for r in q.get("triplets", [])),
        },
    }


def common_decomposition(model_summary: Dict[str, Any]) -> Dict[str, Any]:
    c = model_summary["common_source_reversal"]
    out = {
        "condition_delta_margin_T1": {},
        "condition_delta_margin_Tfit": {},
        "condition_delta_rank_advantage": {},
        "source_follow_delta_swing_T1": c["source_follow"].get("mean_delta_source_follow_swing_T1_vs_parent"),
        "source_follow_delta_swing_Tfit": c["source_follow"].get("mean_delta_source_follow_swing_Tfit_vs_parent"),
        "source_follow_delta_rank_swing": c["source_follow"].get("mean_delta_source_follow_rank_swing_vs_parent"),
        "both_correct_T1": c["source_follow"].get("both_correct_T1"),
        "both_rank_better": c["source_follow"].get("both_rank_better"),
    }
    for cond, vals in c["by_condition"].items():
        out["condition_delta_margin_T1"][cond] = vals.get("mean_delta_margin_T1_vs_parent")
        out["condition_delta_margin_Tfit"][cond] = vals.get("mean_delta_margin_Tfit_vs_parent")
        out["condition_delta_rank_advantage"][cond] = vals.get("mean_delta_rank_advantage_vs_parent")
    # Row-level decomposition of which items gain swing and ranks.
    rows = c.get("source_follow_rows", [])
    out["source_follow_row_distributions"] = {
        "n": len(rows),
        "share_swing_T1_delta_positive": mean(1.0 if r.get("delta_source_follow_swing_T1_vs_parent", 0) > 0 else 0.0 for r in rows),
        "share_swing_Tfit_delta_positive": mean(1.0 if r.get("delta_source_follow_swing_Tfit_vs_parent", 0) > 0 else 0.0 for r in rows),
        "share_rank_swing_delta_positive": mean(1.0 if r.get("delta_source_follow_rank_swing_vs_parent", 0) > 0 else 0.0 for r in rows),
        "median_delta_swing_T1": median(r.get("delta_source_follow_swing_T1_vs_parent") for r in rows),
        "median_delta_swing_Tfit": median(r.get("delta_source_follow_swing_Tfit_vs_parent") for r in rows),
        "median_delta_rank_swing": median(r.get("delta_source_follow_rank_swing_vs_parent") for r in rows),
    }
    return out


def seed_agreement(obj: Dict[str, Any], a: str, b: str) -> Dict[str, Any]:
    sa = obj["model_summaries"][a]
    sb = obj["model_summaries"][b]
    qa = {r["base_task_id"]: r for r in sa["qwen_source"].get("triplets", [])}
    qb = {r["base_task_id"]: r for r in sb["qwen_source"].get("triplets", [])}
    common_ids = sorted(set(qa) & set(qb))
    q_metrics = {}
    for metric in ["delta_specific_advantage_T1_vs_parent", "delta_specific_advantage_Tfit_vs_parent", "delta_specific_rank_advantage_vs_parent", "delta_total_rank_advantage_vs_parent"]:
        xs = [float(qa[i].get(metric, 0.0)) for i in common_ids]
        ys = [float(qb[i].get(metric, 0.0)) for i in common_ids]
        q_metrics[metric] = {"n": len(xs), "pearson": pearson(xs, ys), "mean_abs_diff": mean(abs(x-y) for x, y in zip(xs, ys)), "same_sign_fraction": mean(1.0 if (x > 0) == (y > 0) else 0.0 for x, y in zip(xs, ys))}
    ca = {r["task_id"]: r for r in sa["common_source_reversal"].get("source_follow_rows", [])}
    cb = {r["task_id"]: r for r in sb["common_source_reversal"].get("source_follow_rows", [])}
    cids = sorted(set(ca) & set(cb))
    c_metrics = {}
    for metric in ["delta_source_follow_swing_T1_vs_parent", "delta_source_follow_swing_Tfit_vs_parent", "delta_source_follow_rank_swing_vs_parent"]:
        xs = [float(ca[i].get(metric, 0.0)) for i in cids]
        ys = [float(cb[i].get(metric, 0.0)) for i in cids]
        c_metrics[metric] = {"n": len(xs), "pearson": pearson(xs, ys), "mean_abs_diff": mean(abs(x-y) for x, y in zip(xs, ys)), "same_sign_fraction": mean(1.0 if (x > 0) == (y > 0) else 0.0 for x, y in zip(xs, ys))}
    return {"qwen_source_triplet_seed_agreement": q_metrics, "common_source_follow_seed_agreement": c_metrics}


def write_md(path: pathlib.Path, res: Dict[str, Any]) -> None:
    lines = ["# research source-margin decomposition\n\n"]
    lines.append("This decomposes dense and sparse source-readout changes into correct-source support, wrong/absent-source degradation, and rank/order movement. It uses already-scored research rows and is not official scoring.\n\n")
    for name, summ in res["model_decompositions"].items():
        if name == "coherent86":
            continue
        q = summ["qwen_source"]
        c = summ["common_source_reversal"]
        lines.append(f"## {name}\n")
        lines.append(f"- Qwen Δspecific T1={q['delta_specific_advantage_T1']}, Tfit={q['delta_specific_advantage_Tfit']}, Δrank={q['delta_specific_rank_advantage']}\n")
        lines.append(f"- Qwen condition ΔNLL T1: correct={q['condition_delta_nll_T1']['correct_source']}, wrong={q['condition_delta_nll_T1']['wrong_source']}, view-only={q['condition_delta_nll_T1']['view_only']}\n")
        lines.append(f"- Qwen condition Δrank: correct={q['condition_delta_rank']['correct_source']}, wrong={q['condition_delta_rank']['wrong_source']}, view-only={q['condition_delta_rank']['view_only']}\n")
        lines.append(f"- Common swing ΔT1={c['source_follow_delta_swing_T1']}, ΔTfit={c['source_follow_delta_swing_Tfit']}, Δrank={c['source_follow_delta_rank_swing']}; both-correct={c['both_correct_T1']}/25, both-rank={c['both_rank_better']}/25\n")
        lines.append(f"- Common condition Δmargin T1: {c['condition_delta_margin_T1']}\n\n")
    lines.append("## Seed agreement\n")
    lines.append(json.dumps(res.get("dense_seed_agreement", {}), indent=2) + "\n")
    lines.append("\n## Interpretation\n")
    for k, v in res.get("interpretation", {}).items():
        lines.append(f"- **{k}**: {v}\n")
    path.write_text("".join(lines), encoding="utf-8")


def main() -> None:
    in_path = DEFAULT_IN
    out_path = DEFAULT_OUT
    obj = json.loads(in_path.read_text(encoding="utf-8"))
    decomps: Dict[str, Any] = {}
    for name, summ in obj["model_summaries"].items():
        decomps[name] = {"qwen_source": qwen_decomposition(summ), "common_source_reversal": common_decomposition(summ)}
    res = {
        "status": "SOURCE_MARGIN_DECOMPOSITION_DONE",
        "input": rel(in_path),
        "model_decompositions": decomps,
        "dense_seed_agreement": seed_agreement(obj, "dense_focus_seed62064", "dense_focus_seed62065"),
        "interpretation": {
            "qwen_dense_mechanism": "Dense's Qwen source-specific margin is not a pure correct-source likelihood gain. In both dense seeds, correct-source NLL improves slightly while wrong-source and view-only NLL/ranks worsen substantially; the correct-vs-wrong advantage therefore reflects increased penalty for wrong evidence as well as small correct-evidence support.",
            "rank_order_component": "Dense changes ranks/order on source probes, so the effect is not reducible to a global positive temperature scale. But rank advantage also increases mainly because wrong-source target ranks worsen more than correct-source ranks, not because every correct-source target rank improves.",
            "method_design": "The next method should preserve rank-changing evidence discrimination while reducing broad rank/NLL degradation on CDI, grammar, and supervised transfer. Dense-mask/sparse-label should therefore be evaluated on condition-wise and rank-wise decomposition, not only aggregate margins.",
        },
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(res, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_md(out_path.with_suffix(".md"), res)
    print(json.dumps({"status": res["status"], "out_json": rel(out_path), "out_md": rel(out_path.with_suffix('.md'))}, indent=2), flush=True)


if __name__ == "__main__":
    main()
