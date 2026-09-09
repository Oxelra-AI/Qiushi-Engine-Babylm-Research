#!/usr/bin/env python3
"""research: row-level association between overwrite residuals and update-sensitive EWoK rows.

Uses existing EWoK row records and research synthetic metrics.  For each official
update-sensitive EWoK row, correlate model-level overwrite metrics with that row's
per-model correctness, stable-failure flag, and interaction margin.  This is still
no-training evidence and does not touch endpoint artifacts.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import csv
import json
import math
import statistics
import time
from pathlib import Path
from typing import Any

USER_ROOT = _public_path('.')
A01_WS = _public_path('experiments/archive/representation_and_objectives')
OUT_DIR = _public_path('experiments/archive/representation_and_objectives/data/official_row_association')
OUT_JSON = _public_path('experiments/archive/representation_and_objectives/data/official_row_association/official_row_association.json')
OUT_MD = _public_path('research/notes/representation_and_objectives/official_row_association.md')
SUMMARY = _public_path('experiments/archive/representation_and_objectives/data/overwrite_decomposition/overwrite_decomposition_summary.json')
RECORDS = _public_path('experiments/archive/representation_and_objectives/data/full_ewok_interaction_specificity/ewok_full_interaction_specificity_records_fullcpu.csv')

ROW_RECORDS = {
    "legal16k_base_100M_seed43022": _public_path('experiments/archive/representation_and_objectives/data/scale1p75_matched_ewok_interaction/legal16k_100M/ewok_interaction_records.csv'),
    "scale1p75_100M": _public_path('experiments/archive/representation_and_objectives/data/scale1p75_matched_ewok_interaction/scale1p75_100M/ewok_interaction_records.csv'),
    "fw_compact_100M_seed43022": _public_path('experiments/archive/representation_and_objectives/data/fw_ewok_interaction_reader/fw_compact_fullbatch_seed43022/ewok_interaction_records.csv'),
    "fw_rowblock_100M_seed43022": _public_path('experiments/archive/representation_and_objectives/data/fw_ewok_interaction_reader/fw_breadth_rowblock_fullbatch_seed43022/ewok_interaction_records.csv'),
    "mlm_only_20M": _public_path('experiments/archive/representation_and_objectives/data/full_ewok_coupled_turnover/targets/mlm_only_20M/ewok_interaction_records.csv'),
    "coupled_shuffled_20M": _public_path('experiments/archive/representation_and_objectives/data/full_ewok_coupled_turnover/targets/coupled_shuffled_20M/ewok_interaction_records.csv'),
}
LEGAL40_FROM_STEP092 = {
    "legal40k_8x480_100M_seed43022": "legal40_8x480_43022",
    "legal40k_depth12_100M_seed43022": "legal40_depth_12x384_43022",
}
UPDATE_DOMAINS = {"material-dynamics", "physical-dynamics"}


def fnum(x: Any) -> float | None:
    try:
        v = float(x)
        return v if math.isfinite(v) else None
    except Exception:
        return None


def pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 4:
        return None
    mx, my = statistics.fmean(xs), statistics.fmean(ys)
    sx = math.sqrt(sum((x - mx) ** 2 for x in xs) / (len(xs) - 1))
    sy = math.sqrt(sum((y - my) ** 2 for y in ys) / (len(ys) - 1))
    if sx < 1e-12 or sy < 1e-12:
        return None
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / ((len(xs) - 1) * sx * sy)


def ranks(vs: list[float]) -> list[float]:
    idx = sorted(range(len(vs)), key=lambda i: vs[i])
    out = [0.0] * len(vs)
    i = 0
    while i < len(vs):
        j = i + 1
        while j < len(vs) and vs[idx[j]] == vs[idx[i]]:
            j += 1
        r = (i + j - 1) / 2.0
        for k in range(i, j):
            out[idx[k]] = r
        i = j
    return out


def spearman(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 4:
        return None
    return pearson(ranks(xs), ranks(ys))


def qstats(vals: list[float]) -> dict[str, Any]:
    xs = sorted(v for v in vals if math.isfinite(v))
    if not xs:
        return {"n": 0}
    def q(p: float) -> float:
        if len(xs) == 1:
            return xs[0]
        pos = p * (len(xs) - 1)
        lo, hi = math.floor(pos), math.ceil(pos)
        if lo == hi:
            return xs[lo]
        return xs[lo] * (hi - pos) + xs[hi] * (pos - lo)
    return {"n": len(xs), "min": xs[0], "p05": q(0.05), "p25": q(0.25), "mean": statistics.fmean(xs), "median": statistics.median(xs), "p75": q(0.75), "p95": q(0.95), "max": xs[-1]}


def bool01(x: Any) -> float | None:
    if x in (True, "True", "true", "1", 1):
        return 1.0
    if x in (False, "False", "false", "0", 0):
        return 0.0
    return None


def rel(path: Path) -> str:
    return str(path.resolve().relative_to(USER_ROOT))


def synthetic_metrics() -> dict[str, dict[str, float]]:
    obj = json.loads(SUMMARY.read_text(encoding="utf-8"))
    out: dict[str, dict[str, float]] = {}
    for s in obj["targets"]:
        t = s["target"]
        byc = s["by_condition"]
        dec = s["decomposition"]["by_label"]
        out[t] = {
            "direct_nonadditive_median": dec.get("direct_adjacent", {}).get("residual_mean", {}).get("median"),
            "targetfree_nonadditive_median": dec.get("targetfree_adjacent", {}).get("residual_mean", {}).get("median"),
            "action_crossed": byc.get("action_only", {}).get("crossed"),
            "targetfree_action_crossed": byc.get("tf_action_only", {}).get("crossed"),
            "direct_negative_nonadditive_frac": dec.get("direct_adjacent", {}).get("negative_nonadditive_frac"),
            "targetfree_negative_nonadditive_frac": dec.get("targetfree_adjacent", {}).get("negative_nonadditive_frac"),
        }
    return out


def read_standard_rows(target: str, path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for r in reader:
            if r.get("domain") not in UPDATE_DOMAINS:
                continue
            rows.append({
                "target": target,
                "global_index": int(float(r["global_index"])),
                "domain": r.get("domain"),
                "correct": bool01(r.get("saved_model_correct_flag")),
                "stable_failure": bool01(r.get("conditional_reversal_failure_stable")),
                "interaction_sum": fnum(r.get("interaction_sum")),
            })
    return rows


def read_legal40_rows() -> list[dict[str, Any]]:
    out = []
    with RECORDS.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        inv = {v: k for k, v in LEGAL40_FROM_STEP092.items()}
        for r in reader:
            if r.get("domain") not in UPDATE_DOMAINS:
                continue
            model = r.get("model")
            if model not in inv:
                continue
            out.append({
                "target": inv[model],
                "global_index": int(float(r["global_index"])),
                "domain": r.get("domain"),
                "correct": bool01(r.get("saved_model_correct_flag")),
                "stable_failure": bool01(r.get("conditional_reversal_failure_stable")),
                "interaction_sum": fnum(r.get("interaction_sum")),
            })
    return out


def load_rows() -> dict[int, dict[str, dict[str, Any]]]:
    by_row: dict[int, dict[str, dict[str, Any]]] = {}
    for target, path in ROW_RECORDS.items():
        for r in read_standard_rows(target, path):
            by_row.setdefault(r["global_index"], {})[target] = r
    for r in read_legal40_rows():
        by_row.setdefault(r["global_index"], {})[r["target"]] = r
    return by_row


def row_level_association(by_row: dict[int, dict[str, dict[str, Any]]], syn: dict[str, dict[str, float]]) -> dict[str, Any]:
    metrics = [
        "direct_nonadditive_median",
        "targetfree_nonadditive_median",
        "action_crossed",
        "targetfree_action_crossed",
        "direct_negative_nonadditive_frac",
        "targetfree_negative_nonadditive_frac",
    ]
    outcomes = ["correct", "stable_failure", "interaction_sum"]
    per_metric: dict[str, dict[str, list[float]]] = {m: {o: [] for o in outcomes} for m in metrics}
    row_records = []
    for gi, model_rows in sorted(by_row.items()):
        domain = next(iter(model_rows.values()))["domain"] if model_rows else None
        row_entry = {"global_index": gi, "domain": domain, "n_models": 0, "correlations": {}}
        for m in metrics:
            for o in outcomes:
                xs, ys, models = [], [], []
                for target, rr in model_rows.items():
                    x = syn.get(target, {}).get(m)
                    y = rr.get(o)
                    if x is None or y is None:
                        continue
                    if not (math.isfinite(float(x)) and math.isfinite(float(y))):
                        continue
                    xs.append(float(x)); ys.append(float(y)); models.append(target)
                row_entry["n_models"] = max(row_entry["n_models"], len(xs))
                pr = pearson(xs, ys)
                sr = spearman(xs, ys)
                key = f"{m}__{o}"
                row_entry["correlations"][key] = {"n": len(xs), "pearson": pr, "spearman": sr}
                if pr is not None:
                    per_metric[m][o].append(pr)
        row_records.append(row_entry)
    summary = {"n_rows": len(row_records), "by_metric_outcome": {}}
    for m in metrics:
        for o in outcomes:
            vals = per_metric[m][o]
            summary["by_metric_outcome"][f"{m}__{o}"] = {
                "pearson_distribution": qstats(vals),
                "frac_positive": (sum(1 for v in vals if v > 0) / len(vals)) if vals else None,
                "frac_negative": (sum(1 for v in vals if v < 0) / len(vals)) if vals else None,
                "n_rows_with_defined_r": len(vals),
            }
    # Also keep compact row file for later inspection.
    row_path = _public_path('experiments/archive/representation_and_objectives/data/official_row_association/row_level_associations.jsonl')
    with row_path.open("w", encoding="utf-8") as fh:
        for r in row_records:
            fh.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")
    summary["row_records_path"] = rel(row_path)
    return summary


def stacked_association(by_row: dict[int, dict[str, dict[str, Any]]], syn: dict[str, dict[str, float]]) -> dict[str, Any]:
    # Model × official-row points, useful as a weighted view but not independent rows.
    metrics = ["direct_nonadditive_median", "targetfree_nonadditive_median", "action_crossed", "targetfree_action_crossed"]
    outcomes = ["correct", "stable_failure", "interaction_sum"]
    out = {}
    for m in metrics:
        for o in outcomes:
            xs, ys = [], []
            for model_rows in by_row.values():
                for target, rr in model_rows.items():
                    x = syn.get(target, {}).get(m)
                    y = rr.get(o)
                    if x is None or y is None:
                        continue
                    if math.isfinite(float(x)) and math.isfinite(float(y)):
                        xs.append(float(x)); ys.append(float(y))
            out[f"{m}__{o}"] = {"n": len(xs), "pearson": pearson(xs, ys), "spearman": spearman(xs, ys)}
    return out


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    syn = synthetic_metrics()
    rows = load_rows()
    row_assoc = row_level_association(rows, syn)
    stack_assoc = stacked_association(rows, syn)
    out = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "status": "ROW_ASSOCIATION_DONE",
        "update_domains": sorted(UPDATE_DOMAINS),
        "synthetic_metrics_by_target": syn,
        "n_update_sensitive_global_rows": len(rows),
        "targets_with_rows": sorted({t for row in rows.values() for t in row}),
        "row_level": row_assoc,
        "stacked_model_row": stack_assoc,
        "source_summary": rel(SUMMARY),
    }
    OUT_JSON.write_text(json.dumps(out, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")

    def fmt(v: Any) -> str:
        return "n/a" if v is None or (isinstance(v, float) and not math.isfinite(v)) else f"{float(v):.4f}"
    lines = [
        "# research official row association",
        "",
        "Status: **ROW_LEVEL_AGGREGATED** from existing EWoK records; no new inference.",
        "",
        f"Update-sensitive EWoK rows: {len(rows)} global rows from domains {', '.join(sorted(UPDATE_DOMAINS))}.",
        "",
        "## Row-wise correlation distributions",
        "",
        "Each official row is treated separately; within that row, the association is across models. A strong synthetic route signal would show a consistent sign over many official update rows, not only a model-average association.",
        "",
        "| synthetic metric vs row outcome | rows with r | median r | mean r | frac positive | frac negative |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for k, v in sorted(row_assoc["by_metric_outcome"].items()):
        dist = v["pearson_distribution"]
        lines.append(f"| {k} | {v['n_rows_with_defined_r']} | {fmt(dist.get('median'))} | {fmt(dist.get('mean'))} | {fmt(v['frac_positive'])} | {fmt(v['frac_negative'])} |")
    lines.extend(["", "## Stacked model×row association", "", "| synthetic metric vs row outcome | n | Pearson | Spearman |", "|---|---:|---:|---:|"])
    for k, v in sorted(stack_assoc.items()):
        lines.append(f"| {k} | {v['n']} | {fmt(v['pearson'])} | {fmt(v['spearman'])} |")
    lines.extend(["", f"JSON: `{rel(OUT_JSON)}`", f"Row records: `{row_assoc['row_records_path']}`"])
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": "ROW_ASSOCIATION_DONE", "out_json": rel(OUT_JSON), "out_md": rel(OUT_MD)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
