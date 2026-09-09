#!/usr/bin/env python3
"""research: endpoint-only AoA surprisal profile for coherent86 and dense endpoints.

This is not an AoA measurement because it lacks the full early-stop trajectory.  It is
an endpoint diagnostic over the official CDI target/context set: how the final
candidate endpoints shift masked-token surprisal relative to coherent86, and whether
the two dense seeds agree.  The profile helps interpret later measured AoA and guards
against treating a scalar AoA value as unexplained.
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
import time
from collections import defaultdict
from typing import Any

ROOT = _public_path('.')
STUDY = _public_path('experiments/archive/functional_learning')
DEFAULT_OUT = _public_path('experiments/archive/functional_learning/data/aoa_endpoint_surprisal_profile')
CDI_HUMAN = _public_path('experiments/archive/initial_model_studies/repos/babylm-eval/strict/evaluation_data/full_eval/aoa/cdi_human.csv')
ENDPOINT_SURP = {
    "coherent86": _public_path('experiments/archive/functional_learning/data/batched_aoa_full_endpoints/endpoints/coherent86/full/surprisal.json'),
    "dense_seed62064": _public_path('experiments/archive/functional_learning/data/batched_aoa_full_endpoints/endpoints/dense_seed62064/full/surprisal.json'),
    "dense_seed62065": _public_path('experiments/archive/functional_learning/data/batched_aoa_full_endpoints/endpoints/dense_seed62065/full/surprisal.json'),
}
ENDPOINT_MANIFEST = {k: p.parent / "manifest.json" for k, p in ENDPOINT_SURP.items()}


def rel(path: pathlib.Path | str) -> str:
    try:
        return str(pathlib.Path(path).resolve().relative_to(ROOT))
    except Exception:
        return str(path)


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def load_json(path: pathlib.Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def mean(xs: list[float]) -> float | None:
    vals = [x for x in xs if math.isfinite(x)]
    return sum(vals) / len(vals) if vals else None


def se(xs: list[float]) -> float | None:
    vals = [x for x in xs if math.isfinite(x)]
    if len(vals) <= 1:
        return None
    return statistics.stdev(vals) / math.sqrt(len(vals))


def read_cdi() -> dict[str, dict[str, float | None]]:
    out: dict[str, dict[str, float | None]] = {}
    with CDI_HUMAN.open(newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        months = [k for k in r.fieldnames or [] if k.isdigit()]
        for row in r:
            vals = []
            for m in months:
                try:
                    v = float(row[m])
                except Exception:
                    v = float("nan")
                if math.isfinite(v):
                    vals.append(v)
            # Approximate normative maturity descriptors; not official AoA.
            final = vals[-1] if vals else float("nan")
            auc = sum(vals) / len(vals) if vals else float("nan")
            first50 = None
            for m, v in zip(months, vals, strict=False):
                if v >= 0.5:
                    first50 = float(m)
                    break
            out[row["word"]] = {"cdi_final_30m": final, "cdi_mean_16_30m": auc, "cdi_first_month_ge_0p5": first50}
    return out


def load_endpoint(path: pathlib.Path) -> dict[tuple[str, int, str], float]:
    data = load_json(path)
    rows = data.get("results", [])
    out = {}
    for r in rows:
        try:
            v = float(r.get("surprisal"))
        except Exception:
            continue
        out[(str(r.get("target_word")), int(r.get("context_id")), str(r.get("context")))] = v
    return out


def pearson(xs: list[float], ys: list[float]) -> float | None:
    pairs = [(float(x), float(y)) for x, y in zip(xs, ys, strict=False) if math.isfinite(float(x)) and math.isfinite(float(y))]
    if len(pairs) < 3:
        return None
    x = [p[0] for p in pairs]
    y = [p[1] for p in pairs]
    mx = sum(x) / len(x)
    my = sum(y) / len(y)
    vx = sum((a - mx) ** 2 for a in x)
    vy = sum((b - my) ** 2 for b in y)
    if vx <= 0 or vy <= 0:
        return None
    return sum((a - mx) * (b - my) for a, b in pairs) / math.sqrt(vx * vy)


def summarize_deltas(rows: list[dict[str, Any]], field: str = "delta") -> dict[str, Any]:
    vals = [float(r[field]) for r in rows if math.isfinite(float(r[field]))]
    vals_sorted = sorted(vals)
    def q(p: float) -> float | None:
        if not vals_sorted:
            return None
        idx = min(len(vals_sorted)-1, max(0, int(round(p*(len(vals_sorted)-1)))))
        return vals_sorted[idx]
    return {
        "n": len(vals),
        "mean": mean(vals),
        "se": se(vals),
        "median": q(0.5),
        "q10": q(0.1),
        "q90": q(0.9),
        "n_improved_lower_surprisal": sum(1 for v in vals if v < 0),
        "n_worse_higher_surprisal": sum(1 for v in vals if v > 0),
        "fraction_improved": sum(1 for v in vals if v < 0) / len(vals) if vals else None,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", type=pathlib.Path, default=DEFAULT_OUT)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    cdi = read_cdi()
    maps = {k: load_endpoint(p) for k, p in ENDPOINT_SURP.items()}
    common = sorted(set.intersection(*(set(m.keys()) for m in maps.values())), key=lambda x: (x[0], x[1], x[2]))
    context_rows: list[dict[str, Any]] = []
    word_acc: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for key in common:
        word, cid, context = key
        c = maps["coherent86"][key]
        d64 = maps["dense_seed62064"][key]
        d65 = maps["dense_seed62065"][key]
        rec = {
            "target_word": word,
            "context_id": cid,
            "context": context,
            "coherent86": c,
            "dense_seed62064": d64,
            "dense_seed62065": d65,
            "d64_minus_coherent": d64 - c,
            "d65_minus_coherent": d65 - c,
            "d65_minus_d64": d65 - d64,
        }
        rec.update(cdi.get(word, {}))
        context_rows.append(rec)
        for k2 in ["coherent86", "dense_seed62064", "dense_seed62065", "d64_minus_coherent", "d65_minus_coherent", "d65_minus_d64"]:
            word_acc[word][k2].append(float(rec[k2]))
    word_rows: list[dict[str, Any]] = []
    for word, vals in word_acc.items():
        rec = {"target_word": word, "n_contexts": len(vals["coherent86"])}
        for k in vals:
            rec[k] = mean(vals[k])
        rec.update(cdi.get(word, {}))
        word_rows.append(rec)
    word_rows.sort(key=lambda r: str(r["target_word"]))
    def top(rows: list[dict[str, Any]], key: str, n: int = 12, reverse: bool = False) -> list[dict[str, Any]]:
        return sorted(rows, key=lambda r: float(r.get(key, 0.0)), reverse=reverse)[:n]
    d64 = [r["d64_minus_coherent"] for r in context_rows]
    d65 = [r["d65_minus_coherent"] for r in context_rows]
    wd64 = [r["d64_minus_coherent"] for r in word_rows]
    wd65 = [r["d65_minus_coherent"] for r in word_rows]
    cdi_mean = [r.get("cdi_mean_16_30m", float("nan")) for r in word_rows]
    cdi_final = [r.get("cdi_final_30m", float("nan")) for r in word_rows]
    report = {
        "status": "AOA_ENDPOINT_SURPRISAL_PROFILE_DONE",
        "created_utc": now(),
        "script": rel(_public_path('experiments/archive/functional_learning/scripts/aoa_endpoint_surprisal_profile.py')),
        "note": "Endpoint-only CDI surprisal profile. This is not AoA: measured AoA requires full shared-ancestry plus endpoint trajectory and platform curve scoring.",
        "inputs": {k: {"surprisal": rel(v), "manifest": rel(ENDPOINT_MANIFEST[k]), "manifest_status": (load_json(ENDPOINT_MANIFEST[k]).get("status") if ENDPOINT_MANIFEST[k].is_file() else None)} for k, v in ENDPOINT_SURP.items()},
        "n_common_context_rows": len(common),
        "n_words": len(word_rows),
        "context_delta_summary": {
            "dense62064_minus_coherent": summarize_deltas([{**r, "delta": r["d64_minus_coherent"]} for r in context_rows]),
            "dense62065_minus_coherent": summarize_deltas([{**r, "delta": r["d65_minus_coherent"]} for r in context_rows]),
            "dense62065_minus_dense62064": summarize_deltas([{**r, "delta": r["d65_minus_d64"]} for r in context_rows]),
        },
        "word_mean_delta_summary": {
            "dense62064_minus_coherent": summarize_deltas([{**r, "delta": r["d64_minus_coherent"]} for r in word_rows]),
            "dense62065_minus_coherent": summarize_deltas([{**r, "delta": r["d65_minus_coherent"]} for r in word_rows]),
            "dense62065_minus_dense62064": summarize_deltas([{**r, "delta": r["d65_minus_d64"]} for r in word_rows]),
        },
        "seed_agreement": {
            "context_delta_pearson": pearson(d64, d65),
            "word_mean_delta_pearson": pearson(wd64, wd65),
            "context_same_sign_fraction_excluding_zero": sum(1 for a,b in zip(d64,d65, strict=False) if a*b>0)/sum(1 for a,b in zip(d64,d65, strict=False) if a!=0 or b!=0),
            "word_same_sign_fraction_excluding_zero": sum(1 for a,b in zip(wd64,wd65, strict=False) if a*b>0)/sum(1 for a,b in zip(wd64,wd65, strict=False) if a!=0 or b!=0),
        },
        "cdi_relations_word_level": {
            "corr_d64_delta_with_cdi_mean_16_30m": pearson(wd64, cdi_mean),
            "corr_d65_delta_with_cdi_mean_16_30m": pearson(wd65, cdi_mean),
            "corr_d64_delta_with_cdi_final_30m": pearson(wd64, cdi_final),
            "corr_d65_delta_with_cdi_final_30m": pearson(wd65, cdi_final),
        },
        "top_word_improvements_lower_surprisal_d64": top(word_rows, "d64_minus_coherent", 15, reverse=False),
        "top_word_worsenings_higher_surprisal_d64": top(word_rows, "d64_minus_coherent", 15, reverse=True),
        "top_word_improvements_lower_surprisal_d65": top(word_rows, "d65_minus_coherent", 15, reverse=False),
        "top_word_worsenings_higher_surprisal_d65": top(word_rows, "d65_minus_coherent", 15, reverse=True),
    }
    # Write compact CSVs for later detailed inspection.
    def write_csv(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
        if not rows:
            path.write_text("\n", encoding="utf-8")
            return
        fields: list[str] = []
        seen = set()
        for r in rows:
            for k in r:
                if k not in seen:
                    fields.append(k); seen.add(k)
        with path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader(); w.writerows(rows)
    write_csv(args.out_dir / "endpoint_context_deltas.csv", context_rows)
    write_csv(args.out_dir / "endpoint_word_mean_deltas.csv", word_rows)
    out_json = args.out_dir / "aoa_endpoint_surprisal_profile.json"
    out_md = args.out_dir / "aoa_endpoint_surprisal_profile.md"
    out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    lines = ["# research AoA endpoint surprisal profile\n\n", report["note"] + "\n\n"]
    lines.append(f"Common context rows: `{len(common)}`; words: `{len(word_rows)}`.\n\n")
    lines.append("## Endpoint delta summaries (positive = higher/worse surprisal than coherent86)\n\n")
    for scope in ["context_delta_summary", "word_mean_delta_summary"]:
        lines.append(f"### {scope}\n\n")
        for k, v in report[scope].items():
            lines.append(f"- {k}: mean `{v.get('mean')}`, median `{v.get('median')}`, SE `{v.get('se')}`, improved fraction `{v.get('fraction_improved')}`.\n")
        lines.append("\n")
    lines.append("## Dense seed agreement\n\n")
    lines.append(json.dumps(report["seed_agreement"], indent=2) + "\n\n")
    lines.append("## CDI word-level correlations\n\n")
    lines.append(json.dumps(report["cdi_relations_word_level"], indent=2) + "\n")
    out_md.write_text("".join(lines), encoding="utf-8")
    print(json.dumps({"status": report["status"], "out_json": rel(out_json), "out_md": rel(out_md), "n_context_rows": len(common), "n_words": len(word_rows)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
