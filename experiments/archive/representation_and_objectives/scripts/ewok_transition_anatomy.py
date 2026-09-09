#!/usr/bin/env python3
"""research: paired EWoK row-transition anatomy for scale1.75.

Reads the research EWoK inference-ablation row-level records and compares shared
EWoK items row-by-row across scale1.75 live/disabled/rescaled variants and the
matched research legal16k base. This is evaluation readout only; it must not be
used as a training source. The goal is to decide whether scale1.75's EWoK loss is
localized in identifiable item classes or a distributed trajectory shift.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import csv
import json
import math
import pathlib
import statistics
import time
from collections import Counter, defaultdict
from typing import Any


def find_user_root() -> pathlib.Path:
    return _PUBLIC_ROOT

ROOT = find_user_root()
BASE_DIR = ROOT / "experiments/archive/representation_and_objectives/data/scale1p75_ewok_inference_ablation/readouts"
OUT = ROOT / "experiments/archive/representation_and_objectives/data/ewok_transition_anatomy"
NOTE = ROOT / "research/notes/representation_and_objectives/ewok_transition_anatomy.md"

VARIANTS = [
    "matched_legal16k_80M",
    "scale1p75_disabled",
    "scale1p75_scale0p5",
    "scale1p75_scale1p0",
    "scale1p75_live",
    "scale1p75_scale2p5",
]
COMPARES = [
    ("matched_legal16k_80M", "scale1p75_live", "live_minus_matched"),
    ("matched_legal16k_80M", "scale1p75_disabled", "disabled_minus_matched"),
    ("scale1p75_disabled", "scale1p75_live", "live_minus_disabled"),
    ("scale1p75_live", "scale1p75_scale0p5", "scale0p5_minus_live"),
    ("scale1p75_live", "scale1p75_scale2p5", "scale2p5_minus_live"),
]
KEY_COLS = [
    "global_index", "domain", "local_index", "ContextType", "ContextDiff", "TargetDiff", "ConceptA", "ConceptB",
    "context_diff_c1_texts_joined", "context_diff_c2_texts_joined", "target1", "target2",
]
NUM_COLS = [
    "official_margin_t1_sum", "official_margin_t2_sum", "within_context_margin_c1_sum", "within_context_margin_c2_sum",
    "interaction_sum", "interaction_mean", "interaction_minus_deletion_sum",
]
BOOL_COLS = [
    "saved_model_correct_flag", "saved_model_wrong_flag", "both_official_sum_positive", "both_within_context_sum_positive",
    "both_within_context_mean_positive", "stable_nonpositive_interaction", "conditional_reversal_failure_stable",
    "local_both_actual_over_swapped_positive",
]


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def rel(p: pathlib.Path | str) -> str:
    try:
        return str(pathlib.Path(p).resolve().relative_to(ROOT))
    except Exception:
        return str(p)


def fnum(x: Any) -> float:
    try:
        v = float(x)
        if math.isfinite(v):
            return v
    except Exception:
        pass
    return float("nan")


def bval(x: Any) -> bool:
    return str(x).strip().lower() in {"true", "1", "yes"}


def qstats(vals: list[float]) -> dict[str, Any]:
    xs = sorted(v for v in vals if math.isfinite(v))
    if not xs:
        return {"n": 0}
    def q(p: float) -> float:
        if len(xs) == 1:
            return xs[0]
        return xs[int(round(p * (len(xs)-1)))]
    return {"n": len(xs), "mean": statistics.fmean(xs), "median": statistics.median(xs), "p10": q(0.10), "p90": q(0.90), "min": xs[0], "max": xs[-1]}


def read_variant(name: str) -> dict[int, dict[str, Any]]:
    path = BASE_DIR / name / "ewok_interaction_records.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    rows: dict[int, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            gi = int(r["global_index"])
            slim: dict[str, Any] = {k: r.get(k, "") for k in KEY_COLS}
            for k in NUM_COLS:
                slim[k] = fnum(r.get(k))
            for k in BOOL_COLS:
                slim[k] = bval(r.get(k))
            rows[gi] = slim
    return rows


def base_meta(row: dict[str, Any]) -> dict[str, Any]:
    return {k: row.get(k, "") for k in KEY_COLS}


def write_csv(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: list[str] = []
    seen = set()
    for r in rows:
        for k in r.keys():
            if k not in seen:
                seen.add(k); keys.append(k)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def compare(a_name: str, b_name: str, label: str, data: dict[str, dict[int, dict[str, Any]]]) -> dict[str, Any]:
    a = data[a_name]; b = data[b_name]
    common = sorted(set(a) & set(b))
    rows = []
    counts = Counter()
    by_domain: dict[str, Counter] = defaultdict(Counter)
    by_context: dict[str, Counter] = defaultdict(Counter)
    deltas: dict[str, list[float]] = defaultdict(list)
    for gi in common:
        ar = a[gi]; br = b[gi]
        ac = ar["saved_model_correct_flag"]; bc = br["saved_model_correct_flag"]
        ast = ar["conditional_reversal_failure_stable"]; bst = br["conditional_reversal_failure_stable"]
        key = ("aC" if ac else "aW") + "_to_" + ("bC" if bc else "bW")
        counts[key] += 1
        skey = ("aStable" if ast else "aNotStable") + "_to_" + ("bStable" if bst else "bNotStable")
        counts[skey] += 1
        if ac and not bc:
            counts["lost_correct"] += 1
        if (not ac) and bc:
            counts["gained_correct"] += 1
        if (not ast) and bst:
            counts["added_stable_failure"] += 1
        if ast and (not bst):
            counts["removed_stable_failure"] += 1
        if ac and (not bc) and bst:
            counts["lost_correct_to_stable_failure"] += 1
        if ast and bc:
            counts["stable_failure_to_correct"] += 1
        dom = str(ar.get("domain", ""))
        ctx = str(ar.get("ContextDiff", ""))
        for ckey in [key, skey]:
            by_domain[dom][ckey] += 1
            by_context[ctx][ckey] += 1
        by_domain[dom]["n"] += 1; by_context[ctx]["n"] += 1
        if ac and not bc: by_domain[dom]["lost_correct"] += 1; by_context[ctx]["lost_correct"] += 1
        if (not ac) and bc: by_domain[dom]["gained_correct"] += 1; by_context[ctx]["gained_correct"] += 1
        if (not ast) and bst: by_domain[dom]["added_stable_failure"] += 1; by_context[ctx]["added_stable_failure"] += 1
        if ast and (not bst): by_domain[dom]["removed_stable_failure"] += 1; by_context[ctx]["removed_stable_failure"] += 1
        rec = base_meta(ar)
        for col in NUM_COLS:
            av = fnum(ar[col]); bv = fnum(br[col]); dv = bv - av
            rec[f"a_{col}"] = av; rec[f"b_{col}"] = bv; rec[f"delta_{col}"] = dv
            deltas[col].append(dv)
        for col in BOOL_COLS:
            rec[f"a_{col}"] = ar[col]; rec[f"b_{col}"] = br[col]
        rec["transition_correct"] = key
        rec["transition_stable"] = skey
        rec["lost_correct"] = ac and not bc
        rec["gained_correct"] = (not ac) and bc
        rec["added_stable_failure"] = (not ast) and bst
        rec["removed_stable_failure"] = ast and (not bst)
        rows.append(rec)
    out_dir = OUT / label
    out_dir.mkdir(parents=True, exist_ok=True)
    # Most relevant row files.
    write_csv(out_dir / "lost_correct_rows.csv", [r for r in rows if r["lost_correct"]])
    write_csv(out_dir / "gained_correct_rows.csv", [r for r in rows if r["gained_correct"]])
    write_csv(out_dir / "added_stable_failure_rows.csv", [r for r in rows if r["added_stable_failure"]])
    write_csv(out_dir / "removed_stable_failure_rows.csv", [r for r in rows if r["removed_stable_failure"]])
    # Full transition file is useful but large; still small enough compared with inputs.
    write_csv(out_dir / "all_transition_rows.csv", rows)
    def group_table(groups: dict[str, Counter], field: str) -> list[dict[str, Any]]:
        out = []
        for k, c in groups.items():
            n = c.get("n", 0)
            out.append({
                field: k, "n": n,
                "lost_correct": c.get("lost_correct", 0),
                "gained_correct": c.get("gained_correct", 0),
                "net_correct_gain_b_minus_a": c.get("gained_correct", 0) - c.get("lost_correct", 0),
                "added_stable_failure": c.get("added_stable_failure", 0),
                "removed_stable_failure": c.get("removed_stable_failure", 0),
                "net_stable_failure_b_minus_a": c.get("added_stable_failure", 0) - c.get("removed_stable_failure", 0),
                "lost_frac": c.get("lost_correct", 0) / n if n else None,
                "gain_frac": c.get("gained_correct", 0) / n if n else None,
            })
        return sorted(out, key=lambda r: (-abs(r["net_stable_failure_b_minus_a"]), -abs(r["net_correct_gain_b_minus_a"]), str(r[field])))
    dom_rows = group_table(by_domain, "domain")
    ctx_rows = group_table(by_context, "ContextDiff")
    write_csv(out_dir / "by_domain_transition.csv", dom_rows)
    write_csv(out_dir / "by_contextdiff_transition.csv", ctx_rows)
    summary = {
        "label": label,
        "a": a_name,
        "b": b_name,
        "n_common": len(common),
        "counts": dict(counts),
        "correct_accuracy_a": counts.get("aC_to_bC", 0) / len(common) + counts.get("aC_to_bW", 0) / len(common) if common else None,
        "correct_accuracy_b": counts.get("aC_to_bC", 0) / len(common) + counts.get("aW_to_bC", 0) / len(common) if common else None,
        "net_correct_b_minus_a": counts.get("gained_correct", 0) - counts.get("lost_correct", 0),
        "net_stable_failure_b_minus_a": counts.get("added_stable_failure", 0) - counts.get("removed_stable_failure", 0),
        "delta_stats": {col: qstats(vals) for col, vals in deltas.items()},
        "top_domain_rows": dom_rows[:20],
        "top_contextdiff_rows": ctx_rows[:20],
        "files": {
            "all": rel(out_dir / "all_transition_rows.csv"),
            "lost_correct": rel(out_dir / "lost_correct_rows.csv"),
            "gained_correct": rel(out_dir / "gained_correct_rows.csv"),
            "added_stable_failure": rel(out_dir / "added_stable_failure_rows.csv"),
            "removed_stable_failure": rel(out_dir / "removed_stable_failure_rows.csv"),
            "by_domain": rel(out_dir / "by_domain_transition.csv"),
            "by_contextdiff": rel(out_dir / "by_contextdiff_transition.csv"),
        }
    }
    (out_dir / "transition_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    data = {v: read_variant(v) for v in VARIANTS}
    sizes = {v: len(d) for v, d in data.items()}
    summaries = {}
    for a, b, label in COMPARES:
        summaries[label] = compare(a, b, label, data)
    global_summary = {
        "status": "EWOK_TRANSITION_ANATOMY_DONE",
        "created_utc": now(),
        "description": "Row-level EWoK transitions among scale1.75 live/disabled/rescaled variants and matched research legal16k base; evaluation readout only, no training source.",
        "variant_rows": sizes,
        "summaries": summaries,
    }
    summary_path = OUT / "ewok_transition_anatomy_summary.json"
    summary_path.write_text(json.dumps(global_summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = [
        "# research EWoK transition anatomy\n\n",
        f"Summary JSON: `{rel(summary_path)}`. Row-level transition files are under `{rel(OUT)}`.\n\n",
        "This uses official-format EWoK rows as a diagnostic only, not a training source. It asks whether the scale1.75 EWoK cost is row-localized enough to target with a natural fork, or distributed across the trajectory.\n\n",
    ]
    for label, s in summaries.items():
        c = s["counts"]
        lines.append(f"## {label}: `{s['a']}` -> `{s['b']}`\n")
        lines.append(f"n={s['n_common']}; acc_a={s['correct_accuracy_a']:.6f}; acc_b={s['correct_accuracy_b']:.6f}; net_correct={s['net_correct_b_minus_a']}; net_stable_failure={s['net_stable_failure_b_minus_a']}.\n")
        lines.append(f"lost_correct={c.get('lost_correct',0)}, gained_correct={c.get('gained_correct',0)}, added_stable_failure={c.get('added_stable_failure',0)}, removed_stable_failure={c.get('removed_stable_failure',0)}, lost_correct_to_stable={c.get('lost_correct_to_stable_failure',0)}, stable_to_correct={c.get('stable_failure_to_correct',0)}.\n")
        inter = s["delta_stats"]["interaction_sum"]
        lines.append(f"delta interaction_sum mean={inter.get('mean')}, median={inter.get('median')}, p10={inter.get('p10')}, p90={inter.get('p90')}.\n")
        lines.append(f"Top domain/context transition tables: `{s['files']['by_domain']}`, `{s['files']['by_contextdiff']}`.\n\n")
    NOTE.parent.mkdir(parents=True, exist_ok=True)
    NOTE.write_text("".join(lines), encoding="utf-8")
    print(json.dumps({"status": global_summary["status"], "summary": rel(summary_path), "note": rel(NOTE)}, indent=2), flush=True)

if __name__ == "__main__":
    main()
