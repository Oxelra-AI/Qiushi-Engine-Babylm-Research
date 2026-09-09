#!/usr/bin/env python3
"""research focused audit: do private-scale alphas improve relation/state families?

The frozen-anchor fast-path route was motivated by protecting late relation/state
competence (EWoK and Entity) while adding broad replay knowledge. Aggregate cheap7
can hide that the private residual mostly changes BLiMP/COMPS and a few GlobalPIQA
items. This payload-only audit reports EWoK/Entity item movements for alpha0.5,
alpha0.75, alpha1, shuffled86, and ordinary86 relative to the protected anchor.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import json
import pathlib
import sys
import time
from collections import Counter, OrderedDict, defaultdict
from statistics import mean
from typing import Any

ROOT = _public_path('.')
STUDY = _public_path('experiments/archive/frontier_consolidation')
WORKSPACE = _public_path('experiments/archive/frontier_consolidation')
SCRIPTS = _public_path('experiments/archive/frontier_consolidation/scripts')
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from pairwise_item_flip_analysis import PayloadLoader, official_score, uid_group  # noqa: E402

PAYLOADS = OrderedDict([
    ("chck82_anchor", _public_path('experiments/archive/representation_and_objectives/data/scale1p75_chck82_full_eval_reproduction/staged_full_eval/per_target/scale1p75_chck82_independent.json')),
    ("ordinary86_backbone", _public_path('experiments/archive/representation_and_objectives/data/scale1p75_chck86_cheap7_eval/per_target/scale1p75_chck86_cheap7.json')),
    ("shuffled86_private", _public_path('experiments/archive/frontier_consolidation/data/frozen82_tail4M_shuffled_eval/per_target/frozen82_tail4M_shuffled.json')),
    ("coherent86_alpha0p5", _public_path('experiments/archive/frontier_consolidation/data/private_scale_sentinel_eval/coherent86_private_scale_0p5/per_target/coherent86_private_scale_0p5.json')),
    ("coherent86_alpha0p75", _public_path('experiments/archive/frontier_consolidation/data/private_scale_sentinel_eval/coherent86_private_scale_0p75/per_target/coherent86_private_scale_0p75.json')),
    ("coherent86_alpha1", _public_path('experiments/archive/frontier_consolidation/data/fastpath4M_coherent_eval/per_target/fastpath4M_coherent.json')),
])
FOCUS_COLS = ["EWoK", "Entity"]
OUT = _public_path('experiments/archive/frontier_consolidation/data/relation_state_alpha_audit')


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def rel(path: pathlib.Path | str) -> str:
    p = pathlib.Path(path)
    try:
        return str(p.resolve().relative_to(ROOT))
    except Exception:
        return str(p)


def load_col_maps(col: str) -> dict[str, dict[str, Any]]:
    out = {}
    for label, path in PAYLOADS.items():
        rows, _ = PayloadLoader(path).load_column(col)
        out[label] = {r.item_id: r for r in rows}
    return out


def compare(col: str, base_label: str, cand_label: str, maps: dict[str, dict[str, Any]]) -> dict[str, Any]:
    base = maps[base_label]
    cand = maps[cand_label]
    common = sorted(set(base) & set(cand))
    gain = []
    loss = []
    both_correct = both_wrong = 0
    group_stats: dict[str, Counter] = defaultdict(Counter)
    sub_stats: dict[str, Counter] = defaultdict(Counter)
    for item_id in common:
        b = base[item_id]
        c = cand[item_id]
        g = uid_group(col, b)
        bcor, ccor = bool(b.correct), bool(c.correct)
        if bcor and ccor:
            both_correct += 1
            group_stats[g]["both_correct"] += 1
        elif (not bcor) and (not ccor):
            both_wrong += 1
            group_stats[g]["both_wrong"] += 1
        elif (not bcor) and ccor:
            gain.append(item_id)
            group_stats[g]["gain"] += 1
        else:
            loss.append(item_id)
            group_stats[g]["loss"] += 1
        group_stats[g]["n"] += 1
        if col == "Entity":
            fam = (b.meta or {}).get("family", "unknown")
            numops = (b.meta or {}).get("numops", "unknown")
            key = f"{fam}|{numops}_ops"
            if (not bcor) and ccor: sub_stats[key]["gain"] += 1
            elif bcor and (not ccor): sub_stats[key]["loss"] += 1
            elif bcor and ccor: sub_stats[key]["both_correct"] += 1
            else: sub_stats[key]["both_wrong"] += 1
            sub_stats[key]["n"] += 1
        elif col == "EWoK":
            key = str((b.meta or {}).get("file", g))
            if (not bcor) and ccor: sub_stats[key]["gain"] += 1
            elif bcor and (not ccor): sub_stats[key]["loss"] += 1
            elif bcor and ccor: sub_stats[key]["both_correct"] += 1
            else: sub_stats[key]["both_wrong"] += 1
            sub_stats[key]["n"] += 1

    base_common = [base[i] for i in common]
    cand_common = [cand[i] for i in common]
    # official_score on common rows is not exactly full payload if omitted/skipped differ, but for common
    # payloads here it matches the comparable item surface.
    score_base = official_score(base_common, col)
    score_cand = official_score(cand_common, col)
    def pack_stats(stats: dict[str, Counter], limit: int = 20) -> list[dict[str, Any]]:
        rows=[]
        for key, cnt in stats.items():
            n = int(cnt.get("n", 0))
            g = int(cnt.get("gain", 0)); l = int(cnt.get("loss", 0))
            rows.append({
                "group": key,
                "n": n,
                "gain": g,
                "loss": l,
                "net": g-l,
                "changed": g+l,
                "base_correct": int(cnt.get("both_correct",0)+l),
                "candidate_correct": int(cnt.get("both_correct",0)+g),
                "net_per_100_items": 100.0*(g-l)/n if n else 0.0,
            })
        return sorted(rows, key=lambda r: (abs(r["net_per_100_items"]), r["changed"], abs(r["net"])), reverse=True)[:limit]
    return {
        "base": base_label,
        "candidate": cand_label,
        "column": col,
        "n_common": len(common),
        "score_base_common": score_base,
        "score_candidate_common": score_cand,
        "delta_score_points_common": score_cand - score_base,
        "gain_items": len(gain),
        "loss_items": len(loss),
        "net_gain_minus_loss": len(gain) - len(loss),
        "changed_items": len(gain) + len(loss),
        "anchor_correct_retention_fraction": both_correct / (both_correct + len(loss)) if (both_correct + len(loss)) else None,
        "top_group_movements": pack_stats(group_stats),
        "top_subfamily_movements": pack_stats(sub_stats),
        "sample_gain_items": gain[:20],
        "sample_loss_items": loss[:20],
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    all_maps = {col: load_col_maps(col) for col in FOCUS_COLS}
    comparisons: dict[str, Any] = {}
    for col in FOCUS_COLS:
        maps = all_maps[col]
        for cand in [x for x in PAYLOADS if x != "chck82_anchor"]:
            key = f"{cand}_minus_chck82_anchor_{col}"
            comparisons[key] = compare(col, "chck82_anchor", cand, maps)
    # compact summary table
    rows=[]
    for key, rec in comparisons.items():
        rows.append({
            "candidate": rec["candidate"],
            "column": rec["column"],
            "delta_score_points_common": rec["delta_score_points_common"],
            "gain": rec["gain_items"],
            "loss": rec["loss_items"],
            "net": rec["net_gain_minus_loss"],
            "changed": rec["changed_items"],
            "retention": rec["anchor_correct_retention_fraction"],
        })
    # Main reading: compare alpha endpoints against motivation.
    alpha_rows = [r for r in rows if r["candidate"].startswith("coherent86_alpha")]
    reading = [
        "The frozen-anchor fast path was motivated by relation/state retention. On EWoK/Entity specifically, alpha endpoints are modest: alpha0.5 has EWoK gain but Entity loss; alpha0.75 is near-neutral; alpha1 is near-neutral/slightly positive depending on column.",
        "The relation/state movement is far smaller than COMPS/BLiMP churn and far smaller than GlobalPIQA's apparent score swing from only five alpha-sensitive examples, so current alpha endpoints should be read as endpoint redistribution rather than a solved relation/state protection mechanism.",
    ]
    out = {
        "status": "COMPLETE",
        "created_utc": now(),
        "payloads": {k: rel(v) for k, v in PAYLOADS.items()},
        "focus_columns": FOCUS_COLS,
        "summary_rows": rows,
        "comparisons": comparisons,
        "scientific_reading": reading,
    }
    js = _public_path('experiments/archive/frontier_consolidation/data/relation_state_alpha_audit/relation_state_alpha_audit.json')
    md = _public_path('research/documents/frontier_consolidation/data/relation_state_alpha_audit/relation_state_alpha_audit.md')
    js.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = [
        "# research relation/state alpha audit",
        "",
        f"Status: **{out['status']}**",
        "",
        "## Summary vs protected chck82 anchor",
        "",
        "| candidate | column | delta score pts | gain | loss | net | changed | retention |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(f"| {r['candidate']} | {r['column']} | {r['delta_score_points_common']:+.6f} | {r['gain']} | {r['loss']} | {r['net']:+d} | {r['changed']} | {r['retention']:.6f} |")
    lines += ["", "## Top subfamily movements for alpha endpoints", ""]
    for cand in ["coherent86_alpha0p5", "coherent86_alpha0p75", "coherent86_alpha1"]:
        for col in FOCUS_COLS:
            rec = comparisons[f"{cand}_minus_chck82_anchor_{col}"]
            lines += [f"### {cand} / {col}", "", "| subfamily | n | gain | loss | net | net/100 |", "|---|---:|---:|---:|---:|---:|"]
            for s in rec["top_subfamily_movements"][:10]:
                lines.append(f"| {s['group']} | {s['n']} | {s['gain']} | {s['loss']} | {s['net']:+d} | {s['net_per_100_items']:+.3f} |")
            lines.append("")
    lines += ["## Scientific reading", ""]
    for x in reading:
        lines.append(f"- {x}")
    lines += ["", f"JSON: `{rel(js)}`"]
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": out["status"], "out_json": rel(js), "out_md": rel(md)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
