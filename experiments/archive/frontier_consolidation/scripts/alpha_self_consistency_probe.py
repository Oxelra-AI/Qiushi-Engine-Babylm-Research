#!/usr/bin/env python3
"""research payload-only probe: alpha-scale self-consistency as a label-free signal.

This tests simple, pre-specified decision rules over saved alpha-scale predictions:
  * majority of alpha0.5/0.75/1, falling back to anchor if all three disagree;
  * majority of anchor+alpha0.5+alpha0.75+alpha1, ties falling back to anchor;
  * private consensus only: use private prediction only if all three scaled alphas agree.

The purpose is not to create a benchmark-tuned submission. It asks whether alpha-scale
self-consistency contains enough signal to justify a future official-compatible custom
model wrapper that averages or gates logits without benchmark labels. If these simple
winner-level rules do not improve the discrete surface, the route is weak.
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
WORKSPACE = _public_path('experiments/archive/frontier_consolidation')
SCRIPTS = _public_path('experiments/archive/frontier_consolidation/scripts')
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from pairwise_item_flip_analysis import DISCRETE_COLUMNS, PayloadLoader, ItemRow, official_score  # noqa: E402

PAYLOADS = OrderedDict([
    ("a0_anchor", _public_path('experiments/archive/representation_and_objectives/data/scale1p75_chck82_full_eval_reproduction/staged_full_eval/per_target/scale1p75_chck82_independent.json')),
    ("a0p5", _public_path('experiments/archive/frontier_consolidation/data/private_scale_sentinel_eval/coherent86_private_scale_0p5/per_target/coherent86_private_scale_0p5.json')),
    ("a0p75", _public_path('experiments/archive/frontier_consolidation/data/private_scale_sentinel_eval/coherent86_private_scale_0p75/per_target/coherent86_private_scale_0p75.json')),
    ("a1", _public_path('experiments/archive/frontier_consolidation/data/fastpath4M_coherent_eval/per_target/fastpath4M_coherent.json')),
])
OUT = _public_path('experiments/archive/frontier_consolidation/data/alpha_self_consistency_probe')


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def rel(p: pathlib.Path | str) -> str:
    q = pathlib.Path(p)
    try:
        return str(q.resolve().relative_to(ROOT))
    except Exception:
        return str(q)


def choose_majority(preds: list[str | None], fallback: str | None) -> str | None:
    cnt = Counter([p for p in preds if p is not None])
    if not cnt:
        return fallback
    most = cnt.most_common()
    if len(most) == 1 or most[0][1] > most[1][1]:
        return most[0][0]
    if fallback in [p for p, c in most if c == most[0][1]]:
        return fallback
    return fallback


def make_row(template: ItemRow, pred: str | None) -> ItemRow:
    return ItemRow(
        item_id=template.item_id,
        uid=template.uid,
        correct=(pred == template.gold),
        pred=pred,
        gold=template.gold,
        column=template.column,
        sub=template.sub,
        meta=template.meta,
    )


def rule_prediction(rule: str, rows: dict[str, ItemRow]) -> str | None:
    anchor = rows["a0_anchor"].pred
    scaled = [rows[a].pred for a in ["a0p5", "a0p75", "a1"]]
    if rule == "a0_anchor":
        return anchor
    if rule in rows:
        return rows[rule].pred
    if rule == "scaled_majority_else_anchor":
        return choose_majority(scaled, anchor)
    if rule == "anchor_plus_scaled_majority_tie_anchor":
        return choose_majority([anchor, *scaled], anchor)
    if rule == "private_unanimous_else_anchor":
        vals = [p for p in scaled if p is not None]
        if len(vals) == 3 and vals[0] == vals[1] == vals[2]:
            return vals[0]
        return anchor
    if rule == "a0p5_unless_075_1_agree_else_that":
        if rows["a0p75"].pred is not None and rows["a0p75"].pred == rows["a1"].pred:
            return rows["a0p75"].pred
        return rows["a0p5"].pred
    raise KeyError(rule)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rules = [
        "a0_anchor", "a0p5", "a0p75", "a1",
        "scaled_majority_else_anchor", "anchor_plus_scaled_majority_tie_anchor",
        "private_unanimous_else_anchor", "a0p5_unless_075_1_agree_else_that",
    ]
    scores: dict[str, dict[str, float]] = {r: {} for r in rules}
    movement: dict[str, dict[str, Any]] = {r: {} for r in rules if r != "a0_anchor"}
    rule_rows_by_col: dict[str, dict[str, list[ItemRow]]] = {r: {} for r in rules}

    for col in DISCRETE_COLUMNS:
        maps = {}
        for label, path in PAYLOADS.items():
            rows, _ = PayloadLoader(path).load_column(col)
            maps[label] = {x.item_id: x for x in rows}
        common = sorted(set.intersection(*(set(m) for m in maps.values())))
        for rule in rules:
            out_rows = []
            for item_id in common:
                rowset = {a: maps[a][item_id] for a in PAYLOADS}
                pred = rule_prediction(rule, rowset)
                out_rows.append(make_row(rowset["a0_anchor"], pred))
            rule_rows_by_col[rule][col] = out_rows
            scores[rule][col] = official_score(out_rows, col)
        # movement against anchor by exact item correctness
        anchor_rows = {r.item_id: r for r in rule_rows_by_col["a0_anchor"][col]}
        for rule in [r for r in rules if r != "a0_anchor"]:
            gain = loss = changed = 0
            both_correct = 0
            for rr in rule_rows_by_col[rule][col]:
                aa = anchor_rows[rr.item_id]
                if bool(aa.correct) != bool(rr.correct):
                    changed += 1
                    if (not aa.correct) and rr.correct:
                        gain += 1
                    elif aa.correct and (not rr.correct):
                        loss += 1
                elif aa.correct and rr.correct:
                    both_correct += 1
            movement[rule][col] = {"gain": gain, "loss": loss, "net": gain-loss, "changed": changed, "retention": both_correct/(both_correct+loss) if (both_correct+loss) else None}

    summary = {}
    for rule in rules:
        discrete6 = mean(scores[rule][c] for c in DISCRETE_COLUMNS)
        summary[rule] = {
            "scores": scores[rule],
            "discrete6_mean": discrete6,
            "delta_discrete6_vs_anchor": discrete6 - mean(scores["a0_anchor"][c] for c in DISCRETE_COLUMNS),
        }
        if rule != "a0_anchor":
            total_gain = sum(movement[rule][c]["gain"] for c in DISCRETE_COLUMNS)
            total_loss = sum(movement[rule][c]["loss"] for c in DISCRETE_COLUMNS)
            summary[rule]["movement_vs_anchor"] = {"gain": total_gain, "loss": total_loss, "net": total_gain-total_loss, "changed": total_gain+total_loss, "by_column": movement[rule]}

    reading = []
    best = max([r for r in rules if r != "a0_anchor"], key=lambda r: summary[r]["delta_discrete6_vs_anchor"])
    reading.append(f"Best simple self-consistency rule on the six discrete columns is {best} with delta {summary[best]['delta_discrete6_vs_anchor']:+.6f} points vs anchor.")
    reading.append("Because this uses saved winners, not logits, it is only a headroom check. A future custom multi-alpha model wrapper would require a frozen label-free rule before evaluation, and should not be pursued if this winner-level headroom is weak or purely redistribution.")

    out = {"status": "COMPLETE", "created_utc": now(), "payloads": {k: rel(v) for k,v in PAYLOADS.items()}, "summary": summary, "scientific_reading": reading}
    js = _public_path('experiments/archive/frontier_consolidation/data/alpha_self_consistency_probe/alpha_self_consistency_probe.json')
    md = _public_path('research/documents/frontier_consolidation/data/alpha_self_consistency_probe/alpha_self_consistency_probe.md')
    js.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = [
        "# research alpha self-consistency probe",
        "",
        f"Status: **{out['status']}**",
        "",
        "This is a payload-only headroom check on six discrete columns; Reading/SuperGLUE/AoA are not included.",
        "",
        "| rule | discrete6 mean | delta vs anchor | gain | loss | net | changed |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for rule, rec in sorted(summary.items(), key=lambda kv: kv[1]["delta_discrete6_vs_anchor"], reverse=True):
        mov = rec.get("movement_vs_anchor", {"gain":0,"loss":0,"net":0,"changed":0})
        lines.append(f"| {rule} | {rec['discrete6_mean']:.6f} | {rec['delta_discrete6_vs_anchor']:+.6f} | {mov['gain']} | {mov['loss']} | {mov['net']:+d} | {mov['changed']} |")
    lines += ["", "## By-column scores", "", "| rule | BLiMP | Supplement | EWoK | Entity | COMPS | GlobalPIQA |", "|---|---:|---:|---:|---:|---:|---:|"]
    for rule, rec in sorted(summary.items(), key=lambda kv: kv[1]["delta_discrete6_vs_anchor"], reverse=True):
        s = rec["scores"]
        lines.append(f"| {rule} | {s['BLiMP']:.4f} | {s['Supplement']:.4f} | {s['EWoK']:.4f} | {s['Entity']:.4f} | {s['COMPS']:.4f} | {s['GlobalPIQA']:.4f} |")
    lines += ["", "## Scientific reading", ""]
    for x in reading:
        lines.append(f"- {x}")
    lines += ["", f"JSON: `{rel(js)}`"]
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": out["status"], "out_json": rel(js), "out_md": rel(md), "best_rule": best, "best_delta_discrete6": summary[best]["delta_discrete6_vs_anchor"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
