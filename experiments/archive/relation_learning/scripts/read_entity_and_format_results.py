#!/usr/bin/env python3
"""Collect research terminal result files into a compact scientific readout.

This can be run after the evaluation outputs are complete.  It does not evaluate models; it
summarizes already-written Entity and corrected-format outputs.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative

import csv, json, math, pathlib, statistics, time

ROOT = _public_path('.')
OUT = _public_path('experiments/archive/relation_learning/data/result_readout')
ENTITY = _public_path('experiments/archive/relation_learning/data/tight_private_entity_eval')
FORMAT_TRAIN = _public_path('experiments/archive/relation_learning/data/format_replay_corrected')
FORMAT_EVAL = _public_path('experiments/archive/relation_learning/data/eval_format_replay_corrected')
PAIR = _public_path('experiments/archive/relation_learning/data/format_pairwise_item_flips/pairwise_item_flips.json')

KEY_GROUPS = ["ALL", "rel_eq0", "rel_eq0_irrelevant_ops_gt0", "rel_ge1", "rel_ge1_postrel_ops0", "rel_ge1_postrel_ops_gt0", "rel_ge3", "rel_ge3_postrel_ops0", "rel_ge3_postrel_ops_gt0", "stale_available_not_gold"]


def rel(p):
    try: return str(pathlib.Path(p).resolve().relative_to(ROOT))
    except Exception: return str(p)


def now(): return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

def read_json(p): return json.loads(pathlib.Path(p).read_text(encoding="utf-8"))

def read_csv(p):
    with pathlib.Path(p).open(encoding="utf-8") as f:
        return list(csv.DictReader(f))

def maybe(p): return pathlib.Path(p).exists()

def summarize_entity():
    out = {"available": False}
    sp = _public_path('experiments/archive/relation_learning/data/tight_private_entity_eval/summary.json')
    if not sp.exists():
        return out
    s = read_json(sp)
    ev = s.get("entity_eval", {})
    out.update({"available": True, "score": ev.get("score"), "predictions": ev.get("predictions"), "summary": rel(sp), "diagnostics": s.get("diagnostics", [])})
    for base in ["chck82", "coherent86"]:
        dp = ENTITY / f"recency_vs_{base}" / "recency_pairwise_deltas.csv"
        if dp.exists():
            rows = {r["group"]: r for r in read_csv(dp)}
            out[f"deltas_vs_{base}"] = {g: rows[g] for g in KEY_GROUPS if g in rows}
    return out

def summarize_format_train():
    out = {}
    for arm in ["isolated_all", "half_coherent_half_isolated"]:
        arm_rows = {}
        for seed in [98097, 98098]:
            d = FORMAT_TRAIN / arm / f"seed{seed}"
            sp = d / "summary.json"
            lp = d / "training_log.jsonl"
            if sp.exists():
                s = read_json(sp)
                logs = []
                if lp.exists():
                    logs = [json.loads(x) for x in lp.read_text(encoding="utf-8").splitlines() if x.strip()]
                arm_rows[str(seed)] = {
                    "summary": rel(sp),
                    "updates": s.get("updates"),
                    "total_words": s.get("total_words"),
                    "first_update": s.get("first_update") or (logs[0] if logs else None),
                    "last_update": s.get("last_update") or (logs[-1] if logs else None),
                }
        out[arm] = arm_rows
    return out

def summarize_format_eval():
    out = {}
    for arm in ["isolated_all", "half_coherent_half_isolated"]:
        arm_rows = {}
        for seed in [98097, 98098]:
            label = f"step098_{arm}_seed{seed}_alpha0p75"
            sp = FORMAT_EVAL / arm / "summary" / f"{label}_summary.json"
            if sp.exists():
                s = read_json(sp)
                arm_rows[str(seed)] = {"summary": rel(sp), "cheap7": s.get("cheap7"), "deltas_vs_chck82": s.get("deltas_vs_chck82"), "scores": s.get("scores"), "payload_path": s.get("payload_path")}
        out[arm] = arm_rows
    return out

def summarize_pairwise():
    if not PAIR.exists():
        return {"available": False}
    p = read_json(PAIR)
    agg = {k: v.get("aggregate") for k, v in p.get("comparisons", {}).items() if k.startswith("step098_")}
    return {"available": True, "json": rel(PAIR), "md": rel(PAIR.with_suffix('.md')), "cheap7": p.get("cheap7"), "deltas_vs_anchor": p.get("deltas_vs_anchor"), "aggregates": agg}

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    readout = {"status": "RESULT_READOUT", "created_utc": now(), "entity": summarize_entity(), "format_train": summarize_format_train(), "format_eval": summarize_format_eval(), "format_pairwise": summarize_pairwise()}
    (_public_path('experiments/archive/relation_learning/data/result_readout/readout.json')).write_text(json.dumps(readout, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = ["# research result readout", "", f"Created UTC: {readout['created_utc']}", ""]
    ent = readout["entity"]
    lines += ["## Tight-private Entity transfer", ""]
    if ent.get("available"):
        lines += [f"Score: `{ent.get('score')}`; predictions: `{ent.get('predictions')}`", ""]
        for base in ["chck82", "coherent86"]:
            ds = ent.get(f"deltas_vs_{base}") or {}
            lines += [f"### Deltas vs {base}", "", "| group | delta acc | base acc | candidate acc | n |", "|---|---:|---:|---:|---:|"]
            for g, r in ds.items():
                lines.append(f"| {g} | {r.get('delta_accuracy_pct')} | {r.get('baseline_accuracy_pct')} | {r.get('accuracy_pct')} | {r.get('n')} |")
            lines.append("")
    else:
        lines.append("Not available yet.\n")
    lines += ["## Corrected format training invariants", ""]
    for arm, seeds in readout["format_train"].items():
        lines += [f"### {arm}", "", "| seed | updates | total words | first targets/non-special | first CE | first readout KL | final CE | final readout KL |", "|---:|---:|---:|---|---:|---:|---:|---:|"]
        for seed, rec in seeds.items():
            fu = rec.get("first_update") or {}; lu = rec.get("last_update") or {}
            lines.append(f"| {seed} | {rec.get('updates')} | {rec.get('total_words')} | {fu.get('main_targets')}/{fu.get('non_special_tokens')} ({fu.get('target_ratio')}) | {fu.get('main_ce')} | {fu.get('readout_neutral_kl')} | {lu.get('main_ce')} | {lu.get('readout_neutral_kl')} |")
        lines.append("")
    lines += ["## Corrected format cheap7", ""]
    for arm, seeds in readout["format_eval"].items():
        lines += [f"### {arm}", "", "| seed | cheap7 | delta vs chck82 | scores |", "|---:|---:|---:|---|"]
        for seed, rec in seeds.items():
            lines.append(f"| {seed} | {rec.get('cheap7')} | {rec.get('deltas_vs_chck82', {}).get('cheap7')} | `{rec.get('scores')}` |")
        lines.append("")
    pw = readout["format_pairwise"]
    if pw.get("available"):
        lines += ["## Pairwise item flips", "", f"JSON: `{pw.get('json')}`; MD: `{pw.get('md')}`", "", "```json", json.dumps(pw.get("aggregates"), indent=2, ensure_ascii=False), "```", ""]
    (_public_path('research/documents/relation_learning/data/result_readout/readout.md')).write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"status": readout["status"], "json": rel(_public_path('experiments/archive/relation_learning/data/result_readout/readout.json')), "md": rel(_public_path('research/documents/relation_learning/data/result_readout/readout.md'))}, indent=2))

if __name__ == "__main__":
    main()
