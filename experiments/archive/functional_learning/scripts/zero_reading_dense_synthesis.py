#!/usr/bin/env python3
"""research: synthesize current dense zero-shot/Reading official-sized results.

The official background jobs are not terminal because SuperGLUE/AoA are incomplete.
This script summarizes the completed zero-shot/Reading columns currently present in
both dense payloads and compares seed concordance on that larger surface.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import json
import pathlib
import sys
from typing import Any, Dict, Iterable, Tuple

SCRIPT = _public_path('experiments/archive/functional_learning/scripts/zero_reading_dense_synthesis.py')
ROOT = _public_path('.')
STUDY = _public_path('experiments/archive/functional_learning')
SCRIPTS = _public_path('experiments/archive/functional_learning/scripts')
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import official_transition_compare as cmp  # noqa: E402

PARENT = "experiments/archive/frontier_consolidation/data/private_scale_sentinel_eval/coherent86_private_scale_0p75/per_target/coherent86_private_scale_0p75.json"
DENSE64 = "experiments/archive/functional_learning/data/dense_focus_official_eval_seed62064/per_target/dense_focus_seed62064_u0080.json"
DENSE65 = "experiments/archive/functional_learning/data/dense_focus_official_eval_seed62065/per_target/dense_focus_seed62065_u0080.json"
TR64 = "experiments/archive/functional_learning/data/dense_seed62064_zero_reading_transition/dense62064_vs_coherent86_zero_reading_current_official_transition.json"
TR65 = "experiments/archive/functional_learning/data/dense_seed62065_zero_reading_transition/dense62065_vs_coherent86_zero_reading_current_official_transition.json"
TR6564 = "experiments/archive/functional_learning/data/dense_seed62065_vs_62064_zero_reading_transition/dense62065_vs_dense62064_zero_reading_current_official_transition.json"
OUT = _public_path('experiments/archive/functional_learning/data/zero_reading_dense_synthesis')
CHEAP7 = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA", "Reading"]


def rel(path: pathlib.Path | str) -> str:
    try:
        return str(pathlib.Path(path).resolve().relative_to(ROOT))
    except Exception:
        return str(path)


def load(path: str | pathlib.Path) -> Any:
    p = pathlib.Path(path)
    if not p.is_absolute():
        p = ROOT / p
    return json.loads(p.read_text(encoding="utf-8"))


def rows_by_col(trans: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    return {r["column"]: r for r in trans.get("score_summary", {}).get("rows", [])}


def score_delta_table(trans: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    rb = rows_by_col(trans)
    out = {}
    for col in CHEAP7 + ["GlobalPIQA_parallel", "GlobalPIQA_nonparallel", "cheap7_mean"]:
        r = rb.get(col)
        if r:
            out[col] = {
                "parent_or_a_computed": r.get("a_computed"),
                "dense_or_b_computed": r.get("b_computed"),
                "computed_delta_b_minus_a": r.get("computed_delta_b_minus_a"),
                "payload_delta_b_minus_a": r.get("payload_delta_b_minus_a"),
            }
    return out


def item_overlap(parent_payload: Dict[str, Any], s64_payload: Dict[str, Any], s65_payload: Dict[str, Any]) -> Dict[str, Any]:
    parent_items = cmp.all_items(parent_payload, include_superglue=False)
    s64_items = cmp.all_items(s64_payload, include_superglue=False)
    s65_items = cmp.all_items(s65_payload, include_superglue=False)
    all_cols = [c for c in parent_items if c in s64_items and c in s65_items]

    def key_with_col(r: Dict[str, Any]) -> Tuple[str, str, int, str]:
        return (str(r["column"]), str(r["subtask"]), int(r["index"]), str(r.get("id", "")))

    def flat(items: Dict[str, list[Dict[str, Any]]]) -> Dict[Tuple[str, str, int, str], Dict[str, Any]]:
        out = {}
        for col in all_cols:
            for r in items[col]:
                out[key_with_col(r)] = r
        return out

    pmap, m64, m65 = flat(parent_items), flat(s64_items), flat(s65_items)
    keys = sorted(set(pmap) & set(m64) & set(m65))

    def summarize(keys_subset: Iterable[Tuple[str, str, int, str]]) -> Dict[str, Any]:
        keys_list = list(keys_subset)
        n = len(keys_list)
        if not n:
            return {"n": 0}
        gain64 = set(); gain65 = set(); loss64 = set(); loss65 = set(); disagree = 0
        p_correct = s64_correct = s65_correct = 0
        for k in keys_list:
            pc = bool(pmap[k]["correct"]); c64 = bool(m64[k]["correct"]); c65 = bool(m65[k]["correct"])
            p_correct += int(pc); s64_correct += int(c64); s65_correct += int(c65)
            disagree += int(c64 != c65)
            if (not pc) and c64: gain64.add(k)
            if (not pc) and c65: gain65.add(k)
            if pc and (not c64): loss64.add(k)
            if pc and (not c65): loss65.add(k)
        return {
            "n": n,
            "accuracies": {"parent": 100*p_correct/n, "seed62064": 100*s64_correct/n, "seed62065": 100*s65_correct/n},
            "seed_agreement_fraction": 1.0 - disagree/n,
            "seed_disagree_items": disagree,
            "gain_loss_counts_vs_parent": {
                "seed62064_gains": len(gain64), "seed62065_gains": len(gain65), "shared_gains": len(gain64 & gain65),
                "gain_jaccard": len(gain64 & gain65) / len(gain64 | gain65) if (gain64 | gain65) else None,
                "seed62064_losses": len(loss64), "seed62065_losses": len(loss65), "shared_losses": len(loss64 & loss65),
                "loss_jaccard": len(loss64 & loss65) / len(loss64 | loss65) if (loss64 | loss65) else None,
                "net_shared_item_delta": len(gain64 & gain65) - len(loss64 & loss65),
            },
        }

    by_col = {col: summarize(k for k in keys if k[0] == col) for col in all_cols}
    return {"columns": all_cols, "overall": summarize(keys), "by_column": by_col}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    tr64 = load(TR64); tr65 = load(TR65); tr6564 = load(TR6564)
    parent = cmp.load_payloads([PARENT]); s64 = cmp.load_payloads([DENSE64]); s65 = cmp.load_payloads([DENSE65])
    overlap = item_overlap(parent, s64, s65)
    result = {
        "status": "ZERO_READING_DENSE_SYNTHESIS",
        "script": rel(SCRIPT),
        "scope": "Completed zero-shot/Reading columns present in current dense payloads; SuperGLUE and AoA remain nonterminal/incomplete.",
        "inputs": {"parent": PARENT, "dense64": DENSE64, "dense65": DENSE65, "transitions": {"seed64_vs_parent": TR64, "seed65_vs_parent": TR65, "seed65_vs_seed64": TR6564}},
        "score_deltas_vs_coherent86": {"seed62064": score_delta_table(tr64), "seed62065": score_delta_table(tr65)},
        "seed62065_minus_seed62064_scores": score_delta_table(tr6564),
        "zero_reading_item_overlap": overlap,
        "interpretation": {
            "not_final_official": "This synthesis excludes completed SuperGLUE primary-metric mean and AoA0 recording; it cannot establish v5.",
            "scientific_read": "The larger zero-shot/Reading surface preserves a reproducible dense gain in Entity, COMPS, GlobalPIQA, and Reading, with reproducible BLiMP/Supplement/EWoK costs. Whether this is net useful depends heavily on completed SuperGLUE and exact Overall arithmetic.",
        },
    }
    out_json = _public_path('experiments/archive/functional_learning/data/zero_reading_dense_synthesis/zero_reading_dense_synthesis.json')
    out_md = _public_path('research/documents/functional_learning/data/zero_reading_dense_synthesis/zero_reading_dense_synthesis.md')
    out_json.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = ["# research dense zero-shot/Reading synthesis\n"]
    lines.append("Scope: completed zero-shot/Reading columns currently present in both dense official-compatible payloads. SuperGLUE and AoA are not complete here, so this is not final v5 evidence.\n")
    lines.append("## Score deltas vs coherent86\n")
    for seed in ["seed62064", "seed62065"]:
        lines.append(f"### {seed}")
        tab = result["score_deltas_vs_coherent86"][seed]
        for col in ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA_parallel", "GlobalPIQA_nonparallel", "GlobalPIQA", "Reading", "cheap7_mean"]:
            if col in tab:
                r = tab[col]
                lines.append(f"- {col}: parent `{r['parent_or_a_computed']}`, dense `{r['dense_or_b_computed']}`, delta `{r['computed_delta_b_minus_a']}`.")
        lines.append("")
    lines.append("## Seed concordance on full zero-shot/Reading items\n")
    ov = overlap["overall"]
    gl = ov["gain_loss_counts_vs_parent"]
    lines.append(f"Common items `{ov['n']}`, seed agreement `{ov['seed_agreement_fraction']}`, shared gains `{gl['shared_gains']}`, shared losses `{gl['shared_losses']}`, shared net `{gl['net_shared_item_delta']}`, gain/loss Jaccard `{gl['gain_jaccard']}` / `{gl['loss_jaccard']}`.\n")
    lines.append("## By-column item concordance\n")
    for col, rec in overlap["by_column"].items():
        gl = rec["gain_loss_counts_vs_parent"]
        lines.append(f"- {col}: n `{rec['n']}`, seed agreement `{rec['seed_agreement_fraction']}`, shared gains/losses `{gl['shared_gains']}`/`{gl['shared_losses']}`, shared net `{gl['net_shared_item_delta']}`.")
    lines.append("\n## Scientific interpretation\n")
    lines.append(result["interpretation"]["scientific_read"])
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "out_json": rel(out_json), "out_md": rel(out_md), "common_items": ov["n"]}, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
