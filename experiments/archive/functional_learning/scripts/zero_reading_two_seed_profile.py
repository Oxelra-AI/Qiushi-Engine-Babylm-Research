#!/usr/bin/env python3
"""research: two-seed localization on completed zero-shot/Reading columns.

This script complements `zero_reading_dense_synthesis.py` by localizing the
current official-sized zero-shot/Reading movement for both dense seeds at the group
level.  SuperGLUE/AoA remain incomplete and are intentionally excluded.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import json
import pathlib
import re
import sys
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Tuple

SCRIPT = _public_path('experiments/archive/functional_learning/scripts/zero_reading_two_seed_profile.py')
ROOT = _public_path('.')
STUDY = _public_path('experiments/archive/functional_learning')
SCRIPTS = _public_path('experiments/archive/functional_learning/scripts')
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import official_transition_compare as cmp  # noqa: E402

PARENT = "experiments/archive/frontier_consolidation/data/private_scale_sentinel_eval/coherent86_private_scale_0p75/per_target/coherent86_private_scale_0p75.json"
DENSE = {
    "seed62064": "experiments/archive/functional_learning/data/dense_focus_official_eval_seed62064/per_target/dense_focus_seed62064_u0080.json",
    "seed62065": "experiments/archive/functional_learning/data/dense_focus_official_eval_seed62065/per_target/dense_focus_seed62065_u0080.json",
}
OUT = _public_path('experiments/archive/functional_learning/data/zero_reading_two_seed_profile')


def rel(path: pathlib.Path | str) -> str:
    try:
        return str(pathlib.Path(path).resolve().relative_to(ROOT))
    except Exception:
        return str(path)


def items_col(payload: Dict[str, Any], col: str) -> List[Dict[str, Any]]:
    if col == "BLiMP":
        return cmp.blimp_like_items(payload, "BLiMP", cmp.PRISTINE_FULL / "blimp_filtered")
    if col == "Supplement":
        return cmp.blimp_like_items(payload, "Supplement", cmp.PRISTINE_FULL / "supplement_filtered")
    if col == "EWoK":
        return cmp.ewok_items(payload)
    if col == "Entity":
        return cmp.entity_items(payload)
    if col == "COMPS":
        return cmp.comps_items(payload)
    if col == "GlobalPIQA_parallel":
        return cmp.globalpiqa_items(payload, "GlobalPIQA_parallel", "global_piqa_parallel")
    if col == "GlobalPIQA_nonparallel":
        return cmp.globalpiqa_items(payload, "GlobalPIQA_nonparallel", "global_piqa_nonparallel")
    raise KeyError(col)


def keyed(rows: Iterable[Dict[str, Any]]) -> Dict[Tuple[str, int, str], Dict[str, Any]]:
    return {cmp.key_item(r): r for r in rows}


def compare_group(a_rows: List[Dict[str, Any]], b_rows: List[Dict[str, Any]], key_fn) -> Dict[str, Dict[str, Any]]:
    amap = keyed(a_rows); bmap = keyed(b_rows)
    buckets: Dict[str, Dict[str, Any]] = defaultdict(lambda: {"n": 0, "a_correct": 0, "b_correct": 0, "a_only": 0, "b_only": 0})
    for k in sorted(set(amap) & set(bmap)):
        ar, br = amap[k], bmap[k]
        g = str(key_fn(ar))
        ac, bc = bool(ar["correct"]), bool(br["correct"])
        x = buckets[g]
        x["n"] += 1
        x["a_correct"] += int(ac); x["b_correct"] += int(bc)
        x["a_only"] += int(ac and not bc); x["b_only"] += int((not ac) and bc)
    out = {}
    for g, x in buckets.items():
        n = x["n"]
        aa = 100.0 * x["a_correct"] / n if n else None
        ba = 100.0 * x["b_correct"] / n if n else None
        out[g] = {"n": n, "parent_acc": aa, "dense_acc": ba, "delta": None if aa is None or ba is None else ba-aa, "net_items": x["b_correct"]-x["a_correct"], "parent_only_correct": x["a_only"], "dense_only_correct": x["b_only"]}
    return out


def m_entity(subtask: str) -> Tuple[str, str]:
    m = re.match(r"(.+)_([0-9]+)_ops$", subtask)
    return (m.group(1), f"{m.group(2)}_ops") if m else (subtask, "unknown")


def entity_family(subtask: str) -> str:
    return m_entity(subtask)[0]


def entity_depth(subtask: str) -> str:
    return m_entity(subtask)[1]


def blimp_family(subtask: str) -> str:
    s = subtask.lower()
    if "agreement" in s or "determiner_noun" in s:
        return "agreement_and_number"
    if any(x in s for x in ["island", "wh_", "coordinate_structure", "left_branch", "complex_np"]):
        return "extraction_and_islands"
    if any(x in s for x in ["npi", "negation", "only", "quantifier", "existential_there"]):
        return "quantifier_npi_scope"
    if any(x in s for x in ["principle_a", "anaphor"]):
        return "binding_anaphora"
    if any(x in s for x in ["transitive", "passive", "inchoative", "causative", "drop_argument", "ellipsis", "raising", "tough"]):
        return "argument_structure_ellipsis"
    return "other_blimp"


def merged_two_seed(group_maps: Dict[str, Dict[str, Dict[str, Any]]]) -> List[Dict[str, Any]]:
    groups = sorted(set().union(*(set(m.keys()) for m in group_maps.values())))
    out = []
    for g in groups:
        row = {"group": g}
        vals = []
        for seed, mp in group_maps.items():
            r = mp.get(g)
            if r:
                row[f"{seed}_delta"] = r["delta"]
                row[f"{seed}_net_items"] = r["net_items"]
                row[f"{seed}_parent_acc"] = r["parent_acc"]
                row[f"{seed}_dense_acc"] = r["dense_acc"]
                row["n"] = r["n"]
                vals.append(float(r["delta"]))
            else:
                row[f"{seed}_delta"] = None
                row[f"{seed}_net_items"] = None
        row["mean_delta"] = sum(vals)/len(vals) if vals else None
        row["same_sign"] = (len(vals) == 2 and ((vals[0] > 0 and vals[1] > 0) or (vals[0] < 0 and vals[1] < 0) or (vals[0] == 0 and vals[1] == 0)))
        out.append(row)
    out.sort(key=lambda r: (abs(float(r.get("mean_delta") or 0.0)), r.get("n") or 0), reverse=True)
    return out


def score_rows(payload: Dict[str, Any]) -> Dict[str, float | None]:
    tasks = payload.get("tasks") or {}
    scores: Dict[str, float | None] = {}
    for col in ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA_parallel", "GlobalPIQA_nonparallel"]:
        rec = tasks.get(col) or {}
        scores[col] = None if rec.get("score") is None else float(rec["score"])
    if scores.get("GlobalPIQA_parallel") is not None and scores.get("GlobalPIQA_nonparallel") is not None:
        scores["GlobalPIQA"] = (float(scores["GlobalPIQA_parallel"]) + float(scores["GlobalPIQA_nonparallel"])) / 2.0
    r = (tasks.get("Reading") or {}).get("scores") or {}
    scores["Reading"] = None if r.get("Reading") is None else float(r["Reading"])
    if all(scores.get(k) is not None for k in ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA", "Reading"]):
        scores["cheap7_mean_payload"] = sum(float(scores[k]) for k in ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA", "Reading"]) / 7.0
    return scores


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    parent = cmp.load_payloads([PARENT])
    dense_payloads = {k: cmp.load_payloads([v]) for k, v in DENSE.items()}
    parent_scores = score_rows(parent)
    dense_scores = {k: score_rows(v) for k, v in dense_payloads.items()}
    deltas = {seed: {k: (None if parent_scores.get(k) is None or sc.get(k) is None else float(sc[k]) - float(parent_scores[k])) for k in sorted(set(parent_scores) | set(sc))} for seed, sc in dense_scores.items()}

    cols = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA_parallel", "GlobalPIQA_nonparallel"]
    group_profiles: Dict[str, Any] = {}
    for col in cols:
        p_rows = items_col(parent, col)
        maps_by_seed = {}
        for seed, payload in dense_payloads.items():
            b_rows = items_col(payload, col)
            if col == "Entity":
                maps_by_seed.setdefault("Entity_by_depth", {})[seed] = compare_group(p_rows, b_rows, lambda r: entity_depth(str(r["subtask"])))
                maps_by_seed.setdefault("Entity_by_family", {})[seed] = compare_group(p_rows, b_rows, lambda r: entity_family(str(r["subtask"])))
            elif col == "BLiMP":
                maps_by_seed.setdefault("BLiMP_by_family", {})[seed] = compare_group(p_rows, b_rows, lambda r: blimp_family(str(r["subtask"])))
                maps_by_seed.setdefault("BLiMP_by_subtask", {})[seed] = compare_group(p_rows, b_rows, lambda r: str(r["subtask"]))
            else:
                maps_by_seed.setdefault(f"{col}_by_subtask", {})[seed] = compare_group(p_rows, b_rows, lambda r: str(r["subtask"]))
        for name, seed_maps in maps_by_seed.items():
            group_profiles[name] = merged_two_seed(seed_maps)

    result = {
        "status": "ZERO_READING_TWO_SEED_PROFILE",
        "script": rel(SCRIPT),
        "scope": "Completed zero-shot/Reading official-sized columns for dense seed62064 and seed62065; SuperGLUE/AoA excluded.",
        "inputs": {"parent": PARENT, "dense": DENSE},
        "payload_score_deltas_vs_parent": deltas,
        "group_profiles": group_profiles,
        "interpretation": "The profile identifies stable gains and costs on completed official-sized surfaces. It should guide repair/control only after terminal official jobs establish the full SuperGLUE/AoA0 outcome.",
    }
    out_json = _public_path('experiments/archive/functional_learning/data/zero_reading_two_seed_profile/zero_reading_two_seed_profile.json')
    out_md = _public_path('research/documents/functional_learning/data/zero_reading_two_seed_profile/zero_reading_two_seed_profile.md')
    out_json.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = ["# research zero-shot/Reading two-seed profile\n"]
    lines.append("Scope: current completed zero-shot/Reading official-sized columns only; SuperGLUE and AoA are excluded.\n")
    lines.append("## Payload score deltas vs coherent86\n")
    for seed in ["seed62064", "seed62065"]:
        lines.append(f"### {seed}")
        for col in ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA", "Reading", "cheap7_mean_payload"]:
            if col in deltas[seed]:
                lines.append(f"- {col}: `{deltas[seed][col]}`")
        lines.append("")
    lines.append("## Entity by operation depth\n")
    for r in group_profiles.get("Entity_by_depth", []):
        lines.append(f"- {r['group']}: mean delta `{r['mean_delta']}`, seed62064 `{r['seed62064_delta']}`, seed62065 `{r['seed62065_delta']}`, same sign `{r['same_sign']}`, n `{r.get('n')}`.")
    lines.append("\n## Entity by task family\n")
    for r in group_profiles.get("Entity_by_family", []):
        lines.append(f"- {r['group']}: mean delta `{r['mean_delta']}`, seed62064 `{r['seed62064_delta']}`, seed62065 `{r['seed62065_delta']}`, same sign `{r['same_sign']}`, n `{r.get('n')}`.")
    lines.append("\n## BLiMP family movement\n")
    for r in group_profiles.get("BLiMP_by_family", []):
        lines.append(f"- {r['group']}: mean delta `{r['mean_delta']}`, seed62064 `{r['seed62064_delta']}`, seed62065 `{r['seed62065_delta']}`, same sign `{r['same_sign']}`, n `{r.get('n')}`.")
    lines.append("\n## Largest stable BLiMP subtask movements\n")
    for r in [x for x in group_profiles.get("BLiMP_by_subtask", []) if x.get("same_sign")][:20]:
        lines.append(f"- {r['group']}: mean delta `{r['mean_delta']}`, seed62064 `{r['seed62064_delta']}`, seed62065 `{r['seed62065_delta']}`, n `{r.get('n')}`.")
    lines.append("\n## Other completed-column largest grouped movements\n")
    for name in ["Supplement_by_subtask", "EWoK_by_subtask", "COMPS_by_subtask", "GlobalPIQA_parallel_by_subtask", "GlobalPIQA_nonparallel_by_subtask"]:
        lines.append(f"### {name}")
        for r in group_profiles.get(name, [])[:12]:
            lines.append(f"- {r['group']}: mean delta `{r['mean_delta']}`, seed62064 `{r['seed62064_delta']}`, seed62065 `{r['seed62065_delta']}`, same sign `{r['same_sign']}`, n `{r.get('n')}`.")
    lines.append("\n## Scientific use\n")
    lines.append("The completed official-sized non-SuperGLUE surface shows two-seed reproducible gains in Entity operation depths 2–5, COMPS base/wugs-dist-before, GlobalPIQA, and Reading, with stable costs in BLiMP families and EWoK. If terminal official results fail only through localized costs, these rows identify what the dense-mask/sparse-label control should try to preserve or reduce.")
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "out_json": rel(out_json), "out_md": rel(out_md)}, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
