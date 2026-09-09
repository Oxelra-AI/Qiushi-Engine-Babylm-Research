#!/usr/bin/env python3
"""research: localize the reproducible dense-focus movement before official jobs finish.

The official-compatible seed62064 evaluation is still in progress. This script
therefore restricts itself to the columns already recorded as completed in research
(BLiMP, Supplement, EWoK, Entity) and to frozen fast-screen payloads built in Steps
073/075.  It does not inspect or infer newly written unfinished columns.

Scientific purpose: identify whether dense unchanged-Qwen focus produces a coherent
structured gain/cost profile that should guide the next training contrast if the
full official Overall is not a simple win/loss.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import json
import math
import pathlib
import re
import sys
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Tuple

SCRIPT = _public_path('experiments/archive/functional_learning/scripts/dense_profile_localization.py')
ROOT = _public_path('.')
STUDY = _public_path('experiments/archive/functional_learning')
SCRIPTS = _public_path('experiments/archive/functional_learning/scripts')
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import official_transition_compare as cmp  # noqa: E402

PARENT_OFFICIAL_ZERO = "experiments/archive/frontier_consolidation/data/private_scale_sentinel_eval/coherent86_private_scale_0p75/per_target/coherent86_private_scale_0p75.json"
PARENT_OFFICIAL_SG = "experiments/archive/frontier_consolidation/data/private_scale_superglue_eval/coherent86_private_scale_0p75/per_target/coherent86_private_scale_0p75_sg_retry.json"
DENSE64_OFFICIAL = "experiments/archive/functional_learning/data/dense_focus_official_eval_seed62064/per_target/dense_focus_seed62064_u0080.json"
PARENT_FAST = "experiments/archive/functional_learning/data/dense_fast_payloads/coherent86_alpha075_fast_payload.json"
DENSE64_FAST = "experiments/archive/functional_learning/data/dense_fast_payloads/dense_focus_seed62064_u0080_fast_payload.json"
DENSE65_FAST = "experiments/archive/functional_learning/data/dense_seed62065_fast_payloads/dense_focus_seed62065_u0080_fast_payload.json"
FAST_OVERLAP = "experiments/archive/functional_learning/data/dense_seed_item_overlap/dense_seed_item_overlap.json"
OUT = _public_path('experiments/archive/functional_learning/data/dense_profile_localization')
KNOWN_OFFICIAL_COLS = ["BLiMP", "Supplement", "EWoK", "Entity"]


def rel(path: pathlib.Path | str) -> str:
    try:
        return str(pathlib.Path(path).resolve().relative_to(ROOT))
    except Exception:
        return str(path)


def load_json(path: str | pathlib.Path) -> Any:
    p = pathlib.Path(path)
    if not p.is_absolute():
        p = ROOT / p
    return json.loads(p.read_text(encoding="utf-8"))


def items_for_known_column(payload: Dict[str, Any], col: str) -> List[Dict[str, Any]]:
    # Keep this intentionally narrow to avoid reading unfinished columns from the
    # in-progress official job.
    if col == "BLiMP":
        return cmp.blimp_like_items(payload, "BLiMP", cmp.PRISTINE_FULL / "blimp_filtered")
    if col == "Supplement":
        return cmp.blimp_like_items(payload, "Supplement", cmp.PRISTINE_FULL / "supplement_filtered")
    if col == "EWoK":
        return cmp.ewok_items(payload)
    if col == "Entity":
        return cmp.entity_items(payload)
    raise KeyError(col)


def subgroup_rows(rows: List[Dict[str, Any]], key_fn) -> Dict[str, List[Dict[str, Any]]]:
    d: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        d[str(key_fn(r))].append(r)
    return dict(d)


def keyed(rows: Iterable[Dict[str, Any]]) -> Dict[Tuple[str, int, str], Dict[str, Any]]:
    return {cmp.key_item(r): r for r in rows}


def group_compare(a_rows: List[Dict[str, Any]], b_rows: List[Dict[str, Any]], key_fn) -> List[Dict[str, Any]]:
    amap = keyed(a_rows)
    bmap = keyed(b_rows)
    keys = sorted(set(amap) & set(bmap))
    buckets: Dict[str, Dict[str, Any]] = defaultdict(lambda: {"n": 0, "a_correct": 0, "b_correct": 0, "a_only_correct": 0, "b_only_correct": 0})
    for k in keys:
        ar, br = amap[k], bmap[k]
        g = str(key_fn(ar))
        ac, bc = bool(ar["correct"]), bool(br["correct"])
        rec = buckets[g]
        rec["n"] += 1
        rec["a_correct"] += int(ac)
        rec["b_correct"] += int(bc)
        rec["a_only_correct"] += int(ac and not bc)
        rec["b_only_correct"] += int((not ac) and bc)
    out = []
    for g, rec in buckets.items():
        n = int(rec["n"])
        a_acc = 100.0 * rec["a_correct"] / n if n else None
        b_acc = 100.0 * rec["b_correct"] / n if n else None
        out.append({
            "group": g,
            "n": n,
            "a_acc": a_acc,
            "b_acc": b_acc,
            "delta_b_minus_a": None if a_acc is None or b_acc is None else b_acc - a_acc,
            "net_item_delta": int(rec["b_correct"] - rec["a_correct"]),
            "a_only_correct": int(rec["a_only_correct"]),
            "b_only_correct": int(rec["b_only_correct"]),
        })
    out.sort(key=lambda x: (abs(float(x["delta_b_minus_a"] or 0.0)), x["n"]), reverse=True)
    return out


def subtask_delta_map(a_rows: List[Dict[str, Any]], b_rows: List[Dict[str, Any]]) -> Dict[str, float]:
    return {r["group"]: float(r["delta_b_minus_a"]) for r in group_compare(a_rows, b_rows, lambda x: x["subtask"]) if r["delta_b_minus_a"] is not None}


def pearson(xs: List[float], ys: List[float]) -> float | None:
    if len(xs) < 2:
        return None
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx <= 0 or vy <= 0:
        return None
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / math.sqrt(vx * vy)


def concordance(official: Dict[str, float], fast: Dict[str, float]) -> Dict[str, Any]:
    shared = sorted(set(official) & set(fast))
    xs = [fast[k] for k in shared]
    ys = [official[k] for k in shared]
    nz = [(x, y) for x, y in zip(xs, ys) if abs(x) > 1e-12 or abs(y) > 1e-12]
    same = sum(1 for x, y in nz if (x == 0 and y == 0) or (x > 0 and y > 0) or (x < 0 and y < 0))
    return {
        "n_shared_subtasks": len(shared),
        "pearson_fast_vs_official": pearson(xs, ys),
        "same_sign_fraction_nonzero": (same / len(nz)) if nz else None,
        "largest_disagreements": sorted(
            [{"subtask": k, "fast_delta": fast[k], "official_delta": official[k], "abs_difference": abs(fast[k] - official[k])} for k in shared],
            key=lambda r: r["abs_difference"], reverse=True
        )[:12],
    }


def blimp_family(subtask: str) -> str:
    s = subtask.lower()
    if "agreement" in s or "determiner_noun" in s:
        return "agreement_and_number"
    if any(x in s for x in ["island", "wh_", "wh", "coordinate_structure", "left_branch", "complex_np"]):
        return "extraction_and_islands"
    if any(x in s for x in ["npi", "negation", "only", "quantifier", "existential_there"]):
        return "quantifier_npi_scope"
    if any(x in s for x in ["principle_a", "anaphor"]):
        return "binding_anaphora"
    if any(x in s for x in ["transitive", "passive", "inchoative", "causative", "drop_argument", "ellipsis", "raising", "tough"]):
        return "argument_structure_ellipsis"
    return "other_blimp"


def entity_family_depth(subtask: str) -> str:
    m = re.match(r"(.+)_([0-9]+)_ops$", subtask)
    if not m:
        return subtask
    return f"{m.group(1)}/{m.group(2)}_ops"


def entity_family(subtask: str) -> str:
    m = re.match(r"(.+)_([0-9]+)_ops$", subtask)
    return m.group(1) if m else subtask


def entity_depth(subtask: str) -> str:
    m = re.match(r"(.+)_([0-9]+)_ops$", subtask)
    return f"{m.group(2)}_ops" if m else subtask


def official_thresholds(score_rows: List[Dict[str, Any]], fast64_final: Dict[str, float], fast65_final: Dict[str, float]) -> Dict[str, Any]:
    known = {}
    for r in score_rows:
        col = r["column"]
        if col in KNOWN_OFFICIAL_COLS and r.get("computed_delta_b_minus_a") is not None:
            known[col] = float(r["computed_delta_b_minus_a"])
    known_sum = sum(known.values())
    need_unknown_sum_to_tie = -known_sum
    unknown_cols = ["COMPS", "GlobalPIQA", "Reading", "SuperGLUE"]
    def non_sg_fast(final: Dict[str, float]) -> float:
        return float(final.get("COMPS", 0.0) + final.get("GlobalPIQA", 0.0) + final.get("Reading", 0.0))
    f64 = non_sg_fast(fast64_final)
    f65 = non_sg_fast(fast65_final)
    return {
        "known_official_completed_deltas": known,
        "known_delta_sum": known_sum,
        "unknown_columns_for_aoa0_overall": unknown_cols,
        "unknown_sum_needed_to_tie_coherent86": need_unknown_sum_to_tie,
        "unknown_average_needed_to_tie": need_unknown_sum_to_tie / len(unknown_cols),
        "fast_seed62064_non_superglue_unknown_proxy_sum": f64,
        "fast_seed62064_superglue_drop_tolerated_if_proxy_exact": f64 - need_unknown_sum_to_tie,
        "fast_seed62065_non_superglue_unknown_proxy_sum": f65,
        "fast_seed62065_superglue_drop_tolerated_if_proxy_exact": f65 - need_unknown_sum_to_tie,
        "note": "Threshold arithmetic only; official unfinished columns must be read from terminal payloads, not substituted by fast proxies.",
    }


def score_final_deltas(trans: Dict[str, Any]) -> Dict[str, float]:
    rows = trans["score_summary"]["rows"]
    out = {}
    for r in rows:
        c = r["column"]
        if r.get("computed_delta_b_minus_a") is not None:
            out[c] = float(r["computed_delta_b_minus_a"])
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    parent_official = cmp.load_payloads([PARENT_OFFICIAL_ZERO])
    dense64_official = cmp.load_payloads([DENSE64_OFFICIAL])
    parent_fast = cmp.load_payloads([PARENT_FAST])
    dense64_fast = cmp.load_payloads([DENSE64_FAST])
    dense65_fast = cmp.load_payloads([DENSE65_FAST])

    fast64_transition = load_json("experiments/archive/functional_learning/data/dense_fast_transition_compare/dense_vs_coherent86_fast_official_transition.json")
    fast65_transition = load_json("experiments/archive/functional_learning/data/dense_seed62065_vs_coherent86_fast_transition/dense62065_vs_coherent86_fast_official_transition.json")
    fast_overlap = load_json(FAST_OVERLAP)
    fast64_deltas = score_final_deltas(fast64_transition)
    fast65_deltas = score_final_deltas(fast65_transition)

    official_col = {}
    official_vs_fast = {}
    group_summaries = {}
    for col in KNOWN_OFFICIAL_COLS:
        ao = items_for_known_column(parent_official, col)
        bo = items_for_known_column(dense64_official, col)
        official_col[col] = cmp.compare_standard_column(ao, bo, max_examples=8)

        af = items_for_known_column(parent_fast, col)
        b64f = items_for_known_column(dense64_fast, col)
        b65f = items_for_known_column(dense65_fast, col)
        od = subtask_delta_map(ao, bo)
        f64d = subtask_delta_map(af, b64f)
        f65d = subtask_delta_map(af, b65f)
        favg = {k: (f64d[k] + f65d[k]) / 2.0 for k in set(f64d) & set(f65d)}
        official_vs_fast[col] = {
            "seed62064_fast_vs_official": concordance(od, f64d),
            "seed62065_fast_vs_official": concordance(od, f65d),
            "two_seed_fast_average_vs_official": concordance(od, favg),
        }

        if col == "BLiMP":
            group_summaries["BLiMP_by_scientific_family"] = group_compare(ao, bo, lambda r: blimp_family(str(r["subtask"])))
        elif col == "Entity":
            group_summaries["Entity_by_task_and_depth"] = group_compare(ao, bo, lambda r: entity_family_depth(str(r["subtask"])))
            group_summaries["Entity_by_task_family"] = group_compare(ao, bo, lambda r: entity_family(str(r["subtask"])))
            group_summaries["Entity_by_operation_depth"] = group_compare(ao, bo, lambda r: entity_depth(str(r["subtask"])))
        elif col == "EWoK":
            group_summaries["EWoK_by_domain"] = group_compare(ao, bo, lambda r: str(r["subtask"]))
        elif col == "Supplement":
            group_summaries["Supplement_by_subtask"] = group_compare(ao, bo, lambda r: str(r["subtask"]))

    score_rows = []
    for col in KNOWN_OFFICIAL_COLS:
        rec = official_col[col]
        score_rows.append({
            "column": col,
            "parent_official_macro": rec.get("a_accuracy_official_macro"),
            "dense64_official_macro": rec.get("b_accuracy_official_macro"),
            "computed_delta_b_minus_a": rec.get("delta_accuracy_official_macro_b_minus_a"),
            "net_item_delta": rec.get("net_item_delta_b_minus_a"),
            "n_common": rec.get("n_common"),
        })

    thresholds = official_thresholds(score_rows, fast64_deltas, fast65_deltas)

    # Fast seed-stable cost/gain localization.  Use by-subtask records produced in
    # research and retain only the strongest shared movements.
    stable_fast_subtasks = []
    for k, rec in (fast_overlap.get("by_subtask") or {}).items():
        gl = rec.get("gain_loss_counts_vs_parent") or {}
        stable_fast_subtasks.append({
            "subtask": k,
            "n": rec.get("n"),
            "parent_acc": (rec.get("accuracies") or {}).get("parent"),
            "seed62064_acc": (rec.get("accuracies") or {}).get("seed62064"),
            "seed62065_acc": (rec.get("accuracies") or {}).get("seed62065"),
            "net_shared_item_delta": gl.get("net_shared_item_delta"),
            "shared_gains": gl.get("shared_gains"),
            "shared_losses": gl.get("shared_losses"),
            "seed_agreement_fraction": rec.get("seed_agreement_fraction"),
        })
    stable_fast_subtasks.sort(key=lambda r: abs(int(r.get("net_shared_item_delta") or 0)), reverse=True)

    result = {
        "status": "DENSE_PROFILE_LOCALIZATION",
        "script": rel(SCRIPT),
        "inputs": {
            "known_official_dense64_payload": DENSE64_OFFICIAL,
            "known_official_columns_used_only": KNOWN_OFFICIAL_COLS,
            "parent_official_zero": PARENT_OFFICIAL_ZERO,
            "parent_official_superglue_reference": PARENT_OFFICIAL_SG,
            "fast_payloads": {"parent": PARENT_FAST, "seed62064": DENSE64_FAST, "seed62065": DENSE65_FAST},
            "fast_overlap": FAST_OVERLAP,
        },
        "official_completed_profile_seed62064_vs_coherent86": {
            "score_rows": score_rows,
            "thresholds_for_pending_full_overall": thresholds,
            "by_column": official_col,
            "group_summaries": group_summaries,
        },
        "fast_to_official_subtask_concordance": official_vs_fast,
        "two_seed_fast_stable_subtasks_top_abs_shared_net": stable_fast_subtasks[:60],
        "interpretation": {
            "frontier_status": "No v5 conclusion is made here; official-compatible background jobs remain authoritative for terminal scores.",
            "main_scientific_read": "Dense focus has seed-stable fast and trained-material source-conditioned movement. The available full-official columns show an Entity gain but BLiMP/Supplement/EWoK costs; the pending columns decide whether that structured effect is net beneficial.",
            "if_full_profile_keeps_gains_but_overall_fails": "Use the dense-mask/sparse-label training contrast to test whether local-clue removal can preserve evidence-responsive/entity gains without the same supervised-coverage cost profile, rather than starting a coefficient sweep.",
            "if_full_gains_disappear": "Treat dense as mostly trained-material/fast-surface movement and reduce investment unless another independent endpoint supports it.",
        },
    }
    out_json = _public_path('experiments/archive/functional_learning/data/dense_profile_localization/dense_profile_localization.json')
    out_md = _public_path('research/documents/functional_learning/data/dense_profile_localization/dense_profile_localization.md')
    out_json.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    lines = []
    lines.append("# research dense-focus profile localization\n")
    lines.append("This readout uses only research-known official seed62064 columns (BLiMP, Supplement, EWoK, Entity) plus frozen fast-screen payloads. It does not infer unfinished official columns.\n")
    lines.append("## Completed official-column profile: dense seed62064 vs coherent86\n")
    for r in score_rows:
        lines.append(f"- {r['column']}: parent `{r['parent_official_macro']:.6f}`, dense `{r['dense64_official_macro']:.6f}`, delta `{r['computed_delta_b_minus_a']:.6f}`, net items `{r['net_item_delta']}` / n `{r['n_common']}`.")
    lines.append("")
    lines.append("## Pending-official arithmetic\n")
    th = thresholds
    lines.append(f"Known completed-column delta sum: `{th['known_delta_sum']:.6f}`. The four unresolved non-AoA columns (COMPS, GlobalPIQA, Reading, SuperGLUE) need total delta `>{th['unknown_sum_needed_to_tie_coherent86']:.6f}` (average `>{th['unknown_average_needed_to_tie']:.6f}`) for the AoA0 projected Overall to exceed coherent86.")
    lines.append(f"Fast seed62064 proxy for COMPS+GlobalPIQA+Reading is `{th['fast_seed62064_non_superglue_unknown_proxy_sum']:.6f}`, which would tolerate SuperGLUE delta down to about `-{th['fast_seed62064_superglue_drop_tolerated_if_proxy_exact']:.6f}` if the proxy were exact. This is only arithmetic context; terminal official payloads dominate.")
    lines.append("")
    lines.append("## Entity localization in the completed official payload\n")
    for r in group_summaries.get("Entity_by_operation_depth", []):
        lines.append(f"- depth {r['group']}: delta `{r['delta_b_minus_a']:.4f}`, parent `{r['a_acc']:.4f}`, dense `{r['b_acc']:.4f}`, net items `{r['net_item_delta']}`, n `{r['n']}`.")
    lines.append("")
    lines.append("## BLiMP completed-payload family movement\n")
    for r in group_summaries.get("BLiMP_by_scientific_family", []):
        lines.append(f"- {r['group']}: delta `{r['delta_b_minus_a']:.4f}`, parent `{r['a_acc']:.4f}`, dense `{r['b_acc']:.4f}`, net `{r['net_item_delta']}`, n `{r['n']}`.")
    lines.append("")
    lines.append("## Fast-to-official subtask concordance\n")
    for col, rec in official_vs_fast.items():
        avg = rec["two_seed_fast_average_vs_official"]
        lines.append(f"- {col}: shared subtasks `{avg['n_shared_subtasks']}`, Pearson(two-seed fast avg, official) `{avg['pearson_fast_vs_official']}`, same-sign fraction `{avg['same_sign_fraction_nonzero']}`.")
    lines.append("")
    lines.append("## Strongest two-seed-stable fast subtask movements\n")
    for r in stable_fast_subtasks[:20]:
        lines.append(f"- {r['subtask']}: shared net `{r['net_shared_item_delta']}`, shared gains/losses `{r['shared_gains']}`/`{r['shared_losses']}`, seed agreement `{r['seed_agreement_fraction']}`.")
    lines.append("")
    lines.append("## Scientific use\n")
    lines.append("If full official evaluation retains Entity/source-responsive gains but BLiMP/Supplement or SuperGLUE costs erase the aggregate, the prepared dense-mask/sparse-label contrast becomes a principled repair experiment: it separates input-side clue removal from dense supervised target coverage. If the full gains themselves disappear, dense should be treated as a fast/trained-material movement with weaker practical value.")
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "out_json": rel(out_json), "out_md": rel(out_md)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
