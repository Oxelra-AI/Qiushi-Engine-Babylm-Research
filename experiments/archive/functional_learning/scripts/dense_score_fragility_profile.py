#!/usr/bin/env python3
"""research: dense-focus score fragility and GlobalPIQA dependence profile.

This CPU-only analysis separates two questions that became entangled after the
seed62064/seed62065 official-sized zero-shot/Reading readouts:

1. Frontier arithmetic: does the legal checkpoint beat coherent86 on the official
   coordinate when SuperGLUE and AoA are complete?
2. Scientific method: did dense unchanged-Qwen focus produce a broadly useful
   learning change, or mainly a stable redistribution whose small aggregate surplus
   is carried by a tiny evaluation slice?

The script uses only already-written zero-shot/Reading payloads.  It does not read
external process state and it does not treat current SuperGLUE fields as complete.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import collections
import json
import math
import pathlib
import statistics
import sys
import time
from typing import Any, Dict, Iterable, List, Tuple

try:
    from scipy.stats import binomtest as scipy_binomtest
except Exception:  # pragma: no cover - fallback only
    scipy_binomtest = None


def find_root() -> pathlib.Path:
    return _PUBLIC_ROOT


ROOT = find_root()
STUDY = ROOT / "experiments/archive/functional_learning"
SCRIPT_DIR = STUDY / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))
import official_transition_compare as cmp  # noqa: E402

PARENT_PAYLOAD = ROOT / "experiments/archive/frontier_consolidation/data/private_scale_sentinel_eval/coherent86_private_scale_0p75/per_target/coherent86_private_scale_0p75.json"
SEED64_PAYLOAD = STUDY / "data/dense_focus_official_eval_seed62064/per_target/dense_focus_seed62064_u0080.json"
SEED65_PAYLOAD = STUDY / "data/dense_focus_official_eval_seed62065/per_target/dense_focus_seed62065_u0080.json"
INPUT_LABEL_PROFILE = STUDY / "data/dense_input_label_profile/dense_input_label_profile.json"
SEED64_COMMON_TARGET = STUDY / "data/unchanged_dense_focus_eval/common_target/summary_unchanged_correspondence_focus_weighted_u0080.json"
SEED65_COMMON_TARGET = STUDY / "data/dense_focus_rep_seed62065_eval/common_target/summary_unchanged_correspondence_focus_weighted_u0080.json"
SEED64_QWEN_VIEW = STUDY / "data/unchanged_dense_focus_eval/qwen_view_surface/summary.json"
SEED65_QWEN_VIEW = STUDY / "data/dense_focus_rep_seed62065_eval/qwen_view_surface/summary.json"
OUT_DIR = STUDY / "data/dense_score_fragility_profile"

CHEAP7 = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA", "Reading"]
OTHER6 = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "Reading"]
NON_GP_NON_ENTITY5 = ["BLiMP", "Supplement", "EWoK", "COMPS", "Reading"]
GRAMMAR_KNOWLEDGE3 = ["BLiMP", "Supplement", "EWoK"]
ITEM_COLUMNS = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA_parallel", "GlobalPIQA_nonparallel"]


def rel(path: pathlib.Path | str) -> str:
    try:
        return str(pathlib.Path(path).resolve().relative_to(ROOT))
    except Exception:
        return str(path)


def load_json(path: pathlib.Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def payload_scores(payload: dict[str, Any], items: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    scores = cmp.scores_from_payload(payload)
    comp = cmp.computed_scores_for_payload(items)
    # Reading is not reconstructed from item files by the comparator.
    if "Reading" in scores and "Reading" not in comp:
        comp["Reading"] = scores["Reading"]
    if all(k in comp for k in CHEAP7):
        comp["cheap7_mean"] = sum(float(comp[k]) for k in CHEAP7) / len(CHEAP7)
    return {"payload": scores, "computed": comp}


def delta_table(parent_scores: dict[str, float], child_scores: dict[str, float], keys: list[str]) -> list[dict[str, Any]]:
    rows = []
    for k in keys:
        av = parent_scores.get(k)
        bv = child_scores.get(k)
        rows.append({"column": k, "parent": av, "dense": bv, "delta": None if av is None or bv is None else bv - av})
    return rows


def sum_delta(parent_scores: dict[str, float], child_scores: dict[str, float], keys: list[str]) -> float | None:
    if not all(k in parent_scores and k in child_scores for k in keys):
        return None
    return sum(float(child_scores[k]) - float(parent_scores[k]) for k in keys)


def key_item_full(row: dict[str, Any]) -> tuple[str, str, int, str]:
    return (str(row["column"]), str(row["subtask"]), int(row["index"]), str(row.get("id", "")))


def as_map(rows: Iterable[dict[str, Any]]) -> dict[tuple[str, str, int, str], dict[str, Any]]:
    return {key_item_full(r): r for r in rows}


def exact_binom_two_sided(discordant_successes: int, discordant_failures: int) -> float | None:
    # Two-sided sign/binomial test for matched gains vs losses under p=0.5.
    # scipy handles both small GlobalPIQA counts and large BLiMP/COMPS counts
    # without overflowing.  The fallback is only for small n.
    n = int(discordant_successes + discordant_failures)
    if n <= 0:
        return None
    if scipy_binomtest is not None:
        return float(scipy_binomtest(int(discordant_successes), n, p=0.5, alternative="two-sided").pvalue)
    if n > 1000:
        z = (abs(float(discordant_successes - discordant_failures)) - 1.0) / math.sqrt(n)
        return float(math.erfc(z / math.sqrt(2.0)))
    k = min(int(discordant_successes), int(discordant_failures))
    prob_tail = sum(math.comb(n, i) for i in range(k + 1)) / (2.0 ** n)
    return min(1.0, 2.0 * prob_tail)


def transition_summary(parent_items: dict[str, list[dict[str, Any]]], child_items: dict[str, list[dict[str, Any]]], columns: list[str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for col in columns:
        pmap = as_map(parent_items.get(col, []))
        cmap = as_map(child_items.get(col, []))
        keys = sorted(set(pmap) & set(cmap))
        gains = losses = both_correct = both_wrong = 0
        for k in keys:
            pc = bool(pmap[k]["correct"])
            cc = bool(cmap[k]["correct"])
            if pc and cc:
                both_correct += 1
            elif pc and not cc:
                losses += 1
            elif (not pc) and cc:
                gains += 1
            else:
                both_wrong += 1
        out[col] = {
            "n_common": len(keys),
            "parent_correct": both_correct + losses,
            "child_correct": both_correct + gains,
            "both_correct": both_correct,
            "parent_only_correct_losses": losses,
            "child_only_correct_gains": gains,
            "both_wrong": both_wrong,
            "net_item_delta_child_minus_parent": gains - losses,
            "discordant": gains + losses,
            "mcnemar_exact_two_sided_p": exact_binom_two_sided(gains, losses),
        }
    # Add combined GlobalPIQA item count without reweighting across parallel/nonparallel.
    gp_rows = [out[c] for c in ["GlobalPIQA_parallel", "GlobalPIQA_nonparallel"] if c in out]
    if len(gp_rows) == 2:
        gains = sum(r["child_only_correct_gains"] for r in gp_rows)
        losses = sum(r["parent_only_correct_losses"] for r in gp_rows)
        n_common = sum(r["n_common"] for r in gp_rows)
        parent_correct = sum(r["parent_correct"] for r in gp_rows)
        child_correct = sum(r["child_correct"] for r in gp_rows)
        out["GlobalPIQA_combined_micro"] = {
            "n_common": n_common,
            "parent_correct": parent_correct,
            "child_correct": child_correct,
            "parent_accuracy_micro": 100.0 * parent_correct / n_common,
            "child_accuracy_micro": 100.0 * child_correct / n_common,
            "child_only_correct_gains": gains,
            "parent_only_correct_losses": losses,
            "net_item_delta_child_minus_parent": gains - losses,
            "discordant": gains + losses,
            "mcnemar_exact_two_sided_p": exact_binom_two_sided(gains, losses),
        }
    return out


def two_seed_patterns(parent_items: dict[str, list[dict[str, Any]]], s64_items: dict[str, list[dict[str, Any]]], s65_items: dict[str, list[dict[str, Any]]], columns: list[str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for col in columns:
        pmap = as_map(parent_items.get(col, []))
        m64 = as_map(s64_items.get(col, []))
        m65 = as_map(s65_items.get(col, []))
        keys = sorted(set(pmap) & set(m64) & set(m65))
        patterns: collections.Counter[str] = collections.Counter()
        for k in keys:
            pat = f"{int(bool(pmap[k]['correct']))}{int(bool(m64[k]['correct']))}{int(bool(m65[k]['correct']))}"
            patterns[pat] += 1
        shared_gain = patterns.get("011", 0)
        shared_loss = patterns.get("100", 0)
        seed64_only_gain = patterns.get("010", 0)
        seed65_only_gain = patterns.get("001", 0)
        seed64_only_loss = patterns.get("101", 0)
        seed65_only_loss = patterns.get("110", 0)
        out[col] = {
            "n_common_threeway": len(keys),
            "patterns_parent_seed64_seed65": dict(sorted(patterns.items())),
            "shared_dense_gains_parent_wrong_both_dense_correct": shared_gain,
            "shared_dense_losses_parent_correct_both_dense_wrong": shared_loss,
            "shared_net_item_delta": shared_gain - shared_loss,
            "seed64_specific_gain_pattern010": seed64_only_gain,
            "seed65_specific_gain_pattern001": seed65_only_gain,
            "seed64_specific_loss_pattern101": seed64_only_loss,
            "seed65_specific_loss_pattern110": seed65_only_loss,
            "dense_seed_agreement_fraction": (patterns.get("000", 0) + patterns.get("011", 0) + patterns.get("100", 0) + patterns.get("111", 0)) / len(keys) if keys else None,
        }
    if all(c in out for c in ["GlobalPIQA_parallel", "GlobalPIQA_nonparallel"]):
        merged = collections.Counter()
        n = 0
        for c in ["GlobalPIQA_parallel", "GlobalPIQA_nonparallel"]:
            merged.update(out[c]["patterns_parent_seed64_seed65"])
            n += out[c]["n_common_threeway"]
        shared_gain = merged.get("011", 0)
        shared_loss = merged.get("100", 0)
        out["GlobalPIQA_combined_micro"] = {
            "n_common_threeway": n,
            "patterns_parent_seed64_seed65": dict(sorted(merged.items())),
            "shared_dense_gains_parent_wrong_both_dense_correct": shared_gain,
            "shared_dense_losses_parent_correct_both_dense_wrong": shared_loss,
            "shared_net_item_delta": shared_gain - shared_loss,
            "dense_seed_agreement_fraction": (merged.get("000", 0) + merged.get("011", 0) + merged.get("100", 0) + merged.get("111", 0)) / n if n else None,
        }
    return out


def load_globalpiqa_data(kind: str) -> dict[str, dict[str, Any]]:
    if kind == "parallel":
        ddir = ROOT / "experiments/archive/initial_model_studies/repos/babylm-eval/strict/evaluation_data/full_eval/global_piqa_parallel"
    elif kind == "nonparallel":
        ddir = ROOT / "experiments/archive/initial_model_studies/repos/babylm-eval/strict/evaluation_data/full_eval/global_piqa_nonparallel"
    else:
        raise ValueError(kind)
    return {r["example_id"]: r for r in read_jsonl(ddir / "eng_latn.jsonl")}


def parse_supplement(x: Any) -> Any:
    if not isinstance(x, str):
        return x
    try:
        return json.loads(x)
    except Exception:
        return x


def globalpiqa_flip_records(parent_items: dict[str, list[dict[str, Any]]], s64_items: dict[str, list[dict[str, Any]]], s65_items: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    data = {
        "GlobalPIQA_parallel": load_globalpiqa_data("parallel"),
        "GlobalPIQA_nonparallel": load_globalpiqa_data("nonparallel"),
    }
    out: dict[str, list[dict[str, Any]]] = {
        "shared_dense_gains": [],
        "shared_dense_losses": [],
        "seed_specific_changes": [],
        "all_discordant": [],
    }
    for col in ["GlobalPIQA_parallel", "GlobalPIQA_nonparallel"]:
        pmap = as_map(parent_items[col])
        m64 = as_map(s64_items[col])
        m65 = as_map(s65_items[col])
        for k in sorted(set(pmap) & set(m64) & set(m65)):
            p, a, b = pmap[k], m64[k], m65[k]
            pc, ac, bc = bool(p["correct"]), bool(a["correct"]), bool(b["correct"])
            if pc == ac == bc:
                continue
            eid = p["id"]
            gold = data[col].get(eid, {})
            label = gold.get("label")
            rec = {
                "column": col,
                "example_id": eid,
                "pattern_parent_seed64_seed65": f"{int(pc)}{int(ac)}{int(bc)}",
                "parent_correct": pc,
                "seed62064_correct": ac,
                "seed62065_correct": bc,
                "prompt": gold.get("prompt"),
                "target": p.get("target"),
                "parent_pred": p.get("pred"),
                "seed62064_pred": a.get("pred"),
                "seed62065_pred": b.get("pred"),
                "label": label,
                "options": {f"solution{i}": gold.get(f"solution{i}") for i in range(4)},
                "categories": gold.get("categories"),
                "supplement": parse_supplement(gold.get("supplement")),
            }
            out["all_discordant"].append(rec)
            if (not pc) and ac and bc:
                out["shared_dense_gains"].append(rec)
            elif pc and (not ac) and (not bc):
                out["shared_dense_losses"].append(rec)
            else:
                out["seed_specific_changes"].append(rec)
    return out


def score_decomposition(parent_score_pack: dict[str, Any], child_score_pack: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for source in ["payload", "computed"]:
        p = parent_score_pack[source]
        c = child_score_pack[source]
        cheap_sum = sum_delta(p, c, CHEAP7)
        gp_delta = sum_delta(p, c, ["GlobalPIQA"])
        other6 = sum_delta(p, c, OTHER6)
        non_gp_non_entity5 = sum_delta(p, c, NON_GP_NON_ENTITY5)
        grammar3 = sum_delta(p, c, GRAMMAR_KNOWLEDGE3)
        out[source] = {
            "column_deltas": delta_table(p, c, CHEAP7),
            "cheap7_sum_delta": cheap_sum,
            "cheap7_mean_delta": None if cheap_sum is None else cheap_sum / 7.0,
            "globalpiqa_delta": gp_delta,
            "globalpiqa_fraction_of_cheap7_sum_delta": None if cheap_sum in (None, 0) or gp_delta is None else gp_delta / cheap_sum,
            "other6_excluding_globalpiqa_sum_delta": other6,
            "other6_excluding_globalpiqa_mean_delta": None if other6 is None else other6 / 6.0,
            "non_globalpiqa_non_entity5_sum_delta": non_gp_non_entity5,
            "non_globalpiqa_non_entity5_mean_delta": None if non_gp_non_entity5 is None else non_gp_non_entity5 / 5.0,
            "grammar_knowledge3_sum_delta": grammar3,
            "grammar_knowledge3_mean_delta": None if grammar3 is None else grammar3 / 3.0,
        }
    return out


def get_nested(d: dict[str, Any], path: list[str], default: Any = None) -> Any:
    cur: Any = d
    for p in path:
        if not isinstance(cur, dict) or p not in cur:
            return default
        cur = cur[p]
    return cur


def mechanism_summary() -> dict[str, Any]:
    p = load_json(INPUT_LABEL_PROFILE)
    c64 = load_json(SEED64_COMMON_TARGET)
    c65 = load_json(SEED65_COMMON_TARGET)
    q64 = load_json(SEED64_QWEN_VIEW)
    q65 = load_json(SEED65_QWEN_VIEW)
    return {
        "training_geometry": {
            "prefix_info": p.get("prefix_info"),
            "sparse_seed62064": p["profiles"].get("sparse_seed62064"),
            "dense_seed62064": p["profiles"].get("dense_seed62064"),
            "densemask_sparselabel_seed62064": p["profiles"].get("densemask_sparselabel_seed62064"),
            "geometry_contrast": {
                "sparse_focus_tokens": get_nested(p, ["profiles", "sparse_seed62064", "label_tokens"]),
                "dense_focus_tokens": get_nested(p, ["profiles", "dense_seed62064", "label_tokens"]),
                "densemask_sparse_label_focus_tokens": get_nested(p, ["profiles", "densemask_sparselabel_seed62064", "label_tokens"]),
                "densemask_sparse_label_mask_tokens": get_nested(p, ["profiles", "densemask_sparselabel_seed62064", "mask_tokens"]),
                "densemask_sparse_label_mask_only_tokens": get_nested(p, ["profiles", "densemask_sparselabel_seed62064", "mask_only_tokens"]),
            },
        },
        "qwen_view_source_help": {
            "seed62064": get_nested(q64, ["model_summaries", "unchanged_correspondence_focus_weighted_u0080", "source_help"]),
            "seed62065": get_nested(q65, ["model_summaries", "unchanged_correspondence_focus_weighted_u0080", "source_help"]),
            "seed62064_view_only_delta": get_nested(q64, ["model_summaries", "unchanged_correspondence_focus_weighted_u0080", "by_condition", "view_only", "mean_delta_nll_vs_parent"]),
            "seed62064_with_source_delta": get_nested(q64, ["model_summaries", "unchanged_correspondence_focus_weighted_u0080", "by_condition", "with_source", "mean_delta_nll_vs_parent"]),
            "seed62065_view_only_delta": get_nested(q65, ["model_summaries", "unchanged_correspondence_focus_weighted_u0080", "by_condition", "view_only", "mean_delta_nll_vs_parent"]),
            "seed62065_with_source_delta": get_nested(q65, ["model_summaries", "unchanged_correspondence_focus_weighted_u0080", "by_condition", "with_source", "mean_delta_nll_vs_parent"]),
        },
        "common_target_source_response": {
            "seed62064": {
                "mean_delta": 0.20787532545847906,
                "source_original_delta": 0.26365687021763906,
                "source_altered_delta": 0.33972819037114577,
                "held_source_delta": 0.3662232840563067,
                "no_source_delta": 0.020240915786652325,
                "both_source_conditions_correct": get_nested(c64, ["source_follow", "both_source_conditions_correct"]),
                "n": get_nested(c64, ["source_follow", "n"]),
            },
            "seed62065": {
                "mean_delta": 0.20564517308957875,
                "source_original_delta": 0.2572671786110316,
                "source_altered_delta": 0.33878437042546766,
                "held_source_delta": 0.36692036312888376,
                "no_source_delta": 0.02088397023223698,
                "both_source_conditions_correct": get_nested(c65, ["source_follow", "both_source_conditions_correct"]),
                "n": get_nested(c65, ["source_follow", "n"]),
            },
        },
    }


def fmt_float(x: Any, digits: int = 6) -> str:
    if x is None:
        return "null"
    if isinstance(x, float):
        return f"{x:.{digits}g}"
    return str(x)


def make_md(result: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# research dense-focus score fragility and interpretation profile")
    lines.append("")
    lines.append("Scope: already-written official-sized zero-shot/Reading payloads for dense seed62064 and seed62065, compared with the true coherent86/v4 reference. SuperGLUE and AoA are deliberately excluded here because the official evaluations are not yet complete at this stage and interim SuperGLUE fields may contain only partial task entries.")
    lines.append("")
    lines.append("## Seven-column arithmetic versus scientific breadth")
    for seed in ["seed62064", "seed62065"]:
        dec = result["score_decomposition"][seed]["payload"]
        lines.append(f"### {seed} payload-column deltas")
        for row in dec["column_deltas"]:
            lines.append(f"- {row['column']}: delta `{fmt_float(row['delta'], 8)}` (parent `{fmt_float(row['parent'], 8)}`, dense `{fmt_float(row['dense'], 8)}`).")
        lines.append(f"- Seven-column sum delta `{fmt_float(dec['cheap7_sum_delta'], 8)}`; mean delta `{fmt_float(dec['cheap7_mean_delta'], 8)}`.")
        lines.append(f"- GlobalPIQA delta `{fmt_float(dec['globalpiqa_delta'], 8)}`, fraction of seven-column sum `{fmt_float(dec['globalpiqa_fraction_of_cheap7_sum_delta'], 8)}`.")
        lines.append(f"- Other six columns excluding GlobalPIQA: sum `{fmt_float(dec['other6_excluding_globalpiqa_sum_delta'], 8)}`, mean `{fmt_float(dec['other6_excluding_globalpiqa_mean_delta'], 8)}`.")
        lines.append(f"- Excluding both GlobalPIQA and Entity: five-column sum `{fmt_float(dec['non_globalpiqa_non_entity5_sum_delta'], 8)}`, mean `{fmt_float(dec['non_globalpiqa_non_entity5_mean_delta'], 8)}`.")
        lines.append(f"- BLiMP+Supplement+EWoK sum `{fmt_float(dec['grammar_knowledge3_sum_delta'], 8)}`, mean `{fmt_float(dec['grammar_knowledge3_mean_delta'], 8)}`.")
        lines.append("")
    lines.append("Interpretation: the two-seed official-sized non-SuperGLUE surface preserves a real dense-induced redistribution. The positive seven-column mean is not broad in the scalar sense: GlobalPIQA carries nearly all of seed62064's positive sum and more than all of seed62065's positive sum, while the other six columns are near zero in aggregate and the five columns excluding Entity and GlobalPIQA are negative.")
    lines.append("")
    lines.append("## GlobalPIQA item-level dependence")
    patterns = result["two_seed_patterns"]["GlobalPIQA_combined_micro"]
    lines.append(f"Across both GlobalPIQA slices there are `{patterns['n_common_threeway']}` examples. Parent/seed62064/seed62065 correctness patterns are `{patterns['patterns_parent_seed64_seed65']}`.")
    lines.append(f"Shared dense gains: `{patterns['shared_dense_gains_parent_wrong_both_dense_correct']}`; shared dense losses: `{patterns['shared_dense_losses_parent_correct_both_dense_wrong']}`; shared net item delta: `{patterns['shared_net_item_delta']}`; dense-seed agreement fraction `{fmt_float(patterns['dense_seed_agreement_fraction'], 8)}`.")
    for seed in ["seed62064", "seed62065"]:
        gp = result["transition_summaries"][seed]["GlobalPIQA_combined_micro"]
        lines.append(f"- {seed}: parent correct `{gp['parent_correct']}/{gp['n_common']}`, dense correct `{gp['child_correct']}/{gp['n_common']}`, gains `{gp['child_only_correct_gains']}`, losses `{gp['parent_only_correct_losses']}`, net `{gp['net_item_delta_child_minus_parent']}`, exact paired sign-test p `{fmt_float(gp['mcnemar_exact_two_sided_p'], 6)}`.")
    lines.append("")
    lines.append("### Shared GlobalPIQA gains")
    for rec in result["globalpiqa_flips"]["shared_dense_gains"]:
        lines.append(f"- `{rec['column']}` `{rec['example_id']}` [{rec.get('categories')}]: target={json.dumps(rec['target'], ensure_ascii=False)}; parent={json.dumps(rec['parent_pred'], ensure_ascii=False)}; dense64={json.dumps(rec['seed62064_pred'], ensure_ascii=False)}; dense65={json.dumps(rec['seed62065_pred'], ensure_ascii=False)}; prompt={json.dumps(rec['prompt'], ensure_ascii=False)}")
    lines.append("")
    lines.append("### Shared GlobalPIQA losses")
    for rec in result["globalpiqa_flips"]["shared_dense_losses"]:
        lines.append(f"- `{rec['column']}` `{rec['example_id']}` [{rec.get('categories')}]: target={json.dumps(rec['target'], ensure_ascii=False)}; parent={json.dumps(rec['parent_pred'], ensure_ascii=False)}; dense64={json.dumps(rec['seed62064_pred'], ensure_ascii=False)}; dense65={json.dumps(rec['seed62065_pred'], ensure_ascii=False)}; prompt={json.dumps(rec['prompt'], ensure_ascii=False)}")
    seed_specific = result["globalpiqa_flips"]["seed_specific_changes"]
    if seed_specific:
        lines.append("")
        lines.append("### Seed-specific GlobalPIQA changes")
        for rec in seed_specific:
            lines.append(f"- `{rec['column']}` `{rec['example_id']}` pattern `{rec['pattern_parent_seed64_seed65']}` target={json.dumps(rec['target'], ensure_ascii=False)} parent={json.dumps(rec['parent_pred'], ensure_ascii=False)} dense64={json.dumps(rec['seed62064_pred'], ensure_ascii=False)} dense65={json.dumps(rec['seed62065_pred'], ensure_ascii=False)}")
    lines.append("")
    lines.append("The identical GlobalPIQA score across seeds is therefore shared predictions on the same small example set, not independent support over new evaluation cases. It remains legitimate in the official aggregate, but it cannot carry the broader method claim by itself.")
    lines.append("")
    lines.append("## Larger effect surfaces that do not rely on GlobalPIQA")
    ent64 = result["transition_summaries"]["seed62064"]["Entity"]
    ent65 = result["transition_summaries"]["seed62065"]["Entity"]
    bl64 = result["transition_summaries"]["seed62064"]["BLiMP"]
    bl65 = result["transition_summaries"]["seed62065"]["BLiMP"]
    lines.append(f"Entity: seed62064 gains/losses `{ent64['child_only_correct_gains']}/{ent64['parent_only_correct_losses']}` (net `{ent64['net_item_delta_child_minus_parent']}` over `{ent64['n_common']}`); seed62065 `{ent65['child_only_correct_gains']}/{ent65['parent_only_correct_losses']}` (net `{ent65['net_item_delta_child_minus_parent']}`).")
    lines.append(f"BLiMP: seed62064 gains/losses `{bl64['child_only_correct_gains']}/{bl64['parent_only_correct_losses']}` (net `{bl64['net_item_delta_child_minus_parent']}` over `{bl64['n_common']}`); seed62065 `{bl65['child_only_correct_gains']}/{bl65['parent_only_correct_losses']}` (net `{bl65['net_item_delta_child_minus_parent']}`).")
    prof_note = result["profile_pointer"]
    lines.append(f"Entity depth/family and BLiMP family localization remains in `{prof_note}`: dense helps Entity operation depths 2--5 and loses at 0_ops; BLiMP costs are spread across all coarse families rather than a single isolated subtask.")
    mech = result["mechanism_summary"]
    q64 = mech["qwen_view_source_help"]["seed62064"]
    q65 = mech["qwen_view_source_help"]["seed62065"]
    ct64 = mech["common_target_source_response"]["seed62064"]
    ct65 = mech["common_target_source_response"]["seed62065"]
    lines.append(f"Trained unchanged-Qwen source-help also reproduces: seed62064 Δsource-help `{fmt_float(q64['mean_delta_source_help_vs_parent'], 8)}` and seed62065 `{fmt_float(q65['mean_delta_source_help_vs_parent'], 8)}` over 4,783 targets / 1,200 pairs.")
    lines.append(f"Controlled common-target evidence response reproduces: seed62064 source-original/source-altered/held-source deltas `{fmt_float(ct64['source_original_delta'], 8)}`/`{fmt_float(ct64['source_altered_delta'], 8)}`/`{fmt_float(ct64['held_source_delta'], 8)}`, seed62065 `{fmt_float(ct65['source_original_delta'], 8)}`/`{fmt_float(ct65['source_altered_delta'], 8)}`/`{fmt_float(ct65['held_source_delta'], 8)}`; no-source movement stays near `0.02` in both seeds.")
    lines.append("")
    lines.append("## Consequence for the dense-mask/sparse-label control")
    geom = mech["training_geometry"]["geometry_contrast"]
    lines.append(f"Sparse focus labels/masks `{geom['sparse_focus_tokens']}` Qwen target tokens; dense focus labels/masks `{geom['dense_focus_tokens']}`; dense-mask/sparse-label would keep `{geom['densemask_sparse_label_focus_tokens']}` labels while masking `{geom['densemask_sparse_label_mask_tokens']}` content tokens, of which `{geom['densemask_sparse_label_mask_only_tokens']}` are mask-only. This directly separates input-side clue suppression from added supervised target coverage.")
    lines.append("The control should be judged by whether it preserves the large Entity and source-responsive signals while reducing BLiMP/Supplement/EWoK costs. It is scientifically valuable even if the official aggregate narrowly wins, because the benchmark surplus is currently fragile whereas the gain/cost redistribution is stable and mechanistically important.")
    lines.append("")
    lines.append("## Files")
    for k, v in result["input_paths"].items():
        lines.append(f"- {k}: `{v}`")
    return "\n".join(lines) + "\n"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    parent_payload = cmp.load_payloads([rel(PARENT_PAYLOAD)])
    seed64_payload = cmp.load_payloads([rel(SEED64_PAYLOAD)])
    seed65_payload = cmp.load_payloads([rel(SEED65_PAYLOAD)])

    parent_items = cmp.all_items(parent_payload, include_superglue=False)
    seed64_items = cmp.all_items(seed64_payload, include_superglue=False)
    seed65_items = cmp.all_items(seed65_payload, include_superglue=False)

    score_packs = {
        "parent": payload_scores(parent_payload, parent_items),
        "seed62064": payload_scores(seed64_payload, seed64_items),
        "seed62065": payload_scores(seed65_payload, seed65_items),
    }
    transitions = {
        "seed62064": transition_summary(parent_items, seed64_items, ITEM_COLUMNS),
        "seed62065": transition_summary(parent_items, seed65_items, ITEM_COLUMNS),
    }
    patterns = two_seed_patterns(parent_items, seed64_items, seed65_items, ITEM_COLUMNS)
    flips = globalpiqa_flip_records(parent_items, seed64_items, seed65_items)

    result: dict[str, Any] = {
        "status": "DENSE_SCORE_FRAGILITY_PROFILE",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "input_paths": {
            "coherent86_zero_reading_payload": rel(PARENT_PAYLOAD),
            "dense_seed62064_zero_reading_payload": rel(SEED64_PAYLOAD),
            "dense_seed62065_zero_reading_payload": rel(SEED65_PAYLOAD),
            "input_label_profile": rel(INPUT_LABEL_PROFILE),
            "seed62064_common_target": rel(SEED64_COMMON_TARGET),
            "seed62065_common_target": rel(SEED65_COMMON_TARGET),
            "seed62064_qwen_view": rel(SEED64_QWEN_VIEW),
            "seed62065_qwen_view": rel(SEED65_QWEN_VIEW),
        },
        "scope_warning": "Only zero-shot/Reading official-sized payload columns are used. SuperGLUE and AoA are not complete here; current SuperGLUE fields in dense payloads may be interim.",
        "score_decomposition": {
            "seed62064": score_decomposition(score_packs["parent"], score_packs["seed62064"]),
            "seed62065": score_decomposition(score_packs["parent"], score_packs["seed62065"]),
        },
        "transition_summaries": transitions,
        "two_seed_patterns": patterns,
        "globalpiqa_flips": flips,
        "mechanism_summary": mechanism_summary(),
        "profile_pointer": "research/documents/functional_learning/data/zero_reading_two_seed_profile/zero_reading_two_seed_profile.md",
        "scientific_interpretation": {
            "frontier": "Official aggregate is still the frontier-promotion coordinate, but complete SuperGLUE/AoA are incomplete at the time.",
            "method": "Dense unchanged-Qwen focus has reproducible Entity/source-responsive gains and reproducible grammar/knowledge costs. The current seven-column surplus depends heavily on three GlobalPIQA net decisions, so the broader learning-method question cannot be reduced to the scalar cheap7 gain.",
            "control": "Dense-mask/sparse-label is the direct next causal contrast because it separates dense input masking from dense supervised coverage and can reveal whether Entity/source-response gains can decouple from grammar costs.",
        },
    }
    out_json = OUT_DIR / "dense_score_fragility_profile.json"
    out_md = OUT_DIR / "dense_score_fragility_profile.md"
    out_json.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    out_md.write_text(make_md(result), encoding="utf-8")
    print(json.dumps({"status": result["status"], "out_json": rel(out_json), "out_md": rel(out_md), "globalpiqa_shared_net": patterns["GlobalPIQA_combined_micro"]["shared_net_item_delta"]}, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
