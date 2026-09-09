#!/usr/bin/env python3
"""research: support-floor substrate alignment before any minfreq50 H100 run.

This CPU-only reader combines existing score vectors with existing tokenizer/corpus
measurements.  It asks whether the prepared minfreq50 support-floored tokenizer
mechanically touches the UID families that the legal research endpoint lost relative
to the non-submittable old-tokenizer reference.  It does not train, evaluate, or
use evaluation text to construct training data.
"""
from __future__ import annotations

import csv
import json
import math
import pathlib
import statistics
import time
from typing import Any

STUDY = pathlib.Path("experiments/archive/frontier_consolidation")
DEFICIT_JSON = STUDY / "data/legal_deficit_subtask_decomposition/legal_deficit_subtask_decomposition.json"
FRAG_JSON = STUDY / "data/tokenizer_fragmentation_deficit_alignment/tokenizer_fragmentation_deficit_alignment.json"
SUPPORT_JSON = STUDY / "data/tokenizer_support_identity_alignment/tokenizer_support_identity_alignment.json"
FACTOR_JSON = STUDY / "data/supportfloor_factor_audit/supportfloor_factor_audit.json"
OUT_DIR = STUDY / "data/supportfloor_substrate_alignment"
NOTE = (STUDY.parents[2] / 'research/notes/frontier_consolidation/supportfloor_substrate_alignment.md')

UID_FOCUS = {
    "principle_A_reconstruction", "wh_questions_object_gap", "animate_subject_trans",
    "regular_plural_subject_verb_agreement_1", "tough_vs_raising_1", "anaphor_gender_agreement",
    "matrix_question_npi_licensor_present", "qa_congruence_easy", "qa_congruence_tricky",
    "turn_taking", "subject_aux_inversion", "hypernym", "physical-dynamics", "material-dynamics",
    "social-properties", "quantitative-properties", "spatial-relations", "material-properties",
    "social-interactions", "physical-relations", "social-relations",
}
EWOK_RELATION = {"spatial-relations", "physical-relations", "social-relations", "physical-dynamics", "material-dynamics", "physical-interactions", "social-interactions"}
EWOK_PROPERTY = {"agent-properties", "material-properties", "social-properties", "quantitative-properties"}


def now_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def load(path: pathlib.Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def key(row: dict[str, Any]) -> tuple[str, str]:
    return (str(row.get("column")), str(row.get("uid")))


def finite(x: Any) -> bool:
    return isinstance(x, (int, float)) and math.isfinite(float(x))


def mean(vals: list[Any]) -> float | None:
    xs = [float(v) for v in vals if finite(v)]
    return sum(xs) / len(xs) if xs else None


def median(vals: list[Any]) -> float | None:
    xs = sorted(float(v) for v in vals if finite(v))
    return statistics.median(xs) if xs else None


def pearson(xs: list[Any], ys: list[Any]) -> float | None:
    pairs = [(float(x), float(y)) for x, y in zip(xs, ys) if finite(x) and finite(y)]
    if len(pairs) < 3:
        return None
    xv, yv = zip(*pairs)
    mx, my = statistics.mean(xv), statistics.mean(yv)
    vx = sum((x - mx) ** 2 for x in xv)
    vy = sum((y - my) ** 2 for y in yv)
    if vx <= 0.0 or vy <= 0.0:
        return None
    return sum((x - mx) * (y - my) for x, y in pairs) / math.sqrt(vx * vy)


def ranks(vals: list[float]) -> list[float]:
    order = sorted(range(len(vals)), key=lambda i: vals[i])
    out = [0.0] * len(vals)
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and vals[order[j]] == vals[order[i]]:
            j += 1
        avg = (i + 1 + j) / 2.0
        for k in range(i, j):
            out[order[k]] = avg
        i = j
    return out


def spearman(xs: list[Any], ys: list[Any]) -> float | None:
    pairs = [(float(x), float(y)) for x, y in zip(xs, ys) if finite(x) and finite(y)]
    if len(pairs) < 3:
        return None
    return pearson(ranks([x for x, _ in pairs]), ranks([y for _, y in pairs]))


def safe_round(x: Any, nd: int = 6) -> Any:
    return round(float(x), nd) if finite(x) else None


def fmt(x: Any, nd: int = 4) -> str:
    if x is None:
        return "NA"
    if finite(x):
        return f"{float(x):.{nd}f}"
    return str(x)


def corr_block(rows: list[dict[str, Any]], metrics: list[str]) -> dict[str, Any]:
    out: dict[str, Any] = {"n_uid": len(rows)}
    legal = [r.get("legal_delta_step35_minus_old") for r in rows]
    deficit = [r.get("legal_deficit_magnitude") for r in rows]
    for m in metrics:
        vals = [r.get(m) for r in rows]
        out[m] = {
            "mean": safe_round(mean(vals)),
            "median": safe_round(median(vals)),
            "pearson_legal_delta_vs_metric": safe_round(pearson(legal, vals)),
            "spearman_legal_delta_vs_metric": safe_round(spearman(legal, vals)),
            "pearson_deficit_magnitude_vs_metric": safe_round(pearson(deficit, vals)),
            "spearman_deficit_magnitude_vs_metric": safe_round(spearman(deficit, vals)),
        }
    return out


def load_deficit_map(deficit: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    out: dict[tuple[str, str], dict[str, Any]] = {}
    for col, rows in deficit.get("columns", {}).items():
        for r in rows:
            d = r.get("delta_legal_step35")
            if d is None:
                continue
            out[(col, r["uid"])] = {
                "column": col,
                "uid": r["uid"],
                "old_ref_42033_score": r.get("old_ref_42033"),
                "legal_step35_score": r.get("legal_step35"),
                "legal_delta_step35_minus_old": d,
                "legal_deficit_magnitude": max(0.0, -float(d)),
            }
    return out


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    NOTE.parent.mkdir(parents=True, exist_ok=True)
    deficit = load(DEFICIT_JSON)
    frag = load(FRAG_JSON)
    support = load(SUPPORT_JSON)
    factor = load(FACTOR_JSON)

    base = load_deficit_map(deficit)
    frag_map = {key(r): r for r in frag.get("per_uid", [])}
    support_map = {key(r): r for r in support.get("per_uid", [])}

    rows: list[dict[str, Any]] = []
    for k, b in sorted(base.items()):
        fr = frag_map.get(k, {})
        sp = support_map.get(k, {})
        row = dict(b)
        # Token length movement on evaluation strings.
        s35_tpw = fr.get("legal_step35_16k_tokens_per_word")
        old_tpw = fr.get("old_inherited_16k_nonlegal_tokens_per_word")
        mf50_tpw = fr.get("legal_40k_minfreq50_tokens_per_word")
        mf25_tpw = fr.get("legal_40k_minfreq25_tokens_per_word")
        k40_tpw = fr.get("legal_40k_tokens_per_word")
        for name, val in [("old_tpw", old_tpw), ("tpw", s35_tpw), ("minfreq50_tpw", mf50_tpw), ("minfreq25_tpw", mf25_tpw), ("legal40k_tpw", k40_tpw)]:
            row[name] = val
        if finite(s35_tpw) and finite(mf50_tpw):
            row["minfreq50_minus_step35_tpw"] = float(mf50_tpw) - float(s35_tpw)
            row["minfreq50_token_reduction_pct_vs_step35"] = (float(s35_tpw) - float(mf50_tpw)) / float(s35_tpw) if float(s35_tpw) else None
        if finite(s35_tpw) and finite(mf25_tpw):
            row["minfreq25_minus_step35_tpw"] = float(mf25_tpw) - float(s35_tpw)
        if finite(s35_tpw) and finite(k40_tpw):
            row["legal40k_minus_step35_tpw"] = float(k40_tpw) - float(s35_tpw)
        if finite(s35_tpw) and finite(old_tpw):
            row["minus_old_tpw"] = float(s35_tpw) - float(old_tpw)
            denom = float(s35_tpw) - float(old_tpw)
            row["minfreq50_closes_step35_old_tpw_gap_frac"] = ((float(s35_tpw) - float(mf50_tpw)) / denom) if abs(denom) > 1e-12 and finite(mf50_tpw) else None

        # Support and identity movement on evaluation strings, with pool counts from allowed 10M corpus.
        support_pairs = [
            ("mean_log10_pool_count_plus1", "support_log_gain"),
            ("frac_pool_count_lt_50", "frac_lt50_delta"),
            ("frac_pool_count_lt_100", "frac_lt100_delta"),
            ("frac_pool_count_lt_200", "frac_lt200_delta"),
        ]
        for suffix, outname in support_pairs:
            a = sp.get(f"legal_step35_16k_{suffix}")
            bval = sp.get(f"legal_40k_minfreq50_{suffix}")
            row[f"step35_{suffix}"] = a
            row[f"minfreq50_{suffix}"] = bval
            if finite(a) and finite(bval):
                row[f"minfreq50_minus_step35_{outname}"] = float(bval) - float(a)
        for stem in ["token_multiset_jaccard", "b_not_in_a_frac", "a_not_in_b_frac", "a_common_frac", "b_common_frac"]:
            s35 = sp.get(f"legal_step35_16k_vs_old_{stem}")
            mf50 = sp.get(f"legal_40k_minfreq50_vs_old_{stem}")
            row[f"vs_old_{stem}"] = s35
            row[f"minfreq50_vs_old_{stem}"] = mf50
            if finite(s35) and finite(mf50):
                row[f"minfreq50_minus_step35_vs_old_{stem}"] = float(mf50) - float(s35)
        ev_s35 = sp.get("legal_step35_16k_eval_tokens")
        ev_mf50 = sp.get("legal_40k_minfreq50_eval_tokens")
        row["eval_tokens"] = ev_s35
        row["minfreq50_eval_tokens"] = ev_mf50
        if finite(ev_s35) and finite(ev_mf50):
            row["minfreq50_minus_step35_eval_tokens"] = float(ev_mf50) - float(ev_s35)
            row["minfreq50_eval_token_reduction_pct"] = (float(ev_s35) - float(ev_mf50)) / float(ev_s35) if float(ev_s35) else None
        row["in_focus_set"] = row["uid"] in UID_FOCUS
        if row["column"] == "EWoK":
            if row["uid"] in EWOK_RELATION:
                row["ewok_domain_family"] = "relation_or_dynamics"
            elif row["uid"] in EWOK_PROPERTY:
                row["ewok_domain_family"] = "property"
            else:
                row["ewok_domain_family"] = "other"
        rows.append(row)

    metrics = [
        "minfreq50_minus_step35_tpw",
        "minfreq50_token_reduction_pct_vs_step35",
        "minfreq50_closes_step35_old_tpw_gap_frac",
        "minfreq50_minus_step35_support_log_gain",
        "minfreq50_minus_step35_frac_lt50_delta",
        "minfreq50_minus_step35_frac_lt100_delta",
        "minfreq50_minus_step35_vs_old_token_multiset_jaccard",
        "minfreq50_minus_step35_vs_old_b_not_in_a_frac",
        "minfreq50_minus_step35_eval_tokens",
        "minfreq50_eval_token_reduction_pct",
        "legal40k_minus_step35_tpw",
        "minfreq25_minus_step35_tpw",
    ]
    subsets: dict[str, list[dict[str, Any]]] = {
        "ALL": rows,
        "BLiMP": [r for r in rows if r["column"] == "BLiMP"],
        "Supplement": [r for r in rows if r["column"] == "Supplement"],
        "EWoK": [r for r in rows if r["column"] == "EWoK"],
        "Worst20LegalLosses": sorted(rows, key=lambda r: r["legal_delta_step35_minus_old"])[:20],
        "Focus": [r for r in rows if r.get("in_focus_set")],
    }
    corr = {name: corr_block(rs, metrics) for name, rs in subsets.items()}

    # Compact comparison of worst losses vs neutral/gain UIDs.
    worst20 = subsets["Worst20LegalLosses"]
    nonloss = [r for r in rows if float(r["legal_delta_step35_minus_old"]) >= 0.0]
    moderate = [r for r in rows if float(r["legal_delta_step35_minus_old"]) < 0.0 and r not in worst20]
    group_means: dict[str, Any] = {}
    for name, rs in [("worst20", worst20), ("other_losses", moderate), ("nonloss", nonloss), ("ewok_relation", [r for r in rows if r.get("ewok_domain_family") == "relation_or_dynamics"]), ("ewok_property", [r for r in rows if r.get("ewok_domain_family") == "property"] )]:
        group_means[name] = {"n": len(rs), "mean_legal_delta": safe_round(mean([r["legal_delta_step35_minus_old"] for r in rs]))}
        for m in metrics[:10]:
            group_means[name][f"mean_{m}"] = safe_round(mean([r.get(m) for r in rs]))

    factor_comp = factor.get("token_accounting", {}).get("comparisons", {}).get("minfreq50_supportfloor_minus_step35_legal16k", {})
    init_info = factor.get("initialization_audit", {})
    prefix = "delta_minfreq50_supportfloor_minus_step35_legal16k_"
    factor_summary = {
        "global_raw_tokens_per_word_delta": factor_comp.get(prefix + "raw_tokens_per_word"),
        "global_visible_tokens_per_word_delta": factor_comp.get(prefix + "visible_tokens_per_word"),
        "global_visible_groups_per_word_delta": factor_comp.get(prefix + "visible_groups_per_word"),
        "global_over256_row_delta": factor_comp.get(prefix + "over256_rows"),
        "global_truncated_tokens_delta": factor_comp.get(prefix + "truncated_tokens"),
        "relative_expected_target_tokens": factor_comp.get("relative_expected_target_tokens"),
        "relative_expected_selected_groups": factor_comp.get("relative_expected_selected_groups"),
        "standard_same_seed_random_like_exact_fraction": init_info.get("standard_same_seed_audit", {}).get("random_like_exact_numel_fraction"),
        "initmatched_random_like_exact_fraction": init_info.get("init_matched_copy_simulation", {}).get("postcopy_compare_to_step35", {}).get("random_like_exact_numel_fraction"),
    }

    result = {
        "status": "SUPPORTFLOOR_SUBSTRATE_ALIGNMENT",
        "created_utc": now_utc(),
        "scope": "CPU-only synthesis of existing tokenizer/corpus measurements and existing UID score deltas; no training or model evaluation.",
        "scientific_question": "Does the minfreq50 support-floored legal tokenizer mechanically align with the UID families where the research legal endpoint lost relative to the old non-submittable tokenizer reference?",
        "inputs": {
            "deficit_json": str(DEFICIT_JSON),
            "fragmentation_json": str(FRAG_JSON),
            "support_identity_json": str(SUPPORT_JSON),
            "supportfloor_factor_json": str(FACTOR_JSON),
        },
        "factor_summary": factor_summary,
        "correlations": corr,
        "group_means": group_means,
        "worst_20_legal_losses": worst20,
        "focus_rows": [r for r in rows if r.get("in_focus_set")],
        "per_uid": rows,
    }
    out_json = OUT_DIR / "supportfloor_substrate_alignment.json"
    out_csv = OUT_DIR / "supportfloor_substrate_alignment_per_uid.csv"
    out_json.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    columns = [
        "column", "uid", "legal_delta_step35_minus_old", "legal_deficit_magnitude",
        "tpw", "minfreq50_tpw", "minfreq50_minus_step35_tpw", "minfreq50_token_reduction_pct_vs_step35",
        "mean_log10_pool_count_plus1", "minfreq50_mean_log10_pool_count_plus1", "minfreq50_minus_step35_support_log_gain",
        "frac_pool_count_lt_50", "minfreq50_frac_pool_count_lt_50", "minfreq50_minus_step35_frac_lt50_delta",
        "vs_old_token_multiset_jaccard", "minfreq50_vs_old_token_multiset_jaccard", "minfreq50_minus_step35_vs_old_token_multiset_jaccard",
        "vs_old_b_not_in_a_frac", "minfreq50_vs_old_b_not_in_a_frac", "minfreq50_minus_step35_vs_old_b_not_in_a_frac",
    ]
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=columns)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c) for c in columns})

    lines: list[str] = []
    lines.append("# research support-floor substrate alignment")
    lines.append("")
    lines.append(result["scope"])
    lines.append("")
    lines.append(result["scientific_question"])
    lines.append("")
    lines.append("## Global support-floor movement")
    lines.append(f"- minfreq50 global raw tokens/word delta vs research: {fmt(factor_summary['global_raw_tokens_per_word_delta'], 6)}; visible tokens/word delta {fmt(factor_summary['global_visible_tokens_per_word_delta'], 6)}.")
    lines.append(f"- over-256 rows delta {fmt(factor_summary['global_over256_row_delta'], 0)}; truncated tokens delta {fmt(factor_summary['global_truncated_tokens_delta'], 0)}.")
    lines.append(f"- expected target-token ratio {fmt(factor_summary['relative_expected_target_tokens'], 6)}; expected selected-group ratio {fmt(factor_summary['relative_expected_selected_groups'], 6)}.")
    lines.append(f"- same-shape initialization match after copy: {fmt(factor_summary['initmatched_random_like_exact_fraction'], 4)}; ordinary same-seed match {fmt(factor_summary['standard_same_seed_random_like_exact_fraction'], 4)}.")
    lines.append("")
    lines.append("## Correlation with research legal loss")
    lines.append("Negative legal delta means the legal research endpoint lost against the old non-submittable reference. Positive deficit magnitude is the lost amount clipped at zero.")
    lines.append("| subset | n | metric | mean | Pearson(legal delta, metric) | Pearson(deficit magnitude, metric) |")
    lines.append("|---|---:|---|---:|---:|---:|")
    for subset in ["ALL", "BLiMP", "Supplement", "EWoK", "Worst20LegalLosses"]:
        for m in metrics[:10]:
            rec = corr[subset][m]
            lines.append(f"| {subset} | {corr[subset]['n_uid']} | `{m}` | {fmt(rec['mean'], 6)} | {fmt(rec['pearson_legal_delta_vs_metric'], 4)} | {fmt(rec['pearson_deficit_magnitude_vs_metric'], 4)} |")
    lines.append("")
    lines.append("## Group means")
    lines.append("| group | n | mean legal delta | mean minfreq50-research tpw | mean support log gain | mean frac<50 delta | mean Jaccard-vs-old gain |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for name in ["worst20", "other_losses", "nonloss", "ewok_relation", "ewok_property"]:
        g = group_means[name]
        lines.append(f"| {name} | {g['n']} | {fmt(g['mean_legal_delta'], 4)} | {fmt(g['mean_minfreq50_minus_step35_tpw'], 6)} | {fmt(g['mean_minfreq50_minus_step35_support_log_gain'], 6)} | {fmt(g['mean_minfreq50_minus_step35_frac_lt50_delta'], 6)} | {fmt(g['mean_minfreq50_minus_step35_vs_old_token_multiset_jaccard'], 6)} |")
    lines.append("")
    lines.append("## Twenty largest research legal losses")
    lines.append("| column | uid | legal delta | minfreq50-research tpw | support log gain | frac<50 delta | Jaccard-vs-old gain | new-vs-old frac delta |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|")
    for r in worst20:
        lines.append(f"| {r['column']} | {r['uid']} | {fmt(r['legal_delta_step35_minus_old'], 2)} | {fmt(r.get('minfreq50_minus_step35_tpw'), 6)} | {fmt(r.get('minfreq50_minus_step35_support_log_gain'), 6)} | {fmt(r.get('minfreq50_minus_step35_frac_lt50_delta'), 6)} | {fmt(r.get('minfreq50_minus_step35_vs_old_token_multiset_jaccard'), 6)} | {fmt(r.get('minfreq50_minus_step35_vs_old_b_not_in_a_frac'), 6)} |")
    lines.append("")
    lines.append("## Scientific reading")
    lines.append("- The support-floor route remains a broad legal representation experiment, not a targeted repair derived from the worst UID labels. Its strongest prior value is small but clean: fewer fragmented tokens, lower truncation, lower target-token burden, and much less unsupported tail than large 40k while preserving compact-view reinvestment.")
    lines.append("- If its future 70M/80M score surface helps, the result should be read as a representation/support/segmentation package. If it hurts Entity or GlobalPIQA like the larger 40k route, the support-floor route should stop rather than be expanded by vocabulary size alone.")
    lines.append("- Existing eval-string correlations here should not be used to tune the tokenizer; they only explain what a fixed, already-trained legal tokenizer changes.")
    lines.append("")
    lines.append(f"JSON: `{out_json}`")
    lines.append(f"CSV: `{out_csv}`")
    NOTE.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(json.dumps({
        "status": result["status"],
        "out_json": str(out_json),
        "out_note": str(NOTE),
        "out_csv": str(out_csv),
        "n_uid": len(rows),
        "factor_summary": factor_summary,
        "all_key_correlations": {m: corr["ALL"][m] for m in metrics[:6]},
        "worst20_group_means": group_means["worst20"],
    }, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
