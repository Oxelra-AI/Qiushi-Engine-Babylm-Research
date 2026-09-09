#!/usr/bin/env python3
"""research/054: synthesize corrected-tokenizer endpoint interpretation.

CPU-only. Run after `compare_corrected_tokenizer_two_seed_results.py`
has produced corrected full official vectors. It combines:
  - current public Strict-Small leader surface;
  - corrected-tokenizer two-seed vectors and corrected-vs-inherited movement;
  - exact tokenizer surface contingency;
  - corrected-tokenizer training completion facts;
  - old-coordinate relation dynamics baseline;
  - full old EWoK atlas, concept-exposure link, and byte-coverage record.

It is not final writing or packaging; it preserves the scientific reading and
next research action that follows from actual full official scores.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import json
import math
import time
from pathlib import Path
from typing import Any

HERE = _public_path('experiments/archive/representation_and_objectives/scripts/corrected_endpoint_interpretation.py')
A01 = _public_path('experiments/archive/representation_and_objectives')
USER_ROOT = _public_path('.')
WORKSPACE = _public_path('experiments/archive/representation_and_objectives')
OUT_DIR = _public_path('experiments/archive/representation_and_objectives/data/corrected_endpoint_interpretation')
OUT_JSON = _public_path('experiments/archive/representation_and_objectives/data/corrected_endpoint_interpretation/corrected_endpoint_interpretation.json')
OUT_MD = _public_path('research/notes/representation_and_objectives/corrected_endpoint_interpretation.md')
COMPARE = _public_path('experiments/archive/representation_and_objectives/data/corrected_tokenizer_two_seed_comparison/corrected_tokenizer_two_seed_comparison.json')
TRAINING = _public_path('experiments/archive/representation_and_objectives/data/strictsmalltok_training_completion/strictsmalltok_training_completion_summary.json')
TOKEN_SURFACE = _public_path('experiments/archive/representation_and_objectives/data/tokenizer_surface_contingency/tokenizer_surface_contingency.json')
REL_DYN = _public_path('experiments/archive/representation_and_objectives/data/relation_phase_dynamics_summary/relation_phase_dynamics_summary.json')
OLD_ATLAS = _public_path('experiments/archive/representation_and_objectives/data/full_old_ewok_atlas_synthesis/full_old_ewok_atlas_synthesis.json')
EXPOSURE_LINK = _public_path('experiments/archive/representation_and_objectives/data/ewok_old_atlas_training_exposure_link/ewok_old_atlas_training_exposure_link.json')
BYTE_RECORD = _public_path('experiments/archive/representation_and_objectives/data/tokenizer_byte_coverage_audit/tokenizer_byte_coverage_audit.json')
FOCUS_TOKEN_MARGIN_LINK = _public_path('experiments/archive/representation_and_objectives/data/ewok_focus_tokenization_margin_link/ewok_focus_tokenization_margin_link.json')
LEADERBOARD = _public_path('experiments/archive/representation_and_objectives/data/babylm2026_live_surface/leaderboard_parsed.json')
COLUMNS = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "SuperGLUE", "GlobalPIQA", "Reading", "AoA"]


def now_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def rel(p: Path | str) -> str:
    p = Path(p)
    try:
        return str(p.relative_to(USER_ROOT))
    except Exception:
        return str(p)


def load(path: Path) -> Any:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def num(x: Any) -> float | None:
    if isinstance(x, (int, float)) and math.isfinite(float(x)):
        return float(x)
    try:
        v = float(x)
        return v if math.isfinite(v) else None
    except Exception:
        return None


def current_leader(lb: Any) -> dict[str, Any]:
    if not isinstance(lb, dict):
        return {"exists": False, "overall": 41.8, "model": "unknown"}
    top = (lb.get("top_by_track") or {}).get("strict-small") or []
    row = top[0] if top else {}
    return {
        "exists": bool(row),
        "model": row.get("Model_plain"),
        "repo": row.get("HF Repo"),
        "overall": num(row.get("Overall Average")) or 41.8,
        "fetched_utc": lb.get("fetched_utc"),
        "track_counts": lb.get("track_counts"),
        "row": row,
    }


def get_scores(compare: dict[str, Any], seed: str) -> dict[str, float]:
    rec = ((compare.get("corrected_tokenizer_results") or {}).get(seed) or {}).get("scores") or {}
    return {k: float(v) for k, v in rec.items() if isinstance(v, (int, float)) and math.isfinite(float(v))}


def group_row(rows: list[dict[str, Any]], group: str) -> dict[str, Any]:
    for r in rows:
        if r.get("group") == group:
            return r
    return {}


def surface_summary(ts: Any) -> dict[str, Any]:
    if not isinstance(ts, dict):
        return {"exists": False}
    eval_rows = ts.get("eval_occurrence_rows") or []
    pool_rows = ts.get("reinvest_pool_occurrence_rows") or []
    keep_groups = [
        "ALL",
        "family::BLiMP",
        "family::Supplement",
        "family::EWoK",
        "family::Entity",
        "family::COMPS",
        "family::GlobalPIQA_parallel",
        "family::GlobalPIQA_nonparallel",
        "family::SuperGLUE",
    ]
    out = {
        "exists": True,
        "new_tokenizer_sha256": ts.get("new_tokenizer_sha256"),
        "vocab_shared_fraction": (ts.get("vocab") or {}).get("shared_fraction"),
        "pretrain_pool_all_ratio": group_row(pool_rows, "ALL").get("new_over_old_token_ratio"),
        "changed_block_ratio": group_row(pool_rows, "reinvest_changed_block").get("new_over_old_token_ratio"),
        "eval_groups": {},
        "ewok_domains": {},
    }
    for g in keep_groups:
        r = group_row(eval_rows, g)
        if r:
            out["eval_groups"][g] = {
                "new_over_old_token_ratio": r.get("new_over_old_token_ratio"),
                "old_only_occ_pct": 100.0 * float(r.get("old_only_occ_fraction", 0.0)),
                "mean_token_delta": (r.get("delta_len") or {}).get("mean"),
            }
    for r in eval_rows:
        g = r.get("group", "")
        if str(g).startswith("ewok::"):
            out["ewok_domains"][str(g).replace("ewok::", "")] = {
                "new_over_old_token_ratio": r.get("new_over_old_token_ratio"),
                "old_only_occ_pct": 100.0 * float(r.get("old_only_occ_fraction", 0.0)),
                "mean_token_delta": (r.get("delta_len") or {}).get("mean"),
            }
    return out


def old_atlas_summary(old_atlas: Any) -> dict[str, Any]:
    if not isinstance(old_atlas, dict):
        return {"exists": False}
    overall = old_atlas.get("overall") or {}
    row = old_atlas.get("row_summary") or {}
    te = (overall.get("treatment_effect_micro_pp") or {}) if isinstance(overall, dict) else {}
    top = (((old_atlas.get("top_group_interactions") or {}).get("domain") or {}).get("most_negative") or [])[:5]
    return {
        "exists": True,
        "preflight_selected_rows": old_atlas.get("preflight_selected_rows"),
        "official_prediction_match_rates": {k: v.get("official_prediction_match_rate") for k, v in overall.items() if isinstance(v, dict)},
        "micro_accuracy": {k: v.get("micro_accuracy") for k, v in overall.items() if isinstance(v, dict)},
        "treatment_effect_micro_pp": te,
        "negative_rows": row.get("n_negative_accuracy_interaction_rows"),
        "negative_rows_all_four_margins_abs_lt_1_frac": row.get("negative_rows_all_four_margins_abs_lt_1_frac"),
        "seed_polarized_0110_rows": row.get("n_old_seed_polarized_0110"),
        "seed_polarized_1001_rows": row.get("n_old_seed_polarized_1001"),
        "top_negative_domains": [
            {
                "domain": r.get("group_value"),
                "n": r.get("n"),
                "interaction_pp": (r.get("treatment_effect_pp") or {}).get("interaction_te431_minus_te430"),
                "te430_pp": (r.get("treatment_effect_pp") or {}).get("seed43022"),
                "te431_pp": (r.get("treatment_effect_pp") or {}).get("seed43122"),
            }
            for r in top
        ],
    }


def exposure_summary(exposure: Any) -> dict[str, Any]:
    if not isinstance(exposure, dict):
        return {"exists": False}
    subsets = exposure.get("subset_summary") or {}
    neg = subsets.get("negative_accuracy_interaction") or {}
    all_rows = subsets.get("all") or {}
    old0110 = subsets.get("old_0110_seed430_help_seed431_hurt") or {}
    return {
        "exists": True,
        "n_unique_concept_keys": exposure.get("n_unique_concept_keys"),
        "correlations": exposure.get("correlations") or {},
        "all_both_seen_full_frac": all_rows.get("both_seen_all_frac"),
        "all_both_seen_changed_frac": all_rows.get("both_seen_changed_frac"),
        "negative_both_seen_full_frac": neg.get("both_seen_all_frac"),
        "negative_both_seen_changed_frac": neg.get("both_seen_changed_frac"),
        "old0110_both_seen_full_frac": old0110.get("both_seen_all_frac"),
        "old0110_both_seen_changed_frac": old0110.get("both_seen_changed_frac"),
    }


def byte_summary(byte_record: Any) -> dict[str, Any]:
    if not isinstance(byte_record, dict):
        return {"exists": False}
    pool_total = ((byte_record.get("pool_audit") or {}).get("total") or {})
    eval_total = ((byte_record.get("eval_audit") or {}).get("total") or {})
    old_cmp = byte_record.get("compare_old_vs_a01_eval") or {}
    return {
        "exists": True,
        "a01_sha256": byte_record.get("a01_sha256"),
        "sha_matches_expected": byte_record.get("sha_matches_expected"),
        "missing_bytelevel_alphabet_count": byte_record.get("missing_bytelevel_alphabet_count"),
        "pool_unk_tokens": pool_total.get("unk_tokens"),
        "pool_strings_with_unk": pool_total.get("strings_with_unk"),
        "eval_unk_tokens": eval_total.get("unk_tokens"),
        "eval_strings_with_unk": eval_total.get("strings_with_unk"),
        "eval_strings": eval_total.get("strings"),
        "old_vs_a01_eval_token_ratio": old_cmp.get("token_ratio_b_over_a"),
        "a02byte_vs_a01_eval_token_ratio": (byte_record.get("compare_a02_bytealpha_vs_a01_eval") or {}).get("token_ratio_b_over_a"),
    }


def focus_token_margin_summary(focus: Any) -> dict[str, Any]:
    if not isinstance(focus, dict):
        return {"exists": False}
    glob = focus.get("global") or {}
    keys = [k for k in glob if k.startswith("pearson_")]
    max_abs = None
    if keys:
        vals = [abs(float(glob[k])) for k in keys if isinstance(glob.get(k), (int, float)) and math.isfinite(float(glob[k]))]
        max_abs = max(vals) if vals else None
    return {
        "exists": True,
        "n_rows": glob.get("n_rows"),
        "max_abs_pearson_token_length_feature_vs_margin": max_abs,
        "pearsons": {k: glob.get(k) for k in keys},
    }


def relation_phase_summary(rel_dyn: Any) -> dict[str, Any]:
    if not isinstance(rel_dyn, dict):
        return {"exists": False}
    strict = rel_dyn.get("strict_trajectory") or []
    old = rel_dyn.get("old_trajectory") or {}
    return {
        "exists": True,
        "strict_trajectory": strict,
        "old_summary": old,
    }


def largest_movements(diff: dict[str, float], n: int = 5) -> list[dict[str, Any]]:
    rows = [{"column": k, "delta": diff[k]} for k in COLUMNS if k in diff]
    return sorted(rows, key=lambda r: abs(r["delta"]), reverse=True)[:n]


def fmt_pct(x: Any) -> str:
    v = num(x)
    if v is None:
        return "missing"
    return f"{100.0*v:.2f}%"


def append_mechanism_context(lines: list[str], olda: dict[str, Any], exp: dict[str, Any], byt: dict[str, Any], focus: dict[str, Any], rels: dict[str, Any]) -> None:
    if byt.get("exists"):
        if byt.get("missing_bytelevel_alphabet_count") == 0 and byt.get("pool_unk_tokens") == 0 and byt.get("eval_unk_tokens") == 0:
            lines.append(
                f"A01 legal tokenizer SHA {str(byt.get('a01_sha256'))[:12]} has zero missing ByteLevel alphabet entries, zero <unk> on the 10M pool, and zero <unk> across {byt.get('eval_strings')} official scored-text strings; A02's separate same-pool tokenizer construction issue does not reclassify A01's running evaluations."
            )
        else:
            lines.append(
                f"A01 tokenizer byte coverage is not clean in the research record: missing alphabet {byt.get('missing_bytelevel_alphabet_count')}, pool <unk> {byt.get('pool_unk_tokens')}, eval <unk> {byt.get('eval_unk_tokens')}; endpoint interpretation must separate tokenizer-construction effects from compact-view effects."
            )
    if olda.get("exists"):
        te = olda.get("treatment_effect_micro_pp") or {}
        domains = ", ".join(f"{r['domain']} {num(r.get('interaction_pp')):+.2f}pp" for r in olda.get("top_negative_domains", []) if r.get("domain") is not None and num(r.get("interaction_pp")) is not None)
        lines.append(
            f"Old inherited-tokenizer EWoK atlas is a full-coordinate mechanism baseline: {olda.get('preflight_selected_rows')} rows, treatment effect seed43022 {num(te.get('seed43022')):+.3f}pp, seed43122 {num(te.get('seed43122')):+.3f}pp, interaction {num(te.get('interaction_te431_minus_te430')):+.3f}pp, negative-interaction rows {olda.get('negative_rows')}, and only {fmt_pct(olda.get('negative_rows_all_four_margins_abs_lt_1_frac'))} of those have all four margins within ±1. Strongest old negative domains: {domains}."
        )
    if exp.get("exists"):
        corr = exp.get("correlations") or {}
        lines.append(
            f"Training-text concept exposure does not explain the old EWoK instability: negative-interaction rows have both EWoK concepts in the full 10M pool {fmt_pct(exp.get('negative_both_seen_full_frac'))} of the time and in the compact changed block {fmt_pct(exp.get('negative_both_seen_changed_frac'))}; exposure correlations with old accuracy interaction are near zero (min-full r={num(corr.get('corr_log1p_min_all_occ_vs_accuracy_interaction')):+.4f}, sum-full r={num(corr.get('corr_log1p_sum_all_occ_vs_accuracy_interaction')):+.4f})."
        )
    if rels.get("exists"):
        strict100 = None
        for r in rels.get("strict_trajectory") or []:
            label = str(r.get("label") or r.get("checkpoint") or r.get("source") or "")
            if "100" in label or str(r.get("checkpoint", "")).endswith("100M"):
                strict100 = r
        # research/52 summaries use a compact dict keyed by labels; fall back if needed.
        if strict100 is None:
            for r in rels.get("strict_trajectory") or []:
                if "acc_43022" in r and "acc_43122" in r:
                    strict100 = r
        if strict100:
            a430 = strict100.get("acc_43022") or strict100.get("accuracy_43022")
            a431 = strict100.get("acc_43122") or strict100.get("accuracy_43122")
            corr_old = strict100.get("corr_with_old100_seed_delta") or strict100.get("strict100_seed_delta_corr_with_old100_seed_delta")
            parts = []
            if a430 is not None and a431 is not None:
                parts.append(f"focused corrected 100M EWoK old-instability subset scores {num(a430):.2f}% vs {num(a431):.2f}% for seeds 43022/43122")
            if corr_old is not None:
                parts.append(f"seed-delta correlation with old endpoint {num(corr_old):+.3f}")
            if parts:
                lines.append("Corrected-tokenizer focused EWoK endpoint probe before full official collation: " + ", ".join(parts) + "; treat this as mechanism-local evidence only.")
    if focus.get("exists") and focus.get("max_abs_pearson_token_length_feature_vs_margin") is not None:
        lines.append(
            f"On the same 553-row focused EWoK subset, item-level old-vs-A01 token-length features have max |Pearson r| {focus.get('max_abs_pearson_token_length_feature_vs_margin'):.3f} with corrected 100M margins, so the local corrected relation behavior is not a crude length-fragmentation artifact."
        )


def interpret(compare: dict[str, Any], leader: dict[str, Any], surf: dict[str, Any], olda: dict[str, Any], exp: dict[str, Any], byt: dict[str, Any], focus: dict[str, Any], rels: dict[str, Any]) -> tuple[list[str], str]:
    lines: list[str] = []
    next_action = "wait_for_full_official_results"
    complete = compare.get("complete_corrected_seeds") or []
    if len(complete) < 2:
        lines.append("Corrected-tokenizer full official vectors are not both present; endpoint judgment remains unavailable.")
        lines.append("Do not infer endpoint quality from training loss, partial seven-column surfaces, or old-tokenizer seed behavior.")
        append_mechanism_context(lines, olda, exp, byt, focus, rels)
        return lines, next_action
    scores = {s: get_scores(compare, s) for s in ["43022", "43122"]}
    ov = {s: scores[s].get("Overall") for s in scores}
    clears = {s: (ov[s] is not None and ov[s] > leader["overall"]) for s in ov}
    mean = ((compare.get("comparisons") or {}).get("corrected_two_seed_mean") or {}).get("Overall")
    spread = (compare.get("comparisons") or {}).get("corrected_overall_spread_abs")
    if all(clears.values()):
        lines.append(f"Both corrected-tokenizer seeds clear the current visible Strict-Small leader {leader['overall']}; compact_view_reinvest survives the compliant representation as a strong endpoint family.")
        next_action = "protect_compliant_endpoint_family_with_full_provenance_and_submission_files"
    elif any(clears.values()):
        winner = [s for s, c in clears.items() if c][0]
        lines.append(f"Only seed{winner} clears the current visible Strict-Small leader {leader['overall']}; the route yields a compliant candidate endpoint but seed sensitivity remains central.")
        next_action = "protect_winning_seed_and_choose_third_seed_or_mechanism_repair_from_column_movement"
    else:
        lines.append(f"Neither corrected-tokenizer seed clears the current visible Strict-Small leader {leader['overall']}; the old inherited-tokenizer 42.033 result remains mechanism evidence, not a compliant endpoint.")
        next_action = "rebuild_legal_representation_data_or_learning_dynamics_from_score_movement"
    if mean is not None and spread is not None:
        lines.append(f"Corrected two-seed mean Overall {float(mean):.6f}; seed spread {float(spread):.6f}.")
    append_mechanism_context(lines, olda, exp, byt, focus, rels)
    # Tokenizer-surface interpretation conditioned on score movements.
    for seed, sc in scores.items():
        olddiff = ((compare.get("comparisons") or {}).get(f"corrected_minus_inherited_seed{seed}") or {})
        if olddiff:
            top = largest_movements(olddiff, 4)
            lines.append(f"Seed{seed} corrected-minus-inherited largest column movements: " + ", ".join(f"{r['column']} {r['delta']:+.3f}" for r in top) + ".")
            ewok_delta = olddiff.get("EWoK")
            comps_delta = olddiff.get("COMPS")
            gp_delta = olddiff.get("GlobalPIQA")
            if surf.get("exists"):
                eg = surf.get("eval_groups", {})
                ewok_pressure = (eg.get("family::EWoK") or {}).get("new_over_old_token_ratio")
                comps_pressure = (eg.get("family::COMPS") or {}).get("new_over_old_token_ratio")
                if ewok_delta is not None and ewok_delta < -1.0 and ewok_pressure and ewok_pressure > 1.02:
                    lines.append(f"Seed{seed} EWoK falls by {ewok_delta:+.3f} while A01 tokenizer lengthens EWoK text by ratio {ewok_pressure:.4f}; inspect EWoK domains and old-atlas row classes before attributing loss to the compact-view data mechanism alone.")
                if comps_delta is not None and comps_delta < -1.0 and comps_pressure and comps_pressure > 1.005:
                    lines.append(f"Seed{seed} COMPS falls by {comps_delta:+.3f} with mild COMPS tokenization pressure ratio {comps_pressure:.4f}; animal/object word fragmentation is a plausible localization handle.")
                if gp_delta is not None and gp_delta < -1.0:
                    lines.append(f"Seed{seed} GlobalPIQA falls by {gp_delta:+.3f}; compare to practical-object fragmentation in the A01 tokenizer surface and avoid repeating the rejected narrow lexicon-swap route.")
    return lines, next_action


def write_md(payload: dict[str, Any]) -> None:
    lines = ["# Corrected-tokenizer endpoint interpretation\n\n"]
    lines.append(f"Current public Strict-Small leader: `{payload['current_leader'].get('model')}` Overall `{payload['current_leader'].get('overall')}` fetched `{payload['current_leader'].get('fetched_utc')}`.\n\n")
    lines.append(f"Training artifacts complete: `{payload.get('training_both_complete')}`. Corrected seeds present: `{payload.get('complete_corrected_seeds')}`.\n\n")
    lines.append("## Scientific reading\n\n")
    for x in payload.get("interpretation", []):
        lines.append(f"- {x}\n")
    lines.append("\n## Next action\n\n")
    lines.append(f"`{payload.get('next_action_class')}`\n")
    lines.append("\n## Files used\n\n")
    for k, v in payload.get("files", {}).items():
        lines.append(f"- {k}: `{v}`\n")
    OUT_MD.write_text("".join(lines), encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    compare = load(COMPARE) or {}
    training = load(TRAINING) or {}
    token_surface = load(TOKEN_SURFACE) or {}
    rel_dyn = load(REL_DYN) or {}
    old_atlas = load(OLD_ATLAS) or {}
    exposure = load(EXPOSURE_LINK) or {}
    byte_record = load(BYTE_RECORD) or {}
    focus_link = load(FOCUS_TOKEN_MARGIN_LINK) or {}
    leaderboard = load(LEADERBOARD) or {}
    leader = current_leader(leaderboard)
    surf = surface_summary(token_surface)
    olda = old_atlas_summary(old_atlas)
    exp = exposure_summary(exposure)
    byt = byte_summary(byte_record)
    focus = focus_token_margin_summary(focus_link)
    rels = relation_phase_summary(rel_dyn)
    interp, next_action = interpret(compare, leader, surf, olda, exp, byt, focus, rels)
    payload = {
        "status": "CORRECTED_ENDPOINT_INTERPRETATION_WITH_STEP053_EVIDENCE",
        "created_utc": now_utc(),
        "current_leader": leader,
        "training_both_complete": training.get("both_complete_training_artifacts") if isinstance(training, dict) else None,
        "complete_corrected_seeds": compare.get("complete_corrected_seeds") if isinstance(compare, dict) else [],
        "missing_or_incomplete_corrected_seeds": compare.get("missing_or_incomplete_corrected_seeds") if isinstance(compare, dict) else ["43022", "43122"],
        "corrected_score_comparison_status": compare.get("status") if isinstance(compare, dict) else None,
        "tokenizer_surface_summary": surf,
        "old_ewok_atlas_summary": olda,
        "training_exposure_summary": exp,
        "byte_coverage_summary": byt,
        "focus_tokenization_margin_summary": focus,
        "relation_phase_summary": rels,
        "interpretation": interp,
        "next_action_class": next_action,
        "files": {
            "corrected_two_seed_comparison": rel(COMPARE),
            "training_completion": rel(TRAINING),
            "a01_tokenizer_surface": rel(TOKEN_SURFACE),
            "relation_phase_dynamics": rel(REL_DYN),
            "old_ewok_atlas": rel(OLD_ATLAS),
            "training_exposure_link": rel(EXPOSURE_LINK),
            "a01_byte_coverage_record": rel(BYTE_RECORD),
            "focus_tokenization_margin_link": rel(FOCUS_TOKEN_MARGIN_LINK),
            "leaderboard": rel(LEADERBOARD),
            "note": rel(OUT_MD),
        },
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_md(payload)
    print(json.dumps({
        "status": payload["status"],
        "out_json": rel(OUT_JSON),
        "note": rel(OUT_MD),
        "leader_overall": leader.get("overall"),
        "training_both_complete": payload["training_both_complete"],
        "complete_corrected_seeds": payload["complete_corrected_seeds"],
        "next_action_class": next_action,
        "interpretation_head": interp[:5],
    }, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
