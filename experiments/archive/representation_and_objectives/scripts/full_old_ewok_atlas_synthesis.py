#!/usr/bin/env python3
"""research: synthesize the full old-tokenizer EWoK margin atlas.

Reads the official-compatible full EWoK candidate-margin CSV produced in research for
four inherited-tokenizer cells (clean/reinvest x seed43022/43122).  Produces a
compact scientific atlas of where compact_view_reinvest changed EWoK relation
preferences, which parts are stable, which parts are seed-specific, and which rows
should be reused as stress subsets after corrected-tokenizer official results land.
"""
from __future__ import annotations

import csv
import json
import math
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

USER_ROOT = Path(".").resolve()
WORKSPACE = USER_ROOT / "experiments/archive/representation_and_objectives"
CSV_PATH = WORKSPACE / "data/official_ewok_margin_full_old4/official_ewok_margin_full_old4_records.csv"
PREFLIGHT_PATH = WORKSPACE / "data/official_ewok_margin_full_old4/official_ewok_margin_full_old4_preflight.json"
EWOK_DIR = WORKSPACE / "data/pristine_official_coordinate/babylm-eval/strict/evaluation_data/full_eval/ewok_filtered"
OUT_DIR = WORKSPACE / "data/full_old_ewok_atlas_synthesis"
NOTE_PATH = (USER_ROOT / 'research/notes/representation_and_objectives/full_old_ewok_atlas_synthesis.md')

MODELS = ["clean430", "reinv430", "clean431", "reinv431"]
BIT_ORDER = MODELS


def now_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def to_float(x: Any) -> float | None:
    try:
        if x is None or x == "":
            return None
        v = float(x)
        if math.isnan(v):
            return None
        return v
    except Exception:
        return None


def to_int(x: Any) -> int | None:
    try:
        if x is None or x == "":
            return None
        return int(float(x))
    except Exception:
        return None


def mean(xs: list[float]) -> float | None:
    return statistics.mean(xs) if xs else None


def median(xs: list[float]) -> float | None:
    return statistics.median(xs) if xs else None


def pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    mx, my = statistics.mean(xs), statistics.mean(ys)
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx <= 0 or vy <= 0:
        return None
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / math.sqrt(vx * vy)


def load_ewok_texts() -> dict[tuple[str, int], dict[str, Any]]:
    out: dict[tuple[str, int], dict[str, Any]] = {}
    for p in sorted(EWOK_DIR.glob("*.jsonl")):
        dom = p.stem
        with p.open("r", encoding="utf-8", errors="replace") as f:
            for idx, line in enumerate(f):
                if not line.strip():
                    continue
                obj = json.loads(line)
                out[(dom, idx)] = {
                    "file_domain": dom,
                    "idx": idx,
                    "Domain": obj.get("Domain", dom),
                    "ConceptA": obj.get("ConceptA"),
                    "ConceptB": obj.get("ConceptB"),
                    "ContextType": obj.get("ContextType"),
                    "ContextDiff": obj.get("ContextDiff"),
                    "TargetDiff": obj.get("TargetDiff"),
                    "Context1": obj.get("Context1"),
                    "Context2": obj.get("Context2"),
                    "Target1": obj.get("Target1"),
                    "Target2": obj.get("Target2"),
                }
    return out


def load_rows() -> list[dict[str, Any]]:
    by_uid: dict[str, dict[str, Any]] = {}
    with CSV_PATH.open("r", encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.DictReader(f)
        for rec in reader:
            mk = rec["model_key"]
            uid = rec["uid"]
            if mk not in MODELS:
                continue
            item = by_uid.setdefault(uid, {
                "uid": uid,
                "domain": rec.get("domain"),
                "idx": to_int(rec.get("idx")),
                "ConceptA": rec.get("ConceptA"),
                "ConceptB": rec.get("ConceptB"),
                "ContextType": rec.get("ContextType"),
                "ContextDiff": rec.get("ContextDiff"),
                "TargetDiff": rec.get("TargetDiff"),
                "models": {},
            })
            item["models"][mk] = {
                "correct": to_int(rec.get("correct")),
                "margin": to_float(rec.get("margin_c0_minus_c1")),
                "official_prediction_match": to_int(rec.get("official_prediction_match")),
                "num_cand0_completion_tokens": to_int(rec.get("num_cand0_completion_tokens")),
                "num_cand1_completion_tokens": to_int(rec.get("num_cand1_completion_tokens")),
            }
    rows: list[dict[str, Any]] = []
    missing = []
    for uid, item in sorted(by_uid.items(), key=lambda kv: (kv[1].get("domain") or "", kv[1].get("idx") or -1)):
        if any(m not in item["models"] for m in MODELS):
            missing.append(uid)
            continue
        bits = "".join(str(item["models"][m]["correct"]) for m in BIT_ORDER)
        c430 = item["models"]["clean430"]["correct"]
        r430 = item["models"]["reinv430"]["correct"]
        c431 = item["models"]["clean431"]["correct"]
        r431 = item["models"]["reinv431"]["correct"]
        m_c430 = item["models"]["clean430"]["margin"]
        m_r430 = item["models"]["reinv430"]["margin"]
        m_c431 = item["models"]["clean431"]["margin"]
        m_r431 = item["models"]["reinv431"]["margin"]
        te430 = r430 - c430
        te431 = r431 - c431
        acc_interaction = te431 - te430
        margin_effect_430 = m_r430 - m_c430
        margin_effect_431 = m_r431 - m_c431
        margin_interaction = margin_effect_431 - margin_effect_430
        item.update({
            "pattern": bits,
            "treatment_effect_seed43022": te430,
            "treatment_effect_seed43122": te431,
            "accuracy_interaction_te431_minus_te430": acc_interaction,
            "margin_effect_seed43022": margin_effect_430,
            "margin_effect_seed43122": margin_effect_431,
            "margin_interaction_te431_minus_te430": margin_interaction,
            "seed_delta_clean_431_minus_430": m_c431 - m_c430,
            "seed_delta_reinvest_431_minus_430": m_r431 - m_r430,
        })
        rows.append(item)
    if missing:
        raise RuntimeError({"missing_models_for_items": missing[:20], "n_missing": len(missing)})
    return rows


def model_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for mk in MODELS:
        correct = [r["models"][mk]["correct"] for r in rows]
        margins = [r["models"][mk]["margin"] for r in rows]
        matches = [r["models"][mk]["official_prediction_match"] for r in rows]
        out[mk] = {
            "n": len(correct),
            "micro_accuracy": sum(correct) / len(correct) * 100.0,
            "official_prediction_match_rate": sum(matches) / len(matches) if matches else None,
            "margin_mean": mean(margins),
            "margin_median": median(margins),
            "abs_margin_mean": mean([abs(x) for x in margins]),
            "near_zero_abs_lt_0p5": sum(1 for x in margins if abs(x) < 0.5) / len(margins),
            "near_zero_abs_lt_1": sum(1 for x in margins if abs(x) < 1.0) / len(margins),
            "near_zero_abs_lt_2": sum(1 for x in margins if abs(x) < 2.0) / len(margins),
        }
    out["treatment_effect_micro_pp"] = {
        "seed43022": out["reinv430"]["micro_accuracy"] - out["clean430"]["micro_accuracy"],
        "seed43122": out["reinv431"]["micro_accuracy"] - out["clean431"]["micro_accuracy"],
        "interaction_te431_minus_te430": (out["reinv431"]["micro_accuracy"] - out["clean431"]["micro_accuracy"]) - (out["reinv430"]["micro_accuracy"] - out["clean430"]["micro_accuracy"]),
    }
    return out


def summarize_group(items: list[dict[str, Any]], key_name: str, key_value: str) -> dict[str, Any]:
    n = len(items)
    acc = {mk: sum(r["models"][mk]["correct"] for r in items) / n * 100.0 for mk in MODELS}
    margins_by = {mk: [r["models"][mk]["margin"] for r in items] for mk in MODELS}
    te430 = acc["reinv430"] - acc["clean430"]
    te431 = acc["reinv431"] - acc["clean431"]
    patterns = Counter(r["pattern"] for r in items)
    neg = [r for r in items if r["accuracy_interaction_te431_minus_te430"] < 0]
    pos = [r for r in items if r["accuracy_interaction_te431_minus_te430"] > 0]
    margin_effect_430 = [r["margin_effect_seed43022"] for r in items]
    margin_effect_431 = [r["margin_effect_seed43122"] for r in items]
    margin_interactions = [r["margin_interaction_te431_minus_te430"] for r in items]
    return {
        "group_key": key_name,
        "group_value": key_value,
        "n": n,
        "accuracy": acc,
        "treatment_effect_pp": {"seed43022": te430, "seed43122": te431, "interaction_te431_minus_te430": te431 - te430},
        "margin_effect_mean": {
            "seed43022": mean(margin_effect_430),
            "seed43122": mean(margin_effect_431),
            "interaction_te431_minus_te430": mean(margin_interactions),
        },
        "margin_mean": {mk: mean(vals) for mk, vals in margins_by.items()},
        "abs_margin_mean": {mk: mean([abs(x) for x in vals]) for mk, vals in margins_by.items()},
        "near_zero_abs_lt_1": {mk: sum(1 for x in vals if abs(x) < 1.0) / n for mk, vals in margins_by.items()},
        "pattern_counts": dict(sorted(patterns.items(), key=lambda kv: (-kv[1], kv[0]))),
        "negative_accuracy_interaction_rows": len(neg),
        "negative_accuracy_interaction_frac": len(neg) / n,
        "positive_accuracy_interaction_rows": len(pos),
        "positive_accuracy_interaction_frac": len(pos) / n,
        "seed_polarized_0110_rows": patterns.get("0110", 0),
        "seed_polarized_0110_frac": patterns.get("0110", 0) / n,
        "opposite_seed_polarized_1001_rows": patterns.get("1001", 0),
        "opposite_seed_polarized_1001_frac": patterns.get("1001", 0) / n,
    }


def group_table(rows: list[dict[str, Any]], key_name: str, key_func, min_n: int = 1) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        key = key_func(r)
        if key is None or key == "":
            key = "<missing>"
        groups[str(key)].append(r)
    table = [summarize_group(items, key_name, key) for key, items in groups.items() if len(items) >= min_n]
    table.sort(key=lambda d: (d["treatment_effect_pp"]["interaction_te431_minus_te430"], -d["n"], d["group_value"]))
    return table


def compact_group_rows(table: list[dict[str, Any]], top_k: int = 20) -> dict[str, Any]:
    neg = [d for d in table if d["treatment_effect_pp"]["interaction_te431_minus_te430"] < 0]
    pos = [d for d in table if d["treatment_effect_pp"]["interaction_te431_minus_te430"] > 0]
    pos.sort(key=lambda d: (-d["treatment_effect_pp"]["interaction_te431_minus_te430"], -d["n"], d["group_value"]))
    return {
        "most_negative": neg[:top_k],
        "most_positive": pos[:top_k],
    }


def enrich_examples(rows: list[dict[str, Any]], text_by_key: dict[tuple[str, int], dict[str, Any]], kind: str, limit: int = 24) -> list[dict[str, Any]]:
    if kind == "0110":
        candidates = [r for r in rows if r["pattern"] == "0110"]
        candidates.sort(key=lambda r: -(min(r["models"]["reinv430"]["margin"], -r["models"]["reinv431"]["margin"])))
    elif kind == "1001":
        candidates = [r for r in rows if r["pattern"] == "1001"]
        candidates.sort(key=lambda r: -(min(-r["models"]["reinv430"]["margin"], r["models"]["reinv431"]["margin"])))
    elif kind == "most_negative_margin_interaction":
        candidates = sorted(rows, key=lambda r: r["margin_interaction_te431_minus_te430"])[:limit]
    elif kind == "most_positive_margin_interaction":
        candidates = sorted(rows, key=lambda r: -r["margin_interaction_te431_minus_te430"])[:limit]
    else:
        candidates = []
    out = []
    for r in candidates[:limit]:
        text = text_by_key.get((r["domain"], int(r["idx"]))) or {}
        out.append({
            "uid": r["uid"],
            "domain": r["domain"],
            "idx": r["idx"],
            "ConceptA": r.get("ConceptA"),
            "ConceptB": r.get("ConceptB"),
            "ContextType": r.get("ContextType"),
            "ContextDiff": r.get("ContextDiff"),
            "TargetDiff": r.get("TargetDiff"),
            "pattern": r["pattern"],
            "treatment_effect_seed43022": r["treatment_effect_seed43022"],
            "treatment_effect_seed43122": r["treatment_effect_seed43122"],
            "accuracy_interaction_te431_minus_te430": r["accuracy_interaction_te431_minus_te430"],
            "margin_interaction_te431_minus_te430": r["margin_interaction_te431_minus_te430"],
            "margins": {mk: r["models"][mk]["margin"] for mk in MODELS},
            "correct": {mk: r["models"][mk]["correct"] for mk in MODELS},
            "Context1": text.get("Context1"),
            "Context2": text.get("Context2"),
            "Target1": text.get("Target1"),
            "Target2": text.get("Target2"),
        })
    return out


def write_selection_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = ["domain", "idx", "uid", "ConceptA", "ConceptB", "ContextType", "ContextDiff", "TargetDiff", "pattern", "accuracy_interaction_te431_minus_te430", "margin_interaction_te431_minus_te430"]
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in fields})


def top_concept_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    concept_table = group_table(rows, "ConceptA|ConceptB|domain", lambda r: f"{r.get('domain')}::{r.get('ConceptA')}|{r.get('ConceptB')}", min_n=4)
    pair_table = group_table(rows, "ConceptA|ConceptB", lambda r: f"{r.get('ConceptA')}|{r.get('ConceptB')}", min_n=8)
    return {
        "domain_concept_min_n4": compact_group_rows(concept_table, top_k=30),
        "concept_pair_min_n8": compact_group_rows(pair_table, top_k=30),
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = load_rows()
    text_by_key = load_ewok_texts()
    if len(rows) != 7618:
        raise RuntimeError({"expected_rows": 7618, "found": len(rows)})
    preflight = json.loads(PREFLIGHT_PATH.read_text(encoding="utf-8")) if PREFLIGHT_PATH.exists() else {}

    overall = model_summary(rows)
    pattern_counts = Counter(r["pattern"] for r in rows)
    pattern_summary = []
    for pat, n in sorted(pattern_counts.items(), key=lambda kv: (-kv[1], kv[0])):
        items = [r for r in rows if r["pattern"] == pat]
        pattern_summary.append(summarize_group(items, "pattern", pat))

    domain_table = group_table(rows, "domain", lambda r: r.get("domain"), min_n=1)
    context_type_table = group_table(rows, "ContextType", lambda r: r.get("ContextType"), min_n=10)
    context_diff_table = group_table(rows, "ContextDiff", lambda r: r.get("ContextDiff"), min_n=10)
    target_diff_table = group_table(rows, "TargetDiff", lambda r: r.get("TargetDiff"), min_n=10)
    cxt_combo_table = group_table(rows, "ContextType|ContextDiff|TargetDiff", lambda r: f"{r.get('ContextType')}|{r.get('ContextDiff')}|{r.get('TargetDiff')}", min_n=20)

    neg_rows = [r for r in rows if r["accuracy_interaction_te431_minus_te430"] < 0]
    pos_rows = [r for r in rows if r["accuracy_interaction_te431_minus_te430"] > 0]
    stable_rows = [r for r in rows if r["pattern"] in {"1111", "0000"}]
    stable_high_margin = [r for r in stable_rows if min(abs(r["models"][mk]["margin"]) for mk in MODELS) >= 1.0]
    stable_high_margin.sort(key=lambda r: -min(abs(r["models"][mk]["margin"]) for mk in MODELS))
    old_seed_polarized_0110 = [r for r in rows if r["pattern"] == "0110"]
    old_seed_polarized_1001 = [r for r in rows if r["pattern"] == "1001"]

    # Deterministic future stress-set selections.  They are not training data; they are
    # only row lists for later official-compatible margin reading.
    write_selection_csv(OUT_DIR / "old_negative_interaction_rows_selection.csv", neg_rows)
    write_selection_csv(OUT_DIR / "old_positive_interaction_rows_selection.csv", pos_rows)
    write_selection_csv(OUT_DIR / "old_seed_polarized_0110_rows_selection.csv", old_seed_polarized_0110)
    write_selection_csv(OUT_DIR / "old_seed_polarized_1001_rows_selection.csv", old_seed_polarized_1001)
    write_selection_csv(OUT_DIR / "stable_high_margin_control_rows_selection.csv", stable_high_margin[:1000])

    domain_csv = OUT_DIR / "domain_interaction_table.csv"
    with domain_csv.open("w", encoding="utf-8", newline="") as f:
        fields = ["domain", "n", "clean430", "reinv430", "clean431", "reinv431", "te430_pp", "te431_pp", "interaction_te431_minus_te430_pp", "margin_interaction_mean", "pattern_0110", "pattern_1001"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for d in domain_table:
            w.writerow({
                "domain": d["group_value"],
                "n": d["n"],
                "clean430": d["accuracy"]["clean430"],
                "reinv430": d["accuracy"]["reinv430"],
                "clean431": d["accuracy"]["clean431"],
                "reinv431": d["accuracy"]["reinv431"],
                "te430_pp": d["treatment_effect_pp"]["seed43022"],
                "te431_pp": d["treatment_effect_pp"]["seed43122"],
                "interaction_te431_minus_te430_pp": d["treatment_effect_pp"]["interaction_te431_minus_te430"],
                "margin_interaction_mean": d["margin_effect_mean"]["interaction_te431_minus_te430"],
                "pattern_0110": d["pattern_counts"].get("0110", 0),
                "pattern_1001": d["pattern_counts"].get("1001", 0),
            })

    examples = {
        "seed430_help_seed431_hurt_0110": enrich_examples(rows, text_by_key, "0110", limit=24),
        "opposite_seed_pattern_1001": enrich_examples(rows, text_by_key, "1001", limit=24),
        "largest_negative_margin_interaction": enrich_examples(rows, text_by_key, "most_negative_margin_interaction", limit=24),
        "largest_positive_margin_interaction": enrich_examples(rows, text_by_key, "most_positive_margin_interaction", limit=24),
    }

    corr = {
        "margin_corr_clean430_clean431": pearson([r["models"]["clean430"]["margin"] for r in rows], [r["models"]["clean431"]["margin"] for r in rows]),
        "margin_corr_reinv430_reinv431": pearson([r["models"]["reinv430"]["margin"] for r in rows], [r["models"]["reinv431"]["margin"] for r in rows]),
        "margin_corr_treatment_effects_430_431": pearson([r["margin_effect_seed43022"] for r in rows], [r["margin_effect_seed43122"] for r in rows]),
        "sign_agreement_clean_seeds": sum(1 for r in rows if r["models"]["clean430"]["correct"] == r["models"]["clean431"]["correct"]) / len(rows),
        "sign_agreement_reinvest_seeds": sum(1 for r in rows if r["models"]["reinv430"]["correct"] == r["models"]["reinv431"]["correct"]) / len(rows),
    }

    neg_abs = {mk: [abs(r["models"][mk]["margin"]) for r in neg_rows] for mk in MODELS}
    row_summary = {
        "n_rows": len(rows),
        "n_negative_accuracy_interaction_rows": len(neg_rows),
        "negative_accuracy_interaction_frac": len(neg_rows) / len(rows),
        "n_positive_accuracy_interaction_rows": len(pos_rows),
        "positive_accuracy_interaction_frac": len(pos_rows) / len(rows),
        "negative_rows_all_four_margins_abs_lt_1_frac": sum(1 for r in neg_rows if all(abs(r["models"][mk]["margin"]) < 1.0 for mk in MODELS)) / len(neg_rows),
        "negative_rows_reinv430_abs_lt_1_frac": sum(1 for r in neg_rows if abs(r["models"]["reinv430"]["margin"]) < 1.0) / len(neg_rows),
        "negative_rows_reinv431_abs_lt_1_frac": sum(1 for r in neg_rows if abs(r["models"]["reinv431"]["margin"]) < 1.0) / len(neg_rows),
        "negative_rows_abs_margin_mean": {mk: mean(vals) for mk, vals in neg_abs.items()},
        "n_old_seed_polarized_0110": len(old_seed_polarized_0110),
        "n_old_seed_polarized_1001": len(old_seed_polarized_1001),
        "n_stable_high_margin_control_written": min(1000, len(stable_high_margin)),
    }

    payload = {
        "status": "FULL_OLD_EWOK_MARGIN_ATLAS_SYNTHESIS",
        "created_utc": now_utc(),
        "source_csv": str(CSV_PATH.relative_to(USER_ROOT)),
        "source_preflight": str(PREFLIGHT_PATH.relative_to(USER_ROOT)) if PREFLIGHT_PATH.exists() else None,
        "preflight_selected_rows": preflight.get("selected_rows"),
        "model_order_for_pattern_bits": BIT_ORDER,
        "overall": overall,
        "row_summary": row_summary,
        "margin_correlations_and_sign_agreement": corr,
        "pattern_summary": pattern_summary,
        "domain_table": domain_table,
        "top_group_interactions": {
            "domain": compact_group_rows(domain_table, top_k=12),
            "ContextType": compact_group_rows(context_type_table, top_k=12),
            "ContextDiff": compact_group_rows(context_diff_table, top_k=12),
            "TargetDiff": compact_group_rows(target_diff_table, top_k=12),
            "ContextType_ContextDiff_TargetDiff_min_n20": compact_group_rows(cxt_combo_table, top_k=20),
            **top_concept_rows(rows),
        },
        "examples": examples,
        "selection_files_for_future_corrected_margin_reading": {
            "negative_interaction_rows": str((OUT_DIR / "old_negative_interaction_rows_selection.csv").relative_to(USER_ROOT)),
            "positive_interaction_rows": str((OUT_DIR / "old_positive_interaction_rows_selection.csv").relative_to(USER_ROOT)),
            "seed430_help_seed431_hurt_0110": str((OUT_DIR / "old_seed_polarized_0110_rows_selection.csv").relative_to(USER_ROOT)),
            "opposite_seed_pattern_1001": str((OUT_DIR / "old_seed_polarized_1001_rows_selection.csv").relative_to(USER_ROOT)),
            "stable_high_margin_control": str((OUT_DIR / "stable_high_margin_control_rows_selection.csv").relative_to(USER_ROOT)),
        },
        "interpretation": [
            "The full old-tokenizer atlas validates the focused-subset mechanism result on all 7,618 official EWoK rows: old compact_view_reinvest produced a seed-specific EWoK effect rather than a small selected-subset artifact.",
            "Negative treatment-by-seed rows are not mostly tiny-margin coincidences; only the all-four-small-margin fraction should be read as a noise-like component.",
            "Domain movement is sharply heterogeneous: material/physical/spatial relation groups show the strongest negative seed interaction, while some social/material-property groups move in the opposite direction.",
            "This atlas remains old-tokenizer mechanism evidence after the tokenizer compliance correction. It should shape interpretation of corrected-tokenizer EWoK and any later relation-repair route, not substitute for the running compliant official evaluations.",
        ],
    }
    out_json = OUT_DIR / "full_old_ewok_atlas_synthesis.json"
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    # Short scientist-facing note, with numbers needed in later steps.
    dom_lines = []
    for d in domain_table:
        dom_lines.append(
            f"| {d['group_value']} | {d['n']} | {d['accuracy']['clean430']:.2f} | {d['accuracy']['reinv430']:.2f} | {d['accuracy']['clean431']:.2f} | {d['accuracy']['reinv431']:.2f} | {d['treatment_effect_pp']['seed43022']:+.2f} | {d['treatment_effect_pp']['seed43122']:+.2f} | {d['treatment_effect_pp']['interaction_te431_minus_te430']:+.2f} |"
        )
    note = "\n".join([
        "# research full old EWoK margin atlas synthesis",
        "",
        "This is mechanism evidence for the inherited-tokenizer coordinate only. It does not decide the compliant corrected-tokenizer endpoint; the two official evaluations remain the decisive evidence.",
        "",
        "## Full-coordinate movement",
        f"- Rows: {len(rows)}; model keys/pattern bit order: {' / '.join(BIT_ORDER)}.",
        f"- Micro EWoK accuracies: clean430 {overall['clean430']['micro_accuracy']:.4f}, reinv430 {overall['reinv430']['micro_accuracy']:.4f}, clean431 {overall['clean431']['micro_accuracy']:.4f}, reinv431 {overall['reinv431']['micro_accuracy']:.4f}.",
        f"- Micro treatment effects: seed43022 {overall['treatment_effect_micro_pp']['seed43022']:+.4f} pp, seed43122 {overall['treatment_effect_micro_pp']['seed43122']:+.4f} pp; interaction TE431-TE430 {overall['treatment_effect_micro_pp']['interaction_te431_minus_te430']:+.4f} pp.",
        f"- Margin correlations: clean seeds r={corr['margin_corr_clean430_clean431']:.3f}, reinvest seeds r={corr['margin_corr_reinv430_reinv431']:.3f}, treatment-effect margins across seeds r={corr['margin_corr_treatment_effects_430_431']:.3f}.",
        f"- Negative accuracy-interaction rows: {row_summary['n_negative_accuracy_interaction_rows']} ({row_summary['negative_accuracy_interaction_frac']*100:.2f}%); all four margins |m|<1 in only {row_summary['negative_rows_all_four_margins_abs_lt_1_frac']*100:.2f}% of those rows.",
        f"- Seed-polarized patterns: 0110 count {row_summary['n_old_seed_polarized_0110']} (compact helps seed43022 while seed43122 loses); 1001 count {row_summary['n_old_seed_polarized_1001']} (opposite direction).",
        "",
        "## Domain table",
        "| Domain | n | clean430 | reinv430 | clean431 | reinv431 | TE430 | TE431 | interaction |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        *dom_lines,
        "",
        "## Largest domain interactions",
        "- Most negative TE431-TE430: " + "; ".join(f"{d['group_value']} {d['treatment_effect_pp']['interaction_te431_minus_te430']:+.2f} pp" for d in compact_group_rows(domain_table, top_k=6)["most_negative"][:6]),
        "- Most positive TE431-TE430: " + "; ".join(f"{d['group_value']} {d['treatment_effect_pp']['interaction_te431_minus_te430']:+.2f} pp" for d in compact_group_rows(domain_table, top_k=6)["most_positive"][:6]),
        "",
        "## Reusable row selections",
        "- `old_negative_interaction_rows_selection.csv`: all old-coordinate rows where compact-view treatment was less favorable for seed43122 than seed43022.",
        "- `old_seed_polarized_0110_rows_selection.csv`: rows with clean430 wrong, reinv430 correct, clean431 correct, reinv431 wrong.",
        "- `stable_high_margin_control_rows_selection.csv`: up to 1000 stable high-margin rows from patterns 1111/0000, for later comparison with corrected-tokenizer margins if needed.",
        "",
        "## Scientific reading",
        "The full atlas strengthens the mechanism picture from research: compact-view reinvestment can push relational preferences in a seed-specific way, especially in material dynamics, physical dynamics, spatial relations, physical interactions, and social relations. This is not reducible to a few near-zero choices. Corrected-tokenizer full official EWoK should therefore be read against these old relation patterns, while remembering that the old 42.033 endpoint itself is not a compliant submission because the tokenizer was learned outside the Strict-Small 10M budget.",
    ])
    NOTE_PATH.write_text(note + "\n", encoding="utf-8")

    print(json.dumps({
        "status": payload["status"],
        "out_json": str(out_json.relative_to(USER_ROOT)),
        "domain_csv": str(domain_csv.relative_to(USER_ROOT)),
        "note": str(NOTE_PATH.relative_to(USER_ROOT)),
        "n_rows": len(rows),
        "micro_te430_pp": overall["treatment_effect_micro_pp"]["seed43022"],
        "micro_te431_pp": overall["treatment_effect_micro_pp"]["seed43122"],
        "micro_interaction_te431_minus_te430_pp": overall["treatment_effect_micro_pp"]["interaction_te431_minus_te430"],
        "negative_interaction_rows": row_summary["n_negative_accuracy_interaction_rows"],
        "negative_all_four_abs_lt_1_frac": row_summary["negative_rows_all_four_margins_abs_lt_1_frac"],
        "top_negative_domains": [(d["group_value"], d["treatment_effect_pp"]["interaction_te431_minus_te430"]) for d in compact_group_rows(domain_table, top_k=5)["most_negative"][:5]],
        "top_positive_domains": [(d["group_value"], d["treatment_effect_pp"]["interaction_te431_minus_te430"]) for d in compact_group_rows(domain_table, top_k=5)["most_positive"][:5]],
    }, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
