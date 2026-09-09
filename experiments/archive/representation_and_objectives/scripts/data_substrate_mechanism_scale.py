#!/usr/bin/env python3
"""research CPU analysis for a mechanism-scale FineWeb source+compact route.

This script reads existing experiment artifacts only. It does not create a training
corpus, does not train a tokenizer/model, and does not run official evaluation.
It answers whether a post-SGCR data-substrate route should be framed as broad
source+compact pairing rather than U256 or leader imitation, and what cheap
preparation is needed before H100 time.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path("experiments/archive/representation_and_objectives")
A02_ROOT = Path("experiments/archive/frontier_consolidation")
OUT_DIR = ROOT / "data/data_substrate_mechanism_scale"
NOTE = (ROOT.parents[2] / 'research/notes/representation_and_objectives/data_substrate_mechanism_scale.md')

FILES = {
    "current_source_breakdown": ROOT / "data/corpus_lineage_multiplicity_audit/source_class_word_breakdown_10M.csv",
    "current_target_burden": ROOT / "data/wwm_target_burden_map/source_class_target_burden.csv",
    "u256_note_json": ROOT / "data/u256_leverage_and_data_route_reading/u256_leverage_and_data_route_reading.json",
    "v3_summary": ROOT / "data/live_fineweb_source_selector_v3/live_fineweb_source_selector_v3_summary.json",
    "v3_stable": ROOT / "data/live_fineweb_source_selector_v3/live_fineweb_selector_v3_stable.jsonl",
    "v5_summary": ROOT / "data/live_fineweb_selector_v5_strictstable/live_fineweb_selector_v5_summary.json",
    "v5_kept": ROOT / "data/live_fineweb_selector_v5_strictstable/live_fineweb_selector_v5_kept.jsonl",
    "quality_summary": ROOT / "data/live_fineweb_quality_tiers/live_fineweb_quality_tiers_summary.json",
    "quality_kept": ROOT / "data/live_fineweb_quality_tiers/live_fineweb_quality_v2_kept.jsonl",
    "a02_compact_metadata": A02_ROOT / "data/density_core_reinvestment_medium_riskhard/density_core_reinvestment_metadata.json",
    "a02_compact_pairs": A02_ROOT / "data/density_core_reinvestment_medium_riskhard/selected_compact_reinvest_pairs.jsonl",
    "sgcr_partial_payload": ROOT / "data/legal40k_12x384_sgcrK50d64_seed43022_full_eval/per_target/legal40k_12x384_sgcrK50d64_seed43022.json",
    "depth_vector": ROOT / "data/depth_vector_decision/depth_vector_decision.json",
    "legal40_vector": ROOT / "data/legal40k_two_seed_comparison/legal40k_two_seed_comparison.json",
}

LEADER = {
    "BLiMP": 67.20,
    "Supplement": 56.01,
    "EWoK": 56.07,
    "Entity": 28.45,
    "COMPS": 53.57,
    "GlobalPIQA": 39.67,
    "SuperGLUE": 69.79,
    "Reading": 5.42,
    "AoA": 0.0,
    "Overall": 41.80,
}


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def jsonl_iter(path: Path):
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def norm_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def text_hash(text: str) -> str:
    return hashlib.sha1(norm_text(text).encode("utf-8")).hexdigest()


def domain_tokens(rec: dict[str, Any], flavor: str) -> list[str]:
    vals: list[str] = []
    for key in ["selector_v5_types", "selector_v3_types", "core_domain_hits", "domain_hits"]:
        v = rec.get(key)
        if isinstance(v, list):
            vals.extend(str(x) for x in v)
    if flavor == "quality":
        for x in rec.get("core_domain_hits", []) or []:
            vals.append(str(x))
    if not vals:
        vals = ["no_domain"]
    return sorted(set(vals))


def source_word(rec: dict[str, Any]) -> int:
    for k in ["selector_v5_words", "selector_v3_words", "words", "source_words"]:
        v = rec.get(k)
        if isinstance(v, (int, float)):
            return int(v)
    text = rec.get("selector_v5_text") or rec.get("selector_v3_repaired_text") or rec.get("text") or rec.get("source_text") or ""
    return len(str(text).split())


def source_text(rec: dict[str, Any]) -> str:
    return str(rec.get("selector_v5_text") or rec.get("selector_v3_repaired_text") or rec.get("text") or rec.get("source_text") or "")


def summarize_words(words: list[int]) -> dict[str, Any]:
    if not words:
        return {"n": 0, "sum": 0}
    s = sorted(words)
    def pct(p: float) -> int:
        return s[min(len(s)-1, max(0, int(round((len(s)-1)*p))))]
    return {
        "n": len(words),
        "sum": int(sum(words)),
        "mean": round(float(statistics.mean(words)), 4),
        "median": float(statistics.median(words)),
        "p90": pct(0.90),
        "p95": pct(0.95),
        "min": s[0],
        "max": s[-1],
    }


def summarize_source_pool(name: str, path: Path, flavor: str) -> dict[str, Any]:
    rows = 0
    words: list[int] = []
    docs: set[str] = set()
    domains = Counter()
    hashes: set[str] = set()
    examples: list[dict[str, Any]] = []
    for rec in jsonl_iter(path):
        rows += 1
        w = source_word(rec)
        words.append(w)
        docs.add(str(rec.get("doc_id", rec.get("doc_index", ""))))
        for d in domain_tokens(rec, flavor):
            domains[d] += 1
        h = text_hash(source_text(rec))
        hashes.add(h)
        if len(examples) < 3:
            examples.append({"text": source_text(rec)[:220], "words": w, "domains": domain_tokens(rec, flavor), "doc_id": rec.get("doc_id")})
    return {
        "name": name,
        "path": str(path),
        "rows": rows,
        "source_word_stats": summarize_words(words),
        "unique_docs": len(docs),
        "unique_normalized_texts": len(hashes),
        "domain_counts": dict(domains.most_common()),
        "examples": examples,
    }


def load_source_union(pool_specs: list[tuple[str, Path, str]]) -> dict[str, Any]:
    union: dict[str, dict[str, Any]] = {}
    by_pool = {}
    for name, path, flavor in pool_specs:
        pool_hashes = set()
        for rec in jsonl_iter(path):
            text = source_text(rec)
            h = text_hash(text)
            pool_hashes.add(h)
            w = source_word(rec)
            ds = domain_tokens(rec, flavor)
            if h not in union:
                union[h] = {
                    "hash": h,
                    "text": text,
                    "words": w,
                    "doc_id": str(rec.get("doc_id", rec.get("doc_index", ""))),
                    "domains": set(ds),
                    "pools": set([name]),
                }
            else:
                union[h]["pools"].add(name)
                union[h]["domains"].update(ds)
                if w > union[h]["words"]:
                    union[h]["words"] = w
                    union[h]["text"] = text
        by_pool[name] = {"unique_hashes": len(pool_hashes)}
    words = [int(v["words"]) for v in union.values()]
    docs = {v["doc_id"] for v in union.values()}
    dom = Counter()
    pool_membership = Counter()
    for v in union.values():
        for d in v["domains"]:
            dom[d] += 1
        for p in v["pools"]:
            pool_membership[p] += 1
    return {
        "unique_sources": len(union),
        "source_word_stats": summarize_words(words),
        "unique_docs_lower_bound": len(docs),
        "domain_counts": dict(dom.most_common()),
        "membership_counts": dict(pool_membership.most_common()),
        "by_pool_unique_hashes": by_pool,
    }


def summarize_compact_pairs(path: Path) -> dict[str, Any]:
    rows = 0
    source_words = []
    rewrite_words = []
    pair_words = []
    docs = set()
    domains = Counter()
    hashes = set()
    recalls = []
    entity_recalls = []
    number_recalls = []
    examples = []
    for rec in jsonl_iter(path):
        rows += 1
        sw = int(rec.get("source_words", len(str(rec.get("source_text", "")).split())))
        rw = int(rec.get("rewrite_words", len(str(rec.get("rewrite_text", "")).split())))
        pw = int(rec.get("pair_words", sw + rw))
        source_words.append(sw)
        rewrite_words.append(rw)
        pair_words.append(pw)
        docs.add(str(rec.get("doc_id", "")))
        hashes.add(text_hash(str(rec.get("source_text", ""))))
        for d in rec.get("domain_hits", []) or ["no_domain"]:
            domains[str(d)] += 1
        for k, arr in [("content_recall", recalls), ("entity_recall", entity_recalls), ("number_recall", number_recalls)]:
            v = rec.get(k)
            if isinstance(v, (int, float)):
                arr.append(float(v))
        if len(examples) < 3:
            examples.append({
                "source": str(rec.get("source_text", ""))[:180],
                "rewrite": str(rec.get("rewrite_text", ""))[:180],
                "source_words": sw,
                "rewrite_words": rw,
                "domains": rec.get("domain_hits", []),
            })
    ratio = sum(rewrite_words) / max(1, sum(source_words))
    return {
        "rows": rows,
        "source_words": int(sum(source_words)),
        "rewrite_words": int(sum(rewrite_words)),
        "pair_words": int(sum(pair_words)),
        "rewrite_to_source_ratio": ratio,
        "source_word_stats": summarize_words(source_words),
        "rewrite_word_stats": summarize_words(rewrite_words),
        "pair_word_stats": summarize_words(pair_words),
        "unique_docs": len(docs),
        "unique_normalized_source_texts": len(hashes),
        "domain_counts": dict(domains.most_common()),
        "content_recall_mean": statistics.mean(recalls) if recalls else None,
        "entity_recall_mean": statistics.mean(entity_recalls) if entity_recalls else None,
        "number_recall_mean": statistics.mean(number_recalls) if number_recalls else None,
        "examples": examples,
    }


def current_corpus_summary() -> dict[str, Any]:
    rows = read_csv(FILES["current_source_breakdown"])
    out = {}
    for r in rows:
        out[r["source_class"]] = {
            "rows_10M": int(r["rows_10M"]),
            "words_10M": int(float(r["words_10M"])),
            "fraction_words_10M": float(r["fraction_words_10M"]),
        }
    return out


def current_target_summary() -> dict[str, Any]:
    rows = read_csv(FILES["current_target_burden"])
    out = {}
    for r in rows:
        cls = r["source_class"]
        out[cls] = {k: (float(v) if "." in v else int(v)) for k, v in r.items() if k != "source_class"}
    return out


def sgcr_partial_reading() -> dict[str, Any]:
    p = FILES["sgcr_partial_payload"]
    if not p.exists():
        return {"status": "no_partial_payload"}
    payload = load_json(p)
    tasks = payload.get("tasks", {})
    vals: dict[str, float] = {}
    for col in ["BLiMP", "Supplement", "Entity", "COMPS"]:
        rec = tasks.get(col)
        if isinstance(rec, dict) and rec.get("score") is not None:
            vals[col] = float(rec["score"])
    gp = []
    for col in ["GlobalPIQA_parallel", "GlobalPIQA_nonparallel"]:
        rec = tasks.get(col)
        if isinstance(rec, dict) and rec.get("score") is not None:
            gp.append(float(rec["score"]))
    if len(gp) == 2:
        vals["GlobalPIQA"] = sum(gp) / 2
        vals["GlobalPIQA_parallel"] = gp[0]
        vals["GlobalPIQA_nonparallel"] = gp[1]
    rec = tasks.get("Reading")
    if isinstance(rec, dict) and isinstance(rec.get("scores"), dict) and rec["scores"].get("Reading") is not None:
        vals["Reading"] = float(rec["scores"]["Reading"])
    completed_for_overall = ["BLiMP", "Supplement", "Entity", "COMPS", "GlobalPIQA", "Reading"]
    completed_sum = sum(vals[k] for k in completed_for_overall if k in vals)
    needed_total = LEADER["Overall"] * 9.0
    missing_needed_if_aoa0 = needed_total - completed_sum
    # missing columns would be EWoK, SuperGLUE, AoA; AoA has been 0.0 in visible endpoint records.
    missing_two_needed_if_aoa0 = missing_needed_if_aoa0
    required_avg_ewok_superglue_if_aoa0 = missing_two_needed_if_aoa0 / 2.0
    return {
        "status": "partial_zero_shot_reading_available_superglue_failed_oom",
        "partial_scores": vals,
        "completed_columns_for_overall": completed_for_overall,
        "completed_sum": completed_sum,
        "needed_total_for_41p80": needed_total,
        "required_EWoK_plus_SuperGLUE_sum_if_AoA_zero": missing_two_needed_if_aoa0,
        "required_EWoK_SuperGLUE_average_if_AoA_zero": required_avg_ewok_superglue_if_aoa0,
        "leader_EWoK_plus_SuperGLUE": LEADER["EWoK"] + LEADER["SuperGLUE"],
        "with_leader_EWoK_and_SuperGLUE_and_AoA0_overall": (completed_sum + LEADER["EWoK"] + LEADER["SuperGLUE"] + 0.0) / 9.0,
        "scientific_reading": "The partial official-compatible columns already make a near-frontier SGCR endpoint unlikely unless the missing EWoK/SuperGLUE columns greatly exceed the public leader; complete collation is still required before closing the route formally.",
    }


def estimate_scales(compact: dict[str, Any], source_union: dict[str, Any], current: dict[str, Any]) -> list[dict[str, Any]]:
    ratio = compact["rewrite_to_source_ratio"]
    cached_source_words = source_union["source_word_stats"]["sum"]
    current_fineweb_pair = current["fineweb_source_qwen_compact_rewrite_pair_row"]["words_10M"]
    current_official_qwen = current["inherited_official_source_qwen_paraphrase_pair_row"]["words_10M"]
    out = []
    for budget in [1_000_000, 1_500_000, 1_750_000, 2_000_000, 2_500_000, 3_500_000, 5_000_000]:
        source_needed = budget / (1.0 + ratio)
        rewrite_needed = budget - source_needed
        add_pair_words = max(0, budget - current_fineweb_pair)
        qwen_after_replace_first = max(0, current_official_qwen - add_pair_words)
        official_source_words_preserved_if_replace_qwen_first = current["official_babylm_source_row"]["words_10M"]
        official_source_displaced_after_qwen_exhausted = max(0, add_pair_words - current_official_qwen)
        out.append({
            "fineweb_pair_budget_words": budget,
            "fineweb_pair_fraction_of_10M": budget / 10_000_000,
            "multiple_of_current_fineweb_pair_words": budget / current_fineweb_pair,
            "fraction_of_public_leader_pair_words": budget / 9_999_969,
            "source_words_needed_at_current_compact_ratio": round(source_needed, 1),
            "rewrite_words_needed_at_current_compact_ratio": round(rewrite_needed, 1),
            "cached_source_words_fraction_of_need": cached_source_words / source_needed if source_needed else None,
            "cached_sources_enough_without_new_scan": cached_source_words >= source_needed,
            "current_official_qwen_words_remaining_if_extra_pair_replaces_that_block_first": qwen_after_replace_first,
            "official_source_words_preserved_if_replace_qwen_first": official_source_words_preserved_if_replace_qwen_first,
            "official_source_words_displaced_after_qwen_block_exhausted": official_source_displaced_after_qwen_exhausted,
        })
    return out


def write_csv_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    keys = list(rows[0].keys())
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    current = current_corpus_summary()
    target_burden = current_target_summary()
    summaries = {
        "v3_stable": summarize_source_pool("v3_stable", FILES["v3_stable"], "v3"),
        "v5_strictstable": summarize_source_pool("v5_strictstable", FILES["v5_kept"], "v5"),
        "quality_v2_kept": summarize_source_pool("quality_v2_kept", FILES["quality_kept"], "quality"),
    }
    compact = summarize_compact_pairs(FILES["a02_compact_pairs"])
    pool_specs = [
        ("v3_stable", FILES["v3_stable"], "v3"),
        ("v5_strictstable", FILES["v5_kept"], "v5"),
        ("quality_v2_kept", FILES["quality_kept"], "quality"),
        ("a02_selected_compact_sources", FILES["a02_compact_pairs"], "compact"),
    ]
    source_union = load_source_union(pool_specs)
    scale_rows = estimate_scales(compact, source_union, current)
    write_csv_rows(OUT_DIR / "scale_options.csv", scale_rows)

    sgcr = sgcr_partial_reading()
    u256 = load_json(FILES["u256_note_json"]) if FILES["u256_note_json"].exists() else None
    a02_meta = load_json(FILES["a02_compact_metadata"])
    summary = {
        "status": "DATA_SUBSTRATE_MECHANISM_SCALE",
        "purpose": "CPU-only route reading for broad FineWeb source+compact pairing as an extension of the replicated compact-view principle, distinct from U256 and from merely copying the public leader.",
        "no_training_or_eval_launched_by_this_script": True,
        "inputs": {k: str(v) for k, v in FILES.items() if v.exists()},
        "current_corpus_words": current,
        "current_target_burden_by_source_class": target_burden,
        "source_pool_summaries": summaries,
        "a02_compact_reinvest_pair_summary_from_jsonl": compact,
        "a02_compact_reinvest_metadata_selection": a02_meta.get("selection", {}),
        "cached_source_union": source_union,
        "scale_options": scale_rows,
        "u256_reference": {
            "hidden_words_per_10M": (u256 or {}).get("u256_mass", {}).get("hidden_words_per_10M") or (u256 or {}).get("hidden_words_per_10m"),
            "fineweb_hidden_words": (u256 or {}).get("u256_mass", {}).get("hidden_fineweb_words") or (u256 or {}).get("hidden_fineweb_words"),
            "active_token_relative_increase": (u256 or {}).get("u256_mass", {}).get("active_token_relative_increase"),
            "masked_target_relative_increase": (u256 or {}).get("u256_mass", {}).get("masked_target_relative_increase"),
        },
        "sgcr_partial_endpoint_reading": sgcr,
        "mechanism_reading": {
            "replicated_principle": "Small-scale source+compact pairing was the strongest validated data mechanism, but current FineWeb compact-pair mass is only 423,511 words.",
            "not_leader_copy": "The proposed successor is not to import the leader recipe wholesale; it scales the source-aligned compact-view mechanism with matched source-repeat controls and exact provenance.",
            "source_vs_pairing_distinction": "A broad source-repeat arm with the same selected FineWeb sources, row packing, word budget, model family, and per-corpus legal tokenizer is needed to distinguish adding FineWeb content from compact-view consolidation.",
            "budget_strategy": "Replacing the inherited official-source Qwen paraphrase block before displacing official originals can raise FineWeb source+compact mass to about 1.75-2.0M words while preserving the 7.92M original official-source words.",
            "why_u256_is_lower_priority": "U256 changes only about 1.7% realized targets and exposes almost no FineWeb compact material, while the broad data route changes the source and view substrate at 4-5x current FineWeb pair mass.",
        },
        "proposed_next_materialization": {
            "first_scale": "FW1p75M source+compact family",
            "fineweb_pair_words": 1_750_000,
            "source_words_needed_approx": next(r for r in scale_rows if r["fineweb_pair_budget_words"] == 1_750_000)["source_words_needed_at_current_compact_ratio"],
            "rewrite_words_needed_approx": next(r for r in scale_rows if r["fineweb_pair_budget_words"] == 1_750_000)["rewrite_words_needed_at_current_compact_ratio"],
            "arms": [
                "compact_view: selected FineWeb source followed by faithful compact rewrite, row-packed like research, official original sources preserved, most inherited official-source Qwen rows displaced first",
                "source_repeat: same selected FineWeb sources and row lengths, repeated/source-only material replacing the compact rewrites, same official original source preservation",
            ],
            "pre_h100_work": [
                "deduplicate and freeze selected FineWeb sources independent of evaluation text",
                "generate or reuse faithful compact rewrites with entity/number/content retention measurements and source-line provenance",
                "materialize matched 10M corpora with exact word counts, row hashes, source accounting, and tokenizer-training provenance",
                "train legal tokenizers separately on each exact 10M corpus and measure support/visibility/WWM target burden before any model run",
            ],
            "first_expensive_use_if_sgcr_is_weak": "Launch treatment and source-repeat control only after the corpora and tokenizer measurements show the 1.75M pair family is mechanically clean; use the best completed compliant backbone unless the completed SGCR vector changes that reading.",
        },
    }
    out_json = OUT_DIR / "data_substrate_mechanism_scale.json"
    out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    # Human-readable note.
    enough_175 = next(r for r in scale_rows if r["fineweb_pair_budget_words"] == 1_750_000)
    enough_2m = next(r for r in scale_rows if r["fineweb_pair_budget_words"] == 2_000_000)
    partial = sgcr.get("partial_scores", {}) if isinstance(sgcr, dict) else {}
    note = []
    note.append("# research — data-substrate mechanism scale reading\n")
    note.append("CPU-only analysis of existing pools and partial SGCR evaluation artifacts. No model training, no tokenizer training, and no official evaluation were run by this script.\n")
    note.append("## SGCR partial endpoint context\n")
    if partial:
        note.append(f"The failed SGCR evaluator already produced completed official-compatible partial columns before SuperGLUE boolq OOM: BLiMP {partial.get('BLiMP'):.2f}, Supplement {partial.get('Supplement'):.2f}, Entity {partial.get('Entity'):.2f}, COMPS {partial.get('COMPS'):.2f}, GlobalPIQA {partial.get('GlobalPIQA'):.2f}, Reading {partial.get('Reading'):.2f}. With AoA at zero, EWoK+SuperGLUE would need {sgcr['required_EWoK_plus_SuperGLUE_sum_if_AoA_zero']:.2f} total (average {sgcr['required_EWoK_SuperGLUE_average_if_AoA_zero']:.2f}) to reach 41.80, above the public leader's EWoK+SuperGLUE {sgcr['leader_EWoK_plus_SuperGLUE']:.2f}. This does not replace the resumed full official collation, but it makes near-frontier SGCR unlikely and argues against seed43122 or uniform SGCR work unless the completed vector overturns the pattern.\n")
    else:
        note.append("No usable SGCR partial payload was available.\n")
    note.append("## Existing data scale\n")
    note.append(f"Current corpus words: official original {current['official_babylm_source_row']['words_10M']:,}, inherited official-source Qwen paraphrase pairs {current['inherited_official_source_qwen_paraphrase_pair_row']['words_10M']:,}, FineWeb source+compact pairs {current['fineweb_source_qwen_compact_rewrite_pair_row']['words_10M']:,}. The FineWeb compact block is {current['fineweb_source_qwen_compact_rewrite_pair_row']['fraction_words_10M']*100:.2f}% of the 10M budget.\n")
    note.append(f"A02 compact-reinvest pairs contain {compact['rows']:,} pairs, {compact['source_words']:,} source words and {compact['rewrite_words']:,} rewrite words, rewrite/source ratio {compact['rewrite_to_source_ratio']:.3f}, pair words {compact['pair_words']:,}. Mean content recall {compact['content_recall_mean']:.3f}, entity recall {compact['entity_recall_mean']:.3f}, number recall {compact['number_recall_mean']:.3f}.\n")
    note.append("## Cached source pool capacity\n")
    note.append(f"Unioning A01 v3 stable, v5 strict-stable, quality_v2, and A02 compact sources by normalized text gives {source_union['unique_sources']:,} unique source sentences, {source_union['source_word_stats']['sum']:,} source words, and at least {source_union['unique_docs_lower_bound']:,} document ids. Domain counts in the union are led by {list(source_union['domain_counts'].items())[:8]}.\n")
    note.append(f"At the current compact rewrite/source ratio, a 1.75M FineWeb source+compact family needs about {enough_175['source_words_needed_at_current_compact_ratio']:,.0f} source words and {enough_175['rewrite_words_needed_at_current_compact_ratio']:,.0f} rewrite words; cached source coverage is {enough_175['cached_source_words_fraction_of_need']:.2f}x. A 2.0M family needs about {enough_2m['source_words_needed_at_current_compact_ratio']:,.0f} source words; cached source coverage is {enough_2m['cached_source_words_fraction_of_need']:.2f}x.\n")
    note.append("## Mechanism-scale route\n")
    note.append("The stronger successor to U256 is a broad source+compact family, not a copy of the public leader. It should scale the replicated compact-view mechanism from 0.423M FineWeb pair words to roughly 1.75M first, using sources selected independently of evaluation text and compact rewrites that preserve entities, numbers, and source propositions.\n")
    note.append("The route must distinguish compact-view consolidation from merely adding FineWeb. The matched source-repeat arm should use the same selected FineWeb sources, row packing, word budget, official-original preservation, model family, and per-corpus compliant tokenizer, but replace compact rewrites by repeated/source-only material. Because tokenizer learning is part of the legal data package, each arm must train its tokenizer only on its exact 10M corpus; the result is a compliant data+tokenizer package comparison rather than an illegal shared-tokenizer isolation.\n")
    note.append("A useful budget strategy is to displace the inherited official-source Qwen paraphrase rows before displacing official originals. This can raise FineWeb source+compact mass to about 1.75-2.0M words while retaining the 7.92M original official-source words, giving a large change in factual/physical source substrate without discarding the core BabyLM language mixture.\n")
    note.append("## Before H100 time\n")
    note.append("Materialize only research-facing assets first: frozen source list, rewrite provenance and retention tables, matched source-repeat corpus, exact word accounting, row hashes, tokenizer-only support/visibility/WWM burden measurements. If SGCR remains weak after resumed full collation and these assets are mechanically clean, the proposed comparison is the FW1p75M pair family, followed by the paired treatment/control on the best completed compliant backbone unless the finished SGCR vector changes the backbone choice.\n")
    note.append(f"\nFiles: JSON `{out_json}`; scale CSV `{OUT_DIR / 'scale_options.csv'}`.\n")
    NOTE.write_text("".join(note), encoding="utf-8")
    print(json.dumps({
        "status": summary["status"],
        "out_json": str(out_json),
        "note": str(NOTE),
        "cached_source_words": source_union["source_word_stats"]["sum"],
        "fineweb_1p75_cached_coverage": enough_175["cached_source_words_fraction_of_need"],
        "sgcr_partial_required_ewok_superglue_avg_if_aoa0": sgcr.get("required_EWoK_SuperGLUE_average_if_AoA_zero") if isinstance(sgcr, dict) else None,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
