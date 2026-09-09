#!/usr/bin/env python3
"""Measure source-pool relation motifs and benchmark lexical overlap for FineWeb routes.

This is a CPU-only research aid for the next FineWeb family, not training evidence.
It compares the cached seqsafe96 FineWeb replacement block, its official length-matched
control block, and the live v3/v5 candidate sources against the official central
BabyLM Strict-Small benchmark text.  It asks two practical questions:

1. Does a source pool expose more of the object/property/relation motifs that dominate
   the current deficit cluster (EWoK, Entity, COMPS, GlobalPIQA)?
2. Are there suspicious exact benchmark n-gram overlaps that a future materializer must
   exclude before any source+view training?

The analysis is deliberately lexical and conservative.  It cannot predict downstream
scores; it preserves interpretable source-selection evidence for later route decisions.
"""
from __future__ import annotations

import json
import math
import pathlib
import re
from collections import Counter, defaultdict
from typing import Any, Iterable

ROOT = pathlib.Path("experiments/archive/representation_and_objectives")
WORK = ROOT
STRICT = pathlib.Path("experiments/archive/initial_model_studies/repos/babylm-eval/strict")
EVAL = STRICT / "evaluation_data/full_eval"
OUT_DIR = WORK / "data/fineweb_source_benchmark_overlap"
NOTE = (WORK.parents[2] / 'research/notes/representation_and_objectives/fineweb_source_benchmark_motif_overlap.md')

POOLS = {
    "cached_fineweb_seqsafe96_block": {
        "path": WORK / "training/data/cached_fineweb_seqsafe96_candidate/cleanqwen_seqsafe_fineweb_single_doc_10M.jsonl",
        "source_prefix": "fineweb_edu_random_quality_single_doc_seqsafe_cached_initial_model_studies",
        "text_key": "text",
        "words_key": "words",
        "meaning": "the 1.753M-word cached FineWeb block being tested by the repaired research run",
    },
    "official_lengthmatched_control_block": {
        "path": WORK / "training/data/cached_fineweb_seqsafe96_candidate/cleanqwen_official_lengthmatched_seqsafe_control_10M.jsonl",
        "source_prefix": "official_lengthmatched_to_seqsafe_fineweb",
        "text_key": "text",
        "words_key": "words",
        "meaning": "the matched official-corpus replacement block used as research control",
    },
    "live_v3_relation_rich_doccap8": {
        "path": WORK / "data/live_fineweb_source_selector_v3/live_fineweb_selector_v3_stable_doccap8.jsonl",
        "source_prefix": None,
        "text_key": "selector_v3_repaired_text",
        "fallback_text_key": "text",
        "words_key": "selector_v3_words",
        "meaning": "relation-rich live FineWeb candidate retaining more causal/discourse context",
    },
    "live_v5_self_contained_doccap8": {
        "path": WORK / "data/live_fineweb_selector_v5_strictstable/live_fineweb_selector_v5_doccap8.jsonl",
        "source_prefix": None,
        "text_key": "selector_v5_text",
        "fallback_text_key": "selector_v3_repaired_text",
        "words_key": "selector_v5_words",
        "meaning": "strict self-contained live FineWeb fact candidate",
    },
}

STOP = {
    "the", "and", "that", "this", "with", "for", "from", "into", "onto", "over", "under", "then", "than",
    "when", "what", "where", "which", "while", "because", "about", "after", "before", "there", "their", "them",
    "they", "these", "those", "would", "could", "should", "were", "was", "are", "is", "be", "been", "being",
    "have", "has", "had", "having", "will", "may", "might", "can", "cannot", "does", "did", "do", "done",
    "not", "but", "or", "if", "in", "on", "at", "to", "of", "as", "by", "an", "a", "it", "its", "his",
    "her", "hers", "him", "he", "she", "we", "you", "your", "our", "ours", "my", "me", "i", "all", "one",
    "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "some", "any", "each", "other",
    "same", "more", "most", "less", "least", "also", "only", "such", "like", "through", "between", "among",
}
WORD_RE = re.compile(r"[A-Za-z][A-Za-z'\-]*|\d+(?:\.\d+)?")

RELATION_MARKERS = {
    "causal": "cause causes caused causing lead leads led result results resulted make makes made prevent prevents prevented allow allows enabled enables increase increases decrease decreases reduce reduces damage damages break breaks broke broken improve improves affect affects change changes".split(),
    "physical_dynamic": "move moves moving moved push pushes pushed pull pulls pulled fall falls falling fell drop drops dropped bounce bounces bounced slide slides slid roll rolls rolled expand expands shrink shrinks melt melts melted freeze freezes frozen absorb absorbs heat heats cool cools bend bends stretch stretches".split(),
    "object_property": "hard soft heavy light rough smooth sharp blunt wet dry hot cold warm cool flexible rigid transparent opaque magnetic plastic metal wooden glass paper air water liquid solid gas pressure weight color size shape".split(),
    "spatial_state": "inside outside above below under over near far between around beside behind front left right contain contains contained holding placed put open closed empty full".split(),
    "social_agent": "person people child children teacher student mother father friend group help helps helped teach teaches learn learns ask asks tell tells give gives receive receives want wants know knows believe believes".split(),
    "temporal_process": "first next later earlier before after during finally eventually started begins became becomes continued ended discovered developed published founded built".split(),
}
REL_ALL = {w for vals in RELATION_MARKERS.values() for w in vals}


def iter_jsonl(path: pathlib.Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def raw_tokens(text: str) -> list[str]:
    return [m.group(0).lower().strip("'-") for m in WORD_RE.finditer(text or "") if m.group(0).strip("'-")]


def content_tokens(text: str) -> list[str]:
    toks = []
    for t in raw_tokens(text):
        if len(t) < 3 or t in STOP:
            continue
        toks.append(t)
    return toks


def ngrams(tokens: list[str], n: int) -> set[str]:
    if len(tokens) < n:
        return set()
    return {" ".join(tokens[i:i+n]) for i in range(len(tokens) - n + 1)}


def text_fields_for_record(row: dict[str, Any]) -> list[str]:
    vals: list[str] = []
    for k, v in row.items():
        if isinstance(v, str):
            if k.lower() in {"id", "uid", "language", "field", "linguistics_term", "negative_sample_type", "categories", "example_id"}:
                continue
            vals.append(v)
        elif isinstance(v, list):
            vals.extend(str(x) for x in v if isinstance(x, (str, int, float)))
    return vals


def add_eval_file(path: pathlib.Path, group: str, lex: dict[str, Counter], eval_grams: Counter, group_counts: Counter, phrase_examples: dict[str, list[str]]) -> None:
    for row in iter_jsonl(path):
        fields = text_fields_for_record(row)
        text = " ".join(fields)
        toks = content_tokens(text)
        lex[group].update(toks)
        group_counts[group] += 1
        for gram in ngrams(raw_tokens(text), 7):
            eval_grams[gram] += 1
        # Preserve a few high-information examples for interpreting lexical categories.
        if len(phrase_examples[group]) < 6 and text:
            phrase_examples[group].append(text[:240])


def build_benchmark_assets() -> dict[str, Any]:
    lex: dict[str, Counter] = defaultdict(Counter)
    eval_grams: Counter = Counter()
    group_counts: Counter = Counter()
    examples: dict[str, list[str]] = defaultdict(list)

    for p in sorted((EVAL / "ewok_filtered").glob("*.jsonl")):
        add_eval_file(p, "EWoK", lex, eval_grams, group_counts, examples)
    for p in sorted((EVAL / "comps").glob("*.jsonl")):
        add_eval_file(p, "COMPS", lex, eval_grams, group_counts, examples)
    for p in sorted((EVAL / "global_piqa_parallel").glob("*.jsonl")) + sorted((EVAL / "global_piqa_nonparallel").glob("*.jsonl")):
        add_eval_file(p, "GlobalPIQA", lex, eval_grams, group_counts, examples)
    for p in sorted((EVAL / "entity_tracking").glob("*.jsonl")):
        add_eval_file(p, "Entity", lex, eval_grams, group_counts, examples)

    lex_sets: dict[str, set[str]] = {}
    top_terms: dict[str, list[tuple[str, int]]] = {}
    for group, ctr in lex.items():
        # Use moderately frequent benchmark anchors, not every incidental word.
        total_records = max(group_counts[group], 1)
        min_count = 2 if total_records < 1000 else 5
        terms = {w for w, c in ctr.items() if c >= min_count and w not in STOP and len(w) >= 3}
        lex_sets[group] = terms
        top_terms[group] = ctr.most_common(40)

    return {
        "lex_sets": lex_sets,
        "eval_7grams": set(eval_grams.keys()),
        "eval_7gram_counts": eval_grams,
        "group_record_counts": dict(group_counts),
        "top_terms": top_terms,
        "examples": dict(examples),
    }


def get_pool_row_text(row: dict[str, Any], spec: dict[str, Any]) -> tuple[str, int]:
    key = spec.get("text_key")
    text = row.get(key) if key else None
    if not isinstance(text, str) or not text.strip():
        fk = spec.get("fallback_text_key")
        text = row.get(fk) if fk else row.get("text", "")
    words_key = spec.get("words_key")
    w = row.get(words_key) if words_key else row.get("words")
    try:
        words = int(w)
    except Exception:
        words = len(raw_tokens(text or ""))
    return str(text or ""), words


def iter_pool_rows(spec: dict[str, Any]) -> Iterable[dict[str, Any]]:
    prefix = spec.get("source_prefix")
    for row in iter_jsonl(pathlib.Path(spec["path"])):
        if prefix is not None and row.get("source") != prefix:
            continue
        yield row


def analyze_pool(name: str, spec: dict[str, Any], assets: dict[str, Any]) -> dict[str, Any]:
    lex_sets: dict[str, set[str]] = assets["lex_sets"]
    eval_grams: set[str] = assets["eval_7grams"]
    row_count = 0
    word_count = 0
    task_hit_rows = Counter()
    task_hits_total = Counter()
    task_hits_distinct_total = Counter()
    cross_task_rows = Counter()
    relation_hit_rows = Counter()
    relation_hits_total = Counter()
    leak_rows_ge1 = 0
    leak_rows_ge3 = 0
    leak_max = 0
    suspicious_samples: list[dict[str, Any]] = []
    match_samples: dict[str, list[dict[str, Any]]] = defaultdict(list)
    relation_samples: dict[str, list[dict[str, Any]]] = defaultdict(list)
    word_len_hist = Counter()

    for row in iter_pool_rows(spec):
        text, words = get_pool_row_text(row, spec)
        if not text.strip():
            continue
        row_count += 1
        word_count += words
        word_len_hist[min((words // 20) * 20, 200)] += 1
        toks_raw = raw_tokens(text)
        toks = [t for t in toks_raw if len(t) >= 3 and t not in STOP]
        token_ctr = Counter(toks)
        token_set = set(token_ctr)

        tasks_with_hit = 0
        for group, terms in lex_sets.items():
            matches = token_set & terms
            distinct = len(matches)
            total = sum(token_ctr[t] for t in matches)
            task_hits_total[group] += total
            task_hits_distinct_total[group] += distinct
            if distinct >= 2 or total >= 3:
                task_hit_rows[group] += 1
                tasks_with_hit += 1
                if len(match_samples[group]) < 8:
                    match_samples[group].append({
                        "text": text[:360],
                        "words": words,
                        "distinct_matches": distinct,
                        "sample_matches": sorted(matches)[:20],
                        "row_source": row.get("source"),
                        "doc_id": row.get("doc_id"),
                    })
        cross_task_rows[str(tasks_with_hit)] += 1

        for rgroup, terms in RELATION_MARKERS.items():
            matches = token_set & set(terms)
            total = sum(token_ctr[t] for t in matches)
            relation_hits_total[rgroup] += total
            if total > 0:
                relation_hit_rows[rgroup] += 1
                if len(relation_samples[rgroup]) < 5:
                    relation_samples[rgroup].append({"text": text[:360], "words": words, "matches": sorted(matches), "doc_id": row.get("doc_id")})

        row_grams = ngrams(toks_raw, 7)
        shared = row_grams & eval_grams
        n_shared = len(shared)
        if n_shared:
            leak_rows_ge1 += 1
            leak_max = max(leak_max, n_shared)
            if n_shared >= 3:
                leak_rows_ge3 += 1
            if n_shared >= 2 and len(suspicious_samples) < 12:
                suspicious_samples.append({
                    "text": text[:500],
                    "words": words,
                    "shared_7gram_count": n_shared,
                    "sample_shared_7grams": sorted(shared)[:8],
                    "row_source": row.get("source"),
                    "doc_id": row.get("doc_id"),
                })

    def pct(x: int) -> float:
        return round(x / row_count * 100.0, 4) if row_count else 0.0

    task_summary = {}
    for group in sorted(lex_sets):
        task_summary[group] = {
            "row_hit_count": int(task_hit_rows[group]),
            "row_hit_pct": pct(task_hit_rows[group]),
            "distinct_term_hits_per_1k_words": round(task_hits_distinct_total[group] / max(word_count, 1) * 1000.0, 4),
            "term_hits_per_1k_words": round(task_hits_total[group] / max(word_count, 1) * 1000.0, 4),
            "samples": match_samples[group],
        }
    relation_summary = {}
    for group in RELATION_MARKERS:
        relation_summary[group] = {
            "row_hit_count": int(relation_hit_rows[group]),
            "row_hit_pct": pct(relation_hit_rows[group]),
            "term_hits_per_1k_words": round(relation_hits_total[group] / max(word_count, 1) * 1000.0, 4),
            "samples": relation_samples[group],
        }
    return {
        "pool": name,
        "meaning": spec.get("meaning"),
        "path": str(spec["path"]),
        "source_prefix": spec.get("source_prefix"),
        "rows": row_count,
        "words": word_count,
        "word_len_hist_20bin": dict(sorted(word_len_hist.items())),
        "task_overlap": task_summary,
        "relation_marker_overlap": relation_summary,
        "cross_task_rows": {k: int(v) for k, v in sorted(cross_task_rows.items(), key=lambda kv: int(kv[0]))},
        "suspicious_eval_7gram_overlap": {
            "rows_ge1_shared_7gram": leak_rows_ge1,
            "rows_ge1_pct": pct(leak_rows_ge1),
            "rows_ge3_shared_7gram": leak_rows_ge3,
            "rows_ge3_pct": pct(leak_rows_ge3),
            "max_shared_7grams_in_one_row": leak_max,
            "samples_ge2": suspicious_samples,
            "interpretation": "Exact seven-token overlap with benchmark text is a leakage screen, not proof of contamination; common formulaic strings can match. Rows with repeated/high-count overlap should be excluded before future training materialization.",
        },
    }


def ratio(a: float | None, b: float | None) -> float | None:
    if a is None or b is None or b == 0:
        return None
    return round(a / b, 4)


def compare_pools(pools: dict[str, Any]) -> dict[str, Any]:
    comps: dict[str, Any] = {}
    pairs = [
        ("cached_fineweb_vs_official_control", "cached_fineweb_seqsafe96_block", "official_lengthmatched_control_block"),
        ("live_v3_vs_cached_fineweb", "live_v3_relation_rich_doccap8", "cached_fineweb_seqsafe96_block"),
        ("live_v5_vs_cached_fineweb", "live_v5_self_contained_doccap8", "cached_fineweb_seqsafe96_block"),
        ("live_v3_vs_live_v5", "live_v3_relation_rich_doccap8", "live_v5_self_contained_doccap8"),
    ]
    for label, a, b in pairs:
        if a not in pools or b not in pools:
            continue
        pa, pb = pools[a], pools[b]
        d: dict[str, Any] = {"a": a, "b": b, "task_term_hit_density_ratio_a_over_b": {}, "task_row_hit_pct_delta_a_minus_b": {}, "relation_density_ratio_a_over_b": {}, "relation_row_pct_delta_a_minus_b": {}}
        for group in pa["task_overlap"]:
            av = pa["task_overlap"][group]["term_hits_per_1k_words"]
            bv = pb["task_overlap"][group]["term_hits_per_1k_words"]
            d["task_term_hit_density_ratio_a_over_b"][group] = ratio(av, bv)
            d["task_row_hit_pct_delta_a_minus_b"][group] = round(pa["task_overlap"][group]["row_hit_pct"] - pb["task_overlap"][group]["row_hit_pct"], 4)
        for group in pa["relation_marker_overlap"]:
            av = pa["relation_marker_overlap"][group]["term_hits_per_1k_words"]
            bv = pb["relation_marker_overlap"][group]["term_hits_per_1k_words"]
            d["relation_density_ratio_a_over_b"][group] = ratio(av, bv)
            d["relation_row_pct_delta_a_minus_b"][group] = round(pa["relation_marker_overlap"][group]["row_hit_pct"] - pb["relation_marker_overlap"][group]["row_hit_pct"], 4)
        comps[label] = d
    return comps


def write_note(payload: dict[str, Any], out_json: pathlib.Path) -> None:
    lines = ["# research FineWeb source/benchmark motif overlap\n\n"]
    lines.append("CPU-only source analysis for interpreting the repaired seqsafe96 run and preparing a possible later four-arm FineWeb family. It does not measure model competence.\n\n")
    lines.append("## Pool summary\n\n")
    lines.append("| pool | rows | words | EWoK row hit % | Entity row hit % | COMPS row hit % | GPIQA row hit % | causal row % | physical row % | object-property row % | eval 7-gram rows >=1 |\n")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
    for name, p in payload["pools"].items():
        to = p["task_overlap"]
        ro = p["relation_marker_overlap"]
        leak = p["suspicious_eval_7gram_overlap"]
        lines.append(f"| {name} | {p['rows']} | {p['words']} | {to['EWoK']['row_hit_pct']:.2f} | {to['Entity']['row_hit_pct']:.2f} | {to['COMPS']['row_hit_pct']:.2f} | {to['GlobalPIQA']['row_hit_pct']:.2f} | {ro['causal']['row_hit_pct']:.2f} | {ro['physical_dynamic']['row_hit_pct']:.2f} | {ro['object_property']['row_hit_pct']:.2f} | {leak['rows_ge1_shared_7gram']} |\n")
    lines.append("\n## Main comparisons\n\n")
    for label, comp in payload["comparisons"].items():
        lines.append(f"### {label}\n\n")
        lines.append("Task hit-density ratio a/b: `" + json.dumps(comp["task_term_hit_density_ratio_a_over_b"], ensure_ascii=False) + "`\n\n")
        lines.append("Relation hit-density ratio a/b: `" + json.dumps(comp["relation_density_ratio_a_over_b"], ensure_ascii=False) + "`\n\n")
    lines.append("## Interpretation for the next route\n\n")
    lines.append(payload["interpretation"] + "\n\n")
    lines.append(f"Full JSON with samples: `{out_json}`\n")
    NOTE.parent.mkdir(parents=True, exist_ok=True)
    NOTE.write_text("".join(lines), encoding="utf-8")


def main() -> None:
    assets = build_benchmark_assets()
    pool_summaries: dict[str, Any] = {}
    for name, spec in POOLS.items():
        pool_summaries[name] = analyze_pool(name, spec, assets)
    comparisons = compare_pools(pool_summaries)

    interpretation_bits = []
    c = comparisons.get("cached_fineweb_vs_official_control", {})
    task_ratio = c.get("task_term_hit_density_ratio_a_over_b", {})
    rel_ratio = c.get("relation_density_ratio_a_over_b", {})
    if task_ratio:
        interpretation_bits.append(
            "Cached FineWeb and official length-matched control differ in benchmark-motif exposure: "
            f"EWoK ratio {task_ratio.get('EWoK')}, Entity {task_ratio.get('Entity')}, COMPS {task_ratio.get('COMPS')}, GlobalPIQA {task_ratio.get('GlobalPIQA')}."
        )
    if rel_ratio:
        interpretation_bits.append(
            "Relation marker ratios cached/control: "
            f"causal {rel_ratio.get('causal')}, physical_dynamic {rel_ratio.get('physical_dynamic')}, object_property {rel_ratio.get('object_property')}, spatial_state {rel_ratio.get('spatial_state')}."
        )
    interpretation_bits.append(
        "Use these numbers only to interpret source substrates.  A positive research score would still be downstream evidence; a lexical advantage here only suggests why a later stratified v3/v5 mixture might be worth materializing.  Any rows with repeated exact benchmark 7-gram overlap must be excluded before future training materialization, even when the overlap appears formulaic."
    )

    payload = {
        "status": "FINEWEB_SOURCE_BENCHMARK_MOTIF_OVERLAP",
        "benchmark_record_counts": assets["group_record_counts"],
        "benchmark_top_terms": assets["top_terms"],
        "benchmark_examples": assets["examples"],
        "pools": pool_summaries,
        "comparisons": comparisons,
        "interpretation": " ".join(interpretation_bits),
        "caveat": "Lexical/motif overlap is not BabyLM evaluation evidence and cannot replace real training/evaluation. It is a source-quality and leakage-risk instrument for future corpus materialization.",
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_json = OUT_DIR / "fineweb_source_benchmark_motif_overlap.json"
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_note(payload, out_json)
    print(json.dumps({"status": payload["status"], "out_json": str(out_json), "note": str(NOTE), "pools": {k: {"rows": v["rows"], "words": v["words"]} for k, v in pool_summaries.items()}}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
