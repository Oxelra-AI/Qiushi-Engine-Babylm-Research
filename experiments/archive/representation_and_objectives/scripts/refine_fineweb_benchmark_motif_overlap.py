#!/usr/bin/env python3
"""Refine the research FineWeb/benchmark motif analysis with field-aware text.

The first research overlap pass was useful as a fast smoke test and leakage screen, but
its benchmark vocabulary included metadata strings for EWoK and nonce/template tokens
for COMPS.  This repaired analysis builds benchmark lexical sets only from actual
natural-language evaluation prompts/options/properties, removes artificial tokens, and
keeps the exact 7-gram overlap screen on the same field-aware text.

It remains source analysis only; downstream BabyLM evidence still comes from training
and official-compatible evaluation.
"""
from __future__ import annotations

import json
import pathlib
import re
from collections import Counter, defaultdict
from typing import Any, Iterable

ROOT = pathlib.Path("experiments/archive/representation_and_objectives")
WORK = ROOT
STRICT = pathlib.Path("experiments/archive/initial_model_studies/repos/babylm-eval/strict")
EVAL = STRICT / "evaluation_data/full_eval"
OUT_DIR = WORK / "data/fineweb_source_benchmark_overlap_refined"
NOTE = (WORK.parents[2] / 'research/notes/representation_and_objectives/fineweb_source_benchmark_motif_overlap_refined.md')
RAW_PASS = WORK / "data/fineweb_source_benchmark_overlap/fineweb_source_benchmark_motif_overlap.json"

POOLS = {
    "cached_fineweb_seqsafe96_block": {
        "path": WORK / "training/data/cached_fineweb_seqsafe96_candidate/cleanqwen_seqsafe_fineweb_single_doc_10M.jsonl",
        "source_prefix": "fineweb_edu_random_quality_single_doc_seqsafe_cached_initial_model_studies",
        "text_key": "text",
        "words_key": "words",
    },
    "official_lengthmatched_control_block": {
        "path": WORK / "training/data/cached_fineweb_seqsafe96_candidate/cleanqwen_official_lengthmatched_seqsafe_control_10M.jsonl",
        "source_prefix": "official_lengthmatched_to_seqsafe_fineweb",
        "text_key": "text",
        "words_key": "words",
    },
    "live_v3_relation_rich_doccap8": {
        "path": WORK / "data/live_fineweb_source_selector_v3/live_fineweb_selector_v3_stable_doccap8.jsonl",
        "source_prefix": None,
        "text_key": "selector_v3_repaired_text",
        "fallback_text_key": "text",
        "words_key": "selector_v3_words",
    },
    "live_v5_self_contained_doccap8": {
        "path": WORK / "data/live_fineweb_selector_v5_strictstable/live_fineweb_selector_v5_doccap8.jsonl",
        "source_prefix": None,
        "text_key": "selector_v5_text",
        "fallback_text_key": "selector_v3_repaired_text",
        "words_key": "selector_v5_words",
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
    "sentence", "good", "bad", "target", "context", "direct", "indirect", "swap", "concept", "variable",
    "therefore", "undistracted", "dist", "before", "between",
}
ARTIFICIAL = {
    "wug", "wugs", "fep", "feps", "blicket", "blickets", "dax", "daxes", "toma", "tomas", "zup", "zups",
    "ali", "mohammed", "jesse", "chao", "maria", "carmen", "wei", "yan",
}
WORD_RE = re.compile(r"[A-Za-z][A-Za-z'\-]*|\d+(?:\.\d+)?")

RELATION_MARKERS = {
    "causal": "cause causes caused causing lead leads led result results resulted make makes made prevent prevents prevented allow allows enabled enables increase increases decrease decreases reduce reduces damage damages break breaks broke broken improve improves affect affects change changes force forces forced".split(),
    "physical_dynamic": "move moves moving moved push pushes pushed pull pulls pulled fall falls falling fell drop drops dropped bounce bounces bounced slide slides slid roll rolls rolled expand expands shrink shrinks melt melts melted freeze freezes frozen absorb absorbs heat heats cool cools bend bends stretch stretches throw throws thrown hit hits pour pours".split(),
    "object_property": "hard soft heavy light rough smooth sharp blunt wet dry hot cold warm cool flexible rigid transparent opaque magnetic plastic metal wooden glass paper air water liquid solid gas pressure weight color size shape open closed full empty".split(),
    "spatial_state": "inside outside above below under over near far between around beside behind front left right contain contains contained holding placed put opening center edge surface floor top bottom".split(),
    "social_agent": "person people child children teacher student mother father friend group help helps helped teach teaches learn learns ask asks tell tells give gives receive receives want wants know knows believe believes speak speaks said says".split(),
    "temporal_process": "first next later earlier before after during finally eventually started begins became becomes continued ended discovered developed published founded built created born died".split(),
}
REL_SETS = {k: set(v) for k, v in RELATION_MARKERS.items()}


def iter_jsonl(path: pathlib.Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def raw_tokens(text: str) -> list[str]:
    return [m.group(0).lower().strip("'-") for m in WORD_RE.finditer(text or "") if m.group(0).strip("'-")]


def content_tokens(text: str) -> list[str]:
    out = []
    for t in raw_tokens(text):
        if len(t) < 3 or t in STOP or t in ARTIFICIAL:
            continue
        out.append(t)
    return out


def ngrams(toks: list[str], n: int) -> set[str]:
    return {" ".join(toks[i:i+n]) for i in range(max(0, len(toks) - n + 1))}


def benchmark_texts_for(group: str, row: dict[str, Any]) -> list[str]:
    if group == "EWoK":
        keys = ["ConceptA", "ConceptB", "Context1", "Context2", "Target1", "Target2"]
    elif group == "COMPS":
        keys = ["property", "acceptable_concept", "unacceptable_concept", "prefix_acceptable", "property_phrase", "prefix_unacceptable"]
    elif group == "GlobalPIQA":
        keys = ["prompt", "solution0", "solution1", "solution2", "solution3"]
    elif group == "Entity":
        keys = ["input_prefix", "options"]
    else:
        keys = []
    vals = []
    for k in keys:
        v = row.get(k)
        if isinstance(v, str):
            vals.append(v)
        elif isinstance(v, list):
            vals.extend(str(x) for x in v if isinstance(x, (str, int, float)))
    return vals


def add_eval_file(path: pathlib.Path, group: str, lex: dict[str, Counter], grams: Counter, records: Counter, examples: dict[str, list[str]]) -> None:
    for row in iter_jsonl(path):
        vals = benchmark_texts_for(group, row)
        text = " ".join(vals)
        toks = content_tokens(text)
        if not toks:
            continue
        lex[group].update(toks)
        for gram in ngrams(raw_tokens(text), 7):
            grams[gram] += 1
        records[group] += 1
        if len(examples[group]) < 5:
            examples[group].append(text[:260])


def build_benchmark() -> dict[str, Any]:
    lex: dict[str, Counter] = defaultdict(Counter)
    grams: Counter = Counter()
    records: Counter = Counter()
    examples: dict[str, list[str]] = defaultdict(list)
    for p in sorted((EVAL / "ewok_filtered").glob("*.jsonl")):
        add_eval_file(p, "EWoK", lex, grams, records, examples)
    for p in sorted((EVAL / "comps").glob("*.jsonl")):
        add_eval_file(p, "COMPS", lex, grams, records, examples)
    for p in sorted((EVAL / "global_piqa_parallel").glob("*.jsonl")) + sorted((EVAL / "global_piqa_nonparallel").glob("*.jsonl")):
        add_eval_file(p, "GlobalPIQA", lex, grams, records, examples)
    for p in sorted((EVAL / "entity_tracking").glob("*.jsonl")):
        add_eval_file(p, "Entity", lex, grams, records, examples)

    sets: dict[str, set[str]] = {}
    for group, ctr in lex.items():
        # Keep terms frequent enough to represent a benchmark motif, but not so frequent
        # that tiny GlobalPIQA's useful vocabulary disappears.
        min_count = {"GlobalPIQA": 2, "EWoK": 4, "Entity": 8, "COMPS": 12}.get(group, 5)
        sets[group] = {w for w, c in ctr.items() if c >= min_count and w not in STOP and w not in ARTIFICIAL and len(w) >= 3}
    return {
        "term_sets": sets,
        "term_counts": lex,
        "eval_7grams": set(grams.keys()),
        "record_counts": dict(records),
        "top_terms": {g: ctr.most_common(50) for g, ctr in lex.items()},
        "examples": dict(examples),
    }


def get_pool_text(row: dict[str, Any], spec: dict[str, Any]) -> tuple[str, int]:
    text = row.get(spec.get("text_key", "text"))
    if not isinstance(text, str) or not text.strip():
        text = row.get(spec.get("fallback_text_key", "text"), "")
    words_key = spec.get("words_key")
    try:
        w = int(row.get(words_key)) if words_key else int(row.get("words"))
    except Exception:
        w = len(raw_tokens(str(text)))
    return str(text or ""), w


def iter_pool(spec: dict[str, Any]) -> Iterable[dict[str, Any]]:
    prefix = spec.get("source_prefix")
    for row in iter_jsonl(pathlib.Path(spec["path"])):
        if prefix is not None and row.get("source") != prefix:
            continue
        yield row


def analyze_pool(name: str, spec: dict[str, Any], bench: dict[str, Any]) -> dict[str, Any]:
    row_n = 0
    word_n = 0
    task_row_hits = Counter()
    task_distinct = Counter()
    task_total = Counter()
    relation_row_hits = Counter()
    relation_total = Counter()
    exact7_ge1 = 0
    exact7_ge2 = 0
    exact7_samples = []
    task_samples: dict[str, list[dict[str, Any]]] = defaultdict(list)
    relation_samples: dict[str, list[dict[str, Any]]] = defaultdict(list)
    eval7 = bench["eval_7grams"]
    term_sets: dict[str, set[str]] = bench["term_sets"]
    cross_counts = Counter()

    for row in iter_pool(spec):
        text, words = get_pool_text(row, spec)
        if not text.strip():
            continue
        row_n += 1
        word_n += words
        raw = raw_tokens(text)
        toks = content_tokens(text)
        ctr = Counter(toks)
        st = set(ctr)
        tasks = 0
        for group, terms in term_sets.items():
            m = st & terms
            distinct = len(m)
            total = sum(ctr[t] for t in m)
            task_distinct[group] += distinct
            task_total[group] += total
            # two distinct benchmark terms is a minimal row-level motif hit; for Entity,
            # a single object term repeats across synthetic options too easily.
            if distinct >= 2:
                task_row_hits[group] += 1
                tasks += 1
                if len(task_samples[group]) < 8:
                    task_samples[group].append({"text": text[:360], "words": words, "matches": sorted(m)[:25], "doc_id": row.get("doc_id"), "source": row.get("source")})
        cross_counts[str(tasks)] += 1
        for group, terms in REL_SETS.items():
            m = st & terms
            total = sum(ctr[t] for t in m)
            relation_total[group] += total
            if total:
                relation_row_hits[group] += 1
                if len(relation_samples[group]) < 5:
                    relation_samples[group].append({"text": text[:360], "words": words, "matches": sorted(m), "doc_id": row.get("doc_id"), "source": row.get("source")})
        shared = ngrams(raw, 7) & eval7
        if shared:
            exact7_ge1 += 1
            if len(shared) >= 2:
                exact7_ge2 += 1
                if len(exact7_samples) < 12:
                    exact7_samples.append({"text": text[:500], "words": words, "shared_7grams": sorted(shared)[:10], "doc_id": row.get("doc_id"), "source": row.get("source")})

    def pct(x: int) -> float:
        return round(100.0 * x / row_n, 4) if row_n else 0.0
    task_summary = {}
    for group in sorted(term_sets):
        task_summary[group] = {
            "term_set_size": len(term_sets[group]),
            "row_hit_count": int(task_row_hits[group]),
            "row_hit_pct": pct(task_row_hits[group]),
            "distinct_term_hits_per_1k_words": round(task_distinct[group] / max(word_n, 1) * 1000, 4),
            "term_hits_per_1k_words": round(task_total[group] / max(word_n, 1) * 1000, 4),
            "samples": task_samples[group],
        }
    relation_summary = {}
    for group in RELATION_MARKERS:
        relation_summary[group] = {
            "row_hit_count": int(relation_row_hits[group]),
            "row_hit_pct": pct(relation_row_hits[group]),
            "term_hits_per_1k_words": round(relation_total[group] / max(word_n, 1) * 1000, 4),
            "samples": relation_samples[group],
        }
    return {
        "rows": row_n,
        "words": word_n,
        "task_overlap_field_aware": task_summary,
        "relation_marker_overlap": relation_summary,
        "cross_task_row_counts": dict(sorted(cross_counts.items(), key=lambda kv: int(kv[0]))),
        "field_aware_exact_eval_7gram_overlap": {
            "rows_ge1": exact7_ge1,
            "rows_ge1_pct": pct(exact7_ge1),
            "rows_ge2": exact7_ge2,
            "rows_ge2_pct": pct(exact7_ge2),
            "samples_ge2": exact7_samples,
        },
    }


def div(a: float, b: float) -> float | None:
    return None if b == 0 else round(a / b, 4)


def compare(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    out = {"task_term_density_ratio": {}, "task_row_pct_delta": {}, "relation_density_ratio": {}, "relation_row_pct_delta": {}}
    for g in a["task_overlap_field_aware"]:
        av = a["task_overlap_field_aware"][g]["term_hits_per_1k_words"]
        bv = b["task_overlap_field_aware"][g]["term_hits_per_1k_words"]
        out["task_term_density_ratio"][g] = div(av, bv)
        out["task_row_pct_delta"][g] = round(a["task_overlap_field_aware"][g]["row_hit_pct"] - b["task_overlap_field_aware"][g]["row_hit_pct"], 4)
    for g in a["relation_marker_overlap"]:
        av = a["relation_marker_overlap"][g]["term_hits_per_1k_words"]
        bv = b["relation_marker_overlap"][g]["term_hits_per_1k_words"]
        out["relation_density_ratio"][g] = div(av, bv)
        out["relation_row_pct_delta"][g] = round(a["relation_marker_overlap"][g]["row_hit_pct"] - b["relation_marker_overlap"][g]["row_hit_pct"], 4)
    return out


def main() -> None:
    bench = build_benchmark()
    pools = {name: analyze_pool(name, spec, bench) for name, spec in POOLS.items()}
    comparisons = {
        "cached_fineweb_vs_official_control": compare(pools["cached_fineweb_seqsafe96_block"], pools["official_lengthmatched_control_block"]),
        "live_v3_vs_cached_fineweb": compare(pools["live_v3_relation_rich_doccap8"], pools["cached_fineweb_seqsafe96_block"]),
        "live_v5_vs_cached_fineweb": compare(pools["live_v5_self_contained_doccap8"], pools["cached_fineweb_seqsafe96_block"]),
        "live_v3_vs_live_v5": compare(pools["live_v3_relation_rich_doccap8"], pools["live_v5_self_contained_doccap8"]),
    }
    cf = comparisons["cached_fineweb_vs_official_control"]
    interpretation = (
        "After removing metadata and nonce/template tokens, the cached seqsafe96 FineWeb block still does not look like a clean lexical match to every deficit column: "
        f"term-density ratios cached/control are {json.dumps(cf['task_term_density_ratio'], ensure_ascii=False)}. "
        f"It is much richer in causal/object/temporal markers but weaker in physical/spatial/social markers: {json.dumps(cf['relation_density_ratio'], ensure_ascii=False)}. "
        "This makes the repaired research downstream result especially informative: broad gains would mean source breadth carries more than these simple surface motifs; flat or harmful movement would not falsify cleaner stratified FineWeb, because the cached block is noisy and misbalanced. "
        "The live v3/v5 pools have lower benchmark-term density than the cached block but higher causal/temporal density; a future live source+view family should deliberately mix v3 relation-rich context with v5 self-contained facts and not optimize solely for isolated factual sentences."
    )
    payload = {
        "status": "FINEWEB_SOURCE_BENCHMARK_MOTIF_OVERLAP_REFINED",
        "raw_pass_corrected": str(RAW_PASS),
        "benchmark_record_counts": bench["record_counts"],
        "benchmark_top_terms_field_aware": bench["top_terms"],
        "benchmark_examples_field_aware": bench["examples"],
        "pools": pools,
        "comparisons": comparisons,
        "interpretation": interpretation,
        "scientific_status": "CPU-only source and leakage-risk analysis; not training/evaluation evidence.",
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_json = OUT_DIR / "fineweb_source_benchmark_motif_overlap_refined.json"
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    lines = ["# research refined FineWeb source/benchmark motif overlap\n\n"]
    lines.append("This repairs the earlier motif pass by using field-aware benchmark text: EWoK uses concepts/contexts/targets rather than metadata; COMPS removes nonce/template tokens; Entity and GlobalPIQA use prompts/options. It is still source analysis only.\n\n")
    lines.append("| pool | rows | words | EWoK row % | Entity row % | COMPS row % | GPIQA row % | causal row % | physical row % | spatial row % | social row % | exact eval 7gram rows |\n")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
    for name, p in pools.items():
        t = p["task_overlap_field_aware"]
        r = p["relation_marker_overlap"]
        e = p["field_aware_exact_eval_7gram_overlap"]
        lines.append(f"| {name} | {p['rows']} | {p['words']} | {t['EWoK']['row_hit_pct']:.2f} | {t['Entity']['row_hit_pct']:.2f} | {t['COMPS']['row_hit_pct']:.2f} | {t['GlobalPIQA']['row_hit_pct']:.2f} | {r['causal']['row_hit_pct']:.2f} | {r['physical_dynamic']['row_hit_pct']:.2f} | {r['spatial_state']['row_hit_pct']:.2f} | {r['social_agent']['row_hit_pct']:.2f} | {e['rows_ge1']} |\n")
    lines.append("\n## Comparisons\n\n")
    for label, comp in comparisons.items():
        lines.append(f"### {label}\n\n")
        lines.append("Task term-density ratio: `" + json.dumps(comp["task_term_density_ratio"], ensure_ascii=False) + "`\n\n")
        lines.append("Relation term-density ratio: `" + json.dumps(comp["relation_density_ratio"], ensure_ascii=False) + "`\n\n")
    lines.append("## Route implication\n\n")
    lines.append(interpretation + "\n\n")
    lines.append(f"JSON with samples: `{out_json}`\n")
    NOTE.write_text("".join(lines), encoding="utf-8")
    print(json.dumps({"status": payload["status"], "out_json": str(out_json), "note": str(NOTE), "cached_vs_control_task_ratios": cf["task_term_density_ratio"], "cached_vs_control_relation_ratios": cf["relation_density_ratio"]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
