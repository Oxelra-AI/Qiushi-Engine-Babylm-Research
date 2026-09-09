#!/usr/bin/env python3
"""Audit available FineWeb source pools for a low-confound source-breadth/rewrite experiment.

Inputs are CPU-side JSONL source files already prepared in representation_and_objectives/frontier_consolidation.
The audit deduplicates normalized sentences, measures source mass and factual anchors,
and estimates feasible paired-source+rewrite corpus mass without inflating repetition.
"""
from __future__ import annotations

import json
import pathlib
import re
from collections import Counter, defaultdict
from typing import Any

ROOT = pathlib.Path("experiments/archive/representation_and_objectives")
OUT_ROOT = ROOT / "data" / "fineweb_source_pool_audit"
NOTE = (ROOT / 'notes'.parents[3] / 'research/notes/representation_and_objectives/fineweb_source_pool_audit.md')

POOLS = {
    "A01_step011_544_stratified": pathlib.Path("experiments/archive/representation_and_objectives/training/data/fineweb_rewrite_faithfulness_slice/fineweb_rewrite_faithfulness_sources.jsonl"),
    "A01_step010_balanced_source_by_rewrite": pathlib.Path("experiments/archive/representation_and_objectives/training/data/fineweb_sentence_screens/balanced_source_by_rewrite_sources.jsonl"),
    "A01_step010_strict_factual_expository": pathlib.Path("experiments/archive/representation_and_objectives/training/data/fineweb_sentence_screens/strict_factual_expository_sources.jsonl"),
    "A02_step011_high_precision": pathlib.Path("experiments/archive/frontier_consolidation/data/fineweb_source_by_rewrite_repair/fineweb_factual_high_precision_sources.jsonl"),
    "A02_step011_medium_repaired": pathlib.Path("experiments/archive/frontier_consolidation/data/fineweb_source_by_rewrite_repair/fineweb_factual_medium_repaired_sources.jsonl"),
    "A02_step011_high_anchor": pathlib.Path("experiments/archive/frontier_consolidation/data/fineweb_high_anchor_slice/fineweb_high_anchor_sources.jsonl"),
}

WORD_RE = re.compile(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)?")
NUM_RE = re.compile(r"\b\d+(?:[,.]\d+)*(?:st|nd|rd|th|%|s)?\b")
ENTITY_RE = re.compile(r"\b(?:[A-Z][a-z]+|[A-Z]{2,})(?:\s+(?:of|the|and|de|du|van|von|[A-Z][a-z]+|[A-Z]{2,}))*\b")
RELATION_TERMS = [
    "because", "cause", "caused", "causes", "result", "results", "therefore", "due to", "led to", "leads to",
    "became", "become", "formed", "created", "invented", "founded", "located", "contains", "include", "includes",
    "used", "works", "produces", "prevents", "protects", "allows", "requires", "measured", "compared", "between",
    "during", "after", "before", "when", "while", "if", "than", "from", "into", "within",
]
DOMAIN_PATTERNS = {
    "science_physical": re.compile(r"\b(water|energy|temperature|chemical|species|plant|animal|virus|disease|climate|carbon|oxygen|earth|soil|river|cell|protein|planet|star|magnetic|electric|radiation|material|pressure|gas|fluid|engine|machine)\b", re.I),
    "causal_relational": re.compile(r"\b(because|cause[sd]?|therefore|due to|result(?:ed|s)?|lead[s]? to|prevent[s]?|allow[s]?|require[s]?|in order to|by which|so that|as a result)\b", re.I),
    "quant_numeric": re.compile(r"\b\d+(?:[,.]\d+)*(?:st|nd|rd|th|%|s)?\b"),
    "geography_places": re.compile(r"\b(city|town|state|country|province|region|river|mountain|lake|island|north|south|east|west|county|district|capital|border)\b", re.I),
    "institutions_society": re.compile(r"\b(government|court|law|school|university|company|organization|church|army|military|police|hospital|committee|council|election|parliament|treaty)\b", re.I),
    "people_history": re.compile(r"\b(born|died|king|queen|president|minister|war|battle|century|founded|appointed|elected|author|scientist|artist|emperor)\b", re.I),
    "media_culture": re.compile(r"\b(book|film|movie|music|song|album|television|radio|newspaper|magazine|novel|poem|painting|museum|festival)\b", re.I),
    "physical_affordance": re.compile(r"\b(cut|pull|push|hold|carry|open|close|break|fall|move|heat|cool|melt|freeze|burn|float|sink|wash|dry|eat|drink|cook|clean|drive|fly|walk|run)\b", re.I),
}
BAD_HINTS = re.compile(r"\b(volume|issue|doi|view all|click here|subscribe|comments|advertisement|privacy policy|terms of use|posted by|copyright|all rights reserved)\b", re.I)


def words(text: str) -> list[str]:
    return WORD_RE.findall(text or "")


def normalize(text: str) -> str:
    toks = [w.lower() for w in words(text)]
    return " ".join(toks)


def get_text(row: dict[str, Any]) -> str:
    for key in ["text", "source_text", "source", "sentence"]:
        v = row.get(key)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return ""


def extract(row: dict[str, Any], pool: str) -> dict[str, Any]:
    text = get_text(row)
    w = row.get("words")
    if not isinstance(w, int):
        w = len(words(text))
    ents = row.get("entities")
    if not isinstance(ents, list):
        ents = sorted(set(e.strip() for e in ENTITY_RE.findall(text) if len(e.strip()) > 1))
    nums = row.get("numbers")
    if not isinstance(nums, list):
        nums = NUM_RE.findall(text)
    domain_hits = row.get("domain_hits")
    if not isinstance(domain_hits, list):
        domain_hits = [k for k, rx in DOMAIN_PATTERNS.items() if rx.search(text)]
    rel = row.get("relation_count")
    if not isinstance(rel, int):
        lo = text.lower()
        rel = sum(1 for t in RELATION_TERMS if t in lo)
    flags = row.get("flags")
    if not isinstance(flags, list):
        flags = []
    quality_flags = row.get("source_row_quality_flags")
    if not isinstance(quality_flags, list):
        quality_flags = []
    bad_hint = bool(BAD_HINTS.search(text))
    norm = normalize(text)
    return {
        "pool": pool,
        "sentence_id": row.get("sentence_id"),
        "doc_id": str(row.get("doc_id", "")),
        "text": text,
        "norm": norm,
        "words": int(w),
        "entities": [str(x) for x in ents],
        "numbers": [str(x) for x in nums],
        "domain_hits": [str(x) for x in domain_hits],
        "relation_count": int(rel),
        "flags": [str(x) for x in flags],
        "source_row_quality_flags": [str(x) for x in quality_flags],
        "bad_hint": bad_hint,
    }


def load_pool(name: str, path: pathlib.Path) -> list[dict[str, Any]]:
    rows = []
    with path.open() as f:
        for line in f:
            if line.strip():
                try:
                    rows.append(extract(json.loads(line), name))
                except Exception as e:
                    rows.append({"pool": name, "load_error": repr(e), "text": line[:200], "norm": "", "words": 0, "entities": [], "numbers": [], "domain_hits": [], "relation_count": 0, "flags": ["load_error"], "source_row_quality_flags": [], "bad_hint": True})
    return rows


def quantiles(xs: list[int | float]) -> dict[str, float | int | None]:
    if not xs:
        return {"n": 0, "min": None, "mean": None, "median": None, "p95": None, "max": None, "sum": 0}
    s = sorted(float(x) for x in xs)
    def q(p: float) -> float:
        if len(s) == 1:
            return s[0]
        pos = p * (len(s) - 1)
        lo = int(pos); hi = min(lo + 1, len(s) - 1); frac = pos - lo
        return s[lo] * (1 - frac) + s[hi] * frac
    return {"n": len(s), "min": s[0], "mean": sum(s)/len(s), "median": q(0.5), "p95": q(0.95), "max": s[-1], "sum": sum(s)}


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "rows": len(rows),
        "unique_norms": len({r["norm"] for r in rows if r.get("norm")}),
        "unique_docs": len({r["doc_id"] for r in rows if r.get("doc_id")}),
        "words": sum(int(r.get("words", 0)) for r in rows),
        "word_stats": quantiles([r.get("words", 0) for r in rows]),
        "entity_count_stats": quantiles([len(r.get("entities", [])) for r in rows]),
        "number_count_stats": quantiles([len(r.get("numbers", [])) for r in rows]),
        "relation_count_stats": quantiles([r.get("relation_count", 0) for r in rows]),
        "domain_hit_counts": dict(Counter(h for r in rows for h in r.get("domain_hits", []))),
        "bad_hint_rows": sum(1 for r in rows if r.get("bad_hint")),
        "flag_counts": dict(Counter(x for r in rows for x in (r.get("flags", []) + r.get("source_row_quality_flags", []))))
    }


def choose_priority(all_unique: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """Priority source set: high_anchor first, then high_precision remainder, then medium remainder.

    This mirrors independent review: do not inflate scarce high-precision source mass. Keep source
    identity explicit and allow later materializers to cut at the desired word budget.
    """
    # priority by pool and by anchor strength
    pool_rank = {
        "A02_step011_high_anchor": 0,
        "A02_step011_high_precision": 1,
        "A01_step010_strict_factual_expository": 2,
        "A02_step011_medium_repaired": 3,
        "A01_step010_balanced_source_by_rewrite": 4,
        "A01_step011_544_stratified": 5,
    }
    def score(r: dict[str, Any]) -> tuple[int, int, int, int, int, str]:
        return (
            pool_rank.get(r["pool"], 99),
            -len(r.get("domain_hits", [])),
            -int(r.get("relation_count", 0)),
            -len(r.get("entities", [])),
            -len(r.get("numbers", [])),
            r.get("norm", ""),
        )
    rows = list(all_unique.values())
    rows.sort(key=score)
    return rows


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    loaded = {}
    all_rows = []
    missing = []
    for name, path in POOLS.items():
        if not path.exists():
            missing.append({"pool": name, "path": str(path)})
            continue
        rows = load_pool(name, path)
        loaded[name] = rows
        all_rows.extend(rows)

    per_pool = {name: summarize(rows) for name, rows in loaded.items()}
    by_norm: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in all_rows:
        if r.get("norm"):
            by_norm[r["norm"]].append(r)
    overlap_counts = Counter(len(v) for v in by_norm.values())
    cross_pool_norms = [norm for norm, vs in by_norm.items() if len({v["pool"] for v in vs}) > 1]
    cross_pool_pairs = Counter()
    for norm in cross_pool_norms:
        pools = sorted({v["pool"] for v in by_norm[norm]})
        for i in range(len(pools)):
            for j in range(i+1, len(pools)):
                cross_pool_pairs[(pools[i], pools[j])] += 1

    # choose one canonical record per norm, preferring high-anchor/high-precision.
    canonical: dict[str, dict[str, Any]] = {}
    pool_priority = [
        "A02_step011_high_anchor",
        "A02_step011_high_precision",
        "A01_step010_strict_factual_expository",
        "A02_step011_medium_repaired",
        "A01_step010_balanced_source_by_rewrite",
        "A01_step011_544_stratified",
    ]
    rank = {p: i for i, p in enumerate(pool_priority)}
    for norm, vs in by_norm.items():
        canonical[norm] = sorted(vs, key=lambda r: (rank.get(r["pool"], 99), -len(r.get("domain_hits", [])), -int(r.get("relation_count", 0))))[0]

    priority_rows = choose_priority(canonical)
    # cumulative unique budgets for source-only and source+rewrite at rewrite ratios.
    budgets = []
    cum_words = 0
    for idx, r in enumerate(priority_rows, 1):
        cum_words += int(r.get("words", 0))
        if idx in {512, 1024, 2048, 4096, 8192, 12000, 16000, 20000, len(priority_rows)}:
            budgets.append({
                "rows": idx,
                "source_words": cum_words,
                "pair_words_ratio_0p75": int(round(cum_words * 1.75)),
                "pair_words_ratio_0p90": int(round(cum_words * 1.90)),
                "pair_words_ratio_1p05": int(round(cum_words * 2.05)),
                "unique_docs": len({x.get("doc_id") for x in priority_rows[:idx]}),
                "domain_hit_counts": dict(Counter(h for x in priority_rows[:idx] for h in x.get("domain_hits", []))),
            })

    # write priority source pool for possible later generation planning; not a launch target.
    priority_path = OUT_ROOT / "fineweb_priority_unique_sources.jsonl"
    with priority_path.open("w") as f:
        for i, r in enumerate(priority_rows):
            out = {k: v for k, v in r.items() if k != "norm"}
            out["priority_index"] = i
            f.write(json.dumps(out, ensure_ascii=False) + "\n")

    sample_path = OUT_ROOT / "fineweb_priority_samples.json"
    samples = {
        "first_20": [{k: v for k, v in r.items() if k in {"pool", "doc_id", "sentence_id", "words", "entities", "numbers", "domain_hits", "relation_count", "bad_hint", "text"}} for r in priority_rows[:20]],
        "by_pool_first_5": {name: [{k: v for k, v in r.items() if k in {"doc_id", "sentence_id", "words", "entities", "numbers", "domain_hits", "relation_count", "bad_hint", "text"}} for r in rows[:5]] for name, rows in loaded.items()},
    }
    sample_path.write_text(json.dumps(samples, indent=2, ensure_ascii=False))

    result = {
        "status": "FINEWEB_SOURCE_POOL_AUDITED",
        "purpose": "Quantify available unique FineWeb source mass before any Qwen generation or H100 training allocation.",
        "inputs": {name: str(path) for name, path in POOLS.items()},
        "missing_inputs": missing,
        "per_pool": per_pool,
        "all_loaded_rows": len(all_rows),
        "unique_norms_total": len(by_norm),
        "overlap_multiplicity_counts": dict(overlap_counts),
        "cross_pool_unique_norms": len(cross_pool_norms),
        "cross_pool_pair_overlap_counts": {" || ".join(k): v for k, v in sorted(cross_pool_pairs.items())},
        "canonical_priority_summary": summarize(priority_rows),
        "cumulative_priority_budgets": budgets,
        "priority_unique_sources_jsonl": str(priority_path),
        "samples_json": str(sample_path),
        "interpretation": "High-anchor/high-precision sources support a clean small rewrite test; a multi-million-word high-quality rewrite block is not yet substantiated without using lower-quality balanced/medium material or heavy repetition.",
    }
    (OUT_ROOT / "fineweb_source_pool_audit.json").write_text(json.dumps(result, indent=2, ensure_ascii=False))

    lines = []
    lines.append("# research FineWeb source-pool audit\n\n")
    lines.append("This CPU-only audit quantifies available FineWeb source sentences across representation_and_objectives and frontier_consolidation before any Qwen generation or H100 training.\n\n")
    lines.append("## Per-pool scale\n\n")
    lines.append("| pool | rows | unique norms | words | unique docs | entities/row mean | numbers/row mean | relation/row mean | bad-hint rows |\n")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|\n")
    for name, s in per_pool.items():
        lines.append(f"| {name} | {s['rows']} | {s['unique_norms']} | {s['words']} | {s['unique_docs']} | {s['entity_count_stats']['mean']:.3f} | {s['number_count_stats']['mean']:.3f} | {s['relation_count_stats']['mean']:.3f} | {s['bad_hint_rows']} |\n")
    lines.append("\n## Deduplicated priority pool\n\n")
    ps = result["canonical_priority_summary"]
    lines.append(f"All loaded rows: {len(all_rows)}; unique normalized sentences: {len(by_norm)}; cross-pool duplicated normalized sentences: {len(cross_pool_norms)}.\n")
    lines.append(f"Priority unique pool: {ps['rows']} rows, {ps['words']} source words, {ps['unique_docs']} docs.\n")
    lines.append("\nCumulative clean budgets (priority order high-anchor -> high-precision -> strict factual -> medium -> balanced):\n\n")
    lines.append("| rows | source words | paired words @0.9 rewrite/source | unique docs |\n")
    lines.append("|---:|---:|---:|---:|\n")
    for b in budgets:
        lines.append(f"| {b['rows']} | {b['source_words']} | {b['pair_words_ratio_0p90']} | {b['unique_docs']} |\n")
    lines.append("\n## Interpretation\n\n")
    lines.append("The audited high-anchor/high-precision material is sufficient for a small source-by-rewrite generation and training probe but does not by itself justify a 1.5-2M-word high-quality rewrite block. Inflating to that size would require lower-quality balanced/medium material or heavy repetition, changing the mechanism. The clean next experiment should therefore begin with generation/faithfulness measurement on the high-anchor/high-precision pool, then materialize a three-arm contrast at the actually accepted unique-word scale: protected slot control, FineWeb source repetition, and FineWeb source+accepted rewrite.\n\n")
    lines.append(f"JSON: `{OUT_ROOT / 'fineweb_source_pool_audit.json'}`\n")
    lines.append(f"Priority sources: `{priority_path}`\n")
    lines.append(f"Samples: `{sample_path}`\n")
    NOTE.write_text("".join(lines))

    print(json.dumps({"status": result["status"], "out": str(OUT_ROOT / "fineweb_source_pool_audit.json"), "note": str(NOTE), "priority_rows": ps["rows"], "priority_words": ps["words"], "unique_docs": ps["unique_docs"], "cross_pool_unique_norms": len(cross_pool_norms)}, indent=2))

if __name__ == "__main__":
    main()
