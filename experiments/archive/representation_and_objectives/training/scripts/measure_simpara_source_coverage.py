#!/usr/bin/env python3
"""Measure what the pending Qwen simplification/paraphrase shard can and cannot add.

The research slice showed that simplification/paraphrase generations are mostly
source-conservative.  The research question is different: do those 30k prompts add
new factual/entity breadth, or are they multiple linguistic views of the same
SimpleWiki source rows?  This script measures the prompt source overlap, article
coverage, entity/number density, and an evidence-based estimate of the maximum
semantic-view word mass before the full generations are available.
"""
from __future__ import annotations

import collections
import hashlib
import json
import math
import pathlib
import re
import statistics
from typing import Iterable

ROOT = pathlib.Path("experiments/archive/representation_and_objectives")
PROMPTS = ROOT / "training/data/factual_prompts_shard1_simpara.jsonl"
SLICE_SUMMARY = ROOT / "training/data/generation_slice/slice_quality_summary.json"
OUT_DIR = ROOT / "training/data/semantic_view"
OUT_JSON = OUT_DIR / "simpara_prompt_source_coverage.json"
OUT_NOTE = (ROOT.parents[2] / 'research/notes/representation_and_objectives/simpara_source_coverage.md')

WORD_RE = re.compile(r"[A-Za-z][A-Za-z'’-]*")
NUM_RE = re.compile(r"\b(?:\d{1,4}(?:[,.]\d+)*(?:[-–]\d{1,4})?|\d+(?:\.\d+)?\s*(?:%|km|m|ft|million|billion|thousand)|[A-Z]{2,}\d*)\b")
CAP_SEQ_RE = re.compile(r"\b(?:[A-Z][\w'’.-]*(?:\s+|$)){1,6}")
STOP = {
    "the", "a", "an", "and", "or", "of", "in", "to", "for", "on", "with", "as", "by", "from", "at",
    "is", "are", "was", "were", "be", "been", "being", "it", "its", "this", "that", "these", "those",
    "he", "she", "they", "them", "his", "her", "their", "which", "who", "whom", "when", "where", "while",
    "also", "one", "two", "new", "old", "first", "second", "may", "can", "could", "would", "should",
    "about", "into", "after", "before", "during", "over", "under", "between", "through", "than", "then",
}
DOMAIN_TERMS = {
    "science_physical": {"volcanic","lava","glacier","molecular","cloud","dark","matter","galaxy","energy","electric","chemical","silicon","virus","respiratory","species","freshwater","cyclone","weather","planet","star","comet","biology","engineering"},
    "geography_places": {"city","province","county","district","municipality","canton","region","capital","river","mountain","island","country","state","switzerland","italy","iceland","australia","canada","hong","kong"},
    "people_history": {"born","died","served","mayor","governor","senate","war","emperor","lawyer","professor","leader","olympics","champion","president","secretary","minister","actress","writer"},
    "institutions_society": {"company","university","club","government","council","education","law","act","organization","competition","factory","brand","team","league","prize","school","bank"},
    "media_culture": {"film","album","band","song","movie","game","book","published","released","television","magazine","novel","music","dragon","streetlight"},
    "quant_numeric": {"january","february","march","april","may","june","july","august","september","october","november","december","km","metres","feet","century","year","million","percent"},
    "causal_relational": {"because","therefore","caused","cause","effect","result","led","leading","formed","became","created","due","reason","replaced","allows","requires","helps","influenced"},
}


def sha256_text(text: str) -> str:
    return hashlib.sha256(" ".join(text.split()).encode("utf-8")).hexdigest()


def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def words(text: str) -> list[str]:
    return text.split()


def word_stats(vals: Iterable[int | float]) -> dict:
    xs = list(vals)
    if not xs:
        return {"n": 0}
    xs_sorted = sorted(xs)
    def q(p: float) -> float:
        i = min(len(xs_sorted)-1, max(0, round((len(xs_sorted)-1)*p)))
        return xs_sorted[i]
    return {
        "n": len(xs),
        "min": min(xs),
        "p05": q(0.05),
        "mean": round(statistics.mean(xs), 4),
        "median": statistics.median(xs),
        "p95": q(0.95),
        "max": max(xs),
        "sum": sum(xs),
    }


def content_tokens(text: str) -> set[str]:
    return {m.group(0).lower().strip("'’-") for m in WORD_RE.finditer(text) if len(m.group(0)) > 3 and m.group(0).lower() not in STOP}


def entities(text: str) -> set[str]:
    ents: set[str] = set()
    for m in CAP_SEQ_RE.finditer(text):
        ent = " ".join(m.group(0).split()).strip(" ,.;:()[]{}\"'")
        if not ent:
            continue
        parts = ent.split()
        if len(parts) == 1 and (parts[0].lower() in STOP or parts[0] in {"The", "This", "These", "A", "An", "In", "On", "After", "Before", "During", "He", "She", "It", "They"}):
            continue
        ents.add(ent)
    return ents


def numbers(text: str) -> set[str]:
    return {m.group(0) for m in NUM_RE.finditer(text)}


def domain_hits(text: str) -> list[str]:
    toks = content_tokens(text)
    return [k for k, v in DOMAIN_TERMS.items() if toks & v]


def load_jsonl(path: pathlib.Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    prompts = load_jsonl(PROMPTS)
    by_type: dict[str, list[dict]] = collections.defaultdict(list)
    by_key: dict[str, list[dict]] = collections.defaultdict(list)
    for i, r in enumerate(prompts):
        r = dict(r)
        r["row_index"] = i
        r["source_key"] = sha256_text(str(r["source_text"]))
        r["source_words_actual"] = len(str(r["source_text"]).split())
        by_type[str(r["type"])].append(r)
        by_key[r["source_key"]].append(r)

    type_keys = {typ: {r["source_key"] for r in rows} for typ, rows in by_type.items()}
    common_simp_para = sorted(type_keys.get("simplification", set()) & type_keys.get("paraphrase", set()))
    union_keys = sorted(set().union(*type_keys.values())) if type_keys else []

    unique_source_rows = []
    for key in union_keys:
        first = by_key[key][0]
        unique_source_rows.append({
            "source_key": key,
            "source_text": first["source_text"],
            "source_article": first.get("source_article", ""),
            "source_words": int(first.get("source_words", first["source_words_actual"])),
            "prompt_types": sorted({r["type"] for r in by_key[key]}),
            "prompt_rows": [r["row_index"] for r in by_key[key]],
        })

    article_counter = collections.Counter(str(r.get("source_article", "")) for r in unique_source_rows)
    article_word_counter = collections.Counter()
    domain_counter = collections.Counter()
    source_entity_counts = []
    source_number_counts = []
    for r in unique_source_rows:
        article_word_counter[str(r["source_article"])] += int(r["source_words"])
        for d in domain_hits(str(r["source_text"])):
            domain_counter[d] += 1
        source_entity_counts.append(len(entities(str(r["source_text"]))))
        source_number_counts.append(len(numbers(str(r["source_text"]))))

    slice_ratios = {}
    if SLICE_SUMMARY.exists():
        ss = json.loads(SLICE_SUMMARY.read_text(encoding="utf-8"))
        for typ in ["simplification", "paraphrase"]:
            if typ in ss.get("by_type", {}):
                slice_ratios[typ] = ss["by_type"][typ].get("length_ratio_mean")
    simp_ratio = float(slice_ratios.get("simplification", 0.882))
    para_ratio = float(slice_ratios.get("paraphrase", 0.953))

    # Because research generated simplification and paraphrase for the same 15k good paragraphs,
    # the unique factual-source word mass is not the sum of both prompt types.  Estimate the
    # maximum semantic-view pool if each accepted source obtains one original view plus both
    # generated views at the observed slice ratios.
    unique_words = sum(int(r["source_words"]) for r in unique_source_rows)
    common_words = sum(int(by_key[k][0].get("source_words", by_key[k][0]["source_words_actual"])) for k in common_simp_para)
    simp_only_words = sum(int(by_key[k][0].get("source_words", by_key[k][0]["source_words_actual"])) for k in type_keys.get("simplification", set()) - type_keys.get("paraphrase", set()))
    para_only_words = sum(int(by_key[k][0].get("source_words", by_key[k][0]["source_words_actual"])) for k in type_keys.get("paraphrase", set()) - type_keys.get("simplification", set()))
    estimated_view_words = unique_words + common_words * (simp_ratio + para_ratio) + simp_only_words * simp_ratio + para_only_words * para_ratio

    report = {
        "status": "SIMPARA_PROMPT_SOURCE_COVERAGE_MEASURED",
        "purpose": "separate semantic-view learning potential from factual/entity breadth before spending H100 time on a new corpus arm",
        "prompt_path": str(PROMPTS),
        "prompt_sha256": sha256_file(PROMPTS),
        "prompt_rows": len(prompts),
        "prompt_type_counts": {k: len(v) for k, v in sorted(by_type.items())},
        "unique_source_texts": len(unique_source_rows),
        "unique_source_words": unique_words,
        "source_texts_with_both_simplification_and_paraphrase": len(common_simp_para),
        "source_texts_simplification_only": len(type_keys.get("simplification", set()) - type_keys.get("paraphrase", set())),
        "source_texts_paraphrase_only": len(type_keys.get("paraphrase", set()) - type_keys.get("simplification", set())),
        "source_word_stats_unique": word_stats([int(r["source_words"]) for r in unique_source_rows]),
        "source_entity_count_stats_unique": word_stats(source_entity_counts),
        "source_number_count_stats_unique": word_stats(source_number_counts),
        "unique_source_articles": len(article_counter),
        "top_articles_by_prompt_source_count": article_counter.most_common(30),
        "top_articles_by_source_words": article_word_counter.most_common(30),
        "domain_row_hits_unique_sources": dict(domain_counter),
        "slice_length_ratios_used_for_estimate": {"simplification": simp_ratio, "paraphrase": para_ratio},
        "estimated_max_semantic_view_words_if_all_outputs_accepted": round(estimated_view_words),
        "estimated_max_semantic_view_fraction_of_10M": round(estimated_view_words / 10_000_000, 6),
        "interpretation": {
            "source_overlap": "The simplification and paraphrase prompts mostly target the same source rows, so their main possible effect is multiple linguistic views of existing SimpleWiki facts rather than new factual breadth.",
            "matched_contrast_needed": "Training movement from this corpus must be compared to an original-only corpus with the same source articles, row lengths, and word exposure so transformation benefit is not confounded with repeated allocation of the same source content.",
        },
        "samples": unique_source_rows[:12],
    }
    OUT_JSON.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    md = []
    md.append("# research — simpara source coverage and semantic-view interpretation\n\n")
    md.append("## Measurement\n")
    md.append(f"- Prompt file: `{PROMPTS}`; rows={len(prompts)}; sha256 `{report['prompt_sha256']}`.\n")
    md.append(f"- Type counts: {json.dumps(report['prompt_type_counts'], ensure_ascii=False)}.\n")
    md.append(f"- Unique source texts: {report['unique_source_texts']} with {unique_words:,} source words across {report['unique_source_articles']} SimpleWiki article labels.\n")
    md.append(f"- Source texts with both simplification and paraphrase prompts: {len(common_simp_para)}.\n")
    md.append(f"- Using research slice length ratios, the maximum semantic-view pool if every generated row were accepted is about {round(estimated_view_words):,} words ({estimated_view_words/10_000_000:.2%} of a 10M corpus).\n")
    md.append("- Domain row hits over unique source texts: " + json.dumps(dict(domain_counter), ensure_ascii=False) + "\n")
    md.append("\n## Scientific meaning\n")
    md.append("The pending simplification/paraphrase shard is not a broad new factual source: it is two generated linguistic views of the same 15k SimpleWiki source rows. That can still be scientifically useful, but as a semantic-view learning mechanism rather than as evidence that factual/entity/causal breadth alone closes the leader gap.\n\n")
    md.append("The next corpus comparison therefore needs two matched arms: a treatment that uses original+accepted simplification/paraphrase views, and an original-only contrast that allocates the same word exposure to the same SimpleWiki source articles with the same row-length sequence. This makes any score movement interpretable as transformation/view learning rather than mere extra passes over the selected articles.\n\n")
    md.append("## Top source articles by selected source rows\n")
    for art, n in article_counter.most_common(20):
        md.append(f"- {art}: {n}\n")
    md.append(f"\nFull JSON: `{OUT_JSON}`\n")
    OUT_NOTE.write_text("".join(md), encoding="utf-8")
    print(json.dumps({
        "status": report["status"],
        "prompt_rows": len(prompts),
        "unique_source_texts": len(unique_source_rows),
        "unique_source_words": unique_words,
        "both_simp_para": len(common_simp_para),
        "estimated_max_view_words": round(estimated_view_words),
        "json": str(OUT_JSON),
        "note": str(OUT_NOTE),
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
