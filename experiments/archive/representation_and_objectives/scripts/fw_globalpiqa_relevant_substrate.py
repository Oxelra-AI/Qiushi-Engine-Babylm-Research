#!/usr/bin/env python3
"""research: compare FW compact-vs-breadth substrate in broad GlobalPIQA-relevant terms.

This CPU-only analysis does not use official evaluation examples to build training data.
It asks whether the existing, already-materialized FW comparison arms differ in
broad physical/temporal/spatial/affordance/causal substrate within the changed companion
budget.  The purpose is later interpretation: if compact improves GlobalPIQA despite having
less independent physical/consequence content, same-proposition consolidation is plausible;
if breadth improves, extra coherent coverage may be the active mechanism.
"""
from __future__ import annotations

import csv
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Any, List, Tuple

ROOT = Path("experiments/archive/representation_and_objectives")
OUT = ROOT / "data" / "fw_globalpiqa_substrate"
NOTE = (ROOT / 'notes'.parents[3] / 'research/notes/representation_and_objectives/fw_globalpiqa_relevant_substrate.md')
PAIRS = ROOT / "data/fw_full_preservation/full26k_usable_pairs_for_materializer.jsonl"
BREADTH = ROOT / "data/fw_source_breadth_wholesentence_arm/source_breadth_wholesentence_companion_sources.jsonl"
MANIFEST = ROOT / "data/fw_source_breadth_interleaved_wholesentence_arm/fw_source_breadth_interleaved_wholesentence_manifest.json"

# Broad cognitive-domain lexicons, not item-specific strings. They are used for reading
# existing corpora only; they should not be used as a direct data selector for GlobalPIQA.
CATS = {
    "physical_motion_force": [
        r"\bmove[sd]?\b", r"\bmoving\b", r"\bpush(?:ed|es|ing)?\b", r"\bpull(?:ed|s|ing)?\b",
        r"\bdrop(?:ped|s|ping)?\b", r"\bfall(?:s|ing|en)?\b", r"\bbounce(?:s|d|ing)?\b",
        r"\bbreak(?:s|ing)?\b", r"\bbroken\b", r"\bspill(?:s|ed|ing)?\b", r"\broll(?:s|ed|ing)?\b",
        r"\bfloat(?:s|ed|ing)?\b", r"\bsink(?:s|ing)?\b", r"\bskid(?:s|ded|ding)?\b",
        r"\bslide(?:s|d|ing)?\b", r"\btwist(?:s|ed|ing)?\b", r"\bturn(?:s|ed|ing)?\b",
        r"\bgravity\b", r"\bpressure\b", r"\bforce\b", r"\bfriction\b", r"\bweight\b",
    ],
    "object_material_state": [
        r"\bwater\b", r"\bair\b", r"\bglass\b", r"\bmetal\b", r"\bplastic\b", r"\bwood(?:en)?\b",
        r"\bfabric\b", r"\bcloth\b", r"\bpaper\b", r"\bceramic\b", r"\brubber\b", r"\bliquid\b",
        r"\bsolid\b", r"\bheat(?:ed|s|ing)?\b", r"\bcold\b", r"\bwet\b", r"\bdry\b",
        r"\bsoft\b", r"\bhard\b", r"\blight\b", r"\bheavy\b", r"\btransparent\b",
    ],
    "spatial_direction_geometry": [
        r"\bleft\b", r"\bright\b", r"\bup\b", r"\bdown\b", r"\babove\b", r"\bbelow\b",
        r"\bunder\b", r"\bover\b", r"\binside\b", r"\boutside\b", r"\bfront\b", r"\bback\b",
        r"\bnorth\b", r"\bsouth\b", r"\beast\b", r"\bwest\b", r"\bnear\b", r"\bfar\b",
        r"\bthrough\b", r"\baround\b", r"\bacross\b", r"\bangle\b", r"\bdirection\b",
    ],
    "time_count_order": [
        r"\byear(?:s)?\b", r"\bmonth(?:s)?\b", r"\bweek(?:s)?\b", r"\bday(?:s)?\b",
        r"\bhour(?:s)?\b", r"\bminute(?:s)?\b", r"\bsecond(?:s)?\b", r"\bmorning\b", r"\bafternoon\b",
        r"\bevening\b", r"\bnight\b", r"\bbefore\b", r"\bafter\b", r"\bnext\b", r"\blast\b",
        r"\bearlier\b", r"\blater\b", r"\bfirst\b", r"\bsecond\b", r"\bthird\b", r"\bevery\b", r"\b\d{1,4}\b",
    ],
    "affordance_tool_action": [
        r"\buse(?:d|s|ing)?\b", r"\btool(?:s)?\b", r"\butensil(?:s)?\b", r"\bhold(?:s|ing)?\b",
        r"\bwear(?:s|ing)?\b", r"\bstore(?:s|d|ing)?\b", r"\bopen(?:s|ed|ing)?\b", r"\bclose(?:s|d|ing)?\b",
        r"\bcut(?:s|ting)?\b", r"\bwrite(?:s|ing)?\b", r"\bcook(?:s|ed|ing)?\b", r"\bclean(?:s|ed|ing)?\b",
        r"\bserve(?:s|d|ing)?\b", r"\bmake(?:s|making|made)?\b", r"\bbest\b", r"\bsafe(?:ly|ty)?\b",
    ],
    "explicit_causal_conditional": [
        r"\bbecause\b", r"\bdue to\b", r"\bso that\b", r"\bin order to\b", r"\btherefore\b", r"\bthus\b",
        r"\bas a result\b", r"\bcaus(?:e|es|ed|ing)\b", r"\bresult(?:s|ed|ing)?\b", r"\bif\b", r"\bwhen\b", r"\bthen\b",
        r"\bleads? to\b", r"\ballows?\b", r"\bprevent(?:s|ed|ing)?\b", r"\bmake(?:s)? .* (?:more|less|faster|slower|easier|harder)\b",
    ],
    "history_society_named_entity": [
        r"\bwar\b", r"\bgovernment\b", r"\bcountry\b", r"\bcity\b", r"\bpresident\b", r"\bking\b",
        r"\bqueen\b", r"\bcentury\b", r"\bpopulation\b", r"\belection\b", r"\bcourt\b", r"\blaw\b",
        r"\bcompany\b", r"\buniversity\b", r"\bchurch\b", r"\bstate\b", r"\bprovince\b",
    ],
}

WORD_RE = re.compile(r"\b\w+(?:['’-]\w+)?\b", re.UNICODE)


def words(text: str) -> List[str]:
    return WORD_RE.findall(text)


def count_matches(text: str) -> Dict[str, int]:
    out = {}
    for cat, pats in CATS.items():
        n = 0
        for pat in pats:
            n += len(re.findall(pat, text, flags=re.I))
        out[cat] = n
    return out


def rec_weight(rec: Dict[str, Any], field: str) -> int:
    if field == "source_text":
        return int(rec.get("source_words", len(words(str(rec.get(field,""))))))
    if field == "rewrite_text":
        return int(rec.get("rewrite_words", len(words(str(rec.get(field,""))))))
    if field == "text":
        return int(rec.get("words", len(words(str(rec.get(field,""))))))
    return len(words(str(rec.get(field,""))))


def load_pairs() -> List[Dict[str, Any]]:
    return [json.loads(l) for l in PAIRS.read_text().splitlines() if l.strip()]


def load_breadth() -> List[Dict[str, Any]]:
    return [json.loads(l) for l in BREADTH.read_text().splitlines() if l.strip()]


def aggregate(records: List[Dict[str, Any]], text_field: str, label: str) -> Dict[str, Any]:
    total_words = 0
    n_records = 0
    any_count = Counter()
    hit_word_count = Counter()
    match_count = Counter()
    domain_counts = Counter()
    domain_words = Counter()
    sample_hits = defaultdict(list)
    for rec in records:
        text = str(rec.get(text_field, ""))
        w = rec_weight(rec, text_field)
        total_words += w
        n_records += 1
        cm = count_matches(text)
        any_hit = False
        for cat, n in cm.items():
            if n > 0:
                any_count[cat] += 1
                hit_word_count[cat] += w
                match_count[cat] += n
                any_hit = True
                if len(sample_hits[cat]) < 8:
                    sample_hits[cat].append({
                        "text": text[:260],
                        "words": w,
                        "domains": rec.get("domains"),
                        "doc_id": rec.get("doc_id"),
                        "norm_hash": rec.get("norm_hash"),
                    })
        if not any_hit:
            any_count["no_broad_hits"] += 1
            hit_word_count["no_broad_hits"] += w
        ds = rec.get("domains") or []
        if isinstance(ds, str):
            ds = [ds]
        if not ds:
            ds = ["no_domain_metadata"]
        for d in ds:
            domain_counts[d] += 1
            domain_words[d] += w
    density = {cat: match_count[cat] / total_words * 10000 for cat in CATS}
    hit_frac_records = {cat: any_count[cat] / n_records for cat in CATS}
    hit_frac_words = {cat: hit_word_count[cat] / total_words for cat in CATS}
    composite_gpiqa_like = ["physical_motion_force", "object_material_state", "spatial_direction_geometry", "time_count_order", "affordance_tool_action", "explicit_causal_conditional"]
    comp_any_records = 0
    comp_words = 0
    for rec in records:
        text = str(rec.get(text_field, ""))
        w = rec_weight(rec, text_field)
        if any(count_matches(text)[cat] > 0 for cat in composite_gpiqa_like):
            comp_any_records += 1
            comp_words += w
    return {
        "label": label,
        "n_records": n_records,
        "total_words": total_words,
        "category_match_count": dict(match_count),
        "category_matches_per_10k_words": density,
        "category_record_hit_fraction": hit_frac_records,
        "category_word_hit_fraction": hit_frac_words,
        "composite_gpiqa_broad_hit_record_fraction": comp_any_records / n_records,
        "composite_gpiqa_broad_hit_word_fraction": comp_words / total_words,
        "domain_counts_top": domain_counts.most_common(30),
        "domain_words_top": domain_words.most_common(30),
        "sample_hits": {k:v for k,v in sample_hits.items()},
    }


def ratio(a: float, b: float) -> float | None:
    if b == 0:
        return None
    return a / b


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    pairs = load_pairs()
    breadth = load_breadth()
    common_source = aggregate(pairs, "source_text", "common compact/breadth FineWeb source anchors")
    compact_rewrite = aggregate(pairs, "rewrite_text", "compact arm changed companion: Qwen3.5 compact rewrites")
    breadth_comp = aggregate(breadth, "text", "breadth arm changed companion: independent FineWeb whole sentences")
    records = [common_source, compact_rewrite, breadth_comp]
    comparison = []
    for cat in CATS:
        cr = compact_rewrite["category_matches_per_10k_words"].get(cat,0.0)
        br = breadth_comp["category_matches_per_10k_words"].get(cat,0.0)
        cs = common_source["category_matches_per_10k_words"].get(cat,0.0)
        comparison.append({
            "category": cat,
            "common_source_per10k": cs,
            "compact_rewrite_per10k": cr,
            "breadth_companion_per10k": br,
            "breadth_over_compact_rewrite": ratio(br, cr),
            "compact_rewrite_over_common_source": ratio(cr, cs),
            "breadth_over_common_source": ratio(br, cs),
            "compact_rewrite_record_hit_fraction": compact_rewrite["category_record_hit_fraction"].get(cat,0.0),
            "breadth_companion_record_hit_fraction": breadth_comp["category_record_hit_fraction"].get(cat,0.0),
        })
    payload = {
        "status": "FW_GLOBALPIQA_RELEVANT_SUBSTRATE_DONE",
        "inputs": {"pairs": str(PAIRS), "breadth": str(BREADTH), "interleaved_manifest": str(MANIFEST)},
        "important_budget_fact": "The compact-vs-interleaved-breadth arms share 494,154 FineWeb source words and differ only in a matched 318,851-word companion budget: compact rewrites vs independent whole-sentence FineWeb breadth.",
        "records": records,
        "comparison": comparison,
        "interpretation": {
            "what_this_can_support": "This file can interpret post-training vectors. It cannot prove a causal result until the matched models are trained and officially evaluated.",
            "if_breadth_has_more_physical_causal_substrate": "A GlobalPIQA gain in breadth would naturally point to missing broad physical/temporal/spatial/action experience; a compact win despite lower such substrate would strengthen consolidation/restatement as the active mechanism.",
            "if_compact_has_more_causal_conditional_markers": "A compact gain may reflect denser explicit relation wording rather than only repetition; then the research source_repeat arm remains the attribution comparator.",
            "caution": "The lexicons are broad and only descriptive. They must not be used as evaluation-item shaping for future training data.",
        },
    }
    (OUT/"fw_globalpiqa_relevant_substrate.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    # CSV tables
    with (OUT/"fw_substrate_category_comparison.csv").open("w", newline="") as f:
        writer=csv.DictWriter(f, fieldnames=list(comparison[0].keys()))
        writer.writeheader(); writer.writerows(comparison)
    with (OUT/"fw_substrate_domain_counts.csv").open("w", newline="") as f:
        writer=csv.writer(f)
        writer.writerow(["substrate","rank","domain","record_count","word_count"])
        for rec in records:
            domain_words = dict(rec["domain_words_top"])
            for i,(dom,cnt) in enumerate(rec["domain_counts_top"], start=1):
                writer.writerow([rec["label"], i, dom, cnt, domain_words.get(dom,"")])
    # Note
    def fmt(x):
        return "NA" if x is None else f"{x:.3f}"
    comp_map={r["category"]: r for r in comparison}
    lines=[]
    lines.append("# research — FW compact/breadth substrate and GlobalPIQA-relevant domains")
    lines.append("")
    lines.append("## Budget fact")
    lines.append("")
    lines.append("The active FW experiment shares the same 494,154 FineWeb source-anchor words. The only changed FineWeb companion budget is matched at 318,851 words: compact arm uses preserved Qwen3.5 compact rewrites, while interleaved breadth uses independent whole FineWeb sentences. This note reads that changed budget before interpreting future scores.")
    lines.append("")
    lines.append("## Broad category density in the changed companion budget")
    lines.append("")
    lines.append("Densities are regex matches per 10k words, descriptive only.")
    lines.append("")
    lines.append("| broad category | common source | compact rewrites | breadth companions | breadth/compact |")
    lines.append("|---|---:|---:|---:|---:|")
    for cat in CATS:
        r=comp_map[cat]
        lines.append(f"| `{cat}` | {r['common_source_per10k']:.2f} | {r['compact_rewrite_per10k']:.2f} | {r['breadth_companion_per10k']:.2f} | {fmt(r['breadth_over_compact_rewrite'])} |")
    lines.append("")
    lines.append("Composite broad-GlobalPIQA hit fractions:")
    for rec in records:
        lines.append(f"- {rec['label']}: record fraction {rec['composite_gpiqa_broad_hit_record_fraction']:.3f}, word fraction {rec['composite_gpiqa_broad_hit_word_fraction']:.3f}, words {rec['total_words']}")
    lines.append("")
    lines.append("## Reading for the upcoming FW results")
    lines.append("")
    lines.append("If compact beats both breadth layouts while breadth has equal or richer broad physical/causal substrate, the result is more naturally about same-proposition consolidation, shorter denser restatement, and repeated relation surfaces than about seeing more raw physical facts. If breadth beats compact, especially on GlobalPIQA_parallel and EWoK, the current compact block is not the right use of the 318,851-word companion budget; the next data route should mine a general, independently selected physical/temporal/spatial/action-consequence substrate rather than add more compact paraphrases indiscriminately. If compact and breadth are close but differ by row-block/interleaved layout, local alternation/packing is active and the research source_repeat arm becomes the attribution comparator.")
    lines.append("")
    lines.append("## Files")
    lines.append("")
    lines.append(f"- JSON: `{OUT/'fw_globalpiqa_relevant_substrate.json'}`")
    lines.append(f"- category CSV: `{OUT/'fw_substrate_category_comparison.csv'}`")
    lines.append(f"- domain CSV: `{OUT/'fw_substrate_domain_counts.csv'}`")
    NOTE.write_text("\n".join(lines)+"\n")
    print(json.dumps({
        "status": payload["status"],
        "json": str(OUT/"fw_globalpiqa_relevant_substrate.json"),
        "category_csv": str(OUT/"fw_substrate_category_comparison.csv"),
        "domain_csv": str(OUT/"fw_substrate_domain_counts.csv"),
        "note": str(NOTE),
        "compact_rewrite_words": compact_rewrite["total_words"],
        "breadth_companion_words": breadth_comp["total_words"],
        "common_source_words": common_source["total_words"],
    }, indent=2))

if __name__ == "__main__":
    main()
