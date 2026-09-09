#!/usr/bin/env python3
"""research: build a balanced label-free corpus-internal retention probe.

Unlike the first random probe, this deliberately oversamples closed-class and
low-frequency structural carriers (negation, quantifiers, wh, auxiliaries,
prepositions, pronouns, numbers, rare content) because the official late loss is a
vector-retention phenomenon in relation/state abilities rather than scalar MLM
convergence. Categories are defined ONLY from corpus text and English word classes,
not from official benchmark examples or labels.

Output: data/retention_probe/targeted_retention_probe.json
"""
from __future__ import annotations
import json, random, re, hashlib
from pathlib import Path
from collections import Counter, defaultdict

ROOT = Path("experiments/archive/frontier_consolidation")
POOL = ROOT / "data/density_cleanqwen_overlay_medium_riskhard/cleanqwen_fineweb_compact_view_reinvest_10M.jsonl"
OUT_DIR = ROOT / "data/retention_probe"
OUT_DIR.mkdir(parents=True, exist_ok=True)

NEGATION = set("no not n't never neither nor nobody nothing nowhere without cannot".split())
QUANT = set("all any every each few many most much several some none both either neither one two three four five six seven eight nine ten more less fewer enough half whole multiple various".split())
WH = set("what which who whom whose when where why how".split())
AUX = set("am is are was were be been being do does did have has had having will would shall should can could may might must ought need dare".split())
PREP = set("of in on at by for with about against between into through during before after above below to from up down over under across along around behind beneath beside besides beyond inside outside toward towards upon within near off out onto per via until since than as like".split())
PRONDET = set("i you he she it we they me him us them my your his her its our their mine yours hers ours theirs this that these those a an the another other same such whose".split())
CONJ = set("and but or yet because if unless although though while whereas whether either nor so then once".split())
NUMBER_RE = re.compile(r"^[+-]?\d+(?:[.,:]\d+)*%?$")
WORD_CLEAN_RE = re.compile(r"^[\W_]+|[\W_]+$")


def norm(w: str) -> str:
    return WORD_CLEAN_RE.sub("", w).lower()


def category(w: str, freq: Counter, rare_cut: int) -> str:
    x = norm(w)
    if not x:
        return "punct_or_symbol"
    if NUMBER_RE.match(x):
        return "number"
    if x in NEGATION:
        return "negation"
    if x in WH:
        return "wh"
    if x in QUANT:
        return "quantifier"
    if x in AUX:
        return "aux_modal"
    if x in PREP:
        return "preposition_relation"
    if x in PRONDET:
        return "pronoun_determiner"
    if x in CONJ:
        return "conjunction_marker"
    if freq[x] <= rare_cut:
        return "rare_content"
    return "common_content"


def main(max_per_source_category=160, seed=1357, max_words=260):
    rows=[]
    freq=Counter()
    with open(POOL) as f:
        for line in f:
            o=json.loads(line)
            if o["source"].startswith("neutral_"):
                continue
            words=o["text"].split()
            if 8 <= len(words) <= max_words:
                rows.append(o)
                freq.update(norm(w) for w in words if norm(w))
    # Rare cut: corpus word types with count <= 3 are truly long tail in this 10M pool.
    rare_cut=3
    buckets=defaultdict(list)
    for ri,o in enumerate(rows):
        source=o["source"]
        words=o["text"].split()
        for idx,w in enumerate(words):
            if idx==0 or idx==len(words)-1:
                continue
            cat=category(w, freq, rare_cut)
            if cat == "punct_or_symbol":
                continue
            key=(source, cat)
            # one record per target word position
            buckets[key].append({
                "source": source,
                "structure": cat,
                "text": o["text"],
                "words": words,
                "masked_idx": idx,
                "target_word": w,
                "is_function_word": cat in {"negation","wh","quantifier","aux_modal","preposition_relation","pronoun_determiner","conjunction_marker"},
                "word_frequency_in_10M_pool": int(freq[norm(w)]),
                "example_id": o.get("example_id"),
            })
    rng=random.Random(seed)
    selected=[]
    bucket_counts={}
    for key, items in sorted(buckets.items()):
        rng.shuffle(items)
        chosen=items[:max_per_source_category]
        # Stable order inside output independent of original random shuffle artifact.
        chosen=sorted(chosen, key=lambda r: hashlib.sha256((r["source"]+"|"+r["structure"]+"|"+r["text"]+"|"+str(r["masked_idx"])).encode()).hexdigest())
        selected.extend(chosen)
        bucket_counts[f"{key[0]}::{key[1]}"]=len(chosen)
    summary={
        "targets_total": len(selected),
        "max_per_source_category": max_per_source_category,
        "seed": seed,
        "pool": str(POOL),
        "rare_cut_frequency_leq": rare_cut,
        "category_definitions": {
            "negation": sorted(NEGATION),
            "quantifier": sorted(QUANT),
            "wh": sorted(WH),
            "aux_modal": sorted(AUX),
            "preposition_relation": sorted(PREP),
            "pronoun_determiner": sorted(PRONDET),
            "conjunction_marker": sorted(CONJ),
            "number": "regex numeric",
            "rare_content": f"non-function corpus word frequency <= {rare_cut}",
            "common_content": f"remaining content words frequency > {rare_cut}",
        },
        "bucket_counts": bucket_counts,
        "source_counts": dict(Counter(r["source"] for r in selected)),
        "structure_counts": dict(Counter(r["structure"] for r in selected)),
    }
    out={"rows": selected, "summary": summary}
    out_path=OUT_DIR/"targeted_retention_probe.json"
    out_path.write_text(json.dumps(out), encoding="utf-8")
    md=[]
    md.append("# research targeted retention probe")
    md.append("")
    md.append(f"Targets: {len(selected)}; max per source×category: {max_per_source_category}; rare_content frequency <= {rare_cut}.")
    md.append("")
    md.append("## Source counts")
    for k,v in sorted(summary["source_counts"].items()): md.append(f"- {k}: {v}")
    md.append("")
    md.append("## Structure counts")
    for k,v in sorted(summary["structure_counts"].items()): md.append(f"- {k}: {v}")
    md.append("")
    md.append(f"JSON: `{out_path}`")
    ((OUT_DIR.parents[4] / 'research/documents/frontier_consolidation/data/retention_probe/targeted_retention_probe.md')).write_text("\n".join(md), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)

if __name__ == "__main__":
    main()
