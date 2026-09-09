#!/usr/bin/env python3
"""Analyze the research Qwen generation slice for factual faithfulness, breadth, and word accounting.

This is a conservative automated preflight, not a truth oracle. It checks whether the
larger generation route deserves H100 time by measuring things we can compute from
source/output pairs: schema integrity, source entity/number/date retention, output/source
length ratios, repetition and disclaimer artifacts, lexical/domain breadth, and evidence of
unsupported entity injection in expansions.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import statistics as stats
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path("experiments/archive/representation_and_objectives")
PROMPTS = ROOT / "training/data/generation_slice/slice_prompts.jsonl"
OUTPUTS = ROOT / "training/runs/generation_slice_qwen/outputs.jsonl"
OUT_DIR = ROOT / "training/data/generation_slice"
REPORT = (ROOT.parents[2] / 'research/notes/representation_and_objectives/generation_slice_quality.md')

STOP = {
    "the", "a", "an", "and", "or", "of", "in", "to", "for", "on", "with", "as", "by", "from", "at", "is", "are", "was", "were", "be", "been", "being",
    "it", "its", "this", "that", "these", "those", "he", "she", "they", "them", "his", "her", "their", "which", "who", "whom", "when", "where", "while",
    "also", "one", "two", "new", "old", "first", "second", "major", "minor", "may", "can", "could", "would", "should", "about", "into", "after", "before",
    "during", "over", "under", "between", "through", "than", "then", "there", "not", "only", "other", "such", "more", "most", "less", "very", "same",
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
BAD_PATTERNS = [
    re.compile(p, re.I) for p in [
        r"^\s*(sure|here('| i)s|certainly)\b", r"as an ai", r"i (cannot|can't)", r"please provide", r"output only", r"simplified text\s*:",
        r"the original text", r"not named in the original", r"potentially a confusion", r"while the specific .* is not named",
    ]
]
CAP_SEQ_RE = re.compile(r"\b(?:[A-Z][\w'’.-]*(?:\s+|$)){1,6}")
NUM_RE = re.compile(r"\b(?:\d{1,4}(?:[,.]\d+)*(?:[-–]\d{1,4})?|\d+(?:\.\d+)?\s*(?:%|km|m|ft|million|billion|thousand)|[A-Z]{2,}\d*)\b")
WORD_RE = re.compile(r"[A-Za-z][A-Za-z'’.-]*")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def tokens(text: str) -> list[str]:
    return [w.lower().strip("'’.-") for w in WORD_RE.findall(text)]


def content_tokens(text: str) -> set[str]:
    return {t for t in tokens(text) if len(t) > 3 and t not in STOP}


def entities(text: str) -> set[str]:
    ents: set[str] = set()
    for m in CAP_SEQ_RE.finditer(text):
        ent = " ".join(m.group(0).split()).strip(" ,.;:()[]{}\"'")
        parts = ent.split()
        if not parts:
            continue
        if len(parts) == 1 and parts[0].lower() in STOP:
            continue
        # Ignore sentence-initial generic words unless multiword or strongly name-like.
        if len(parts) == 1 and parts[0] in {"The", "This", "These", "A", "An", "In", "On", "After", "Before", "During", "He", "She", "It", "They"}:
            continue
        if len(ent) >= 2:
            ents.add(ent)
    return ents


def nums(text: str) -> set[str]:
    return {m.group(0).strip() for m in NUM_RE.finditer(text)}


def domain_hits(text: str) -> dict[str, int]:
    toks = content_tokens(text)
    return {k: len(toks & v) for k, v in DOMAIN_TERMS.items()}


def jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / max(1, len(a | b))


def pct(vals: list[float], q: float) -> float | None:
    if not vals:
        return None
    xs = sorted(vals)
    idx = min(len(xs)-1, max(0, int(round((len(xs)-1)*q))))
    return xs[idx]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--prompts", default=str(PROMPTS))
    p.add_argument("--outputs", default=str(OUTPUTS))
    p.add_argument("--out-dir", default=str(OUT_DIR))
    p.add_argument("--report", default=str(REPORT))
    args = p.parse_args()

    prompt_rows = read_jsonl(Path(args.prompts))
    out_rows = read_jsonl(Path(args.outputs))
    if len(prompt_rows) != len(out_rows):
        raise RuntimeError(f"row count mismatch prompts={len(prompt_rows)} outputs={len(out_rows)}")

    rows = []
    for p_row, o_row in zip(prompt_rows, out_rows):
        if int(o_row.get("index")) != int(p_row.get("slice_index")):
            raise RuntimeError(f"index mismatch output {o_row.get('index')} vs prompt slice_index {p_row.get('slice_index')}")
        source = str(p_row["source_text"])
        output = str(o_row.get("output", "")).strip()
        stoks = content_tokens(source)
        otoks = content_tokens(output)
        sents = entities(source)
        oents = entities(output)
        snums = nums(source)
        onums = nums(output)
        bad = [rx.pattern for rx in BAD_PATTERNS if rx.search(output)]
        row = {
            "slice_index": p_row["slice_index"],
            "id": p_row["id"],
            "type": p_row["type"],
            "source_article": p_row.get("source_article"),
            "source_words_field": int(p_row.get("source_words", len(source.split()))),
            "source_words_actual": len(source.split()),
            "output_words": len(output.split()),
            "length_ratio": len(output.split()) / max(1, len(source.split())),
            "content_jaccard": jaccard(stoks, otoks),
            "source_entities": sorted(sents),
            "output_entities": sorted(oents),
            "entity_recall": len({e.lower() for e in sents} & {e.lower() for e in oents}) / max(1, len(sents)),
            "new_entity_count": len({e.lower() for e in oents} - {e.lower() for e in sents}),
            "source_numbers": sorted(snums),
            "output_numbers": sorted(onums),
            "number_recall": len(snums & onums) / max(1, len(snums)) if snums else 1.0,
            "new_number_count": len(onums - snums),
            "bad_patterns": bad,
            "bad_pattern_count": len(bad),
            "domain_hits": domain_hits(output),
            "source": source,
            "output": output,
            "generated_tokens": o_row.get("generated_tokens"),
        }
        rows.append(row)

    by_type: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_type[r["type"]].append(r)

    summary = {
        "status": "GENERATION_SLICE_ANALYZED",
        "prompts": str(Path(args.prompts)),
        "outputs": str(Path(args.outputs)),
        "prompt_sha256": sha256_file(Path(args.prompts)),
        "output_sha256": sha256_file(Path(args.outputs)),
        "n": len(rows),
        "overall": {},
        "by_type": {},
    }

    for typ, rs in [("overall", rows)] + sorted(by_type.items()):
        lengths = [r["output_words"] for r in rs]
        ratios = [r["length_ratio"] for r in rs]
        jac = [r["content_jaccard"] for r in rs]
        erec = [r["entity_recall"] for r in rs]
        nrec = [r["number_recall"] for r in rs]
        bad_count = sum(1 for r in rs if r["bad_pattern_count"])
        expansion_inj = sum(1 for r in rs if r["type"] == "expansion" and (r["new_entity_count"] >= 3 or r["new_number_count"] >= 2 or r["bad_pattern_count"]))
        dom_counter = Counter()
        for r in rs:
            for k, v in r["domain_hits"].items():
                if v:
                    dom_counter[k] += 1
        obj = {
            "n": len(rs),
            "words_total": sum(lengths),
            "output_words_mean": round(stats.mean(lengths), 2) if lengths else None,
            "output_words_p10_p50_p90": [pct(lengths, q) for q in (0.1, 0.5, 0.9)],
            "length_ratio_mean": round(stats.mean(ratios), 3) if ratios else None,
            "length_ratio_p10_p50_p90": [round(pct(ratios, q), 3) for q in (0.1, 0.5, 0.9)] if ratios else None,
            "content_jaccard_mean": round(stats.mean(jac), 3) if jac else None,
            "entity_recall_mean": round(stats.mean(erec), 3) if erec else None,
            "number_recall_mean": round(stats.mean(nrec), 3) if nrec else None,
            "new_entities_per_row_mean": round(stats.mean([r["new_entity_count"] for r in rs]), 3) if rs else None,
            "new_numbers_per_row_mean": round(stats.mean([r["new_number_count"] for r in rs]), 3) if rs else None,
            "bad_pattern_rows": bad_count,
            "bad_pattern_rate": round(bad_count / max(1, len(rs)), 3),
            "domain_row_hits": dict(dom_counter),
        }
        if typ == "overall":
            summary["overall"] = obj
        else:
            obj["high_risk_rows"] = expansion_inj if typ == "expansion" else sum(1 for r in rs if r["bad_pattern_count"] or r["number_recall"] < 0.5 or r["entity_recall"] < 0.5)
            summary["by_type"][typ] = obj

    # Identify examples worth human/source inspection.
    risk_sorted = sorted(rows, key=lambda r: (
        r["bad_pattern_count"],
        1 if r["type"] == "expansion" and (r["new_entity_count"] >= 3 or r["new_number_count"] >= 2) else 0,
        1-r["number_recall"], 1-r["entity_recall"], r["new_entity_count"] + r["new_number_count"]
    ), reverse=True)
    representative = []
    for typ in ["simplification", "paraphrase", "expansion"]:
        typ_rows = by_type[typ]
        representative.extend(sorted(typ_rows, key=lambda r: abs(r["length_ratio"] - (1 if typ != "expansion" else 3)))[:3])
    summary["risk_sample"] = [
        {k: r[k] for k in ["slice_index", "id", "type", "source_article", "output_words", "length_ratio", "entity_recall", "new_entity_count", "number_recall", "new_number_count", "bad_patterns", "source", "output"]}
        for r in risk_sorted[:24]
    ]
    summary["representative_sample"] = [
        {k: r[k] for k in ["slice_index", "id", "type", "source_article", "output_words", "length_ratio", "entity_recall", "new_entity_count", "number_recall", "new_number_count", "source", "output"]}
        for r in representative
    ]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "slice_quality_summary.json"
    rows_path = out_dir / "slice_quality_rows.jsonl"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    with rows_path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Compose concise scientific note.
    def fmt_type(typ: str) -> str:
        o = summary["by_type"][typ]
        return (f"- {typ}: n={o['n']}, words={o['words_total']}, mean ratio={o['length_ratio_mean']}, "
                f"entity recall={o['entity_recall_mean']}, number recall={o['number_recall_mean']}, "
                f"new entities/row={o['new_entities_per_row_mean']}, bad-pattern rows={o['bad_pattern_rows']}, "
                f"high-risk rows={o['high_risk_rows']}")

    exp = summary["by_type"].get("expansion", {})
    verdict = []
    if exp:
        if exp["bad_pattern_rate"] > 0.05 or exp["new_entities_per_row_mean"] > 2.0 or exp["new_numbers_per_row_mean"] > 1.0:
            verdict.append("The expansion prompt is scientifically unsafe as a bulk pretraining source without filtering: it often injects new names/numbers or meta-text not anchored in the SimpleWiki source.")
        else:
            verdict.append("The expansion prompt did not show severe automated risk on this slice, but manual spot-checking remains necessary before scaling.")
    simp = summary["by_type"].get("simplification", {})
    para = summary["by_type"].get("paraphrase", {})
    if simp and para:
        verdict.append("Simplification/paraphrase outputs are closer to the source than expansions and are better candidates for a low-risk source-paired data arm.")
    verdict.append("The larger 50k generation should not proceed unchanged; a safer arm should keep original SimpleWiki/official text plus source-conservative rewrites, and treat free expansions as either excluded or heavily filtered.")

    md = []
    md.append("# research — Qwen generation slice quality and provenance check\n")
    md.append("## Purpose\n")
    md.append("The slice tests whether the research plan can safely scale Qwen-generated SimpleWiki derivatives before spending both H100s on the full 50k generation, 40k tokenizer, and 12×384 training package. The measurements are source/output process checks, not a substitute for official training evaluation.\n")
    md.append("## Run evidence\n")
    md.append(f"- Prompts: `{args.prompts}` ({len(prompt_rows)} rows, sha256 `{summary['prompt_sha256']}`)\n")
    md.append(f"- Outputs: `{args.outputs}` ({len(out_rows)} rows, sha256 `{summary['output_sha256']}`)\n")
    md.append(f"- Summary JSON: `{summary_path}`\n")
    md.append(f"- Row metrics JSONL: `{rows_path}`\n")
    md.append("\n## Aggregate measurements\n")
    for typ in ["simplification", "paraphrase", "expansion"]:
        md.append(fmt_type(typ) + "\n")
    md.append("\nOverall domain row hits: " + json.dumps(summary["overall"]["domain_row_hits"], ensure_ascii=False) + "\n")
    md.append("\n## Scientific interpretation\n")
    for v in verdict:
        md.append(f"- {v}\n")
    md.append("\n## High-risk examples to inspect\n")
    for r in summary["risk_sample"][:8]:
        md.append(f"\n### {r['id']} ({r['type']}, article={r['source_article']})\n")
        md.append(f"- length_ratio={r['length_ratio']:.2f}, entity_recall={r['entity_recall']:.2f}, new_entity_count={r['new_entity_count']}, number_recall={r['number_recall']:.2f}, new_number_count={r['new_number_count']}, bad_patterns={r['bad_patterns']}\n")
        md.append(f"- Source: {r['source']}\n")
        md.append(f"- Output: {r['output']}\n")
    md.append("\n## Next experimental implication\n")
    md.append("Do a matched factor cross before any confounded leader imitation: (1) build a source-conservative SimpleWiki rewrite data arm under the protected 8×480/16k recipe, with exact word accounting and provenance; (2) separately change recipe/architecture on fixed clean-Qwen or mix25 data only if there is a representative reason to believe the previous 12×384/LAMB/40k negative came from data mismatch rather than the recipe itself.\n")
    Path(args.report).write_text("".join(md), encoding="utf-8")
    print(json.dumps({"status": "ok", "summary": str(summary_path), "rows": str(rows_path), "report": args.report}, indent=2))


if __name__ == "__main__":
    main()
