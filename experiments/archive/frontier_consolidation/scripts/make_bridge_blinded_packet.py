#!/usr/bin/env python3
"""Make a blinded reader packet for research retained source-attested bridge outputs.

The packet omits source_copy_degree and the research automatic edit class. A reader sees
only source, candidate, natural compact reference, and prompt bucket/regime. The goal is
to judge whether candidates are proposition-preserving transformations or polished
extracts, without the flawed lemma-overlap statistic.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import json
from pathlib import Path

ROOT = _public_path('.')
WORKSPACE = _public_path('experiments/archive/frontier_consolidation')
ACCEPTED = _public_path('experiments/archive/frontier_consolidation/data/improved_fluent_bridge/final_accepted.jsonl')
OUT_DIR = _public_path('experiments/archive/frontier_consolidation/data/bridge_blinded_reading')
OUT_MD = _public_path('research/documents/frontier_consolidation/data/bridge_blinded_reading/bridge_blinded_reader_packet.md')
OUT_JSONL = _public_path('experiments/archive/frontier_consolidation/data/bridge_blinded_reading/bridge_blinded_cases.jsonl')


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    with ACCEPTED.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    md = [
        "# Blinded reader packet: retained source-attested bridge outputs",
        "",
        "Scientific question: Are these candidates fluent proposition-preserving transformations, or mostly polished extracts/deletions?",
        "",
        "Important: Do not judge by whether candidate content words occur in the source; the construction intentionally asks for source-attested lexical material. Judge structural transformation, clause/predicate-argument recoding, function-word restructuring, and proposition preservation.",
        "",
        "Suggested labels per case:",
        "- transformation: substantive clause/argument/function-word re-expression while preserving the source proposition.",
        "- light_transform: small but real grammatical compression or reordering beyond contiguous extraction.",
        "- extract: contiguous or near-contiguous source substring with deletion only or nearly source-order keyword packing.",
        "- damaged: ungrammatical, proposition changed, important modifier/negation/complement lost, or argument reassigned.",
        "",
    ]
    out_rows = []
    for i, r in enumerate(rows, 1):
        rec = {
            "case_id": f"B{i:03d}",
            "bucket": r.get("prototype_bucket"),
            "regime": r.get("regime"),
            "source_text": r.get("source_text"),
            "candidate_text": r.get("generated_text") or r.get("raw_output"),
            "natural_compact_reference": r.get("natural_compact_text"),
            "source_words": r.get("source_words"),
            "candidate_words": r.get("generated_words"),
            "natural_words": r.get("natural_words"),
            "compact_absent_content_lemma_count": r.get("compact_absent_content_lemma_count"),
            "relation_count_proxy": r.get("relation_count_proxy"),
        }
        out_rows.append(rec)
        md.extend([
            f"## {rec['case_id']} ({rec['bucket']}; {rec['regime']})",
            f"Source ({rec['source_words']}w): {rec['source_text']}",
            f"Candidate ({rec['candidate_words']}w): {rec['candidate_text']}",
            f"Natural compact reference ({rec['natural_words']}w): {rec['natural_compact_reference']}",
            "",
        ])
    OUT_JSONL.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in out_rows), encoding="utf-8")
    OUT_MD.write_text("\n".join(md), encoding="utf-8")
    print(json.dumps({"status":"BRIDGE_BLINDED_PACKET_WRITTEN", "n": len(out_rows), "md": str(OUT_MD.relative_to(ROOT)), "jsonl": str(OUT_JSONL.relative_to(ROOT))}, indent=2), flush=True)


if __name__ == "__main__":
    main()
