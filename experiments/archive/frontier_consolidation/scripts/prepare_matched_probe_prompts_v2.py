#!/usr/bin/env python3
"""research v2 prompt builder for matched graph-transfer probes.

The v1 generator mostly copied the answer words into the context views.  This v2
prompt hard-codes lexicalization rules: context views must use held-out clue
words (later/earlier, resulted/although, approved/refused, rose/fell) while the
MLM query uses the answer pair only at the masked position.  It remains a small
construction viability test, not a training corpus.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import json
from pathlib import Path
from typing import Any

ROOT = _public_path('.')
WORKSPACE = _public_path('experiments/archive/frontier_consolidation')
PACKET = _public_path('experiments/archive/frontier_consolidation/data/graph_packet_v0/graph_packet_v0.jsonl')
OUT_DIR = _public_path('experiments/archive/frontier_consolidation/data/matched_graph_transfer_probe')
PROMPTS = _public_path('experiments/archive/frontier_consolidation/data/matched_graph_transfer_probe/matched_probe_generation_prompts_v2.jsonl')
META = _public_path('experiments/archive/frontier_consolidation/data/matched_graph_transfer_probe/prompt_manifest_v2.json')

FAMILY_RULES = {
    "entity_state_update": "Prefer temporal/state. Use target/distractor after/before or entered/left. Context clue words: later than vs earlier than, moved into vs moved away from, became present vs became absent. Neutral clue: mentions both entities/events without order or state direction.",
    "event_temporal_causal": "Prefer temporal/causal. Use after/before or because/despite. Context clue words: later than vs earlier than, resulted from vs happened although. Neutral clue: mentions both events without direction or cause.",
    "quantity_change_compare": "Prefer quantity. Use more/less or gained/lost. Context clue words: rose/increased vs fell/decreased, added vs removed. Neutral clue: amount changed or was measured without sign.",
    "social_belief_report": "Prefer belief/social polarity. Use accepted/rejected or true/false. Context clue words: approved/believed vs refused/doubted. Neutral clue: considered or discussed without stance.",
    "polarity_contrast_event": "Prefer polarity/causal contrast. Use true/false, accepted/rejected, or because/despite. Context clue words: confirmed vs denied, approved vs refused, resulted from vs happened although. Neutral clue: says the issue was discussed without verdict.",
}

EXAMPLE = """Example (do not copy its content):
{
  "ok": true,
  "relation_family": "temporal",
  "target": "after",
  "distractor": "before",
  "view_structured": "Mira joined the academy later than the festival announcement in the town record.",
  "view_neutral": "The town record mentions Mira, the academy, and the festival announcement in one account.",
  "view_reversed": "Mira joined the academy earlier than the festival announcement in the town record.",
  "query_prefix": "Mira joined the academy",
  "query_suffix": "the festival announcement appeared in the town record.",
  "correct_full_query": "Mira joined the academy after the festival announcement appeared in the town record.",
  "distractor_full_query": "Mira joined the academy before the festival announcement appeared in the town record.",
  "edge_correct": "joining is later than announcement",
  "edge_reversed": "joining is earlier than announcement",
  "new_names": ["Mira", "the academy", "the festival announcement", "the town record"]
}
Notice: neither after nor before appears in any view or in query_prefix/query_suffix; the words appear only in correct_full_query and distractor_full_query at the answer slot."""


def load_rows() -> list[dict[str, Any]]:
    rows=[]
    with PACKET.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                obj=json.loads(line)
                issues=obj.get("verification_issues") or []
                graph=obj.get("graph") or {}
                rels=graph.get("relations") or []
                ents=graph.get("entities") or []
                score=0
                score += 4 if not issues else -2*len(issues)
                score += min(6, len(rels))
                score += min(4, len(ents))
                score += 2 if obj.get("source_bucket") == "SimpleWiki" else 0
                # Penalize graph_compact rows that already looked suspect.
                if any("low entity coverage" in str(x) for x in issues): score -= 3
                obj["_score"] = score
                rows.append(obj)
    return rows


def graph_digest(row: dict[str, Any]) -> str:
    g=row.get("graph") or {}
    ents=g.get("entities") or []
    rels=g.get("relations") or []
    evs=g.get("key_events") or []
    out=[]
    out.append("Entities: " + "; ".join(str(e.get("name")) for e in ents[:7]))
    out.append("Relations: " + "; ".join(f"{r.get('type')}: {r.get('arg1')} -> {r.get('arg2')} ({r.get('detail')})" for r in rels[:7]))
    out.append("Events: " + "; ".join(str(e.get("event")) for e in evs[:5]))
    return "\n".join(out)[:2000]


def make_prompt(row: dict[str, Any], idx: int) -> dict[str, Any]:
    fam=str(row.get("family"))
    source=str(row.get("source_text") or "")[:1300]
    prompt=f"""Return exactly one compact JSON object. Build a masked-language-model diagnostic item from the legal source row and graph.

Goal: test RELATIONAL TRANSFER without answer-word copying. The context views must have matched style, matched entities, similar length, and equal answer-word availability: the target word and distractor word are ABSENT from all views and absent from query_prefix/query_suffix.

Family rule for this row: {FAMILY_RULES.get(fam, 'Use a clean temporal, causal, state, quantity, belief, polarity, or social edge.')}

{EXAMPLE}

Source row (for factual inspiration only; use new names in your item):
{source}

Extracted graph:
{graph_digest(row)}

Schema:
{{"ok": true, "relation_family": "temporal|causal|state|quantity|belief|polarity|social", "target": "after|before|because|despite|more|less|gained|lost|entered|left|accepted|rejected|true|false", "distractor": "opposite from allowed pairs", "view_structured": "18-34 words; uses clue synonyms, not target/distractor", "view_neutral": "18-34 words; same names/content but no direction/cause/state/sign/stance", "view_reversed": "18-34 words; minimally flips only the edge using clue synonyms, not target/distractor", "query_prefix": "prefix ending just before answer word; no target/distractor", "query_suffix": "suffix after answer word; no target/distractor", "correct_full_query": "prefix + target + suffix", "distractor_full_query": "prefix + distractor + suffix", "edge_correct": "short", "edge_reversed": "short", "new_names": ["..."]}}

Allowed opposite pairs only: after/before, because/despite, more/less, gained/lost, entered/left, accepted/rejected, true/false.
Forbidden in view_structured/view_neutral/view_reversed/query_prefix/query_suffix: the exact target word and exact distractor word. Do not use labels, arrows, parentheses, lists, or quotation marks in the views. If you cannot satisfy every constraint, return {{"ok": false, "reason": "..."}}.
"""
    return {"prompt_id": f"v2_{idx:03d}_row{row.get('row_index')}", "index_hint": idx, "row_index": row.get("row_index"), "family": fam, "source_bucket": row.get("source_bucket"), "selection_score": row.get("_score"), "prompt": prompt}


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows=load_rows()
    by={}
    for r in rows:
        by.setdefault(str(r.get("family")), []).append(r)
    for xs in by.values():
        xs.sort(key=lambda r: (-int(r.get("_score",0)), int(r.get("row_index", 10**9))))
    selected=[]
    for fam in sorted(by):
        selected.extend(by[fam][:6])
    selected=selected[:30]
    prompts=[make_prompt(r,i) for i,r in enumerate(selected)]
    with PROMPTS.open("w", encoding="utf-8") as f:
        for p in prompts:
            f.write(json.dumps(p, ensure_ascii=False)+"\n")
    meta={"status":"MATCHED_PROBE_V2_PROMPTS_READY", "selected":len(prompts), "by_family":{fam:sum(1 for r in selected if r.get('family')==fam) for fam in sorted(by)}, "prompt_file":str(PROMPTS.relative_to(ROOT)), "output_file":str((_public_path('experiments/archive/frontier_consolidation/data/matched_graph_transfer_probe/matched_probe_generation_outputs_v2.jsonl')).relative_to(ROOT)), "purpose":"small construction viability test; no training"}
    META.write_text(json.dumps(meta, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")
    print(json.dumps(meta, indent=2, ensure_ascii=False), flush=True)

if __name__ == "__main__":
    main()
