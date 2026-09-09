#!/usr/bin/env python3
"""research: Extract cross-context pairs (Route 3) and procedural passages (Route 1)
from the legal 10M pool for cheap no-training screening.

Route 3: Two extraction strategies:
  A) Antonym-pair cross-context: find sentences with property P_a from pair (P_a, P_b)
     and sentences with P_b; construct crossed MLM items.
  B) Same-entity multi-property: find entities appearing with ≥2 different physical
     properties in different sentences.

Route 1: Find passages with ≥3 temporal markers AND ≥2 action verbs.

Operates on the exact legal 10M pool (SHA 215944...) using regex only. No GPU.
"""

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import json, re, hashlib, random, sys
from collections import defaultdict
from pathlib import Path

SCRIPT_DIR = _public_path('experiments/archive/representation_and_objectives/scripts')          # scripts/
WORKSPACE = _public_path('experiments/archive/representation_and_objectives')                          # 
STUDY = _public_path('experiments/archive/representation_and_objectives')                             # experiments/archive/representation_and_objectives
USER_ROOT = _public_path('.')                      # project root
CORPUS = _public_path('experiments/archive/frontier_consolidation/data/density_cleanqwen_overlay_medium_riskhard/cleanqwen_fineweb_compact_view_reinvest_10M.jsonl')
OUT = _public_path('experiments/archive/representation_and_objectives/data/orthogonal_route_extraction')
OUT.mkdir(parents=True, exist_ok=True)

# ══════════════════ ROUTE 3: Cross-context property pairs ══════════════════

# Antonym pairs for crossed-context items
ANTONYM_PAIRS = [
    ("hot", "cold"), ("warm", "cool"), ("wet", "dry"),
    ("soft", "hard"), ("open", "closed"), ("full", "empty"),
    ("broken", "intact"), ("bright", "dark"), ("clean", "dirty"),
    ("strong", "weak"), ("new", "old"), ("deep", "shallow"),
    ("thick", "thin"), ("heavy", "light"), ("large", "small"),
    ("tight", "loose"), ("solid", "liquid"), ("frozen", "melted"),
    ("flat", "sharp"), ("smooth", "rough"),
]

# All physical state words (used for same-entity extraction too)
PHYS_WORDS = set()
for a, b in ANTONYM_PAIRS:
    PHYS_WORDS.add(a)
    PHYS_WORDS.add(b)
PHYS_WORDS.update("brittle rigid stiff flexible elastic damp moist soaked boiling heated cooled warm damaged cracked shattered destroyed pure transparent opaque safe dangerous alive dead fresh stale sealed exposed".split())

# Entity-state extraction pattern (more precise: require determiner or proper position)
COPULAS = r"(?:is|was|are|were|became|becomes|turned|turns|gets|got|remains|remained|stays|stayed|feels|felt|looks|looked|seems|seemed)"
DET = r"(?:[Tt]he|[Aa]n?|[Tt]his|[Tt]hat|[Ii]ts|[Tt]heir|[Oo]ur|[Mm]y|[Hh]is|[Hh]er|[Tt]hese|[Tt]hose)"
# Pattern: [det] [noun(s)] [copula] [optional modifier] [state_adj]
PAT_ENTITY_STATE = re.compile(
    rf'\b({DET}\s+\w+(?:\s+\w+)?)\s+({COPULAS})\s+(?:(?:very|quite|extremely|rather|still|now|also|often|usually|always)\s+)?(\w+)\b',
    re.IGNORECASE
)

# Sentence splitter (robust)
SENT_SPLIT = re.compile(r'(?<=[.!?])\s+(?=[A-Z])')

# Temporal markers for Route 1
TEMPORAL_RE = re.compile(
    r'\b(first|second|third|then|next|after(?:\s+that)?|before|finally|'
    r'subsequently|later|previously|meanwhile|as\s+a\s+result|consequently|'
    r'therefore|once|until|step\s+\d+|initially|lastly|afterwards)\b',
    re.IGNORECASE
)
# Action verbs for Route 1
ACTION_RE = re.compile(
    r'\b(add|remove|place|put|take|pour|heat|cool|mix|stir|cut|break|open|close|'
    r'turn|press|push|pull|move|lift|drop|fill|empty|connect|attach|insert|'
    r'apply|wash|clean|dry|paint|build|assemble|install|measure|test|adjust|'
    r'set|start|stop|activate|separate|combine|dissolve|filter|boil|freeze|'
    r'melt|fold|wrap|unwrap|tie|untie|charge|discharge|light|extinguish)\b',
    re.IGNORECASE
)

# ══════════════════ PROCESSING ══════════════════

def process_corpus():
    # Stores for Route 3
    # state_word -> list of (entity, sentence, row_idx, match_start, match_end)
    state_sentences = defaultdict(list)
    # entity_norm -> list of (state, sentence, row_idx)
    entity_states = defaultdict(list)
    
    # Stores for Route 1
    procedural_passages = []
    
    total_es_matches = 0
    total_phys_matches = 0
    
    with open(CORPUS) as f:
        for row_idx, line in enumerate(f):
            row = json.loads(line)
            text = row["text"]
            
            # Split into sentences
            sentences = SENT_SPLIT.split(text)
            
            for sent_idx, sent in enumerate(sentences):
                sent = sent.strip()
                if len(sent) < 20 or len(sent) > 500:
                    continue
                
                # Extract entity-state patterns
                for m in PAT_ENTITY_STATE.finditer(sent):
                    entity_raw = m.group(1).strip()
                    copula = m.group(2).strip().lower()
                    state = m.group(3).strip().lower()
                    total_es_matches += 1
                    
                    if state in PHYS_WORDS:
                        total_phys_matches += 1
                        # Store for antonym-pair extraction
                        state_sentences[state].append({
                            "entity": entity_raw,
                            "sentence": sent,
                            "row_idx": row_idx,
                            "copula": copula,
                            "state_start": m.start(3),
                            "state_end": m.end(3),
                        })
                        # Store for same-entity extraction
                        entity_norm = re.sub(r'^(the|a|an|this|that|its|their|our|my|his|her)\s+', '', entity_raw.lower()).strip()
                        entity_states[entity_norm].append({
                            "state": state,
                            "sentence": sent,
                            "row_idx": row_idx,
                            "state_start": m.start(3),
                            "state_end": m.end(3),
                        })
            
            # Route 1: Check entire text for procedural structure
            temporal_hits = TEMPORAL_RE.findall(text)
            action_hits = ACTION_RE.findall(text)
            if len(temporal_hits) >= 3 and len(action_hits) >= 2:
                # Find the best procedural segment
                for sent in sentences:
                    sent = sent.strip()
                    if len(sent) < 40:
                        continue
                    t_count = len(TEMPORAL_RE.findall(sent))
                    a_count = len(ACTION_RE.findall(sent))
                    if t_count >= 2 and a_count >= 1:
                        procedural_passages.append({
                            "text": sent,
                            "row_idx": row_idx,
                            "n_temporal": t_count,
                            "n_action": a_count,
                            "temporal_markers": TEMPORAL_RE.findall(sent)[:8],
                            "action_verbs": ACTION_RE.findall(sent)[:8],
                            "words": len(sent.split()),
                        })
            
            if (row_idx + 1) % 10000 == 0:
                print(f"  Row {row_idx+1}: {total_es_matches} ES matches, {total_phys_matches} physical, {len(procedural_passages)} procedural", flush=True)
    
    print(f"\nCorpus scan complete: {row_idx+1} rows")
    print(f"Total entity-state matches: {total_es_matches}")
    print(f"Physical state matches: {total_phys_matches}")
    print(f"Procedural passages: {len(procedural_passages)}")
    
    return state_sentences, entity_states, procedural_passages


def build_antonym_pairs(state_sentences, max_per_pair=200):
    """Strategy A: antonym-pair cross-context items."""
    pairs = []
    pair_stats = {}
    
    for prop_a, prop_b in ANTONYM_PAIRS:
        sents_a = state_sentences.get(prop_a, [])
        sents_b = state_sentences.get(prop_b, [])
        
        if not sents_a or not sents_b:
            pair_stats[f"{prop_a}/{prop_b}"] = {"a": len(sents_a), "b": len(sents_b), "pairs": 0}
            continue
        
        # Sample if too many combinations
        if len(sents_a) * len(sents_b) > max_per_pair * 10:
            sents_a_sample = random.sample(sents_a, min(len(sents_a), max_per_pair))
            sents_b_sample = random.sample(sents_b, min(len(sents_b), max_per_pair))
        else:
            sents_a_sample = sents_a
            sents_b_sample = sents_b
        
        pair_count = 0
        for sa in sents_a_sample:
            for sb in sents_b_sample:
                # Must be from different rows
                if sa["row_idx"] == sb["row_idx"]:
                    continue
                # Both sentences must be reasonable length
                if len(sa["sentence"]) < 25 or len(sb["sentence"]) < 25:
                    continue
                if len(sa["sentence"]) > 400 or len(sb["sentence"]) > 400:
                    continue
                
                pairs.append({
                    "pair_type": "antonym",
                    "antonym_pair": f"{prop_a}/{prop_b}",
                    "context_a": sa["sentence"],
                    "target_a": prop_a,
                    "foil_a": prop_b,
                    "entity_a": sa["entity"],
                    "row_a": sa["row_idx"],
                    "state_start_a": sa["state_start"],
                    "state_end_a": sa["state_end"],
                    "context_b": sb["sentence"],
                    "target_b": prop_b,
                    "foil_b": prop_a,
                    "entity_b": sb["entity"],
                    "row_b": sb["row_idx"],
                    "state_start_b": sb["state_start"],
                    "state_end_b": sb["state_end"],
                })
                pair_count += 1
                if pair_count >= max_per_pair:
                    break
            if pair_count >= max_per_pair:
                break
        
        pair_stats[f"{prop_a}/{prop_b}"] = {"a": len(sents_a), "b": len(sents_b), "pairs": pair_count}
    
    return pairs, pair_stats


def build_same_entity_pairs(entity_states, max_per_entity=20):
    """Strategy B: same-entity different-state cross-context items."""
    pairs = []
    entity_stats = {}
    
    for entity, items in entity_states.items():
        if len(entity) < 2:
            continue
        # Group by state
        by_state = defaultdict(list)
        for item in items:
            by_state[item["state"]].append(item)
        
        if len(by_state) < 2:
            continue
        
        entity_stats[entity] = {s: len(v) for s, v in by_state.items()}
        
        states = list(by_state.keys())
        for i in range(len(states)):
            for j in range(i+1, len(states)):
                sa_list = by_state[states[i]]
                sb_list = by_state[states[j]]
                pair_count = 0
                for sa in sa_list:
                    for sb in sb_list:
                        if sa["row_idx"] == sb["row_idx"]:
                            continue
                        if len(sa["sentence"]) < 25 or len(sb["sentence"]) < 25:
                            continue
                        if len(sa["sentence"]) > 400 or len(sb["sentence"]) > 400:
                            continue
                        
                        pairs.append({
                            "pair_type": "same_entity",
                            "entity": entity,
                            "context_a": sa["sentence"],
                            "target_a": states[i],
                            "foil_a": states[j],
                            "row_a": sa["row_idx"],
                            "state_start_a": sa["state_start"],
                            "state_end_a": sa["state_end"],
                            "context_b": sb["sentence"],
                            "target_b": states[j],
                            "foil_b": states[i],
                            "row_b": sb["row_idx"],
                            "state_start_b": sb["state_start"],
                            "state_end_b": sb["state_end"],
                        })
                        pair_count += 1
                        if pair_count >= max_per_entity:
                            break
                    if pair_count >= max_per_entity:
                        break
    
    return pairs, entity_stats


def main():
    random.seed(194)
    print("research: Extracting route candidates from legal pool...", flush=True)
    print(f"Corpus: {CORPUS}", flush=True)
    
    state_sentences, entity_states, procedural_passages = process_corpus()
    
    # ─── Route 3A: Antonym-pair cross-context ───
    print("\n=== Route 3A: Antonym-pair cross-context ===", flush=True)
    antonym_pairs, ant_stats = build_antonym_pairs(state_sentences, max_per_pair=100)
    print(f"Total antonym pairs: {len(antonym_pairs)}")
    for k, v in sorted(ant_stats.items(), key=lambda x: -x[1]["pairs"])[:15]:
        print(f"  {k}: {v['a']}×{v['b']} → {v['pairs']} pairs")
    
    # ─── Route 3B: Same-entity multi-property ───
    print("\n=== Route 3B: Same-entity multi-property ===", flush=True)
    entity_pairs, ent_stats = build_same_entity_pairs(entity_states)
    print(f"Total same-entity pairs: {len(entity_pairs)}")
    print(f"Unique multi-state entities: {len(ent_stats)}")
    for e, states in list(sorted(ent_stats.items(), key=lambda x: -len(x[1])))[:15]:
        print(f"  \"{e}\": {states}")
    
    # ─── Route 1: Procedural passages ───
    print(f"\n=== Route 1: Procedural passages ===", flush=True)
    # Sort by quality score (temporal + action density)
    procedural_passages.sort(key=lambda x: -(x["n_temporal"] + x["n_action"]))
    top_proc = procedural_passages[:500]
    print(f"Total candidates: {len(procedural_passages)}")
    print(f"Top 500 selected (n_temporal range: {top_proc[-1]['n_temporal']}-{top_proc[0]['n_temporal']})")
    if top_proc:
        print(f"Sample top passage: {top_proc[0]['text'][:200]}")
    
    # ─── Save outputs ───
    out_r3a = _public_path('experiments/archive/representation_and_objectives/data/orthogonal_route_extraction/route3_antonym_cross_context.jsonl')
    with open(out_r3a, "w") as f:
        for p in antonym_pairs:
            f.write(json.dumps(p) + "\n")
    
    out_r3b = _public_path('experiments/archive/representation_and_objectives/data/orthogonal_route_extraction/route3_same_entity_cross_context.jsonl')
    with open(out_r3b, "w") as f:
        for p in entity_pairs:
            f.write(json.dumps(p) + "\n")
    
    out_r3_all = _public_path('experiments/archive/representation_and_objectives/data/orthogonal_route_extraction/route3_all_cross_context.jsonl')
    all_r3 = antonym_pairs + entity_pairs
    random.shuffle(all_r3)
    with open(out_r3_all, "w") as f:
        for p in all_r3:
            f.write(json.dumps(p) + "\n")
    
    out_r1 = _public_path('experiments/archive/representation_and_objectives/data/orthogonal_route_extraction/route1_procedural_passages.jsonl')
    with open(out_r1, "w") as f:
        for p in top_proc:
            f.write(json.dumps(p) + "\n")
    
    # ─── Summary JSON ───
    summary = {
        "status": "ROUTE_CANDIDATE_EXTRACTION",
        "corpus_sha256": "215944978157394dbecf2039f2c1e1806bfcbb9701421920f78605eb58975a23",
        "route3_antonym_pairs": len(antonym_pairs),
        "route3_same_entity_pairs": len(entity_pairs),
        "route3_total": len(all_r3),
        "route3_antonym_stats": ant_stats,
        "route3_unique_antonym_pairs_with_items": sum(1 for v in ant_stats.values() if v["pairs"] > 0),
        "route3_same_entity_unique_entities": len(ent_stats),
        "route1_total_candidates": len(procedural_passages),
        "route1_selected": len(top_proc),
        "files": {
            "route3_antonym": str(out_r3a),
            "route3_same_entity": str(out_r3b),
            "route3_all": str(out_r3_all),
            "route1": str(out_r1),
        },
    }
    
    out_json = _public_path('experiments/archive/representation_and_objectives/data/orthogonal_route_extraction/extraction_summary.json')
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n=== Files saved ===")
    for k, v in summary["files"].items():
        print(f"  {k}: {v}")
    print(f"  summary: {out_json}")
    print(json.dumps({"status": summary["status"], "route3_total": summary["route3_total"], 
                       "route1_selected": summary["route1_selected"]}, indent=2))


if __name__ == "__main__":
    main()
