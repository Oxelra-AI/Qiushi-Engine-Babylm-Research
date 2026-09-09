#!/usr/bin/env python3
"""research: Micro-world feasibility extraction from the legal 10M pool.

CPU-only. Scans for sentences with explicit relational patterns that can
be transformed into crossed-sign binding frames.

Reports frame counts by type/subtype and sample frames.
"""
import json, re, sys, pathlib, collections
from typing import Any

POOL = pathlib.Path("experiments/archive/frontier_consolidation/data/density_cleanqwen_overlay_medium_riskhard/cleanqwen_fineweb_compact_view_reinvest_10M.jsonl")
OUT_DIR = pathlib.Path("experiments/archive/representation_and_objectives/data/micro_world_feasibility")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# --- Pattern definitions ---

# Type A: Direct spatial relation
SPATIAL_ANTONYMS = {
    "in": "outside", "inside": "outside", "on": "under",
    "above": "below", "behind": "in front of", "near": "far from",
    "under": "on", "below": "above", "outside": "inside",
    "before": "after", "after": "before",
}
# Simpler bidirectional pairs (both directions extractable)
SPATIAL_PAIRS = [
    ("in", "outside"), ("inside", "outside"), ("on", "under"),
    ("above", "below"), ("near", "far from"),
]

# Pattern: entity_phrase is/was RELATION entity_phrase
# We match "is/was/are/were RELATION" 
SPATIAL_RE = re.compile(
    r'\b(?:is|was|are|were)\s+'
    r'(in|inside|on|under|above|below|behind|outside|near|before|after)\s+'
    r'(?:the\s+|a\s+|an\s+)?(\w[\w\s]{0,30}?)(?:\.|,|;|\s+and\s+|\s+or\s+|\s+but\s+)',
    re.IGNORECASE
)

# Type B: Argument-role binding (transfer verbs)
TRANSFER_RE = re.compile(
    r'\b([A-Z]\w+)\s+(?:gave|sent|handed|passed|showed|taught|told|sold|offered)\s+'
    r'(?:the\s+|a\s+|an\s+)?(\w[\w\s]{0,20}?)\s+to\s+([A-Z]\w+)',
    re.IGNORECASE
)

# Type C: Belief attribution 
BELIEF_POSITIVE = {"believes", "thinks", "knows", "expects", "hopes", "assumes", "feels"}
BELIEF_NEGATIVE = {"doubts", "denies", "fears", "suspects"}
BELIEF_RE = re.compile(
    r'\b([A-Z]\w+)\s+(believes|thinks|knows|doubts|denies|expects|hopes|assumes|fears|suspects)\s+that\s+',
    re.IGNORECASE
)

# Type D: State-change verbs
STATE_PAIRS = [
    ("opened", "closed"), ("locked", "unlocked"), ("turned on", "turned off"),
    ("added", "removed"), ("attached", "detached"),
]
STATE_VERBS = set()
for a, b in STATE_PAIRS:
    STATE_VERBS.add(a.lower()); STATE_VERBS.add(b.lower())

STATE_RE = re.compile(
    r'\b([A-Z]\w+)\s+(opened|closed|locked|unlocked|added|removed|attached|detached)\s+'
    r'(?:the\s+|a\s+|an\s+)?(\w[\w\s]{0,20}?)(?:\.|,|;)',
    re.IGNORECASE
)

# --- Scan ---
def split_sentences(text: str) -> list[str]:
    """Rough sentence splitter."""
    parts = re.split(r'(?<=[.!?])\s+', text)
    return [p.strip() for p in parts if len(p.strip()) > 10]

def scan_spatial(sent: str) -> list[dict]:
    frames = []
    for m in SPATIAL_RE.finditer(sent):
        rel = m.group(1).lower()
        ant = SPATIAL_ANTONYMS.get(rel)
        if ant:
            frames.append({
                "type": "A_spatial", "subtype": f"{rel}_vs_{ant}",
                "relation": rel, "antonym": ant,
                "sentence": sent, "match": m.group(0)
            })
    return frames

def scan_transfer(sent: str) -> list[dict]:
    frames = []
    for m in TRANSFER_RE.finditer(sent):
        agent, obj, patient = m.group(1), m.group(2).strip(), m.group(3)
        if agent.lower() != patient.lower() and len(agent) > 1 and len(patient) > 1:
            frames.append({
                "type": "B_transfer", "subtype": "possession_swap",
                "agent": agent, "patient": patient, "object": obj,
                "sentence": sent, "match": m.group(0)
            })
    return frames

def scan_belief(sent: str) -> list[dict]:
    frames = []
    for m in BELIEF_RE.finditer(sent):
        verb = m.group(2).lower()
        if verb in BELIEF_POSITIVE:
            antonym_set = BELIEF_NEGATIVE
            polarity = "positive"
        else:
            antonym_set = BELIEF_POSITIVE
            polarity = "negative"
        frames.append({
            "type": "C_belief", "subtype": f"belief_{polarity}",
            "subject": m.group(1), "verb": verb,
            "antonym_candidates": sorted(antonym_set),
            "sentence": sent, "match": m.group(0),
        })
    return frames

def scan_state_change(sent: str) -> list[dict]:
    frames = []
    for m in STATE_RE.finditer(sent):
        verb = m.group(2).lower()
        # find antonym
        ant = None
        for a, b in STATE_PAIRS:
            if verb == a.lower(): ant = b; break
            if verb == b.lower(): ant = a; break
        if ant:
            frames.append({
                "type": "D_state_change", "subtype": f"{verb}_vs_{ant}",
                "agent": m.group(1), "verb": verb, "antonym": ant,
                "object": m.group(3).strip(),
                "sentence": sent, "match": m.group(0),
            })
    return frames

def main():
    if not POOL.exists():
        print(json.dumps({"error": f"Pool not found: {POOL}"})); sys.exit(1)
    
    counts = collections.Counter()
    subtype_counts = collections.Counter()
    samples = collections.defaultdict(list)
    MAX_SAMPLES = 8
    total_sentences = 0
    total_rows = 0
    
    with open(POOL, "r") as f:
        for line_no, line in enumerate(f):
            row = json.loads(line)
            text = row.get("text", "")
            total_rows += 1
            sents = split_sentences(text)
            total_sentences += len(sents)
            
            for sent in sents:
                for scanner in [scan_spatial, scan_transfer, scan_belief, scan_state_change]:
                    for frame in scanner(sent):
                        ftype = frame["type"]
                        subtype = frame["subtype"]
                        counts[ftype] += 1
                        subtype_counts[f"{ftype}/{subtype}"] += 1
                        if len(samples[ftype]) < MAX_SAMPLES:
                            samples[ftype].append(frame)
    
    result = {
        "status": "PASS",
        "pool_rows": total_rows,
        "total_sentences": total_sentences,
        "frame_counts_by_type": dict(counts.most_common()),
        "frame_counts_by_subtype": dict(subtype_counts.most_common(40)),
        "samples": {k: v for k, v in samples.items()},
    }
    
    out_json = OUT_DIR / "micro_world_feasibility.json"
    out_json.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    
    # Markdown summary
    lines = ["# research micro-world feasibility extraction\n"]
    lines.append(f"Pool: {POOL}")
    lines.append(f"Rows scanned: {total_rows}")
    lines.append(f"Sentences scanned: {total_sentences}\n")
    lines.append("## Frame counts by type\n")
    lines.append("| Type | Count |")
    lines.append("|---|---:|")
    for ftype, cnt in counts.most_common():
        lines.append(f"| {ftype} | {cnt} |")
    lines.append("\n## Top subtypes\n")
    lines.append("| Subtype | Count |")
    lines.append("|---|---:|")
    for sub, cnt in subtype_counts.most_common(20):
        lines.append(f"| {sub} | {cnt} |")
    lines.append("\n## Samples\n")
    for ftype in sorted(samples.keys()):
        lines.append(f"### {ftype}\n")
        for s in samples[ftype][:4]:
            lines.append(f"- **{s.get('subtype','')}**: `{s.get('match','')}`")
            lines.append(f"  Sentence: {s['sentence'][:200]}")
            lines.append("")
    
    out_md = (OUT_DIR.parents[4] / 'research/documents/representation_and_objectives/data/micro_world_feasibility/micro_world_feasibility.md')
    out_md.write_text("\n".join(lines) + "\n")
    
    print(json.dumps({
        "status": "PASS",
        "out_json": str(out_json),
        "out_md": str(out_md),
        "frame_counts": dict(counts.most_common()),
        "total_rows": total_rows,
        "total_sentences": total_sentences,
    }))

if __name__ == "__main__":
    main()
