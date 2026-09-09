#!/usr/bin/env python3
"""Step039c: Broader relation extraction from BabyLM Simple Wikipedia.

Expands beyond narrow regex to capture more relation types:
- Biographical: birthplace, birth year, death place, death year
- Role/title: leader of, member of, plays for
- Location: located in, capital of, part of
- Achievement: won, received, awarded
- Founded/established: year, founder
- Numeric: ranked, numbered, population

Uses more flexible patterns and entity filtering.
Output: augmented triple set for pair matching.
"""

import json, re, pathlib, collections

SOURCE_SENTENCES = [
    "experiments/archive/compact_experience/data/qwen_aligned/source_sentences.jsonl",
    "experiments/archive/compact_experience/data/qwen_clean_aligned/extra_source_sentences.jsonl",
]

OUT_DIR = pathlib.Path("experiments/archive/functional_learning/data/broad_extraction")
OUT_DIR.mkdir(parents=True, exist_ok=True)

PRONOUN_BLOCK = {
    "He", "She", "It", "They", "Who", "This", "That", "The", "His", "Her",
    "Its", "Their", "Some", "Many", "Most", "All", "Each", "Every", "Any",
    "Both", "One", "Two", "Three", "Four", "Five", "Several",
}

NAME_RE = r"([A-Z][a-z]+(?:(?:\s|-)[A-Z]?[a-z]+){0,3})"


def extract_all_triples(source_paths):
    """Extract triples using broader patterns."""
    triples = []
    
    for path in source_paths:
        with open(path) as f:
            for line in f:
                row = json.loads(line)
                if row["source"] != "simple_wiki":
                    continue
                text = row["text"]
                sid = row.get("id", f'ex_{row["example_id"]}')
                eid = row.get("example_id")
                
                # --- Birthplace (expanded) ---
                for m in re.finditer(
                    NAME_RE + r"\s+was\s+born\s+(?:on\s+[^.]*?\s)?in\s+" + NAME_RE,
                    text
                ):
                    ent, val = m.group(1).strip(), m.group(2).strip().rstrip(",.")
                    if ent.split()[0] in PRONOUN_BLOCK or len(val) < 3:
                        continue
                    vi = text.find(val, m.start(2))
                    if vi >= 0:
                        triples.append(make_triple(sid, eid, ent, "birthplace", val, vi, text))
                
                # --- X is from Y ---
                for m in re.finditer(
                    NAME_RE + r"\s+(?:is|was|comes?)\s+from\s+" + NAME_RE,
                    text
                ):
                    ent, val = m.group(1).strip(), m.group(2).strip().rstrip(",.")
                    if ent.split()[0] in PRONOUN_BLOCK or len(val) < 3:
                        continue
                    vi = text.find(val, m.start(2))
                    if vi >= 0:
                        triples.append(make_triple(sid, eid, ent, "from_place", val, vi, text))
                
                # --- Death place (expanded) ---
                for m in re.finditer(
                    NAME_RE + r"\s+died\s+(?:on\s+[^.]*?\s)?in\s+" + NAME_RE,
                    text
                ):
                    ent, val = m.group(1).strip(), m.group(2).strip().rstrip(",.")
                    if ent.split()[0] in PRONOUN_BLOCK or len(val) < 3:
                        continue
                    vi = text.find(val, m.start(2))
                    if vi >= 0:
                        triples.append(make_triple(sid, eid, ent, "death_place", val, vi, text))
                
                # --- Birth year ---
                for m in re.finditer(
                    NAME_RE + r"\s+was\s+born\s+(?:on\s+)?(?:[A-Z][a-z]+\s+\d{1,2},?\s+)?(\d{4})",
                    text
                ):
                    ent, val = m.group(1).strip(), m.group(2)
                    if ent.split()[0] in PRONOUN_BLOCK:
                        continue
                    vi = text.find(val, m.start(2))
                    if vi >= 0:
                        triples.append(make_triple(sid, eid, ent, "birth_year", val, vi, text))
                
                # Also: (born DATE) pattern
                for m in re.finditer(
                    r"\(born\s+(?:[A-Z][a-z]+\s+\d{1,2},?\s+)?(\d{4})\)",
                    text
                ):
                    # Find preceding entity name
                    before = text[:m.start()].strip()
                    ent_m = re.search(NAME_RE + r"\s*$", before)
                    if ent_m:
                        ent = ent_m.group(1).strip()
                        val = m.group(1)
                        if ent.split()[0] not in PRONOUN_BLOCK:
                            vi = text.find(val, m.start(1))
                            if vi >= 0:
                                triples.append(make_triple(sid, eid, ent, "birth_year", val, vi, text))
                
                # --- Located in (expanded) ---
                for m in re.finditer(
                    NAME_RE + r"\s+is\s+(?:a|an|the)\s+(?:\w+\s+)?(?:city|town|village|district|municipality|province|region|state|county|island|area|borough|suburb|commune)\s+(?:in|of)\s+(?:the\s+)?" + NAME_RE,
                    text
                ):
                    ent, val = m.group(1).strip(), m.group(2).strip()
                    if ent.split()[0] in PRONOUN_BLOCK:
                        continue
                    vi = text.find(val, m.start(2))
                    if vi >= 0:
                        triples.append(make_triple(sid, eid, ent, "located_in", val, vi, text))
                
                # --- Capital of ---
                for m in re.finditer(
                    NAME_RE + r"\s+is\s+the\s+capital\s+(?:city\s+)?of\s+" + NAME_RE,
                    text
                ):
                    val, ent = m.group(1).strip(), m.group(2).strip()
                    vi = text.find(val, m.start(1))
                    if vi >= 0:
                        triples.append(make_triple(sid, eid, ent, "capital", val, vi, text))
                
                # --- Founded year (expanded) ---
                for m in re.finditer(
                    NAME_RE + r"\s+was\s+(?:founded|established|created|formed|started|opened)\s+(?:in|on)\s+(?:[A-Z][a-z]+\s+\d{1,2},?\s+)?(\d{4})",
                    text
                ):
                    ent, val = m.group(1).strip(), m.group(2)
                    if ent.split()[0] in PRONOUN_BLOCK:
                        continue
                    vi = text.find(val, m.start(2))
                    if vi >= 0:
                        triples.append(make_triple(sid, eid, ent, "founded_year", val, vi, text))
                
                # --- Plays/played for ---
                for m in re.finditer(
                    NAME_RE + r"\s+(?:plays|played|plays?|competed)\s+for\s+" + NAME_RE,
                    text
                ):
                    ent, val = m.group(1).strip(), m.group(2).strip()
                    if ent.split()[0] in PRONOUN_BLOCK:
                        continue
                    vi = text.find(val, m.start(2))
                    if vi >= 0:
                        triples.append(make_triple(sid, eid, ent, "plays_for", val, vi, text))
                
                # --- Leader/president/prime minister of ---
                for m in re.finditer(
                    NAME_RE + r"\s+(?:is|was|became)\s+(?:the\s+)?(?:president|prime minister|king|queen|emperor|governor|mayor|chancellor|leader|chairman|director|head|chief)\s+of\s+(?:the\s+)?" + NAME_RE,
                    text
                ):
                    ent, val = m.group(1).strip(), m.group(2).strip()
                    if ent.split()[0] in PRONOUN_BLOCK:
                        continue
                    vi = text.find(val, m.start(2))
                    if vi >= 0:
                        triples.append(make_triple(sid, eid, ent, "leader_of", val, vi, text))
                
                # Also: "X of Y" role patterns where X is a role
                for m in re.finditer(
                    r"(?:president|prime minister|king|queen|governor|mayor|chairman)\s+of\s+(?:the\s+)?" + NAME_RE + r"\s*,?\s+" + NAME_RE,
                    text, re.I
                ):
                    val, ent = m.group(1).strip(), m.group(2).strip()
                    if ent.split()[0] in PRONOUN_BLOCK:
                        continue
                    vi = text.find(ent, m.start(2))
                    if vi >= 0:
                        triples.append(make_triple(sid, eid, val, "has_leader", ent, vi, text))
                
                # --- Nationality + occupation ---
                for m in re.finditer(
                    NAME_RE + r"\s+(?:is|was)\s+(?:a|an)\s+([A-Z][a-z]+)\s+(politician|singer|actor|actress|writer|author|painter|composer|scientist|mathematician|physicist|footballer|player|musician|director|poet|artist|architect|engineer|lawyer|doctor|philosopher|historian|journalist|economist|general|admiral|pilot|astronaut|athlete)",
                    text
                ):
                    ent = m.group(1).strip()
                    nationality = m.group(2)
                    occupation = m.group(3)
                    if ent.split()[0] in PRONOUN_BLOCK:
                        continue
                    vi = text.find(nationality, m.start(2))
                    if vi >= 0:
                        triples.append(make_triple(sid, eid, ent, "nationality", nationality, vi, text))
                    oi = text.find(occupation, m.start(3))
                    if oi >= 0:
                        triples.append(make_triple(sid, eid, ent, "occupation", occupation, oi, text))
    
    return triples


def make_triple(sid, eid, entity, relation, value, vi, text):
    return {
        "sentence_id": sid, "example_id": eid,
        "entity": entity, "relation": relation, "value": value,
        "value_span": [vi, vi + len(value)],
        "sentence": text, "source_type": "simple_wiki",
        "grounding": "raw_source_exact_span",
    }


def main():
    triples = extract_all_triples(SOURCE_SENTENCES)
    
    # Deduplicate: same entity + relation + value
    seen = set()
    unique = []
    for t in triples:
        key = (t["entity"].lower(), t["relation"], t["value"].lower())
        if key not in seen:
            seen.add(key)
            unique.append(t)
    
    by_rel = collections.Counter(t["relation"] for t in unique)
    print(f"Total unique triples: {len(unique)}")
    print("By relation:")
    for rel, count in sorted(by_rel.items(), key=lambda x: -x[1]):
        print(f"  {rel}: {count}")
    
    # Save
    with open(OUT_DIR / "broad_triples.jsonl", "w") as f:
        for t in unique:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")
    
    # Show examples per type
    for rel in sorted(by_rel.keys()):
        examples = [t for t in unique if t["relation"] == rel][:3]
        print(f"\n=== {rel} ===")
        for ex in examples:
            print(f"  {ex['entity']} → {ex['value']} ({ex['sentence_id']})")
            print(f"    \"{ex['sentence'][:120]}\"")
    
    print(f"\nSaved to {OUT_DIR / 'broad_triples.jsonl'}")
    
    # Quick pair count estimate
    for rel in sorted(by_rel.keys()):
        group = [t for t in unique if t["relation"] == rel]
        # Count unique entities
        ents = set(t["entity"].lower() for t in group)
        vals = set(t["value"].lower() for t in group)
        print(f"\n{rel}: {len(group)} triples, {len(ents)} unique entities, {len(vals)} unique values")


if __name__ == "__main__":
    main()
