#!/usr/bin/env python3
"""research: Route B directed-event yield measurement.

Purpose: Measure how many attested directed-event pairs with reversed arguments
exist in the FineWeb source corpus, as Gate 1 for the interaction-isolating
training signal proposed by independent review research generator memo.

Gate 1 threshold (independent review memo): Route B yield "must be measured directly.
If yield is small, stop — regardless of quality."

The miner:
1. Extracts entities and directed-event structures from each FineWeb source sentence
2. Builds a global entity-pair index to find reversed-argument pairs
3. Also mines within-document reversed-role structures
4. For source-compact pairs, checks direction preservation
5. Reports yield by relation family

Also mines Route A comparative-frame candidates: sentences containing
asymmetric entity pairs suitable for synthesized comparative frames.
"""
import json, re, pathlib, collections, sys, hashlib

SOURCES = pathlib.Path("experiments/archive/representation_and_objectives/data/fw_mechanism_source_selection/fw_mechanism_frozen_sources.jsonl")
OUT_DIR = pathlib.Path("experiments/archive/representation_and_objectives/data/route_b_yield")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Stop words for entity extraction ──
STOP_CAPS = frozenset([
    "The","This","That","These","Those","However","Although","While","When",
    "Where","Because","Since","After","Before","Also","Furthermore","Moreover",
    "Nevertheless","Therefore","Meanwhile","Additionally","Similarly","But",
    "And","Or","Yet","So","If","Then","There","It","He","She","They","We",
    "Our","His","Her","Its","Many","Some","Most","Several","Few","All",
    "Each","Every","Other","Another","Such","Much","More","Less","Both",
    "One","Two","Three","Four","Five","First","Second","Third","Fourth",
    "No","Not","New","Old","Very","People","Children","May","According",
    "January","February","March","April","June","July","August","September",
    "October","November","December","During","Between","Within","Without",
    "About","Under","Over","Around","Through","Across","Along","Against",
    "Among","Throughout","Beyond","Despite","Until","Upon","Toward",
    "For","From","Into","Onto","To","In","On","At","By","With","Of",
])

# ── Directed relation families ──
COMPARATIVE_KW = [
    "bigger","smaller","taller","shorter","heavier","lighter","faster","slower",
    "stronger","weaker","hotter","colder","older","younger","longer","wider",
    "deeper","higher","lower","larger","thicker","thinner","richer","poorer",
    "warmer","cooler","louder","quieter","brighter","darker","denser","softer",
    "harder","wetter","drier","greater","lesser","superior","inferior",
    "more","less","better","worse","outperform","surpass","exceed","outweigh",
]
CAUSAL_KW = [
    "caused","causes","causing","cause","ledto","resultsin","resultedin",
    "producedby","triggeredby","created","produces","generates","triggered",
    "induces","induced","provoked","stimulated","brought",
]
TEMPORAL_KW = [
    "before","after","preceded","preceded by","followed","following",
    "prior to","subsequent to","then","until","since",
]
SPATIAL_KW = [
    "north of","south of","east of","west of","above","below","left of",
    "right of","upstream","downstream","uphill","downhill","beyond",
    "inside","outside","beneath","atop",
]
TRANSITIVE_VERBS = [
    "defeated","attacked","conquered","invaded","destroyed","taught","trained",
    "employed","hired","fired","married","killed","founded","discovered",
    "invented","built","designed","wrote","composed","directed","produced",
    "managed","led","governed","ruled","commanded","arrested","kidnapped",
    "rescued","saved","fought","beat","pushed","pulled","threw","broke",
    "bought","sold","gave","sent","showed","told","asked","paid","punished",
    "rewarded","praised","blamed","criticized","influenced","inspired",
    "defeated","captured","liberated","annexed","colonized","occupied",
    "bombed","shelled","besieged","overthrew","assassinated","succeeded",
    "appointed","dismissed","elected","nominated","promoted","demoted",
    "mentored","coached","tutored","supervised","oversaw","chaired",
    "supported","opposed","endorsed","rejected","accepted","denied",
    "acquired","purchased","inherited","donated","granted","awarded",
]

def extract_entities(text):
    """Extract capitalized entity-like spans from text.
    Returns list of (entity_text, char_start)."""
    ents = []
    # First, find multi-word capitalized sequences (2-4 words)
    for m in re.finditer(r'\b([A-Z][a-z]+(?:\s+(?:of|the|and|de|von|van|el|al|la|le)\s+)?[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})\b', text):
        ent = m.group(1)
        if ent.split()[0] not in STOP_CAPS:
            ents.append((ent, m.start()))
    # Then find single capitalized words NOT already covered
    covered = set()
    for e, s in ents:
        for i in range(s, s + len(e)):
            covered.add(i)
    for m in re.finditer(r'\b([A-Z][a-z]{2,})\b', text):
        if m.start() not in covered and m.group(1) not in STOP_CAPS:
            ents.append((m.group(1), m.start()))
    return ents

def classify_directed_relation(text_lower):
    """Classify sentence into directed-relation families. Returns set of families."""
    families = set()
    # Comparative
    for kw in COMPARATIVE_KW:
        if kw in text_lower:
            if "than" in text_lower or kw in ("outperform","surpass","exceed","outweigh"):
                families.add("comparative")
                break
    # Causal
    for kw in CAUSAL_KW:
        if kw in text_lower.replace(" ", ""):
            families.add("causal")
            break
    # Temporal
    for kw in TEMPORAL_KW:
        if kw in text_lower:
            families.add("temporal")
            break
    # Spatial  
    for kw in SPATIAL_KW:
        if kw in text_lower:
            families.add("spatial")
            break
    return families

def find_transitive_events(text, entities):
    """Find (agent, verb, patient) triples from entities in text.
    Uses simple heuristic: entity before verb = agent, after = patient."""
    events = []
    text_lower = text.lower()
    for verb in TRANSITIVE_VERBS:
        for m in re.finditer(r'\b' + re.escape(verb) + r'\b', text_lower):
            verb_pos = m.start()
            # Find closest entity before verb (agent) and after verb (patient)
            agents = [(e, s) for e, s in entities if s < verb_pos]
            patients = [(e, s) for e, s in entities if s > verb_pos + len(verb)]
            if agents and patients:
                agent = max(agents, key=lambda x: x[1])[0]  # closest before
                patient = min(patients, key=lambda x: x[1])[0]  # closest after
                if agent != patient:
                    events.append((agent, verb, patient, "agent_patient"))
    return events

def main():
    print("Loading FineWeb sources...")
    sources = []
    with open(SOURCES) as f:
        for line in f:
            sources.append(json.loads(line))
    print(f"  Loaded {len(sources)} sources")
    
    # ── Phase 1: Extract directed events from all sources ──
    print("\nPhase 1: Extracting directed events...")
    directed_sentences = []  # (idx, text, entities, families, events)
    entity_pair_index = collections.defaultdict(list)  # (sorted_pair) -> [(idx, direction)]
    doc_sentences = collections.defaultdict(list)  # doc_id -> [idx]
    
    family_counts = collections.Counter()
    total_events = 0
    has_entities_2plus = 0
    
    for idx, src in enumerate(sources):
        text = src["text"]
        text_lower = text.lower()
        entities = extract_entities(text)
        
        if len(entities) >= 2:
            has_entities_2plus += 1
        
        # Classify directed relations
        families = classify_directed_relation(text_lower)
        
        # Find transitive events
        events = find_transitive_events(text, entities)
        total_events += len(events)
        
        for fam in families:
            family_counts[fam] += 1
        
        if families or events:
            directed_sentences.append({
                "idx": idx,
                "text": text,
                "doc_id": src.get("doc_id", ""),
                "entities": [e[0] for e in entities],
                "families": list(families),
                "events": [(a, v, p, t) for a, v, p, t in events],
                "has_rewrite": src.get("has_rewrite", False),
                "rewrite_text": src.get("rewrite_text", ""),
            })
        
        # Index entity pairs from transitive events
        for agent, verb, patient, _ in events:
            pair_key = tuple(sorted([agent.lower(), patient.lower()]))
            direction = "forward" if agent.lower() == pair_key[0] else "reverse"
            entity_pair_index[pair_key].append({
                "idx": idx, "direction": direction, "verb": verb,
                "agent": agent, "patient": patient,
                "doc_id": src.get("doc_id", ""),
            })
        
        # Index entity co-occurrences for role analysis
        if len(entities) >= 2:
            doc_id = src.get("doc_id", "")
            if doc_id:
                doc_sentences[doc_id].append(idx)
            # Index all entity pairs by position (first entity = potential agent)
            for i, (e1, s1) in enumerate(entities):
                for j, (e2, s2) in enumerate(entities):
                    if i != j and e1.lower() != e2.lower():
                        pair_key = tuple(sorted([e1.lower(), e2.lower()]))
                        direction = "forward" if e1.lower() == pair_key[0] else "reverse"
                        # Record positional direction (first-mention = potential subject)
                        entity_pair_index[pair_key].append({
                            "idx": idx, "direction": f"positional_{direction}",
                            "verb": "co-occur", "agent": e1, "patient": e2,
                            "doc_id": src.get("doc_id", ""),
                        })
    
    print(f"  Sentences with ≥2 entities: {has_entities_2plus}")
    print(f"  Sentences with directed families: {sum(family_counts.values())}")
    print(f"  Family counts: {dict(family_counts)}")
    print(f"  Transitive events found: {total_events}")
    print(f"  Unique entity pairs in index: {len(entity_pair_index)}")
    
    # ── Phase 2: Find reversed-argument pairs ──
    print("\nPhase 2: Finding reversed-argument pairs...")
    
    # 2a: From transitive events — same verb, reversed agent-patient
    verb_reversed = []
    for pair_key, entries in entity_pair_index.items():
        verb_entries = [e for e in entries if e["verb"] != "co-occur"]
        if len(verb_entries) < 2:
            continue
        # Group by verb
        by_verb = collections.defaultdict(list)
        for e in verb_entries:
            by_verb[e["verb"]].append(e)
        for verb, vents in by_verb.items():
            dirs = set(e["direction"] for e in vents)
            if "forward" in dirs and "reverse" in dirs:
                fwd = [e for e in vents if e["direction"] == "forward"]
                rev = [e for e in vents if e["direction"] == "reverse"]
                verb_reversed.append({
                    "pair": pair_key, "verb": verb,
                    "forward_count": len(fwd), "reverse_count": len(rev),
                    "forward_docs": list(set(e["doc_id"] for e in fwd)),
                    "reverse_docs": list(set(e["doc_id"] for e in rev)),
                    "forward_example": sources[fwd[0]["idx"]]["text"],
                    "reverse_example": sources[rev[0]["idx"]]["text"],
                })
    
    # 2b: Same entity pair, any transitive verb, reversed direction
    any_verb_reversed = []
    for pair_key, entries in entity_pair_index.items():
        verb_entries = [e for e in entries if e["verb"] != "co-occur"]
        if len(verb_entries) < 2:
            continue
        dirs = set(e["direction"] for e in verb_entries)
        if "forward" in dirs and "reverse" in dirs:
            fwd = [e for e in verb_entries if e["direction"] == "forward"]
            rev = [e for e in verb_entries if e["direction"] == "reverse"]
            any_verb_reversed.append({
                "pair": pair_key,
                "forward_verbs": list(set(e["verb"] for e in fwd)),
                "reverse_verbs": list(set(e["verb"] for e in rev)),
                "forward_count": len(fwd),
                "reverse_count": len(rev),
                "cross_doc": bool(set(e["doc_id"] for e in fwd) != set(e["doc_id"] for e in rev)),
            })
    
    # 2c: Positional role reversals — same pair, one as first-mention and one as second
    positional_reversed = 0
    positional_same_doc_reversed = 0
    for pair_key, entries in entity_pair_index.items():
        pos_entries = [e for e in entries if e["verb"] == "co-occur"]
        if len(pos_entries) < 2:
            continue
        dirs = set(e["direction"] for e in pos_entries)
        if "positional_forward" in dirs and "positional_reverse" in dirs:
            positional_reversed += 1
            # Check if any reversal is within the same document
            fwd_docs = set(e["doc_id"] for e in pos_entries if e["direction"] == "positional_forward")
            rev_docs = set(e["doc_id"] for e in pos_entries if e["direction"] == "positional_reverse")
            if fwd_docs & rev_docs:
                positional_same_doc_reversed += 1
    
    print(f"\n  Verb-specific reversed pairs: {len(verb_reversed)}")
    print(f"  Any-verb reversed pairs: {len(any_verb_reversed)}")
    print(f"  Positional role reversals (entity pairs): {positional_reversed}")
    print(f"  Positional role reversals (same doc): {positional_same_doc_reversed}")
    
    # ── Phase 3: Route A — comparative frame candidates ──
    print("\nPhase 3: Counting Route A comparative-frame candidates...")
    comparative_sentences = [s for s in directed_sentences if "comparative" in s["families"]]
    comparative_with_entities = [s for s in comparative_sentences if len(s["entities"]) >= 2]
    
    print(f"  Comparative sentences: {len(comparative_sentences)}")
    print(f"  Comparative with ≥2 entities: {len(comparative_with_entities)}")
    
    # ── Phase 4: Source-compact direction preservation ──
    print("\nPhase 4: Source-compact direction preservation...")
    directed_with_rewrite = [s for s in directed_sentences if s["has_rewrite"] and s["rewrite_text"]]
    preserved = 0
    checked = 0
    direction_samples = []
    
    for s in directed_with_rewrite[:500]:  # Sample up to 500
        src_lower = s["text"].lower()
        rw_lower = s["rewrite_text"].lower()
        
        # Check if the same direction markers appear in both
        src_families = classify_directed_relation(src_lower)
        rw_families = classify_directed_relation(rw_lower)
        
        if src_families:
            checked += 1
            if src_families & rw_families:  # At least one shared family
                preserved += 1
            direction_samples.append({
                "source": s["text"][:120],
                "rewrite": s["rewrite_text"][:120],
                "src_families": list(src_families),
                "rw_families": list(rw_families),
                "preserved": bool(src_families & rw_families),
            })
    
    preservation_rate = preserved / checked if checked else 0
    print(f"  Directed sentences with rewrites: {len(directed_with_rewrite)}")
    print(f"  Checked (sampled): {checked}")
    print(f"  Direction preserved: {preserved} ({preservation_rate:.3f})")
    
    # ── Phase 5: Within-document entity role analysis ──
    print("\nPhase 5: Within-document directed-event pairs...")
    within_doc_pairs = 0
    within_doc_examples = []
    for doc_id, indices in doc_sentences.items():
        if len(indices) < 2:
            continue
        # Find directed events in this document
        doc_events = []
        for idx in indices:
            src = sources[idx]
            ents = extract_entities(src["text"])
            evts = find_transitive_events(src["text"], ents)
            for a, v, p, t in evts:
                doc_events.append({"idx": idx, "agent": a, "patient": p, "verb": v})
        
        # Find reversed pairs within document
        for i, e1 in enumerate(doc_events):
            for j, e2 in enumerate(doc_events):
                if i >= j:
                    continue
                if (e1["agent"].lower() == e2["patient"].lower() and 
                    e1["patient"].lower() == e2["agent"].lower()):
                    within_doc_pairs += 1
                    if len(within_doc_examples) < 20:
                        within_doc_examples.append({
                            "doc_id": doc_id,
                            "sent1": sources[e1["idx"]]["text"][:150],
                            "event1": f"{e1['agent']} {e1['verb']} {e1['patient']}",
                            "sent2": sources[e2["idx"]]["text"][:150],
                            "event2": f"{e2['agent']} {e2['verb']} {e2['patient']}",
                        })
    
    print(f"  Within-document reversed event pairs: {within_doc_pairs}")
    
    # ── Yield summary ──
    print("\n" + "="*60)
    print("ROUTE B YIELD SUMMARY")
    print("="*60)
    total_reversed = len(any_verb_reversed)
    print(f"Total FineWeb sources: {len(sources)}")
    print(f"Sources with ≥2 entities: {has_entities_2plus}")
    print(f"Sources with directed families: {sum(family_counts.values())}")
    print(f"Transitive events extracted: {total_events}")
    print(f"")
    print(f"Reversed-argument pairs (same verb): {len(verb_reversed)}")
    print(f"Reversed-argument pairs (any verb): {total_reversed}")
    print(f"Within-document reversed pairs: {within_doc_pairs}")
    print(f"Positional role reversals: {positional_reversed}")
    print(f"  (same doc): {positional_same_doc_reversed}")
    print(f"")
    print(f"Comparative sentences (Route A input): {len(comparative_sentences)}")
    print(f"  with ≥2 entities: {len(comparative_with_entities)}")
    print(f"Direction preservation rate: {preservation_rate:.3f}")
    print(f"")
    
    # Gate 1 assessment
    route_b_yield = total_reversed + within_doc_pairs
    if route_b_yield < 50:
        gate1 = "CLOSE: Route B yield is critically low (<50 reversed pairs)"
    elif route_b_yield < 500:
        gate1 = "MARGINAL: Route B yield is low (<500), may be insufficient for pretraining"
    elif route_b_yield < 5000:
        gate1 = "MODERATE: Route B has moderate yield, proceed to Gate 2 cautiously"
    else:
        gate1 = "OPEN: Route B has substantial yield, proceed to Gate 2"
    print(f"Gate 1 assessment: {gate1}")
    
    # Save results
    result = {
        "status": "ROUTE_B_YIELD",
        "n_sources": len(sources),
        "n_entities_2plus": has_entities_2plus,
        "n_directed_family_sentences": sum(family_counts.values()),
        "family_counts": dict(family_counts),
        "n_transitive_events": total_events,
        "n_unique_entity_pairs": len(entity_pair_index),
        "verb_reversed_pairs": len(verb_reversed),
        "any_verb_reversed_pairs": total_reversed,
        "within_doc_reversed_pairs": within_doc_pairs,
        "positional_reversed": positional_reversed,
        "positional_same_doc_reversed": positional_same_doc_reversed,
        "comparative_sentences": len(comparative_sentences),
        "comparative_with_entities": len(comparative_with_entities),
        "direction_preservation": {
            "checked": checked,
            "preserved": preserved,
            "rate": preservation_rate,
        },
        "gate1_route_b_yield": route_b_yield,
        "gate1_assessment": gate1,
        "verb_reversed_examples": verb_reversed[:20],
        "within_doc_examples": within_doc_examples[:20],
        "direction_preservation_samples": direction_samples[:20],
    }
    
    out_json = OUT_DIR / "route_b_yield.json"
    with open(out_json, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False, default=str)
    
    note_path = pathlib.Path("research/notes/representation_and_objectives/route_b_directed_event_yield.md")
    note_path.parent.mkdir(parents=True, exist_ok=True)
    with open(note_path, "w") as f:
        f.write(f"# research — Route B directed-event yield\n\n")
        f.write(f"Gate 1 measurement for interaction-isolating training signal.\n\n")
        f.write(f"- FineWeb sources: {len(sources)}\n")
        f.write(f"- Transitive events: {total_events}\n")
        f.write(f"- Verb-reversed pairs: {len(verb_reversed)}\n")
        f.write(f"- Any-verb reversed pairs: {total_reversed}\n")
        f.write(f"- Within-doc reversed pairs: {within_doc_pairs}\n")
        f.write(f"- Positional reversals: {positional_reversed} ({positional_same_doc_reversed} same-doc)\n")
        f.write(f"- **Gate 1 yield: {route_b_yield}** — {gate1}\n")
        f.write(f"- Direction preservation rate: {preservation_rate:.3f}\n\n")
        f.write(f"Artifacts:\n- `{out_json}`\n")
    
    print(f"\nSaved: {out_json}")
    print(f"Note: {note_path}")
    return result

if __name__ == "__main__":
    main()
