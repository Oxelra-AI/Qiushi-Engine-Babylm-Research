#!/usr/bin/env python3
"""research: Paired contrastive entity-binding packet generator and validator.

Scientific purpose
------------------
Build natural-language experience where only entity-specific update binding, not
source retention or recency copying, solves both UPDATE and RETAIN targets.

Scientific design requirements:
1. Each pair shares the SAME source sentence and SAME use-sentence frame
2. UPDATE row: target entity gets new state → use sentence answer is NEW state
3. RETAIN row: distractor entity gets new state → use sentence answer is SOURCE state
4. Answer and foil spans are explicitly located with token positions
5. No temporal cue words (now/still/remain/current/new/recently)
6. The use-sentence frame is IDENTICAL between paired rows
7. Held-out splits by source/entity combination

The generator outputs paired JSONL with full_text, explicit span annotations, and
metadata for downstream MLM mask analysis and training corpus materialization.

Usage
-----
  python paired_packet_generator.py \\
    --mode template-pilot \\
    --out-dir experiments/archive/functional_learning/data/paired_packets

  python paired_packet_generator.py \\
    --mode from-sources \\
    --source-jsonl <source_file> \\
    --out-dir experiments/archive/functional_learning/data/paired_packets
"""
import argparse, json, pathlib, sys, re, hashlib
from typing import Dict, List, Tuple, Optional, Any

# Cue words that must NOT appear in use sentences
FORBIDDEN_CUE_WORDS = {
    "now", "still", "remain", "remains", "remaining", "remained",
    "current", "currently", "recent", "recently", "new", "newly",
    "old", "previous", "previously", "former", "formerly",
    "original", "originally", "unchanged", "changed", "updated",
}

def validate_no_cue_words(text: str) -> Tuple[bool, List[str]]:
    """Check that use sentence contains no temporal/persistence cue words."""
    words = set(re.findall(r'\b\w+\b', text.lower()))
    violations = words & FORBIDDEN_CUE_WORDS
    return len(violations) == 0, list(violations)


def find_text_spans(full_text: str, target: str) -> List[Tuple[int, int]]:
    """Find all character-level (start, end) spans of target in full_text."""
    spans = []
    start = 0
    while True:
        idx = full_text.find(target, start)
        if idx < 0:
            break
        spans.append((idx, idx + len(target)))
        start = idx + 1
    return spans


def make_paired_packet(
    pair_id: str,
    source_text: str,
    target_entity: str,
    distractor_entity: str,
    target_source_state: str,
    distractor_source_state: str,
    target_new_state: str,
    distractor_new_state: str,
    update_sentence_target: str,
    update_sentence_distractor: str,
    use_sentence_frame: str,
    target_attribute: str = "",
) -> List[Dict[str, Any]]:
    """Build an UPDATE/RETAIN pair with validated structure.
    
    The use_sentence_frame should contain exactly one placeholder {STATE}
    that will be filled with the correct answer for each row.
    """
    packets = []
    
    for ptype in ["UPDATE", "RETAIN"]:
        if ptype == "UPDATE":
            update_sent = update_sentence_target
            answer_state = target_new_state
            foil_state = target_source_state
            updated_entity = target_entity
        else:
            update_sent = update_sentence_distractor
            answer_state = target_source_state
            foil_state = target_new_state
            updated_entity = distractor_entity
        
        use_sent = use_sentence_frame.replace("{STATE}", answer_state)
        full_text = f"{source_text} {update_sent} {use_sent}"
        
        # Validate no cue words in use sentence
        ok, violations = validate_no_cue_words(use_sent)
        
        # Find answer spans in full text
        answer_spans = find_text_spans(full_text, answer_state)
        # The answer in the use sentence is the LAST occurrence
        use_start = full_text.rfind(use_sent)
        answer_in_use = [s for s in answer_spans if s[0] >= use_start]
        answer_elsewhere = [s for s in answer_spans if s[0] < use_start]
        
        # Find foil spans
        foil_spans = find_text_spans(full_text, foil_state)
        
        # Find entity spans
        target_entity_spans = find_text_spans(full_text, target_entity)
        distractor_entity_spans = find_text_spans(full_text, distractor_entity)
        updated_entity_spans = find_text_spans(full_text, updated_entity)
        
        packet = {
            "pair_id": pair_id,
            "packet_type": ptype,
            "source_sentence": source_text,
            "update_sentence": update_sent,
            "use_sentence": use_sent,
            "use_sentence_frame": use_sentence_frame,
            "full_text": full_text,
            "answer_text": answer_state,
            "foil_text": foil_state,
            "entity_name": target_entity,
            "update_entity": updated_entity,
            "update_state_text": target_new_state if ptype == "UPDATE" else distractor_new_state,
            "source_state_text": target_source_state,
            "target_attribute": target_attribute,
            "answer_in_use_spans": answer_in_use,
            "answer_elsewhere_spans": answer_elsewhere,
            "foil_spans": foil_spans,
            "n_answer_occurrences": len(answer_spans),
            "n_foil_occurrences": len(foil_spans),
            "answer_also_in_source_or_update": len(answer_elsewhere) > 0,
            "cue_word_clean": ok,
            "cue_word_violations": violations,
            "word_count": len(full_text.split()),
        }
        packets.append(packet)
    
    return packets


def generate_template_pilot() -> List[Dict[str, Any]]:
    """Generate a pilot set of paired packets from templates."""
    all_packets = []
    
    templates = [
        # Pair 1: Simple possessions
        {
            "pair_id": "tp_001",
            "source": "Alice carried a red umbrella and Bob wore a blue hat.",
            "target_entity": "Alice",
            "distractor_entity": "Bob",
            "target_source_state": "red",
            "distractor_source_state": "blue",
            "target_new_state": "green",
            "distractor_new_state": "yellow",
            "update_target": "Later that day, Alice started carrying a green umbrella instead.",
            "update_distractor": "Later that day, Bob started wearing a yellow hat instead.",
            "use_frame": "When they met for dinner, Alice was holding a {STATE} umbrella.",
            "attribute": "color",
        },
        # Pair 2: Professional tools
        {
            "pair_id": "tp_002",
            "source": "Professor Chen kept leather-bound journals on the shelf and Dr. Reyes stored handwritten field notes in her cabinet.",
            "target_entity": "Professor Chen",
            "distractor_entity": "Dr. Reyes",
            "target_source_state": "leather-bound journals",
            "distractor_source_state": "handwritten field notes",
            "target_new_state": "digital tablets",
            "distractor_new_state": "printed spreadsheets",
            "update_target": "After the renovation, Professor Chen replaced the materials with digital tablets for cataloging.",
            "update_distractor": "After the renovation, Dr. Reyes replaced the materials with printed spreadsheets for record-keeping.",
            "use_frame": "A visiting scholar found {STATE} in Professor Chen's workspace.",
            "attribute": "workspace_materials",
        },
        # Pair 3: Vehicles
        {
            "pair_id": "tp_003",
            "source": "Tom drove a silver sedan to work while Sarah rode a black bicycle.",
            "target_entity": "Tom",
            "distractor_entity": "Sarah",
            "target_source_state": "silver sedan",
            "distractor_source_state": "black bicycle",
            "target_new_state": "white truck",
            "distractor_new_state": "orange scooter",
            "update_target": "Tom traded his vehicle for a white truck at the dealership.",
            "update_distractor": "Sarah traded her vehicle for an orange scooter at the shop.",
            "use_frame": "In the parking lot, Tom was seen loading groceries into a {STATE}.",
            "attribute": "vehicle",
        },
        # Pair 4: Locations
        {
            "pair_id": "tp_004",
            "source": "Maria lived in a small apartment near the university and James occupied a large house by the lake.",
            "target_entity": "Maria",
            "distractor_entity": "James",
            "target_source_state": "small apartment",
            "distractor_source_state": "large house",
            "target_new_state": "converted warehouse",
            "distractor_new_state": "garden cottage",
            "update_target": "Maria moved into a converted warehouse across town for more studio space.",
            "update_distractor": "James moved into a garden cottage in the countryside for quieter mornings.",
            "use_frame": "Friends visiting Maria found her in a {STATE} with high ceilings.",
            "attribute": "residence",
        },
        # Pair 5: Pets
        {
            "pair_id": "tp_005",
            "source": "Elena had a tabby cat named Whiskers and Marcus had a golden retriever named Buddy.",
            "target_entity": "Elena",
            "distractor_entity": "Marcus",
            "target_source_state": "tabby cat",
            "distractor_source_state": "golden retriever",
            "target_new_state": "spotted rabbit",
            "distractor_new_state": "gray parrot",
            "update_target": "Elena adopted a spotted rabbit from the shelter to keep as a companion.",
            "update_distractor": "Marcus adopted a gray parrot from the shelter to keep as a companion.",
            "use_frame": "At the neighborhood picnic, Elena brought a {STATE} for everyone to meet.",
            "attribute": "pet",
        },
        # Pair 6: Food preferences
        {
            "pair_id": "tp_006",
            "source": "Chef Kim prepared spicy tofu dishes at the restaurant while Chef Park specialized in grilled salmon plates.",
            "target_entity": "Chef Kim",
            "distractor_entity": "Chef Park",
            "target_source_state": "spicy tofu",
            "distractor_source_state": "grilled salmon",
            "target_new_state": "mushroom risotto",
            "distractor_new_state": "lamb stew",
            "update_target": "Chef Kim redesigned the menu around mushroom risotto as the signature dish.",
            "update_distractor": "Chef Park redesigned the menu around lamb stew as the signature dish.",
            "use_frame": "Food critics praised Chef Kim for the exceptional {STATE} served at the tasting.",
            "attribute": "signature_dish",
        },
        # Pair 7: Musical instruments  
        {
            "pair_id": "tp_007",
            "source": "David played the acoustic guitar at concerts and Sophia performed on the grand piano.",
            "target_entity": "David",
            "distractor_entity": "Sophia",
            "target_source_state": "acoustic guitar",
            "distractor_source_state": "grand piano",
            "target_new_state": "electric violin",
            "distractor_new_state": "steel drums",
            "update_target": "David switched to performing with an electric violin for the jazz festival.",
            "update_distractor": "Sophia switched to performing with steel drums for the world music showcase.",
            "use_frame": "At the benefit concert, David captivated the audience with his {STATE}.",
            "attribute": "instrument",
        },
        # Pair 8: Clothing
        {
            "pair_id": "tp_008",
            "source": "Rebecca wore a wool coat through the winter and Nathan preferred a leather jacket.",
            "target_entity": "Rebecca",
            "distractor_entity": "Nathan",
            "target_source_state": "wool coat",
            "distractor_source_state": "leather jacket",
            "target_new_state": "denim vest",
            "distractor_new_state": "flannel shirt",
            "update_target": "When spring arrived, Rebecca started wearing a denim vest as her everyday layer.",
            "update_distractor": "When spring arrived, Nathan started wearing a flannel shirt as his everyday layer.",
            "use_frame": "Passing Rebecca on the street, you could recognize her by the {STATE} she always had on.",
            "attribute": "outerwear",
        },
    ]
    
    for t in templates:
        pair = make_paired_packet(
            pair_id=t["pair_id"],
            source_text=t["source"],
            target_entity=t["target_entity"],
            distractor_entity=t["distractor_entity"],
            target_source_state=t["target_source_state"],
            distractor_source_state=t["distractor_source_state"],
            target_new_state=t["target_new_state"],
            distractor_new_state=t["distractor_new_state"],
            update_sentence_target=t["update_target"],
            update_sentence_distractor=t["update_distractor"],
            use_sentence_frame=t["use_frame"],
            target_attribute=t.get("attribute", ""),
        )
        all_packets.extend(pair)
    
    return all_packets


def validate_packets(packets: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Run comprehensive validation on packet set."""
    issues = []
    stats = {
        "total": len(packets),
        "pairs": len(set(p["pair_id"] for p in packets)),
        "UPDATE": sum(1 for p in packets if p["packet_type"] == "UPDATE"),
        "RETAIN": sum(1 for p in packets if p["packet_type"] == "RETAIN"),
        "cue_word_violations": 0,
        "answer_only_in_use": 0,
        "answer_also_elsewhere": 0,
        "mean_word_count": 0,
    }
    
    # Check paired structure
    by_pair = {}
    for p in packets:
        pid = p["pair_id"]
        if pid not in by_pair:
            by_pair[pid] = {}
        by_pair[pid][p["packet_type"]] = p
    
    for pid, pair in by_pair.items():
        if "UPDATE" not in pair or "RETAIN" not in pair:
            issues.append(f"Pair {pid} missing {'UPDATE' if 'RETAIN' in pair else 'RETAIN'}")
            continue
        
        u, r = pair["UPDATE"], pair["RETAIN"]
        
        # Check same use-sentence frame
        if u["use_sentence_frame"] != r["use_sentence_frame"]:
            issues.append(f"Pair {pid}: use_sentence_frame differs between UPDATE and RETAIN")
        
        # Check same source
        if u["source_sentence"] != r["source_sentence"]:
            issues.append(f"Pair {pid}: source_sentence differs")
        
        # Check same target entity
        if u["entity_name"] != r["entity_name"]:
            issues.append(f"Pair {pid}: entity_name differs")
        
        # Check different update entities
        if u["update_entity"] == r["update_entity"]:
            issues.append(f"Pair {pid}: same update_entity in both rows")
        
        # Check different answers (this is the counterfactual structure)
        if u["answer_text"] == r["answer_text"]:
            issues.append(f"Pair {pid}: same answer_text in both rows — no counterfactual contrast")
        
        # Paired shortcut analysis: can a fixed strategy solve both?
        u_copy_from_update = u["answer_also_in_source_or_update"] and any(
            s[0] < u["full_text"].find(u["use_sentence"]) 
            for s in find_text_spans(u["full_text"], u["answer_text"])
            if s[0] >= u["full_text"].find(u["update_sentence"])
        )
        r_copy_from_source = r["answer_also_in_source_or_update"] and any(
            s[0] < r["full_text"].find(r["update_sentence"])
            for s in find_text_spans(r["full_text"], r["answer_text"])
        )
    
    for p in packets:
        if not p["cue_word_clean"]:
            stats["cue_word_violations"] += 1
            issues.append(f"{p['pair_id']} {p['packet_type']}: cue words {p['cue_word_violations']}")
        
        if not p["answer_also_in_source_or_update"]:
            stats["answer_only_in_use"] += 1
        else:
            stats["answer_also_elsewhere"] += 1
    
    wcs = [p["word_count"] for p in packets]
    stats["mean_word_count"] = sum(wcs) / max(1, len(wcs))
    stats["min_word_count"] = min(wcs) if wcs else 0
    stats["max_word_count"] = max(wcs) if wcs else 0
    stats["n_issues"] = len(issues)
    stats["issues"] = issues[:20]  # cap
    
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["template-pilot", "from-sources"], default="template-pilot")
    ap.add_argument("--source-jsonl", type=str, default=None)
    ap.add_argument("--out-dir", type=str, required=True)
    args = ap.parse_args()
    
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    if args.mode == "template-pilot":
        packets = generate_template_pilot()
    else:
        print("from-sources mode not yet implemented", file=sys.stderr)
        sys.exit(1)
    
    # Validate
    vstats = validate_packets(packets)
    
    # Write packets
    pkt_path = out_dir / "paired_packets.jsonl"
    with open(pkt_path, "w") as f:
        for p in packets:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    
    # Write validation
    val_path = out_dir / "validation_stats.json"
    with open(val_path, "w") as f:
        json.dump(vstats, f, indent=2, ensure_ascii=False)
    
    # Write summary
    summary = {
        "status": "PAIRED_PACKETS_GENERATED",
        "mode": args.mode,
        "n_packets": len(packets),
        "n_pairs": vstats["pairs"],
        "n_issues": vstats["n_issues"],
        "mean_word_count": vstats["mean_word_count"],
        "cue_violations": vstats["cue_word_violations"],
        "answer_only_in_use": vstats["answer_only_in_use"],
        "answer_also_elsewhere": vstats["answer_also_elsewhere"],
        "packet_jsonl": str(pkt_path),
        "validation_json": str(val_path),
    }
    
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
