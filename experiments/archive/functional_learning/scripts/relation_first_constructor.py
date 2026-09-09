#!/usr/bin/env python3
"""research: Relation-first packet constructor.

Reverses the entity-first dependency from research: starts from actually
supported relations with explicit source-text evidence, then finds two
independently addressable instances and constructs clean recipient-contrast
packets.

The intended object is a small state map: entity has a value for a specified
relation at the relevant time. The update changes one entry while the queried
other entry remains intact.

Phase 1: Extract (entity, relation_type, value) triples from BabyLM Simple
         Wikipedia sentences with exact raw-text span validation.
Phase 2: Match pairs of entities sharing a relation type with different
         compatible values. Filter for quality.
Phase 3: Construct source-grounded update/retain/neutral contexts with
         explicit counterfactual new values (labeled as augmentation).
Phase 4: Validate spans and produce scorer-compatible output.
Phase 5: Score with coherent86 using symmetric multi-token MLM scoring.
"""

import json, re, pathlib, sys, random, collections, math, itertools
import argparse

# ──────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────

SOURCE_SENTENCES = [
    "experiments/archive/compact_experience/data/qwen_aligned/source_sentences.jsonl",
    "experiments/archive/compact_experience/data/qwen_clean_aligned/extra_source_sentences.jsonl",
]

COHERENT86_PATH = "models/frontier"

OUT_DIR = pathlib.Path("experiments/archive/functional_learning/data/relation_first_packets")

REPLACEMENT_CITIES = [
    "Amsterdam", "Barcelona", "Dublin", "Milan", "Tokyo", "Sydney",
    "Vienna", "Copenhagen", "Stockholm", "Athens", "Lisbon", "Prague",
    "Budapest", "Helsinki", "Cairo", "Montreal", "Zurich", "Brisbane",
    "Edinburgh", "Florence", "Geneva", "Hamburg", "Kyoto", "Munich",
    "Osaka", "Salzburg", "Venice", "Brussels", "Oslo", "Marseille",
    "Ankara", "Lima", "Bogota", "Manila", "Jakarta", "Havana",
    "Nairobi", "Doha", "Riyadh", "Bangalore",
]

SEED = 39039
HELD_FRACTION = 0.25  # fraction of pairs for held-out evaluation

# Relation-specific update and use templates
RELATION_TEMPLATES = {
    "birthplace": {
        "update": "However, recent records show that {ENTITY} was actually born in {NEW_VALUE}.",
        "use": "According to this information, the birthplace of {QUERY} is {STATE}.",
        "relation_phrase": "was born in",
    },
    "death_place": {
        "update": "However, updated records confirm that {ENTITY} actually died in {NEW_VALUE}.",
        "use": "According to this information, the place where {QUERY} died is {STATE}.",
        "relation_phrase": "died in",
    },
    "located_in": {
        "update": "However, after a boundary change, {ENTITY} is now part of {NEW_VALUE}.",
        "use": "According to current records, {QUERY} is located in {STATE}.",
        "relation_phrase": "is located in",
    },
    "founded_year": {
        "update": "However, newly discovered documents show that {ENTITY} was actually founded in {NEW_VALUE}.",
        "use": "According to the latest records, {QUERY} was founded in {STATE}.",
        "relation_phrase": "was founded in",
    },
}

REPLACEMENT_YEARS = [str(y) for y in range(1880, 2010, 5)]


# ──────────────────────────────────────────────────────────────
# Phase 1: Extract triples
# ──────────────────────────────────────────────────────────────

def extract_triples(source_paths):
    """Extract explicit (entity, relation_type, value) triples with spans."""
    triples = []
    
    ENTITY_RE = r"([A-Z][a-z]+(?:\s[A-Z][a-z]+){0,2})"
    PRONOUN_BLOCK = {"He", "She", "It", "They", "Who", "This", "That", "The", "His", "Her"}
    
    for path in source_paths:
        with open(path) as f:
            for line in f:
                row = json.loads(line)
                if row["source"] != "simple_wiki":
                    continue
                text = row["text"]
                sid = row.get("id", f'ex_{row["example_id"]}')
                eid = row.get("example_id")
                
                # --- birthplace ---
                for m in re.finditer(
                    ENTITY_RE + r"\s+was\s+born\s+(?:on\s+[^.]*?\s)?in\s+" + ENTITY_RE,
                    text
                ):
                    ent, val = m.group(1), m.group(2).rstrip(",.")
                    if ent in PRONOUN_BLOCK or len(ent) < 3 or len(val) < 3:
                        continue
                    # Find exact value span in original text
                    vi = text.find(val)
                    if vi < 0:
                        continue
                    triples.append({
                        "sentence_id": sid, "example_id": eid,
                        "entity": ent, "relation": "birthplace", "value": val,
                        "value_span": [vi, vi + len(val)],
                        "sentence": text, "source_type": "simple_wiki",
                        "grounding": "raw_source_exact_span",
                    })
                
                # --- death_place ---
                for m in re.finditer(
                    ENTITY_RE + r"\s+died\s+(?:on\s+[^.]*?\s)?in\s+" + ENTITY_RE,
                    text
                ):
                    ent, val = m.group(1), m.group(2).rstrip(",.")
                    if ent in PRONOUN_BLOCK or len(ent) < 3 or len(val) < 3:
                        continue
                    vi = text.find(val)
                    if vi < 0:
                        continue
                    triples.append({
                        "sentence_id": sid, "example_id": eid,
                        "entity": ent, "relation": "death_place", "value": val,
                        "value_span": [vi, vi + len(val)],
                        "sentence": text, "source_type": "simple_wiki",
                        "grounding": "raw_source_exact_span",
                    })
                
                # --- located_in ---
                for m in re.finditer(
                    ENTITY_RE + r"\s+is\s+(?:a|an|the)\s+(?:city|town|village|district|municipality|province|region|island)\s+in\s+(?:the\s+)?" + ENTITY_RE,
                    text
                ):
                    ent, val = m.group(1), m.group(2)
                    if ent in PRONOUN_BLOCK:
                        continue
                    vi = text.find(val, m.start(2))
                    if vi < 0:
                        continue
                    triples.append({
                        "sentence_id": sid, "example_id": eid,
                        "entity": ent, "relation": "located_in", "value": val,
                        "value_span": [vi, vi + len(val)],
                        "sentence": text, "source_type": "simple_wiki",
                        "grounding": "raw_source_exact_span",
                    })
                
                # --- founded_year ---
                for m in re.finditer(
                    ENTITY_RE + r"\s+was\s+founded\s+in\s+(\d{4})",
                    text
                ):
                    ent, val = m.group(1), m.group(2)
                    if ent in PRONOUN_BLOCK:
                        continue
                    vi = text.find(val, m.start(2))
                    if vi < 0:
                        continue
                    triples.append({
                        "sentence_id": sid, "example_id": eid,
                        "entity": ent, "relation": "founded_year", "value": val,
                        "value_span": [vi, vi + len(val)],
                        "sentence": text, "source_type": "simple_wiki",
                        "grounding": "raw_source_exact_span",
                    })
    
    return triples


# ──────────────────────────────────────────────────────────────
# Phase 2: Match pairs
# ──────────────────────────────────────────────────────────────

def match_pairs(triples, rng):
    """Match pairs of triples sharing relation type with different values."""
    by_relation = collections.defaultdict(list)
    for t in triples:
        by_relation[t["relation"]].append(t)
    
    pairs = []
    pair_id_counter = 0
    
    for rel, group in by_relation.items():
        # Deduplicate: keep one triple per entity (prefer shorter sentence)
        seen_entities = {}
        for t in sorted(group, key=lambda x: len(x["sentence"])):
            ent_lower = t["entity"].lower()
            if ent_lower not in seen_entities:
                seen_entities[ent_lower] = t
        
        unique_triples = list(seen_entities.values())
        
        # Match all pairs with different values
        for i, ta in enumerate(unique_triples):
            for j, tb in enumerate(unique_triples):
                if j <= i:
                    continue
                # Skip if same value (e.g., both born in London)
                if ta["value"].lower() == tb["value"].lower():
                    continue
                # Skip if entity name appears in the other's sentence (confusion)
                if ta["entity"].lower() in tb["sentence"].lower():
                    continue
                if tb["entity"].lower() in ta["sentence"].lower():
                    continue
                # Skip if value appears in the other's sentence (ambiguity)
                if ta["value"].lower() in tb["sentence"].lower():
                    continue
                if tb["value"].lower() in ta["sentence"].lower():
                    continue
                
                pair_id_counter += 1
                pairs.append({
                    "pair_id": f"rf_{rel}_{pair_id_counter:04d}",
                    "relation": rel,
                    "triple_a": ta,
                    "triple_b": tb,
                })
    
    rng.shuffle(pairs)
    return pairs


# ──────────────────────────────────────────────────────────────
# Phase 3: Construct contexts
# ──────────────────────────────────────────────────────────────

def select_new_value(relation, current_values, source_text, rng):
    """Select a plausible new value not in current values or source text."""
    if relation == "founded_year":
        pool = REPLACEMENT_YEARS
    else:
        pool = REPLACEMENT_CITIES
    
    candidates = [v for v in pool
                  if v.lower() not in [cv.lower() for cv in current_values]
                  and v.lower() not in source_text.lower()]
    
    if not candidates:
        return None
    return rng.choice(candidates)


def construct_pair_contexts(pair, rng):
    """Construct all contexts for a matched pair."""
    ta, tb = pair["triple_a"], pair["triple_b"]
    rel = pair["relation"]
    templates = RELATION_TEMPLATES.get(rel)
    if not templates:
        return None
    
    # Source context: compose two real sentences
    source_context = ta["sentence"].rstrip() + " " + tb["sentence"].rstrip()
    
    # Select new values
    all_current = [ta["value"], tb["value"]]
    new_a = select_new_value(rel, all_current, source_context, rng)
    if new_a is None:
        return None
    new_b = select_new_value(rel, all_current + [new_a], source_context, rng)
    if new_b is None:
        return None
    
    # Update sentences (augmented)
    update_a = templates["update"].format(ENTITY=ta["entity"], NEW_VALUE=new_a)
    update_b = templates["update"].format(ENTITY=tb["entity"], NEW_VALUE=new_b)
    
    # Use frames for each entity
    use_a = templates["use"].format(QUERY=ta["entity"], STATE="{STATE}")
    use_b = templates["use"].format(QUERY=tb["entity"], STATE="{STATE}")
    
    # Neutral context (no update)
    neutral_use_a = templates["use"].format(QUERY=ta["entity"], STATE="{STATE}")
    neutral_use_b = templates["use"].format(QUERY=tb["entity"], STATE="{STATE}")
    
    return {
        "pair_id": pair["pair_id"],
        "relation": rel,
        "entity_a": ta["entity"],
        "entity_b": tb["entity"],
        "value_a": ta["value"],  # source value for A
        "value_b": tb["value"],  # source value for B
        "new_value_a": new_a,    # counterfactual for A
        "new_value_b": new_b,    # counterfactual for B
        "source_a_sentence": ta["sentence"],
        "source_b_sentence": tb["sentence"],
        "source_a_id": ta["sentence_id"],
        "source_b_id": tb["sentence_id"],
        "source_a_value_span": ta["value_span"],
        "source_b_value_span": tb["value_span"],
        "source_context": source_context,
        "update_a_sentence": update_a,
        "update_b_sentence": update_b,
        "use_frame_a": use_a,
        "use_frame_b": use_b,
        "grounding_a": ta["grounding"],
        "grounding_b": tb["grounding"],
        "augmentation_note": "Update sentences are controlled counterfactual augmentation; source sentences are exact BabyLM text.",
        # Full contexts for scoring
        "context_update_a": source_context + " " + update_a,
        "context_update_b": source_context + " " + update_b,
        "context_neutral": source_context,
        # 4 evaluation conditions:
        # 1. Update A, Query A → answer=new_a (UPDATE)
        # 2. Update A, Query B → answer=value_b (RETAIN)
        # 3. Update B, Query A → answer=value_a (RETAIN)
        # 4. Update B, Query B → answer=new_b (UPDATE)
    }


def build_scoring_rows(pair_ctx):
    """Build scorer-compatible rows from a constructed pair context."""
    rows = []
    pc = pair_ctx
    
    # Row 1: Update A, Query A → UPDATE
    rows.append({
        "pair_id": pc["pair_id"],
        "row_type": "update_a_query_a",
        "role": "UPDATE",
        "updated_entity": pc["entity_a"],
        "query_entity": pc["entity_a"],
        "answer_text": pc["new_value_a"],
        "foil_text": pc["value_a"],
        "use_sentence_frame": pc["context_update_a"] + " " + pc["use_frame_a"],
        "relation": pc["relation"],
    })
    
    # Row 2: Update A, Query B → RETAIN
    rows.append({
        "pair_id": pc["pair_id"],
        "row_type": "update_a_query_b",
        "role": "RETAIN",
        "updated_entity": pc["entity_a"],
        "query_entity": pc["entity_b"],
        "answer_text": pc["value_b"],
        "foil_text": pc["new_value_a"],
        "use_sentence_frame": pc["context_update_a"] + " " + pc["use_frame_b"],
        "relation": pc["relation"],
    })
    
    # Row 3: Update B, Query A → RETAIN
    rows.append({
        "pair_id": pc["pair_id"],
        "row_type": "update_b_query_a",
        "role": "RETAIN",
        "updated_entity": pc["entity_b"],
        "query_entity": pc["entity_a"],
        "answer_text": pc["value_a"],
        "foil_text": pc["new_value_b"],
        "use_sentence_frame": pc["context_update_b"] + " " + pc["use_frame_a"],
        "relation": pc["relation"],
    })
    
    # Row 4: Update B, Query B → UPDATE
    rows.append({
        "pair_id": pc["pair_id"],
        "row_type": "update_b_query_b",
        "role": "UPDATE",
        "updated_entity": pc["entity_b"],
        "query_entity": pc["entity_b"],
        "answer_text": pc["new_value_b"],
        "foil_text": pc["value_b"],
        "use_sentence_frame": pc["context_update_b"] + " " + pc["use_frame_b"],
        "relation": pc["relation"],
    })
    
    # Neutral rows (no update)
    rows.append({
        "pair_id": pc["pair_id"],
        "row_type": "neutral_query_a",
        "role": "NEUTRAL",
        "updated_entity": "none",
        "query_entity": pc["entity_a"],
        "answer_text": pc["value_a"],
        "foil_text": pc["new_value_a"],  # arbitrary foil
        "use_sentence_frame": pc["context_neutral"] + " " + pc["use_frame_a"],
        "relation": pc["relation"],
    })
    
    rows.append({
        "pair_id": pc["pair_id"],
        "row_type": "neutral_query_b",
        "role": "NEUTRAL",
        "updated_entity": "none",
        "query_entity": pc["entity_b"],
        "answer_text": pc["value_b"],
        "foil_text": pc["new_value_b"],
        "use_sentence_frame": pc["context_neutral"] + " " + pc["use_frame_b"],
        "relation": pc["relation"],
    })
    
    return rows


# ──────────────────────────────────────────────────────────────
# Phase 4: Validate
# ──────────────────────────────────────────────────────────────

def validate_pair(pair_ctx):
    """Validate a constructed pair context."""
    issues = []
    
    # Check value spans are exact
    sa = pair_ctx["source_a_sentence"]
    sb = pair_ctx["source_b_sentence"]
    va = pair_ctx["value_a"]
    vb = pair_ctx["value_b"]
    
    if sa[pair_ctx["source_a_value_span"][0]:pair_ctx["source_a_value_span"][1]] != va:
        issues.append("value_a_span_mismatch")
    if sb[pair_ctx["source_b_value_span"][0]:pair_ctx["source_b_value_span"][1]] != vb:
        issues.append("value_b_span_mismatch")
    
    # Check new values are plausible (in replacement pool)
    # Check no value confusion
    if pair_ctx["new_value_a"].lower() == pair_ctx["value_a"].lower():
        issues.append("new_value_a_same_as_source")
    if pair_ctx["new_value_b"].lower() == pair_ctx["value_b"].lower():
        issues.append("new_value_b_same_as_source")
    
    # Check entities are different
    if pair_ctx["entity_a"].lower() == pair_ctx["entity_b"].lower():
        issues.append("same_entity")
    
    # Check no entity–value confusion
    if pair_ctx["entity_a"].lower() in pair_ctx["value_a"].lower():
        issues.append("entity_a_in_value_a")
    if pair_ctx["entity_b"].lower() in pair_ctx["value_b"].lower():
        issues.append("entity_b_in_value_b")
    
    # Check entity names are real (not stopwords)
    for ent in [pair_ctx["entity_a"], pair_ctx["entity_b"]]:
        if len(ent) < 3 or ent.lower() in ("the", "this", "that", "here"):
            issues.append(f"suspicious_entity_{ent}")
    
    return issues


# ──────────────────────────────────────────────────────────────
# Phase 5: Score with coherent86
# ──────────────────────────────────────────────────────────────

def score_with_model(scoring_rows, model_path, device="cpu"):
    """Score rows using symmetric multi-token MLM scoring."""
    import torch
    from transformers import AutoModelForMaskedLM, AutoTokenizer
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForMaskedLM.from_pretrained(model_path, local_files_only=True)
    model.to(device).eval()
    
    mask_id = tokenizer.mask_token_id
    
    def score_text(text, target_text, full_frame):
        """Score target tokens in context using mean per-token log-prob."""
        # Find where target appears in the full frame
        target_start = full_frame.rfind(target_text)
        if target_start < 0:
            return float("nan")
        target_end = target_start + len(target_text)
        
        # Tokenize full text
        enc = tokenizer(full_frame, return_tensors="pt", truncation=True, max_length=512)
        input_ids = enc["input_ids"][0]
        
        # Find token positions for target
        # Use offset mapping to locate target tokens
        enc_off = tokenizer(full_frame, return_offsets_mapping=True, truncation=True, max_length=512)
        offsets = enc_off["offset_mapping"]
        
        target_token_positions = []
        for idx, (s, e) in enumerate(offsets):
            if s >= target_start and e <= target_end and s < e:
                target_token_positions.append(idx)
        
        if not target_token_positions:
            return float("nan")
        
        # Mask target tokens and score
        total_logprob = 0.0
        with torch.no_grad():
            for pos in target_token_positions:
                masked = input_ids.clone().unsqueeze(0).to(device)
                true_id = masked[0, pos].item()
                masked[0, pos] = mask_id
                logits = model(input_ids=masked, attention_mask=enc["attention_mask"].to(device)).logits[0]
                lp = torch.log_softmax(logits[pos], dim=-1)
                total_logprob += lp[true_id].item()
        
        return total_logprob / len(target_token_positions)
    
    scored = []
    for row in scoring_rows:
        frame = row["use_sentence_frame"]
        ans = row["answer_text"]
        foil = row["foil_text"]
        
        # Create full text with answer inserted
        ans_frame = frame.replace("{STATE}", ans)
        foil_frame = frame.replace("{STATE}", foil)
        
        # Score answer and foil
        s_ans = score_text(ans, ans, ans_frame)
        s_foil = score_text(foil, foil, foil_frame)
        
        margin = s_ans - s_foil
        correct = margin > 0
        
        scored_row = dict(row)
        scored_row["score_answer"] = s_ans
        scored_row["score_foil"] = s_foil
        scored_row["margin"] = margin
        scored_row["correct"] = correct
        scored.append(scored_row)
    
    return scored


def compute_pair_metrics(scored_rows):
    """Compute U, R, neutral, beta, alpha, gamma per pair."""
    by_pair = collections.defaultdict(dict)
    for r in scored_rows:
        by_pair[r["pair_id"]][r["row_type"]] = r
    
    metrics = []
    for pid, rows in by_pair.items():
        # UPDATE margins: answer is the new value, margin > 0 means correct
        u_aa = rows.get("update_a_query_a", {}).get("margin", float("nan"))
        u_bb = rows.get("update_b_query_b", {}).get("margin", float("nan"))
        
        # RETAIN margins: answer is the source value, margin > 0 means correct
        r_ab = rows.get("update_a_query_b", {}).get("margin", float("nan"))
        r_ba = rows.get("update_b_query_a", {}).get("margin", float("nan"))
        
        # Neutral margins
        n_a = rows.get("neutral_query_a", {}).get("margin", float("nan"))
        n_b = rows.get("neutral_query_b", {}).get("margin", float("nan"))
        
        # Average across the two update/retain conditions
        U = (u_aa + u_bb) / 2  # mean UPDATE margin
        R = (r_ab + r_ba) / 2  # mean RETAIN margin (positive = correct)
        N = (n_a + n_b) / 2    # mean NEUTRAL margin
        
        beta = (U + R) / 2     # recipient dependence
        alpha = (U - R) / 2    # shared preference
        gamma = beta - abs(alpha)
        
        # Joint correctness: both UPDATE conditions correct AND both RETAIN correct
        update_correct = (u_aa > 0) and (u_bb > 0)
        retain_correct = (r_ab > 0) and (r_ba > 0)
        joint = update_correct and retain_correct
        
        metrics.append({
            "pair_id": pid,
            "relation": rows.get("update_a_query_a", {}).get("relation", "?"),
            "U": round(U, 4), "R": round(R, 4), "N": round(N, 4),
            "beta": round(beta, 4), "alpha": round(alpha, 4), "gamma": round(gamma, 4),
            "update_correct": update_correct,
            "retain_correct": retain_correct,
            "joint": joint,
            "u_aa": round(u_aa, 4), "u_bb": round(u_bb, 4),
            "r_ab": round(r_ab, 4), "r_ba": round(r_ba, 4),
            "n_a": round(n_a, 4), "n_b": round(n_b, 4),
        })
    
    return metrics


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--score", action="store_true", help="Score with coherent86")
    parser.add_argument("--device", default="cpu", help="Device for scoring")
    parser.add_argument("--max-pairs", type=int, default=120, help="Max pairs to construct")
    args = parser.parse_args()
    
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)
    
    # Phase 1: Extract
    print("Phase 1: Extracting triples...", flush=True)
    triples = extract_triples(SOURCE_SENTENCES)
    by_rel = collections.Counter(t["relation"] for t in triples)
    print(f"  Extracted {len(triples)} triples: {dict(by_rel)}", flush=True)
    
    # Save triples
    with open(OUT_DIR / "extracted_triples.jsonl", "w") as f:
        for t in triples:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")
    
    # Phase 2: Match pairs
    print("\nPhase 2: Matching pairs...", flush=True)
    pairs = match_pairs(triples, rng)
    print(f"  Matched {len(pairs)} candidate pairs", flush=True)
    
    # Limit to max_pairs
    if len(pairs) > args.max_pairs:
        pairs = pairs[:args.max_pairs]
        print(f"  Limited to {args.max_pairs} pairs", flush=True)
    
    # Phase 3: Construct contexts
    print("\nPhase 3: Constructing contexts...", flush=True)
    constructed = []
    for pair in pairs:
        ctx = construct_pair_contexts(pair, rng)
        if ctx is None:
            continue
        constructed.append(ctx)
    print(f"  Constructed {len(constructed)} pair contexts", flush=True)
    
    # Phase 4: Validate
    print("\nPhase 4: Validating...", flush=True)
    valid_pairs = []
    invalid_count = 0
    for ctx in constructed:
        issues = validate_pair(ctx)
        ctx["validation_issues"] = issues
        if issues:
            invalid_count += 1
        else:
            valid_pairs.append(ctx)
    
    print(f"  Valid: {len(valid_pairs)}, Invalid: {invalid_count}", flush=True)
    
    # Split train/held
    n_held = max(1, int(len(valid_pairs) * HELD_FRACTION))
    rng.shuffle(valid_pairs)
    held_pairs = valid_pairs[:n_held]
    train_pairs = valid_pairs[n_held:]
    
    for p in held_pairs:
        p["split"] = "held"
    for p in train_pairs:
        p["split"] = "train"
    
    all_pairs = train_pairs + held_pairs
    print(f"  Train: {len(train_pairs)}, Held: {len(held_pairs)}", flush=True)
    
    # Save pair contexts
    with open(OUT_DIR / "relation_first_pairs.jsonl", "w") as f:
        for p in all_pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    
    # Build scoring rows
    all_scoring_rows = []
    for p in all_pairs:
        rows = build_scoring_rows(p)
        for r in rows:
            r["split"] = p["split"]
        all_scoring_rows.append(rows)
    
    flat_rows = [r for group in all_scoring_rows for r in group]
    with open(OUT_DIR / "scoring_rows.jsonl", "w") as f:
        for r in flat_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    
    print(f"\n  Total scoring rows: {len(flat_rows)}", flush=True)
    
    # Phase 5: Score (if requested)
    if args.score:
        print(f"\nPhase 5: Scoring with coherent86 on {args.device}...", flush=True)
        scored = score_with_model(flat_rows, COHERENT86_PATH, args.device)
        
        # Save scored rows
        with open(OUT_DIR / "scored_rows.jsonl", "w") as f:
            for r in scored:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        
        # Compute pair metrics
        pair_metrics = compute_pair_metrics(scored)
        
        # Split metrics
        train_metrics = [m for m in pair_metrics if any(
            p["pair_id"] == m["pair_id"] and p["split"] == "train" for p in all_pairs)]
        held_metrics = [m for m in pair_metrics if any(
            p["pair_id"] == m["pair_id"] and p["split"] == "held" for p in all_pairs)]
        
        def summarize(metrics, label):
            n = len(metrics)
            if n == 0:
                return {"label": label, "n": 0}
            mean_U = sum(m["U"] for m in metrics) / n
            mean_R = sum(m["R"] for m in metrics) / n
            mean_beta = sum(m["beta"] for m in metrics) / n
            mean_alpha = sum(m["alpha"] for m in metrics) / n
            mean_abs_alpha = sum(abs(m["alpha"]) for m in metrics) / n
            mean_gamma = sum(m["gamma"] for m in metrics) / n
            n_update = sum(1 for m in metrics if m["update_correct"])
            n_retain = sum(1 for m in metrics if m["retain_correct"])
            n_joint = sum(1 for m in metrics if m["joint"])
            return {
                "label": label, "n": n,
                "mean_U": round(mean_U, 4), "mean_R": round(mean_R, 4),
                "mean_beta": round(mean_beta, 4),
                "mean_alpha": round(mean_alpha, 4),
                "mean_abs_alpha": round(mean_abs_alpha, 4),
                "mean_gamma": round(mean_gamma, 4),
                "n_update_correct": n_update, "n_retain_correct": n_retain,
                "n_joint": n_joint,
                "update_rate": round(n_update / n, 4),
                "retain_rate": round(n_retain / n, 4),
                "joint_rate": round(n_joint / n, 4),
            }
        
        all_summary = summarize(pair_metrics, "all")
        train_summary = summarize(train_metrics, "train")
        held_summary = summarize(held_metrics, "held")
        
        # Per-relation breakdown
        by_rel_metrics = collections.defaultdict(list)
        for m in pair_metrics:
            by_rel_metrics[m["relation"]].append(m)
        rel_summaries = {rel: summarize(ms, rel) for rel, ms in by_rel_metrics.items()}
        
        result = {
            "status": "RELATION_FIRST_PACKETS",
            "n_triples": len(triples),
            "n_candidate_pairs": len(pairs),
            "n_constructed": len(constructed),
            "n_valid": len(valid_pairs),
            "n_train": len(train_pairs),
            "n_held": len(held_pairs),
            "all": all_summary,
            "train": train_summary,
            "held": held_summary,
            "by_relation": rel_summaries,
        }
        
        with open(OUT_DIR / "relation_first_summary.json", "w") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        # Save pair metrics
        with open(OUT_DIR / "pair_metrics.jsonl", "w") as f:
            for m in pair_metrics:
                f.write(json.dumps(m) + "\n")
        
        print("\n" + "=" * 60, flush=True)
        print(json.dumps(result, indent=2), flush=True)
    else:
        # Just save construction summary without scoring
        result = {
            "status": "RELATION_FIRST_PACKETS_CONSTRUCTED",
            "n_triples": len(triples),
            "n_candidate_pairs": len(pairs),
            "n_constructed": len(constructed),
            "n_valid": len(valid_pairs),
            "n_train": len(train_pairs),
            "n_held": len(held_pairs),
            "relations": dict(by_rel),
            "sample_pair": all_pairs[0] if all_pairs else None,
        }
        with open(OUT_DIR / "relation_first_summary.json", "w") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        print("\n" + "=" * 60, flush=True)
        print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
