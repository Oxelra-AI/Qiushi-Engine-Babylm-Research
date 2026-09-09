#!/usr/bin/env python3
"""research: Validate state-update generations and materialize matched training streams.

Reads raw training generation outputs, validates state-update quality, packs valid
pairs into ~160-word rows, fills with official filler, creates 10M pool and 100M
training JSONL matching the exact COMPACT_EXPERIENCE Qwen pair construction.

For non-amenable originals (no entity detected), keeps the original Qwen restatement
pairs from COMPACT_EXPERIENCE.

The output stream can be directly compared to:
- chck_82M / chck_84M (base-level comparison)  
- The existing Qwen restatement stream (relation-type comparison)
"""
from __future__ import annotations

import collections
import hashlib
import json
import pathlib
import random
import re
import statistics
import time

ROOT = pathlib.Path.cwd()
STUDY = ROOT / "experiments/archive/relation_learning"

# Input paths
GEN_OUTPUTS = STUDY / "data/state_update_generation/raw_state_updates_qwen3_1p7b.jsonl"
EXISTING_PAIRS = ROOT / "experiments/archive/compact_experience/data/qwen_clean_aligned/selected_pairs.jsonl"
NONAM_IDS = STUDY / "data/state_update_generation/not_amenable_pair_ids.json"
OFFICIAL_POOL = ROOT / "experiments/archive/compact_experience/data/mixture/official_pool.jsonl"

# Output paths
OUT_DIR = STUDY / "data/state_update_materialization"
CORPUS_DIR = OUT_DIR / "training_corpora"

# Constants matching COMPACT_EXPERIENCE research
TOTAL_WORDS = 10_000_000
PASSES = 10
WORDS_PER_OFFICIAL_ROW = 160
MAX_PACKED_EXAMPLE_WORDS = 160
RNG_SEED = 41042

# Validation thresholds for state-update companions
MIN_GEN_WORDS = 6
MAX_GEN_WORDS = 70
MIN_LEN_RATIO = 0.35   # state-updates can be shorter/longer than originals
MAX_LEN_RATIO = 2.5
MAX_CONTENT_OVERLAP = 0.85  # should NOT be a pure restatement
MIN_ENTITY_RECALL = 0.3     # at least some entity reference preserved
MAX_COPY_OVERLAP = 0.90     # reject near-copies of original

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "being", "but", "by", "can", "could",
    "did", "do", "does", "for", "from", "had", "has", "have", "he", "her", "hers", "him", "his",
    "i", "if", "in", "into", "is", "it", "its", "just", "may", "might", "must", "not", "of", "on",
    "or", "our", "she", "should", "so", "some", "such", "than", "that", "the", "their", "them",
    "then", "there", "these", "they", "this", "to", "too", "under", "up", "very", "was", "we", "were",
    "what", "when", "where", "which", "who", "will", "with", "would", "you", "your",
}

META_PREFIXES = (
    "here is", "here's", "sure", "certainly", "of course", "the state", "state-update",
    "sentence:", "output:", "i'm sorry", "i cannot", "i can",
)

WORD_RE = re.compile(r"[A-Za-z0-9]+(?:[\u2019'][A-Za-z0-9]+)?")


def normalize_ws(text: str) -> str:
    return " ".join(text.replace("\u00a0", " ").split())


def strip_outer_quotes(text: str) -> str:
    text = normalize_ws(text.strip())
    changed = True
    while changed and len(text) >= 2:
        changed = False
        for a, b in [("\"", "\""), ("'", "'"), ("\u201c", "\u201d"), ("\u2018", "\u2019")]:
            if text.startswith(a) and text.endswith(b):
                text = normalize_ws(text[1:-1].strip())
                changed = True
    return text


def clean_output(raw: str) -> str:
    text = strip_outer_quotes(raw or "")
    low = text.lower()
    for marker in ("state-update sentence:", "state update:", "output:", "sentence:"):
        if low.startswith(marker):
            text = normalize_ws(text[len(marker):].strip())
            low = text.lower()
    return strip_outer_quotes(text)


def norm_tokens(text: str) -> list[str]:
    return [m.group(0).lower() for m in WORD_RE.finditer(text)]


def content_tokens(text: str) -> set[str]:
    return {t for t in norm_tokens(text) if len(t) >= 3 and t not in STOPWORDS}


def entity_tokens(text: str) -> list[str]:
    """Extract likely entity tokens (capitalized non-first words, CHILDES marks)."""
    words = text.split()
    ents = []
    for i, w in enumerate(words):
        stripped = w.strip(".,!?;:()[]{}\"'\u201c\u201d\u2018\u2019")
        if not stripped:
            continue
        if stripped.startswith("*") and len(stripped) > 2:
            ents.append(stripped.lower().rstrip(":"))
            continue
        letters = re.sub(r"[^A-Za-z]", "", stripped)
        if not letters or len(letters) < 2:
            continue
        has_internal_cap = any(c.isupper() for c in letters[1:])
        all_caps = letters.isupper() and len(letters) >= 2
        is_cap = letters[0].isupper()
        if i == 0 and not has_internal_cap and not all_caps:
            continue  # sentence-initial capitalization
        if is_cap or has_internal_cap or all_caps:
            if stripped.lower() not in STOPWORDS:
                ents.append(stripped.lower())
    return list(dict.fromkeys(ents))


def has_repetition(text: str) -> bool:
    toks = norm_tokens(text)
    if len(toks) < 8:
        return False
    run = 1
    for a, b in zip(toks, toks[1:]):
        run = run + 1 if a == b else 1
        if run >= 4:
            return True
    grams = [tuple(toks[i:i+4]) for i in range(len(toks) - 3)]
    c = collections.Counter(grams)
    return any(v >= 2 for v in c.values())


def validate_state_update(original: str, generated: str) -> tuple[bool, list[str]]:
    """Validate that generated text is a good entity-state-update companion."""
    reasons = []
    gen = clean_output(generated)
    if not gen:
        return False, ["empty"]
    
    gen_words = len(gen.split())
    orig_words = len(original.split())
    
    if gen_words < MIN_GEN_WORDS:
        reasons.append(f"too_short_{gen_words}")
    if gen_words > MAX_GEN_WORDS:
        reasons.append(f"too_long_{gen_words}")
    
    ratio = gen_words / max(1, orig_words)
    if ratio < MIN_LEN_RATIO or ratio > MAX_LEN_RATIO:
        reasons.append(f"len_ratio_{ratio:.2f}")
    
    # Check meta prefixes
    low = gen.lower().strip()
    if any(low.startswith(p) for p in META_PREFIXES):
        reasons.append("meta_prefix")
    
    # Check sentence completeness
    if not re.search(r"[.!?][\"'\u201d\u2019\)\]]*$", gen):
        reasons.append("incomplete_sentence")
    
    if has_repetition(gen):
        reasons.append("repetition")
    
    # Content overlap with original (should be moderate, not too high)
    orig_content = content_tokens(original)
    gen_content = content_tokens(gen)
    if orig_content and gen_content:
        overlap = len(orig_content & gen_content) / max(1, min(len(orig_content), len(gen_content)))
    else:
        overlap = 0.0
    
    if overlap > MAX_COPY_OVERLAP:
        reasons.append(f"copy_{overlap:.2f}")
    
    # Entity recall: at least some entity reference preserved
    orig_ents = set(entity_tokens(original))
    gen_ents = set(entity_tokens(gen))
    if orig_ents:
        recall = len(orig_ents & gen_ents) / len(orig_ents)
        if recall < MIN_ENTITY_RECALL and gen_words >= 10:
            # Also check for pronoun references or partial matches
            gen_lower = gen.lower()
            has_pronoun_ref = any(p in gen_lower for p in ["he ", "she ", "it ", "they ", "his ", "her ", "its ", "their "])
            if not has_pronoun_ref:
                reasons.append(f"low_entity_recall_{recall:.2f}")
    
    # State change check: the generation should NOT be too similar to original
    # (it should describe a CHANGE, not a restatement)
    norm_orig = set(norm_tokens(original))
    norm_gen = set(norm_tokens(gen))
    if norm_orig and norm_gen:
        token_overlap = len(norm_orig & norm_gen) / max(1, min(len(norm_orig), len(norm_gen)))
        if token_overlap > MAX_CONTENT_OVERLAP:
            reasons.append(f"too_similar_{token_overlap:.2f}")
    
    return len(reasons) == 0, reasons


def pack_pairs(pairs: list[dict], max_words: int = MAX_PACKED_EXAMPLE_WORDS) -> list[dict]:
    """Pack original+companion pairs into ~160-word rows."""
    rows = []
    cur_segments, cur_pair_ids, cur_words = [], [], 0
    cur_sources = collections.Counter()
    base_id = 800000
    
    for p in pairs:
        pair_words = p["original_words"] + p["companion_words"]
        
        if pair_words > max_words:
            if cur_segments:
                rows.append({
                    "text": " ".join(cur_segments), "words": cur_words,
                    "source": "state_update_packed", "example_id": base_id + len(rows),
                    "pair_ids": list(cur_pair_ids), "n_pairs": len(cur_pair_ids),
                    "component_sources": dict(cur_sources), "relation_type": p.get("relation_type", "mixed"),
                })
                cur_segments, cur_pair_ids, cur_words, cur_sources = [], [], 0, collections.Counter()
            rows.append({
                "text": f"{p['original']} {p['companion']}", "words": pair_words,
                "source": "state_update_packed", "example_id": base_id + len(rows),
                "pair_ids": [p["pair_id"]], "n_pairs": 1,
                "component_sources": {p.get("original_source", "unknown"): pair_words},
                "relation_type": p.get("relation_type", "state_update"),
            })
            continue
        
        if cur_words and cur_words + pair_words > max_words:
            rows.append({
                "text": " ".join(cur_segments), "words": cur_words,
                "source": "state_update_packed", "example_id": base_id + len(rows),
                "pair_ids": list(cur_pair_ids), "n_pairs": len(cur_pair_ids),
                "component_sources": dict(cur_sources), "relation_type": "mixed",
            })
            cur_segments, cur_pair_ids, cur_words, cur_sources = [], [], 0, collections.Counter()
        
        cur_segments.append(p["original"])
        cur_segments.append(p["companion"])
        cur_pair_ids.append(p["pair_id"])
        cur_words += pair_words
        cur_sources[p.get("original_source", "unknown")] += pair_words
    
    if cur_segments:
        rows.append({
            "text": " ".join(cur_segments), "words": cur_words,
            "source": "state_update_packed", "example_id": base_id + len(rows),
            "pair_ids": list(cur_pair_ids), "n_pairs": len(cur_pair_ids),
            "component_sources": dict(cur_sources), "relation_type": "mixed",
        })
    
    return rows


def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    
    # 1. Read generation outputs
    print("Reading generation outputs...", flush=True)
    if not GEN_OUTPUTS.exists():
        print(f"ERROR: {GEN_OUTPUTS} not found. Run generation first.", flush=True)
        raise SystemExit(1)
    
    gen_records = {}
    with GEN_OUTPUTS.open() as f:
        for line in f:
            rec = json.loads(line)
            gen_records[rec.get("pair_id", rec.get("id", ""))] = rec
    print(f"  Read {len(gen_records)} generation outputs", flush=True)
    
    # 2. Read existing Qwen pairs (for non-amenable fallback)
    print("Reading existing Qwen pairs...", flush=True)
    existing_pairs = {}
    with EXISTING_PAIRS.open() as f:
        for line in f:
            p = json.loads(line)
            existing_pairs[p["pair_id"]] = p
    print(f"  Read {len(existing_pairs)} existing pairs", flush=True)
    
    # 3. Read non-amenable IDs
    nonam_data = json.loads(NONAM_IDS.read_text())
    nonam_ids = set(nonam_data["pair_ids"])
    print(f"  Non-amenable: {len(nonam_ids)}", flush=True)
    
    # 4. Validate state-update generations
    print("Validating state-update generations...", flush=True)
    valid_state_updates = []
    rejection_reasons = collections.Counter()
    
    for pair_id, gen_rec in gen_records.items():
        ok, reasons = validate_state_update(
            gen_rec["source_sentence"],
            gen_rec.get("generated", gen_rec.get("output", "")),
        )
        if ok:
            cleaned = clean_output(gen_rec.get("generated", gen_rec.get("output", "")))
            valid_state_updates.append({
                "pair_id": pair_id,
                "original": gen_rec["source_sentence"],
                "companion": cleaned,
                "original_words": len(gen_rec["source_sentence"].split()),
                "companion_words": len(cleaned.split()),
                "relation_type": "state_update",
                "original_source": existing_pairs.get(pair_id, {}).get("source", "unknown"),
            })
        else:
            for r in reasons:
                rejection_reasons[r] += 1
    
    accept_rate = len(valid_state_updates) / max(1, len(gen_records))
    print(f"  Valid state-updates: {len(valid_state_updates)} / {len(gen_records)} "
          f"({100*accept_rate:.1f}%)", flush=True)
    print(f"  Top rejection reasons: {rejection_reasons.most_common(10)}", flush=True)
    
    # 5. Add non-amenable originals with their existing Qwen restatements
    restatement_fallbacks = []
    for pair_id in nonam_ids:
        if pair_id in existing_pairs:
            p = existing_pairs[pair_id]
            restatement_fallbacks.append({
                "pair_id": pair_id,
                "original": p["original"],
                "companion": p["rewrite"],
                "original_words": len(p["original"].split()),
                "companion_words": len(p["rewrite"].split()),
                "relation_type": "restatement",
                "original_source": p.get("source", "unknown"),
            })
    
    # Also add rejected state-updates as restatement fallback
    rejected_amenable_ids = set(gen_records.keys()) - {p["pair_id"] for p in valid_state_updates}
    for pair_id in rejected_amenable_ids:
        if pair_id in existing_pairs:
            p = existing_pairs[pair_id]
            restatement_fallbacks.append({
                "pair_id": pair_id,
                "original": p["original"],
                "companion": p["rewrite"],
                "original_words": len(p["original"].split()),
                "companion_words": len(p["rewrite"].split()),
                "relation_type": "restatement_fallback",
                "original_source": p.get("source", "unknown"),
            })
    
    print(f"  Restatement fallbacks: {len(restatement_fallbacks)}", flush=True)
    
    # 6. Combine all pairs
    all_pairs = valid_state_updates + restatement_fallbacks
    rng = random.Random(RNG_SEED)
    rng.shuffle(all_pairs)
    
    # 7. Select pairs up to target budget (matching COMPACT_EXPERIENCE pair fraction)
    target_pair_words = int(TOTAL_WORDS * 0.25)  # 25% of 10M = 2.5M words
    selected_pairs = []
    total_pair_words = 0
    n_su, n_rs = 0, 0
    
    # Prioritize state-update pairs
    su_pairs = [p for p in all_pairs if p["relation_type"] == "state_update"]
    rs_pairs = [p for p in all_pairs if p["relation_type"] != "state_update"]
    
    for p in su_pairs:
        pw = p["original_words"] + p["companion_words"]
        if total_pair_words + pw <= target_pair_words:
            selected_pairs.append(p)
            total_pair_words += pw
            n_su += 1
    
    for p in rs_pairs:
        pw = p["original_words"] + p["companion_words"]
        if total_pair_words + pw <= target_pair_words:
            selected_pairs.append(p)
            total_pair_words += pw
            n_rs += 1
    
    # Round down to whole official rows for filler
    while selected_pairs and total_pair_words % WORDS_PER_OFFICIAL_ROW != 0:
        p = selected_pairs.pop()
        total_pair_words -= (p["original_words"] + p["companion_words"])
        if p["relation_type"] == "state_update":
            n_su -= 1
        else:
            n_rs -= 1
    
    print(f"  Selected pairs: {len(selected_pairs)} "
          f"(state_update={n_su}, restatement={n_rs})", flush=True)
    print(f"  Total pair words: {total_pair_words}", flush=True)
    
    # 8. Pack pairs into rows
    rng.shuffle(selected_pairs)
    packed_rows = pack_pairs(selected_pairs)
    packed_words = sum(r["words"] for r in packed_rows)
    assert packed_words == total_pair_words, f"{packed_words} != {total_pair_words}"
    print(f"  Packed rows: {len(packed_rows)}, words: {packed_words}", flush=True)
    
    # 9. Read official pool and fill remaining words
    print("Reading official pool...", flush=True)
    official_pool = []
    with OFFICIAL_POOL.open() as f:
        for line in f:
            official_pool.append(json.loads(line))
    
    filler_words_needed = TOTAL_WORDS - packed_words
    filler_rows_needed = filler_words_needed // WORDS_PER_OFFICIAL_ROW
    assert filler_words_needed == filler_rows_needed * WORDS_PER_OFFICIAL_ROW
    
    # Exclude source example IDs from filler
    source_example_ids = set()
    for p in selected_pairs:
        if p["pair_id"] in existing_pairs:
            source_example_ids.add(existing_pairs[p["pair_id"]].get("example_id", -1))
    
    rng2 = random.Random(RNG_SEED + 1)
    filler_candidates = [r for r in official_pool if r.get("example_id", -1) not in source_example_ids]
    remaining = [r for r in official_pool if r.get("example_id", -1) in source_example_ids]
    rng2.shuffle(filler_candidates)
    rng2.shuffle(remaining)
    filler_pool = (filler_candidates + remaining)[:filler_rows_needed]
    
    if len(filler_pool) < filler_rows_needed:
        raise RuntimeError(f"Need {filler_rows_needed} filler rows, only have {len(filler_pool)}")
    
    filler_rows = [{
        "text": r["text"], "words": int(r.get("words", len(r["text"].split()))),
        "source": r["source"], "example_id": r["example_id"],
    } for r in filler_pool]
    
    filler_total = sum(r["words"] for r in filler_rows)
    assert filler_total == filler_words_needed, f"Filler {filler_total} != {filler_words_needed}"
    
    # 10. Combine and write 10M pool
    pool_rows = packed_rows + filler_rows
    pool_total = sum(r["words"] for r in pool_rows)
    assert pool_total == TOTAL_WORDS, f"Pool {pool_total} != {TOTAL_WORDS}"
    print(f"  Pool: {len(pool_rows)} rows, {pool_total} words", flush=True)
    
    pool_path = CORPUS_DIR / "state_update_10M.jsonl"
    meta_path = CORPUS_DIR / "state_update_rows_meta.jsonl"
    with pool_path.open("w", encoding="utf-8") as f, meta_path.open("w", encoding="utf-8") as mf:
        for i, r in enumerate(pool_rows):
            assert r["words"] == len(r["text"].split()), f"Row {i} mismatch"
            f.write(json.dumps({
                "text": r["text"], "words": r["words"],
                "example_id": r.get("example_id", i), "source": r["source"],
            }, ensure_ascii=False) + "\n")
            if r.get("pair_ids"):
                mf.write(json.dumps({
                    "row_index": i, "example_id": r.get("example_id", i),
                    "words": r["words"], "pair_ids": r["pair_ids"],
                    "n_pairs": r["n_pairs"],
                    "component_sources": r.get("component_sources", {}),
                    "relation_type": r.get("relation_type", "mixed"),
                }, ensure_ascii=False) + "\n")
    
    # 11. Create 100M training file (10 shuffled passes)
    print("Creating 100M training file...", flush=True)
    training_path = CORPUS_DIR / "state_update_100M.jsonl"
    pass_orders = []
    indices = list(range(len(pool_rows)))
    for pass_i in range(PASSES):
        order = list(indices)
        random.Random(RNG_SEED + 100 + pass_i).shuffle(order)
        pass_orders.append(order)
    
    total_training_words = 0
    with training_path.open("w", encoding="utf-8") as f:
        for order in pass_orders:
            for idx in order:
                r = pool_rows[idx]
                f.write(json.dumps({
                    "text": r["text"], "words": r["words"],
                    "example_id": r.get("example_id", idx), "source": r["source"],
                }, ensure_ascii=False) + "\n")
                total_training_words += r["words"]
    
    assert total_training_words == TOTAL_WORDS * PASSES
    print(f"  Training file: {total_training_words} words ({PASSES} passes)", flush=True)
    
    # 12. Compute hashes
    pool_sha = sha256_file(pool_path)
    training_sha = sha256_file(training_path)
    
    # 13. Save metadata
    metadata = {
        "status": "STATE_UPDATE_MATERIALIZED",
        "total_generation_outputs": len(gen_records),
        "valid_state_updates": len(valid_state_updates),
        "acceptance_rate": round(accept_rate, 4),
        "restatement_fallbacks": len(restatement_fallbacks),
        "selected_state_update_pairs": n_su,
        "selected_restatement_pairs": n_rs,
        "total_selected_pairs": len(selected_pairs),
        "total_pair_words": total_pair_words,
        "packed_rows": len(packed_rows),
        "filler_rows": len(filler_rows),
        "filler_words": filler_words_needed,
        "pool_total_words": TOTAL_WORDS,
        "training_total_words": TOTAL_WORDS * PASSES,
        "passes": PASSES,
        "pool_path": str(pool_path),
        "training_path": str(training_path),
        "meta_path": str(meta_path),
        "pool_sha256": pool_sha,
        "training_sha256": training_sha,
        "rejection_reasons": dict(rejection_reasons.most_common(20)),
        "state_update_word_stats": {
            "original_mean": round(statistics.mean(p["original_words"] for p in valid_state_updates), 2) if valid_state_updates else 0,
            "companion_mean": round(statistics.mean(p["companion_words"] for p in valid_state_updates), 2) if valid_state_updates else 0,
        },
        "elapsed_sec": round(time.time() - t0, 1),
    }
    
    meta_out = OUT_DIR / "state_update_materialization_metadata.json"
    meta_out.write_text(json.dumps(metadata, indent=2, ensure_ascii=False))
    
    # 14. Save selected pairs for reference
    pairs_out = OUT_DIR / "selected_state_update_pairs.jsonl"
    with pairs_out.open("w", encoding="utf-8") as f:
        for p in selected_pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
