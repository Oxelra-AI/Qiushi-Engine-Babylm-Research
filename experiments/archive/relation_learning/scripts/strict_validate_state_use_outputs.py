#!/usr/bin/env python3
"""research: strict validation and train/held-out split for natural state-use packets.

This replaces the research exact-substring validator.  The important training
signal is not only that the use sentence names the entity, but that its masked
state-bearing tokens express the right current state:

- UPDATED_USE: use sentence must carry new_state content and must not carry
  source_state-specific content.
- UNCHANGED_DISTRACTOR_USE: use sentence must carry source_state content and must
  not carry distractor new_state content.

The validator uses only deterministic lexical tests.  It also reserves a balanced
held-out slice of accepted packets for a source/update/use margin probe, and
writes the remaining packets for stream assembly.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import random
import re
import time
from collections import Counter, defaultdict
from typing import Any

ROOT = pathlib.Path.cwd()
DATA = ROOT / "experiments/archive/relation_learning/data/state_use_generation"
FULL_PROMPTS = DATA / "state_use_prompts_9b.jsonl"
MAIN_RAW = DATA / "raw_outputs_9b.jsonl"
CONT_PROMPTS = DATA / "state_use_prompts_9b_remaining_after_step043.jsonl"
CONT_RAW = DATA / "raw_outputs_9b_continuation_step044.jsonl"
FORCED_UPDATED_PROMPTS = DATA / "state_use_prompts_9b_updated_forced_step044.jsonl"
FORCED_UPDATED_RAW = DATA / "raw_outputs_9b_updated_forced_step044.jsonl"

OUT_DIR = DATA

WORD_RE = re.compile(r"[A-Za-z0-9]+(?:[\u2019'][A-Za-z0-9]+)?")
REQUIRED_KEYS = {"target_entity", "source_state", "updated_entity", "new_state", "update_sentence", "use_sentence"}
BENCHMARK_VOCAB = {"box", "basket", "marble", "container", "containers", "baskets", "boxes", "marbles"}

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "being", "but", "by", "can", "could",
    "did", "do", "does", "for", "from", "had", "has", "have", "he", "her", "hers", "him", "his",
    "i", "if", "in", "into", "is", "it", "its", "just", "may", "might", "must", "not", "of", "on",
    "or", "our", "she", "should", "so", "some", "such", "than", "that", "the", "their", "them",
    "then", "there", "these", "they", "this", "to", "too", "under", "up", "very", "was", "we",
    "were", "what", "when", "where", "which", "who", "will", "with", "would", "you", "your",
    "before", "after", "while", "over", "about", "also", "through", "each", "every", "all", "any",
    "now", "still", "continues", "continue", "continued", "continually", "currently", "former", "formerly",
}

ENTITY_GENERIC = {
    "mr", "mrs", "ms", "miss", "dr", "sir", "madam", "man", "woman", "person", "people", "child",
    "children", "boy", "girl", "men", "women", "city", "town", "country", "state", "province", "village",
    "place", "area", "region", "room", "house", "home", "building", "site", "part", "way", "thing", "things",
    "group", "team", "company", "organization", "institution", "government", "system", "event", "story",
    "january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november", "december",
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday", "mot", "chi", "mommy", "daddy",
}

STATE_GENERIC = {
    "state", "status", "condition", "role", "position", "place", "location", "site", "area", "region", "part",
    "new", "old", "current", "original", "changed", "different", "same", "located", "situated", "based",
    "living", "lives", "live", "lived", "resides", "reside", "resided", "working", "works", "worked",
    "serving", "serves", "served", "became", "become", "becomes", "being", "having", "remains", "remain",
    "keeps", "kept", "stays", "stayed", "still", "now", "after", "before", "during", "through", "near",
    "meeting", "office", "home", "house", "room", "place", "local", "main", "central", "around", "along",
    "month", "months", "date", "day", "days", "year", "years",
}

RNG_SEED = 44044
HELDOUT_TARGET_TOTAL = 1000


def norm_ws(text: Any) -> str:
    return " ".join(str(text).replace("\u00a0", " ").split())


def raw_words(text: str) -> list[str]:
    return [m.group(0) for m in WORD_RE.finditer(str(text))]


def wc(text: str) -> int:
    return len(raw_words(text))


def stem(word: str) -> str:
    w = word.lower().strip("\u2019'")
    if w.endswith("'s"):
        w = w[:-2]
    if len(w) > 6 and w.endswith("ing"):
        return w[:-3]
    if len(w) > 5 and w.endswith("ied"):
        return w[:-3] + "y"
    if len(w) > 5 and w.endswith("ed"):
        return w[:-2]
    if len(w) > 5 and w.endswith("es"):
        return w[:-2]
    if len(w) > 4 and w.endswith("s") and not w.endswith("ss"):
        return w[:-1]
    return w


def content_words(text: str, *, kind: str) -> list[str]:
    out: list[str] = []
    extra = ENTITY_GENERIC if kind == "entity" else STATE_GENERIC
    for w in raw_words(text):
        wl = stem(w)
        if len(wl) < 3:
            continue
        if wl in STOPWORDS or wl in extra:
            continue
        if kind == "state" and wl.isdigit() and len(wl) <= 3:
            continue
        out.append(wl)
    # Deduplicate preserving order
    seen = set()
    uniq = []
    for w in out:
        if w not in seen:
            seen.add(w)
            uniq.append(w)
    return uniq


def contains_content(sentence: str, terms: list[str], *, need: int | None = None) -> tuple[bool, list[str]]:
    sent = set(content_words(sentence, kind="state")) | set(content_words(sentence, kind="entity"))
    hits = [t for t in terms if t in sent]
    if need is None:
        need = 1 if len(terms) <= 2 else 2
    return len(hits) >= min(need, len(terms)), hits


def required_state_hits(terms: list[str]) -> int:
    """Demand enough lexical support that one accidental content overlap does not pass."""
    if len(terms) <= 1:
        return 1
    return min(len(terms), max(2, (len(terms) + 1) // 2))


def entity_matches_candidate(entity_terms: list[str], candidates: list[Any]) -> tuple[bool, list[str]]:
    """Generated target entity must correspond to a contentful extracted source entity."""
    if not entity_terms:
        return False, []
    e = set(entity_terms)
    matched = []
    for cand in candidates or []:
        cterms = content_words(str(cand), kind="entity")
        if not cterms:
            continue
        # Require either exact term subset/overlap for short names, or any distinctive
        # proper-name token for longer names. Articles and generic labels have already
        # been removed from cterms/entity_terms.
        inter = e & set(cterms)
        if inter and (len(e) == 1 or len(cterms) == 1 or len(inter) >= min(2, len(e), len(cterms))):
            matched.append(str(cand))
    return bool(matched), matched


def state_changed(source_state: str, new_state: str) -> tuple[bool, dict[str, Any]]:
    s = content_words(source_state, kind="state")
    n = content_words(new_state, kind="state")
    if not s:
        return False, {"reason": "source_state_weak_content", "source_terms": s, "new_terms": n}
    if not n:
        return False, {"reason": "new_state_weak_content", "source_terms": s, "new_terms": n}
    ss, ns = set(s), set(n)
    inter = ss & ns
    overlap_coef = len(inter) / max(1, min(len(ss), len(ns)))
    jaccard = len(inter) / max(1, len(ss | ns))
    new_specific = sorted(ns - ss)
    source_specific = sorted(ss - ns)
    if not new_specific:
        return False, {"reason": "new_state_no_new_content", "source_terms": s, "new_terms": n, "overlap_coef": overlap_coef, "jaccard": jaccard}
    if overlap_coef >= 0.67 and jaccard >= 0.50:
        return False, {"reason": "state_content_too_similar", "source_terms": s, "new_terms": n, "overlap_coef": overlap_coef, "jaccard": jaccard}
    return True, {
        "source_terms": s,
        "new_terms": n,
        "source_specific": source_specific,
        "new_specific": new_specific,
        "overlap_coef": overlap_coef,
        "jaccard": jaccard,
    }


def parse_output(raw_text: str) -> tuple[dict[str, Any] | None, str]:
    txt = raw_text.strip()
    if txt.startswith("```"):
        txt = "\n".join(l for l in txt.splitlines() if not l.strip().startswith("```"))
    start = txt.find("{")
    end = txt.rfind("}") + 1
    if start < 0 or end <= start:
        return None, "no_json"
    try:
        obj = json.loads(txt[start:end])
    except json.JSONDecodeError:
        return None, "json_parse_error"
    missing = REQUIRED_KEYS - set(obj.keys())
    if missing:
        return None, "missing_keys:" + ",".join(sorted(missing))
    return obj, ""


def validate_output(raw_text: str, prompt_rec: dict[str, Any]) -> tuple[dict[str, Any] | None, str, dict[str, Any]]:
    obj, reason = parse_output(raw_text)
    if obj is None:
        return None, reason, {}

    packet_type = prompt_rec["packet_type"]
    source_sentence = norm_ws(prompt_rec["source_sentence"])

    target_ent = norm_ws(obj["target_entity"])
    updated_ent = norm_ws(obj["updated_entity"])
    source_state = norm_ws(obj["source_state"])
    new_state = norm_ws(obj["new_state"])
    update_sent = norm_ws(obj["update_sentence"])
    use_sent = norm_ws(obj["use_sentence"])

    if target_ent.strip().startswith("*") or updated_ent.strip().startswith("*"):
        return None, "transcript_speaker_label_entity", {"target_entity": target_ent, "updated_entity": updated_ent}
    t_ent_terms = content_words(target_ent, kind="entity")
    u_ent_terms = content_words(updated_ent, kind="entity")
    if not t_ent_terms:
        return None, "target_entity_weak_content", {"target_entity": target_ent}
    if not u_ent_terms:
        return None, "updated_entity_weak_content", {"updated_entity": updated_ent}

    target_in_candidates, matched_candidates = entity_matches_candidate(t_ent_terms, prompt_rec.get("candidate_entities", []))
    if not target_in_candidates:
        return None, "target_entity_not_in_source_candidates", {"target_entity": target_ent, "target_terms": t_ent_terms, "candidates": prompt_rec.get("candidate_entities", [])}
    ok_target_in_source, target_source_hits = contains_content(source_sentence, t_ent_terms, need=1)
    if not ok_target_in_source:
        return None, "target_entity_not_in_source", {"target_entity": target_ent, "target_terms": t_ent_terms, "source_sentence": source_sentence}

    target_key = " ".join(t_ent_terms)
    updated_key = " ".join(u_ent_terms)
    if packet_type == "UPDATED_USE" and target_key != updated_key:
        return None, "updated_entity_mismatch", {"target_terms": t_ent_terms, "updated_terms": u_ent_terms}
    if packet_type == "UNCHANGED_DISTRACTOR_USE" and target_key == updated_key:
        return None, "distractor_same_entity", {"target_terms": t_ent_terms, "updated_terms": u_ent_terms}

    changed, state_info = state_changed(source_state, new_state)
    if not changed:
        return None, state_info.get("reason", "state_change_failed"), state_info

    source_terms = state_info["source_terms"]
    new_terms = state_info["new_terms"]
    source_specific = state_info["source_specific"]
    new_specific = state_info["new_specific"]

    # The declared source_state should be strongly grounded in the source sentence;
    # one shared word such as a person name or date is not enough.
    ok_source_ground, source_ground_hits = contains_content(source_sentence, source_terms, need=required_state_hits(source_terms))
    if not ok_source_ground:
        return None, "source_state_not_in_source_strong", {**state_info, "source_sentence": source_sentence, "source_ground_hits": source_ground_hits, "needed": required_state_hits(source_terms)}

    # Entity mentions use content words only, not articles or generic labels.
    ok_update_ent, upd_ent_hits = contains_content(update_sent, u_ent_terms, need=1)
    if not ok_update_ent:
        return None, "update_missing_entity_content", {"updated_terms": u_ent_terms, "update_sentence": update_sent}
    ok_use_ent, use_ent_hits = contains_content(use_sent, t_ent_terms, need=1)
    if not ok_use_ent:
        return None, "use_missing_target_content", {"target_terms": t_ent_terms, "use_sentence": use_sent}

    # Update sentence should carry the changed state for the updated entity.
    ok_update_state, update_state_hits = contains_content(update_sent, new_terms, need=1)
    if not ok_update_state:
        return None, "update_missing_new_state_content", {**state_info, "update_sentence": update_sent}

    # Use sentence should carry the right current state and avoid the wrong/stale one.
    if packet_type == "UPDATED_USE":
        ok_new_in_use, new_use_hits = contains_content(use_sent, new_terms)
        if not ok_new_in_use:
            return None, "use_missing_new_state_content", {**state_info, "use_sentence": use_sent}
        stale_terms = source_specific or source_terms
        stale_hits = [t for t in stale_terms if t in set(content_words(use_sent, kind="state"))]
        if stale_hits:
            return None, "use_contains_source_state_content", {**state_info, "stale_hits": stale_hits, "use_sentence": use_sent}
    else:
        ok_source_in_use, source_use_hits = contains_content(use_sent, source_terms)
        if not ok_source_in_use:
            return None, "use_missing_source_state_content", {**state_info, "use_sentence": use_sent}
        wrong_terms = new_specific or new_terms
        wrong_hits = [t for t in wrong_terms if t in set(content_words(use_sent, kind="state"))]
        if wrong_hits:
            return None, "use_contains_distractor_new_state_content", {**state_info, "wrong_hits": wrong_hits, "use_sentence": use_sent}
        new_use_hits = []
        source_use_hits = [t for t in source_terms if t in set(content_words(use_sent, kind="state"))]

    update_wc = wc(update_sent)
    use_wc = wc(use_sent)
    if update_wc < 6 or update_wc > 30:
        return None, f"update_wordcount:{update_wc}", {"update_sentence": update_sent}
    if use_wc < 4 or use_wc > 24:
        return None, f"use_wordcount:{use_wc}", {"use_sentence": use_sent}

    update_words = set(stem(w) for w in raw_words(update_sent))
    use_words = set(stem(w) for w in raw_words(use_sent))
    if update_words & BENCHMARK_VOCAB or use_words & BENCHMARK_VOCAB:
        return None, "benchmark_vocab", {}

    if update_sent.lower() == source_sentence.lower():
        return None, "update_copies_source", {}
    if use_sent.lower() == source_sentence.lower():
        return None, "use_copies_source", {}

    packet = {
        "pair_id": prompt_rec["pair_id"],
        "packet_type": packet_type,
        "source_sentence": source_sentence,
        "source_words": prompt_rec.get("source_words", wc(source_sentence)),
        "qwen_rewrite": prompt_rec.get("qwen_rewrite", ""),
        "rewrite_words": prompt_rec.get("rewrite_words", None),
        "target_entity": target_ent,
        "source_state": source_state,
        "updated_entity": updated_ent,
        "new_state": new_state,
        "update_sentence": update_sent,
        "use_sentence": use_sent,
        "update_words": update_wc,
        "use_words": use_wc,
        "target_entity_terms": t_ent_terms,
        "updated_entity_terms": u_ent_terms,
        "source_state_terms": source_terms,
        "new_state_terms": new_terms,
        "source_specific_terms": source_specific,
        "new_specific_terms": new_specific,
        "state_overlap_coef": state_info["overlap_coef"],
        "state_jaccard": state_info["jaccard"],
        "source_ground_hits": source_ground_hits,
        "matched_source_candidates": matched_candidates,
        "update_entity_hits": upd_ent_hits,
        "use_entity_hits": use_ent_hits,
        "update_state_hits": update_state_hits,
        "use_new_state_hits": new_use_hits if packet_type == "UPDATED_USE" else [],
        "use_source_state_hits": source_use_hits if packet_type == "UNCHANGED_DISTRACTOR_USE" else [],
        "raw_output": raw_text,
    }
    return packet, "", {}


def load_jsonl(path: pathlib.Path, *, tolerate_partial: bool = False) -> list[dict[str, Any]]:
    rows = []
    if not path.exists():
        return rows
    with path.open() as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                if tolerate_partial:
                    continue
                raise RuntimeError(f"JSON parse failed in {path} line {line_no}")
    return rows


def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def process_outputs(raw_path: pathlib.Path, prompt_path: pathlib.Path, *, source_label: str, tolerate_partial: bool) -> tuple[dict[str, dict], Counter, list[dict[str, Any]], dict[str, Any]]:
    prompts = load_jsonl(prompt_path)
    raws = load_jsonl(raw_path, tolerate_partial=tolerate_partial)
    accepted: dict[str, dict] = {}
    rejections = Counter()
    rejection_examples: list[dict[str, Any]] = []
    duplicate_pairs = 0
    for rec in raws:
        idx = rec.get("index")
        if not isinstance(idx, int) or idx < 0 or idx >= len(prompts):
            rejections["index_out_of_range"] += 1
            continue
        prompt_rec = prompts[idx]
        global_idx = prompt_rec.get("original_prompt_index", idx)
        raw_output = rec.get("output", rec.get("generated_text", ""))
        pkt, reason, detail = validate_output(raw_output, prompt_rec)
        if pkt is None:
            rejections[reason] += 1
            if len(rejection_examples) < 200:
                rejection_examples.append({
                    "source_label": source_label,
                    "index": idx,
                    "original_prompt_index": global_idx,
                    "pair_id": prompt_rec.get("pair_id"),
                    "packet_type": prompt_rec.get("packet_type"),
                    "reason": reason,
                    "detail": detail,
                    "output": raw_output,
                    "source_sentence": prompt_rec.get("source_sentence"),
                })
            continue
        pkt["generation_source"] = source_label
        pkt["local_output_index"] = idx
        pkt["original_prompt_index"] = global_idx
        if pkt["pair_id"] in accepted:
            duplicate_pairs += 1
            continue
        accepted[pkt["pair_id"]] = pkt
    info = {
        "source_label": source_label,
        "raw_path": str(raw_path),
        "prompt_path": str(prompt_path),
        "prompt_records": len(prompts),
        "raw_outputs": len(raws),
        "accepted": len(accepted),
        "duplicate_pairs": duplicate_pairs,
        "rejections": dict(rejections.most_common()),
    }
    return accepted, rejections, rejection_examples, info


def write_jsonl(path: pathlib.Path, rows: list[dict[str, Any]]):
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def split_train_heldout(valid_rows: list[dict[str, Any]], heldout_target_total: int) -> tuple[list[dict], list[dict]]:
    rng = random.Random(RNG_SEED)
    by_type: dict[str, list[dict]] = defaultdict(list)
    for r in valid_rows:
        by_type[r["packet_type"]].append(r)
    heldout: list[dict] = []
    per_type_target = heldout_target_total // 2
    for typ in ["UPDATED_USE", "UNCHANGED_DISTRACTOR_USE"]:
        rows = by_type.get(typ, [])
        rng.shuffle(rows)
        take = min(per_type_target, max(0, len(rows) // 5), len(rows))
        heldout.extend(rows[:take])
    heldout_ids = {r["pair_id"] for r in heldout}
    train = [r for r in valid_rows if r["pair_id"] not in heldout_ids]
    return train, heldout


def write_review_sample(path: pathlib.Path, rows: list[dict[str, Any]], n: int):
    rng = random.Random(RNG_SEED + 1)
    sample = rows[:]
    rng.shuffle(sample)
    sample = sample[:min(n, len(sample))]
    lines = []
    lines.append("# research Accepted Packet Manual Read Sample\n\n")
    lines.append(f"Sampled {len(sample)} accepted packets from the strict lexical validator. Read these before relying on acceptance rate.\n\n")
    for i, r in enumerate(sample, start=1):
        lines.append(f"## {i}. {r['pair_id']} — {r['packet_type']}\n\n")
        lines.append(f"SOURCE: {r['source_sentence']}\n\n")
        lines.append(f"TARGET: {r['target_entity']} | UPDATED: {r['updated_entity']}\n\n")
        lines.append(f"SOURCE_STATE: {r['source_state']} | NEW_STATE: {r['new_state']}\n\n")
        lines.append(f"UPDATE: {r['update_sentence']}\n\n")
        lines.append(f"USE: {r['use_sentence']}\n\n")
        lines.append(f"STATE_TERMS source={r['source_state_terms']} new={r['new_state_terms']} hits_update={r['update_state_hits']} hits_use_new={r['use_new_state_hits']} hits_use_source={r['use_source_state_hits']}\n\n")
    path.write_text("".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--include-continuation", action="store_true", help="Include the research continuation output file if present")
    ap.add_argument("--include-forced-updated", action="store_true", help="Include the research forced UPDATED_USE output file if present; valid forced packets override earlier UPDATED_USE packets for the same pair")
    ap.add_argument("--tolerate-partial", action="store_true", help="Skip partial JSON lines, useful only for non-final inspection")
    ap.add_argument("--heldout-total", type=int, default=HELDOUT_TARGET_TOTAL)
    ap.add_argument("--review-sample", type=int, default=50)
    args = ap.parse_args()

    t0 = time.time()
    merged: dict[str, dict] = {}
    all_rejections = Counter()
    all_examples: list[dict[str, Any]] = []
    source_infos = []

    main_acc, main_rej, main_ex, main_info = process_outputs(MAIN_RAW, FULL_PROMPTS, source_label="main", tolerate_partial=args.tolerate_partial)
    merged.update(main_acc)
    all_rejections.update(main_rej)
    all_examples.extend(main_ex)
    source_infos.append(main_info)

    if args.include_continuation and CONT_RAW.exists():
        cont_acc, cont_rej, cont_ex, cont_info = process_outputs(CONT_RAW, CONT_PROMPTS, source_label="continuation", tolerate_partial=args.tolerate_partial)
        # Continuation should cover pair ids not in main; keep main if duplicate.
        dup = 0
        for pid, pkt in cont_acc.items():
            if pid in merged:
                dup += 1
            else:
                merged[pid] = pkt
        cont_info["duplicates_against_merged"] = dup
        all_rejections.update(cont_rej)
        all_examples.extend(cont_ex)
        source_infos.append(cont_info)

    if args.include_forced_updated and FORCED_UPDATED_RAW.exists():
        forced_acc, forced_rej, forced_ex, forced_info = process_outputs(FORCED_UPDATED_RAW, FORCED_UPDATED_PROMPTS, source_label="forced_updated", tolerate_partial=args.tolerate_partial)
        overwritten = 0
        added = 0
        for pid, pkt in forced_acc.items():
            # Forced UPDATED_USE prompts were built to place the new-state phrase in
            # both update and use; prefer them over an earlier looser UPDATED_USE.
            if pid in merged:
                overwritten += 1
            else:
                added += 1
            merged[pid] = pkt
        forced_info["overwritten_or_duplicate_pairs"] = overwritten
        forced_info["added_pairs"] = added
        all_rejections.update(forced_rej)
        all_examples.extend(forced_ex)
        source_infos.append(forced_info)

    valid_rows = sorted(merged.values(), key=lambda r: int(r.get("original_prompt_index", 10**12)))
    train_rows, heldout_rows = split_train_heldout(valid_rows, args.heldout_total)

    valid_path = OUT_DIR / "validated_state_use_packets_strict_all.jsonl"
    train_path = OUT_DIR / "validated_state_use_packets_strict_train.jsonl"
    heldout_path = OUT_DIR / "validated_state_use_packets_strict_heldout.jsonl"
    reject_path = OUT_DIR / "validation_rejection_examples_step044.jsonl"
    review_path = OUT_DIR / "accepted_packet_manual_read_sample_revision_044.md"
    meta_path = OUT_DIR / "validation_metadata_strict_revision_044.json"

    write_jsonl(valid_path, valid_rows)
    write_jsonl(train_path, train_rows)
    write_jsonl(heldout_path, heldout_rows)
    write_jsonl(reject_path, all_examples[:500])
    write_review_sample(review_path, valid_rows, args.review_sample)

    type_counts = Counter(r["packet_type"] for r in valid_rows)
    train_type_counts = Counter(r["packet_type"] for r in train_rows)
    heldout_type_counts = Counter(r["packet_type"] for r in heldout_rows)
    raw_total = sum(info["raw_outputs"] for info in source_infos)
    meta = {
        "status": "STRICT_STATE_USE_VALIDATION_DONE",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "include_continuation": args.include_continuation,
        "include_forced_updated": args.include_forced_updated,
        "source_infos": source_infos,
        "raw_outputs_total": raw_total,
        "valid_packets_total": len(valid_rows),
        "valid_rate_vs_outputs": round(len(valid_rows) / max(1, raw_total), 4),
        "type_counts_all": dict(type_counts),
        "type_counts_train": dict(train_type_counts),
        "type_counts_heldout": dict(heldout_type_counts),
        "heldout_total_requested": args.heldout_total,
        "heldout_total_actual": len(heldout_rows),
        "train_total_actual": len(train_rows),
        "rejection_reasons": dict(all_rejections.most_common()),
        "paths": {
            "all_validated": str(valid_path),
            "train_packets": str(train_path),
            "heldout_packets": str(heldout_path),
            "rejection_examples": str(reject_path),
            "manual_read_sample": str(review_path),
        },
        "sha256": {
            "all_validated": sha256_file(valid_path),
            "train_packets": sha256_file(train_path),
            "heldout_packets": sha256_file(heldout_path),
        },
        "elapsed_sec": round(time.time() - t0, 2),
        "note": "Acceptance rate alone is not evidence of packet quality; read the sample and use held-out margin probes before interpreting training results.",
    }
    meta_path.write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2), flush=True)


if __name__ == "__main__":
    main()
