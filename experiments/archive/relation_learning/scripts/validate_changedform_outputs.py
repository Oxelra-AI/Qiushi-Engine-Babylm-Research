#!/usr/bin/env python3
"""research: validate changed-form natural state/use generation outputs.

This validator supersedes the research strict validator for the practical training
stream.  It keeps the useful state-content tests but adds the missing condition:
the final use sentence must express the relevant state in changed form rather
than copying the source sentence (or, for UPDATED_USE, the update sentence).

The output is a balanced candidate pool: train and held-out splits contain at
most the same number of UPDATED_USE and UNCHANGED_DISTRACTOR_USE packets so the
arm tests supersession plus protected retention rather than re-running DUP.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import pathlib
import random
import re
import statistics
import time
from collections import Counter, defaultdict
from typing import Any, Iterable

ROOT = pathlib.Path.cwd()
DATA = ROOT / "experiments/archive/relation_learning/data/state_use_generation"
FULL_PROMPTS = DATA / "state_use_prompts_9b.jsonl"
UPDATED_PROMPTS = DATA / "state_use_prompts_9b_updated_changedform_step045.jsonl"
DISTRACTOR_PROMPTS = DATA / "state_use_prompts_9b_distractor_changedform_step045.jsonl"
UPDATED_PILOT_PROMPTS = DATA / "state_use_prompts_9b_updated_changedform_step045_pilot256.jsonl"
DISTRACTOR_PILOT_PROMPTS = DATA / "state_use_prompts_9b_distractor_changedform_step045_pilot256.jsonl"
UPDATED_RAW = DATA / "raw_outputs_9b_updated_changedform_step045.jsonl"
DISTRACTOR_RAW = DATA / "raw_outputs_9b_distractor_changedform_step045.jsonl"
UPDATED_PILOT_RAW = DATA / "raw_outputs_9b_updated_changedform_step045_pilot256.jsonl"
DISTRACTOR_PILOT_RAW = DATA / "raw_outputs_9b_distractor_changedform_step045_pilot256.jsonl"

OUT_DIR = DATA
WORD_RE = re.compile(r"[A-Za-z0-9]+(?:[\u2019'][A-Za-z0-9]+)?")
REQUIRED_KEYS = {"target_entity", "source_state", "updated_entity", "new_state", "update_sentence", "use_sentence"}
BENCHMARK_VOCAB = {"box", "basket", "marble", "container", "containers", "baskets", "boxes", "marbles"}
RNG_SEED = 45045
HELDOUT_PER_TYPE_DEFAULT = 500
MAX_SOURCE_USE_CONTENT_JACCARD = 0.80
MAX_SOURCE_USE_OVERLAP_MIN = 0.94
MAX_SOURCE_USE_LCS = 5
MAX_UPDATE_USE_LCS_UPDATED = 5

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "being", "but", "by", "can", "could",
    "did", "do", "does", "for", "from", "had", "has", "have", "he", "her", "hers", "him", "his",
    "i", "if", "in", "into", "is", "it", "its", "just", "may", "might", "must", "not", "of", "on",
    "or", "our", "she", "should", "so", "some", "such", "than", "that", "the", "their", "them",
    "then", "there", "these", "they", "this", "to", "too", "under", "up", "very", "was", "we",
    "were", "what", "when", "where", "which", "who", "will", "with", "would", "you", "your",
    "before", "after", "while", "over", "about", "also", "through", "each", "every", "all", "any",
    "now", "still", "continues", "continue", "continued", "continually", "currently", "former", "formerly",
    "one", "only", "other", "same", "new", "old", "current", "original", "different",
}

ENTITY_GENERIC = {
    "mr", "mrs", "ms", "miss", "dr", "sir", "madam", "man", "woman", "person", "people", "child",
    "children", "boy", "girl", "men", "women", "city", "town", "country", "state", "province", "village",
    "place", "area", "region", "room", "house", "home", "building", "site", "part", "way", "thing", "things",
    "group", "team", "company", "organization", "institution", "government", "system", "event", "story",
    "january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november", "december",
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday", "mot", "chi", "mommy", "daddy",
    "street", "road", "roads", "forest", "entrance", "cover", "meal", "run", "command", "dialog",
}

STATE_GENERIC = {
    "state", "status", "condition", "role", "position", "place", "location", "site", "area", "region", "part",
    "located", "situated", "based", "living", "lives", "live", "lived", "resides", "reside", "resided",
    "working", "works", "worked", "serving", "serves", "served", "became", "become", "becomes", "being",
    "having", "remains", "remain", "keeps", "kept", "stays", "stayed", "near", "meeting", "office",
    "home", "house", "room", "local", "main", "central", "around", "along", "month", "months", "date",
    "day", "days", "year", "years",
}


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


def norm_tokens(text: str) -> list[str]:
    return [stem(w) for w in raw_words(text)]


def content_words(text: str, *, kind: str = "state") -> list[str]:
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
    seen = set(); uniq = []
    for w in out:
        if w not in seen:
            seen.add(w); uniq.append(w)
    return uniq


def contains_content(sentence: str, terms: list[str], *, need: int | None = None) -> tuple[bool, list[str]]:
    sent = set(content_words(sentence, kind="state")) | set(content_words(sentence, kind="entity"))
    hits = [t for t in terms if t in sent]
    if need is None:
        need = 1 if len(terms) <= 2 else 2
    return len(hits) >= min(need, len(terms)), hits


def required_state_hits(terms: list[str]) -> int:
    if len(terms) <= 1:
        return 1
    return min(len(terms), max(2, (len(terms) + 1) // 2))


def entity_matches_candidate(entity_terms: list[str], candidates: list[Any]) -> tuple[bool, list[str]]:
    if not entity_terms:
        return False, []
    e = set(entity_terms)
    matched = []
    for cand in candidates or []:
        cterms = content_words(str(cand), kind="entity")
        if not cterms:
            continue
        inter = e & set(cterms)
        if inter and (len(e) == 1 or len(cterms) == 1 or len(inter) >= min(2, len(e), len(cterms))):
            matched.append(str(cand))
    return bool(matched), matched


def jaccard_words(a: Iterable[str], b: Iterable[str]) -> float:
    A, B = set(a), set(b)
    if not (A or B):
        return 0.0
    return len(A & B) / len(A | B)


def overlap_min(a: Iterable[str], b: Iterable[str]) -> float:
    A, B = set(a), set(b)
    if not A or not B:
        return 0.0
    return len(A & B) / min(len(A), len(B))


def longest_common_contiguous(a: list[str], b: list[str]) -> tuple[int, str]:
    best = 0; best_i = 0
    dp = [0] * (len(b) + 1)
    for i, x in enumerate(a, start=1):
        ndp = [0] * (len(b) + 1)
        for j, y in enumerate(b, start=1):
            if x == y:
                ndp[j] = dp[j - 1] + 1
                if ndp[j] > best:
                    best = ndp[j]
                    best_i = i - best
        dp = ndp
    return best, " ".join(a[best_i:best_i+best]) if best else ""


def exact_norm(a: str, b: str) -> bool:
    return " ".join(norm_tokens(a)) == " ".join(norm_tokens(b))


def is_substring_norm(needle: str, haystack: str) -> bool:
    n = " ".join(norm_tokens(needle)); h = " ".join(norm_tokens(haystack))
    return bool(n) and n in h


def surface_metrics(a: str, b: str) -> dict[str, Any]:
    ta, tb = norm_tokens(a), norm_tokens(b)
    ca, cb = content_words(a), content_words(b)
    lcs, span = longest_common_contiguous(ta, tb)
    return {
        "all_token_jaccard": jaccard_words(ta, tb),
        "content_jaccard": jaccard_words(ca, cb),
        "content_overlap_min": overlap_min(ca, cb),
        "longest_common_contiguous_words": lcs,
        "longest_common_span": span,
        "exact_norm": exact_norm(a, b),
        "b_is_substring_of_a_norm": is_substring_norm(b, a),
    }


def state_changed(source_state: str, new_state: str) -> tuple[bool, dict[str, Any]]:
    s = content_words(source_state, kind="state")
    n = content_words(new_state, kind="state")
    if not s:
        return False, {"reason": "source_state_weak_content", "source_terms": s, "new_terms": n}
    if not n:
        return False, {"reason": "new_state_weak_content", "source_terms": s, "new_terms": n}
    if len(n) < 2:
        return False, {"reason": "new_state_too_few_content_words", "source_terms": s, "new_terms": n}
    ss, ns = set(s), set(n)
    inter = ss & ns
    overlap_coef = len(inter) / max(1, min(len(ss), len(ns)))
    jacc = len(inter) / max(1, len(ss | ns))
    new_specific = sorted(ns - ss)
    source_specific = sorted(ss - ns)
    if not new_specific:
        return False, {"reason": "new_state_no_new_content", "source_terms": s, "new_terms": n, "overlap_coef": overlap_coef, "jaccard": jacc}
    if overlap_coef >= 0.67 and jacc >= 0.50:
        return False, {"reason": "state_content_too_similar", "source_terms": s, "new_terms": n, "overlap_coef": overlap_coef, "jaccard": jacc}
    return True, {"source_terms": s, "new_terms": n, "source_specific": source_specific, "new_specific": new_specific, "overlap_coef": overlap_coef, "jaccard": jacc}


def parse_output(raw_text: str) -> tuple[dict[str, Any] | None, str]:
    txt = (raw_text or "").strip()
    if txt.startswith("```"):
        txt = "\n".join(l for l in txt.splitlines() if not l.strip().startswith("```"))
    start = txt.find("{"); end = txt.rfind("}") + 1
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

    if re.search(r"\*[A-Z]{2,5}:", source_sentence):
        return None, "transcript_marker_in_source", {"source_sentence": source_sentence}
    if "*" in target_ent or "*" in updated_ent:
        return None, "transcript_speaker_label_entity", {"target_entity": target_ent, "updated_entity": updated_ent}
    t_terms = content_words(target_ent, kind="entity")
    u_terms = content_words(updated_ent, kind="entity")
    if not t_terms:
        return None, "target_entity_weak_content", {"target_entity": target_ent}
    if not u_terms:
        return None, "updated_entity_weak_content", {"updated_entity": updated_ent}
    ok_cand, matched = entity_matches_candidate(t_terms, prompt_rec.get("candidate_entities", []))
    if not ok_cand:
        return None, "target_entity_not_in_source_candidates", {"target_entity": target_ent, "target_terms": t_terms, "candidates": prompt_rec.get("candidate_entities", [])}
    ok_tsrc, target_source_hits = contains_content(source_sentence, t_terms, need=1)
    if not ok_tsrc:
        return None, "target_entity_not_in_source", {"target_terms": t_terms}
    target_key = " ".join(t_terms); updated_key = " ".join(u_terms)
    if packet_type == "UPDATED_USE" and target_key != updated_key:
        return None, "updated_entity_mismatch", {"target_terms": t_terms, "updated_terms": u_terms}
    if packet_type == "UNCHANGED_DISTRACTOR_USE" and target_key == updated_key:
        return None, "distractor_same_entity", {"target_terms": t_terms, "updated_terms": u_terms}

    changed, st = state_changed(source_state, new_state)
    if not changed:
        return None, st.get("reason", "state_change_failed"), st
    source_terms = st["source_terms"]; new_terms = st["new_terms"]
    source_specific = st["source_specific"]; new_specific = st["new_specific"]
    if packet_type == "UNCHANGED_DISTRACTOR_USE" and len(source_terms) < 2:
        return None, "distractor_source_state_too_few_content_words", st

    ok_ground, ground_hits = contains_content(source_sentence, source_terms, need=required_state_hits(source_terms))
    if not ok_ground:
        return None, "source_state_not_in_source_strong", {**st, "source_sentence": source_sentence, "source_ground_hits": ground_hits, "needed": required_state_hits(source_terms)}
    ok_up_ent, up_ent_hits = contains_content(update_sent, u_terms, need=1)
    if not ok_up_ent:
        return None, "update_missing_entity_content", {"updated_terms": u_terms, "update_sentence": update_sent}
    ok_use_ent, use_ent_hits = contains_content(use_sent, t_terms, need=1)
    if not ok_use_ent:
        return None, "use_missing_target_content", {"target_terms": t_terms, "use_sentence": use_sent}
    ok_up_state, up_state_hits = contains_content(update_sent, new_terms, need=1)
    if not ok_up_state:
        return None, "update_missing_new_state_content", {**st, "update_sentence": update_sent}

    if packet_type == "UPDATED_USE":
        ok_new_use, new_use_hits = contains_content(use_sent, new_terms)
        if not ok_new_use:
            return None, "use_missing_new_state_content", {**st, "use_sentence": use_sent}
        stale_terms = source_specific or source_terms
        stale_hits = [t for t in stale_terms if t in set(content_words(use_sent, kind="state"))]
        if stale_hits:
            return None, "use_contains_source_state_content", {**st, "stale_hits": stale_hits, "use_sentence": use_sent}
        source_use_hits: list[str] = []
    else:
        ok_source_use, source_use_hits = contains_content(use_sent, source_terms)
        if not ok_source_use:
            return None, "use_missing_source_state_content", {**st, "use_sentence": use_sent}
        wrong_terms = new_specific or new_terms
        wrong_hits = [t for t in wrong_terms if t in set(content_words(use_sent, kind="state"))]
        if wrong_hits:
            return None, "use_contains_distractor_new_state_content", {**st, "wrong_hits": wrong_hits, "use_sentence": use_sent}
        new_use_hits = []

    uwc = wc(update_sent); vwc = wc(use_sent)
    if uwc < 6 or uwc > 30:
        return None, f"update_wordcount:{uwc}", {"update_sentence": update_sent}
    if vwc < 4 or vwc > 24:
        return None, f"use_wordcount:{vwc}", {"use_sentence": use_sent}
    update_wordset = set(norm_tokens(update_sent)); use_wordset = set(norm_tokens(use_sent))
    if update_wordset & BENCHMARK_VOCAB or use_wordset & BENCHMARK_VOCAB:
        return None, "benchmark_vocab", {}

    us = surface_metrics(source_sentence, use_sent)
    uu = surface_metrics(update_sent, use_sent)
    # Reject source-copy or near-copy final use.  This is stricter than the COMPACT_EXPERIENCE Qwen
    # copy rule because the use sentence is short; a six-token source span is already
    # a direct recovery route under WWM.
    if us["exact_norm"]:
        return None, "changedform_use_exact_source_copy", us
    if us["b_is_substring_of_a_norm"]:
        return None, "changedform_use_source_substring", us
    if us["longest_common_contiguous_words"] > MAX_SOURCE_USE_LCS:
        return None, "changedform_use_source_lcs_gt5", us
    if us["content_overlap_min"] > MAX_SOURCE_USE_OVERLAP_MIN and us["content_jaccard"] > 0.80:
        return None, "changedform_use_source_copy_overlap", us
    # Also reject exact source_state phrase reuse in the final use; it is a direct source fragment.
    if is_substring_norm(source_state, use_sent) and wc(source_state) >= 3:
        return None, "changedform_use_repeats_source_state_phrase", {"source_state": source_state, "use_sentence": use_sent, **us}
    if packet_type == "UPDATED_USE":
        if uu["exact_norm"]:
            return None, "changedform_use_exact_update_copy", uu
        if uu["b_is_substring_of_a_norm"]:
            return None, "changedform_use_update_substring", uu
        if uu["longest_common_contiguous_words"] > MAX_UPDATE_USE_LCS_UPDATED:
            return None, "changedform_use_update_lcs_gt5", uu

    pkt = {
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
        "update_words": uwc,
        "use_words": vwc,
        "target_entity_terms": t_terms,
        "updated_entity_terms": u_terms,
        "source_state_terms": source_terms,
        "new_state_terms": new_terms,
        "source_specific_terms": source_specific,
        "new_specific_terms": new_specific,
        "state_overlap_coef": st["overlap_coef"],
        "state_jaccard": st["jaccard"],
        "source_ground_hits": ground_hits,
        "matched_source_candidates": matched,
        "update_entity_hits": up_ent_hits,
        "use_entity_hits": use_ent_hits,
        "update_state_hits": up_state_hits,
        "use_new_state_hits": new_use_hits if packet_type == "UPDATED_USE" else [],
        "use_source_state_hits": source_use_hits if packet_type == "UNCHANGED_DISTRACTOR_USE" else [],
        "surface": {
            "use_vs_source": us,
            "use_vs_update": uu,
        },
        "raw_output": raw_text,
    }
    return pkt, "", {}


def load_jsonl(path: pathlib.Path, *, tolerate_partial: bool = False) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
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


def process_outputs(raw_path: pathlib.Path, prompt_path: pathlib.Path, label: str, tolerate_partial: bool) -> tuple[dict[str, dict], Counter, list[dict], dict]:
    prompts = load_jsonl(prompt_path)
    raws = load_jsonl(raw_path, tolerate_partial=tolerate_partial)
    acc: dict[str, dict] = {}
    rej = Counter(); examples = []
    for rec in raws:
        idx = rec.get("index")
        if not isinstance(idx, int) or idx < 0 or idx >= len(prompts):
            rej["index_out_of_range"] += 1
            continue
        p = prompts[idx]
        global_idx = p.get("original_prompt_index", idx)
        pkt, reason, detail = validate_output(rec.get("output", rec.get("generated_text", "")), p)
        if pkt is None:
            rej[reason] += 1
            if len(examples) < 300:
                examples.append({"source_label": label, "index": idx, "original_prompt_index": global_idx, "pair_id": p.get("pair_id"), "packet_type": p.get("packet_type"), "reason": reason, "detail": detail, "output": rec.get("output", ""), "source_sentence": p.get("source_sentence")})
            continue
        pkt["generation_source"] = label
        pkt["local_output_index"] = idx
        pkt["original_prompt_index"] = global_idx
        # There should be no duplicates within one file, but keep first by prompt order.
        acc.setdefault(pkt["pair_id"], pkt)
    info = {"source_label": label, "raw_path": str(raw_path), "prompt_path": str(prompt_path), "prompt_records": len(prompts), "raw_outputs": len(raws), "accepted": len(acc), "rejections": dict(rej.most_common())}
    return acc, rej, examples, info


def write_jsonl(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def balanced_split(rows: list[dict[str, Any]], *, heldout_per_type: int) -> tuple[list[dict], list[dict], list[dict]]:
    rng = random.Random(RNG_SEED)
    by_type: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_type[r["packet_type"]].append(r)
    for typ in by_type:
        by_type[typ].sort(key=lambda r: int(r.get("original_prompt_index", 10**12)))
        rng.shuffle(by_type[typ])
    n_bal = min(len(by_type.get("UPDATED_USE", [])), len(by_type.get("UNCHANGED_DISTRACTOR_USE", [])))
    balanced = []
    for typ in ["UPDATED_USE", "UNCHANGED_DISTRACTOR_USE"]:
        balanced.extend(by_type.get(typ, [])[:n_bal])
    heldout = []
    train = []
    for typ in ["UPDATED_USE", "UNCHANGED_DISTRACTOR_USE"]:
        chosen = by_type.get(typ, [])[:n_bal]
        take_h = min(heldout_per_type, max(0, len(chosen)//5), len(chosen))
        heldout.extend(chosen[:take_h])
        train.extend(chosen[take_h:])
    heldout_ids = {r["pair_id"] for r in heldout}
    train = [r for r in train if r["pair_id"] not in heldout_ids]
    balanced.sort(key=lambda r: (int(r.get("original_prompt_index", 10**12)), r["packet_type"]))
    train.sort(key=lambda r: (int(r.get("original_prompt_index", 10**12)), r["packet_type"]))
    heldout.sort(key=lambda r: (int(r.get("original_prompt_index", 10**12)), r["packet_type"]))
    return balanced, train, heldout


def pct(vals: list[float], q: float) -> float | None:
    if not vals:
        return None
    vals = sorted(vals)
    if len(vals) == 1:
        return vals[0]
    pos = (len(vals)-1)*q; lo=math.floor(pos); hi=math.ceil(pos)
    return vals[lo] if lo==hi else vals[lo]*(hi-pos)+vals[hi]*(pos-lo)


def stat(vals: list[float]) -> dict[str, Any]:
    if not vals:
        return {"n":0,"min":None,"p05":None,"mean":None,"median":None,"p95":None,"max":None}
    return {"n":len(vals),"min":round(min(vals),4),"p05":round(pct(vals,0.05),4),"mean":round(statistics.mean(vals),4),"median":round(statistics.median(vals),4),"p95":round(pct(vals,0.95),4),"max":round(max(vals),4)}


def surface_summary(rows: list[dict]) -> dict[str, Any]:
    vals = lambda key: [float(r["surface"]["use_vs_source"][key]) for r in rows]
    lcs = [float(r["surface"]["use_vs_source"]["longest_common_contiguous_words"]) for r in rows]
    n = len(rows) or 1
    return {
        "use_source_content_jaccard": stat(vals("content_jaccard")),
        "use_source_overlap_min": stat(vals("content_overlap_min")),
        "use_source_lcs": stat(lcs),
        "frac_lcs_ge6": round(sum(x >= 6 for x in lcs)/n, 4),
        "frac_content_jaccard_ge0p8": round(sum(x >= 0.8 for x in vals("content_jaccard"))/n, 4),
    }


def write_review(path: pathlib.Path, rows: list[dict], n_each: int = 25) -> None:
    rng = random.Random(RNG_SEED + 3)
    by = defaultdict(list)
    for r in rows:
        by[r["packet_type"]].append(r)
    lines = ["# research accepted changed-form packet sample\n\n", "Read this before assembling a training stream. These examples passed deterministic state-content and changed-form filters.\n\n"]
    for typ in ["UPDATED_USE", "UNCHANGED_DISTRACTOR_USE"]:
        xs = by.get(typ, [])[:]
        rng.shuffle(xs)
        lines.append(f"## {typ}\n\n")
        for i, r in enumerate(xs[:min(n_each, len(xs))], 1):
            us = r["surface"]["use_vs_source"]
            uu = r["surface"]["use_vs_update"]
            lines.append(f"### {i}. {r['pair_id']}\n\n")
            lines.append(f"SOURCE: {r['source_sentence']}\n\n")
            lines.append(f"TARGET: {r['target_entity']} | UPDATED: {r['updated_entity']}\n\n")
            lines.append(f"SOURCE_STATE: {r['source_state']} | NEW_STATE: {r['new_state']}\n\n")
            lines.append(f"UPDATE: {r['update_sentence']}\n\n")
            lines.append(f"USE: {r['use_sentence']}\n\n")
            lines.append(f"SURFACE use-source: J={us['content_jaccard']:.3f}, overlap_min={us['content_overlap_min']:.3f}, LCS={us['longest_common_contiguous_words']} `{us['longest_common_span']}`; use-update LCS={uu['longest_common_contiguous_words']} `{uu['longest_common_span']}`\n\n")
            lines.append(f"STATE_TERMS source={r['source_state_terms']} new={r['new_state_terms']} hits_update={r['update_state_hits']} hits_use_new={r['use_new_state_hits']} hits_use_source={r['use_source_state_hits']}\n\n")
    path.write_text("".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["pilot", "full"], default="pilot")
    ap.add_argument("--tolerate-partial", action="store_true")
    ap.add_argument("--heldout-per-type", type=int, default=HELDOUT_PER_TYPE_DEFAULT)
    args = ap.parse_args()
    sources = []
    if args.mode == "pilot":
        sources = [
            (UPDATED_PILOT_RAW, UPDATED_PILOT_PROMPTS, "updated_pilot"),
            (DISTRACTOR_PILOT_RAW, DISTRACTOR_PILOT_PROMPTS, "distractor_pilot"),
        ]
        stem_name = "changedform_pilot"
    else:
        sources = [
            (UPDATED_RAW, UPDATED_PROMPTS, "updated_full"),
            (DISTRACTOR_RAW, DISTRACTOR_PROMPTS, "distractor_full"),
        ]
        stem_name = "changedform_full"

    t0 = time.time(); merged = {}; rejs = Counter(); exs = []; infos = []
    for raw, prompts, label in sources:
        acc, rej, ex, info = process_outputs(raw, prompts, label, args.tolerate_partial)
        infos.append(info); rejs.update(rej); exs.extend(ex)
        for pid, pkt in acc.items():
            if pid not in merged:
                merged[pid] = pkt
    valid = sorted(merged.values(), key=lambda r: (int(r.get("original_prompt_index", 10**12)), r["packet_type"]))
    balanced, train, heldout = balanced_split(valid, heldout_per_type=args.heldout_per_type)

    all_path = OUT_DIR / f"validated_state_use_packets_{stem_name}_all.jsonl"
    bal_path = OUT_DIR / f"validated_state_use_packets_{stem_name}_balanced.jsonl"
    train_path = OUT_DIR / f"validated_state_use_packets_{stem_name}_train.jsonl"
    heldout_path = OUT_DIR / f"validated_state_use_packets_{stem_name}_heldout.jsonl"
    rej_path = OUT_DIR / f"validation_rejection_examples_{stem_name}_step045.jsonl"
    review_path = OUT_DIR / f"accepted_packet_manual_read_sample_{stem_name}_step045.md"
    meta_path = OUT_DIR / f"validation_metadata_{stem_name}_step045.json"
    write_jsonl(all_path, valid); write_jsonl(bal_path, balanced); write_jsonl(train_path, train); write_jsonl(heldout_path, heldout); write_jsonl(rej_path, exs[:600])
    write_review(review_path, balanced if balanced else valid, n_each=50)
    by_type = Counter(r["packet_type"] for r in valid)
    bal_by_type = Counter(r["packet_type"] for r in balanced)
    meta = {
        "status": "CHANGEDFORM_VALIDATION_DONE",
        "mode": args.mode,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source_infos": infos,
        "raw_outputs_total": sum(i["raw_outputs"] for i in infos),
        "valid_packets_total": len(valid),
        "valid_rate_vs_outputs": round(len(valid)/max(1,sum(i["raw_outputs"] for i in infos)),4),
        "type_counts_all": dict(by_type),
        "balanced_total": len(balanced),
        "type_counts_balanced": dict(bal_by_type),
        "train_total": len(train),
        "heldout_total": len(heldout),
        "train_type_counts": dict(Counter(r["packet_type"] for r in train)),
        "heldout_type_counts": dict(Counter(r["packet_type"] for r in heldout)),
        "surface_summary_all": surface_summary(valid),
        "surface_summary_balanced": surface_summary(balanced),
        "rejection_reasons": dict(rejs.most_common()),
        "paths": {"all": str(all_path), "balanced": str(bal_path), "train": str(train_path), "heldout": str(heldout_path), "rejections": str(rej_path), "manual_read_sample": str(review_path)},
        "sha256": {"all": sha256_file(all_path), "balanced": sha256_file(bal_path), "train": sha256_file(train_path), "heldout": sha256_file(heldout_path)},
        "thresholds": {"max_source_use_lcs": MAX_SOURCE_USE_LCS, "max_update_use_lcs_updated": MAX_UPDATE_USE_LCS_UPDATED, "max_source_use_content_jaccard": MAX_SOURCE_USE_CONTENT_JACCARD, "max_source_use_overlap_min": MAX_SOURCE_USE_OVERLAP_MIN, "new_state_min_content_words": 2, "distractor_source_state_min_content_words": 2, "reject_transcript_marker_in_source": True},
        "elapsed_sec": round(time.time()-t0,2),
        "note": "Train only balanced changed-form packets after manual reading; research copy-like accepted packets are not used.",
    }
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(meta, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
