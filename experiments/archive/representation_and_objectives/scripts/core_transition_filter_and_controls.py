#!/usr/bin/env python3
"""research: sharpen the research transition substrate and build length-matched controls.

This script is CPU-only and uses only existing allowed training reservoirs.  It does
not read official evaluation item text and it does not launch training.  The goal is
to turn the broad research transition candidates into a cleaner, inspectable substrate
for a possible later *small* matched probe if the running FineWeb compact/breadth
experiment fails to move relation-conditioned errors.
"""
from __future__ import annotations

import csv
import json
import math
import re
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

USER_ROOT = Path.cwd()
A01_WS = USER_ROOT / "experiments/archive/representation_and_objectives"
OUT = A01_WS / "data/core_transition_filter"
NOTE = A01_WS / "notes/core_transition_filter_and_controls.md"
CANDIDATES = A01_WS / "data/strict_transition_substrate/strict_transition_candidates.jsonl"

BUDGETS = [50_000, 100_000, 200_000]
PRIMARY_BUCKETS = [
    "physical_material_transition",
    "spatial_transition",
    "temporal_quantity_transition",
    "affordance_procedure_transition",
    "explicit_contrast_transition",
]

RESERVOIRS: dict[str, dict[str, Any]] = {
    "compact_experience_aligned_10m_rows": {
        "path": USER_ROOT / "experiments/archive/compact_experience/data/qwen_clean_aligned/training_corpora/qwen_aligned_10M.jsonl",
        "fields": ["text"],
        "kind": "current_lineage_10M_pool",
    },
    "fw_frozen_sources_38167": {
        "path": A01_WS / "data/fw_mechanism_source_selection/fw_mechanism_frozen_sources.jsonl",
        "fields": ["text"],
        "kind": "fineweb_candidate_source_reservoir",
    },
    "fw_compact_rewrites": {
        "path": A01_WS / "data/fw_full_preservation/full26k_usable_pairs_for_materializer.jsonl",
        "fields": ["rewrite_text"],
        "kind": "selected_qwen35_compact_rewrites",
    },
    "fw_breadth_whole_sentence_companions": {
        "path": A01_WS / "data/fw_source_breadth_wholesentence_arm/source_breadth_wholesentence_companion_sources.jsonl",
        "fields": ["text"],
        "kind": "selected_fineweb_breadth_companions",
    },
}

WORD_RE = re.compile(r"\b[\w]+(?:['’-][\w]+)?\b", re.UNICODE)
SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'“‘\[])|\n+")

CAUSAL_RE = re.compile(
    r"\b(if|when|whenever|unless|because|since|therefore|thus|so that|in order to|as a result|results? in|resulted in|leads? to|causes?|caused|prevents?|allows?)\b",
    re.I,
)
CHANGE_RE = re.compile(
    r"\b(becomes?|became|turns? into|changed?|changes?|increase[sd]?|decrease[sd]?|fill(?:ed|s)?|empty|emptied|open(?:ed|s)?|close[sd]?|sealed?|unsealed|broken?|breaks?|melt(?:ed|s)?|freez(?:e|es|ing|en)|dissolv(?:e|es|ed)|fall(?:s|en|ing)?|fell|drop(?:s|ped)?|rise[sn]?|rose|lower(?:ed|s)?|lift(?:ed|s)?|move[sd]?|shifts?|spread(?:s|ing)?|mix(?:ed|es)?|separate[sd]?)\b",
    re.I,
)
CONTRAST_RE = re.compile(
    r"\b(but|whereas|while|although|instead|rather than|compared with|compared to|before|after|first|then|next|finally|from\b.{0,45}\bto)\b",
    re.I,
)
ACTION_RE = re.compile(
    r"\b(push(?:ed|es)?|pull(?:ed|s)?|throw(?:s|n|ing)?|threw|drop(?:ped|s)?|pour(?:ed|s)?|carry|carried|put|puts|place[sd]?|insert(?:ed|s)?|remove[sd]?|cut|cuts|wash(?:ed|es)?|cook(?:ed|s)?|heat(?:ed|s)?|cool(?:ed|s)?|bend(?:s|ing)?|fold(?:ed|s)?|tie[ds]?|untie[ds]?|cover(?:ed|s)?|seal(?:ed|s)?|leak(?:ed|s)?|spill(?:ed|s)?|burn(?:ed|s)?|shake|shook|press(?:ed|es)?|rub(?:bed|s)?|roll(?:ed|s)?|slide[sd]?|float(?:ed|s)?|sink(?:s|ing)?|attach(?:ed|es)?|connect(?:ed|s)?)\b",
    re.I,
)
SPATIAL_RE = re.compile(
    r"\b(in|inside|outside|under|over|above|below|between|through|across|around|near|far|into|onto|out of|up|down|left|right|front|back|behind|beside|toward|towards|away from|close to)\b",
    re.I,
)
AFFORDANCE_RE = re.compile(
    r"\b(can|could|able to|use[sd]?|using|need(?:ed|s)? to|must|should|how to|way to|safe|careful|protect(?:ed|s)?|avoid(?:ed|s)?|prevent(?:ed|s)?|make sure|so you can)\b",
    re.I,
)

# Concrete anchors: deliberately broad but physical/object-biased.  The filter does
# not try to prove semantic equivalence; it raises purity relative to research by
# requiring explicit relation markers plus at least one concrete/action anchor.
CONCRETE_WORDS = {
    "air","water","fire","heat","cold","ice","steam","rain","snow","wind","light","sound","smoke","dust","dirt","mud","oil","gas",
    "bag","ball","box","bottle","bowl","cup","glass","jar","can","bucket","basket","plate","pan","pot","spoon","fork","knife","stick","rope","string","wire","paper","card","book","cloth","towel","shirt","shoe","coat","blanket","bed","chair","table","door","window","wall","floor","roof","room","house","castle","car","truck","bike","wheel","engine","machine","tool","button","handle","key","lock","lid","cap","hole","space","pipe","tube","hose","nest","stone","rock","wood","metal","plastic","rubber","sand","soil","seed","plant","tree","leaf","food","bread","milk","egg","meat","fruit","animal","dog","cat","bird","fish","insect","wasp","cattle","person","hand","foot","body","mouth","eye","skin",
}
ABSTRACT_NOISE_WORDS = {
    "love","happiness","sadness","competition","win","winning","music","song","dance","dancing","peace","war","government","policy","election","market","finance","stock","company","business","rights","law","argument","opinion","belief","idea","story","movie","game","team","score","religion","campaign","internet","website","software","programming","data","research","study",
}
TRANSITION_NEUTRAL_RE = re.compile(
    r"\b(if|when|whenever|unless|because|therefore|thus|so that|in order to|result|results|resulted|cause|causes|caused|leads?|prevent|allows?|"
    r"become|became|change|changed|changes|turns? into|from\b.{0,35}\bto|increase|decrease|more|less|fewer|full|empty|hot|cold|wet|dry|"
    r"push|pull|throw|drop|fall|fell|bounce|break|broke|broken|bend|melt|freeze|dissolve|open|close|up|down|above|below|inside|outside|"
    r"before|after|first|last|next|earlier|later|how to|safe|careful|instead|rather than|whereas|although)\b",
    re.I,
)
NOISE_RE = re.compile(r"https?://|www\.|</?\w+>|\[[^\]]*(illustration|edit|citation)[^\]]*\]|\b(cookie policy|all rights reserved)\b", re.I)
TRANSCRIPT_NOISE_RE = re.compile(r"\*[A-Z]{2,5}:|\.\.\.s*\.\.\.|_{2,}|={2,}")


def norm(text: str) -> str:
    return re.sub(r"\s+", " ", str(text).strip())


def words(text: str) -> list[str]:
    return [w.lower() for w in WORD_RE.findall(text)]


def wc(text: str) -> int:
    return len(words(text))


def split_sentences(text: str) -> list[str]:
    out: list[str] = []
    for s in SENT_SPLIT_RE.split(str(text)):
        s = norm(s)
        if not s:
            continue
        for sub in re.split(r"\s+(?=\*[A-Z]{2,5}:)|\s+(?=Q:)|\s+(?=A:)", s):
            sub = norm(sub)
            if sub:
                out.append(sub)
    return out


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def len_bin(n: int) -> str:
    if n <= 15:
        return "08_15"
    if n <= 25:
        return "16_25"
    if n <= 35:
        return "26_35"
    if n <= 50:
        return "36_50"
    if n <= 70:
        return "51_70"
    return "71_90"


def transition_marker_counts(text: str) -> dict[str, int]:
    return {
        "causal": len(CAUSAL_RE.findall(text)),
        "change": len(CHANGE_RE.findall(text)),
        "contrast_time": len(CONTRAST_RE.findall(text)),
        "action": len(ACTION_RE.findall(text)),
        "spatial": len(SPATIAL_RE.findall(text)),
        "affordance": len(AFFORDANCE_RE.findall(text)),
    }


def semantic_features(row: dict[str, Any]) -> dict[str, Any]:
    text = norm(row.get("text", ""))
    toks = words(text)
    tokset = set(toks)
    hits = row.get("hits") or {}
    marker = transition_marker_counts(text)
    concrete_count = sum(1 for t in toks if t in CONCRETE_WORDS)
    abstract_count = sum(1 for t in toks if t in ABSTRACT_NOISE_WORDS)
    non_social_hit = int(hits.get("physical_action", 0)) + int(hits.get("material_object", 0)) + int(hits.get("spatial_relation", 0)) + int(hits.get("temporal_quantity", 0)) + int(hits.get("affordance_procedure", 0))
    social_hit = int(hits.get("social_agent_relation", 0))
    relation_hit = int(hits.get("condition_cause", 0)) + int(hits.get("change_transition", 0)) + int(bool(row.get("contrast_pairs")))
    marker_sum = marker["causal"] + marker["change"] + marker["contrast_time"]
    concrete_anchor = concrete_count >= 1 or marker["action"] >= 1 or int(hits.get("material_object", 0)) >= 1
    explicit_relation = marker_sum >= 1 or relation_hit >= 2
    action_spatial_temporal = marker["action"] + marker["spatial"] + marker["affordance"] + int(hits.get("spatial_relation", 0)) + int(hits.get("temporal_quantity", 0))
    contains_bad = bool(NOISE_RE.search(text)) or bool(TRANSCRIPT_NOISE_RE.search(text))
    too_fragmentary = text[:1].islower() and len(toks) < 20
    abstract_dominant = abstract_count >= 2 and abstract_count > concrete_count
    social_dominant = social_hit >= 2 and non_social_hit < 3 and concrete_count == 0
    score = 0.0
    score += 2.0 * min(marker["causal"], 2)
    score += 1.6 * min(marker["change"], 2)
    score += 1.3 * min(marker["contrast_time"], 2)
    score += 1.1 * min(marker["action"], 2)
    score += 0.8 * min(marker["spatial"], 3)
    score += 0.7 * min(marker["affordance"], 2)
    score += 1.2 * min(concrete_count, 3)
    score += 0.35 * min(non_social_hit, 8)
    score += 0.25 * min(float(row.get("score", 0.0) or 0.0), 20.0)
    score -= 1.2 * abstract_count
    score -= 1.5 if social_dominant else 0.0
    score -= 2.5 if contains_bad else 0.0
    score -= 1.0 if too_fragmentary else 0.0
    strict = (
        12 <= int(row.get("words", wc(text))) <= 70
        and explicit_relation
        and concrete_anchor
        and non_social_hit >= 3
        and action_spatial_temporal >= 2
        and not contains_bad
        and not social_dominant
        and not abstract_dominant
    )
    very_strict = (
        strict
        and marker["causal"] + marker["change"] + marker["contrast_time"] >= 2
        and concrete_count + marker["action"] + int(hits.get("material_object", 0)) >= 2
        and non_social_hit >= 4
        and abstract_count == 0
    )
    return {
        "marker": marker,
        "concrete_count": concrete_count,
        "abstract_noise_count": abstract_count,
        "non_social_hit": non_social_hit,
        "relation_hit": relation_hit,
        "explicit_relation": explicit_relation,
        "concrete_anchor": concrete_anchor,
        "contains_noise": contains_bad,
        "too_fragmentary": too_fragmentary,
        "social_dominant": social_dominant,
        "abstract_dominant": abstract_dominant,
        "core_score": round(score, 4),
        "core_strict": strict,
        "core_very_strict": very_strict,
        "length_bin": len_bin(int(row.get("words", wc(text)))),
    }


def primary_bucket(row: dict[str, Any]) -> str:
    routes = list(row.get("routes") or [])
    selected = row.get("selected_route_bucket")
    if selected in PRIMARY_BUCKETS:
        return str(selected)
    for b in PRIMARY_BUCKETS:
        if b in routes:
            return b
    return "other"


def attach_features(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for r in rows:
        rr = dict(r)
        feat = semantic_features(rr)
        rr.update(feat)
        rr["selected_route_bucket"] = primary_bucket(rr)
        out.append(rr)
    return out


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total_words = sum(int(r.get("words", wc(r.get("text", "")))) for r in rows)
    by_bucket = Counter(str(r.get("selected_route_bucket", "")) for r in rows)
    by_source_words = Counter()
    by_len_words = Counter()
    for r in rows:
        w = int(r.get("words", wc(r.get("text", ""))))
        by_source_words[str(r.get("source_label", ""))] += w
        by_len_words[str(r.get("length_bin") or len_bin(w))] += w
    scores = [float(r.get("core_score", 0.0)) for r in rows]
    return {
        "sentences": len(rows),
        "words": total_words,
        "mean_words": total_words / len(rows) if rows else 0.0,
        "median_words": statistics.median([int(r.get("words", wc(r.get("text", "")))) for r in rows]) if rows else 0.0,
        "mean_core_score": statistics.mean(scores) if scores else 0.0,
        "bucket_sentences": dict(by_bucket),
        "source_words": dict(by_source_words),
        "length_bin_words": dict(by_len_words),
    }


def build_balanced_slice(pool: list[dict[str, Any]], budget: int) -> list[dict[str, Any]]:
    by_bucket: dict[str, list[dict[str, Any]]] = {b: [] for b in PRIMARY_BUCKETS}
    overflow: list[dict[str, Any]] = []
    for r in pool:
        b = str(r.get("selected_route_bucket", "other"))
        if b in by_bucket:
            by_bucket[b].append(r)
        else:
            overflow.append(r)
    for rows in by_bucket.values():
        rows.sort(key=lambda r: (-float(r.get("core_score", 0.0)), abs(int(r["words"]) - 36), r.get("text_sha256", "")))
    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    total = 0
    quota = budget // len(PRIMARY_BUCKETS)
    # Fill each route to roughly equal word mass to keep the substrate broad.
    for b in PRIMARY_BUCKETS:
        sw = 0
        for r in by_bucket[b]:
            key = str(r.get("text_sha256") or r.get("text"))
            w = int(r["words"])
            if key in selected_ids:
                continue
            if sw + w <= quota and total + w <= budget:
                selected.append(r); selected_ids.add(key); sw += w; total += w
    # Top off with best remaining core rows, preserving no duplicates.
    remaining = [r for r in pool if str(r.get("text_sha256") or r.get("text")) not in selected_ids]
    remaining.sort(key=lambda r: (-float(r.get("core_score", 0.0)), abs(int(r["words"]) - 36), r.get("text_sha256", "")))
    for r in remaining:
        w = int(r["words"])
        if total + w <= budget:
            selected.append(r); selected_ids.add(str(r.get("text_sha256") or r.get("text"))); total += w
        if total >= budget:
            break
    # Small exact-remainder repair: if possible add one row of the remaining word count.
    rem = budget - total
    if rem > 0:
        for r in remaining:
            key = str(r.get("text_sha256") or r.get("text"))
            if key in selected_ids:
                continue
            if int(r["words"]) == rem:
                selected.append(r); selected_ids.add(key); total += rem
                break
    return selected


def neutral_transition_score(text: str) -> int:
    return len(TRANSITION_NEUTRAL_RE.findall(text))


def is_neutral(text: str, nwords: int) -> bool:
    if nwords < 8 or nwords > 90:
        return False
    if NOISE_RE.search(text) or TRANSCRIPT_NOISE_RE.search(text):
        return False
    if neutral_transition_score(text) > 0:
        return False
    # Avoid rows likely to be long proper-name catalog/news lists.
    if len(re.findall(r"\b[A-Z][a-z]{2,}\b", text)) >= max(8, nwords // 5):
        return False
    return True


def source_bin_targets(treatment: list[dict[str, Any]]) -> dict[tuple[str, str], int]:
    targets: Counter[tuple[str, str]] = Counter()
    for r in treatment:
        w = int(r["words"])
        targets[(str(r.get("source_label", "")), len_bin(w))] += w
    return dict(targets)


def collect_neutral_pool(targets: dict[tuple[str, str], int], multiplier: float = 2.5, slack_words: int = 4000) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    target_by_source: Counter[str] = Counter()
    target_by_source_bin: Counter[tuple[str, str]] = Counter()
    for key, val in targets.items():
        src, lb = key
        target_by_source[src] += val
        target_by_source_bin[key] += val
    needed_by_source = {src: int(math.ceil(words_ * multiplier + slack_words)) for src, words_ in target_by_source.items()}
    pool: list[dict[str, Any]] = []
    seen = set()
    scanned: dict[str, Any] = {}
    for label, need in needed_by_source.items():
        spec = RESERVOIRS.get(label) or RESERVOIRS.get("fw_frozen_sources_38167")
        if spec is None:
            continue
        path = Path(spec["path"])
        got_total = 0
        got_by_bin: Counter[str] = Counter()
        scanned_sent = 0; scanned_words = 0; accepted = 0
        with path.open("r", encoding="utf-8", errors="replace") as f:
            for row_index, line in enumerate(f):
                if got_total >= need and all(got_by_bin[lb] >= int(math.ceil(target_by_source_bin.get((label, lb), 0) * multiplier)) for lb in ["08_15","16_25","26_35","36_50","51_70","71_90"]):
                    break
                if not line.strip():
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                for field in spec["fields"]:
                    raw = str(obj.get(field, "") or "")
                    if not raw:
                        continue
                    for sent_i, sent in enumerate(split_sentences(raw)):
                        sent = norm(sent)
                        n = wc(sent)
                        scanned_sent += 1; scanned_words += n
                        if not is_neutral(sent, n):
                            continue
                        key = sent.lower()
                        if key in seen:
                            continue
                        seen.add(key)
                        rec = {
                            "source_label": label,
                            "source_kind": spec["kind"],
                            "input_path": str(path),
                            "row_index": row_index,
                            "sentence_index": sent_i,
                            "field": field,
                            "text": sent,
                            "words": n,
                            "length_bin": len_bin(n),
                            "neutral_transition_score": neutral_transition_score(sent),
                            "origin_source": obj.get("source") or obj.get("pool") or obj.get("source_kind") or obj.get("origin_source"),
                            "doc_id": obj.get("doc_id"),
                            "norm_hash": obj.get("norm_hash"),
                        }
                        pool.append(rec)
                        got_total += n; got_by_bin[len_bin(n)] += n; accepted += 1
        scanned[label] = {
            "target_words": target_by_source[label],
            "collection_need_words": need,
            "collected_words": got_total,
            "collected_by_len_bin_words": dict(got_by_bin),
            "accepted_sentences": accepted,
            "sentences_scanned": scanned_sent,
            "words_scanned": scanned_words,
            "path": str(path),
        }
    return pool, scanned


def pick_length_source_matched_neutral(treatment: list[dict[str, Any]], neutral_pool: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    by_source_bin: defaultdict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    by_source: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    by_bin: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in neutral_pool:
        src = str(r.get("source_label", "")); lb = str(r.get("length_bin") or len_bin(int(r["words"])))
        by_source_bin[(src, lb)].append(r)
        by_source[src].append(r)
        by_bin[lb].append(r)
    for bucket in list(by_source_bin.values()) + list(by_source.values()) + list(by_bin.values()):
        bucket.sort(key=lambda r: (int(r["words"]), r.get("text", "")))
    used: set[int] = set()
    chosen: list[dict[str, Any]] = []
    fallback_counter = Counter()

    def choose_from(cands: list[dict[str, Any]], target_len: int) -> dict[str, Any] | None:
        best_idx = None; best_key = None
        for idx, cand in enumerate(cands):
            cid = id(cand)
            if cid in used:
                continue
            key = (abs(int(cand["words"]) - target_len), abs(int(cand["words"]) - 36), cand.get("text", ""))
            if best_key is None or key < best_key:
                best_key = key; best_idx = idx
        if best_idx is None:
            return None
        cand = cands[best_idx]
        used.add(id(cand))
        return cand

    # Longer rows first: long neutral rows are scarcer and determine length matching.
    for tr in sorted(treatment, key=lambda r: (-int(r["words"]), str(r.get("source_label", "")), str(r.get("text_sha256", "")))):
        src = str(tr.get("source_label", "")); lb = len_bin(int(tr["words"])); tw = int(tr["words"])
        cand = choose_from(by_source_bin[(src, lb)], tw)
        mode = "same_source_same_lenbin"
        if cand is None:
            cand = choose_from(by_source[src], tw); mode = "same_source_any_lenbin"
        if cand is None:
            cand = choose_from(by_bin[lb], tw); mode = "any_source_same_lenbin"
        if cand is None:
            cand = choose_from(neutral_pool, tw); mode = "any_source_any_lenbin"
        if cand is not None:
            rr = dict(cand)
            rr["matched_treatment_words"] = tw
            rr["match_mode"] = mode
            chosen.append(rr)
            fallback_counter[mode] += 1
        else:
            fallback_counter["unmatched"] += 1
    return chosen, {"match_modes": dict(fallback_counter)}


def match_metrics(treatment: list[dict[str, Any]], neutral: list[dict[str, Any]]) -> dict[str, Any]:
    def word_counter(rows: list[dict[str, Any]], keyfunc) -> Counter[str]:
        c: Counter[str] = Counter()
        for r in rows:
            c[str(keyfunc(r))] += int(r["words"])
        return c
    tw = sum(int(r["words"]) for r in treatment)
    nw = sum(int(r["words"]) for r in neutral)
    t_src = word_counter(treatment, lambda r: r.get("source_label", ""))
    n_src = word_counter(neutral, lambda r: r.get("source_label", ""))
    t_bin = word_counter(treatment, lambda r: len_bin(int(r["words"])))
    n_bin = word_counter(neutral, lambda r: len_bin(int(r["words"])))
    def l1(a: Counter[str], b: Counter[str]) -> float:
        keys = set(a) | set(b)
        if not keys:
            return 0.0
        ta = sum(a.values()) or 1
        tb = sum(b.values()) or 1
        return sum(abs(a.get(k,0)/ta - b.get(k,0)/tb) for k in keys)
    length_diffs = [int(n.get("words",0)) - int(n.get("matched_treatment_words",0)) for n in neutral if "matched_treatment_words" in n]
    return {
        "treatment_sentences": len(treatment),
        "neutral_sentences": len(neutral),
        "treatment_words": tw,
        "neutral_words": nw,
        "word_diff_neutral_minus_treatment": nw - tw,
        "treatment_mean_words": tw/len(treatment) if treatment else 0.0,
        "neutral_mean_words": nw/len(neutral) if neutral else 0.0,
        "source_distribution_l1": l1(t_src, n_src),
        "length_bin_distribution_l1": l1(t_bin, n_bin),
        "treatment_source_words": dict(t_src),
        "neutral_source_words": dict(n_src),
        "treatment_lenbin_words": dict(t_bin),
        "neutral_lenbin_words": dict(n_bin),
        "row_length_diff_mean": statistics.mean(length_diffs) if length_diffs else None,
        "row_length_diff_median": statistics.median(length_diffs) if length_diffs else None,
        "row_abs_length_diff_mean": statistics.mean([abs(x) for x in length_diffs]) if length_diffs else None,
        "row_abs_length_diff_median": statistics.median([abs(x) for x in length_diffs]) if length_diffs else None,
    }


def csv_dict_writer(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})


def main() -> None:
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    NOTE.parent.mkdir(parents=True, exist_ok=True)
    raw = load_jsonl(CANDIDATES)
    featured = attach_features(raw)
    strict = [r for r in featured if r.get("core_strict")]
    very_strict = [r for r in featured if r.get("core_very_strict")]
    strict.sort(key=lambda r: (-float(r.get("core_score", 0.0)), abs(int(r["words"]) - 36), r.get("text_sha256", "")))
    very_strict.sort(key=lambda r: (-float(r.get("core_score", 0.0)), abs(int(r["words"]) - 36), r.get("text_sha256", "")))
    write_jsonl(OUT / "core_transition_candidates.jsonl", strict)
    write_jsonl(OUT / "core_transition_very_strict.jsonl", very_strict)

    slices: dict[str, dict[str, Any]] = {}
    all_targets: dict[tuple[str, str], int] = {}
    for budget in BUDGETS:
        rows = build_balanced_slice(strict, budget)
        for r in rows:
            r["slice_budget"] = budget
        write_jsonl(OUT / f"core_transition_treatment_{budget//1000}k.jsonl", rows)
        slices[str(budget)] = {"treatment": rows, "treatment_summary": summarize_rows(rows)}
        for k, v in source_bin_targets(rows).items():
            all_targets[k] = max(all_targets.get(k, 0), v)

    neutral_pool, neutral_scan = collect_neutral_pool(all_targets)
    write_jsonl(OUT / "neutral_candidate_pool_for_length_matching.jsonl", neutral_pool)

    slice_metrics_rows: list[dict[str, Any]] = []
    match_rows: list[dict[str, Any]] = []
    for budget in BUDGETS:
        key = str(budget)
        treatment = slices[key]["treatment"]
        neutral, match_info = pick_length_source_matched_neutral(treatment, neutral_pool)
        write_jsonl(OUT / f"core_transition_neutral_lengthmatched_{budget//1000}k.jsonl", neutral)
        m = match_metrics(treatment, neutral)
        slices[key]["neutral"] = neutral
        slices[key]["neutral_summary"] = summarize_rows(neutral)
        slices[key]["match_metrics"] = m
        slices[key]["match_info"] = match_info
        slice_metrics_rows.append({
            "budget": f"{budget//1000}k",
            "arm": "treatment",
            "sentences": len(treatment),
            "words": sum(int(r["words"]) for r in treatment),
            "mean_words": round(m["treatment_mean_words"], 3),
            "source_l1_vs_treatment": 0.0,
            "lenbin_l1_vs_treatment": 0.0,
            "mean_core_score": round(summarize_rows(treatment)["mean_core_score"], 3),
        })
        slice_metrics_rows.append({
            "budget": f"{budget//1000}k",
            "arm": "neutral_lengthmatched",
            "sentences": len(neutral),
            "words": sum(int(r["words"]) for r in neutral),
            "mean_words": round(m["neutral_mean_words"], 3),
            "source_l1_vs_treatment": round(m["source_distribution_l1"], 6),
            "lenbin_l1_vs_treatment": round(m["length_bin_distribution_l1"], 6),
            "mean_core_score": "",
        })
        match_rows.append({
            "budget": f"{budget//1000}k",
            "word_diff_neutral_minus_treatment": m["word_diff_neutral_minus_treatment"],
            "source_distribution_l1": round(m["source_distribution_l1"], 6),
            "length_bin_distribution_l1": round(m["length_bin_distribution_l1"], 6),
            "row_abs_length_diff_mean": round(float(m["row_abs_length_diff_mean"] or 0.0), 3),
            "row_abs_length_diff_median": round(float(m["row_abs_length_diff_median"] or 0.0), 3),
            "match_modes": json.dumps(match_info["match_modes"], sort_keys=True),
        })

    csv_dict_writer(OUT / "core_slice_summary.csv", slice_metrics_rows, ["budget","arm","sentences","words","mean_words","source_l1_vs_treatment","lenbin_l1_vs_treatment","mean_core_score"])
    csv_dict_writer(OUT / "length_source_match_summary.csv", match_rows, ["budget","word_diff_neutral_minus_treatment","source_distribution_l1","length_bin_distribution_l1","row_abs_length_diff_mean","row_abs_length_diff_median","match_modes"])

    summary = {
        "status": "CORE_TRANSITION_FILTER_DONE",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "input_candidates": str(CANDIDATES),
        "input_candidate_count": len(raw),
        "input_candidate_words": sum(int(r.get("words", wc(r.get("text", "")))) for r in raw),
        "core_strict_count": len(strict),
        "core_strict_words": sum(int(r["words"]) for r in strict),
        "core_very_strict_count": len(very_strict),
        "core_very_strict_words": sum(int(r["words"]) for r in very_strict),
        "core_strict_summary": summarize_rows(strict),
        "core_very_strict_summary": summarize_rows(very_strict),
        "neutral_pool_count": len(neutral_pool),
        "neutral_pool_words": sum(int(r["words"]) for r in neutral_pool),
        "neutral_scan": neutral_scan,
        "slices": {k: {kk: vv for kk, vv in v.items() if kk not in {"treatment","neutral"}} for k, v in slices.items()},
        "files": {
            "core_candidates": str(OUT / "core_transition_candidates.jsonl"),
            "core_very_strict": str(OUT / "core_transition_very_strict.jsonl"),
            "neutral_pool": str(OUT / "neutral_candidate_pool_for_length_matching.jsonl"),
            "slice_summary_csv": str(OUT / "core_slice_summary.csv"),
            "match_summary_csv": str(OUT / "length_source_match_summary.csv"),
            "note": str(NOTE),
        },
        "elapsed_sec": round(time.time() - t0, 3),
    }
    (OUT / "core_transition_filter_and_controls.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    note_lines = [
        "# research — core transition filter and length-matched controls",
        "",
        "## Purpose",
        "",
        "research found a large corpus-derived transition reservoir, but sampling showed metaphor/social/narrative noise. This CPU-only step filters it toward explicit relation markers plus concrete/action/spatial/temporal anchors and builds neutral controls matched by source and sentence-length bins. No official evaluation item text is read, and no training is launched.",
        "",
        "## Main counts",
        "",
        f"- Input research candidates: {len(raw)} sentences / {summary['input_candidate_words']} words.",
        f"- Core strict candidates: {len(strict)} sentences / {summary['core_strict_words']} words.",
        f"- Core very-strict candidates: {len(very_strict)} sentences / {summary['core_very_strict_words']} words.",
        f"- Neutral candidate pool for matching: {len(neutral_pool)} sentences / {summary['neutral_pool_words']} words.",
        "",
        "## Balanced treatment slices and length-matched controls",
        "",
        "See `core_slice_summary.csv` and `length_source_match_summary.csv` for exact matching measurements. The intended use is a future cheap treatment-vs-neutral probe only if FW compact/breadth endpoints fail to move the shared EWoK/GlobalPIQA relational weakness.",
        "",
        "## Scientific reading",
        "",
        "The filter creates a cleaner substrate than research but still remains a research asset, not a training decision. It should not supersede the running FW compact-vs-breadth experiment. Before any GPU probe, sample the actual slice, check whether the neutral control is semantically neutral rather than merely marker-free, and choose the minimum reliable probe scale.",
        "",
        "## Files",
        "",
        f"- summary JSON: `{OUT / 'core_transition_filter_and_controls.json'}`",
        f"- core strict candidates: `{OUT / 'core_transition_candidates.jsonl'}`",
        f"- core very-strict candidates: `{OUT / 'core_transition_very_strict.jsonl'}`",
        f"- treatment slices: `{OUT / 'core_transition_treatment_50k.jsonl'}`, `{OUT / 'core_transition_treatment_100k.jsonl'}`, `{OUT / 'core_transition_treatment_200k.jsonl'}`",
        f"- length-matched controls: `{OUT / 'core_transition_neutral_lengthmatched_50k.jsonl'}`, `{OUT / 'core_transition_neutral_lengthmatched_100k.jsonl'}`, `{OUT / 'core_transition_neutral_lengthmatched_200k.jsonl'}`",
        f"- neutral pool: `{OUT / 'neutral_candidate_pool_for_length_matching.jsonl'}`",
        f"- slice summary CSV: `{OUT / 'core_slice_summary.csv'}`",
        f"- match summary CSV: `{OUT / 'length_source_match_summary.csv'}`",
    ]
    NOTE.write_text("\n".join(note_lines) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": summary["status"],
        "core_strict_count": summary["core_strict_count"],
        "core_strict_words": summary["core_strict_words"],
        "core_very_strict_count": summary["core_very_strict_count"],
        "core_very_strict_words": summary["core_very_strict_words"],
        "neutral_pool_words": summary["neutral_pool_words"],
        "json": str(OUT / "core_transition_filter_and_controls.json"),
        "slice_summary_csv": str(OUT / "core_slice_summary.csv"),
        "match_summary_csv": str(OUT / "length_source_match_summary.csv"),
        "note": str(NOTE),
        "elapsed_sec": summary["elapsed_sec"],
    }, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
