#!/usr/bin/env python3
"""Audit residual original--rewrite similarity in the research shuffled control.

The shuffled control deranges Qwen rewrites within source × rewrite-length bins.  To
interpret aligned-minus-shuffled, we need to know how much lexical/anchor similarity
survives that derangement.  This script reproduces the materializer's shuffle and
compares aligned and shuffled pair similarity distributions using transparent text
features (content-token overlap, Jaccard, number/entity anchor recall, source and
length-bin structure).  It does not use external embeddings or models.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import collections
import json
import pathlib
import random
import re
import statistics
from dataclasses import dataclass
from typing import Iterable

ROOT = _public_path('experiments/archive/compact_experience')
SELECTED = _public_path('experiments/archive/compact_experience/data/qwen_clean_aligned/selected_pairs.jsonl')
OUT = _public_path('experiments/archive/compact_experience/data/shuffled_residual_similarity_audit.json')
RNG_SEED = 28900

STOP = set("""
a an the and or but if while of in on at by for to from with without into onto over under as is are was were be been being am do does did has have had can could may might must should would will shall this that these those it its he she they them his her their we you i me my our your not no yes there here than then so such very just also only
""".split())
NUM_RE = re.compile(r"\b\d+(?:[.,:/-]\d+)*\b")
ENTITY_RE = re.compile(r"\b(?:[A-Z][A-Za-z0-9._'-]+|\*[A-Z]{2,4}:)\b")
TOK_RE = re.compile(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)?")


@dataclass
class Pair:
    pair_id: str
    original: str
    rewrite: str
    original_words: int
    rewrite_words: int
    source: str
    cohort: str


def length_bin(n: int) -> str:
    if n <= 12:
        return "le12"
    if n <= 18:
        return "13_18"
    if n <= 26:
        return "19_26"
    if n <= 38:
        return "27_38"
    return "gt38"


def toks(text: str, content: bool = True) -> list[str]:
    xs = [m.group(0).lower() for m in TOK_RE.finditer(text)]
    if content:
        xs = [x for x in xs if x not in STOP and len(x) > 1]
    return xs


def jaccard(a: Iterable[str], b: Iterable[str]) -> float:
    A, B = set(a), set(b)
    if not A and not B:
        return 1.0
    if not A or not B:
        return 0.0
    return len(A & B) / len(A | B)


def recall(a: Iterable[str], b: Iterable[str]) -> float | None:
    A, B = set(a), set(b)
    if not A:
        return None
    return len(A & B) / len(A)


def num_list(text: str) -> list[str]:
    return [m.group(0).lower() for m in NUM_RE.finditer(text)]


def ent_list(text: str) -> list[str]:
    # Lowercase for matching but keep speaker labels / proper-name tokens.
    return [m.group(0).lower() for m in ENTITY_RE.finditer(text)]


def load_pairs() -> list[Pair]:
    pairs = []
    with SELECTED.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                o = json.loads(line)
                pairs.append(Pair(
                    pair_id=str(o["pair_id"]), original=str(o["original"]), rewrite=str(o["rewrite"]),
                    original_words=int(o["original_words"]), rewrite_words=int(o["rewrite_words"]),
                    source=str(o.get("source", "unknown")), cohort=str(o.get("cohort", "unknown")),
                ))
    return pairs


def reproduce_shuffle(pairs: list[Pair]) -> list[Pair]:
    rng = random.Random(RNG_SEED)
    groups = collections.defaultdict(list)
    for i, p in enumerate(pairs):
        groups[(p.source, length_bin(p.rewrite_words))].append(i)
    shuffled = list(pairs)
    for key, idxs in groups.items():
        if len(idxs) < 2:
            continue
        rewrites = [(pairs[i].rewrite, pairs[i].rewrite_words, pairs[i].pair_id) for i in idxs]
        orig_ids = [pairs[i].pair_id for i in idxs]
        for _ in range(200):
            rng.shuffle(rewrites)
            if all(src_pid != orig_pid for orig_pid, (_, _, src_pid) in zip(orig_ids, rewrites)):
                break
        else:
            rewrites = rewrites[1:] + rewrites[:1]
        for i, (rw, rw_words, src_pid) in zip(idxs, rewrites):
            p = pairs[i]
            shuffled[i] = Pair(
                pair_id=p.pair_id + "__rw_from__" + src_pid,
                original=p.original,
                rewrite=rw,
                original_words=p.original_words,
                rewrite_words=rw_words,
                source=p.source,
                cohort=p.cohort,
            )
    return shuffled


def stats(vals: list[float]) -> dict:
    vals = [v for v in vals if v is not None]
    if not vals:
        return {"n": 0}
    vals_sorted = sorted(vals)

    def pct(q: float) -> float:
        i = min(len(vals_sorted) - 1, max(0, round((len(vals_sorted) - 1) * q)))
        return vals_sorted[i]

    return {
        "n": len(vals), "min": round(min(vals), 6), "mean": round(statistics.mean(vals), 6),
        "median": round(statistics.median(vals), 6), "p05": round(pct(0.05), 6), "p25": round(pct(0.25), 6),
        "p75": round(pct(0.75), 6), "p95": round(pct(0.95), 6), "max": round(max(vals), 6),
    }


def pair_features(p: Pair) -> dict:
    ot = toks(p.original, content=True)
    rt = toks(p.rewrite, content=True)
    all_ot = toks(p.original, content=False)
    all_rt = toks(p.rewrite, content=False)
    nums_o, nums_r = num_list(p.original), num_list(p.rewrite)
    ents_o, ents_r = ent_list(p.original), ent_list(p.rewrite)
    cont_recall = recall(ot, rt)
    return {
        "content_jaccard": jaccard(ot, rt),
        "all_token_jaccard": jaccard(all_ot, all_rt),
        "content_recall_original_to_rewrite": cont_recall,
        "number_recall_original_to_rewrite": recall(nums_o, nums_r),
        "entity_recall_original_to_rewrite": recall(ents_o, ents_r),
        "has_numbers": bool(nums_o),
        "has_entities": bool(ents_o),
    }


def summarize_feature_rows(rows: list[dict]) -> dict:
    out = {}
    for k in ["content_jaccard", "all_token_jaccard", "content_recall_original_to_rewrite", "number_recall_original_to_rewrite", "entity_recall_original_to_rewrite"]:
        vals = [r[k] for r in rows if r[k] is not None]
        out[k] = stats(vals)
    out["fraction_content_jaccard_ge_0p2"] = round(sum(r["content_jaccard"] >= 0.2 for r in rows) / len(rows), 6)
    out["fraction_content_jaccard_ge_0p4"] = round(sum(r["content_jaccard"] >= 0.4 for r in rows) / len(rows), 6)
    out["fraction_all_token_jaccard_ge_0p4"] = round(sum(r["all_token_jaccard"] >= 0.4 for r in rows) / len(rows), 6)
    return out


def main():
    pairs = load_pairs()
    shuf = reproduce_shuffle(pairs)
    aligned_rows = [pair_features(p) for p in pairs]
    shuffled_rows = [pair_features(p) for p in shuf]
    # Confirm derangement and same source/length-bin constraints.
    changed = 0
    same_source = 0
    same_lenbin = 0
    for p, s in zip(pairs, shuf):
        src_pid = s.pair_id.split("__rw_from__")[-1] if "__rw_from__" in s.pair_id else s.pair_id
        if src_pid != p.pair_id:
            changed += 1
        if s.source == p.source:
            same_source += 1
        if length_bin(s.rewrite_words) == length_bin(p.rewrite_words):
            same_lenbin += 1
    by_source = {}
    for source in sorted({p.source for p in pairs}):
        idxs = [i for i,p in enumerate(pairs) if p.source == source]
        by_source[source] = {
            "n": len(idxs),
            "aligned_content_jaccard": stats([aligned_rows[i]["content_jaccard"] for i in idxs]),
            "shuffled_content_jaccard": stats([shuffled_rows[i]["content_jaccard"] for i in idxs]),
            "shuffled_fraction_content_jaccard_ge_0p2": round(sum(shuffled_rows[i]["content_jaccard"] >= 0.2 for i in idxs) / len(idxs), 6) if idxs else None,
        }
    # Examples with surprisingly high residual shuffled lexical overlap.
    high = []
    for p, s, feat in zip(pairs, shuf, shuffled_rows):
        if feat["content_jaccard"] >= 0.4 and len(high) < 80:
            high.append({
                "pair_id": p.pair_id,
                "assigned_rewrite_from": s.pair_id.split("__rw_from__")[-1],
                "source": p.source,
                "rewrite_len_bin": length_bin(s.rewrite_words),
                "content_jaccard": round(feat["content_jaccard"], 4),
                "original": p.original,
                "assigned_rewrite": s.rewrite,
            })
    payload = {
        "status": "SHUFFLED_RESIDUAL_SIMILARITY_AUDIT",
        "selected_pairs": len(pairs),
        "shuffle_reproduced_rng_seed": RNG_SEED,
        "changed_pairs": changed,
        "same_source_pairs": same_source,
        "same_rewrite_length_bin_pairs": same_lenbin,
        "aligned_similarity": summarize_feature_rows(aligned_rows),
        "shuffled_similarity": summarize_feature_rows(shuffled_rows),
        "by_source": by_source,
        "high_residual_shuffled_overlap_examples": high,
        "interpretation_hint": "Low shuffled content-overlap relative to aligned supports the shuffled control as a correspondence-breaking null; residual same-source high-overlap examples mean aligned-minus-shuffled still tests local relatedness/correspondence, not abstract semantic alignment alone.",
    }
    OUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(OUT), "changed_pairs": changed, "aligned_content_jaccard_mean": payload["aligned_similarity"]["content_jaccard"]["mean"], "shuffled_content_jaccard_mean": payload["shuffled_similarity"]["content_jaccard"]["mean"], "shuffled_fraction_ge_0p4": payload["shuffled_similarity"]["fraction_content_jaccard_ge_0p4"]}, indent=2))


if __name__ == "__main__":
    main()
