#!/usr/bin/env python3
"""research: inspect SynCSE relation types used by the relation-ladder screen.

This script examines the interpretation caveat that the `hard_neg` field is not a
clean document-continuation or topic-matched non-rewrite control.  It measures
simple lexical/negation markers and saves representative rows so the training
result is interpreted as a relation-type probe rather than a clean causal split.
"""
from __future__ import annotations

import json
import os
import random
import re
import statistics
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple

SEED = 43
ROOT = Path("experiments/archive/compact_experience")
HF_CACHE = ROOT / "staging/hf_hub"
OUT_DIR = ROOT / "data/syncse_relation_ladder_eval"
OUT_JSON = OUT_DIR / "syncse_relation_probe.json"
OUT_MD = (ROOT.parents[2] / 'research/notes/compact_experience/19_syncse_relation_probe.md')
TARGET_BASE_WORDS = 7_989_120

os.environ["HF_HOME"] = str(HF_CACHE)
os.environ["HF_HUB_CACHE"] = str(HF_CACHE)
os.environ["HF_DATASETS_CACHE"] = str(HF_CACHE)

STOP = set("""
a an the and or but if then else when while as of in on at by for with from to into onto over under through during before after is are was were be been being do does did have has had having i you he she it we they them him her his hers its our ours their theirs this that these those there here not no yes than so very can could should would will may might must about above below up down out just only more most less least such same other another one two three four five new old first last
""".split())
NEG = set("""
not no never none nobody nothing neither nor without cannot can't don't doesn't didn't won't wouldn't shouldn't couldn't isn't aren't wasn't weren't hasn't haven't hadn't fail fails failed false impossible unable lack lacks lacking absent except deny denied denies refuse refused contrary contradiction contradicts opposed opposite unlike instead
""".split())
ROLE_MARKERS = set("""
by from to against with versus vs defeated defeat beat beaten killed kills arrested arrest sentenced sentence accused accuse attacked attack chased chase followed follow led lead gave received bought sold borrowed lent parent child mother father son daughter husband wife man woman boy girl dog cat horse rider player team winner loser
""".split())


def clean(s: Any) -> str:
    return " ".join(str(s).split())


def wlen(s: str) -> int:
    return len(s.split())


def toks(s: str) -> List[str]:
    return re.findall(r"[A-Za-z0-9']+", s.lower())


def content_toks(s: str) -> List[str]:
    return [t for t in toks(s) if t not in STOP and len(t) > 2]


def overlap(a: str, b: str) -> Dict[str, float]:
    ca = Counter(content_toks(a))
    cb = Counter(content_toks(b))
    sa = set(ca)
    sb = set(cb)
    inter = sa & sb
    union = sa | sb
    return {
        "jaccard": (len(inter) / len(union)) if union else 0.0,
        "containment_min": (len(inter) / min(len(sa), len(sb))) if sa and sb else 0.0,
        "shared_content_types": float(len(inter)),
        "len_a_content_types": float(len(sa)),
        "len_b_content_types": float(len(sb)),
    }


def has_neg(s: str) -> bool:
    return any(t in NEG for t in toks(s))


def role_score(a: str, b: str) -> int:
    ta = set(toks(a))
    tb = set(toks(b))
    return len((ta | tb) & ROLE_MARKERS)


def load_rows() -> List[Dict[str, Any]]:
    from datasets import load_dataset

    ds = load_dataset("hkust-nlp/SynCSE-partial-NLI", split="train", cache_dir=str(HF_CACHE))
    rows: List[Dict[str, Any]] = []
    skipped = Counter()
    seen = set()
    for r in ds:
        s0 = clean(r.get("sent0", ""))
        s1 = clean(r.get("sent1", ""))
        hn = clean(r.get("hard_neg", ""))
        if min(wlen(s0), wlen(s1), wlen(hn)) < 3:
            skipped["too_short"] += 1
            continue
        key = (s0.lower(), s1.lower(), hn.lower())
        if key in seen:
            skipped["duplicate_triplet"] += 1
            continue
        seen.add(key)
        if s0.lower() == s1.lower() or s0.lower() == hn.lower():
            skipped["identity"] += 1
            continue
        rows.append({"sent0": s0, "sent1": s1, "hard_neg": hn, "ow": wlen(s0), "pw": wlen(s1), "hw": wlen(hn)})
    rng = random.Random(SEED)
    rng.shuffle(rows)
    return rows


def rows_used_prefix(rows: List[Dict[str, Any]], relation: str) -> List[Dict[str, Any]]:
    out = []
    total = 0
    for r in rows:
        add = r["ow"] + (r["pw"] if relation == "paraphrase" else r["hw"])
        if total + add > TARGET_BASE_WORDS:
            break
        total += add
        out.append(r)
    return out


def summarize_pair(rows: List[Dict[str, Any]], partner_key: str) -> Dict[str, Any]:
    js = []
    cont = []
    shared = []
    neg_partner = 0
    neg_either = 0
    role = []
    for r in rows:
        o = overlap(r["sent0"], r[partner_key])
        js.append(o["jaccard"])
        cont.append(o["containment_min"])
        shared.append(o["shared_content_types"])
        neg_partner += int(has_neg(r[partner_key]))
        neg_either += int(has_neg(r["sent0"]) or has_neg(r[partner_key]))
        role.append(role_score(r["sent0"], r[partner_key]))
    n = len(rows)
    def q(vals: List[float], p: float) -> float:
        vals2 = sorted(vals)
        return vals2[min(len(vals2)-1, max(0, int(round((len(vals2)-1)*p))))]
    return {
        "n_rows": n,
        "mean_content_jaccard": statistics.fmean(js),
        "median_content_jaccard": statistics.median(js),
        "p90_content_jaccard": q(js, 0.9),
        "mean_min_containment": statistics.fmean(cont),
        "mean_shared_content_types": statistics.fmean(shared),
        "partner_negation_marker_rate": neg_partner / n,
        "either_side_negation_marker_rate": neg_either / n,
        "mean_role_marker_count": statistics.fmean(role),
        "role_marker_nonzero_rate": sum(x > 0 for x in role) / n,
    }


def choose_examples(rows: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    scored = []
    for i, r in enumerate(rows):
        oh = overlap(r["sent0"], r["hard_neg"])
        op = overlap(r["sent0"], r["sent1"])
        score = oh["containment_min"] + 0.25 * oh["shared_content_types"] + 0.5 * has_neg(r["hard_neg"]) + 0.15 * role_score(r["sent0"], r["hard_neg"])
        scored.append((score, i, oh, op, r))
    top = sorted(scored, reverse=True)[:24]
    rng = random.Random(SEED + 909)
    random_sample = rng.sample(scored, 24)
    def pack(items: List[Tuple[float, int, Dict[str, float], Dict[str, float], Dict[str, Any]]]) -> List[Dict[str, Any]]:
        out = []
        for score, i, oh, op, r in items:
            out.append({
                "row_index_in_shuffled_stream": i,
                "hardneg_probe_score": round(score, 4),
                "sent0": r["sent0"],
                "sent1_paraphrase": r["sent1"],
                "hard_neg": r["hard_neg"],
                "sent0_hardneg_overlap": {k: round(v, 4) for k, v in oh.items()},
                "sent0_paraphrase_overlap": {k: round(v, 4) for k, v in op.items()},
                "hard_neg_has_neg_marker": has_neg(r["hard_neg"]),
                "either_side_has_neg_marker": has_neg(r["sent0"]) or has_neg(r["hard_neg"]),
                "role_marker_count": role_score(r["sent0"], r["hard_neg"]),
            })
        return out
    return {"high_overlap_or_negation_hardneg_examples": pack(top), "random_examples": pack(random_sample)}


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = load_rows()
    para_rows = rows_used_prefix(rows, "paraphrase")
    hard_rows = rows_used_prefix(rows, "hardneg")
    summary = {
        "status": "SYNCSE_RELATION_PROBE_DONE",
        "purpose": "Ground the interpretation of the relation ladder, including the SynCSE hard_neg control limitation.",
        "seed": SEED,
        "target_base_words": TARGET_BASE_WORDS,
        "filtered_rows_total": len(rows),
        "prefix_rows_used_for_paraphrase_stream_before_boundary": len(para_rows),
        "prefix_rows_used_for_hardneg_stream_before_boundary": len(hard_rows),
        "paraphrase_sent0_sent1_summary": summarize_pair(para_rows, "sent1"),
        "hardneg_sent0_hardneg_summary": summarize_pair(hard_rows, "hard_neg"),
        "interpretation": "SynCSE hard_neg is useful as a hard relation-type stressor with repeated entities/lexical fields, but marker and sample evidence should not be treated as a clean topic-matched coherent non-rewrite control.",
        "examples": choose_examples(hard_rows),
    }
    OUT_JSON.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    md = [
        "# research — SynCSE relation probe",
        "",
        f"JSON: `{OUT_JSON}`",
        "",
        "Interpretation caveat: `hard_neg` is a useful relation-type condition, not a clean document-adjacent or topic-matched non-rewrite control.",
        "",
        "## Aggregate markers",
        "",
        "| pair type | rows | mean content Jaccard | mean min-containment | partner negation-marker rate | either-side negation-marker rate | role-marker nonzero rate |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for label, key in [("sent0–sent1 paraphrase", "paraphrase_sent0_sent1_summary"), ("sent0–hard_neg", "hardneg_sent0_hardneg_summary")]:
        s = summary[key]
        md.append(f"| {label} | {s['n_rows']} | {s['mean_content_jaccard']:.4f} | {s['mean_min_containment']:.4f} | {s['partner_negation_marker_rate']:.4f} | {s['either_side_negation_marker_rate']:.4f} | {s['role_marker_nonzero_rate']:.4f} |")
    md += [
        "",
        "## Interpretation for the training result",
        "",
        "The hard-negative arm can show whether placing semantically incompatible but often lexically related sentences next to each other changes BabyLM transfer. It cannot by itself isolate benign local coherence or same-topic continuity. If the research fast scores make this relation dimension important, the next experiment should use real document adjacency or a topic/cluster-matched non-rewrite neighbor that avoids contradiction-style supervision.",
        "",
        "## Representative high-overlap / marker-heavy hard-negative rows",
        "",
    ]
    for ex in summary["examples"]["high_overlap_or_negation_hardneg_examples"][:8]:
        md.append(f"- sent0: {ex['sent0']}")
        md.append(f"  - hard_neg: {ex['hard_neg']}")
        md.append(f"  - sent1: {ex['sent1_paraphrase']}")
        md.append(f"  - markers: hard_neg_neg={ex['hard_neg_has_neg_marker']}, overlap={ex['sent0_hardneg_overlap']['containment_min']:.3f}, role_markers={ex['role_marker_count']}")
    OUT_MD.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "examples"}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
