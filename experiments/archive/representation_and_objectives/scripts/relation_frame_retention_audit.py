#!/usr/bin/env python3
"""research: relation-frame retention audit for compact_view_reinvest pairs.

CPU-only, no model inference and no use of official benchmark text for training-data
selection.  It reads the existing Qwen compact source/rewrite pairs and asks whether
compactification systematically removes general relational frames (spatial, physical,
material, social, causal/temporal, negation, comparison).  The result is a scientific
mechanism/repair substrate for interpreting corrected-tokenizer evaluations: if EWoK,
COMPS, or GlobalPIQA are weak, we need know whether there is a cheap legal repair
inside the already accepted compact-pair pool or whether new generation/source work is
needed.
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
from typing import Any

USER_ROOT = Path(".").resolve()
A01 = USER_ROOT / "experiments/archive/representation_and_objectives"
A02 = USER_ROOT / "experiments/archive/frontier_consolidation"
WORKSPACE = A01
A02_ANALYSIS = A02 / "data/medium_compact_analysis"
A02_DENSITY = A02 / "data/density_core_reinvestment_medium_riskhard"
ALL_ACCEPTED = A02_ANALYSIS / "medium_compact_ws_accepted_rewrites.jsonl"
SELECTED_REINVEST = A02_DENSITY / "selected_compact_reinvest_pairs.jsonl"
SELECTED_CORE = A02_DENSITY / "selected_compact_core_pairs.jsonl"
SELECTED_ADDED = A02_DENSITY / "selected_compact_added_pairs.jsonl"
OUT_DIR = WORKSPACE / "data/relation_frame_retention_audit"
OUT_JSON = OUT_DIR / "relation_frame_retention_audit.json"
OUT_DROP_CSV = OUT_DIR / "selected_relation_drop_examples.csv"
OUT_CAND_CSV = OUT_DIR / "unused_high_retention_candidate_examples.csv"
OUT_DOMAIN_CSV = OUT_DIR / "domain_category_retention_table.csv"
NOTE = (USER_ROOT / 'research/notes/representation_and_objectives/relation_frame_retention_audit.md')

WORD_RE = re.compile(r"[A-Za-z]+(?:[-'][A-Za-z]+)?|\d+(?:\.\d+)?")

# Broad relation-frame lexicon, deliberately not a copy of benchmark rows.  It is used
# only to audit existing source/rewrite pairs.  Future training-data construction must
# select by these source/rewrite properties, not by scored evaluation examples.
CUE_LEXICON: dict[str, list[str]] = {
    "spatial_state": [
        "above", "below", "under", "over", "inside", "outside", "within", "between",
        "beside", "behind", "front", "near", "far", "across", "through", "around",
        "along", "onto", "into", "beneath", "underneath", "surface", "center", "centre",
        "edge", "top", "bottom", "left", "right", "north", "south", "east", "west",
        "upstream", "downstream", "surround", "contains", "located", "adjacent", "opposite",
    ],
    "physical_dynamics": [
        "move", "moved", "moving", "fall", "falls", "fell", "rise", "rises", "rose",
        "sink", "sinks", "float", "floats", "flow", "flows", "freeze", "freezes", "melt",
        "melts", "boil", "boils", "evaporate", "expand", "contract", "rotate", "spin",
        "accelerate", "decelerate", "vibrate", "collide", "bounce", "roll", "slide",
        "drop", "drops", "dropped", "lift", "lifts", "falling", "rising", "shift", "shifted",
    ],
    "physical_interaction": [
        "push", "pull", "touch", "hit", "kick", "throw", "catch", "break", "breaks", "broke",
        "bend", "tear", "cut", "crush", "press", "strike", "grip", "hold", "carry", "pour",
        "stir", "mix", "rub", "scratch", "open", "close", "attach", "detach", "connect",
        "separate", "damage", "destroy", "repair", "turn", "switch", "insert", "remove",
    ],
    "material_property": [
        "hard", "soft", "rigid", "flexible", "solid", "liquid", "gas", "metal", "wooden",
        "plastic", "glass", "rubber", "wet", "dry", "smooth", "rough", "sharp", "dull",
        "heavy", "light", "hot", "cold", "warm", "cool", "dense", "thin", "thick",
        "strong", "weak", "fragile", "sticky", "slippery", "transparent", "opaque",
    ],
    "social_relation": [
        "parent", "child", "mother", "father", "son", "daughter", "brother", "sister",
        "family", "teacher", "student", "pupil", "boss", "manager", "worker", "employee",
        "employer", "subordinate", "leader", "follower", "friend", "enemy", "neighbor",
        "owner", "tenant", "landlord", "doctor", "patient", "coach", "player", "judge",
        "lawyer", "customer", "seller", "buyer", "husband", "wife", "partner", "colleague",
    ],
    "mental_state_social": [
        "believe", "believes", "believed", "doubt", "doubts", "know", "knows", "knew",
        "think", "thinks", "thought", "remember", "remembers", "forget", "forgets", "want",
        "wants", "need", "needs", "like", "likes", "love", "loves", "hate", "hates",
        "prefer", "prefers", "allow", "allows", "forbid", "forbids", "decide", "decides",
        "ask", "asks", "tell", "tells", "said", "says", "explain", "warn", "promise",
    ],
    "causal_temporal": [
        "because", "cause", "causes", "caused", "due", "result", "results", "therefore", "so",
        "hence", "lead", "leads", "led", "produce", "produces", "created", "creates", "make",
        "makes", "made", "if", "then", "when", "while", "before", "after", "during", "since",
        "until", "once", "following", "become", "became", "becomes", "prevent", "prevents",
        "enable", "enables", "affect", "affects", "depends", "requires", "trigger", "triggers",
    ],
    "negation_modality": [
        "not", "no", "never", "none", "without", "unless", "cannot", "can't", "won't", "don't",
        "doesn't", "didn't", "isn't", "aren't", "wasn't", "weren't", "may", "might", "must",
        "should", "could", "would", "possible", "impossible", "likely", "unlikely", "except",
    ],
    "comparison_order": [
        "more", "less", "most", "least", "larger", "smaller", "bigger", "older", "younger",
        "higher", "lower", "better", "worse", "hotter", "colder", "heavier", "lighter",
        "first", "last", "next", "previous", "same", "different", "opposite", "equal", "than",
        "increase", "increases", "decrease", "decreases", "greater", "lesser", "fewer",
    ],
    "quantity_measure": [
        "one", "two", "three", "four", "five", "many", "few", "several", "each", "every", "all",
        "some", "half", "twice", "percent", "percentage", "number", "amount", "total", "rate",
        "distance", "length", "height", "weight", "temperature", "speed", "volume", "mass",
    ],
}

MULTI_CUES: dict[str, list[tuple[str, ...]]] = {
    "spatial_state": [("in", "front"), ("next", "to"), ("on", "top"), ("far", "from")],
    "causal_temporal": [("as", "a", "result"), ("due", "to"), ("in", "order"), ("so", "that")],
    "comparison_order": [("greater", "than"), ("less", "than"), ("more", "than"), ("different", "from")],
    "negation_modality": [("not", "only"), ("no", "longer")],
}


def now_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def rel(p: Path | str) -> str:
    p = Path(p)
    try:
        return str(p.relative_to(USER_ROOT))
    except Exception:
        return str(p)


def norm_token(t: str) -> str:
    t = t.lower().strip()
    # Light normalization for inflected relation words not already enumerated.
    if len(t) > 5 and t.endswith("ing"):
        stem = t[:-3]
        if stem:
            return stem
    if len(t) > 4 and t.endswith("ed"):
        stem = t[:-2]
        if stem:
            return stem
    if len(t) > 4 and t.endswith("s") and not t.endswith("ss"):
        return t[:-1]
    return t


def text_tokens(text: str) -> list[str]:
    return [norm_token(m.group(0)) for m in WORD_RE.finditer(text)]


def phrase_count(tokens: list[str], phrase: tuple[str, ...]) -> int:
    n = len(phrase)
    if n == 0 or len(tokens) < n:
        return 0
    return sum(1 for i in range(len(tokens) - n + 1) if tuple(tokens[i:i+n]) == phrase)


def cue_counts(text: str) -> dict[str, int]:
    toks = text_tokens(text)
    c = Counter(toks)
    out: dict[str, int] = {}
    for cat, words in CUE_LEXICON.items():
        total = 0
        for w in words:
            total += c.get(norm_token(w), 0)
        for ph in MULTI_CUES.get(cat, []):
            total += phrase_count(toks, tuple(norm_token(x) for x in ph))
        out[cat] = total
    return out


def pair_id_from_obj(obj: dict[str, Any]) -> str:
    if obj.get("pair_id"):
        return str(obj["pair_id"])
    return "compact:" + str(obj.get("prompt_id"))


def primary_domain(obj: dict[str, Any]) -> str:
    hits = obj.get("domain_hits") or []
    if isinstance(hits, list) and hits:
        return str(hits[0])
    return "no_domain"


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def load_selected_ids(path: Path) -> set[str]:
    ids = set()
    for obj in read_jsonl(path):
        ids.add(pair_id_from_obj(obj))
    return ids


def enrich_pair(obj: dict[str, Any], selected_reinvest: set[str], selected_core: set[str], selected_added: set[str]) -> dict[str, Any]:
    pid = pair_id_from_obj(obj)
    src = str(obj.get("source_text", ""))
    rew = str(obj.get("rewrite_text", ""))
    sc = cue_counts(src)
    rc = cue_counts(rew)
    src_cats = [k for k, v in sc.items() if v > 0]
    rew_cats = [k for k, v in rc.items() if v > 0]
    dropped = [k for k in src_cats if rc.get(k, 0) <= 0]
    gained = [k for k in rew_cats if sc.get(k, 0) <= 0]
    retained = [k for k in src_cats if rc.get(k, 0) > 0]
    source_total = sum(sc.values())
    rewrite_total = sum(rc.values())
    retention_frac = len(retained) / len(src_cats) if src_cats else None
    count_retention = min(rewrite_total / source_total, 1.0) if source_total else None
    # Relation-loss score weights loss of whole categories more than lower repeated counts.
    loss_score = 0.0
    if src_cats:
        loss_score += len(dropped)
        loss_score += max(0, source_total - rewrite_total) / max(1, source_total)
        loss_score += 0.35 * max(0.0, 0.70 - float(retention_frac))
    return {
        "pair_id": pid,
        "key": obj.get("key") or f"sid:{obj.get('sentence_id')}|doc:{obj.get('doc_id')}",
        "sentence_id": obj.get("sentence_id"),
        "doc_id": obj.get("doc_id"),
        "source_text": src,
        "rewrite_text": rew,
        "source_words": int(obj.get("source_words", len(src.split()))),
        "rewrite_words": int(obj.get("rewrite_words", len(rew.split()))),
        "pair_words": int(obj.get("pair_words", len(src.split()) + len(rew.split()))),
        "length_ratio": float(obj.get("length_ratio", len(rew.split()) / max(1, len(src.split())))),
        "content_recall": float(obj.get("content_recall", 0.0) or 0.0),
        "content_overlap": float(obj.get("content_overlap", 0.0) or 0.0),
        "entity_recall": float(obj.get("entity_recall", 1.0) or 0.0),
        "number_recall": float(obj.get("number_recall", 1.0) or 0.0),
        "domain_hits": obj.get("domain_hits") or [],
        "primary_domain": primary_domain(obj),
        "selected_reinvest": pid in selected_reinvest,
        "selected_core": pid in selected_core,
        "selected_added": pid in selected_added,
        "unused_accepted": pid not in selected_reinvest,
        "source_cue_counts": sc,
        "rewrite_cue_counts": rc,
        "source_categories": src_cats,
        "rewrite_categories": rew_cats,
        "retained_source_categories": retained,
        "dropped_source_categories": dropped,
        "gained_rewrite_categories": gained,
        "n_source_categories": len(src_cats),
        "n_rewrite_categories": len(rew_cats),
        "source_cue_total": source_total,
        "rewrite_cue_total": rewrite_total,
        "category_retention_frac": retention_frac,
        "count_retention_capped": count_retention,
        "n_dropped_categories": len(dropped),
        "relation_loss_score": loss_score,
    }


def mean(xs: list[float]) -> float | None:
    xs = [x for x in xs if x is not None and math.isfinite(float(x))]
    return statistics.mean(xs) if xs else None


def qtiles(xs: list[float]) -> dict[str, float | None]:
    xs = sorted(float(x) for x in xs if x is not None and math.isfinite(float(x)))
    if not xs:
        return {"p10": None, "p25": None, "p50": None, "p75": None, "p90": None}
    def q(p: float) -> float:
        if len(xs) == 1:
            return xs[0]
        pos = p * (len(xs) - 1)
        lo = int(math.floor(pos)); hi = int(math.ceil(pos))
        if lo == hi:
            return xs[lo]
        return xs[lo] * (hi - pos) + xs[hi] * (pos - lo)
    return {"p10": q(0.10), "p25": q(0.25), "p50": q(0.50), "p75": q(0.75), "p90": q(0.90)}


def summarize(rows: list[dict[str, Any]], label: str) -> dict[str, Any]:
    n = len(rows)
    if not rows:
        return {"label": label, "n": 0}
    with_src = [r for r in rows if r["n_source_categories"] > 0]
    cat_drop = Counter()
    cat_src = Counter()
    cat_rew = Counter()
    domain_counts = Counter()
    for r in rows:
        domain_counts[r["primary_domain"]] += 1
        for c in r["source_categories"]:
            cat_src[c] += 1
        for c in r["rewrite_categories"]:
            cat_rew[c] += 1
        for c in r["dropped_source_categories"]:
            cat_drop[c] += 1
    return {
        "label": label,
        "n": n,
        "pair_words": sum(r["pair_words"] for r in rows),
        "mean_pair_words": mean([r["pair_words"] for r in rows]),
        "mean_source_words": mean([r["source_words"] for r in rows]),
        "mean_rewrite_words": mean([r["rewrite_words"] for r in rows]),
        "mean_length_ratio": mean([r["length_ratio"] for r in rows]),
        "mean_content_recall": mean([r["content_recall"] for r in rows]),
        "mean_entity_recall": mean([r["entity_recall"] for r in rows]),
        "mean_number_recall": mean([r["number_recall"] for r in rows]),
        "source_relation_pair_frac": len(with_src) / n,
        "mean_source_categories": mean([r["n_source_categories"] for r in rows]),
        "mean_rewrite_categories": mean([r["n_rewrite_categories"] for r in rows]),
        "mean_source_cue_total": mean([r["source_cue_total"] for r in rows]),
        "mean_rewrite_cue_total": mean([r["rewrite_cue_total"] for r in rows]),
        "mean_category_retention_frac_on_source_rel_pairs": mean([r["category_retention_frac"] for r in with_src]),
        "category_retention_quantiles_on_source_rel_pairs": qtiles([r["category_retention_frac"] for r in with_src if r["category_retention_frac"] is not None]),
        "drop_any_source_category_frac_on_source_rel_pairs": sum(1 for r in with_src if r["n_dropped_categories"] > 0) / max(1, len(with_src)),
        "drop_majority_source_categories_frac_on_source_rel_pairs": sum(1 for r in with_src if r["category_retention_frac"] is not None and r["category_retention_frac"] < 0.5) / max(1, len(with_src)),
        "mean_relation_loss_score": mean([r["relation_loss_score"] for r in rows]),
        "top_source_categories": dict(cat_src.most_common(12)),
        "top_rewrite_categories": dict(cat_rew.most_common(12)),
        "top_dropped_categories": dict(cat_drop.most_common(12)),
        "primary_domain_counts": dict(domain_counts.most_common(12)),
    }


def domain_category_table(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by = defaultdict(list)
    for r in rows:
        for cat in r["source_categories"] or ["<no_source_relation>"]:
            by[(r["primary_domain"], cat)].append(r)
    table = []
    for (dom, cat), rs in by.items():
        if cat == "<no_source_relation>":
            continue
        n = len(rs)
        table.append({
            "primary_domain": dom,
            "category": cat,
            "n_pairs": n,
            "pair_words": sum(r["pair_words"] for r in rs),
            "drop_frac": sum(1 for r in rs if cat in r["dropped_source_categories"]) / n,
            "mean_category_retention": mean([r["category_retention_frac"] for r in rs if r["category_retention_frac"] is not None]),
            "mean_content_recall": mean([r["content_recall"] for r in rs]),
            "selected_pair_frac": sum(1 for r in rs if r["selected_reinvest"]) / n,
        })
    table.sort(key=lambda r: (-r["drop_frac"], -r["n_pairs"], r["primary_domain"], r["category"]))
    return table


def repair_pool_feasibility(rows: list[dict[str, Any]]) -> dict[str, Any]:
    selected = [r for r in rows if r["selected_reinvest"]]
    unused = [r for r in rows if not r["selected_reinvest"]]
    bad = [
        r for r in selected
        if r["n_source_categories"] >= 2 and r["category_retention_frac"] is not None and r["category_retention_frac"] < 0.5 and r["content_recall"] >= 0.50 and r["entity_recall"] >= 0.99 and r["number_recall"] >= 0.99
    ]
    good_unused = [
        r for r in unused
        if r["n_source_categories"] >= 2 and r["category_retention_frac"] is not None and r["category_retention_frac"] >= 0.75 and r["content_recall"] >= 0.55 and r["entity_recall"] >= 0.99 and r["number_recall"] >= 0.99
    ]
    by_bad_dom = defaultdict(list)
    by_good_dom = defaultdict(list)
    for r in bad:
        by_bad_dom[r["primary_domain"]].append(r)
    for r in good_unused:
        by_good_dom[r["primary_domain"]].append(r)
    dom = {}
    for d in sorted(set(by_bad_dom) | set(by_good_dom)):
        b = by_bad_dom.get(d, [])
        g = by_good_dom.get(d, [])
        dom[d] = {
            "bad_selected_rows": len(b),
            "bad_selected_pair_words": sum(r["pair_words"] for r in b),
            "good_unused_rows": len(g),
            "good_unused_pair_words": sum(r["pair_words"] for r in g),
            "same_domain_pair_word_coverable": min(sum(r["pair_words"] for r in b), sum(r["pair_words"] for r in g)),
        }
    return {
        "bad_selected_definition": "selected reinvest pair with >=2 source relation categories, category retention <0.5, content_recall>=0.50, entity/number recall>=0.99",
        "good_unused_definition": "unused accepted pair with >=2 source relation categories, category retention >=0.75, content_recall>=0.55, entity/number recall>=0.99",
        "bad_selected_rows": len(bad),
        "bad_selected_pair_words": sum(r["pair_words"] for r in bad),
        "good_unused_rows": len(good_unused),
        "good_unused_pair_words": sum(r["pair_words"] for r in good_unused),
        "same_domain_coverable_pair_words_total": sum(v["same_domain_pair_word_coverable"] for v in dom.values()),
        "by_primary_domain": dom,
        "bad_selected_top_examples": [slim_pair(r, include_text=True) for r in sorted(bad, key=lambda x: (-x["relation_loss_score"], x["pair_words"]))[:20]],
        "good_unused_top_examples": [slim_pair(r, include_text=True) for r in sorted(good_unused, key=lambda x: (-x["category_retention_frac"], -x["content_recall"], x["pair_words"]))[:20]],
    }


def slim_pair(r: dict[str, Any], include_text: bool = False) -> dict[str, Any]:
    out = {
        "pair_id": r["pair_id"],
        "primary_domain": r["primary_domain"],
        "domain_hits": r["domain_hits"],
        "pair_words": r["pair_words"],
        "content_recall": r["content_recall"],
        "length_ratio": r["length_ratio"],
        "source_categories": r["source_categories"],
        "rewrite_categories": r["rewrite_categories"],
        "dropped_source_categories": r["dropped_source_categories"],
        "category_retention_frac": r["category_retention_frac"],
        "relation_loss_score": r["relation_loss_score"],
    }
    if include_text:
        out["source_text"] = r["source_text"]
        out["rewrite_text"] = r["rewrite_text"]
    return out


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            rec = {}
            for k in fields:
                v = r.get(k)
                if isinstance(v, (list, dict)):
                    v = json.dumps(v, ensure_ascii=False)
                rec[k] = v
            w.writerow(rec)


def write_note(payload: dict[str, Any]) -> None:
    summaries = {r["label"]: r for r in payload["group_summaries"]}
    sel = summaries.get("selected_reinvest_all", {})
    unused = summaries.get("unused_accepted", {})
    feas = payload["repair_pool_feasibility"]
    lines = ["# research relation-frame retention audit\n\n"]
    lines.append("This CPU-only audit inspects existing Qwen compact source/rewrite pairs. It does not score models and does not use benchmark rows to select training text.\n\n")
    lines.append("## Core measurements\n\n")
    lines.append(f"- Accepted compact pairs scanned: `{payload['n_all_accepted']}`; selected reinvest pairs recovered: `{payload['n_selected_reinvest']}`.\n")
    lines.append(f"- Selected reinvest source-relation pair fraction: `{sel.get('source_relation_pair_frac'):.4f}`; unused accepted: `{unused.get('source_relation_pair_frac'):.4f}`.\n")
    lines.append(f"- Selected mean source categories: `{sel.get('mean_source_categories'):.3f}`, rewrite categories: `{sel.get('mean_rewrite_categories'):.3f}`; mean category retention on source-relational pairs: `{sel.get('mean_category_retention_frac_on_source_rel_pairs'):.4f}`.\n")
    lines.append(f"- Selected source-relational pairs that drop at least one source category: `{sel.get('drop_any_source_category_frac_on_source_rel_pairs'):.4f}`; drop majority: `{sel.get('drop_majority_source_categories_frac_on_source_rel_pairs'):.4f}`.\n")
    lines.append(f"- Top selected dropped categories: `{sel.get('top_dropped_categories')}`.\n")
    lines.append("\n## Repair-pool feasibility inside existing accepted pairs\n\n")
    lines.append(f"- Bad selected definition: {feas['bad_selected_definition']}.\n")
    lines.append(f"- Bad selected rows/pair-words: `{feas['bad_selected_rows']}` / `{feas['bad_selected_pair_words']}`.\n")
    lines.append(f"- Good unused rows/pair-words: `{feas['good_unused_rows']}` / `{feas['good_unused_pair_words']}`.\n")
    lines.append(f"- Same-primary-domain coverable bad pair-words: `{feas['same_domain_coverable_pair_words_total']}`.\n")
    lines.append("\n## Interpretation\n\n")
    for x in payload["interpretation"]:
        lines.append(f"- {x}\n")
    lines.append("\n## Files\n\n")
    for k, v in payload["files"].items():
        lines.append(f"- {k}: `{v}`\n")
    NOTE.write_text("".join(lines), encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    selected_reinvest = load_selected_ids(SELECTED_REINVEST)
    selected_core = load_selected_ids(SELECTED_CORE)
    selected_added = load_selected_ids(SELECTED_ADDED)
    rows = []
    for obj in read_jsonl(ALL_ACCEPTED):
        rows.append(enrich_pair(obj, selected_reinvest, selected_core, selected_added))
    # Sanity: every selected reinvest pair should be in the accepted pool.
    ids_all = {r["pair_id"] for r in rows}
    missing_selected = sorted(selected_reinvest - ids_all)
    selected = [r for r in rows if r["selected_reinvest"]]
    core = [r for r in rows if r["selected_core"]]
    added = [r for r in rows if r["selected_added"]]
    unused = [r for r in rows if not r["selected_reinvest"]]
    group_summaries = [
        summarize(rows, "all_accepted"),
        summarize(selected, "selected_reinvest_all"),
        summarize(core, "selected_core"),
        summarize(added, "selected_added"),
        summarize(unused, "unused_accepted"),
    ]
    dom_table = domain_category_table(selected)
    feas = repair_pool_feasibility(rows)
    selected_drop_examples = [
        r for r in selected
        if r["n_source_categories"] >= 1 and r["n_dropped_categories"] > 0
    ]
    selected_drop_examples.sort(key=lambda r: (-r["relation_loss_score"], -r["n_source_categories"], r["pair_words"]))
    unused_candidates = [
        r for r in unused
        if r["n_source_categories"] >= 2 and r["category_retention_frac"] is not None and r["category_retention_frac"] >= 0.75 and r["entity_recall"] >= 0.99 and r["number_recall"] >= 0.99
    ]
    unused_candidates.sort(key=lambda r: (-r["category_retention_frac"], -r["content_recall"], -r["n_source_categories"], r["pair_words"]))

    csv_fields = [
        "pair_id", "primary_domain", "domain_hits", "pair_words", "source_words", "rewrite_words", "length_ratio",
        "content_recall", "entity_recall", "number_recall", "n_source_categories", "n_rewrite_categories",
        "source_categories", "rewrite_categories", "dropped_source_categories", "category_retention_frac",
        "relation_loss_score", "source_text", "rewrite_text",
    ]
    write_csv(OUT_DROP_CSV, selected_drop_examples[:200], csv_fields)
    write_csv(OUT_CAND_CSV, unused_candidates[:200], csv_fields)
    write_csv(OUT_DOMAIN_CSV, dom_table, ["primary_domain", "category", "n_pairs", "pair_words", "drop_frac", "mean_category_retention", "mean_content_recall", "selected_pair_frac"])

    selected_summary = group_summaries[1]
    unused_summary = group_summaries[4]
    interpretations = [
        "Relation-frame retention is measured from source/rewrite pairs, not from official score items. It is therefore suitable as a legal pretraining-data repair signal if future scores show relation weakness.",
    ]
    if selected_summary.get("drop_any_source_category_frac_on_source_rel_pairs", 0) > 0.25:
        interpretations.append("A substantial fraction of selected source-relational compact pairs drop at least one broad relation category in the rewrite; compact density may therefore remove some relational framing even while preserving entities and numbers.")
    else:
        interpretations.append("Selected compact rewrites do not show broad relation-category deletion under this coarse lexicon; relation instability would more likely come from optimization/seed dynamics or subtler context geometry.")
    if feas["good_unused_pair_words"] > 0:
        interpretations.append("There is an unused accepted-pair pool with high relation-category retention; a future CPU-only corpus dry run can attempt same-budget replacement before any new Qwen generation or GPU training is authorized.")
    else:
        interpretations.append("The unused accepted-pair pool contains little high-retention relational material; if corrected official results require repair, new source selection or teacher generation would be more plausible than reshuffling current accepted pairs.")
    if selected_summary.get("mean_category_retention_frac_on_source_rel_pairs") is not None and unused_summary.get("mean_category_retention_frac_on_source_rel_pairs") is not None:
        diff = selected_summary["mean_category_retention_frac_on_source_rel_pairs"] - unused_summary["mean_category_retention_frac_on_source_rel_pairs"]
        interpretations.append(f"Selected-minus-unused mean category-retention difference is {diff:+.4f}; this indicates whether the current reinvest selector already favored or disfavored relation-preserving pairs.")

    payload = {
        "status": "RELATION_FRAME_RETENTION_AUDIT",
        "created_utc": now_utc(),
        "method": {
            "no_model_inference": True,
            "no_benchmark_text_used_for_training_selection": True,
            "cue_categories": sorted(CUE_LEXICON),
            "lexicon_limit": "Coarse lexical relation-frame retention; misses paraphrased or implicit relations and should guide, not replace, real official evaluation.",
        },
        "inputs": {
            "all_accepted_pairs": rel(ALL_ACCEPTED),
            "selected_reinvest_pairs": rel(SELECTED_REINVEST),
            "selected_core_pairs": rel(SELECTED_CORE),
            "selected_added_pairs": rel(SELECTED_ADDED),
        },
        "n_all_accepted": len(rows),
        "n_selected_reinvest": len(selected_reinvest),
        "n_selected_reinvest_recovered_in_all_accepted": len(selected),
        "n_missing_selected_from_accepted": len(missing_selected),
        "missing_selected_sample": missing_selected[:20],
        "group_summaries": group_summaries,
        "domain_category_retention_table_top_by_drop": dom_table[:80],
        "repair_pool_feasibility": feas,
        "interpretation": interpretations,
        "files": {
            "json": rel(OUT_JSON),
            "selected_relation_drop_examples": rel(OUT_DROP_CSV),
            "unused_high_retention_candidates": rel(OUT_CAND_CSV),
            "domain_category_retention_table": rel(OUT_DOMAIN_CSV),
            "note": rel(NOTE),
        },
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_note(payload)
    print(json.dumps({
        "status": payload["status"],
        "out_json": rel(OUT_JSON),
        "note": rel(NOTE),
        "n_all_accepted": len(rows),
        "n_selected_reinvest_recovered": len(selected),
        "selected_relation_pair_frac": selected_summary.get("source_relation_pair_frac"),
        "selected_mean_category_retention": selected_summary.get("mean_category_retention_frac_on_source_rel_pairs"),
        "selected_drop_any_frac": selected_summary.get("drop_any_source_category_frac_on_source_rel_pairs"),
        "bad_selected_pair_words": feas["bad_selected_pair_words"],
        "good_unused_pair_words": feas["good_unused_pair_words"],
        "same_domain_coverable_pair_words_total": feas["same_domain_coverable_pair_words_total"],
        "interpretation": interpretations,
    }, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
