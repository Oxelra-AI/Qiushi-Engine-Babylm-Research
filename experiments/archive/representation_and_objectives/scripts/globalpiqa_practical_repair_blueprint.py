#!/usr/bin/env python3
"""research: CPU-only practical-affordance repair blueprint for compact-view reinvest.

This repaired version uses the same compact-candidate eligibility as
`materialize_density_core_reinvestment.py` before ranking any unused
packet: accepted row, ratio in [0.35, 0.85], content recall >=0.45, no exact copy,
no source risks, pair_words <=160, and duplicate source keys collapsed by the same
quality rule.  It therefore asks whether the *valid unused compact-candidate bank*
contains exact-pair-word replacement material for a future GlobalPIQA repair.

It does not train, evaluate, generate text, or read managed research outputs.
"""
from __future__ import annotations

import collections
import hashlib
import json
import math
import pathlib
import re
import statistics
import string
from typing import Any, Iterable

ROOT = pathlib.Path(".").resolve()
A01 = ROOT / "experiments/archive" / 'representation_and_objectives'
A02 = ROOT / "experiments/archive" / 'frontier_consolidation'
COMPACT_ROWS = A02 / "data" / "medium_compact_analysis" / "medium_compact_ws_rows.jsonl"
SELECTED_REINVEST = A02 / "data" / "density_core_reinvestment_medium_riskhard" / "selected_compact_reinvest_pairs.jsonl"
DENSITY_META = A02 / "data" / "density_core_reinvestment_medium_riskhard" / "density_core_reinvestment_metadata.json"
GLOBALPIQA_ATLAS = A01 / "data" / "globalpiqa_failure_atlas" / "globalpiqa_failure_atlas.json"
OUT_DIR = A01 / "data" / "globalpiqa_practical_repair_blueprint"
OUT_JSON = OUT_DIR / "globalpiqa_practical_repair_blueprint.json"
OUT_CANDIDATES = OUT_DIR / "top_unselected_practical_compact_pairs.jsonl"
OUT_SWAPS = OUT_DIR / "exact_wordlength_practical_swap_plan.jsonl"
OUT_NOTE = (ROOT / 'research/notes/representation_and_objectives/globalpiqa_practical_repair_blueprint.md')

TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9'\-]*|\d+(?:\.\d+)?")
WORD_RE = re.compile(r"[A-Za-z0-9]+")
PUNCT = str.maketrans("", "", string.punctuation.replace("'", ""))
MAX_PACKET_WORDS = 160
COMPACT_MIN_RATIO = 0.35
COMPACT_MAX_RATIO = 0.85
COMPACT_MIN_RECALL = 0.45

# Generic mechanism families. These are broad everyday physical/action words, not
# harvested from GlobalPIQA prompts or choices.
LEXICONS: dict[str, set[str]] = {
    "affordance_tools": {
        "tool", "tools", "hammer", "screw", "screws", "nail", "nails", "drill", "wrench", "saw", "knife",
        "knives", "scissor", "scissors", "tape", "glue", "rope", "string", "wire", "cord", "chain", "strap",
        "ladder", "rack", "racks", "shelf", "shelves", "basket", "bucket", "broom", "brush", "sponge", "mop",
        "handle", "lid", "cap", "hook", "clip", "clamp", "pin", "needle", "thread", "shovel", "hose", "pump",
        "bike", "bicycle", "car", "wheel", "wheels", "furniture", "chair", "table", "door", "window", "jar",
        "box", "can", "bottle", "bag", "cup", "bowl", "plate", "pan", "pot", "oven", "stove", "cloth", "towel",
    },
    "action_outcomes": {
        "open", "opened", "opening", "close", "closed", "cut", "cuts", "slice", "chop", "peel", "pour", "fill",
        "filled", "empty", "seal", "sealed", "tighten", "loosen", "attach", "attached", "connect", "connected",
        "remove", "clean", "wash", "dry", "wrap", "cover", "tie", "carry", "hold", "lift", "push", "pull", "turn",
        "roll", "fold", "bend", "stretch", "hang", "place", "support", "protect", "block", "catch", "prevent",
        "repair", "fix", "make", "break", "crack", "split", "slide", "stick", "slip", "spill", "leak", "drop",
    },
    "material_response": {
        "plastic", "paper", "glass", "metal", "wood", "wooden", "rubber", "cloth", "fabric", "cotton", "wool",
        "leather", "water", "air", "oil", "sand", "clay", "ice", "snow", "heat", "hot", "cold", "freeze",
        "frozen", "melt", "boil", "burn", "fire", "steam", "soak", "absorb", "absorbs", "wet", "dry", "sticky",
        "smooth", "rough", "flexible", "hard", "soft", "brittle", "heavy", "light", "sharp", "dull", "float",
        "sink", "expand", "shrink", "dissolve", "liquid", "solid", "gas", "pressure", "friction", "gravity",
    },
    "spatial_container_support": {
        "inside", "outside", "above", "below", "under", "over", "between", "near", "across", "through", "around",
        "onto", "into", "surface", "edge", "corner", "top", "bottom", "front", "back", "side", "hole", "opening",
        "container", "room", "wall", "floor", "ceiling", "shelf", "rack", "stand", "support", "base", "cover",
    },
    "time_count_quantity": {
        "second", "seconds", "minute", "minutes", "hour", "hours", "day", "days", "week", "weeks", "before",
        "after", "first", "next", "last", "twice", "half", "double", "single", "count", "counts", "number",
        "numbers", "amount", "many", "few", "less", "more", "full", "empty", "increase", "decrease", "measure",
    },
    "everyday_food_body_home": {
        "food", "bread", "meat", "egg", "eggs", "milk", "salt", "sugar", "butter", "kitchen", "cook", "cooking",
        "eat", "drink", "shirt", "clothes", "shoe", "shoes", "body", "hand", "hands", "face", "hair", "skin",
        "home", "house", "bed", "bath", "bathroom", "garden", "garage", "laundry", "soap", "cleaner", "furniture",
    },
}
WEIGHTS = {
    "affordance_tools": 1.35,
    "action_outcomes": 1.35,
    "material_response": 1.25,
    "spatial_container_support": 0.95,
    "time_count_quantity": 0.9,
    "everyday_food_body_home": 0.85,
}


def read_jsonl(path: pathlib.Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def wc(text: str) -> int:
    return len(str(text or "").split())


def norm_for_copy(text: str) -> str:
    return " ".join(WORD_RE.findall((text or "").lower()))


def source_key_from_fields(sentence_id: Any, doc_id: Any, source_text: str) -> str:
    if sentence_id is not None and str(sentence_id) != "":
        return f"sid:{sentence_id}|doc:{doc_id}"
    return "txt:" + hashlib.sha1(norm_for_copy(source_text).encode("utf-8")).hexdigest()[:20]


def norm_tok(tok: str) -> str:
    tok = tok.lower().strip("'\"")
    if len(tok) > 4 and tok.endswith("ies"):
        tok = tok[:-3] + "y"
    elif len(tok) > 4 and tok.endswith("ves"):
        tok = tok[:-3] + "f"
    elif len(tok) > 3 and tok.endswith("s") and not tok.endswith("ss"):
        tok = tok[:-1]
    return tok


def tokens(text: str) -> set[str]:
    out = set()
    for raw in TOKEN_RE.findall(str(text or "").lower().translate(PUNCT)):
        t = norm_tok(raw)
        if len(t) >= 2:
            out.add(t)
    return out


def key_for(obj: dict[str, Any]) -> str:
    if obj.get("key"):
        return str(obj["key"])
    src = " ".join(str(obj.get("source_text") or "").split())
    return source_key_from_fields(obj.get("sentence_id"), obj.get("doc_id"), src)


def pair_id_for(obj: dict[str, Any]) -> str:
    v = obj.get("pair_id")
    if v:
        return str(v)
    p = obj.get("prompt_id")
    if p:
        return f"compact:{p}"
    return f"compact:{key_for(obj)}"


def pair_quality_tuple(r: dict[str, Any]) -> tuple[float, int, str]:
    return (float(r.get("content_recall") or 0.0), -int(r.get("pair_words") or 0), str(r.get("pair_id") or r.get("prompt_id") or ""))


def practical_features(source: str, rewrite: str) -> dict[str, Any]:
    toks = tokens(f"{source} {rewrite}")
    hits: dict[str, list[str]] = {}
    weighted = 0.0
    norm_lexicons = {fam: {norm_tok(x) for x in lex} for fam, lex in LEXICONS.items()}
    for fam, lex in norm_lexicons.items():
        fam_hits = sorted(toks & lex)
        if fam_hits:
            hits[fam] = fam_hits[:16]
            weighted += WEIGHTS[fam] * math.sqrt(len(fam_hits))
    family_count = len(hits)
    score = weighted + 1.15 * max(0, family_count - 1)
    has_action_core = bool(hits.get("affordance_tools") or hits.get("action_outcomes") or hits.get("material_response"))
    return {
        "practical_score": round(score, 6),
        "practical_family_count": family_count,
        "has_action_material_or_tool": has_action_core,
        "practical_hits": hits,
    }


def base_record_from_obj(obj: dict[str, Any]) -> dict[str, Any]:
    src = " ".join(str(obj.get("source_text") or "").split())
    rew = " ".join(str(obj.get("rewrite_text") or obj.get("raw_output") or "").split())
    sw = wc(src)
    rw = wc(rew)
    pair_words = sw + rw
    rec = {
        "key": key_for(obj),
        "pair_id": pair_id_for(obj),
        "prompt_id": str(obj.get("prompt_id") or obj.get("pair_id") or ""),
        "sentence_id": str(obj.get("sentence_id") or ""),
        "doc_id": str(obj.get("doc_id") or ""),
        "source_text": src,
        "rewrite_text": rew,
        "source_words": sw,
        "rewrite_words": rw,
        "pair_words": pair_words,
        "length_ratio": rw / max(1, sw),
        "content_recall": None if obj.get("content_recall") is None else float(obj.get("content_recall")),
        "content_overlap": None if obj.get("content_overlap") is None else float(obj.get("content_overlap")),
        "entity_recall": None if obj.get("entity_recall") is None else float(obj.get("entity_recall")),
        "number_recall": None if obj.get("number_recall") is None else float(obj.get("number_recall")),
        "domain_hits": [str(x) for x in (obj.get("domain_hits") or [])],
        "soft_flags": [str(x) for x in (obj.get("soft_flags") or [])],
        "source_risks": [str(x) for x in (obj.get("source_risks") or [])],
        "accepted_for_next_construction": bool(obj.get("accepted_for_next_construction", False)),
    }
    rec.update(practical_features(src, rew))
    return rec


def compact_filter_reason(r: dict[str, Any]) -> str | None:
    if not r.get("accepted_for_next_construction", False) or not r.get("source_text") or not r.get("rewrite_text"):
        return "not_accepted_or_malformed"
    if int(r["pair_words"]) > MAX_PACKET_WORDS:
        return "pair_too_long_for_seq256_packet"
    if r.get("source_risks"):
        return "source_risk"
    if norm_for_copy(r["source_text"]) == norm_for_copy(r["rewrite_text"]):
        return "exact_copy"
    ratio = float(r["length_ratio"])
    if ratio < COMPACT_MIN_RATIO or ratio > COMPACT_MAX_RATIO:
        return "outside_compact_ratio_band"
    if r.get("content_recall") is not None and float(r["content_recall"]) < COMPACT_MIN_RECALL:
        return "compact_low_content_recall"
    return None


def compact_candidate_map(rows: Iterable[dict[str, Any]]) -> tuple[dict[str, dict[str, Any]], collections.Counter[str]]:
    out: dict[str, dict[str, Any]] = {}
    reject: collections.Counter[str] = collections.Counter()
    for obj in rows:
        r = base_record_from_obj(obj)
        reason = compact_filter_reason(r)
        if reason:
            reject[reason] += 1
            continue
        old = out.get(r["key"])
        if old is None:
            out[r["key"]] = r
        else:
            if pair_quality_tuple(r) > pair_quality_tuple(old):
                out[r["key"]] = r
                reject["duplicate_source_replaced"] += 1
            else:
                reject["duplicate_source_dropped"] += 1
    return out, reject


def quantiles(vals: list[float]) -> dict[str, float | None]:
    if not vals:
        return {"n": 0, "mean": None, "median": None, "p90": None, "p95": None, "max": None}
    xs = sorted(vals)
    def q(p: float) -> float:
        idx = min(len(xs) - 1, max(0, int(round((len(xs) - 1) * p))))
        return xs[idx]
    return {
        "n": len(xs), "mean": round(sum(xs) / len(xs), 6), "median": round(statistics.median(xs), 6),
        "p90": round(q(0.90), 6), "p95": round(q(0.95), 6), "max": round(xs[-1], 6),
    }


def is_strong_practical(r: dict[str, Any]) -> bool:
    return (
        float(r["practical_score"]) >= 4.2
        and int(r["practical_family_count"]) >= 2
        and bool(r["has_action_material_or_tool"])
        and not r.get("source_risks")
        and float(r["length_ratio"]) <= COMPACT_MAX_RATIO
        and float(r["length_ratio"]) >= COMPACT_MIN_RATIO
        and (r.get("content_recall") is None or float(r["content_recall"]) >= 0.50)
        and (r.get("entity_recall") is None or float(r["entity_recall"]) >= 0.75)
        and (r.get("number_recall") is None or float(r["number_recall"]) >= 1.0)
    )


def removal_ok(r: dict[str, Any]) -> bool:
    # Prefer replacing low-practical packets with no selector domain tag, so the
    # current EWoK/Entity/Supplement-bearing distribution is disturbed as little
    # as possible. This is only a CPU blueprint, not proof of harmlessness.
    domains = set(r.get("domain_hits") or [])
    return (
        float(r["practical_score"]) <= 1.25
        and not domains
        and not r.get("source_risks")
        and int(r["pair_words"]) <= 80
    )


def compact_public_record(r: dict[str, Any]) -> dict[str, Any]:
    return {
        "key": r["key"], "pair_id": r["pair_id"], "doc_id": r["doc_id"], "sentence_id": r["sentence_id"],
        "pair_words": r["pair_words"], "source_words": r["source_words"], "rewrite_words": r["rewrite_words"],
        "length_ratio": r["length_ratio"], "content_recall": r["content_recall"], "content_overlap": r["content_overlap"],
        "entity_recall": r["entity_recall"], "number_recall": r["number_recall"], "domain_hits": r["domain_hits"],
        "practical_score": r["practical_score"], "practical_family_count": r["practical_family_count"],
        "practical_hits": r["practical_hits"], "source_text": r["source_text"], "rewrite_text": r["rewrite_text"],
    }


def summarize_pool(rows: list[dict[str, Any]]) -> dict[str, Any]:
    fam_counts: collections.Counter[str] = collections.Counter()
    domain_counts: collections.Counter[str] = collections.Counter()
    for r in rows:
        for fam in r["practical_hits"]:
            fam_counts[fam] += 1
        for d in r.get("domain_hits") or ["no_domain"]:
            domain_counts[d] += 1
    strong = [r for r in rows if is_strong_practical(r)]
    return {
        "pairs": len(rows),
        "pair_words": sum(int(r["pair_words"]) for r in rows),
        "unique_docs": len({r["doc_id"] for r in rows}),
        "practical_score": quantiles([float(r["practical_score"]) for r in rows]),
        "strong_practical_pairs": len(strong),
        "strong_practical_words": sum(int(r["pair_words"]) for r in strong),
        "strong_practical_fraction_pairs": round(len(strong) / len(rows), 6) if rows else None,
        "family_pair_counts": dict(fam_counts.most_common()),
        "domain_pair_counts_top": dict(domain_counts.most_common(12)),
    }


def build_exact_swap_plan(selected: list[dict[str, Any]], unselected: list[dict[str, Any]], max_swaps: int = 768) -> list[dict[str, Any]]:
    removals_by_len: dict[int, list[dict[str, Any]]] = collections.defaultdict(list)
    for r in selected:
        if removal_ok(r):
            removals_by_len[int(r["pair_words"])].append(r)
    for L in list(removals_by_len):
        removals_by_len[L].sort(key=lambda x: (float(x["practical_score"]), -float(x.get("content_recall") or 0.0), x["key"]))
    candidates = [r for r in unselected if is_strong_practical(r) and int(r["pair_words"]) <= 80]
    candidates.sort(key=lambda x: (float(x["practical_score"]), int(x["practical_family_count"]), -int(x["pair_words"]), x["key"]), reverse=True)
    used_docs: set[str] = set()
    plan: list[dict[str, Any]] = []
    deferred_same_doc: list[dict[str, Any]] = []
    for add in candidates:
        L = int(add["pair_words"])
        bucket = removals_by_len.get(L) or []
        if not bucket:
            continue
        if add["doc_id"] in used_docs:
            deferred_same_doc.append(add)
            continue
        rem = bucket.pop(0)
        used_docs.add(add["doc_id"])
        plan.append({"add": compact_public_record(add), "remove": compact_public_record(rem), "pair_word_delta": 0,
                     "score_delta": round(float(add["practical_score"]) - float(rem["practical_score"]), 6)})
        if len(plan) >= max_swaps:
            return plan
    for add in deferred_same_doc:
        L = int(add["pair_words"])
        bucket = removals_by_len.get(L) or []
        if not bucket:
            continue
        rem = bucket.pop(0)
        plan.append({"add": compact_public_record(add), "remove": compact_public_record(rem), "pair_word_delta": 0,
                     "score_delta": round(float(add["practical_score"]) - float(rem["practical_score"]), 6)})
        if len(plan) >= max_swaps:
            break
    return plan


def hash_paths(paths: list[pathlib.Path]) -> dict[str, str]:
    out = {}
    for p in paths:
        h = hashlib.sha256()
        with p.open("rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        out[str(p)] = h.hexdigest()
    return out


def write_jsonl(path: pathlib.Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_atlas_snapshot() -> dict[str, Any]:
    atlas = json.loads(GLOBALPIQA_ATLAS.read_text(encoding="utf-8")) if GLOBALPIQA_ATLAS.exists() else {}
    return {
        "source": str(GLOBALPIQA_ATLAS),
        "reinvest_globalpiqa_mean": atlas.get("score_summary", {}).get("compact_view_reinvest", {}).get("mean_parallel_nonparallel"),
        "reinvest_parallel": atlas.get("score_summary", {}).get("compact_view_reinvest", {}).get("parallel", {}).get("accuracy"),
        "reinvest_nonparallel": atlas.get("score_summary", {}).get("compact_view_reinvest", {}).get("nonparallel", {}).get("accuracy"),
        "parallel_all_three_wrong": atlas.get("flips", {}).get("parallel", {}).get("all_three_wrong", {}).get("count"),
        "nonparallel_all_three_wrong": atlas.get("flips", {}).get("nonparallel", {}).get("all_three_wrong", {}).get("count"),
        "ranking_uses_exact_globalpiqa_text": False,
    }


def mean_field(rows: list[dict[str, Any]], field: str) -> float | None:
    vals = [float(r[field]) for r in rows if r.get(field) is not None]
    return sum(vals) / len(vals) if vals else None


def write_note(payload: dict[str, Any]) -> None:
    s = payload["pool_summary"]
    plan = payload["swap_plan_summary"]
    lines: list[str] = []
    lines.append("# research GlobalPIQA practical-affordance repair blueprint")
    lines.append("")
    lines.append("This repaired CPU-only pass asks whether the valid unused compact-candidate bank contains source-view packets for a future GlobalPIQA repair. It does not train, evaluate, generate text, or read the running research managed outputs.")
    lines.append("")
    lines.append("The first draft accidentally counted all accepted analysis rows, including rows outside the compact selector's ratio band. This repaired version applies the same compact-candidate rules used by A02's materializer before any practical ranking: ratio 0.35--0.85, content recall >=0.45, no exact copy, no source risk, <=160 words, and duplicate source keys collapsed by content recall/shortness. The repaired candidate count matches the A02 materialization metadata.")
    lines.append("")
    lines.append("The practical ranking uses a broad generic mechanism lexicon for tools, everyday actions, material response, spatial/container/support relations, time/counting, and household/body/food contexts. It does not use exact GlobalPIQA prompt or option text to rank training candidates; the finished fast GlobalPIQA results are used only to identify the weak scientific area.")
    lines.append("")
    lines.append("## Candidate-bank measurement")
    lines.append("")
    lines.append(f"- Valid compact candidates after selector-equivalent filtering: {s['all_compact_candidates']['pairs']} pairs / {s['all_compact_candidates']['pair_words']} pair words.")
    lines.append(f"- Current reinvest selection matched inside that bank: {s['selected_reinvest']['pairs']} pairs / {s['selected_reinvest']['pair_words']} words / {s['selected_reinvest']['unique_docs']} docs.")
    lines.append(f"- Valid unused compact candidates: {s['unselected_compact_candidates']['pairs']} pairs / {s['unselected_compact_candidates']['pair_words']} words / {s['unselected_compact_candidates']['unique_docs']} docs.")
    lines.append(f"- Strong generic practical packets already selected: {s['selected_reinvest']['strong_practical_pairs']} pairs / {s['selected_reinvest']['strong_practical_words']} words ({s['selected_reinvest']['strong_practical_fraction_pairs']:.3f} of selected pairs).")
    lines.append(f"- Strong generic practical packets still unused: {s['unselected_compact_candidates']['strong_practical_pairs']} pairs / {s['unselected_compact_candidates']['strong_practical_words']} words ({s['unselected_compact_candidates']['strong_practical_fraction_pairs']:.3f} of unused valid candidates).")
    lines.append("")
    lines.append("## Exact-length replacement plan")
    lines.append("")
    lines.append(f"- Exact pair-word swap plan: {plan['replacement_count']} replacements; added words {plan['add_words']}, removed words {plan['remove_words']}, net word change {plan['net_pair_words']}.")
    lines.append(f"- Mean practical score rises from {plan['removed_practical_score_mean']:.3f} in removed packets to {plan['added_practical_score_mean']:.3f} in added packets.")
    lines.append(f"- Added packets cover {plan['added_unique_docs']} documents; removed packets cover {plan['removed_unique_docs']} documents.")
    lines.append(f"- Added packet means: content recall {plan['added_content_recall_mean']:.3f}, entity recall {plan['added_entity_recall_mean']:.3f}, number recall {plan['added_number_recall_mean']:.3f}, length ratio {plan['added_length_ratio_mean']:.3f}.")
    lines.append("")
    lines.append("Top practical families in added packets:")
    for fam, count in list(plan["added_family_counts"].items())[:8]:
        lines.append(f"- {fam}: {count}")
    lines.append("")
    lines.append("## Example added packets")
    lines.append("")
    for cand in payload["top_unselected_practical_examples"][:8]:
        hits = ", ".join(cand["practical_hits"].keys())
        lines.append(f"- score {cand['practical_score']:.3f}, ratio {cand['length_ratio']:.3f}, {cand['pair_words']} words, families [{hits}] — {cand['source_text']} / {cand['rewrite_text']}")
    lines.append("")
    lines.append("## Scientific reading")
    lines.append("")
    lines.append("The repaired result is smaller but cleaner than the draft: after enforcing the actual compact-candidate contract, the unused reserve still contains a finite set of compact, anchor-preserving practical-action packets. Exact pair-word replacement is possible without new generation and without changing the 423,520-word changed-block budget.")
    lines.append("This is only a repair blueprint. It does not establish that practical lexical density causes GlobalPIQA improvement, and it does not check tokenizer exposure, seq256 packing, contamination, or tradeoffs with EWoK/Entity/Supplement. If the pending endpoint and seed evidence later justify a repair run, the next CPU work should materialize the swapped corpus in dry-run mode and audit tokenizer length, source:rewrite ratio, joint visibility, duplicate/near-duplicate rates, and overlap with GlobalPIQA prompts/options before any GPU training.")
    lines.append("")
    lines.append(f"JSON: `{OUT_JSON}`")
    lines.append(f"Top valid unused candidates: `{OUT_CANDIDATES}`")
    lines.append(f"Exact pair-word swap plan: `{OUT_SWAPS}`")
    OUT_NOTE.parent.mkdir(parents=True, exist_ok=True)
    OUT_NOTE.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    density_meta = json.loads(DENSITY_META.read_text(encoding="utf-8"))
    selected_keys = {key_for(obj) for obj in read_jsonl(SELECTED_REINVEST)}
    compact_map, reject = compact_candidate_map(read_jsonl(COMPACT_ROWS))
    all_candidates = list(compact_map.values())
    selected = [compact_map[k] for k in selected_keys if k in compact_map]
    selected.sort(key=lambda r: r["key"])
    unselected = [r for k, r in compact_map.items() if k not in selected_keys]

    expected_compact_candidates = int(density_meta.get("candidate_counts", {}).get("compact_candidates", -1))
    if expected_compact_candidates != len(compact_map):
        raise RuntimeError(f"compact candidate count {len(compact_map)} != metadata {expected_compact_candidates}")
    if len(selected) != len(selected_keys):
        missing = sorted(selected_keys - set(compact_map))[:10]
        raise RuntimeError(f"selected keys missing from compact map: {missing}")

    top_unselected = [r for r in unselected if is_strong_practical(r)]
    top_unselected.sort(key=lambda x: (float(x["practical_score"]), int(x["practical_family_count"]), int(x["pair_words"]), x["key"]), reverse=True)
    swap_plan = build_exact_swap_plan(selected, unselected)
    added = [x["add"] for x in swap_plan]
    removed = [x["remove"] for x in swap_plan]
    fam_counts: collections.Counter[str] = collections.Counter()
    for r in added:
        for fam in r["practical_hits"]:
            fam_counts[fam] += 1
    plan_summary = {
        "replacement_count": len(swap_plan),
        "add_words": sum(int(r["pair_words"]) for r in added),
        "remove_words": sum(int(r["pair_words"]) for r in removed),
        "net_pair_words": sum(int(r["pair_words"]) for r in added) - sum(int(r["pair_words"]) for r in removed),
        "added_unique_docs": len({r["doc_id"] for r in added}),
        "removed_unique_docs": len({r["doc_id"] for r in removed}),
        "added_practical_score_mean": round(mean_field(added, "practical_score") or 0.0, 6),
        "removed_practical_score_mean": round(mean_field(removed, "practical_score") or 0.0, 6),
        "added_content_recall_mean": round(mean_field(added, "content_recall") or 0.0, 6),
        "added_entity_recall_mean": round(mean_field(added, "entity_recall") or 0.0, 6),
        "added_number_recall_mean": round(mean_field(added, "number_recall") or 0.0, 6),
        "added_length_ratio_mean": round(mean_field(added, "length_ratio") or 0.0, 6),
        "removed_length_ratio_mean": round(mean_field(removed, "length_ratio") or 0.0, 6),
        "added_family_counts": dict(fam_counts.most_common()),
    }
    payload = {
        "status": "GLOBALPIQA_PRACTICAL_REPAIR_BLUEPRINT_REPAIRED",
        "inputs": {
            "compact_rows": str(COMPACT_ROWS),
            "selected_reinvest_pairs": str(SELECTED_REINVEST),
            "density_metadata": str(DENSITY_META),
            "globalpiqa_atlas_snapshot": load_atlas_snapshot(),
            "input_sha256": hash_paths([COMPACT_ROWS, SELECTED_REINVEST, DENSITY_META]),
        },
        "candidate_filter": {
            "compact_min_ratio": COMPACT_MIN_RATIO,
            "compact_max_ratio": COMPACT_MAX_RATIO,
            "compact_min_recall": COMPACT_MIN_RECALL,
            "reject_exact_copy": True,
            "reject_source_risk": True,
            "max_packet_words": MAX_PACKET_WORDS,
            "dedupe_rule": "same as A02 materializer: one compact candidate per source key, replacing only if (content_recall, -pair_words, pair_id) improves.",
            "reject_counts_recomputed": dict(reject.most_common()),
            "matches_a02_compact_candidate_count": len(compact_map) == expected_compact_candidates,
            "expected_compact_candidates_from_metadata": expected_compact_candidates,
        },
        "ranking_rules": {
            "uses_exact_globalpiqa_prompt_text_for_candidate_ranking": False,
            "lexicon_families": {k: sorted(v) for k, v in LEXICONS.items()},
            "strong_practical_packet_rule": "valid compact candidate; score >= 4.2; at least two practical families; at least one tool/action/material family; content recall >=0.50, entity recall >=0.75, number recall >=1.0.",
            "replacement_rule": "add unused strong practical compact candidate and remove selected low-practical no-domain packet with identical pair_words; this preserves pair-word budget exactly in the blueprint.",
        },
        "selection_match": {
            "selected_reinvest_keys": len(selected_keys),
            "matched_selected_rows_in_compact_candidate_map": len(selected),
            "unmatched_selected_keys": [],
        },
        "pool_summary": {
            "all_compact_candidates": summarize_pool(all_candidates),
            "selected_reinvest": summarize_pool(selected),
            "unselected_compact_candidates": summarize_pool(unselected),
        },
        "top_unselected_practical_examples": [compact_public_record(r) for r in top_unselected[:40]],
        "swap_plan_summary": plan_summary,
        "outputs": {
            "json": str(OUT_JSON),
            "top_candidates_jsonl": str(OUT_CANDIDATES),
            "exact_wordlength_swap_plan_jsonl": str(OUT_SWAPS),
            "note": str(OUT_NOTE),
        },
        "scientific_use": {
            "what_it_supports": "There is a no-new-generation, exact-pair-word source replacement option inside the valid compact-candidate bank if compact_view_reinvest later needs a GlobalPIQA repair while preserving its other advantages.",
            "what_it_does_not_support": "It is not a BabyLM score and not evidence that the replacement improves any official task; tokenizer/packing/contamination audits and endpoint evidence remain necessary before training.",
        },
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_jsonl(OUT_CANDIDATES, (compact_public_record(r) for r in top_unselected[:500]))
    write_jsonl(OUT_SWAPS, swap_plan)
    write_note(payload)
    print(json.dumps({
        "status": payload["status"],
        "json": str(OUT_JSON),
        "note": str(OUT_NOTE),
        "compact_candidates": len(compact_map),
        "selected_pairs": len(selected),
        "unselected_candidates": len(unselected),
        "unselected_strong_practical_pairs": payload["pool_summary"]["unselected_compact_candidates"]["strong_practical_pairs"],
        "exact_wordlength_replacements": plan_summary["replacement_count"],
        "net_pair_words": plan_summary["net_pair_words"],
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
