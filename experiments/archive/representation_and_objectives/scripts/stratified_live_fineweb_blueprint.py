#!/usr/bin/env python3
"""Build a CPU-only stratified live-FineWeb source blueprint for a possible next study.

This does not train, evaluate, or generate rewritten text.  It turns the existing live
v3/v5 source pools into an inspectable source inventory and a small current-scan seed
that preserves both relation-rich context and self-contained facts.  The purpose is to
make the next data materialization fast and less arbitrary if the repaired cached
source-breadth trajectory supports continuing FineWeb.
"""
from __future__ import annotations

import json
import pathlib
import re
import statistics
from collections import Counter, defaultdict
from typing import Any, Iterable

ROOT = pathlib.Path("experiments/archive/representation_and_objectives")
WORK = ROOT
V3_PATH = WORK / "data/live_fineweb_source_selector_v3/live_fineweb_selector_v3_stable_doccap8.jsonl"
V5_PATH = WORK / "data/live_fineweb_selector_v5_strictstable/live_fineweb_selector_v5_doccap8.jsonl"
V3_SUM = WORK / "data/live_fineweb_source_selector_v3/live_fineweb_source_selector_v3_summary.json"
V5_SUM = WORK / "data/live_fineweb_selector_v5_strictstable/live_fineweb_selector_v5_summary.json"
TOKENIZER_ROOT = pathlib.Path("experiments/archive/initial_model_studies/training/runs/fullcycle_official_wwm_debertav2_8x480_seed43_100M_b256/hf_model")
OUT_DIR = WORK / "data/stratified_live_fineweb_blueprint"
NOTE = (WORK.parents[2] / 'research/notes/representation_and_objectives/stratified_live_fineweb_blueprint.md')

WORD_RE = re.compile(r"[A-Za-z][A-Za-z'\-]*|\d+(?:\.\d+)?")
RELATION_MARKERS = {
    "causal": set("cause causes caused causing lead leads led result results resulted make makes made prevent prevents prevented allow allows enabled enables increase increases decrease decreases reduce reduces damage damages break breaks broke broken improve improves affect affects change changes force forces forced".split()),
    "physical_dynamic": set("move moves moving moved push pushes pushed pull pulls pulled fall falls falling fell drop drops dropped bounce bounces bounced slide slides slid roll rolls rolled expand expands shrink shrinks melt melts melted freeze freezes frozen absorb absorbs heat heats cool cools bend bends stretch stretches throw throws thrown hit hits pour pours".split()),
    "object_property": set("hard soft heavy light rough smooth sharp blunt wet dry hot cold warm cool flexible rigid transparent opaque magnetic plastic metal wooden glass paper air water liquid solid gas pressure weight color size shape open closed full empty".split()),
    "spatial_state": set("inside outside above below under over near far between around beside behind front left right contain contains contained holding placed put opening center edge surface floor top bottom".split()),
    "social_agent": set("person people child children teacher student mother father friend group help helps helped teach teaches learn learns ask asks tell tells give gives receive receives want wants know knows believe believes speak speaks said says".split()),
    "temporal_process": set("first next later earlier before after during finally eventually started begins became becomes continued ended discovered developed published founded built created born died".split()),
}


def iter_jsonl(path: pathlib.Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_json(path: pathlib.Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def rid(row: dict[str, Any]) -> str:
    return str(row.get("dedup_hash") or f"{row.get('doc_id')}::{row.get('sentence_index')}::{row.get('text','')[:80]}")


def toks(text: str) -> list[str]:
    return [m.group(0).lower().strip("'-") for m in WORD_RE.finditer(text or "") if m.group(0).strip("'-")]


def row_text(row: dict[str, Any], source: str) -> str:
    if source == "v5":
        return str(row.get("selector_v5_text") or row.get("selector_v3_repaired_text") or row.get("text") or "")
    return str(row.get("selector_v3_repaired_text") or row.get("text") or "")


def row_words(row: dict[str, Any], source: str) -> int:
    for k in (["selector_v5_words", "selector_v3_words", "words"] if source == "v5" else ["selector_v3_words", "words"]):
        try:
            return int(row.get(k))
        except Exception:
            pass
    return len(toks(row_text(row, source)))


def profile_row(row: dict[str, Any], source: str) -> dict[str, Any]:
    text = row_text(row, source)
    token_set = set(toks(text))
    hits = {k: sorted(token_set & v) for k, v in RELATION_MARKERS.items()}
    types = row.get("selector_v5_types") if source == "v5" else row.get("selector_v3_types")
    if not isinstance(types, list):
        types = row.get("selector_v3_types") or []
    flags = {
        "has_causal": bool(hits["causal"]),
        "has_temporal": bool(hits["temporal_process"]),
        "has_physical_spatial_object": bool(hits["physical_dynamic"] or hits["spatial_state"] or hits["object_property"]),
        "has_social_agent": bool(hits["social_agent"]),
        "has_definition_taxonomy": "definition_taxonomy" in types,
        "has_relation_type": any(x in types for x in ["causal_mechanism", "temporal_event", "attribute_relation", "process_relation", "social_relation"]),
    }
    if flags["has_causal"] or flags["has_temporal"]:
        focus = "causal_temporal_process"
    elif flags["has_physical_spatial_object"]:
        focus = "physical_spatial_object"
    elif flags["has_social_agent"]:
        focus = "social_entity_state"
    elif flags["has_definition_taxonomy"]:
        focus = "definition_taxonomic_fact"
    else:
        focus = "other_expository_relation"
    score = 0.0
    for k in ["core_content_score"]:
        try:
            score += float(row.get(k) or 0.0)
        except Exception:
            pass
    score += 1.2 * len([v for v in hits.values() if v])
    score += 0.6 * len(types)
    if source == "v5":
        score += 2.0
    return {
        "text": text,
        "words": row_words(row, source),
        "doc_id": row.get("doc_id"),
        "sentence_index": row.get("sentence_index"),
        "dedup_hash": rid(row),
        "source_pool": source,
        "types": types,
        "relation_hits": hits,
        "flags": flags,
        "focus": focus,
        "selection_score": round(score, 4),
    }


def load_profiles() -> dict[str, dict[str, Any]]:
    v5_profiles = {rid(r): profile_row(r, "v5") for r in iter_jsonl(V5_PATH)}
    profiles: dict[str, dict[str, Any]] = {}
    for r in iter_jsonl(V3_PATH):
        h = rid(r)
        if h in v5_profiles:
            p = v5_profiles[h]
            p["blueprint_class"] = "v5_self_contained_fact"
        else:
            p = profile_row(r, "v3")
            p["blueprint_class"] = "v3_relation_context_not_v5"
        profiles[h] = p
    # If v5 has any row outside v3 doccap8, preserve it as self-contained reserve.
    for h, p in v5_profiles.items():
        if h not in profiles:
            p["blueprint_class"] = "v5_self_contained_fact_outside_v3_doccap8"
            profiles[h] = p
    return profiles


def summarize(profiles: Iterable[dict[str, Any]]) -> dict[str, Any]:
    rows = list(profiles)
    words = sum(int(r["words"]) for r in rows)
    by_class = Counter(r["blueprint_class"] for r in rows)
    words_by_class = Counter()
    by_focus = Counter()
    words_by_focus = Counter()
    rel_rows = Counter()
    rel_words = Counter()
    type_counts = Counter()
    docs = set()
    for r in rows:
        docs.add(r.get("doc_id"))
        words_by_class[r["blueprint_class"]] += int(r["words"])
        by_focus[r["focus"]] += 1
        words_by_focus[r["focus"]] += int(r["words"])
        for t in r.get("types") or []:
            type_counts[t] += 1
        for k, h in r["relation_hits"].items():
            if h:
                rel_rows[k] += 1
                rel_words[k] += int(r["words"])
    return {
        "rows": len(rows),
        "words": words,
        "docs": len(docs),
        "rows_by_class": dict(by_class),
        "words_by_class": dict(words_by_class),
        "rows_by_focus": dict(by_focus),
        "words_by_focus": dict(words_by_focus),
        "relation_row_pct": {k: round(rel_rows[k] / max(len(rows), 1) * 100, 3) for k in RELATION_MARKERS},
        "relation_word_pct": {k: round(rel_words[k] / max(words, 1) * 100, 3) for k in RELATION_MARKERS},
        "top_type_counts": dict(type_counts.most_common(20)),
        "word_mean": round(statistics.mean([r["words"] for r in rows]), 3) if rows else 0.0,
        "word_median": round(statistics.median([r["words"] for r in rows]), 3) if rows else 0.0,
    }


def greedy_select(profiles: list[dict[str, Any]], targets: dict[str, int], total_target: int) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    used = set()
    words_by_bucket = Counter()
    total_words = 0
    def row_bucket(r: dict[str, Any]) -> str:
        if r["blueprint_class"].startswith("v5"):
            return "v5_self_contained_fact"
        return r["focus"]
    # bucket-first fill
    for bucket, target in targets.items():
        candidates = [r for r in profiles if row_bucket(r) == bucket and r["dedup_hash"] not in used]
        candidates.sort(key=lambda r: (r["selection_score"], r["words"]), reverse=True)
        for r in candidates:
            if words_by_bucket[bucket] >= target:
                break
            selected.append(r)
            used.add(r["dedup_hash"])
            words_by_bucket[bucket] += int(r["words"])
            total_words += int(r["words"])
    # fill remainder with relation-rich rows, alternating quality and diversity
    rest = [r for r in profiles if r["dedup_hash"] not in used]
    rest.sort(key=lambda r: (len([h for h in r["relation_hits"].values() if h]), r["selection_score"], r["words"]), reverse=True)
    for r in rest:
        if total_words >= total_target:
            break
        selected.append(r)
        used.add(r["dedup_hash"])
        words_by_bucket[row_bucket(r)] += int(r["words"])
        total_words += int(r["words"])
    for i, r in enumerate(selected):
        r["blueprint_selection_rank"] = i
        r["blueprint_selection_bucket"] = row_bucket(r)
    return selected


def tokenizer_probe(rows: list[dict[str, Any]]) -> dict[str, Any]:
    try:
        from transformers import AutoTokenizer  # type: ignore
        tok = AutoTokenizer.from_pretrained(TOKENIZER_ROOT, trust_remote_code=True)
    except Exception as exc:
        return {"status": "tokenizer_unavailable", "error": repr(exc), "tokenizer_root": str(TOKENIZER_ROOT)}
    total_words = 0
    total_tokens = 0
    visible_tokens = 0
    hidden_rows = 0
    longest = []
    for r in rows:
        ids = tok(row_text_like(r), add_special_tokens=False)["input_ids"]
        n = len(ids)
        w = int(r["words"])
        total_words += w
        total_tokens += n
        visible_tokens += min(n, 256)
        if n > 256:
            hidden_rows += 1
        longest.append((n, w, r["text"][:180], r["dedup_hash"]))
    longest.sort(reverse=True)
    return {
        "status": "ok",
        "tokenizer_root": str(TOKENIZER_ROOT),
        "rows": len(rows),
        "words": total_words,
        "tokens": total_tokens,
        "tokens_per_word": round(total_tokens / max(total_words, 1), 6),
        "seq256_visible_token_fraction": round(visible_tokens / max(total_tokens, 1), 6),
        "rows_over_256_tokens": hidden_rows,
        "longest_rows": [{"tokens": n, "words": w, "text": text, "dedup_hash": h} for n, w, text, h in longest[:10]],
    }


def row_text_like(r: dict[str, Any]) -> str:
    return str(r.get("text") or "")


def project_words(summary: dict[str, Any], source_words: int) -> dict[str, Any]:
    scanned = summary.get("doc_words_scanned") or summary.get("source_doc_words_scanned") or 3756512
    cap8 = summary.get("stable_doccap8_words") or summary.get("doccap8_words") or summary.get("core_doc_cap8_words")
    if not cap8:
        return {}
    return {str(source_words): round(float(scanned) * float(source_words) / float(cap8), 1)}


def main() -> None:
    profiles_by_id = load_profiles()
    profiles = list(profiles_by_id.values())
    current_summary = summarize(profiles)

    # Current scan is not enough for a 1.75M source block. Build an inspectable source seed
    # close to 300k words while preserving both stable facts and relation context.
    targets = {
        "v5_self_contained_fact": 115_000,
        "causal_temporal_process": 80_000,
        "physical_spatial_object": 45_000,
        "social_entity_state": 25_000,
        "definition_taxonomic_fact": 20_000,
        "other_expository_relation": 15_000,
    }
    selected = greedy_select(profiles, targets, total_target=300_000)
    selected_summary = summarize(selected)
    token_probe_all = tokenizer_probe(profiles)
    token_probe_selected = tokenizer_probe(selected)

    v3s = load_json(V3_SUM)
    v5s = load_json(V5_SUM)
    projection = {
        "observed_current_unique_doccap8_words": current_summary["words"],
        "observed_selected_seed_words": selected_summary["words"],
        "v3_projected_doc_words_needed_for_1p75M": (v3s.get("projected_scanned_doc_words_needed") or {}).get("1750000"),
        "v5_projected_doc_words_needed_for_1p75M": (v5s.get("projected_scanned_doc_words_needed") or {}).get("1750000"),
        "interpretation": "Current research scan can support a compact source/rewrite pilot seed, not a 1.75M+ replacement without more live streaming. The projected scan needs should be used before starting any larger source materialization.",
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    selected_path = OUT_DIR / "live_fineweb_stratified_current_seed_300k.jsonl"
    with selected_path.open("w", encoding="utf-8") as f:
        for r in selected:
            keep = {k: r[k] for k in ["dedup_hash", "doc_id", "sentence_index", "text", "words", "source_pool", "blueprint_class", "focus", "types", "relation_hits", "flags", "selection_score", "blueprint_selection_rank", "blueprint_selection_bucket"] if k in r}
            f.write(json.dumps(keep, ensure_ascii=False) + "\n")

    payload = {
        "status": "STRATIFIED_LIVE_FINEWEB_BLUEPRINT",
        "purpose": "CPU-only inventory and current-scan source seed for a possible later FineWeb source/source+view study; not training, evaluation, or Qwen generation.",
        "inputs": {"v3_doccap8": str(V3_PATH), "v5_doccap8": str(V5_PATH), "v3_summary": str(V3_SUM), "v5_summary": str(V5_SUM)},
        "current_union_summary": current_summary,
        "selected_seed_summary": selected_summary,
        "selected_seed_path": str(selected_path),
        "selection_targets_words": targets,
        "tokenizer_probe_current_union": token_probe_all,
        "tokenizer_probe_selected_seed": token_probe_selected,
        "projection": projection,
        "route_use": {
            "if_step017_broad_positive": "Use this as the seed logic for a larger stratified live FineWeb four-arm family, but stream more source text first; keep both v3 relation-rich context and v5 self-contained facts.",
            "if_step017_small_positive": "Use a compact source+view pilot seeded by this file before another large replacement; measure substantive rewrite rate and token exposure.",
            "if_step017_flat_or_negative": "Do not extend cached source repetition. This source seed may still be useful only for a small cleaner source+view test or for interpreting which relation types are missing.",
        },
    }
    out_json = OUT_DIR / "stratified_live_fineweb_blueprint.json"
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    lines = ["# research stratified live-FineWeb blueprint\n\n"]
    lines.append("This is CPU-only preparation for a possible next FineWeb study. It creates no training corpus, no generated text, and no model score.\n\n")
    lines.append(f"Current v3/v5 union: {current_summary['rows']} rows / {current_summary['words']} words / {current_summary['docs']} docs.\n\n")
    lines.append(f"Selected current-scan seed: `{selected_path}` with {selected_summary['rows']} rows / {selected_summary['words']} words / {selected_summary['docs']} docs.\n\n")
    lines.append("## Current union by source class\n\n")
    lines.append("```json\n" + json.dumps(current_summary["words_by_class"], indent=2, ensure_ascii=False) + "\n```\n\n")
    lines.append("## Selected seed by focus\n\n")
    lines.append("```json\n" + json.dumps(selected_summary["words_by_focus"], indent=2, ensure_ascii=False) + "\n```\n\n")
    if token_probe_selected.get("status") == "ok":
        lines.append(f"Baseline16k token probe for selected seed: {token_probe_selected['tokens_per_word']} tokens/word; seq256 visible fraction {token_probe_selected['seq256_visible_token_fraction']}; rows over 256 tokens {token_probe_selected['rows_over_256_tokens']}.\n\n")
    lines.append("Route use: if FineWeb continues, use this as a seed logic and source inspection object, not as a substitute for the required four-arm materialization and real training/evaluation. The scaled family still needs exact 10M/100M accounting, matched natural/control arms, faithful-view acceptance measurement, and trainer-level token exposure measurement.\n\n")
    lines.append(f"Summary JSON: `{out_json}`\n")
    NOTE.write_text("".join(lines), encoding="utf-8")
    print(json.dumps({"status": payload["status"], "out_json": str(out_json), "note": str(NOTE), "selected_seed_path": str(selected_path), "selected_seed_words": selected_summary["words"], "selected_seed_rows": selected_summary["rows"], "token_probe_selected": token_probe_selected}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
