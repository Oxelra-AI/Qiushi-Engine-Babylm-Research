#!/usr/bin/env python3
"""Prepare a live-FineWeb faithful-view pilot and slot manifest.

This is CPU-only preparation.  It writes a preferred source/distinct/repeat slot
manifest from the research 300k live-FineWeb seed, and a compact prompt slice for
later Qwen generation if downstream evidence warrants testing source+view.  It
does not generate text, train a model, or evaluate BabyLM.
"""
from __future__ import annotations

import json
import pathlib
from collections import Counter, defaultdict, deque
from typing import Any

ROOT = pathlib.Path("experiments/archive/representation_and_objectives")
SEED = ROOT / "data/stratified_live_fineweb_blueprint/live_fineweb_stratified_current_seed_300k.jsonl"
PAIRING = ROOT / "data/fineweb_slot_pairing_feasibility/fineweb_slot_pairing_feasibility.json"
OUT_DIR = ROOT / "training/data/live_fineweb_view_pilot"
NOTE = (ROOT.parents[2] / 'research/notes/representation_and_objectives/live_fineweb_view_pilot_prepared.md')
RATIO = 0.88
PILOT_UNIQUE_SOURCES = 256
PROMPT_VARIANTS = {
    "faithful_simplify": (
        "Rewrite the sentence below as one clear sentence for a younger reader. "
        "Use only information in the sentence. Keep every name, date, number, place, object, and cause/effect relation. "
        "Do not add background facts, examples, explanations, guesses, or opinions. Use different wording when possible. "
        "Aim for about {target_words} words, but preserving the facts is more important than exact length. "
        "Output only the rewritten sentence.\n\nSentence: {text}"
    ),
    "relation_preserving_restate": (
        "Restate the sentence as a single factual sentence with simpler wording while preserving the exact relationship it expresses. "
        "Do not introduce any new entity, event, date, number, location, cause, result, comparison, or attribution. "
        "Keep all names and numbers from the source sentence. Avoid copying the whole sentence verbatim when a faithful rewording is possible. "
        "Aim for about {target_words} words. Output only one sentence.\n\nSentence: {text}"
    ),
}
PILOT_FOCUS_QUOTAS = {
    "causal_temporal_process": 96,
    "physical_spatial_object": 64,
    "social_entity_state": 40,
    "other_expository_relation": 40,
    "definition_taxonomic_fact": 16,
}


def load_jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue
            r = json.loads(line)
            r["_i"] = i
            r["words"] = int(r.get("words") or len(str(r.get("text", "")).split()))
            r["focus"] = str(r.get("focus") or "unknown")
            r["doc_id"] = str(r.get("doc_id") or "")
            rows.append(r)
    return rows


def target_len(words: int) -> int:
    return max(10, int(round(words * RATIO)))


def make_buckets(rows: list[dict[str, Any]]) -> dict[tuple[str, int], deque[int]]:
    buckets: dict[tuple[str, int], deque[int]] = defaultdict(deque)
    for r in sorted(rows, key=lambda x: (int(x.get("blueprint_selection_rank", x["_i"])), x["_i"])):
        buckets[(r["focus"], r["words"])].append(r["_i"])
    return buckets


def pop_exact_same_focus(rows_by_i: dict[int, dict[str, Any]], buckets: dict[tuple[str, int], deque[int]], used: set[int], base: dict[str, Any], want_len: int) -> dict[str, Any] | None:
    key = (base["focus"], want_len)
    q = buckets.get(key)
    if not q:
        return None
    skipped = deque()
    match = None
    while q:
        idx = q.popleft()
        cand = rows_by_i[idx]
        if idx in used or idx == base["_i"] or cand["doc_id"] == base["doc_id"]:
            skipped.append(idx)
            continue
        match = cand
        break
    while skipped:
        q.appendleft(skipped.pop())
    return match


def repeat_to_len(text: str, n: int) -> str:
    ws = str(text or "").split()
    if not ws:
        return ""
    out = []
    i = 0
    while len(out) < n:
        out.append(ws[i % len(ws)])
        i += 1
    return " ".join(out)


def build_pairs(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows_by_i = {r["_i"]: r for r in rows}
    buckets = make_buckets(rows)
    used: set[int] = set()
    pairs = []
    for base in sorted(rows, key=lambda x: (int(x.get("blueprint_selection_rank", x["_i"])), x["_i"])):
        if base["_i"] in used:
            continue
        want = target_len(base["words"])
        match = pop_exact_same_focus(rows_by_i, buckets, used, base, want)
        if match is None:
            continue
        used.add(base["_i"]); used.add(match["_i"])
        pairs.append({
            "pair_id": f"fwslot_{len(pairs):05d}",
            "source_index": base["_i"],
            "distinct_index": match["_i"],
            "source_doc": base["doc_id"],
            "distinct_doc": match["doc_id"],
            "source_text": base.get("text"),
            "distinct_text": match.get("text"),
            "repeat_slot_text_simulated": repeat_to_len(str(base.get("text", "")), want),
            "source_words": base["words"],
            "target_view_words_simulated": want,
            "distinct_words": match["words"],
            "packet_words_simulated": base["words"] + want,
            "source_focus": base["focus"],
            "distinct_focus": match["focus"],
            "source_pool": base.get("source_pool"),
            "source_class": base.get("blueprint_class"),
            "distinct_pool": match.get("source_pool"),
            "distinct_class": match.get("blueprint_class"),
            "source_types": base.get("types"),
            "source_relation_hits": base.get("relation_hits"),
            "source_selection_score": base.get("selection_score"),
            "source_dedup_hash": base.get("dedup_hash"),
            "distinct_dedup_hash": match.get("dedup_hash"),
        })
    return pairs


def choose_pilot_sources(pairs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_focus: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for p in pairs:
        by_focus[p["source_focus"]].append(p)
    for focus in by_focus:
        by_focus[focus].sort(key=lambda p: (-float(p.get("source_selection_score") or 0.0), p["pair_id"]))
    selected: list[dict[str, Any]] = []
    used_docs: set[str] = set()
    for focus, quota in PILOT_FOCUS_QUOTAS.items():
        bucket = by_focus.get(focus, [])
        chosen = []
        for p in bucket:
            if len(chosen) >= quota:
                break
            if p["source_doc"] in used_docs:
                continue
            chosen.append(p)
            used_docs.add(p["source_doc"])
        if len(chosen) < quota:
            for p in bucket:
                if len(chosen) >= quota:
                    break
                if p in chosen:
                    continue
                chosen.append(p)
                used_docs.add(p["source_doc"])
        selected.extend(chosen[:quota])
    if len(selected) < PILOT_UNIQUE_SOURCES:
        already = {p["pair_id"] for p in selected}
        rest = sorted([p for p in pairs if p["pair_id"] not in already], key=lambda p: (-float(p.get("source_selection_score") or 0.0), p["pair_id"]))
        for p in rest:
            if len(selected) >= PILOT_UNIQUE_SOURCES:
                break
            selected.append(p)
    return selected[:PILOT_UNIQUE_SOURCES]


def write_jsonl(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = load_jsonl(SEED)
    pairs = build_pairs(rows)
    pilot_sources = choose_pilot_sources(pairs)
    prompts: list[dict[str, Any]] = []
    for p in pilot_sources:
        for variant, template in PROMPT_VARIANTS.items():
            prompts.append({
                "id": f"{p['pair_id']}__{variant}",
                "pair_id": p["pair_id"],
                "variant": variant,
                "prompt": template.format(text=p["source_text"], target_words=p["target_view_words_simulated"]),
                "source_text": p["source_text"],
                "source_words": p["source_words"],
                "target_view_words_simulated": p["target_view_words_simulated"],
                "source_focus": p["source_focus"],
                "source_class": p["source_class"],
                "source_pool": p["source_pool"],
                "source_doc": p["source_doc"],
                "source_types": p["source_types"],
                "source_relation_hits": p["source_relation_hits"],
            })
    paths = {
        "manifest": OUT_DIR / "live_fineweb_slot_manifest_ratio0p88_samefocus_exact.jsonl",
        "pilot_sources": OUT_DIR / "live_fineweb_view_pilot_sources_256.jsonl",
        "prompts": OUT_DIR / "live_fineweb_view_pilot_prompts_512.jsonl",
        "summary": OUT_DIR / "live_fineweb_view_pilot_preparation_summary.json",
        "samples": OUT_DIR / "live_fineweb_view_pilot_samples.json",
        "generation_config": OUT_DIR / "GENERATION_CONFIG.json",
    }
    write_jsonl(paths["manifest"], pairs)
    write_jsonl(paths["pilot_sources"], pilot_sources)
    write_jsonl(paths["prompts"], prompts)
    focus_words = Counter(); focus_rows = Counter(); class_words = Counter()
    for p in pairs:
        focus_words[p["source_focus"]] += p["packet_words_simulated"]
        focus_rows[p["source_focus"]] += 1
        class_words[p["source_class"]] += p["source_words"]
    pilot_focus = Counter(p["source_focus"] for p in pilot_sources)
    pilot_class = Counter(p["source_class"] for p in pilot_sources)
    summary = {
        "status": "LIVE_FINEWEB_VIEW_PILOT_PREPARED",
        "purpose": "CPU-only prompt and slot-manifest preparation for a possible FineWeb source+faithful-view experiment; no generation, training, or evaluation.",
        "source_seed": str(SEED),
        "pairing_feasibility": str(PAIRING),
        "contract": {
            "view_length_ratio_simulated": RATIO,
            "distinct_slot_matching": "same focus and exact simulated view length; different document",
            "actual_view_note": "After generation, distinct-source slots should be rematched to actual accepted view lengths before any training materialization.",
        },
        "manifest_pairs": len(pairs),
        "manifest_source_words": sum(p["source_words"] for p in pairs),
        "manifest_target_view_words_simulated": sum(p["target_view_words_simulated"] for p in pairs),
        "manifest_packet_words_simulated": sum(p["packet_words_simulated"] for p in pairs),
        "manifest_distinct_words": sum(p["distinct_words"] for p in pairs),
        "manifest_focus_packet_words": dict(focus_words.most_common()),
        "manifest_focus_rows": dict(focus_rows.most_common()),
        "manifest_source_class_words": dict(class_words.most_common()),
        "pilot_unique_sources": len(pilot_sources),
        "pilot_prompts": len(prompts),
        "pilot_prompt_variants": list(PROMPT_VARIANTS),
        "pilot_focus_counts": dict(pilot_focus.most_common()),
        "pilot_class_counts": dict(pilot_class.most_common()),
        "paths": {k: str(v) for k, v in paths.items()},
        "generation_is_deferred": True,
        "future_generation": {
            "model_alias": "qwen3.5-9b", "prompts_jsonl": str(paths['prompts']),
            "output_jsonl": "experiments/archive/representation_and_objectives/training/runs/live_fineweb_view_pilot_qwen/outputs.jsonl",
            "batch_size": 64, "max_new_tokens": 80, "temperature": 0.15,
            "device": "cuda", "status": "proposed_not_executed",
        },
        "future_analysis_script": "experiments/archive/representation_and_objectives/training/scripts/analyze_live_fineweb_view_pilot_outputs.py",
        "use_policy": "Analyze faithfulness and substantive changes in generated views before materializing a training corpus.",
    }
    paths["summary"].write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    paths["samples"].write_text(json.dumps({"manifest_head": pairs[:10], "pilot_prompt_head": prompts[:12]}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    paths["generation_config"].write_text(json.dumps(summary["future_generation"], indent=2) + "\n", encoding="utf-8")

    lines = ["# research live FineWeb faithful-view pilot preparation\n\n"]
    lines.append("This CPU-only file set prepares a possible source+view test without launching generation or training. It uses the research live-FineWeb seed and the exact same-focus ratio-0.88 slot contract from the research pairing study.\n\n")
    lines.append(f"Slot manifest: {len(pairs):,} source/distinct pairs; simulated source+view packet words {summary['manifest_packet_words_simulated']:,}; source words {summary['manifest_source_words']:,}; simulated view words {summary['manifest_target_view_words_simulated']:,}.\n\n")
    lines.append(f"Prompt slice: {len(pilot_sources):,} unique sources and {len(prompts):,} prompts across variants {', '.join(PROMPT_VARIANTS)}. Focus counts: {dict(pilot_focus)}.\n\n")
    lines.append("If generation is later run, actual accepted view lengths must be measured and the distinct-source slot should be rebuilt to those actual lengths before any matched training family.\n\n")
    lines.append(f"Summary JSON: `{paths['summary']}`\n\nPrompts: `{paths['prompts']}`\n\nManifest: `{paths['manifest']}`\n")
    NOTE.write_text("".join(lines), encoding="utf-8")
    print(json.dumps({
        "status": summary["status"],
        "summary": str(paths["summary"]),
        "note": str(NOTE),
        "manifest_pairs": len(pairs),
        "manifest_packet_words_simulated": summary["manifest_packet_words_simulated"],
        "pilot_prompts": len(prompts),
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
