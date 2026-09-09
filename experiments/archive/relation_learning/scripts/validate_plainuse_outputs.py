#!/usr/bin/env python3
"""research: validate cue-controlled plain-use state/update outputs.

This wraps the research changed-form validator and adds one new scientific
condition: the final use sentence must not contain temporal, change, or
persistence markers that can tell a learner whether the target state came from
sentence 1 or sentence 2 in the fixed three-sentence packet.  The goal is to keep
source/update/use training interpretable as entity-state practice rather than a
cue-position readout.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import pathlib
import random
import re
import statistics
import time
from collections import Counter, defaultdict
from typing import Any

ROOT = pathlib.Path.cwd()
SCRIPT_DIR = ROOT / "experiments/archive/relation_learning/scripts"
DATA = ROOT / "experiments/archive/relation_learning/data/state_use_generation"

spec = importlib.util.spec_from_file_location("validate_changedform_outputs", SCRIPT_DIR / "validate_changedform_outputs.py")
if spec is None or spec.loader is None:
    raise RuntimeError("Cannot import research validator")
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)

UPDATED_PROMPTS = DATA / "state_use_prompts_9b_updated_plainuse_step047.jsonl"
DISTRACTOR_PROMPTS = DATA / "state_use_prompts_9b_distractor_plainuse_step047.jsonl"
UPDATED_PILOT_PROMPTS = DATA / "state_use_prompts_9b_updated_plainuse_step047_pilot256.jsonl"
DISTRACTOR_PILOT_PROMPTS = DATA / "state_use_prompts_9b_distractor_plainuse_step047_pilot256.jsonl"
UPDATED_RAW = DATA / "raw_outputs_9b_updated_plainuse_step047.jsonl"
DISTRACTOR_RAW = DATA / "raw_outputs_9b_distractor_plainuse_step047.jsonl"
UPDATED_PILOT_RAW = DATA / "raw_outputs_9b_updated_plainuse_step047_pilot256.jsonl"
DISTRACTOR_PILOT_RAW = DATA / "raw_outputs_9b_distractor_plainuse_step047_pilot256.jsonl"

RNG_SEED = 47047
HELDOUT_PER_TYPE_DEFAULT = 500

# Word/phrase cues whose presence in use_sentence makes the packet insufficiently
# diagnostic: they reveal "state from source" or "state from update" by marker and
# position rather than entity-state tracking.
CUE_REGEXES: list[tuple[str, re.Pattern[str]]] = [
    ("still", re.compile(r"\bstill\b", re.I)),
    ("remain", re.compile(r"\bremains?\b|\bremained\b|\bremaining\b", re.I)),
    ("continue", re.compile(r"\bcontinues?\b|\bcontinued\b|\bcontinuing\b", re.I)),
    ("stay", re.compile(r"\bstays?\b|\bstayed\b|\bstaying\b", re.I)),
    ("keep", re.compile(r"\bkeeps?\b|\bkept\b", re.I)),
    ("retain", re.compile(r"\bretains?\b|\bretained\b|\bretaining\b", re.I)),
    ("maintain", re.compile(r"\bmaintains?\b|\bmaintained\b|\bmaintaining\b", re.I)),
    ("unchanged", re.compile(r"\bunchanged\b|\bas before\b", re.I)),
    ("today", re.compile(r"\btoday\b|\bnowadays\b", re.I)),
    ("now", re.compile(r"\bnow\b", re.I)),
    ("current", re.compile(r"\bcurrently\b|\bcurrent\b|\bpresently\b|\bat present\b", re.I)),
    ("new", re.compile(r"\bnew\b|\bnewly\b", re.I)),
    ("no_longer", re.compile(r"\bno longer\b|\banymore\b", re.I)),
    ("again_later", re.compile(r"\bagain\b|\blater\b", re.I)),
    ("former", re.compile(r"\bformer\b|\bformerly\b", re.I)),
]


def cue_hits(text: str) -> list[str]:
    hits: list[str] = []
    for name, rx in CUE_REGEXES:
        if rx.search(text or ""):
            hits.append(name)
    return hits


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


def write_jsonl(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def validate_plain_output(raw_text: str, prompt_rec: dict[str, Any]) -> tuple[dict[str, Any] | None, str, dict[str, Any]]:
    obj, parse_reason = base.parse_output(raw_text)
    if obj is None:
        return None, parse_reason, {}
    use_sent = base.norm_ws(obj.get("use_sentence", ""))
    hits = cue_hits(use_sent)
    if hits:
        return None, "plainuse_cue_in_use_sentence", {"cue_hits": hits, "use_sentence": use_sent}
    pkt, reason, detail = base.validate_output(raw_text, prompt_rec)
    if pkt is None:
        return None, reason, detail
    pkt["plainuse_cue_hits"] = []
    return pkt, "", {}


def process_outputs(raw_path: pathlib.Path, prompt_path: pathlib.Path, label: str, tolerate_partial: bool) -> tuple[dict[str, dict], Counter, list[dict], dict, Counter]:
    prompts = load_jsonl(prompt_path)
    raws = load_jsonl(raw_path, tolerate_partial=tolerate_partial)
    acc: dict[str, dict] = {}
    rej = Counter(); examples: list[dict[str, Any]] = []; cue_counter = Counter()
    for rec in raws:
        idx = rec.get("index")
        if not isinstance(idx, int) or idx < 0 or idx >= len(prompts):
            rej["index_out_of_range"] += 1
            continue
        p = prompts[idx]
        global_idx = p.get("original_prompt_index", idx)
        raw_text = rec.get("output", rec.get("generated_text", ""))
        parsed, _ = base.parse_output(raw_text)
        if isinstance(parsed, dict):
            for h in cue_hits(base.norm_ws(parsed.get("use_sentence", ""))):
                cue_counter[h] += 1
        pkt, reason, detail = validate_plain_output(raw_text, p)
        if pkt is None:
            rej[reason] += 1
            if len(examples) < 400:
                examples.append({"source_label": label, "index": idx, "original_prompt_index": global_idx, "pair_id": p.get("pair_id"), "packet_type": p.get("packet_type"), "reason": reason, "detail": detail, "output": raw_text, "source_sentence": p.get("source_sentence")})
            continue
        pkt["generation_source"] = label
        pkt["local_output_index"] = idx
        pkt["original_prompt_index"] = global_idx
        acc.setdefault(pkt["pair_id"], pkt)
    info = {"source_label": label, "raw_path": str(raw_path), "prompt_path": str(prompt_path), "prompt_records": len(prompts), "raw_outputs": len(raws), "accepted": len(acc), "rejections": dict(rej.most_common()), "raw_parsed_use_cue_hits_before_rejection": dict(cue_counter.most_common())}
    return acc, rej, examples, info, cue_counter


def balanced_split(rows: list[dict[str, Any]], *, heldout_per_type: int) -> tuple[list[dict], list[dict], list[dict]]:
    rng = random.Random(RNG_SEED)
    by_type: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_type[r["packet_type"]].append(r)
    for typ in by_type:
        by_type[typ].sort(key=lambda r: int(r.get("original_prompt_index", 10**12)))
        rng.shuffle(by_type[typ])
    n_bal = min(len(by_type.get("UPDATED_USE", [])), len(by_type.get("UNCHANGED_DISTRACTOR_USE", [])))
    balanced: list[dict[str, Any]] = []
    for typ in ["UPDATED_USE", "UNCHANGED_DISTRACTOR_USE"]:
        balanced.extend(by_type.get(typ, [])[:n_bal])
    heldout: list[dict[str, Any]] = []
    train: list[dict[str, Any]] = []
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
    return vals[lo] if lo == hi else vals[lo]*(hi-pos)+vals[hi]*(pos-lo)


def stat(vals: list[float]) -> dict[str, Any]:
    if not vals:
        return {"n": 0, "min": None, "p05": None, "mean": None, "median": None, "p95": None, "max": None}
    return {"n": len(vals), "min": round(min(vals),4), "p05": round(pct(vals,0.05),4), "mean": round(statistics.mean(vals),4), "median": round(statistics.median(vals),4), "p95": round(pct(vals,0.95),4), "max": round(max(vals),4)}


def surface_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    vals = lambda key: [float(r["surface"]["use_vs_source"][key]) for r in rows]
    lcs = [float(r["surface"]["use_vs_source"]["longest_common_contiguous_words"]) for r in rows]
    n = len(rows) or 1
    return {
        "use_source_content_jaccard": stat(vals("content_jaccard")),
        "use_source_overlap_min": stat(vals("content_overlap_min")),
        "use_source_lcs": stat(lcs),
        "frac_lcs_ge6": round(sum(x >= 6 for x in lcs)/n, 4),
        "frac_content_jaccard_ge0p8": round(sum(x >= 0.8 for x in vals("content_jaccard"))/n, 4),
        "use_cue_hits_in_accepted": dict(Counter(h for r in rows for h in cue_hits(r.get("use_sentence", ""))).most_common()),
        "frac_any_use_cue_in_accepted": round(sum(bool(cue_hits(r.get("use_sentence", ""))) for r in rows)/n, 4),
    }


def write_review(path: pathlib.Path, rows: list[dict[str, Any]], n_each: int = 50) -> None:
    rng = random.Random(RNG_SEED + 3)
    by = defaultdict(list)
    for r in rows:
        by[r["packet_type"]].append(r)
    lines = ["# research accepted cue-controlled plain-use packet sample\n\n", "These examples passed research changed-form state/content checks and the research no-use-cue filter.\n\n"]
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
            lines.append(f"USE_CUES: {cue_hits(r['use_sentence'])}\n\n")
            lines.append(f"SURFACE use-source: J={us['content_jaccard']:.3f}, overlap_min={us['content_overlap_min']:.3f}, LCS={us['longest_common_contiguous_words']} `{us['longest_common_span']}`; use-update LCS={uu['longest_common_contiguous_words']} `{uu['longest_common_span']}`\n\n")
            lines.append(f"STATE_TERMS source={r['source_state_terms']} new={r['new_state_terms']} hits_update={r['update_state_hits']} hits_use_new={r['use_new_state_hits']} hits_use_source={r['use_source_state_hits']}\n\n")
    path.write_text("".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["pilot", "full"], default="pilot")
    ap.add_argument("--tolerate-partial", action="store_true")
    ap.add_argument("--heldout-per-type", type=int, default=HELDOUT_PER_TYPE_DEFAULT)
    args = ap.parse_args()

    if args.mode == "pilot":
        sources = [
            (UPDATED_PILOT_RAW, UPDATED_PILOT_PROMPTS, "updated_plainuse_pilot"),
            (DISTRACTOR_PILOT_RAW, DISTRACTOR_PILOT_PROMPTS, "distractor_plainuse_pilot"),
        ]
        stem = "plainuse_pilot_step047"
    else:
        sources = [
            (UPDATED_RAW, UPDATED_PROMPTS, "updated_plainuse_full"),
            (DISTRACTOR_RAW, DISTRACTOR_PROMPTS, "distractor_plainuse_full"),
        ]
        stem = "plainuse_full_step047"

    t0 = time.time(); merged: dict[str, dict] = {}; rejs = Counter(); examples: list[dict[str, Any]] = []; infos: list[dict[str, Any]] = []; raw_cues = Counter()
    for raw, prompts, label in sources:
        acc, rej, ex, info, cue_counter = process_outputs(raw, prompts, label, args.tolerate_partial)
        infos.append(info); rejs.update(rej); examples.extend(ex); raw_cues.update(cue_counter)
        for pid, pkt in acc.items():
            if pid not in merged:
                merged[pid] = pkt
    valid = sorted(merged.values(), key=lambda r: (int(r.get("original_prompt_index", 10**12)), r["packet_type"]))
    balanced, train, heldout = balanced_split(valid, heldout_per_type=args.heldout_per_type)

    all_path = DATA / f"validated_state_use_packets_{stem}_all.jsonl"
    bal_path = DATA / f"validated_state_use_packets_{stem}_balanced.jsonl"
    train_path = DATA / f"validated_state_use_packets_{stem}_train.jsonl"
    heldout_path = DATA / f"validated_state_use_packets_{stem}_heldout.jsonl"
    rej_path = DATA / f"validation_rejection_examples_{stem}.jsonl"
    review_path = DATA / f"accepted_packet_manual_read_sample_{stem}.md"
    meta_path = DATA / f"validation_metadata_{stem}.json"
    write_jsonl(all_path, valid); write_jsonl(bal_path, balanced); write_jsonl(train_path, train); write_jsonl(heldout_path, heldout); write_jsonl(rej_path, examples[:800])
    write_review(review_path, balanced if balanced else valid, n_each=50)
    by_type = Counter(r["packet_type"] for r in valid)
    bal_by_type = Counter(r["packet_type"] for r in balanced)
    meta = {
        "status": "PLAINUSE_VALIDATION_DONE",
        "mode": args.mode,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source_infos": infos,
        "raw_outputs_total": sum(i["raw_outputs"] for i in infos),
        "valid_packets_total": len(valid),
        "valid_rate_vs_outputs": round(len(valid)/max(1, sum(i["raw_outputs"] for i in infos)), 4),
        "type_counts_all": dict(by_type),
        "balanced_total": len(balanced),
        "type_counts_balanced": dict(bal_by_type),
        "train_total": len(train),
        "heldout_total": len(heldout),
        "train_type_counts": dict(Counter(r["packet_type"] for r in train)),
        "heldout_type_counts": dict(Counter(r["packet_type"] for r in heldout)),
        "surface_summary_all": surface_summary(valid),
        "surface_summary_balanced": surface_summary(balanced),
        "raw_parsed_use_cue_hits_before_rejection": dict(raw_cues.most_common()),
        "rejection_reasons": dict(rejs.most_common()),
        "paths": {"all": str(all_path), "balanced": str(bal_path), "train": str(train_path), "heldout": str(heldout_path), "rejections": str(rej_path), "manual_read_sample": str(review_path)},
        "sha256": {"all": sha256_file(all_path), "balanced": sha256_file(bal_path), "train": sha256_file(train_path), "heldout": sha256_file(heldout_path)},
        "thresholds": {**base.surface_summary([]), "no_use_sentence_cues": [name for name, _ in CUE_REGEXES]},
        "note": "Plain-use acceptance applies all research state/content/changed-form tests and rejects temporal/change/persistence cues in the final use sentence.",
        "elapsed_sec": round(time.time() - t0, 2),
    }
    # Keep real research thresholds explicit; the surface_summary empty dict in thresholds above is harmless but unhelpful, so overwrite it.
    meta["thresholds"] = {
        "max_source_use_lcs": base.MAX_SOURCE_USE_LCS,
        "max_update_use_lcs_updated": base.MAX_UPDATE_USE_LCS_UPDATED,
        "max_source_use_content_jaccard": base.MAX_SOURCE_USE_CONTENT_JACCARD,
        "max_source_use_overlap_min": base.MAX_SOURCE_USE_OVERLAP_MIN,
        "new_state_min_content_words": 2,
        "distractor_source_state_min_content_words": 2,
        "reject_transcript_marker_in_source": True,
        "reject_use_sentence_cues": [name for name, _ in CUE_REGEXES],
    }
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(meta, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
