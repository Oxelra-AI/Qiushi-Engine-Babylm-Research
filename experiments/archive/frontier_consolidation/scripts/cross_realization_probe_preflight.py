#!/usr/bin/env python3
"""research: preflight a saved-checkpoint cross-realization probe.

This script does not train or score models. It constructs a feasible, matched
set of events for the proposed cross-realization inference probe:
source-shared/copied content targets are masked under source, genuine compact,
and matched counterfactual compact contexts. It checks target identity, BPE
piece counts, context lengths, counterfactual no-target condition, and arm
availability.
"""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import math
import pathlib
import random
import re
import statistics
from typing import Any

from transformers import AutoTokenizer

SEQ_LEN = 256
ROOT_A02 = pathlib.Path("experiments/archive/frontier_consolidation")
ROOT_A01 = pathlib.Path("experiments/archive/representation_and_objectives")
TOKENIZER_PATH = ROOT_A02 / "data/compliant_tokenizer"
COMPACT_PAIRS = ROOT_A02 / "data/factorial_view_candidate_audit/compact_candidate_pairs.jsonl"
TRAIN_FIXED_EVENTS = ROOT_A01 / "data/existing_trajectory_denoising_probe/probe_events.jsonl"
SOURCE_DISJOINT_EVENTS = ROOT_A01 / "data/source_disjoint_target_probe/source_disjoint_probe_events.jsonl"
OUT_DIR_DEFAULT = ROOT_A02 / "data/cross_realization_probe_preflight"

ARMS = {
    "full_compact_100M": pathlib.Path("experiments/archive/frontier_consolidation/training/runs/cleanqwen_fineweb_compact_view_reinvest_16k_seed43022/hf_model/chck_100M"),
    "drop_abs_100M": pathlib.Path("experiments/archive/representation_and_objectives/training/runs/packed_drop_abs_content_100M_fast/hf_model/chck_100M"),
    "drop_copied_word_100M": pathlib.Path("experiments/archive/representation_and_objectives/training/runs/packed_drop_copied_content_wholeword_100M_fast/hf_model/chck_100M"),
    "repeat_100M": pathlib.Path("experiments/archive/representation_and_objectives/training/runs/gc_compact_repeat_reinvest_16k_seed43022_r2/hf_model/chck_100M"),
    "adjbreak_100M": pathlib.Path("experiments/archive/representation_and_objectives/training/runs/gc_adjbreak_reinvest_16k_seed43022_r2/hf_model/chck_100M"),
}

FUNC_WORDS = frozenset("""
a an the this that these those my your his her its our their is am are was were be been being
have has had do does did will would shall should can could may might must need ought to of in on
at by for with from into through during before after above below between under over about against
along across around behind beside beyond down near off since toward upon within without and but or
nor so yet both either neither not no if then else than as which who whom whose what when where how
while until because although though even also just only still already very much more most less least
too quite really rather such same other another each every all some any few many i me we us you he
him she her it they them one ones there here up out away again back now however therefore thus hence
moreover furthermore nevertheless meanwhile otherwise instead indeed perhaps maybe probably
""".split())
WORD_RE = re.compile(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)?")


def norm(w: str) -> str:
    m = WORD_RE.search(w)
    return m.group(0).lower() if m else re.sub(r"[^a-z0-9]", "", w.lower())


def is_content(w: str) -> bool:
    n = norm(w)
    return bool(n) and n not in FUNC_WORDS and len(n) > 1


def sha256_file(path: pathlib.Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def token_ids(tok, text: str) -> list[int]:
    if not text:
        return []
    return tok(text, add_special_tokens=False)["input_ids"]


def span_for_word(tok, words: list[str], word_i: int) -> dict[str, Any] | None:
    if word_i < 0 or word_i >= len(words):
        return None
    before = " ".join(words[:word_i])
    through = " ".join(words[: word_i + 1])
    ids_before = token_ids(tok, before)
    ids_after = token_ids(tok, through)
    st, en = len(ids_before), len(ids_after)
    if st >= en or en > SEQ_LEN:
        return None
    ids = ids_after[st:en]
    return {"span": [st, en], "ids": ids, "pieces": len(ids), "rel_pos": word_i / max(1, len(words) - 1)}


def load_jsonl(path: pathlib.Path):
    with path.open(encoding="utf-8") as f:
        for i, line in enumerate(f):
            if line.strip():
                d = json.loads(line)
                d.setdefault("_line_index", i)
                yield d


def parse_doc_from_key(key: str | None) -> str | None:
    if not key:
        return None
    m = re.search(r"doc:([^|]+)", key)
    return m.group(1) if m else None


def load_compact_contexts(tok) -> tuple[list[dict[str, Any]], dict[tuple[int, int, bool], list[int]]]:
    contexts: list[dict[str, Any]] = []
    slots_by_bucket: dict[tuple[int, int, bool], list[int]] = collections.defaultdict(list)
    for row in load_jsonl(COMPACT_PAIRS):
        view_words = str(row.get("view_text", "")).split()
        source_words = str(row.get("source_text", "")).split()
        if not view_words:
            continue
        view_norms = {norm(w) for w in view_words if norm(w)}
        rec = {
            "pair_id": row.get("pair_id"),
            "key": row.get("key"),
            "doc_id": parse_doc_from_key(row.get("key")),
            "source_words": int(row.get("source_words", len(source_words))),
            "view_words_n": int(row.get("view_words", len(view_words))),
            "view_text": row.get("view_text"),
            "source_text": row.get("source_text"),
            "view_words": view_words,
            "view_norms": view_norms,
            "slots": [],
        }
        full_ids = token_ids(tok, " ".join(view_words))
        if len(full_ids) > SEQ_LEN:
            continue
        for wi, w in enumerate(view_words):
            sp = span_for_word(tok, view_words, wi)
            if not sp:
                continue
            n = norm(w)
            slot = {
                "word_index": wi,
                "word": w,
                "norm": n,
                "pieces": int(sp["pieces"]),
                "rel_pos": float(sp["rel_pos"]),
                "is_content": is_content(w),
            }
            rec["slots"].append(slot)
        ci = len(contexts)
        contexts.append(rec)
        for si, slot in enumerate(rec["slots"]):
            dec = min(9, max(0, int(float(slot["rel_pos"]) * 10)))
            # content flag buckets help avoid replacing content targets with pure function slots.
            buckets = [(int(slot["pieces"]), dec, bool(slot["is_content"]))]
            for b in buckets:
                slots_by_bucket[b].append((ci, si))
    return contexts, slots_by_bucket


def load_candidate_events(max_per_eval_set: int, seed: int) -> list[dict[str, Any]]:
    raw: list[dict[str, Any]] = []
    for d in load_jsonl(TRAIN_FIXED_EVENTS):
        if str(d.get("category")) == "retained_content":
            d = dict(d)
            d.setdefault("eval_set", "train_fixed_probe")
            d.setdefault("source_name", "train_fixed_probe")
            raw.append(d)
    for d in load_jsonl(SOURCE_DISJOINT_EVENTS):
        if str(d.get("category")) == "retained_content":
            d = dict(d)
            d.setdefault("eval_set", "source_disjoint")
            d.setdefault("source_name", "source_disjoint")
            raw.append(d)
    rng = random.Random(seed)
    by_set: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for d in raw:
        by_set[str(d.get("eval_set", "unknown"))].append(d)
    out: list[dict[str, Any]] = []
    for k, rows in sorted(by_set.items()):
        rng.shuffle(rows)
        out.extend(rows[:max_per_eval_set] if max_per_eval_set > 0 else rows)
    return out


def find_source_span(tok, source_words: list[str], target_norm: str, target_ids: list[int], target_rel: float) -> tuple[int, dict[str, Any]] | None:
    candidates = []
    for i, w in enumerate(source_words):
        if norm(w) != target_norm:
            continue
        sp = span_for_word(tok, source_words, i)
        if not sp:
            continue
        if sp["ids"] != target_ids:
            continue
        candidates.append((abs(float(sp["rel_pos"]) - target_rel), i, sp))
    if not candidates:
        return None
    candidates.sort(key=lambda x: (x[0], x[1]))
    return candidates[0][1], candidates[0][2]


def choose_counterfactual(tok, contexts: list[dict[str, Any]], slots_by_bucket, event: dict[str, Any], target_surface: str, target_norm: str, target_ids: list[int], target_rel: float, target_pieces: int, rng: random.Random) -> tuple[dict[str, Any] | None, str | None]:
    want_content = True
    dec = min(9, max(0, int(target_rel * 10)))
    candidate_refs: list[tuple[int, int]] = []
    for radius in range(0, 10):
        for d in sorted(set([dec - radius, dec + radius])):
            if d < 0 or d > 9:
                continue
            candidate_refs.extend(slots_by_bucket.get((target_pieces, d, want_content), []))
        if len(candidate_refs) >= 200:
            break
    if not candidate_refs:
        for d in range(10):
            candidate_refs.extend(slots_by_bucket.get((target_pieces, d, want_content), []))
    if not candidate_refs:
        return None, "no_piece_position_bucket"
    ev_pair = str(event.get("pair_id", ""))
    ev_doc = str(event.get("doc_id", parse_doc_from_key(event.get("key")) or ""))
    # Score a deterministic sample first, then if needed broaden.
    refs = list(candidate_refs)
    rng.shuffle(refs)
    refs = refs[:1000]
    best = None
    best_score = 1e9
    for ci, si in refs:
        c = contexts[ci]
        if str(c.get("pair_id")) == ev_pair:
            continue
        if ev_doc and str(c.get("doc_id")) == ev_doc:
            continue
        if target_norm in c.get("view_norms", set()) or target_norm in {norm(w) for w in str(c.get("source_text", "")).split()}:
            continue
        slot = c["slots"][si]
        words = list(c["view_words"])
        if si <= 0:
            # Avoid first-position ByteLevel boundary complications.
            continue
        words[si] = target_surface
        sp = span_for_word(tok, words, si)
        if not sp:
            continue
        if sp["ids"] != target_ids:
            continue
        ids_len = len(token_ids(tok, " ".join(words)))
        if ids_len > SEQ_LEN:
            continue
        score = abs(float(slot["rel_pos"]) - target_rel) + 0.02 * abs(len(words) - int(event.get("side_words_n", len(words))))
        if score < best_score:
            best_score = score
            best = {
                "counterfactual_pair_id": c.get("pair_id"),
                "counterfactual_key": c.get("key"),
                "counterfactual_doc_id": c.get("doc_id"),
                "counterfactual_original_word": slot.get("word"),
                "counterfactual_word_index": si,
                "counterfactual_rel_pos": float(slot["rel_pos"]),
                "counterfactual_view_words": len(words),
                "counterfactual_text_with_target": " ".join(words),
                "counterfactual_target_span": sp["span"],
                "counterfactual_match_score": score,
                "counterfactual_context_no_target_before_insert": True,
            }
    if best is None:
        return None, "no_valid_counterfactual_after_constraints"
    return best, None


def pct(xs, q):
    if not xs:
        return None
    xs = sorted(xs)
    idx = min(len(xs) - 1, max(0, int(round(q * (len(xs) - 1)))))
    return xs[idx]


def stats(xs):
    if not xs:
        return None
    return {
        "n": len(xs),
        "min": min(xs),
        "p05": pct(xs, 0.05),
        "p25": pct(xs, 0.25),
        "mean": statistics.mean(xs),
        "median": statistics.median(xs),
        "p75": pct(xs, 0.75),
        "p95": pct(xs, 0.95),
        "max": max(xs),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(OUT_DIR_DEFAULT))
    ap.add_argument("--max-per-eval-set", type=int, default=2500, help="0 means all retained events")
    ap.add_argument("--max-events-out", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=22243023)
    args = ap.parse_args()

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(str(TOKENIZER_PATH), local_files_only=True)
    rng = random.Random(args.seed)

    contexts, slots_by_bucket = load_compact_contexts(tok)
    candidates = load_candidate_events(args.max_per_eval_set, args.seed)
    skips = collections.Counter()
    prepared = []
    by_set = collections.Counter()
    piece_counts = []
    source_len_tokens = []
    rewrite_len_tokens = []
    counter_len_tokens = []
    rel_diffs = []
    word_len_diffs = []

    for e in candidates:
        if len(prepared) >= args.max_events_out:
            break
        target_surface = str(e.get("word", ""))
        target_norm = norm(target_surface)
        if not is_content(target_surface):
            skips["target_not_content"] += 1
            continue
        source_text = str(e.get("source_text", ""))
        side_text = str(e.get("side_text", e.get("rewrite_text", "")))
        source_words = source_text.split()
        side_words = side_text.split()
        if not source_words or not side_words:
            skips["missing_text"] += 1
            continue
        wi = int(e.get("word_index", -1))
        side_sp = span_for_word(tok, side_words, wi)
        if not side_sp:
            skips["bad_rewrite_span"] += 1
            continue
        if wi == 0 or side_sp["span"][0] == 0:
            # Keep BPE target identity strict and avoid start-token boundary effects.
            skips["rewrite_target_at_start"] += 1
            continue
        target_ids = list(side_sp["ids"])
        target_pieces = int(side_sp["pieces"])
        if target_pieces <= 0 or target_pieces > 8:
            skips["target_piece_count_out_of_range"] += 1
            continue
        src_match = find_source_span(tok, source_words, target_norm, target_ids, float(side_sp["rel_pos"]))
        if not src_match:
            skips["no_exact_source_occurrence_same_bpe"] += 1
            continue
        source_i, source_sp = src_match
        src_ids_len = len(token_ids(tok, " ".join(source_words)))
        rew_ids_len = len(token_ids(tok, " ".join(side_words)))
        if src_ids_len > SEQ_LEN or rew_ids_len > SEQ_LEN:
            skips["context_over_seq_len"] += 1
            continue
        ev = dict(e)
        ev["side_words_n"] = len(side_words)
        cf, reason = choose_counterfactual(tok, contexts, slots_by_bucket, ev, target_surface, target_norm, target_ids, float(side_sp["rel_pos"]), target_pieces, rng)
        if cf is None:
            skips[reason or "no_counterfactual"] += 1
            continue
        cf_ids_len = len(token_ids(tok, str(cf["counterfactual_text_with_target"])))
        rec = {
            "event_uid": str(e.get("event_uid", f"{e.get('eval_set','unknown')}|{e.get('_line_index')}|{e.get('pair_id')}|wi{wi}")),
            "eval_set": str(e.get("eval_set", "unknown")),
            "source_name": str(e.get("source_name", "unknown")),
            "pair_id": str(e.get("pair_id", "")),
            "key": e.get("key"),
            "doc_id": e.get("doc_id", parse_doc_from_key(e.get("key"))),
            "target_word": target_surface,
            "target_norm": target_norm,
            "target_ids": target_ids,
            "target_pieces": target_pieces,
            "source_text": source_text,
            "compact_text": side_text,
            "source_word_index": source_i,
            "compact_word_index": wi,
            "source_target_span": source_sp["span"],
            "compact_target_span": side_sp["span"],
            "source_rel_pos": float(source_sp["rel_pos"]),
            "compact_rel_pos": float(side_sp["rel_pos"]),
            "source_context_tokens": src_ids_len,
            "compact_context_tokens": rew_ids_len,
            "counterfactual_context_tokens": cf_ids_len,
            **cf,
            "match_diagnostics": {
                "abs_compact_counterfactual_relpos_delta": abs(float(side_sp["rel_pos"]) - float(cf["counterfactual_rel_pos"])),
                "abs_compact_counterfactual_wordlen_delta": abs(len(side_words) - int(cf["counterfactual_view_words"])),
                "same_doc_rejected": True,
                "target_absent_from_counterfactual_context_before_insert": True,
                "bpe_ids_identical_across_source_compact_counterfactual": True,
            },
        }
        prepared.append(rec)
        by_set[rec["eval_set"]] += 1
        piece_counts.append(target_pieces)
        source_len_tokens.append(src_ids_len)
        rewrite_len_tokens.append(rew_ids_len)
        counter_len_tokens.append(cf_ids_len)
        rel_diffs.append(rec["match_diagnostics"]["abs_compact_counterfactual_relpos_delta"])
        word_len_diffs.append(rec["match_diagnostics"]["abs_compact_counterfactual_wordlen_delta"])

    events_path = out_dir / "cross_realization_probe_preflight_events.jsonl"
    with events_path.open("w", encoding="utf-8") as f:
        for r in prepared:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    arm_availability = {}
    for name, path in ARMS.items():
        arm_availability[name] = {
            "path": str(path),
            "exists": path.exists(),
            "config_exists": (path / "config.json").exists(),
            "model_safetensors_exists": (path / "model.safetensors").exists(),
            "tokenizer_json_exists": (path / "tokenizer.json").exists(),
        }

    payload = {
        "status": "CROSS_REALIZATION_PROBE_PREFLIGHT",
        "meaning": "No-training feasibility and matching preflight for a saved-checkpoint cross-realization probe of source-shared/copied content targets under source, genuine compact, and counterfactual compact contexts.",
        "inputs": {
            "tokenizer_path": str(TOKENIZER_PATH),
            "tokenizer_json_sha256": sha256_file(TOKENIZER_PATH / "tokenizer.json"),
            "compact_pairs": str(COMPACT_PAIRS),
            "compact_pairs_sha256": sha256_file(COMPACT_PAIRS),
            "train_fixed_events": str(TRAIN_FIXED_EVENTS),
            "train_fixed_events_sha256": sha256_file(TRAIN_FIXED_EVENTS),
            "source_disjoint_events": str(SOURCE_DISJOINT_EVENTS),
            "source_disjoint_events_sha256": sha256_file(SOURCE_DISJOINT_EVENTS),
        },
        "parameters": {"max_per_eval_set": args.max_per_eval_set, "max_events_out": args.max_events_out, "seed": args.seed, "seq_len": SEQ_LEN},
        "counterfactual_pool": {
            "compact_contexts": len(contexts),
            "slot_bucket_count": len(slots_by_bucket),
            "slot_count": sum(len(v) for v in slots_by_bucket.values()),
            "matching_rule": "same target BPE ids; replacement slot same BPE piece count and content flag; close relative position; target absent from counterfactual source/view before insertion; different pair and different doc when doc id is known; final context <=256 tokens",
        },
        "candidate_events_loaded": len(candidates),
        "prepared_events": len(prepared),
        "by_eval_set": dict(sorted(by_set.items())),
        "skips": dict(skips),
        "target_piece_stats": stats(piece_counts),
        "source_context_token_stats": stats(source_len_tokens),
        "compact_context_token_stats": stats(rewrite_len_tokens),
        "counterfactual_context_token_stats": stats(counter_len_tokens),
        "counterfactual_relpos_abs_delta_stats": stats(rel_diffs),
        "counterfactual_wordlen_abs_delta_stats": stats(word_len_diffs),
        "events_file": str(events_path),
        "events_file_sha256": sha256_file(events_path),
        "arm_availability": arm_availability,
        "preflight_pass": bool(len(prepared) >= 1000 and all(v["exists"] and v["config_exists"] and v["model_safetensors_exists"] for v in arm_availability.values())),
        "scientific_reading": "If used for the full probe, these events should be analyzed by eval_set and bootstrapped by pair/doc. They are not selected from model outputs. A positive result must come from arm interactions (full/drop_abs/drop_copied/repeat/adjbreak), not from raw local NLL alone.",
    }
    with (out_dir / "cross_realization_probe_preflight.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    lines = [
        "# research cross-realization probe preflight",
        "",
        f"Prepared events: {len(prepared)} / candidates loaded: {len(candidates)}",
        f"By eval_set: {dict(sorted(by_set.items()))}",
        f"Skips: {dict(skips)}",
        f"Target pieces: {payload['target_piece_stats']}",
        f"Counterfactual rel-pos |delta|: {payload['counterfactual_relpos_abs_delta_stats']}",
        f"Counterfactual word-length |delta|: {payload['counterfactual_wordlen_abs_delta_stats']}",
        "",
        "## Arm availability",
    ]
    for name, a in arm_availability.items():
        lines.append(f"- {name}: exists={a['exists']} config={a['config_exists']} weights={a['model_safetensors_exists']} tokenizer={a['tokenizer_json_exists']} path=`{a['path']}`")
    lines.extend([
        "",
        f"Preflight pass: {payload['preflight_pass']}",
        "",
        "The generated JSONL stores source, compact, and counterfactual contexts plus exact target spans and BPE ids. It is a construction preflight only; no model logits were read.",
    ])
    with (out_dir / "cross_realization_probe_preflight.md").open("w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(json.dumps({
        "status": payload["status"],
        "prepared_events": len(prepared),
        "by_eval_set": payload["by_eval_set"],
        "preflight_pass": payload["preflight_pass"],
        "json": str(out_dir / "cross_realization_probe_preflight.json"),
        "events": str(events_path),
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
