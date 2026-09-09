#!/usr/bin/env python3
"""research: aligned-pair joint-visibility audit for repaired stagewise curricula.

The research row-chunked corpus fixed hidden-word exposure, but it may destroy the
mechanism that made clean-Qwen useful: each aligned original/rewrite pair appearing
jointly inside one attention window.  This script measures that mechanism directly
on the real clean-Qwen 10M corpus and pair metadata, comparing:

  1. inherited fixed-seq row-atomic corpus (seq256, batch256 geometry not needed)
  2. research row-chunked stagewise corpus (stage1 32-word chunks/seq64,
     stage2 64-word chunks/seq128, stage3 160-word chunks/seq256)
  3. a candidate pair-boundary-aware stagewise corpus where each Qwen
     original+rewrite pair is an atomic example and official/non-pair text is
     chunked for the current stage.

It does not read downstream evaluation items and does not train.  The central
outputs are complete/partial joint visibility rates for Qwen pairs by stage and
source, plus examples of pairs split by row chunking.
"""
from __future__ import annotations

import json
import math
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

ROOT = Path("experiments/archive/frontier_consolidation")
COMPACT_EXPERIENCE = Path("experiments/archive/compact_experience")
SCRIPT_DIR = ROOT / "scripts"
sys.path.insert(0, str(SCRIPT_DIR.resolve()))
# Use the exact tokenizer/word-start utilities that the in-flight research training uses
# (research loader handles both fast baseline16k and SentencePiece 40k tokenizers).
COMPACT_EXPERIENCE_SCRIPT_DIR = (COMPACT_EXPERIENCE / "scripts").resolve()
sys.path.insert(0, str(COMPACT_EXPERIENCE_SCRIPT_DIR))
import phase2_sota_trainer as tr  # type: ignore  # noqa: E402

QWEN10 = COMPACT_EXPERIENCE / "data/qwen_clean_aligned/training_corpora/qwen_aligned_10M.jsonl"
SELECTED_PAIRS = COMPACT_EXPERIENCE / "data/qwen_clean_aligned/selected_pairs.jsonl"
PAIR_ROW_META = COMPACT_EXPERIENCE / "data/qwen_clean_aligned/qwen_pair_packed_rows_meta.jsonl"
TOKENIZER16 = COMPACT_EXPERIENCE / "training/runs/qwen_clean_aligned_16k_seed43022/hf_model/chck_1M"
# The in-flight research stagewise run uses this shared 40k tokenizer.
TOKENIZER40 = COMPACT_EXPERIENCE / "data/shared_tokenizer/hf_tokenizer_40k_shared"

OUT_DIR = ROOT / "data/pair_joint_visibility"
JSON_OUT = OUT_DIR / "pair_joint_visibility_audit.json"
NOTE_OUT = (ROOT.parents[2] / 'research/notes/frontier_consolidation/pair_joint_visibility_audit.md')

STAGES = [
    {"stage": 1, "target_words": 3_000_000, "seq": 64, "batch": 512, "max_chunk_words": 32},
    {"stage": 2, "target_words": 3_000_000, "seq": 128, "batch": 256, "max_chunk_words": 64},
    {"stage": 3, "target_words": 4_000_000, "seq": 256, "batch": 128, "max_chunk_words": 160},
]


def load_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    total = 0
    with QWEN10.open(encoding="utf-8") as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue
            r = json.loads(line)
            text = str(r["text"])
            words = int(r.get("words", len(text.split())))
            actual = len(text.split())
            if words != actual:
                raise RuntimeError(f"row {i} word mismatch field={words} actual={actual}")
            rows.append({
                "row_index": i,
                "text": text,
                "words": words,
                "source": str(r.get("source", "")),
                "example_id": r.get("example_id"),
            })
            total += words
    if total != 10_000_000:
        raise RuntimeError(f"expected 10M words, got {total}")
    return rows


def load_pairs_and_meta() -> tuple[dict[str, dict[str, Any]], dict[int, dict[str, Any]]]:
    pairs: dict[str, dict[str, Any]] = {}
    with SELECTED_PAIRS.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            pairs[str(r["pair_id"])] = r
    meta: dict[int, dict[str, Any]] = {}
    with PAIR_ROW_META.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            meta[int(r["row_index"])] = r
    return pairs, meta


def split_words(text: str, n: int) -> list[str]:
    w = text.split()
    return [" ".join(w[i:i+n]) for i in range(0, len(w), n)]


class TokenStats:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.special_ids = set(tokenizer.all_special_ids)
        self.word_start_cache: dict[int, bool] = {}
        self.text_cache: dict[tuple[str, int], dict[str, int]] = {}

    def _word_start_flag(self, token_id: int) -> bool:
        v = self.word_start_cache.get(token_id)
        if v is None:
            s = self.tokenizer.convert_ids_to_tokens(int(token_id))
            v = bool(s is not None and tr.is_word_start(str(s)))
            self.word_start_cache[token_id] = v
        return v

    def visible(self, text: str, seq_len: int) -> dict[str, int]:
        key = (text, int(seq_len))
        got = self.text_cache.get(key)
        if got is not None:
            return got
        ids = self.tokenizer(text, add_special_tokens=False, truncation=False)["input_ids"]
        groups: list[int] = []
        gid = -1
        for i, tid in enumerate(ids):
            if int(tid) in self.special_ids:
                groups.append(-1)
                continue
            if gid < 0 or self._word_start_flag(int(tid)) or i == 0:
                gid += 1
            groups.append(gid)
        full_groups = [g for g in groups if g >= 0]
        visible_groups = [g for g in groups[:seq_len] if g >= 0]
        out = {
            "full_tokens": len(ids),
            "visible_tokens": min(len(ids), int(seq_len)),
            "full_word_groups": (max(full_groups) + 1) if full_groups else 0,
            "visible_word_groups": (max(visible_groups) + 1) if visible_groups else 0,
            "truncated_tokens": max(0, len(ids) - int(seq_len)),
        }
        self.text_cache[key] = out
        return out


def find_subspan(words: list[str], sub: list[str], start_hint: int = 0) -> int | None:
    if not sub:
        return start_hint
    # Usually pair spans are exactly adjacent in metadata order; use the hint first.
    if start_hint >= 0 and start_hint + len(sub) <= len(words) and words[start_hint:start_hint + len(sub)] == sub:
        return start_hint
    for i in range(0, len(words) - len(sub) + 1):
        if words[i:i + len(sub)] == sub:
            return i
    return None


def build_pair_records(rows: list[dict[str, Any]], pairs: dict[str, dict[str, Any]], meta: dict[int, dict[str, Any]]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    missing_meta = 0
    pos_errors = []
    for row in rows:
        if row["source"] != "qwen_pair_packed":
            continue
        m = meta.get(int(row["row_index"]))
        if m is None:
            missing_meta += 1
            continue
        row_words = str(row["text"]).split()
        cursor = 0
        for pid in m["pair_ids"]:
            p = pairs[str(pid)]
            ow = str(p["original"]).split()
            rw = str(p["rewrite"]).split()
            op = find_subspan(row_words, ow, cursor)
            if op is None:
                pos_errors.append({"row_index": row["row_index"], "pair_id": pid, "part": "original"})
                continue
            rp = find_subspan(row_words, rw, op + len(ow))
            if rp is None:
                pos_errors.append({"row_index": row["row_index"], "pair_id": pid, "part": "rewrite"})
                continue
            records.append({
                "pair_id": str(pid),
                "row_index": int(row["row_index"]),
                "row_words": int(row["words"]),
                "row_source": row["source"],
                "source": str(p.get("source")),
                "example_id": p.get("example_id"),
                "original_words": len(ow),
                "rewrite_words": len(rw),
                "pair_words": int(p["pair_words"]),
                "orig_start": op,
                "orig_end": op + len(ow),
                "rew_start": rp,
                "rew_end": rp + len(rw),
                "pair_start": op,
                "pair_end": rp + len(rw),
                "original": p["original"],
                "rewrite": p["rewrite"],
            })
            cursor = rp + len(rw)
    if missing_meta or pos_errors:
        raise RuntimeError(f"pair record build had missing_meta={missing_meta} pos_errors_sample={pos_errors[:5]}")
    if len(records) != len(pairs):
        # selected_pairs is intended to be the Qwen aligned set used in this corpus.
        print(f"WARNING: built {len(records)} pair records but selected_pairs has {len(pairs)}", file=sys.stderr)
    return records


def row_stage_parts(rows: list[dict[str, Any]]) -> dict[int, list[dict[str, Any]]]:
    """Replicate research row-part assignment before within-stage chunking."""
    out: dict[int, list[dict[str, Any]]] = defaultdict(list)
    stage_idx = 0
    stage_words = 0
    part_counter = 0
    for row in rows:
        words = str(row["text"]).split()
        cursor = 0
        while cursor < len(words):
            if stage_idx >= len(STAGES):
                raise RuntimeError("ran out of stages")
            spec = STAGES[stage_idx]
            remaining = int(spec["target_words"]) - stage_words
            if remaining <= 0:
                stage_idx += 1
                stage_words = 0
                continue
            take = min(remaining, len(words) - cursor)
            out[int(row["row_index"])].append({
                "stage": int(spec["stage"]),
                "seq": int(spec["seq"]),
                "max_chunk_words": int(spec["max_chunk_words"]),
                "part_id": part_counter,
                "row_start": cursor,
                "row_end": cursor + take,
            })
            part_counter += 1
            cursor += take
            stage_words += take
            if stage_words == int(spec["target_words"]):
                stage_idx += 1
                stage_words = 0
    return out


def candidate_pair_atomic_stage(pair_start_global: int, pair_words: int) -> int:
    """Stage assignment for a pair-atomic stream by cumulative pair start.

    This mirrors keeping the pair together: a pair starting before a boundary stays
    in that stage, even if it slightly crosses the nominal target.
    """
    if pair_start_global < STAGES[0]["target_words"]:
        return 1
    if pair_start_global < STAGES[0]["target_words"] + STAGES[1]["target_words"]:
        return 2
    return 3


def interval_overlap(a0: int, a1: int, b0: int, b1: int) -> int:
    return max(0, min(a1, b1) - max(a0, b0))


def init_acc() -> dict[str, Any]:
    return {
        "pairs": 0,
        "pair_words": 0,
        "complete_joint_pairs": 0,
        "partial_joint_pairs": 0,
        "orig_only_pairs": 0,
        "rewrite_only_pairs": 0,
        "neither_pairs": 0,
        "split_no_joint_pairs": 0,
        "complete_joint_words": 0,
        "partial_joint_words": 0,
        "sum_best_visible_orig_frac": 0.0,
        "sum_best_visible_rewrite_frac": 0.0,
        "sum_best_visible_pair_frac": 0.0,
        "sum_examples_touching_pair": 0,
        "max_examples_touching_pair": 0,
    }


def add_eval(acc: dict[str, Any], rec: dict[str, Any], ev: dict[str, Any]) -> None:
    pw = int(rec["pair_words"])
    acc["pairs"] += 1
    acc["pair_words"] += pw
    if ev["complete_joint"]:
        acc["complete_joint_pairs"] += 1
        acc["complete_joint_words"] += pw
    if ev["partial_joint"]:
        acc["partial_joint_pairs"] += 1
        acc["partial_joint_words"] += pw
    if ev["orig_seen"] and not ev["rew_seen"]:
        acc["orig_only_pairs"] += 1
    elif ev["rew_seen"] and not ev["orig_seen"]:
        acc["rewrite_only_pairs"] += 1
    elif (not ev["orig_seen"]) and (not ev["rew_seen"]):
        acc["neither_pairs"] += 1
    if (ev["orig_seen"] or ev["rew_seen"]) and not ev["partial_joint"] and ev["examples_touching"] > 1:
        acc["split_no_joint_pairs"] += 1
    acc["sum_best_visible_orig_frac"] += float(ev["best_visible_orig_frac"])
    acc["sum_best_visible_rewrite_frac"] += float(ev["best_visible_rewrite_frac"])
    acc["sum_best_visible_pair_frac"] += float(ev["best_visible_pair_frac"])
    acc["sum_examples_touching_pair"] += int(ev["examples_touching"])
    acc["max_examples_touching_pair"] = max(int(acc["max_examples_touching_pair"]), int(ev["examples_touching"]))


def finalize_acc(acc: dict[str, Any]) -> dict[str, Any]:
    out = dict(acc)
    n = max(1, int(acc["pairs"]))
    w = max(1, int(acc["pair_words"]))
    out["complete_joint_pair_rate"] = round(acc["complete_joint_pairs"] / n, 6)
    out["partial_joint_pair_rate"] = round(acc["partial_joint_pairs"] / n, 6)
    out["orig_only_pair_rate"] = round(acc["orig_only_pairs"] / n, 6)
    out["rewrite_only_pair_rate"] = round(acc["rewrite_only_pairs"] / n, 6)
    out["split_no_joint_pair_rate"] = round(acc["split_no_joint_pairs"] / n, 6)
    out["complete_joint_word_rate"] = round(acc["complete_joint_words"] / w, 6)
    out["partial_joint_word_rate"] = round(acc["partial_joint_words"] / w, 6)
    out["mean_best_visible_orig_frac"] = round(acc["sum_best_visible_orig_frac"] / n, 6)
    out["mean_best_visible_rewrite_frac"] = round(acc["sum_best_visible_rewrite_frac"] / n, 6)
    out["mean_best_visible_pair_frac"] = round(acc["sum_best_visible_pair_frac"] / n, 6)
    out["mean_examples_touching_pair"] = round(acc["sum_examples_touching_pair"] / n, 6)
    return out


def by_source_summary(accs: dict[str, dict[str, Any]]) -> dict[str, Any]:
    return {k: finalize_acc(v) for k, v in sorted(accs.items())}


def evaluate_from_segments(rec: dict[str, Any], segments: list[dict[str, int]], stats: TokenStats, row_words: list[str]) -> dict[str, Any]:
    """Evaluate whether any visible training example contains both views.

    segments have row_start,row_end,seq,stage, and represent possible examples.
    Visible word count is computed by tokenizer for that segment text and clipped
    onto the segment's row-coordinate word interval.
    """
    best_orig = 0
    best_rew = 0
    best_pair = 0
    any_orig = False
    any_rew = False
    any_partial = False
    any_complete = False
    touching = 0
    examples = []
    for seg in segments:
        s0, s1 = int(seg["row_start"]), int(seg["row_end"])
        if interval_overlap(rec["pair_start"], rec["pair_end"], s0, s1) <= 0:
            continue
        text = " ".join(row_words[s0:s1])
        vis = stats.visible(text, int(seg["seq"]))
        # Treat visible word groups as the prefix of segment whitespace words.
        visible_end = min(s1, s0 + int(vis["visible_word_groups"]))
        touching += 1
        orig_vis = interval_overlap(rec["orig_start"], rec["orig_end"], s0, visible_end)
        rew_vis = interval_overlap(rec["rew_start"], rec["rew_end"], s0, visible_end)
        pair_vis = interval_overlap(rec["pair_start"], rec["pair_end"], s0, visible_end)
        orig_full = orig_vis >= int(rec["original_words"])
        rew_full = rew_vis >= int(rec["rewrite_words"])
        pair_full = pair_vis >= int(rec["pair_words"])
        best_orig = max(best_orig, orig_vis)
        best_rew = max(best_rew, rew_vis)
        best_pair = max(best_pair, pair_vis)
        any_orig = any_orig or orig_vis > 0
        any_rew = any_rew or rew_vis > 0
        any_partial = any_partial or (orig_vis > 0 and rew_vis > 0)
        any_complete = any_complete or (orig_full and rew_full and pair_full)
        if len(examples) < 5:
            examples.append({
                "stage": int(seg.get("stage", 0)),
                "row_start": s0,
                "row_end": s1,
                "seq": int(seg["seq"]),
                "visible_row_end": visible_end,
                "orig_visible_words": orig_vis,
                "rewrite_visible_words": rew_vis,
                "pair_visible_words": pair_vis,
                "full_tokens": vis["full_tokens"],
                "visible_tokens": vis["visible_tokens"],
                "truncated_tokens": vis["truncated_tokens"],
            })
    return {
        "complete_joint": bool(any_complete),
        "partial_joint": bool(any_partial),
        "orig_seen": bool(any_orig),
        "rew_seen": bool(any_rew),
        "best_visible_orig_frac": round(best_orig / max(1, int(rec["original_words"])), 6),
        "best_visible_rewrite_frac": round(best_rew / max(1, int(rec["rewrite_words"])), 6),
        "best_visible_pair_frac": round(best_pair / max(1, int(rec["pair_words"])), 6),
        "examples_touching": touching,
        "example_slices": examples,
    }


def fixed_segments_for_pair(rec: dict[str, Any]) -> list[dict[str, int]]:
    return [{"stage": 0, "row_start": 0, "row_end": int(rec["row_words"]), "seq": 256}]


def rowchunk_segments_for_pair(rec: dict[str, Any], parts_by_row: dict[int, list[dict[str, Any]]]) -> list[dict[str, int]]:
    segs: list[dict[str, int]] = []
    for part in parts_by_row[int(rec["row_index"] )]:
        max_words = int(part["max_chunk_words"])
        p0, p1 = int(part["row_start"]), int(part["row_end"])
        for c0 in range(p0, p1, max_words):
            c1 = min(p1, c0 + max_words)
            segs.append({"stage": int(part["stage"]), "row_start": c0, "row_end": c1, "seq": int(part["seq"])})
    return segs


def pair_atomic_eval(rec: dict[str, Any], stage: int, stats: TokenStats) -> dict[str, Any]:
    spec = STAGES[stage - 1]
    text = str(rec["original"]) + " " + str(rec["rewrite"])
    # Represent the pair as its own row-coordinate segment from 0..pair_words.
    pair_rec = dict(rec)
    pair_rec.update({
        "row_words": int(rec["pair_words"]),
        "orig_start": 0,
        "orig_end": int(rec["original_words"]),
        "rew_start": int(rec["original_words"]),
        "rew_end": int(rec["pair_words"]),
        "pair_start": 0,
        "pair_end": int(rec["pair_words"]),
    })
    words = text.split()
    return evaluate_from_segments(pair_rec, [{"stage": stage, "row_start": 0, "row_end": len(words), "seq": int(spec["seq"])}], stats, words)


def summarize_arm(records: list[dict[str, Any]], row_text_words: dict[int, list[str]], stats: TokenStats, arm: str, parts_by_row: dict[int, list[dict[str, Any]]] | None = None, global_pair_starts: dict[str, int] | None = None) -> dict[str, Any]:
    overall = init_acc()
    by_stage: dict[str, dict[str, Any]] = defaultdict(init_acc)
    by_source: dict[str, dict[str, Any]] = defaultdict(init_acc)
    examples_bad: list[dict[str, Any]] = []
    pair_start_cursor = 0
    local_pair_starts: dict[str, int] = {}
    if global_pair_starts is None:
        for rec in records:
            local_pair_starts[str(rec["pair_id"])] = pair_start_cursor
            pair_start_cursor += int(rec["pair_words"])
        global_pair_starts = local_pair_starts

    for rec in records:
        if arm == "fixed_seq256_row_atomic":
            ev = evaluate_from_segments(rec, fixed_segments_for_pair(rec), stats, row_text_words[int(rec["row_index"])])
            stage_key = "fixed_seq256"
        elif arm == "stagewise_row_chunked":
            if parts_by_row is None:
                raise RuntimeError("rowchunk arm needs parts_by_row")
            segs = rowchunk_segments_for_pair(rec, parts_by_row)
            ev = evaluate_from_segments(rec, segs, stats, row_text_words[int(rec["row_index"])])
            # Assign pair to the stage(s) of examples touching it. If a rare stage boundary splits it,
            # use a compound stage key so the loss of joint visibility is visible.
            touched = sorted({int(s["stage"]) for s in segs if interval_overlap(rec["pair_start"], rec["pair_end"], int(s["row_start"]), int(s["row_end"])) > 0})
            stage_key = "+".join(map(str, touched)) if touched else "none"
        elif arm == "candidate_pair_atomic":
            st = candidate_pair_atomic_stage(int(global_pair_starts[str(rec["pair_id"])]), int(rec["pair_words"]))
            ev = pair_atomic_eval(rec, st, stats)
            stage_key = str(st)
        else:
            raise RuntimeError(f"unknown arm {arm}")
        add_eval(overall, rec, ev)
        add_eval(by_stage[stage_key], rec, ev)
        add_eval(by_source[str(rec["source"])], rec, ev)
        if (not ev["complete_joint"] or not ev["partial_joint"]) and len(examples_bad) < 20:
            examples_bad.append({
                "pair_id": rec["pair_id"],
                "row_index": rec["row_index"],
                "source": rec["source"],
                "original_words": rec["original_words"],
                "rewrite_words": rec["rewrite_words"],
                "pair_words": rec["pair_words"],
                "orig_span": [rec["orig_start"], rec["orig_end"]],
                "rewrite_span": [rec["rew_start"], rec["rew_end"]],
                "complete_joint": ev["complete_joint"],
                "partial_joint": ev["partial_joint"],
                "best_visible_pair_frac": ev["best_visible_pair_frac"],
                "examples_touching": ev["examples_touching"],
                "example_slices": ev["example_slices"],
                "original_preview": str(rec["original"])[:180],
                "rewrite_preview": str(rec["rewrite"])[:180],
            })
    return {
        "arm": arm,
        "overall": finalize_acc(overall),
        "by_stage": {k: finalize_acc(v) for k, v in sorted(by_stage.items())},
        "by_source": by_source_summary(by_source),
        "bad_examples_sample": examples_bad,
    }


def word_position_maps(records: list[dict[str, Any]]) -> dict[str, int]:
    out: dict[str, int] = {}
    cursor = 0
    for rec in records:
        out[str(rec["pair_id"])] = cursor
        cursor += int(rec["pair_words"])
    return out


def write_note(payload: dict[str, Any]) -> None:
    lines = [
        "# research — aligned-pair joint visibility audit",
        "",
        f"JSON: `{JSON_OUT}`",
        "",
        "This audit measures whether a Qwen original/rewrite pair is jointly present in one visible training example, not merely whether counted words are visible.",
        "",
        "## Overall pair visibility",
        "",
        "| tokenizer | arm | pairs | complete joint | partial joint | split no joint | mean examples touching | mean visible pair frac |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for tok_label, tok_payload in payload["tokenizers"].items():
        for arm_key in ["fixed_seq256_row_atomic", "stagewise_row_chunked", "candidate_pair_atomic"]:
            arm = tok_payload["arms"][arm_key]
            o = arm["overall"]
            lines.append(
                f"| {tok_label} | {arm_key} | {o['pairs']} | {o['complete_joint_pair_rate']} | "
                f"{o['partial_joint_pair_rate']} | {o['split_no_joint_pair_rate']} | "
                f"{o['mean_examples_touching_pair']} | {o['mean_best_visible_pair_frac']} |"
            )
    lines += ["", "## Stagewise row-chunked rates by stage (40k tokenizer)", "", "| stage key | pairs | complete joint | partial joint | split no joint | mean visible pair frac |", "|---|---:|---:|---:|---:|---:|"]
    row40 = payload["tokenizers"]["leader40k"]["arms"]["stagewise_row_chunked"]["by_stage"]
    for k, v in row40.items():
        lines.append(f"| {k} | {v['pairs']} | {v['complete_joint_pair_rate']} | {v['partial_joint_pair_rate']} | {v['split_no_joint_pair_rate']} | {v['mean_best_visible_pair_frac']} |")
    lines += ["", "## Scientific reading", ""]
    f40 = payload["tokenizers"]["leader40k"]["arms"]["fixed_seq256_row_atomic"]["overall"]
    r40 = payload["tokenizers"]["leader40k"]["arms"]["stagewise_row_chunked"]["overall"]
    p40 = payload["tokenizers"]["leader40k"]["arms"]["candidate_pair_atomic"]["overall"]
    lines += [
        f"- Inherited fixed-seq row-atomic clean-Qwen preserves complete joint visibility for {f40['complete_joint_pair_rate']:.3f} of pairs and partial joint visibility for {f40['partial_joint_pair_rate']:.3f}.",
        f"- research row chunking preserves complete joint visibility for only {r40['complete_joint_pair_rate']:.3f} of pairs, although counted-word visibility was near-unity; partial joint visibility is {r40['partial_joint_pair_rate']:.3f}.",
        f"- Candidate pair-atomic stagewise data would preserve complete joint visibility for {p40['complete_joint_pair_rate']:.3f} of pairs and partial joint visibility for {p40['partial_joint_pair_rate']:.3f} under the leader 40k tokenizer.",
        "- Therefore a fast score from the in-flight row-chunked model cannot by itself decide whether leader-style curriculum helps or hurts the paired-view mechanism; a pair-boundary-aware arm is needed if the row-chunked model is weak or ambiguous.",
    ]
    NOTE_OUT.parent.mkdir(parents=True, exist_ok=True)
    NOTE_OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = load_rows()
    pairs, meta = load_pairs_and_meta()
    records = build_pair_records(rows, pairs, meta)
    row_words = {int(r["row_index"]): str(r["text"]).split() for r in rows if r["source"] == "qwen_pair_packed"}
    parts_by_row = row_stage_parts(rows)
    global_pair_starts = word_position_maps(records)

    tokenizer_specs = {
        "baseline16k": TOKENIZER16,
        "leader40k": TOKENIZER40,
    }
    tok_payloads: dict[str, Any] = {}
    for label, path in tokenizer_specs.items():
        tokenizer = tr.make_portable_tokenizer(str(path))
        stats = TokenStats(tokenizer)
        arms = {}
        for arm in ["fixed_seq256_row_atomic", "stagewise_row_chunked", "candidate_pair_atomic"]:
            arms[arm] = summarize_arm(records, row_words, stats, arm, parts_by_row=parts_by_row, global_pair_starts=global_pair_starts)
        tok_payloads[label] = {
            "tokenizer_path": str(path),
            "vocab_size": len(tokenizer),
            "arms": arms,
        }

    payload = {
        "status": "PAIR_JOINT_VISIBILITY_AUDIT_DONE",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "input_jsonl": str(QWEN10),
        "selected_pairs": str(SELECTED_PAIRS),
        "pair_row_meta": str(PAIR_ROW_META),
        "n_qwen_pairs": len(records),
        "n_qwen_pair_rows": len(row_words),
        "stage_specs": STAGES,
        "mechanism_definition": "complete_joint means all whitespace words of original and rewrite are visible inside one training example/window; partial_joint means at least one visible word from each view occurs inside one training example/window.",
        "tokenizers": tok_payloads,
        "elapsed_sec": round(time.time() - t0, 3),
    }
    JSON_OUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_note(payload)
    print(json.dumps({
        "status": payload["status"],
        "json": str(JSON_OUT),
        "note": str(NOTE_OUT),
        "elapsed_sec": payload["elapsed_sec"],
        "leader40k_fixed_complete": tok_payloads["leader40k"]["arms"]["fixed_seq256_row_atomic"]["overall"]["complete_joint_pair_rate"],
        "leader40k_rowchunk_complete": tok_payloads["leader40k"]["arms"]["stagewise_row_chunked"]["overall"]["complete_joint_pair_rate"],
        "leader40k_pairatomic_complete": tok_payloads["leader40k"]["arms"]["candidate_pair_atomic"]["overall"]["complete_joint_pair_rate"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
