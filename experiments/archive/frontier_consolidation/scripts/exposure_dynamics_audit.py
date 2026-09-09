#!/usr/bin/env python3
"""research: audit exposure dynamics before running leader-recipe pilots.

The research launch script must not be used unchanged: its sequence-length
schedule crops tokenized examples while still counting all whitespace words, and
its baseline arm used the wrong batch size.  This script measures, on the real
clean-Qwen 10M corpus and the real baseline16k tokenizer, how much counted word
exposure is actually visible under the old schedule and under repaired candidate
stagewise schedules.

Outputs a JSON summary and a short Markdown note under data and
notes.  No downstream evaluation data is read.
"""
from __future__ import annotations

import json
import math
import statistics
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import sys

ROOT = Path("experiments/archive/frontier_consolidation")
COMPACT_EXPERIENCE = Path("experiments/archive/compact_experience")
SCRIPT_DIR = ROOT / "scripts"
sys.path.insert(0, str(SCRIPT_DIR.resolve()))
import masking_curriculum_trainer_v2 as tr  # type: ignore

QWEN10 = COMPACT_EXPERIENCE / "data/qwen_clean_aligned/training_corpora/qwen_aligned_10M.jsonl"
SELECTED_PAIRS = COMPACT_EXPERIENCE / "data/qwen_clean_aligned/selected_pairs.jsonl"
PAIR_ROW_META = COMPACT_EXPERIENCE / "data/qwen_clean_aligned/qwen_pair_packed_rows_meta.jsonl"
TOKENIZER_PATH = COMPACT_EXPERIENCE / "training/runs/qwen_clean_aligned_16k_seed43022/hf_model/chck_1M"
OUT_DIR = ROOT / "data/exposure_audit"
JSON_OUT = OUT_DIR / "exposure_dynamics_audit.json"
NOTE_OUT = (ROOT.parents[2] / 'research/notes/frontier_consolidation/exposure_dynamics_audit.md')

STAGE_SPECS = [
    {"stage": 1, "target_words": 3_000_000, "seq": 64, "batch": 512, "max_chunk_words": 32},
    {"stage": 2, "target_words": 3_000_000, "seq": 128, "batch": 256, "max_chunk_words": 64},
    {"stage": 3, "target_words": 4_000_000, "seq": 256, "batch": 128, "max_chunk_words": 160},
]


def load_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    total = 0
    with path.open(encoding="utf-8") as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue
            r = json.loads(line)
            text = str(r["text"])
            words = int(r.get("words", len(text.split())))
            actual = len(text.split())
            if words != actual:
                raise RuntimeError(f"word mismatch row {i}: field={words} actual={actual}")
            rows.append({
                "row_index": i,
                "text": text,
                "words": words,
                "source": str(r.get("source", "")),
                "example_id": r.get("example_id"),
            })
            total += words
    if total != 10_000_000:
        raise RuntimeError(f"Expected 10M words in {path}, got {total}")
    return rows


def load_pair_maps() -> tuple[dict[str, dict[str, Any]], dict[int, dict[str, Any]]]:
    pairs: dict[str, dict[str, Any]] = {}
    with SELECTED_PAIRS.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            pairs[str(r["pair_id"])] = r
    row_meta: dict[int, dict[str, Any]] = {}
    with PAIR_ROW_META.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            row_meta[int(r["row_index"])] = r
    return pairs, row_meta


class TokenStats:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.special_ids = set(tokenizer.all_special_ids)
        self.word_start_cache: dict[int, bool] = {}
        self.text_cache: dict[str, tuple[list[int], list[int]]] = {}

    def _word_start_flag(self, token_id: int) -> bool:
        v = self.word_start_cache.get(token_id)
        if v is None:
            s = self.tokenizer.convert_ids_to_tokens(int(token_id))
            v = bool(s is not None and tr.is_word_start(str(s)))
            self.word_start_cache[token_id] = v
        return v

    def encode_groups(self, text: str) -> tuple[list[int], list[int]]:
        cached = self.text_cache.get(text)
        if cached is not None:
            return cached
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
        out = (list(map(int, ids)), groups)
        self.text_cache[text] = out
        return out

    def visible(self, text: str, seq_len: int) -> dict[str, int]:
        ids, groups = self.encode_groups(text)
        vis_ids = ids[:seq_len]
        vis_groups = [g for g in groups[:seq_len] if g >= 0]
        full_groups = [g for g in groups if g >= 0]
        return {
            "full_tokens": len(ids),
            "visible_tokens": len(vis_ids),
            "full_word_groups": (max(full_groups) + 1) if full_groups else 0,
            "visible_word_groups": (max(vis_groups) + 1) if vis_groups else 0,
            "truncated_tokens": max(0, len(ids) - seq_len),
        }


def empty_acc() -> dict[str, Any]:
    return {
        "rows": 0,
        "counted_words": 0,
        "visible_tokens": 0,
        "full_tokens": 0,
        "visible_word_groups": 0,
        "full_word_groups": 0,
        "truncated_tokens": 0,
        "truncated_rows": 0,
    }


def add_visible(acc: dict[str, Any], row: dict[str, Any], vis: dict[str, int]) -> None:
    acc["rows"] += 1
    acc["counted_words"] += int(row["words"])
    for k in ["visible_tokens", "full_tokens", "visible_word_groups", "full_word_groups", "truncated_tokens"]:
        acc[k] += int(vis[k])
    if vis["truncated_tokens"] > 0:
        acc["truncated_rows"] += 1


def finalize_acc(acc: dict[str, Any]) -> dict[str, Any]:
    out = dict(acc)
    w = max(1, int(out["counted_words"]))
    ft = max(1, int(out["full_tokens"]))
    fg = max(1, int(out["full_word_groups"]))
    out["visible_tokens_per_counted_word"] = round(out["visible_tokens"] / w, 6)
    out["visible_word_groups_per_counted_word"] = round(out["visible_word_groups"] / w, 6)
    out["visible_token_fraction"] = round(out["visible_tokens"] / ft, 6)
    out["visible_group_fraction"] = round(out["visible_word_groups"] / fg, 6)
    out["truncated_row_fraction"] = round(out["truncated_rows"] / max(1, out["rows"]), 6)
    out["hidden_word_group_deficit_vs_counted"] = int(out["counted_words"] - out["visible_word_groups"])
    return out


def summarize_sources(source_acc: dict[str, dict[str, Any]]) -> dict[str, Any]:
    return {k: finalize_acc(v) for k, v in sorted(source_acc.items())}


def seq_for_old_schedule(frac: float) -> int:
    return 64 if frac < 0.7 else 256


def simulate_v2(rows: list[dict[str, Any]], stats: TokenStats, name: str, batch_size: int, scheduled: bool) -> dict[str, Any]:
    total_steps = math.ceil(len(rows) / batch_size)
    total = empty_acc()
    by_source: dict[str, dict[str, Any]] = defaultdict(empty_acc)
    by_phase: dict[str, dict[str, Any]] = defaultdict(empty_acc)
    first_long: dict[str, Any] | None = None
    batch_words: list[int] = []
    padded_positions = 0
    for step in range(1, total_steps + 1):
        batch = rows[(step - 1) * batch_size: step * batch_size]
        frac = (step - 1) / max(1, total_steps)
        seq_len = seq_for_old_schedule(frac) if scheduled else 256
        phase = "early_seq64" if seq_len == 64 else "late_seq256"
        if scheduled and seq_len == 256 and first_long is None:
            first_long = {"step": step, "progress_frac": frac, "counted_words_before_batch": sum(batch_words)}
        bwords = sum(int(r["words"]) for r in batch)
        batch_words.append(bwords)
        padded_positions += len(batch) * seq_len
        for r in batch:
            vis = stats.visible(str(r["text"]), seq_len)
            add_visible(total, r, vis)
            add_visible(by_source[str(r["source"])], r, vis)
            add_visible(by_phase[phase], r, vis)
    bw_mean = statistics.mean(batch_words) if batch_words else 0.0
    return {
        "name": name,
        "kind": "current_v2_schedule_simulation",
        "batch_size": batch_size,
        "scheduled_seq64_to_256": scheduled,
        "total_steps": total_steps,
        "batch_words_mean": round(bw_mean, 3),
        "batch_words_min": min(batch_words) if batch_words else None,
        "batch_words_max": max(batch_words) if batch_words else None,
        "padded_positions": padded_positions,
        "padded_positions_per_counted_word": round(padded_positions / max(1, total["counted_words"]), 6),
        "first_seq256_batch": first_long,
        "overall": finalize_acc(total),
        "by_phase": {k: finalize_acc(v) for k, v in sorted(by_phase.items())},
        "by_source": summarize_sources(by_source),
    }


def split_words(text: str, n: int) -> list[str]:
    words = text.split()
    return [" ".join(words[i:i+n]) for i in range(0, len(words), n)]


def build_pair_atomic_units(rows: list[dict[str, Any]], pairs: dict[str, dict[str, Any]], row_meta: dict[int, dict[str, Any]]) -> list[dict[str, Any]]:
    units: list[dict[str, Any]] = []
    missing = 0
    for r in rows:
        if r["source"] != "qwen_pair_packed":
            units.append({**r, "unit_type": "official_or_filler", "pair_id": None})
            continue
        meta = row_meta.get(int(r["row_index"]))
        if meta is None:
            missing += 1
            # Fall back to row atomic rather than silently splitting the pair row.
            units.append({**r, "unit_type": "qwen_row_atomic_missing_meta", "pair_id": None})
            continue
        for pid in meta["pair_ids"]:
            p = pairs[str(pid)]
            text = str(p["original"]) + " " + str(p["rewrite"])
            words = int(p["pair_words"])
            if words != len(text.split()):
                raise RuntimeError(f"pair word mismatch {pid}: field={words} actual={len(text.split())}")
            units.append({
                "row_index": r["row_index"],
                "text": text,
                "words": words,
                "source": "qwen_pair_atomic",
                "component_source": p.get("source"),
                "example_id": p.get("example_id"),
                "unit_type": "qwen_pair_atomic",
                "pair_id": str(pid),
            })
    if missing:
        print(f"WARNING: {missing} qwen rows missing metadata", file=sys.stderr)
    if sum(int(u["words"]) for u in units) != 10_000_000:
        raise RuntimeError("pair-atomic unit word total changed")
    return units


def assign_units_to_stages(units: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    """Assign units by cumulative word position; never split qwen pair units."""
    stages: list[list[dict[str, Any]]] = [[] for _ in STAGE_SPECS]
    targets = [s["target_words"] for s in STAGE_SPECS]
    stage_idx = 0
    stage_words = 0
    for u in units:
        uwords = int(u["words"])
        # Split official/filler text at stage boundary if needed; qwen pair units stay intact.
        remaining_text_words = u["text"].split()
        while remaining_text_words:
            if stage_idx >= len(STAGE_SPECS):
                raise RuntimeError("ran out of stages")
            remaining_capacity = targets[stage_idx] - stage_words
            if remaining_capacity <= 0:
                stage_idx += 1
                stage_words = 0
                continue
            if u["unit_type"] == "qwen_pair_atomic":
                part_words = len(remaining_text_words)
                stages[stage_idx].append(dict(u))
                stage_words += part_words
                remaining_text_words = []
                if stage_words >= targets[stage_idx]:
                    stage_idx += 1
                    stage_words = 0
            else:
                take = min(len(remaining_text_words), remaining_capacity)
                part = dict(u)
                part["text"] = " ".join(remaining_text_words[:take])
                part["words"] = take
                part["unit_type"] = str(u["unit_type"]) + "_stagepart"
                stages[stage_idx].append(part)
                stage_words += take
                remaining_text_words = remaining_text_words[take:]
                if stage_words >= targets[stage_idx]:
                    stage_idx += 1
                    stage_words = 0
    return stages


def expand_stage_examples(stage_units: list[dict[str, Any]], spec: dict[str, Any], preserve_pairs: bool) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    max_words = int(spec["max_chunk_words"])
    for u in stage_units:
        if preserve_pairs and u.get("unit_type") == "qwen_pair_atomic":
            examples.append(dict(u))
        else:
            for j, chunk in enumerate(split_words(str(u["text"]), max_words)):
                if not chunk:
                    continue
                part = dict(u)
                part["text"] = chunk
                part["words"] = len(chunk.split())
                part["chunk_id"] = j
                if u.get("unit_type") == "qwen_pair_atomic":
                    part["unit_type"] = "qwen_pair_chunked"
                else:
                    part["unit_type"] = str(u.get("unit_type", "unit")) + "_chunked"
                examples.append(part)
    return examples


def simulate_stagewise(stages_units: list[list[dict[str, Any]]], stats: TokenStats, name: str, preserve_pairs: bool) -> dict[str, Any]:
    total = empty_acc()
    by_source: dict[str, dict[str, Any]] = defaultdict(empty_acc)
    stage_out: list[dict[str, Any]] = []
    padded_positions = 0
    total_steps = 0
    qwen_long_units = {"stage1_seq64": 0, "stage2_seq128": 0, "stage3_seq256": 0}
    qwen_long_words = {"stage1_seq64": 0, "stage2_seq128": 0, "stage3_seq256": 0}
    for spec, units in zip(STAGE_SPECS, stages_units):
        examples = expand_stage_examples(units, spec, preserve_pairs=preserve_pairs)
        st_acc = empty_acc()
        st_by_source: dict[str, dict[str, Any]] = defaultdict(empty_acc)
        seq_len = int(spec["seq"])
        batch = int(spec["batch"])
        n_steps = math.ceil(len(examples) / batch)
        total_steps += n_steps
        padded_positions += len(examples) * seq_len
        for ex in examples:
            vis = stats.visible(str(ex["text"]), seq_len)
            add_visible(st_acc, ex, vis)
            add_visible(total, ex, vis)
            add_visible(st_by_source[str(ex["source"])], ex, vis)
            add_visible(by_source[str(ex["source"])], ex, vis)
            if str(ex.get("source")) == "qwen_pair_atomic" and vis["truncated_tokens"] > 0:
                key = f"stage{spec['stage']}_seq{seq_len}"
                qwen_long_units[key] += 1
                qwen_long_words[key] += int(ex["words"])
        stage_out.append({
            "stage": spec["stage"],
            "seq": seq_len,
            "batch": batch,
            "max_chunk_words": spec["max_chunk_words"],
            "target_words_nominal": spec["target_words"],
            "examples": len(examples),
            "steps": n_steps,
            "padded_positions": len(examples) * seq_len,
            "overall": finalize_acc(st_acc),
            "by_source": summarize_sources(st_by_source),
        })
    return {
        "name": name,
        "kind": "repaired_stagewise_candidate_simulation",
        "preserve_qwen_pairs_as_atomic_examples": preserve_pairs,
        "stage_specs": STAGE_SPECS,
        "total_steps": total_steps,
        "padded_positions": padded_positions,
        "padded_positions_per_counted_word": round(padded_positions / max(1, total["counted_words"]), 6),
        "overall": finalize_acc(total),
        "stages": stage_out,
        "by_source": summarize_sources(by_source),
        "qwen_pair_atomic_truncated_units": qwen_long_units,
        "qwen_pair_atomic_truncated_words": qwen_long_words,
    }


def write_note(payload: dict[str, Any]) -> None:
    arms = payload["arms"]
    lines = [
        "# research — exposure dynamics audit before pilot training",
        "",
        f"JSON: `{JSON_OUT}`",
        "",
        "This audit uses only the real clean-Qwen 10M training corpus and baseline16k tokenizer; it does not read any downstream evaluation items.",
        "",
        "## Key measurements",
        "",
        "| schedule | steps | padded/word | visible groups/word | visible token fraction | qwen visible groups/word | qwen visible token fraction | first long batch |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for key in ["old_A0_batch64_fixed256", "repaired_A0_batch256_fixed256", "old_A1_A2_A3_batch256_seq64_to_256", "stagewise_pair_atomic", "stagewise_row_chunked"]:
        a = arms[key]
        q = a.get("by_source", {}).get("qwen_pair_packed") or a.get("by_source", {}).get("qwen_pair_atomic") or {}
        first = a.get("first_seq256_batch") or ""
        if isinstance(first, dict):
            first = f"step {first.get('step')} after {first.get('counted_words_before_batch')} words"
        lines.append(
            f"| {key} | {a.get('total_steps')} | {a.get('padded_positions_per_counted_word')} | "
            f"{a['overall'].get('visible_word_groups_per_counted_word')} | {a['overall'].get('visible_token_fraction')} | "
            f"{q.get('visible_word_groups_per_counted_word')} | {q.get('visible_token_fraction')} | {first} |"
        )
    old = arms["old_A1_A2_A3_batch256_seq64_to_256"]
    fixed = arms["repaired_A0_batch256_fixed256"]
    pa = arms["stagewise_pair_atomic"]
    lines += [
        "",
        "## Scientific reading",
        "",
        f"- The research baseline launch used batch64, which would create {arms['old_A0_batch64_fixed256']['total_steps']} updates for a 10M pass; the repaired baseline batch256 uses {fixed['total_steps']} updates and matches the 2,515-step 100M clean-Qwen geometry (one tenth of the full run).",
        f"- The unchanged seq64→256 schedule with batch256 would count 10M words but expose only {old['overall']['visible_word_groups_per_counted_word']:.3f} visible word groups per counted word overall; for Qwen pair rows the ratio is {old['by_source']['qwen_pair_packed']['visible_word_groups_per_counted_word']:.3f}. This is not the leader's inverse-batch low-truncation dynamics.",
        f"- The repaired pair-atomic stagewise candidate uses dynamic batches 512/256/128 and reaches {pa['overall']['visible_word_groups_per_counted_word']:.3f} visible word groups per counted word overall, but it still preserves Qwen pairs as atomic rows; truncated qwen pair units by stage are {pa['qwen_pair_atomic_truncated_units']}. A full data-mechanism interpretation therefore still needs pair-length-aware construction, not just this training control.",
        "- Loss values from these schedules are not comparable across masking modes; the next decision must use matched fast official-task outcomes across all trained arms.",
    ]
    NOTE_OUT.parent.mkdir(parents=True, exist_ok=True)
    NOTE_OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = load_rows(QWEN10)
    tokenizer = tr.make_portable_tokenizer(str(TOKENIZER_PATH))
    stats = TokenStats(tokenizer)
    pairs, row_meta = load_pair_maps()
    units = build_pair_atomic_units(rows, pairs, row_meta)
    stages_units = assign_units_to_stages(units)

    arms: dict[str, Any] = {}
    arms["old_A0_batch64_fixed256"] = simulate_v2(rows, stats, "old_A0_batch64_fixed256", batch_size=64, scheduled=False)
    arms["repaired_A0_batch256_fixed256"] = simulate_v2(rows, stats, "repaired_A0_batch256_fixed256", batch_size=256, scheduled=False)
    arms["old_A1_A2_A3_batch256_seq64_to_256"] = simulate_v2(rows, stats, "old_A1_A2_A3_batch256_seq64_to_256", batch_size=256, scheduled=True)
    arms["stagewise_pair_atomic"] = simulate_stagewise(stages_units, stats, "stagewise_pair_atomic", preserve_pairs=True)
    arms["stagewise_row_chunked"] = simulate_stagewise(stages_units, stats, "stagewise_row_chunked", preserve_pairs=False)

    payload = {
        "status": "EXPOSURE_DYNAMICS_AUDIT_DONE",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "input_jsonl": str(QWEN10),
        "input_words": sum(int(r["words"]) for r in rows),
        "input_rows": len(rows),
        "tokenizer_path": str(TOKENIZER_PATH),
        "tokenizer_vocab_size": len(tokenizer),
        "selected_pairs": str(SELECTED_PAIRS),
        "pair_row_meta": str(PAIR_ROW_META),
        "stage_specs": STAGE_SPECS,
        "arms": arms,
        "elapsed_sec": round(time.time() - t0, 3),
        "interpretation": (
            "The unchanged research panel should not be launched. Batch64 baseline changes update count; "
            "seq64-to-256 with full 160-word rows hides a large fraction of counted word groups. "
            "Use dynamic-batch, low-truncation stagewise training and matched fast official-task scores."
        ),
    }
    JSON_OUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_note(payload)
    print(json.dumps({
        "status": payload["status"],
        "json": str(JSON_OUT),
        "note": str(NOTE_OUT),
        "elapsed_sec": payload["elapsed_sec"],
        "old_sched_visible_groups_per_word": arms["old_A1_A2_A3_batch256_seq64_to_256"]["overall"]["visible_word_groups_per_counted_word"],
        "stagewise_pair_atomic_visible_groups_per_word": arms["stagewise_pair_atomic"]["overall"]["visible_word_groups_per_counted_word"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
