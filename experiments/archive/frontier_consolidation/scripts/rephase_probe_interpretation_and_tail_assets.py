#!/usr/bin/env python3
"""research: interpret research restart probes and prepare original-tail replay assets.

This is CPU-only. It records how the research restart trainers differ from the
original legal training loop, and computes the exact post-80M row segment from
the frozen 100M stream. The segment is the one to use if a restart score movement
has to be reproduced without changing the post-80M data order.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import json
import pathlib
import time
from collections import Counter
from typing import Any

from transformers import AutoTokenizer

def find_user_root() -> pathlib.Path:
    return _PUBLIC_ROOT


USER_ROOT = find_user_root()
STUDY = USER_ROOT / "experiments/archive/frontier_consolidation"
WORKSPACE = STUDY
OUT = WORKSPACE / "data/rephase_probe_interpretation"

RUN36 = WORKSPACE / "training/runs/complianttok_reinvest_seed43022_r2"
LOG36 = RUN36 / "training_log.jsonl"
METRICS36 = RUN36 / "scientific_metrics.json"
CHCK80 = RUN36 / "hf_model/chck_80M"
TRAIN100M = WORKSPACE / "data/density_cleanqwen_overlay_medium_riskhard/cleanqwen_fineweb_compact_view_reinvest_100M.jsonl"
POOL10M = WORKSPACE / "data/density_cleanqwen_overlay_medium_riskhard/cleanqwen_fineweb_compact_view_reinvest_10M.jsonl"
RESTART_TRAINER = WORKSPACE / "scripts/rephase_restart_trainer.py"
ORIGINAL_TRAINER = USER_ROOT / "experiments/archive/compact_experience/scripts/masking_curriculum_trainer.py"
TAIL_REPLAY_TRAINER = WORKSPACE / "scripts/original_tail_replay_trainer.py"

BATCH_SIZE = 256
SEQ_LEN = 256
TARGET_80M = 80_000_000
TOTAL_100M = 100_000_000
POOL_WORDS = 10_000_000


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def read_jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def count_jsonl_rows_words(path: pathlib.Path) -> tuple[int, int]:
    n = 0
    w = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            n += 1
            obj = json.loads(line)
            text = str(obj.get("text", ""))
            words = int(obj.get("words", len(text.split())))
            if words != len(text.split()):
                raise RuntimeError(f"word mismatch in {path} at row {n}: field={words} actual={len(text.split())}")
            w += words
    return n, w


def find_chck80_step(log_rows: list[dict[str, Any]]) -> dict[str, Any]:
    for r in log_rows:
        if int(r["cumulative_word_exposure"]) >= TARGET_80M:
            return r
    raise RuntimeError("No training-log step reaches 80M")


def tail_segment_summary(row_start: int, rows_per_pool: int) -> dict[str, Any]:
    total_rows = 0
    total_words = 0
    pass_words: Counter[int] = Counter()
    pass_rows: Counter[int] = Counter()
    first_rows: list[dict[str, Any]] = []
    last_rows: list[dict[str, Any]] = []
    with TRAIN100M.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if idx < row_start:
                continue
            if not line.strip():
                continue
            obj = json.loads(line)
            words = int(obj.get("words", len(str(obj.get("text", "")).split())))
            total_rows += 1
            total_words += words
            p = idx // rows_per_pool + 1
            pass_words[p] += words
            pass_rows[p] += 1
            mini = {k: obj.get(k) for k in obj.keys() if k != "text"}
            mini["global_row_index0"] = idx
            mini["pass_index1"] = p
            if len(first_rows) < 5:
                first_rows.append(mini)
            last_rows.append(mini)
            if len(last_rows) > 5:
                last_rows.pop(0)
    return {
        "row_start_index0": row_start,
        "tail_rows": total_rows,
        "tail_words": total_words,
        "pass_rows": dict(sorted(pass_rows.items())),
        "pass_words": dict(sorted(pass_words.items())),
        "first_tail_rows_no_text": first_rows,
        "last_tail_rows_no_text": last_rows,
    }


def restart_chunk_summary(tokenizer) -> dict[str, Any]:
    # Mirrors the research ContinuationDataset construction without materializing tensors.
    chunks = 0
    buffer_len = 0
    row_count = 0
    pool_words = 0
    token_count = 0
    with POOL10M.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            row_count += 1
            words = int(obj.get("words", len(str(obj.get("text", "")).split())))
            pool_words += words
            ids = tokenizer.encode(str(obj["text"]), add_special_tokens=False)
            token_count += len(ids)
            buffer_len += len(ids)
            made = buffer_len // SEQ_LEN
            chunks += made
            buffer_len = buffer_len % SEQ_LEN
    padded_tail_chunk = buffer_len > SEQ_LEN // 4
    if padded_tail_chunk:
        chunks += 1
    steps_per_pass = (chunks + BATCH_SIZE - 1) // BATCH_SIZE
    return {
        "pool_rows": row_count,
        "pool_words": pool_words,
        "pool_tokens_no_specials": token_count,
        "restart_token_chunks_per_pass": chunks,
        "restart_tail_buffer_tokens": buffer_len,
        "restart_has_padded_tail_chunk": padded_tail_chunk,
        "restart_steps_per_pass": steps_per_pass,
        "restart_two_pass_steps_nominal": steps_per_pass * 2,
        "restart_words_per_full_chunk_step_approx": POOL_WORDS / max(1, chunks) * BATCH_SIZE,
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    required = [LOG36, METRICS36, CHCK80, TRAIN100M, POOL10M, RESTART_TRAINER, ORIGINAL_TRAINER]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(missing)

    log_rows = read_jsonl(LOG36)
    metrics = json.loads(METRICS36.read_text(encoding="utf-8"))
    chck80_log = find_chck80_step(log_rows)
    parent_step = int(chck80_log["step"])
    parent_actual_words = int(chck80_log["cumulative_word_exposure"])
    row_start = parent_step * BATCH_SIZE

    pool_rows, pool_words = count_jsonl_rows_words(POOL10M)
    train_rows, train_words = count_jsonl_rows_words(TRAIN100M)
    if pool_words != POOL_WORDS or train_words != TOTAL_100M:
        raise RuntimeError({"pool_words": pool_words, "train_words": train_words})

    tokenizer = AutoTokenizer.from_pretrained(str(CHCK80), use_fast=True)
    chunk_info = restart_chunk_summary(tokenizer)
    tail_info = tail_segment_summary(row_start, pool_rows)

    original_tail_log = [r for r in log_rows if int(r["step"]) > parent_step]
    log_tail_words = sum(int(r["batch_words"]) for r in original_tail_log)

    trainer_differences = [
        {
            "axis": "post_80M_text_order",
            "original_continuous": "fixed order from frozen 100M stream after the batch that saved chck_80M",
            "restart": "fresh random shuffle of token chunks from the 10M pool for each continuation pass, train_rng_seed=43044",
            "scientific_consequence": "a score movement cannot by itself be assigned to optimizer-state reset",
        },
        {
            "axis": "training_unit",
            "original_continuous": "one JSONL row is one example, tokenized then truncated/padded to length 256",
            "restart": "all rows are concatenated into a token buffer and split into length-256 chunks",
            "scientific_consequence": "context boundaries, truncation, and source/rewrite adjacency differ from the original run",
        },
        {
            "axis": "word_accounting",
            "original_continuous": "batch word exposure is the sum of row word counts",
            "restart": "batch word exposure is approximated from selected token-chunk fraction of the 10M pool",
            "scientific_consequence": "nominal total exposure is close, but row/source composition around each checkpoint differs",
        },
        {
            "axis": "gradient_step_update",
            "original_continuous": "AdamW plus gradient clipping at norm 1.0",
            "restart": "AdamW without gradient clipping",
            "scientific_consequence": "large-update behavior is not identical, especially for base-LR restart",
        },
        {
            "axis": "learning_rate_schedule",
            "original_continuous": "single 2529-step cosine schedule, LR about 1.07e-4 at 80M and decaying to zero by 100M",
            "matched_lr": "fresh 20M cosine tail starting near 1.08e-4 with no warmup",
            "base_lr": "fresh 20M cosine tail at peak 1e-3 with warmup 0.06",
            "scientific_consequence": "matched-LR is closer to the inherited COMPACT_EXPERIENCE restart; base-LR is a separate late-schedule perturbation",
        },
    ]

    replay_plan = {
        "if_restart_score_moves_up": "run original-tail replay from chck_80M before interpreting the movement",
        "original_tail_replay_trainer": str(TAIL_REPLAY_TRAINER.relative_to(USER_ROOT)),
        "matched_lr_command_template": (
            "PYTHONDONTWRITEBYTECODE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "
            "python3 -B experiments/archive/frontier_consolidation/scripts/original_tail_replay_trainer.py "
            "--init_checkpoint experiments/archive/frontier_consolidation/training/runs/complianttok_reinvest_seed43022_r2/hf_model/chck_80M "
            "--train_file experiments/archive/frontier_consolidation/data/density_cleanqwen_overlay_medium_riskhard/cleanqwen_fineweb_compact_view_reinvest_100M.jsonl "
            "--run_dir experiments/archive/frontier_consolidation/training/runs/original_tail_replay_matched_lr_seed43044 "
            "--lr_mode matched --gpu 0 --train_rng_seed 43044"
        ),
        "base_lr_command_template": (
            "PYTHONDONTWRITEBYTECODE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "
            "python3 -B experiments/archive/frontier_consolidation/scripts/original_tail_replay_trainer.py "
            "--init_checkpoint experiments/archive/frontier_consolidation/training/runs/complianttok_reinvest_seed43022_r2/hf_model/chck_80M "
            "--train_file experiments/archive/frontier_consolidation/data/density_cleanqwen_overlay_medium_riskhard/cleanqwen_fineweb_compact_view_reinvest_100M.jsonl "
            "--run_dir experiments/archive/frontier_consolidation/training/runs/original_tail_replay_base_lr_seed43044 "
            "--lr_mode base --gpu 0 --train_rng_seed 43044"
        ),
    }

    summary: dict[str, Any] = {
        "status": "REPHASE_PROBE_INTERPRETATION_READY",
        "created_utc": now(),
        "parent_run": str(RUN36.relative_to(USER_ROOT)),
        "checkpoint_80M_log_record": chck80_log,
        "parent_step": parent_step,
        "parent_actual_words": parent_actual_words,
        "row_start_after_chck80": row_start,
        "pool_rows": pool_rows,
        "pool_words": pool_words,
        "train100m_rows": train_rows,
        "train100m_words": train_words,
        "tail_segment": tail_info,
        "tail_words_from_training_log_after_parent_step": log_tail_words,
        "tail_words_agree_with_100M_minus_parent": log_tail_words == TOTAL_100M - parent_actual_words,
        "restart_chunk_presentation": chunk_info,
        "trainer_differences": trainer_differences,
        "replay_plan": replay_plan,
    }

    out_json = OUT / "rephase_probe_interpretation_and_tail_assets.json"
    out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    md: list[str] = []
    md.append("# research rephase probe interpretation and original-tail replay assets")
    md.append("")
    md.append("CPU-only reading. No new training or evaluation was launched here.")
    md.append("")
    md.append("## Exact post-80M segment from the legal continuous run")
    md.append(f"- chck_80M was saved after training step `{parent_step}` with cumulative words `{parent_actual_words}`.")
    md.append(f"- Original row start after that checkpoint: `{row_start}` (step × batch_size = {parent_step} × {BATCH_SIZE}).")
    md.append(f"- Tail rows in frozen 100M stream: `{tail_info['tail_rows']}`.")
    md.append(f"- Tail words in frozen 100M stream: `{tail_info['tail_words']}`.")
    md.append(f"- Tail words from original training log after step {parent_step}: `{log_tail_words}`.")
    md.append(f"- Agreement with 100M-parent words: `{summary['tail_words_agree_with_100M_minus_parent']}`.")
    md.append(f"- Tail pass rows: `{tail_info['pass_rows']}`.")
    md.append(f"- Tail pass words: `{tail_info['pass_words']}`.")
    md.append("")
    md.append("## research restart presentation")
    for k, v in chunk_info.items():
        md.append(f"- `{k}`: `{v}`")
    md.append("")
    md.append("## Why research is only a bounded score probe")
    md.append("| Axis | Original continuous run | research restart | Consequence |")
    md.append("|---|---|---|---|")
    for d in trainer_differences:
        orig = d.get("original_continuous", "")
        restart = d.get("restart") or f"matched: {d.get('matched_lr')} / base: {d.get('base_lr')}"
        md.append(f"| {d['axis']} | {orig} | {restart} | {d['scientific_consequence']} |")
    md.append("")
    md.append("## If a restart improves")
    md.append("Before assigning the movement to optimizer-state reset or spending more GPU, run the same restart style on the original post-80M stream segment.")
    md.append(f"- Replay trainer: `{replay_plan['original_tail_replay_trainer']}`")
    md.append("- Matched-LR template:")
    md.append(f"  `{replay_plan['matched_lr_command_template']}`")
    md.append("- Base-LR template:")
    md.append(f"  `{replay_plan['base_lr_command_template']}`")
    md.append("")
    md.append(f"JSON: `{out_json.relative_to(USER_ROOT)}`")
    (OUT / "rephase_probe_interpretation_and_tail_assets.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    print(json.dumps({
        "status": summary["status"],
        "out_json": str(out_json.relative_to(USER_ROOT)),
        "out_md": str((OUT / "rephase_probe_interpretation_and_tail_assets.md").relative_to(USER_ROOT)),
        "parent_step": parent_step,
        "parent_actual_words": parent_actual_words,
        "row_start": row_start,
        "tail_words": tail_info["tail_words"],
        "restart_chunks_per_pass": chunk_info["restart_token_chunks_per_pass"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
