#!/usr/bin/env python3
"""research: accumulated-trainer 16k equivalence control.

Purpose
-------
The legal-40k compact_view_reinvest runs use the research accumulated trainer
(effective batch 256 via 4x64 microbatches) after direct batch-256 training OOMed.
The legal-16k corrected-tokenizer baseline used the original full-batch trainer.
This script prepares a cheap post-hoc control: run the accumulated trainer with the
EXACT legal-16k tokenizer and seed43022 for the first 4,030,900 words (the same
row-boundary exposure as the base trainer's chck_4M, research) while forcing the
LR scheduler denominator to the full 100M run length (2529 steps). It then compares
loss/log geometry against the existing base-16k seed43022 log.

Do not launch while the two legal-40k 100M trainings occupy the GPUs. Use
`--mode preflight` now; use `--mode train` only after a GPU is free.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import shutil
import statistics
import subprocess
import sys
from typing import Any

STUDY = pathlib.Path("experiments/archive/representation_and_objectives")
WS = STUDY
OUT_DIR = WS / "data/accum16k_equivalence_control"
AI_RUN_ROOT = WS / "training/runs"

ACCUM_TRAINER = WS / "scripts/accumulated_masking_curriculum_trainer.py"
TOKENIZER16_DIR = WS / "data/strictsmall_tokenizer_retrain/strictsmall_compact_reinvest_16k_tokenizer"
TRAIN_100M = pathlib.Path("experiments/archive/frontier_consolidation/data/density_cleanqwen_overlay_medium_riskhard/cleanqwen_fineweb_compact_view_reinvest_100M.jsonl")
META = pathlib.Path("experiments/archive/frontier_consolidation/data/density_cleanqwen_overlay_medium_riskhard/density_cleanqwen_rowholdout_overlay_metadata.json")
BASE16_RUN = WS / "training/runs/strictsmalltok_compact_view_reinvest_seed43022"
RUN_DIR = AI_RUN_ROOT / "accum16k_equivalence_seed43022_4M"

EXPECTED_TOK16_SHA = "4a95a2a2ead21813a409c5ad7c2c008fbf418dcf3e0dc17bd970bd9ce25ea738"
EXPECTED_100M_SHA = "3dd19f09deeca44d6b07340e70cd15baa4b5c7bffe0bd462ff37536e470e7691"
STOP_WORDS = 4_030_900  # base-16k seed43022 chck_4M actual_cumulative_word_exposure, row boundary
FULL_RUN_TOTAL_STEPS = 2529  # from base-16k seed43022 scientific_metrics.json
COMPARE_STEPS = 102  # base-16k step at STOP_WORDS


def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path: pathlib.Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def command() -> list[str]:
    return [
        sys.executable,
        "-B",
        str(ACCUM_TRAINER),
        "--example_jsonl", str(TRAIN_100M),
        "--example_jsonl_label", "cleanqwen_fineweb_compact_view_reinvest_accum16k_equivalence_4M",
        "--example_jsonl_meta", str(META),
        "--output_dir", str(RUN_DIR),
        "--tokenizer_path", str(TOKENIZER16_DIR),
        "--tokenizer_label", "strictsmall_compact_reinvest_16k_accum_equivalence",
        "--hidden_size", "480",
        "--n_layer", "8",
        "--n_head", "8",
        "--ffn_mult", "4",
        "--seed", "43",
        "--extra_init_seed", "43022",
        "--train_rng_seed", "43023",
        "--batch_size", "256",
        "--micro_batch_size", "64",
        "--seq_length", "256",
        "--max_seq_length", "256",
        "--learning_rate", "0.001",
        "--warmup_fraction", "0.06",
        "--lr_total_steps", str(FULL_RUN_TOTAL_STEPS),
        "--weight_decay", "0.01",
        "--masking_curriculum", "wwm_fixed",
        "--mask_prob_start", "0.15",
        "--mask_prob_end", "0.15",
        "--checkpoint_words", "1000000",
        "--max_word_exposure", str(STOP_WORDS),
        "--num_workers", "0",
        "--log_every", "10",
        "--dynamics_trace_every", "200",
    ]


def base_prefix() -> list[dict[str, Any]]:
    logs = read_jsonl(BASE16_RUN / "training_log.jsonl")
    return logs[:COMPARE_STEPS]


def summarize_base_prefix() -> dict[str, Any]:
    prefix = base_prefix()
    last = prefix[-1]
    return {
        "steps": len(prefix),
        "last_step": last["step"],
        "last_cumulative_word_exposure": last["cumulative_word_exposure"],
        "last_lr": last["lr"],
        "last_loss": last["loss"],
        "mean_loss_first_10": statistics.mean(r["loss"] for r in prefix[:10]),
        "mean_loss_last_10": statistics.mean(r["loss"] for r in prefix[-10:]),
        "masked_tokens_first_step": prefix[0]["masked_tokens"],
        "batch_words_first_step": prefix[0]["batch_words"],
        "checksum_first_5": [(r["step"], r["batch_words"], r["masked_tokens"], round(r["lr"], 12), round(r["loss"], 6)) for r in prefix[:5]],
        "checksum_last_5": [(r["step"], r["batch_words"], r["masked_tokens"], round(r["lr"], 12), round(r["loss"], 6)) for r in prefix[-5:]],
    }


def preflight() -> dict[str, Any]:
    errors: list[str] = []
    for p, label in [
        (ACCUM_TRAINER, "accumulated trainer"),
        (TOKENIZER16_DIR / "tokenizer.json", "legal 16k tokenizer"),
        (TRAIN_100M, "100M stream"),
        (META, "overlay metadata"),
        (BASE16_RUN / "training_log.jsonl", "base-16k training log"),
        (BASE16_RUN / "scientific_metrics.json", "base-16k metrics"),
    ]:
        if not p.exists():
            errors.append(f"missing {label}: {p}")
    if (TOKENIZER16_DIR / "tokenizer.json").exists():
        tok_sha = sha256_file(TOKENIZER16_DIR / "tokenizer.json")
        if tok_sha != EXPECTED_TOK16_SHA:
            errors.append(f"tokenizer16 sha mismatch: {tok_sha} != {EXPECTED_TOK16_SHA}")
    if TRAIN_100M.exists():
        stream_sha = sha256_file(TRAIN_100M)
        if stream_sha != EXPECTED_100M_SHA:
            errors.append(f"100M stream sha mismatch: {stream_sha} != {EXPECTED_100M_SHA}")
    base_summary: dict[str, Any] | None = None
    if (BASE16_RUN / "training_log.jsonl").exists() and (BASE16_RUN / "scientific_metrics.json").exists():
        base_summary = summarize_base_prefix()
        metrics = read_json(BASE16_RUN / "scientific_metrics.json")
        if int(metrics.get("actual_training_steps", -1)) != FULL_RUN_TOTAL_STEPS:
            errors.append(f"base actual_training_steps mismatch: {metrics.get('actual_training_steps')} != {FULL_RUN_TOTAL_STEPS}")
        if base_summary["last_cumulative_word_exposure"] != STOP_WORDS:
            errors.append(f"base step{COMPARE_STEPS} exposure mismatch: {base_summary['last_cumulative_word_exposure']} != {STOP_WORDS}")
    payload = {
        "status": "PREFLIGHT_OK" if not errors else "PREFLIGHT_FAILED",
        "errors": errors,
        "purpose": "bounds the microbatching-only confound between research legal-16k base trainer and research legal-40k accumulated trainer",
        "run_dir": str(RUN_DIR),
        "tokenizer16_dir": str(TOKENIZER16_DIR),
        "tokenizer16_sha256": EXPECTED_TOK16_SHA,
        "train_100m": str(TRAIN_100M),
        "train_100m_sha256": EXPECTED_100M_SHA,
        "stop_words": STOP_WORDS,
        "compare_steps": COMPARE_STEPS,
        "lr_total_steps": FULL_RUN_TOTAL_STEPS,
        "base_prefix_summary": base_summary,
        "command": command(),
    }
    return payload


def run_train(force: bool) -> int:
    pf = preflight()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "accum16k_equivalence_preflight.json").write_text(json.dumps(pf, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in pf.items() if k != "command"}, indent=2, ensure_ascii=False), flush=True)
    if pf["status"] != "PREFLIGHT_OK":
        return 1
    if RUN_DIR.exists():
        if force:
            shutil.rmtree(RUN_DIR)
        else:
            print(f"Run dir exists; use --force to remove: {RUN_DIR}", file=sys.stderr)
            return 2
    env = os.environ.copy()
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    tcmd = command()
    result = subprocess.run(tcmd, text=True, capture_output=True, env=env)
    rec = {
        "status": "TRAIN_COMPLETE" if result.returncode == 0 else "TRAIN_FAILED",
        "returncode": result.returncode,
        "run_dir": str(RUN_DIR),
        "stdout_tail": result.stdout[-4000:],
        "stderr_tail": result.stderr[-4000:],
    }
    (OUT_DIR / "accum16k_equivalence_train_result.json").write_text(json.dumps(rec, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in rec.items() if k not in ("stdout_tail", "stderr_tail")}, indent=2), flush=True)
    if result.returncode != 0:
        print(result.stderr[-1200:], file=sys.stderr)
    return result.returncode


def compare() -> dict[str, Any]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if not (RUN_DIR / "training_log.jsonl").exists():
        payload = {
            "status": "WAITING_FOR_ACCUM16K_CONTROL",
            "missing": str(RUN_DIR / "training_log.jsonl"),
            "run_dir": str(RUN_DIR),
        }
        (OUT_DIR / "accum16k_equivalence_comparison.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        return payload
    base_logs = base_prefix()
    acc_logs = read_jsonl(RUN_DIR / "training_log.jsonl")
    n = min(len(base_logs), len(acc_logs), COMPARE_STEPS)
    rows = []
    for b, a in zip(base_logs[:n], acc_logs[:n]):
        rows.append({
            "step": b["step"],
            "base_words": b["batch_words"],
            "acc_words": a["batch_words"],
            "base_cum": b["cumulative_word_exposure"],
            "acc_cum": a["cumulative_word_exposure"],
            "base_masked": b["masked_tokens"],
            "acc_masked": a["masked_tokens"],
            "base_lr": b["lr"],
            "acc_lr": a["lr"],
            "base_loss": b["loss"],
            "acc_loss": a["loss"],
            "loss_delta_acc_minus_base": a["loss"] - b["loss"],
        })
    deltas = [r["loss_delta_acc_minus_base"] for r in rows]
    payload = {
        "status": "ACCUM16K_EQUIVALENCE_COMPARISON",
        "n_compared_steps": n,
        "same_batch_words_all": all(r["base_words"] == r["acc_words"] for r in rows),
        "same_cumulative_words_all": all(r["base_cum"] == r["acc_cum"] for r in rows),
        "same_masked_tokens_all": all(r["base_masked"] == r["acc_masked"] for r in rows),
        "same_lr_all_close_1e12": all(abs(r["base_lr"] - r["acc_lr"]) < 1e-12 for r in rows),
        "mean_loss_delta": statistics.mean(deltas) if deltas else None,
        "max_abs_loss_delta": max((abs(x) for x in deltas), default=None),
        "first_10_mean_delta": statistics.mean(deltas[:10]) if len(deltas) >= 10 else None,
        "last_10_mean_delta": statistics.mean(deltas[-10:]) if len(deltas) >= 10 else None,
        "final_step": rows[-1] if rows else None,
        "interpretation": (
            "Batch words/cumulative words/masked tokens/LR should match exactly if masking/data/schedule are identical; "
            "residual loss deltas bound the dropout/microbatch-forward effect. This is a short control, not endpoint evidence."
        ),
        "rows": rows,
    }
    (OUT_DIR / "accum16k_equivalence_comparison.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    # Compact CSV for quick inspection
    with (OUT_DIR / "accum16k_equivalence_loss_rows.csv").open("w", encoding="utf-8") as f:
        f.write("step,base_loss,acc_loss,delta,base_masked,acc_masked,base_lr,acc_lr\n")
        for r in rows:
            f.write(f"{r['step']},{r['base_loss']},{r['acc_loss']},{r['loss_delta_acc_minus_base']},{r['base_masked']},{r['acc_masked']},{r['base_lr']},{r['acc_lr']}\n")
    return payload


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["preflight", "train", "compare"], default="preflight")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if args.mode == "preflight":
        pf = preflight()
        (OUT_DIR / "accum16k_equivalence_preflight.json").write_text(json.dumps(pf, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(json.dumps({k: v for k, v in pf.items() if k != "command"}, indent=2, ensure_ascii=False), flush=True)
        sys.exit(0 if pf["status"] == "PREFLIGHT_OK" else 1)
    if args.mode == "train":
        sys.exit(run_train(force=args.force))
    if args.mode == "compare":
        out = compare()
        print(json.dumps({k: v for k, v in out.items() if k != "rows"}, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
