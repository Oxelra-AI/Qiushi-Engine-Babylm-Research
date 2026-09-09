#!/usr/bin/env python3
"""research guarded paired launcher for FW compact-view vs repaired whole-sentence
source-breadth comparison.

This supersedes the research paired launcher for the source-breadth arm. research
found that the research source-breadth stream sliced independent FineWeb sentences
across example boundaries. The corrected breadth arm below keeps the same exact
10M/100M word budgets and row word sequence, but packs whole independent
FineWeb sentences inside each row.

Expensive-work meaning:
  - compact_view: selected FineWeb sources with aligned compact rewrites.
  - source_breadth_wholesentence: same common selected FineWeb source-word budget,
    same row word lengths, same filler, same shared compliant 16k tokenizer, but
    companion words are coherent independent FineWeb source sentences.
The run decides whether aligned compact re-expression or additional coherent
source breadth is the stronger data mechanism under the legal 8x480 recipe.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import subprocess
import sys
import time
from typing import Any

STUDY = pathlib.Path("experiments/archive/representation_and_objectives")
WS = STUDY
AI_RUN_ROOT = WS / "training/runs"
OUT_DIR = WS / "data/fw_compact_vs_wholesentence_breadth_train"

TOKENIZER_DIR = WS / "data/shared_tokenizer/shared_16k_tokenizer"
EXPECTED_TOK_SHA = "e70d167f620668130f88744998bd1bacbe05f6d0735012144fe501b271b04366"
TRAINER = WS / "scripts/accumulated_masking_curriculum_trainer.py"

ARMS = {
    "compact_view": {
        "stream": WS / "data/fw_source_breadth_arm/fw_preserved_compact_view_100M.jsonl",
        "sha256": "c8d7f24b5edd2dad21f589c8cde72671d0b76038d9632fcd3d6c79178a24be68",
        "label": "fw_preserved_compact_view_shared16k",
        "run_dir": AI_RUN_ROOT / "fw_compact_view_shared16k_seed43022",
    },
    "source_breadth_wholesentence": {
        "stream": WS / "data/fw_source_breadth_wholesentence_arm/fw_preserved_source_breadth_wholesentence_100M.jsonl",
        "sha256": "1b98269fb210cc9494885ec47ee26d9fd1b6ead60a1c308f5d9c47d9b385731d",
        "label": "fw_preserved_source_breadth_wholesentence_shared16k",
        "run_dir": AI_RUN_ROOT / "fw_source_breadth_wholesentence_shared16k_seed43022",
    },
}

SEED = {"seed": 43, "extra_init_seed": 43022, "train_rng_seed": 43023}
RECIPE = dict(
    hidden_size=480, n_layer=8, n_head=8, ffn_mult=4,
    batch_size=256, micro_batch_size=64, seq_length=256, max_seq_length=256,
    learning_rate=0.001, warmup_fraction=0.06, weight_decay=0.01,
    masking_curriculum="wwm_fixed", mask_prob_start=0.15, mask_prob_end=0.15,
    checkpoint_words=1_000_000, max_word_exposure=100_000_000,
)


def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def count_words(path: pathlib.Path, cap_rows: int | None = None) -> tuple[int, int]:
    total_words = 0
    rows = 0
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            w = obj.get("words")
            if w is None:
                w = len(str(obj.get("text", "")).split())
            total_words += int(w)
            rows += 1
            if cap_rows is not None and rows >= cap_rows:
                break
    return total_words, rows


def preflight(full_word_check: bool) -> dict[str, Any]:
    errors: list[str] = []
    info: dict[str, Any] = {}
    tok_json = TOKENIZER_DIR / "tokenizer.json"
    if not tok_json.exists():
        errors.append(f"tokenizer.json not found at {tok_json}")
    else:
        tok_sha = sha256_file(tok_json)
        info["tokenizer_sha256"] = tok_sha
        if tok_sha != EXPECTED_TOK_SHA:
            errors.append(f"tokenizer SHA mismatch: {tok_sha} != {EXPECTED_TOK_SHA}")
    if not TRAINER.exists():
        errors.append(f"trainer not found: {TRAINER}")

    arm_info: dict[str, Any] = {}
    for name, a in ARMS.items():
        stream = a["stream"]
        d: dict[str, Any] = {"stream": str(stream), "exists": stream.exists()}
        if not stream.exists():
            errors.append(f"{name}: stream missing {stream}")
            arm_info[name] = d
            continue
        sha = sha256_file(stream)
        d["sha256"] = sha
        if sha != a["sha256"]:
            errors.append(f"{name}: stream SHA mismatch {sha} != {a['sha256']}")
        if full_word_check:
            words, rows = count_words(stream)
            d["words"] = words
            d["rows"] = rows
            if words != RECIPE["max_word_exposure"]:
                errors.append(f"{name}: word count {words} != {RECIPE['max_word_exposure']}")
        arm_info[name] = d

    rd = {name: str(a["run_dir"]) for name, a in ARMS.items()}
    if len(set(rd.values())) != len(rd):
        errors.append(f"run_dir collision: {rd}")
    info["run_dirs"] = rd

    if not errors:
        try:
            from transformers import AutoTokenizer
            tok = AutoTokenizer.from_pretrained(str(TOKENIZER_DIR), use_fast=True)
            assert tok.is_fast, "not a fast tokenizer"
            assert tok.vocab_size == 16384, f"expected 16384 vocab, got {tok.vocab_size}"
            enc = tok("The quick brown fox jumped over the lazy dog", add_special_tokens=False)
            assert enc.word_ids() is not None, "word_ids() failed"
            info["tokenizer_functional"] = {"vocab_size": tok.vocab_size, "is_fast": True}
        except Exception as e:
            errors.append(f"tokenizer functional check failed: {e}")

    return {
        "status": "PREFLIGHT_OK" if not errors else "PREFLIGHT_FAILED",
        "errors": errors,
        "arms": arm_info,
        "info": info,
        "recipe": RECIPE,
        "seed": SEED,
        "supersedes": "research source_breadth stream; use source_breadth_wholesentence for repaired breadth comparison.",
    }


def train_command(name: str) -> tuple[list[str], pathlib.Path]:
    a = ARMS[name]
    run_dir = a["run_dir"]
    cmd = [
        sys.executable, "-B", str(TRAINER),
        "--example_jsonl", str(a["stream"]),
        "--example_jsonl_label", a["label"],
        "--output_dir", str(run_dir),
        "--tokenizer_path", str(TOKENIZER_DIR),
        "--tokenizer_label", "shared_16k",
        "--hidden_size", str(RECIPE["hidden_size"]),
        "--n_layer", str(RECIPE["n_layer"]),
        "--n_head", str(RECIPE["n_head"]),
        "--ffn_mult", str(RECIPE["ffn_mult"]),
        "--seed", str(SEED["seed"]),
        "--extra_init_seed", str(SEED["extra_init_seed"]),
        "--train_rng_seed", str(SEED["train_rng_seed"]),
        "--batch_size", str(RECIPE["batch_size"]),
        "--micro_batch_size", str(RECIPE["micro_batch_size"]),
        "--seq_length", str(RECIPE["seq_length"]),
        "--max_seq_length", str(RECIPE["max_seq_length"]),
        "--learning_rate", str(RECIPE["learning_rate"]),
        "--warmup_fraction", str(RECIPE["warmup_fraction"]),
        "--weight_decay", str(RECIPE["weight_decay"]),
        "--masking_curriculum", RECIPE["masking_curriculum"],
        "--mask_prob_start", str(RECIPE["mask_prob_start"]),
        "--mask_prob_end", str(RECIPE["mask_prob_end"]),
        "--checkpoint_words", str(RECIPE["checkpoint_words"]),
        "--max_word_exposure", str(RECIPE["max_word_exposure"]),
        "--num_workers", "0",
        "--log_every", "50",
        "--dynamics_trace_every", "200",
    ]
    return cmd, run_dir


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=list(ARMS.keys()), default=None)
    ap.add_argument("--gpu", type=int, default=None)
    ap.add_argument("--launch", action="store_true")
    ap.add_argument("--full-word-check", action="store_true")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pf = preflight(full_word_check=args.full_word_check)
    pf_path = OUT_DIR / "paired_preflight.json"
    pf_path.write_text(json.dumps(pf, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({k: pf[k] for k in ("status", "errors")}, indent=2), flush=True)
    print(f"Preflight saved: {pf_path}", flush=True)
    if pf["status"] != "PREFLIGHT_OK":
        sys.exit(1)

    manifest = {"status": "PAIRED_COMMANDS", "arms": {}}
    for name in ARMS:
        cmd, run_dir = train_command(name)
        manifest["arms"][name] = {"run_dir": str(run_dir), "cmd": cmd}
    (OUT_DIR / "paired_commands.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    if not args.launch:
        print("\n[DRY RUN] Repaired compact-vs-whole-sentence-breadth preflight passed. Commands ready; NOT launched.", flush=True)
        return

    if args.arm is None or args.gpu is None:
        print("ERROR: --launch requires both --arm and --gpu", flush=True)
        sys.exit(2)

    cmd, run_dir = train_command(args.arm)
    run_dir.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    print(f"Launching repaired FW comparison arm={args.arm} on GPU{args.gpu}; run_dir={run_dir}", flush=True)
    t0 = time.time()
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
    elapsed = time.time() - t0
    result = {
        "status": "TRAINING_COMPLETE" if proc.returncode == 0 else "TRAINING_FAILED",
        "arm": args.arm,
        "gpu": args.gpu,
        "run_dir": str(run_dir),
        "returncode": proc.returncode,
        "elapsed_sec": round(elapsed, 1),
        "stdout_tail": proc.stdout[-4000:] if proc.stdout else "",
        "stderr_tail": proc.stderr[-4000:] if proc.stderr else "",
    }
    (OUT_DIR / f"train_result_{args.arm}.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k not in ("stdout_tail", "stderr_tail")}, indent=2), flush=True)
    if proc.returncode != 0:
        print(f"STDERR tail: {proc.stderr[-800:]}", flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
