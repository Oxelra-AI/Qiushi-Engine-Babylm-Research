#!/usr/bin/env python3
"""research: launch nested restatement-dose training runs.

This is the dose-specific version of the verified research/research launcher.  It
uses the same compact-view-reinvest SOTA recipe:

  * DeBERTa-v2 8x480 MLM with adapter bottleneck 128 and adapter scale 1.75;
  * compliant16k_reinvest10M tokenizer;
  * whole-word fixed masking at 0.15;
  * batch size 256, LR 0.001, warmup_fraction 0.06, weight_decay 0.01;
  * 100M word exposure and 1M-word checkpoint cadence;
  * seed convention seed=43, extra_init_seed={43022,43122}, train_rng_seed=+1.

The scientific difference from research is only the training stream: the stream
preserves the compact-view-reinvest substrate and replaces ordinary filler rows
with additional aligned source->rewrite pairs at a specified nested dose.  The
launcher metadata therefore records dose accounting rather than state-update
metadata.
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

ROOT = pathlib.Path.cwd()
STUDY = ROOT / "experiments/archive/relation_learning"
TRAINER = ROOT / "experiments/archive/frontier_consolidation/scripts/adapter_scaled_trainer.py"
TOKENIZER = ROOT / "experiments/archive/frontier_consolidation/data/compliant_tokenizer"
RUN_BASE = STUDY / "training/runs"
STREAMS = {
    "dose21": STUDY / "data/nested_restatement_dose_streams/dose21/dose21_compact_view_reinvest_100M.jsonl",
    "dose25": STUDY / "data/nested_restatement_dose_streams/dose25/dose25_compact_view_reinvest_100M.jsonl",
}
META = STUDY / "data/nested_restatement_dose_streams/nested_dose_materialization_metadata.json"

SEED_MAP = {
    43022: {"seed": 43, "extra_init_seed": 43022, "train_rng_seed": 43023},
    43122: {"seed": 43, "extra_init_seed": 43122, "train_rng_seed": 43123},
}


def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def count_stream(path: pathlib.Path) -> tuple[int, int]:
    n = 0
    words = 0
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            n += 1
            words += int(r.get("words", 0))
    return n, words


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dose", choices=sorted(STREAMS), required=True)
    ap.add_argument("--seed", type=int, choices=sorted(SEED_MAP), required=True)
    ap.add_argument("--gpu", type=int, choices=[0, 1], required=True)
    ap.add_argument("--stream", default="", help="Override stream path")
    ap.add_argument("--stream-meta", default=str(META))
    ap.add_argument("--validate-only", action="store_true")
    ap.add_argument("--run-label", default="")
    args = ap.parse_args()

    dose = args.dose
    stream_path = pathlib.Path(args.stream) if args.stream else STREAMS[dose]
    meta_path = pathlib.Path(args.stream_meta) if args.stream_meta else META
    seeds = SEED_MAP[args.seed]

    errors = []
    for p, desc in [(stream_path, "stream"), (TRAINER, "trainer"), (TOKENIZER, "tokenizer"), (meta_path, "stream metadata")]:
        if not p.exists():
            errors.append(f"Missing {desc}: {p}")
    if errors:
        for e in errors:
            print("ERROR:", e, flush=True)
        raise SystemExit(1)

    n_rows, total_words = count_stream(stream_path)
    if total_words != 100_000_000:
        errors.append(f"Stream word count {total_words} != 100000000")
    if n_rows != 647_400:
        errors.append(f"Stream row count {n_rows} != 647400")

    stream_meta = json.loads(meta_path.read_text(encoding="utf-8"))
    mat = stream_meta.get(f"{dose}_materialization", {})
    expected_stream = mat.get("stream_path")
    if expected_stream:
        try:
            expected_resolved = pathlib.Path(expected_stream).resolve()
            launch_resolved = stream_path.resolve()
        except Exception:
            expected_resolved = pathlib.Path(expected_stream)
            launch_resolved = stream_path
        if expected_resolved != launch_resolved:
            errors.append(f"Metadata stream path {expected_stream} does not match launch stream {stream_path}")
    stream_sha = sha256_file(stream_path)
    expected_sha = mat.get("stream_sha256")
    if expected_sha and stream_sha != expected_sha:
        errors.append(f"Stream SHA {stream_sha} != metadata {expected_sha}")
    if int(mat.get("pair_segment_rows_over_256_total", -1)) != 0:
        errors.append(f"Metadata says pair segments over 256: {mat.get('pair_segment_rows_over_256_total')}")
    if errors:
        for e in errors:
            print("ERROR:", e, flush=True)
        raise SystemExit(1)

    label = args.run_label or f"restatement_{dose}_adapter128_scale1p75_seed{args.seed}"
    run_dir = RUN_BASE / label
    if run_dir.exists() and not args.validate_only:
        existing = [p for p in run_dir.iterdir()]
        if existing:
            print(f"ERROR: run_dir already exists and is not empty: {run_dir}", flush=True)
            raise SystemExit(1)

    cmd = [
        sys.executable, "-B", str(TRAINER),
        "--adapter_bottleneck", "128",
        "--adapter_enabled", "1",
        "--adapter_scale", "1.75",
        "--gpu", str(args.gpu),
        "--example_jsonl", str(stream_path),
        "--example_jsonl_label", label,
        "--example_jsonl_meta", str(meta_path),
        "--output_dir", str(run_dir),
        "--tokenizer_path", str(TOKENIZER),
        "--tokenizer_label", "compliant16k_reinvest10M",
        "--hidden_size", "480",
        "--n_layer", "8",
        "--n_head", "8",
        "--ffn_mult", "4",
        "--seed", str(seeds["seed"]),
        "--extra_init_seed", str(seeds["extra_init_seed"]),
        "--train_rng_seed", str(seeds["train_rng_seed"]),
        "--batch_size", "256",
        "--seq_length", "256",
        "--max_seq_length", "256",
        "--learning_rate", "0.001",
        "--warmup_fraction", "0.06",
        "--weight_decay", "0.01",
        "--masking_curriculum", "wwm_fixed",
        "--mask_prob_start", "0.15",
        "--mask_prob_end", "0.15",
        "--checkpoint_words", "1000000",
        "--max_word_exposure", "100000000",
        "--num_workers", "0",
        "--log_every", "50",
        "--dynamics_trace_every", "200",
    ]

    launch_meta = {
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "cmd": cmd,
        "run_label": label,
        "run_dir": str(run_dir),
        "dose": dose,
        "seed": args.seed,
        "seeds": seeds,
        "gpu": args.gpu,
        "stream": str(stream_path),
        "stream_rows": n_rows,
        "stream_words": total_words,
        "stream_sha256": stream_sha,
        "stream_meta": str(meta_path),
        "stream_meta_sha256": sha256_file(meta_path),
        "dose_materialization_summary": {
            "added_pair_words_per_pass": mat.get("added_pair_words_per_pass"),
            "replacement_rows_per_pass": mat.get("replacement_rows_per_pass"),
            "pair_segment_rows_over_256_total": mat.get("pair_segment_rows_over_256_total"),
            "full_rows_over_256_total": mat.get("full_rows_over_256_total"),
            "output_source_words": mat.get("output_source_words"),
        },
        "tokenizer": str(TOKENIZER),
        "trainer": str(TRAINER),
        "adapter_bottleneck": 128,
        "adapter_scale": 1.75,
        "intervention": "nested_aligned_restatement_dose",
        "description": (
            "Nested aligned-restatement dose on the compact-view-reinvest substrate. "
            "Inherited COMPACT_EXPERIENCE ALN and compact-view-reinvest rows are preserved; additional "
            "validated aligned source->rewrite pairs replace ordinary filler at fixed word "
            "budget. The experiment estimates the practiced-class readout return and ordinary "
            "language-fit price of extra relation practice under standard WWM."
        ),
    }

    if args.validate_only:
        print(json.dumps({"status": "VALIDATE_ONLY", **launch_meta}, indent=2), flush=True)
        print("Command:", flush=True)
        print(" ".join(cmd), flush=True)
        return

    if run_dir.exists():
        existing = [p for p in run_dir.iterdir()]
        if existing:
            print(f"ERROR: run_dir already exists and is not empty: {run_dir}", flush=True)
            raise SystemExit(1)
    else:
        run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "launch_command.json").write_text(json.dumps(launch_meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(json.dumps({
        "status": "launching",
        "run_label": label,
        "run_dir": str(run_dir),
        "dose": dose,
        "seed": args.seed,
        "gpu": args.gpu,
        "stream_rows": n_rows,
        "stream_words": total_words,
        "stream_sha256": stream_sha,
        "pair_segment_rows_over_256_total": mat.get("pair_segment_rows_over_256_total"),
        "full_rows_over_256_total": mat.get("full_rows_over_256_total"),
    }, indent=2), flush=True)

    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    env["TOKENIZERS_PARALLELISM"] = "false"
    proc = subprocess.run(cmd, env=env, capture_output=False)
    result = {
        "run_label": label,
        "run_dir": str(run_dir),
        "exit_code": proc.returncode,
        "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    print(json.dumps(result, indent=2), flush=True)
    raise SystemExit(proc.returncode)


if __name__ == "__main__":
    main()
