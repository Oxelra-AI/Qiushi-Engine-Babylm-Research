#!/usr/bin/env python3
"""research: train MAX-geometry DeBERTa clean controls.

Scientific role
---------------
The research/273 readouts show that the dose-amplified Entity view-minus-repeat
carrier is largely a zero-op/nonzero-op allocation effect under the Entity test
mixture, not clean evidence for source-view correspondence. The strongest
remaining broad quantity is therefore view-minus-clean (V-C), especially ex-Entity
V-C. But the current V-C uses an older 1x-geometry clean arm, while MAX view,
repeat, and breadth use a 653,130-row / 2,552-update MAX stream. This launcher
trains a clean-Qwen control on the already materialized MAX length-matched clean
stream so V-C can be read without the row/update geometry confound.

Two basins are supported:
  seed43022: matches first-basin MAX view/repeat/breadth seeds 43/43022/43023.
  seed43122: matches second-basin MAX view/repeat seeds 43/43122/43123.

No evaluation, GlobalPIQA, SuperGLUE, AoA, upload, or leaderboard action is
performed here.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse
import hashlib
import json
import os
import pathlib
import subprocess
import sys
import time
from typing import Any

from transformers import AutoTokenizer, DebertaV2Config, DebertaV2ForMaskedLM


def find_user_root() -> pathlib.Path:
    return _PUBLIC_ROOT


ROOT = find_user_root()
WS = ROOT / "experiments/archive/frontier_consolidation"
TRAINER = ROOT / "experiments/archive/compact_experience/scripts/masking_curriculum_trainer.py"
TOKENIZER = WS / "data/compliant_tokenizer"
POOL_DIR = WS / "data/dose_2p64x_rowholdout_pools"
META_PATH = POOL_DIR / "dose2p64x_rowholdout_metadata.json"
MAT_PATH = POOL_DIR / "cleanqwen_lengthmatched_dose2p64x_100M.materialization.json"
STREAM = POOL_DIR / "cleanqwen_lengthmatched_dose2p64x_100M.jsonl"
RUNS_DIR = WS / "training/runs"
EXPECTED_TOKENIZER_SHA = "91b775514b9f4e2d1f37c28445ab98181007e16d14f763955168547e293ee8f9"
EXPECTED_STREAM_SHA = "64e686d16e0d7d5e81acecc73494d8670a1d6f8ddaffb4ce517277c983a9bed8"
TOTAL_WORDS = 100_000_000
CKS = [f"chck_{i}M" for i in range(10, 101, 10)]

BASE_RECIPE = {
    "parameter_count_expected": 34_467_424,
    "vocab_size_expected": 16_384,
    "hidden_size": 480,
    "n_layer": 8,
    "n_head": 8,
    "ffn_mult": 4,
    "seed": 43,
    "batch_size": 256,
    "seq_length": 256,
    "max_seq_length": 256,
    "learning_rate": 0.001,
    "weight_decay": 0.01,
    "warmup_fraction": 0.06,
    "lr_total_steps": 2529,
    "masking_curriculum": "wwm_fixed",
    "mask_prob_start": 0.15,
    "mask_prob_end": 0.15,
    "checkpoint_words": 10_000_000,
    "max_word_exposure": 100_000_000,
    "num_workers": 0,
    "log_every": 50,
    "dynamics_trace_every": 200,
    "deberta_pos_att_type": "p2c,c2p",
}
SEEDS = {
    "seed43022": {"extra_init_seed": 43022, "train_rng_seed": 43023},
    "seed43122": {"extra_init_seed": 43122, "train_rng_seed": 43123},
}


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def rel(p: pathlib.Path) -> str:
    try:
        return str(p.relative_to(ROOT))
    except Exception:
        return str(p)


def read_json(path: pathlib.Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: pathlib.Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def count_jsonl(path: pathlib.Path) -> dict[str, Any]:
    rows = 0
    words = 0
    by_source: dict[str, int] = {}
    first_rows: list[dict[str, Any]] = []
    last_rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            text = str(obj.get("text") or "")
            w_field = int(obj.get("words", len(text.split())))
            w_actual = len(text.split())
            if w_field != w_actual:
                raise RuntimeError(f"word mismatch at row {rows}: field={w_field} actual={w_actual}")
            rows += 1
            words += w_field
            src = str(obj.get("source") or "")
            by_source[src] = by_source.get(src, 0) + w_field
            rec = {"row": rows - 1, "words": w_field, "source": src, "example_id": obj.get("example_id")}
            if len(first_rows) < 3:
                first_rows.append(rec)
            last_rows.append(rec)
            if len(last_rows) > 3:
                last_rows.pop(0)
    return {
        "rows": rows,
        "words": words,
        "exact_100M": words == TOTAL_WORDS,
        "expected_updates_at_batch256": (rows + BASE_RECIPE["batch_size"] - 1) // BASE_RECIPE["batch_size"],
        "first_rows": first_rows,
        "last_rows": last_rows,
        "source_word_counts": by_source,
    }


def model_variant() -> dict[str, Any]:
    tok = AutoTokenizer.from_pretrained(str(TOKENIZER), use_fast=True)
    cfg = DebertaV2Config(
        vocab_size=len(tok),
        hidden_size=BASE_RECIPE["hidden_size"],
        num_hidden_layers=BASE_RECIPE["n_layer"],
        num_attention_heads=BASE_RECIPE["n_head"],
        intermediate_size=BASE_RECIPE["hidden_size"] * BASE_RECIPE["ffn_mult"],
        max_position_embeddings=512,
        max_relative_positions=256,
        position_buckets=256,
        relative_attention=True,
        pos_att_type=["p2c", "c2p"],
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        pad_token_id=tok.pad_token_id,
        bos_token_id=tok.bos_token_id,
        eos_token_id=tok.eos_token_id,
    )
    m = DebertaV2ForMaskedLM(cfg)
    keys = list(m.state_dict().keys())
    return {
        "parameter_count": sum(p.numel() for p in m.parameters()),
        "vocab_size": len(tok),
        "has_pos_key_proj": any("pos_key_proj" in k for k in keys),
        "has_pos_query_proj": any("pos_query_proj" in k for k in keys),
        "has_encoder_rel_embeddings": any(k.endswith("encoder.rel_embeddings.weight") for k in keys),
    }


def recipe_for(seed_label: str) -> dict[str, Any]:
    r = dict(BASE_RECIPE)
    r.update(SEEDS[seed_label])
    return r


def default_run_dir(seed_label: str) -> pathlib.Path:
    return RUNS_DIR / f"full_p2c_c2p_abs_clean_dose2p64x_matched_rowholdout_deberta100M_{seed_label}"


def build_command(run_dir: pathlib.Path, seed_label: str) -> list[str]:
    r = recipe_for(seed_label)
    return [
        sys.executable, "-B", str(TRAINER),
        "--example_jsonl", str(STREAM),
        "--example_jsonl_label", f"cleanqwen_lengthmatched_dose2p64x_matched_rowholdout_{seed_label}",
        "--output_dir", str(run_dir),
        "--tokenizer_path", str(TOKENIZER),
        "--tokenizer_label", "compliant16k_reinvest10M",
        "--hidden_size", str(r["hidden_size"]),
        "--n_layer", str(r["n_layer"]),
        "--n_head", str(r["n_head"]),
        "--ffn_mult", str(r["ffn_mult"]),
        "--deberta_pos_att_type", str(r["deberta_pos_att_type"]),
        "--seed", str(r["seed"]),
        "--extra_init_seed", str(r["extra_init_seed"]),
        "--train_rng_seed", str(r["train_rng_seed"]),
        "--batch_size", str(r["batch_size"]),
        "--seq_length", str(r["seq_length"]),
        "--max_seq_length", str(r["max_seq_length"]),
        "--learning_rate", str(r["learning_rate"]),
        "--warmup_fraction", str(r["warmup_fraction"]),
        "--weight_decay", str(r["weight_decay"]),
        "--lr_total_steps", str(r["lr_total_steps"]),
        "--masking_curriculum", str(r["masking_curriculum"]),
        "--mask_prob_start", str(r["mask_prob_start"]),
        "--mask_prob_end", str(r["mask_prob_end"]),
        "--checkpoint_words", str(r["checkpoint_words"]),
        "--max_word_exposure", str(r["max_word_exposure"]),
        "--num_workers", str(r["num_workers"]),
        "--log_every", str(r["log_every"]),
        "--dynamics_trace_every", str(r["dynamics_trace_every"]),
    ]


def checkpoint_ok(run_dir: pathlib.Path, ck: str) -> bool:
    d = run_dir / "hf_model" / ck
    return (d / "model.safetensors").exists() or (d / "pytorch_model.bin").exists()


def summarize_metrics(run_dir: pathlib.Path) -> dict[str, Any]:
    p = run_dir / "scientific_metrics.json"
    if not p.exists():
        return {"metrics_exists": False, "ready": False}
    m = read_json(p)
    saved = m.get("saved_checkpoints") or []
    out = {
        "metrics_exists": True,
        "status": m.get("status"),
        "word_exposure": m.get("word_exposure"),
        "actual_training_steps": m.get("actual_training_steps"),
        "loss_first": m.get("loss_first"),
        "loss_last": m.get("loss_last"),
        "parameter_count": m.get("parameter_count"),
        "vocab_size": m.get("vocab_size"),
        "tokenizer_label": m.get("tokenizer_label"),
        "saved_checkpoints": len(saved),
        "first_checkpoint": saved[0].get("name") if saved else None,
        "last_checkpoint": saved[-1].get("name") if saved else None,
        "all_expected_checkpoints_present": all(checkpoint_ok(run_dir, ck) for ck in CKS),
    }
    out["ready"] = (
        int(out.get("word_exposure") or 0) >= TOTAL_WORDS
        and out.get("parameter_count") == BASE_RECIPE["parameter_count_expected"]
        and out.get("vocab_size") == BASE_RECIPE["vocab_size_expected"]
        and out.get("tokenizer_label") == "compliant16k_reinvest10M"
        and out.get("all_expected_checkpoints_present") is True
    )
    return out


def preflight(seed_label: str, run_dir: pathlib.Path, count_words: bool) -> dict[str, Any]:
    if seed_label not in SEEDS:
        raise ValueError(f"unknown seed_label {seed_label}; choices {sorted(SEEDS)}")
    for p in [TRAINER, TOKENIZER, META_PATH, MAT_PATH, STREAM]:
        if not p.exists():
            raise FileNotFoundError(p)
    meta = read_json(META_PATH)
    mat = read_json(MAT_PATH)
    audit = meta.get("audit", {})
    if meta.get("status") != "MATCHED_MAX_ROWHOLDOUT_POOLS_MATERIALIZED":
        raise RuntimeError(f"unexpected metadata status {meta.get('status')}")
    if not (audit.get("all_exact_10M") and audit.get("row_length_sequence_identical_all_arms")):
        raise RuntimeError(f"MAX pool audit not clean: {audit}")
    if pathlib.Path(mat.get("output_100m", "")).name != STREAM.name:
        raise RuntimeError(f"materialization output mismatch: {mat}")
    tok_sha = sha256_file(TOKENIZER / "tokenizer.json")
    if tok_sha != EXPECTED_TOKENIZER_SHA:
        raise RuntimeError(f"tokenizer SHA mismatch {tok_sha}")
    stream_sha = sha256_file(STREAM)
    if stream_sha != EXPECTED_STREAM_SHA or mat.get("output_100m_sha256") != EXPECTED_STREAM_SHA:
        raise RuntimeError(f"clean 100M stream SHA mismatch file={stream_sha} materialization={mat.get('output_100m_sha256')}")
    counts = count_jsonl(STREAM) if count_words else {"count_skipped": True, "exact_100M": True, "rows": 653130, "words": TOTAL_WORDS, "expected_updates_at_batch256": 2552}
    if not counts.get("exact_100M") or int(counts.get("expected_updates_at_batch256") or 0) != 2552:
        raise RuntimeError(f"clean stream count/update mismatch: {counts}")
    mv = model_variant()
    if mv["parameter_count"] != BASE_RECIPE["parameter_count_expected"] or mv["vocab_size"] != BASE_RECIPE["vocab_size_expected"]:
        raise RuntimeError(f"model coordinate mismatch: {mv}")
    cmd = build_command(run_dir, seed_label)
    role = (
        "This H100 run decides whether the surviving broad MAX view-minus-clean signal, especially ex-Entity V-C, remains after clean uses the same MAX row-length sequence, 653130 rows, 2552 optimizer updates, tokenizer, architecture, and basin seeds as the MAX intervention arms. If it collapses, the previous V-C was a row/update geometry artifact; if it survives, the fixed-budget allocation result becomes the main scientific object."
    )
    lowest = (
        "All cheaper work is already exhausted: the MAX clean stream is materialized and audited, but no trained DeBERTa model exists under this geometry. One 100M run with checkpoints is the minimum reliable intervention because the target quantity is downstream competence after the same training schedule, not a static corpus statistic or MLM loss proxy."
    )
    return {
        "status": "MAXGEOM_CLEAN_PREFLIGHT_OK",
        "created_utc": now(),
        "seed_label": seed_label,
        "run_dir": rel(run_dir),
        "stream": rel(STREAM),
        "stream_sha256": stream_sha,
        "stream_accounting": counts,
        "materialization": mat,
        "pool_metadata": rel(META_PATH),
        "tokenizer_dir": rel(TOKENIZER),
        "tokenizer_json_sha256": tok_sha,
        "model_variant": mv,
        "recipe": recipe_for(seed_label),
        "command": [sys.executable if x == sys.executable else str(x) for x in cmd],
        "expensive_work_role": role,
        "lowest_reliable_method": lowest,
        "comparison_targets": {
            "seed43022": "compare to first-basin MAX view/repeat/breadth with same seeds and geometry",
            "seed43122": "compare to second-basin MAX view/repeat with same seeds and geometry",
        },
        "no_evaluation_globalpiqa_superglue_aoa_upload_or_leaderboard": True,
        "mechanism_instrument_not_leaderboard_submission": True,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed-label", choices=sorted(SEEDS), required=True)
    ap.add_argument("--gpu", type=int, required=True)
    ap.add_argument("--run-dir", default="")
    ap.add_argument("--count-words", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--allow-nonempty", action="store_true")
    args = ap.parse_args()

    run_dir = pathlib.Path(args.run_dir) if args.run_dir else default_run_dir(args.seed_label)
    if not run_dir.is_absolute():
        run_dir = ROOT / run_dir
    info = preflight(args.seed_label, run_dir, args.count_words)
    preflight_dir = WS / "data/maxgeom_clean_preflight"
    preflight_dir.mkdir(parents=True, exist_ok=True)
    write_json(preflight_dir / f"{args.seed_label}_preflight.json", info)
    if args.dry_run:
        print(json.dumps({
            "status": info["status"],
            "seed_label": args.seed_label,
            "run_dir": info["run_dir"],
            "rows": info["stream_accounting"].get("rows"),
            "words": info["stream_accounting"].get("words"),
            "expected_updates": info["stream_accounting"].get("expected_updates_at_batch256"),
            "parameter_count": info["model_variant"]["parameter_count"],
            "tokenizer_sha_prefix": info["tokenizer_json_sha256"][:12],
            "stream_sha_prefix": info["stream_sha256"][:12],
            "dryrun_json": rel(preflight_dir / f"{args.seed_label}_preflight.json"),
            "no_training_started": True,
        }, indent=2, ensure_ascii=False), flush=True)
        return

    existing = summarize_metrics(run_dir)
    if existing.get("ready"):
        result = {"status": "MAXGEOM_CLEAN_ALREADY_FINISHED", "seed_label": args.seed_label, "run_dir": rel(run_dir), "metrics": existing}
        print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)
        return
    if run_dir.exists() and any(run_dir.iterdir()) and not args.allow_nonempty:
        raise SystemExit(f"run_dir exists and is non-empty: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    write_json(run_dir / "maxgeom_clean_preflight.json", info)

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    env.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    env["TMPDIR"] = f"/tmp/q_frontier_consolidation_step273_maxgeom_clean_{args.seed_label}_{int(time.time())}"
    pathlib.Path(env["TMPDIR"]).mkdir(parents=True, exist_ok=True)

    stdout_path = run_dir / "train_stdout.log"
    stderr_path = run_dir / "train_stderr.log"
    started = now()
    start_rec = {"status": "MAXGEOM_CLEAN_TRAIN_STARTING", "seed_label": args.seed_label, "gpu": args.gpu, "run_dir": rel(run_dir), "created_utc": started, "expensive_work_role": info["expensive_work_role"], "lowest_reliable_method": info["lowest_reliable_method"], "no_evaluation_globalpiqa_superglue_aoa_upload_or_leaderboard": True}
    print(json.dumps(start_rec, indent=2, ensure_ascii=False), flush=True)
    with stdout_path.open("w", encoding="utf-8") as out, stderr_path.open("w", encoding="utf-8") as err:
        out.write(json.dumps({"event": "maxgeom_clean_launcher_start", "utc": started, "seed_label": args.seed_label, "gpu": args.gpu, "recipe": recipe_for(args.seed_label)}, ensure_ascii=False) + "\n")
        out.flush()
        proc = subprocess.run(build_command(run_dir, args.seed_label), cwd=str(ROOT), env=env, stdout=out, stderr=err, text=True)
    result = {
        "status": "MAXGEOM_CLEAN_TRAIN_FINISHED" if proc.returncode == 0 else "MAXGEOM_CLEAN_TRAIN_FAILED",
        "returncode": proc.returncode,
        "seed_label": args.seed_label,
        "gpu": args.gpu,
        "started_utc": started,
        "finished_utc": now(),
        "run_dir": rel(run_dir),
        "stdout_log": rel(stdout_path),
        "stderr_log": rel(stderr_path),
        "metrics": summarize_metrics(run_dir),
        "no_evaluation_globalpiqa_superglue_aoa_upload_or_leaderboard": True,
        "mechanism_instrument_not_leaderboard_submission": True,
    }
    if proc.returncode != 0 and stderr_path.exists():
        result["stderr_tail"] = stderr_path.read_text(encoding="utf-8", errors="replace")[-4000:]
    write_json(run_dir / "maxgeom_clean_launcher_result.json", result)
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)
    raise SystemExit(proc.returncode)


if __name__ == "__main__":
    main()
