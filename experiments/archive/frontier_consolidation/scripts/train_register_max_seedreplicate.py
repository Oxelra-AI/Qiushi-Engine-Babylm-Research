#!/usr/bin/env python3
"""research: replicate the MAX-register reversal in an independent DeBERTa basin.

The fixed streams were materialized in research.  Both arms admit the identical
MAX FineWeb compact-view block and differ only in which clean register was
removed from the fixed 10M-word budget.  research found a negative
childspeech_removed - adultprose_removed contrast at seed43022, rejecting the
pre-score profile and rate signs.  This launcher repeats that exact data pair
with a new initialization/training-randomness basin (default 43/43122/43123).

No official evaluation, GlobalPIQA, SuperGLUE, AoA, upload, or leaderboard action
is performed here.  Use --dry-run --count-words before the H100 launch.
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


def find_user_root() -> pathlib.Path:
    return _PUBLIC_ROOT


ROOT = find_user_root()
WS = ROOT / "experiments/archive/frontier_consolidation"
TRAINER = ROOT / "experiments/archive/compact_experience/scripts/masking_curriculum_trainer.py"
TOKENIZER = WS / "data/compliant_tokenizer"
POOL_DIR = WS / "data/register_max_rowholdout_pools"
META_PATH = POOL_DIR / "register_max_rowholdout_metadata.json"
RUNS_DIR = WS / "training/runs"
EXPECTED_TOKENIZER_SHA = "91b775514b9f4e2d1f37c28445ab98181007e16d14f763955168547e293ee8f9"
TOTAL_WORDS = 100_000_000
ARM_STREAMS = {
    "childspeech": "regmax_childspeech_samefw_100M.jsonl",
    "adultprose": "regmax_adultprose_samefw_100M.jsonl",
}
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
    "max_word_exposure": TOTAL_WORDS,
    "num_workers": 0,
    "log_every": 50,
    "dynamics_trace_every": 200,
    "deberta_pos_att_type": "p2c,c2p",
}


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def rel(p: pathlib.Path) -> str:
    try:
        return str(p.relative_to(ROOT))
    except Exception:
        return str(p)


def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path: pathlib.Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: pathlib.Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def run_name(arm: str, extra_init_seed: int) -> str:
    return f"regmax_{arm}_samefw_deberta100M_seed{extra_init_seed}"


def verify_tokenizer() -> str:
    sha = sha256_file(TOKENIZER / "tokenizer.json")
    if sha != EXPECTED_TOKENIZER_SHA:
        raise RuntimeError(f"Tokenizer SHA mismatch: {sha}")
    return sha


def count_stream(path: pathlib.Path) -> tuple[int, int]:
    rows = words = 0
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            words += int(r.get("words", len(str(r.get("text", "")).split())))
            rows += 1
    return rows, words


def checkpoint_names_from_metrics(m: dict[str, Any]) -> list[str]:
    vals = m.get("saved_checkpoints") or []
    if isinstance(vals, list):
        names = []
        for x in vals:
            if isinstance(x, dict):
                names.append(str(x.get("name")))
            else:
                names.append(str(x))
        return names
    return []


def summarize_metrics(run_dir: pathlib.Path) -> dict[str, Any]:
    mf = run_dir / "scientific_metrics.json"
    if not mf.exists():
        return {"metrics_exists": False, "ready": False}
    try:
        m = read_json(mf)
    except Exception as exc:
        return {"metrics_exists": True, "ready": False, "error": repr(exc)}
    ck_names = checkpoint_names_from_metrics(m)
    needed = {f"chck_{i}M" for i in range(10, 101, 10)}
    return {
        "metrics_exists": True,
        "status": m.get("status"),
        "word_exposure": m.get("word_exposure"),
        "actual_training_steps": m.get("actual_training_steps"),
        "loss_first": m.get("loss_first"),
        "loss_last": m.get("loss_last"),
        "parameter_count": m.get("parameter_count"),
        "vocab_size": m.get("vocab_size"),
        "tokenizer_label": m.get("tokenizer_label"),
        "saved_checkpoint_count": len(ck_names),
        "saved_checkpoints": ck_names,
        "ready": bool(m.get("word_exposure") == TOTAL_WORDS and needed.issubset(set(ck_names))),
    }


def arm_meta(arm: str) -> dict[str, Any]:
    meta = read_json(META_PATH)
    for a in meta.get("arm_results", []):
        if a.get("arm") == arm:
            return a
    raise KeyError(arm)


def stream_path(arm: str) -> pathlib.Path:
    return POOL_DIR / ARM_STREAMS[arm]


def build_command(arm: str, stream: pathlib.Path, run_dir: pathlib.Path, seed: int, extra_init_seed: int, train_rng_seed: int) -> list[str]:
    r = dict(BASE_RECIPE)
    r.update({"seed": seed, "extra_init_seed": extra_init_seed, "train_rng_seed": train_rng_seed})
    return [
        sys.executable, "-B", str(TRAINER),
        "--example_jsonl", str(stream),
        "--example_jsonl_label", f"regmax_{arm}_samefw_seed{extra_init_seed}",
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


def make_preflight(args: argparse.Namespace, run_dir: pathlib.Path, stream: pathlib.Path, count_words: bool) -> dict[str, Any]:
    for p in [TRAINER, TOKENIZER / "tokenizer.json", META_PATH, stream]:
        if not p.exists():
            raise FileNotFoundError(p)
    meta = arm_meta(args.arm)
    tok_sha = verify_tokenizer()
    stream_sha = sha256_file(stream)
    expected_sha = str(meta.get("sha_100m"))
    if expected_sha and stream_sha != expected_sha:
        raise RuntimeError({"stream_sha": stream_sha, "expected_sha": expected_sha})
    rows_words = None
    if count_words:
        rows_words = count_stream(stream)
        if rows_words[1] != TOTAL_WORDS:
            raise RuntimeError({"stream_words": rows_words[1], "expected": TOTAL_WORDS})
    recipe = dict(BASE_RECIPE)
    recipe.update({"seed": args.seed, "extra_init_seed": args.extra_init_seed, "train_rng_seed": args.train_rng_seed})
    cmd = build_command(args.arm, stream, run_dir, args.seed, args.extra_init_seed, args.train_rng_seed)
    return {
        "status": "REGISTER_SEED_REPLICATE_PREFLIGHT_OK",
        "created_utc": now(),
        "arm": args.arm,
        "run_dir": rel(run_dir),
        "stream": rel(stream),
        "stream_sha256": stream_sha,
        "stream_sha256_matches_step284": True,
        "tokenizer": rel(TOKENIZER),
        "tokenizer_sha256": tok_sha,
        "trainer": rel(TRAINER),
        "seed_triplet": {"seed": args.seed, "extra_init_seed": args.extra_init_seed, "train_rng_seed": args.train_rng_seed},
        "recipe": recipe,
        "command": cmd,
        "arm_metadata_core": {
            "selected_holdout_summary": meta.get("selected_holdout_summary"),
            "pool_summary": meta.get("pool_summary"),
            "topup_summary": meta.get("topup_summary"),
        },
        "rows_words_counted": {"rows": rows_words[0], "words": rows_words[1]} if rows_words else None,
        "scientific_role": "Independent-basin replication of the seed43022 register reversal: with identical FineWeb admission and fixed 10M/100M budget, test whether removing CHILDES/OpenSubtitles/BNC/Switchboard remains more harmful than removing Gutenberg/SimpleWiki.",
        "what_this_run_decides": "If the negative childspeech_removed - adultprose_removed contrast reproduces at 80M/100M on BLiMP/Supplement/EWoK/COMPS, the developmental/spoken core has measurable fixed-budget opportunity value in this coordinate. If it vanishes or reverses, the seed43022 inversion is not stable enough to state as a principle.",
        "lowest_reliable_method": "The streams and hashes already exist; after research showed item deltas and few-tenth family movements do not carry across seeds, a full second-basin training pair is the smallest reliable test of this training-trajectory contrast.",
        "no_official_eval_globalpiqa_superglue_aoa_upload_or_leaderboard": True,
    }


def do_dry_run(args: argparse.Namespace, run_dir: pathlib.Path, stream: pathlib.Path) -> None:
    info = make_preflight(args, run_dir, stream, args.count_words)
    from transformers import DebertaV2Config, DebertaV2ForMaskedLM
    ref_config_dir = RUNS_DIR / "full_p2c_c2p_abs_clean_dose2p64x_matched_rowholdout_deberta100M_seed43022/hf_model/chck_100M"
    cfg = DebertaV2Config.from_pretrained(str(ref_config_dir))
    model = DebertaV2ForMaskedLM(cfg)
    n_params = sum(p.numel() for p in model.parameters())
    info["model_coordinate"] = {
        "parameter_count": n_params,
        "expected_parameter_count": BASE_RECIPE["parameter_count_expected"],
        "params_match": n_params == BASE_RECIPE["parameter_count_expected"],
        "vocab_size": cfg.vocab_size,
        "expected_vocab_size": BASE_RECIPE["vocab_size_expected"],
    }
    if n_params != BASE_RECIPE["parameter_count_expected"] or cfg.vocab_size != BASE_RECIPE["vocab_size_expected"]:
        raise RuntimeError(info["model_coordinate"])
    print(json.dumps(info, indent=2, ensure_ascii=False), flush=True)


def do_train(args: argparse.Namespace, run_dir: pathlib.Path, stream: pathlib.Path) -> None:
    existing = summarize_metrics(run_dir)
    if existing.get("ready"):
        print(json.dumps({"status": "REGISTER_SEED_REPLICATE_ALREADY_COMPLETE", "arm": args.arm, "run_dir": rel(run_dir), "metrics": existing}, indent=2, ensure_ascii=False), flush=True)
        return
    if run_dir.exists() and any(run_dir.iterdir()):
        raise SystemExit(f"run_dir already exists and is non-empty but not complete: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    info = make_preflight(args, run_dir, stream, args.count_words)
    write_json(run_dir / "register_seed_replicate_preflight.json", info)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    env.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    env["TMPDIR"] = f"/tmp/q_frontier_consolidation_step299_regmax_{args.arm}_{args.extra_init_seed}_{int(time.time())}"
    pathlib.Path(env["TMPDIR"]).mkdir(parents=True, exist_ok=True)
    cmd = info["command"]
    print(json.dumps({
        "status": "REGISTER_SEED_REPLICATE_TRAIN_STARTING",
        "created_utc": now(),
        "arm": args.arm,
        "gpu": args.gpu,
        "run_dir": rel(run_dir),
        "stream_sha256": info["stream_sha256"],
        "seed_triplet": info["seed_triplet"],
        "what_this_run_decides": info["what_this_run_decides"],
        "lowest_reliable_method": info["lowest_reliable_method"],
        "no_official_eval_globalpiqa_superglue_aoa_upload_or_leaderboard": True,
    }, indent=2, ensure_ascii=False), flush=True)
    stdout_path = run_dir / "train_stdout.log"
    stderr_path = run_dir / "train_stderr.log"
    with stdout_path.open("w", encoding="utf-8") as out, stderr_path.open("w", encoding="utf-8") as err:
        out.write(json.dumps({"event": "register_seed_replicate_launcher_start", "utc": now(), "arm": args.arm, "gpu": args.gpu, "preflight": info}, ensure_ascii=False) + "\n")
        out.flush()
        proc = subprocess.run(cmd, cwd=str(ROOT), env=env, stdout=out, stderr=err, text=True)
    metrics = summarize_metrics(run_dir)
    result = {
        "status": "REGISTER_SEED_REPLICATE_TRAIN_DONE" if proc.returncode == 0 else "REGISTER_SEED_REPLICATE_TRAIN_FAILED",
        "arm": args.arm,
        "returncode": proc.returncode,
        "finished_utc": now(),
        "run_dir": rel(run_dir),
        "stdout_log": rel(stdout_path),
        "stderr_log": rel(stderr_path),
        "seed_triplet": {"seed": args.seed, "extra_init_seed": args.extra_init_seed, "train_rng_seed": args.train_rng_seed},
        "metrics": metrics,
        "no_official_eval_globalpiqa_superglue_aoa_upload_or_leaderboard": True,
    }
    if proc.returncode != 0 and stderr_path.exists():
        result["stderr_tail"] = stderr_path.read_text(encoding="utf-8", errors="replace")[-4000:]
    write_json(run_dir / "register_seed_replicate_result.json", result)
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arm", required=True, choices=sorted(ARM_STREAMS))
    ap.add_argument("--gpu", type=int, required=True)
    ap.add_argument("--seed", type=int, default=43)
    ap.add_argument("--extra-init-seed", type=int, default=43122)
    ap.add_argument("--train-rng-seed", type=int, default=43123)
    ap.add_argument("--run-dir", default="")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--count-words", action="store_true")
    args = ap.parse_args()
    stream = stream_path(args.arm)
    run_dir = pathlib.Path(args.run_dir) if args.run_dir else RUNS_DIR / run_name(args.arm, args.extra_init_seed)
    if not run_dir.is_absolute():
        run_dir = ROOT / run_dir
    if args.dry_run:
        do_dry_run(args, run_dir, stream)
    else:
        do_train(args, run_dir, stream)


if __name__ == "__main__":
    main()
