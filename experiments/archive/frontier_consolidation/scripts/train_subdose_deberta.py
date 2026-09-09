#!/usr/bin/env python3
"""research: Train sub-1x dose DeBERTa arms in MAX geometry.

Trains the quarter_1x, half_1x, and full_1x dose arms materialized by
materialize_subdose_ladder.py, using the exact same architecture,
optimizer, tokenizer, and seed convention as all other dose family trainings.

Usage:
  python train_subdose_deberta.py --dose quarter_1x --gpu 0
  python train_subdose_deberta.py --dose half_1x --gpu 1
  python train_subdose_deberta.py --dose full_1x --gpu 0
  python train_subdose_deberta.py --dose quarter_1x --gpu 0 --dry-run --count-words

No evaluation, GlobalPIQA, SuperGLUE, AoA, upload, or leaderboard action.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative

import argparse, hashlib, json, os, pathlib, subprocess, sys, time
from typing import Any

def find_user_root() -> pathlib.Path:
    return _PUBLIC_ROOT

ROOT = find_user_root()
WS = ROOT / "experiments/archive/frontier_consolidation"
TRAINER = ROOT / "experiments/archive/compact_experience/scripts/masking_curriculum_trainer.py"
TOKENIZER = WS / "data/compliant_tokenizer"
POOL_DIR = WS / "data/subdose_ladder_maxgeom_pools"
META_PATH = POOL_DIR / "subdose_ladder_metadata.json"
RUNS_DIR = WS / "training/runs"
EXPECTED_TOKENIZER_SHA = "91b775514b9f4e2d1f37c28445ab98181007e16d14f763955168547e293ee8f9"
TOTAL_WORDS = 100_000_000
CKS = [f"chck_{i}M" for i in range(10, 101, 10)]

DOSE_STREAMS = {
    "quarter_1x": "subdose_quarter_1x_view_100M.jsonl",
    "half_1x":    "subdose_half_1x_view_100M.jsonl",
    "full_1x":    "subdose_full_1x_view_100M.jsonl",
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
    "max_word_exposure": 100_000_000,
    "num_workers": 0,
    "log_every": 50,
    "dynamics_trace_every": 200,
    "deberta_pos_att_type": "p2c,c2p",
}
SEED_LABEL = "seed43022"
SEED_OVERRIDES = {"extra_init_seed": 43022, "train_rng_seed": 43023}


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def rel(p: pathlib.Path) -> str:
    try:
        return str(p.relative_to(ROOT))
    except Exception:
        return str(p)


def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def summarize_metrics(run_dir: pathlib.Path) -> dict[str, Any]:
    mf = run_dir / "scientific_metrics.json"
    if not mf.exists():
        return {"ready": False, "metrics_exists": False}
    m = json.loads(mf.read_text())
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
        "saved_checkpoints": m.get("saved_checkpoints"),
        "first_checkpoint": m.get("first_checkpoint"),
        "last_checkpoint": m.get("last_checkpoint"),
        "all_expected_checkpoints_present": m.get("all_expected_checkpoints_present"),
        "ready": m.get("all_expected_checkpoints_present", False) and m.get("word_exposure", 0) == TOTAL_WORDS,
    }


def verify_tokenizer() -> str:
    tok_json = TOKENIZER / "tokenizer.json"
    sha = sha256_file(tok_json)
    assert sha == EXPECTED_TOKENIZER_SHA, f"Tokenizer SHA mismatch: {sha}"
    return sha


def count_stream(path: pathlib.Path) -> tuple[int, int]:
    rows = words = 0
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            words += int(r.get("words", len(r.get("text", "").split())))
            rows += 1
    return rows, words


def do_dry_run(dose: str, stream: pathlib.Path, run_dir: pathlib.Path, count_words: bool):
    print(json.dumps({
        "mode": "dry_run",
        "dose": dose,
        "stream": rel(stream),
        "stream_exists": stream.exists(),
        "stream_bytes": stream.stat().st_size if stream.exists() else 0,
        "tokenizer": rel(TOKENIZER),
        "tokenizer_sha": verify_tokenizer(),
        "trainer": rel(TRAINER),
        "trainer_exists": TRAINER.exists(),
        "run_dir": rel(run_dir),
        "seed_label": SEED_LABEL,
    }, indent=2))

    if count_words and stream.exists():
        print("Counting stream words...")
        rows, words = count_stream(stream)
        print(json.dumps({
            "stream_rows": rows,
            "stream_words": words,
            "expected_words": TOTAL_WORDS,
            "words_match": words == TOTAL_WORDS,
            "expected_updates": rows // BASE_RECIPE["batch_size"],
        }, indent=2))
        assert words == TOTAL_WORDS, f"Stream has {words} words, expected {TOTAL_WORDS}"

    from transformers import DebertaV2Config, DebertaV2ForMaskedLM
    # Use the config from an existing dose-family model for exact match
    ref_config_dir = RUNS_DIR / "full_p2c_c2p_abs_clean_dose2p64x_matched_rowholdout_deberta100M_seed43022/hf_model/chck_100M"
    if ref_config_dir.exists():
        cfg = DebertaV2Config.from_pretrained(str(ref_config_dir))
        model = DebertaV2ForMaskedLM(cfg)
        n_params = sum(p.numel() for p in model.parameters())
        print(json.dumps({
            "parameter_count": n_params,
            "expected_parameter_count": BASE_RECIPE["parameter_count_expected"],
            "params_match": n_params == BASE_RECIPE["parameter_count_expected"],
            "vocab_size": cfg.vocab_size,
        }, indent=2))
        assert n_params == BASE_RECIPE["parameter_count_expected"], (
            f"Param count {n_params} != {BASE_RECIPE['parameter_count_expected']}")
        del model
    else:
        print("  (skipping param count check: no reference config found)")

    print(f"SUBDOSE_TRAIN_PREFLIGHT_OK: {dose}")


def build_command(dose: str, stream: pathlib.Path, run_dir: pathlib.Path) -> list[str]:
    r = dict(BASE_RECIPE)
    r.update(SEED_OVERRIDES)
    return [
        sys.executable, "-B", str(TRAINER),
        "--example_jsonl", str(stream),
        "--example_jsonl_label", f"subdose_{dose}_maxgeom_{SEED_LABEL}",
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


def do_train(dose: str, stream: pathlib.Path, run_dir: pathlib.Path, gpu: int):
    run_dir.mkdir(parents=True, exist_ok=True)

    existing = summarize_metrics(run_dir)
    if existing.get("ready"):
        print(json.dumps({
            "status": "ALREADY_COMPLETE",
            "dose": dose,
            "run_dir": rel(run_dir),
            "metrics": existing,
        }, indent=2))
        return

    verify_tokenizer()
    stream_sha = sha256_file(stream)

    recipe = dict(BASE_RECIPE)
    recipe.update(SEED_OVERRIDES)
    recipe["tokenizer_path"] = str(TOKENIZER)
    recipe["example_jsonl"] = str(stream)
    recipe["example_jsonl_label"] = f"subdose_{dose}_maxgeom_{SEED_LABEL}"
    recipe["output_dir"] = str(run_dir)
    recipe["max_word_exposure"] = TOTAL_WORDS
    recipe["tokenizer_label"] = "compliant16k_reinvest10M"
    recipe["stream_sha256"] = stream_sha
    recipe["launcher_interface"] = "direct masking_curriculum_trainer.py CLI flags; trainer has no --recipe option"

    recipe_path = run_dir / "recipe.json"
    recipe_path.write_text(json.dumps(recipe, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    env.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    env["TMPDIR"] = f"/tmp/q_frontier_consolidation_step280_subdose_{dose}_{int(time.time())}"
    pathlib.Path(env["TMPDIR"]).mkdir(parents=True, exist_ok=True)

    cmd = build_command(dose, stream, run_dir)
    print(json.dumps({
        "status": "SUBDOSE_TRAIN_STARTING",
        "dose": dose,
        "gpu": gpu,
        "run_dir": rel(run_dir),
        "stream": rel(stream),
        "stream_sha": stream_sha,
        "cmd": cmd,
        "utc": now(),
        "expensive_work_role": "Train a sub-1x dose view arm to determine whether the fixed-budget admission effect appears below the old 1x rho=0.042 block or requires more admitted text.",
        "lowest_reliable_method": "The pools are already materialized and audited; one full 100M training with trajectory checkpoints is the minimum reliable test because the target is downstream competence, not static corpus statistics.",
    }, indent=2), flush=True)

    stdout_path = run_dir / "train_stdout.log"
    stderr_path = run_dir / "train_stderr.log"
    with stdout_path.open("w", encoding="utf-8") as out, stderr_path.open("w", encoding="utf-8") as err:
        out.write(json.dumps({"event": "subdose_launcher_start", "utc": now(), "dose": dose, "gpu": gpu, "recipe": recipe}, ensure_ascii=False) + "\n")
        out.flush()
        proc = subprocess.run(cmd, cwd=str(ROOT), env=env, stdout=out, stderr=err, text=True)

    metrics = summarize_metrics(run_dir)
    result = {
        "status": "SUBDOSE_TRAIN_DONE" if proc.returncode == 0 else "SUBDOSE_TRAIN_FAILED",
        "dose": dose,
        "returncode": proc.returncode,
        "finished_utc": now(),
        "run_dir": rel(run_dir),
        "stdout_log": rel(stdout_path),
        "stderr_log": rel(stderr_path),
        "metrics": metrics,
        "no_evaluation_globalpiqa_superglue_aoa_upload_or_leaderboard": True,
    }
    if proc.returncode != 0 and stderr_path.exists():
        result["stderr_tail"] = stderr_path.read_text(encoding="utf-8", errors="replace")[-4000:]
    (run_dir / "subdose_launcher_result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)
    if proc.returncode != 0:
        sys.exit(proc.returncode)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dose", required=True, choices=list(DOSE_STREAMS.keys()))
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--count-words", action="store_true")
    args = parser.parse_args()

    stream = POOL_DIR / DOSE_STREAMS[args.dose]
    run_dir = RUNS_DIR / f"subdose_{args.dose}_deberta100M_{SEED_LABEL}"

    if args.dry_run:
        do_dry_run(args.dose, stream, run_dir, args.count_words)
    else:
        do_train(args.dose, stream, run_dir, args.gpu)


if __name__ == "__main__":
    main()
