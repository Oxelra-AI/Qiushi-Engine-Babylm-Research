#!/usr/bin/env python3
"""research: train position-matched direct register-displacement DeBERTa arms.

The arms were materialized by materialize_register_position_matched_direct_pools.py.
They admit the identical quarter_1x FineWeb source+compact-view block at rho≈0.0106,
while directly replacing row-position-matched clean rows dominated by either
CHILDES/OpenSubtitles or Gutenberg/SimpleWiki. This launcher uses the same
DeBERTa coordinate, tokenizer, seeds, WWM, optimizer, and checkpoint convention as
research sub-dose trainings.

Use --dry-run before any H100 launch. No evaluation, GlobalPIQA, SuperGLUE, AoA,
packaging, upload, or leaderboard action.
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
POOL_DIR = WS / "data/register_position_matched_direct_pools"
RUNS_DIR = WS / "training/runs"
META_PATH = POOL_DIR / "register_position_matched_direct_pools_metadata.json"
EXPECTED_TOKENIZER_SHA = "91b775514b9f4e2d1f37c28445ab98181007e16d14f763955168547e293ee8f9"
TOTAL_WORDS = 100_000_000
SEED_LABEL = "seed43022"
SEED_OVERRIDES = {"extra_init_seed": 43022, "train_rng_seed": 43023}

ARM_STREAMS = {
    "childsub_posmatched": "regpos_childsub_posmatched_samefw_quarter_100M.jsonl",
    "adult_posmatched": "regpos_adult_posmatched_samefw_quarter_100M.jsonl",
}
RUN_NAMES = {
    "childsub_posmatched": f"regpos_childsub_posmatched_samefw_quarter_deberta100M_{SEED_LABEL}",
    "adult_posmatched": f"regpos_adult_posmatched_samefw_quarter_deberta100M_{SEED_LABEL}",
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


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def rel(p: pathlib.Path) -> str:
    try:
        return str(p.relative_to(ROOT))
    except Exception:
        return str(p)


def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_tokenizer() -> str:
    sha = sha256_file(TOKENIZER / "tokenizer.json")
    assert sha == EXPECTED_TOKENIZER_SHA, f"Tokenizer SHA mismatch: {sha}"
    return sha


def count_stream(path: pathlib.Path) -> tuple[int, int]:
    rows = words = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            words += int(r.get("words", len(str(r.get("text", "")).split())))
            rows += 1
    return rows, words


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


def load_arm_metadata(arm: str) -> dict[str, Any]:
    meta = json.loads(META_PATH.read_text())
    for ar in meta.get("arm_results", []):
        if ar.get("arm_key") == arm:
            return ar
    raise KeyError(f"arm metadata not found for {arm}")


def build_command(arm: str, stream: pathlib.Path, run_dir: pathlib.Path) -> list[str]:
    r = dict(BASE_RECIPE)
    r.update(SEED_OVERRIDES)
    return [
        sys.executable, "-B", str(TRAINER),
        "--example_jsonl", str(stream),
        "--example_jsonl_label", f"regpos_{arm}_samefw_quarter_{SEED_LABEL}",
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


def do_dry_run(arm: str, stream: pathlib.Path, run_dir: pathlib.Path, count_words: bool) -> None:
    arm_meta = load_arm_metadata(arm)
    print(json.dumps({
        "mode": "dry_run",
        "arm": arm,
        "stream": rel(stream),
        "stream_exists": stream.exists(),
        "stream_bytes": stream.stat().st_size if stream.exists() else 0,
        "tokenizer": rel(TOKENIZER),
        "tokenizer_sha": verify_tokenizer(),
        "trainer": rel(TRAINER),
        "trainer_exists": TRAINER.exists(),
        "run_dir": rel(run_dir),
        "seed_label": SEED_LABEL,
        "arm_metadata_core": {
            "rho": arm_meta.get("rho"),
            "active_fineweb_rows": arm_meta.get("active_fineweb_rows"),
            "active_fineweb_words": arm_meta.get("active_fineweb_words"),
            "displaced_clean_summary": arm_meta.get("displaced_clean_summary"),
        },
        "expensive_work_role": "Train this register-displacement arm only to decide whether the rho≈0.011 effect depends on the clean register removed while the admitted FineWeb block is fixed.",
        "lowest_reliable_method": "The pools are already materialized from existing data. Full 100M training with stable-family checkpoints is the minimum reliable downstream competence test; static corpus statistics cannot decide it.",
    }, indent=2, ensure_ascii=False), flush=True)
    if count_words:
        rows, words = count_stream(stream)
        print(json.dumps({"stream_rows": rows, "stream_words": words, "expected_words": TOTAL_WORDS, "words_match": words == TOTAL_WORDS, "expected_updates": rows // BASE_RECIPE["batch_size"]}, indent=2), flush=True)
        assert words == TOTAL_WORDS
    from transformers import DebertaV2Config, DebertaV2ForMaskedLM
    ref_config_dir = RUNS_DIR / "full_p2c_c2p_abs_clean_dose2p64x_matched_rowholdout_deberta100M_seed43022/hf_model/chck_100M"
    cfg = DebertaV2Config.from_pretrained(str(ref_config_dir))
    model = DebertaV2ForMaskedLM(cfg)
    n_params = sum(p.numel() for p in model.parameters())
    print(json.dumps({"parameter_count": n_params, "expected_parameter_count": BASE_RECIPE["parameter_count_expected"], "params_match": n_params == BASE_RECIPE["parameter_count_expected"], "vocab_size": cfg.vocab_size}, indent=2), flush=True)
    assert n_params == BASE_RECIPE["parameter_count_expected"]
    print(f"REGISTER_POSMATCHED_TRAIN_PREFLIGHT_OK: {arm}", flush=True)


def do_train(arm: str, stream: pathlib.Path, run_dir: pathlib.Path, gpu: int) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    existing = summarize_metrics(run_dir)
    if existing.get("ready"):
        print(json.dumps({"status": "ALREADY_COMPLETE", "arm": arm, "run_dir": rel(run_dir), "metrics": existing}, indent=2), flush=True)
        return
    verify_tokenizer()
    stream_sha = sha256_file(stream)
    recipe = dict(BASE_RECIPE)
    recipe.update(SEED_OVERRIDES)
    recipe.update({
        "arm": arm,
        "tokenizer_path": str(TOKENIZER),
        "example_jsonl": str(stream),
        "example_jsonl_label": f"regpos_{arm}_samefw_quarter_{SEED_LABEL}",
        "output_dir": str(run_dir),
        "max_word_exposure": TOTAL_WORDS,
        "tokenizer_label": "compliant16k_reinvest10M",
        "stream_sha256": stream_sha,
        "register_position_matched_metadata": str(META_PATH),
        "launcher_interface": "direct masking_curriculum_trainer.py CLI flags; trainer has no --recipe option",
    })
    (run_dir / "recipe.json").write_text(json.dumps(recipe, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    env.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    env["TMPDIR"] = f"/tmp/q_frontier_consolidation_step282_regpos_{arm}_{int(time.time())}"
    pathlib.Path(env["TMPDIR"]).mkdir(parents=True, exist_ok=True)
    cmd = build_command(arm, stream, run_dir)
    print(json.dumps({
        "status": "REGISTER_POSMATCHED_TRAIN_STARTING",
        "arm": arm,
        "gpu": gpu,
        "run_dir": rel(run_dir),
        "stream": rel(stream),
        "stream_sha": stream_sha,
        "cmd": cmd,
        "utc": now(),
        "expensive_work_role": "Train a register-removal contrast arm: identical FineWeb packet, same rho and geometry, different clean register directly removed.",
        "lowest_reliable_method": "After CPU materialization/preflight, full 100M downstream training is the minimum reliable method to compare stable-family competence under this intervention.",
    }, indent=2, ensure_ascii=False), flush=True)
    stdout_path = run_dir / "train_stdout.log"
    stderr_path = run_dir / "train_stderr.log"
    with stdout_path.open("w", encoding="utf-8") as out, stderr_path.open("w", encoding="utf-8") as err:
        out.write(json.dumps({"event": "register_posmatched_launcher_start", "utc": now(), "arm": arm, "gpu": gpu, "recipe": recipe}, ensure_ascii=False)+"\n")
        out.flush()
        proc = subprocess.run(cmd, cwd=str(ROOT), env=env, stdout=out, stderr=err, text=True)
    metrics = summarize_metrics(run_dir)
    result = {
        "status": "REGISTER_POSMATCHED_TRAIN_DONE" if proc.returncode == 0 else "REGISTER_POSMATCHED_TRAIN_FAILED",
        "arm": arm,
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
    (run_dir / "register_posmatched_launcher_result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arm", required=True, choices=list(ARM_STREAMS.keys()))
    ap.add_argument("--gpu", type=int, required=True)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--count-words", action="store_true")
    args = ap.parse_args()
    stream = POOL_DIR / ARM_STREAMS[args.arm]
    run_dir = RUNS_DIR / RUN_NAMES[args.arm]
    if args.dry_run:
        do_dry_run(args.arm, stream, run_dir, args.count_words)
    else:
        do_train(args.arm, stream, run_dir, args.gpu)


if __name__ == "__main__":
    main()
