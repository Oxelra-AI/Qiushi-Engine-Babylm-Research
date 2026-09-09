#!/usr/bin/env python3
"""research: launcher/dry-run wrapper for RoBERTa MAX-dose transfer tests.

This prepares the generality test that should follow the DeBERTa dose readout.
The already materialized research MAX pools are architecture-agnostic text under
the fixed research tokenizer. This wrapper reuses the research RoBERTa MLM trainer
and keeps the recipe of the earlier 1x RoBERTa transfer run, while replacing only
the data stream with the matched 2.64x row-holdout pools.

No GPU training is started in --dry-run mode. Actual launch should wait until the
DeBERTa budget decomposition identifies the load-bearing leg:
  * semantic leg grows -> launch view and repeat;
  * source/freed-budget leg carries growth -> launch view and clean.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse
import hashlib
import json
import math
import os
import pathlib
import shutil
import subprocess
import sys
import time
from typing import Any

from transformers import AutoTokenizer, RobertaConfig, RobertaForMaskedLM


def find_user_root() -> pathlib.Path:
    return _PUBLIC_ROOT


USER_ROOT = find_user_root()
WS = USER_ROOT / "experiments/archive/frontier_consolidation"
TRAINER = WS / "scripts/roberta_mlm_transfer_trainer.py"
TOKENIZER = WS / "data/compliant_tokenizer"
POOL_DIR = WS / "data/dose_2p64x_rowholdout_pools"
META_PATH = POOL_DIR / "dose2p64x_rowholdout_metadata.json"
PREFLIGHT_DIR = WS / "data/roberta_maxdose_transfer_preflight"
RUNS_DIR = WS / "training/runs"
EXPECTED_TOKENIZER_SHA = "91b775514b9f4e2d1f37c28445ab98181007e16d14f763955168547e293ee8f9"
TOTAL_WORDS = 100_000_000
TEN_M_WORDS = 10_000_000

RECIPE = {
    "model_family": "roberta",
    "parameter_count_expected": 30_528_064,
    "vocab_size_expected": 16_384,
    "hidden_size": 480,
    "n_layer": 8,
    "n_head": 8,
    "ffn_mult": 4,
    "seed": 43,
    "extra_init_seed": 43022,
    "train_rng_seed": 43023,
    "batch_size": 256,
    "seq_length": 256,
    "max_seq_length": 256,
    "learning_rate": 0.001,
    "weight_decay": 0.01,
    "warmup_fraction": 0.06,
    "mask_prob": 0.15,
    "checkpoint_words": 10_000_000,
    "max_word_exposure": 100_000_000,
    "lr_total_steps": 2529,
    "num_workers": 0,
    "log_every": 50,
}

STREAM_KEYS = {
    "view": {
        "training_meta_key": "compact_view_dose2p64x",
        "sha_meta_key": "compact_view_dose2p64x_100M.jsonl",
        "label": "roberta_compact_view_dose2p64x_matched_rowholdout",
    },
    "repeat": {
        "training_meta_key": "compact_repeat_dose2p64x",
        "sha_meta_key": "compact_repeat_dose2p64x_100M.jsonl",
        "label": "roberta_compact_repeat_dose2p64x_matched_rowholdout",
    },
    "clean": {
        "pool_10m_name": "cleanqwen_lengthmatched_dose2p64x_10M.jsonl",
        "pool_100m_name": "cleanqwen_lengthmatched_dose2p64x_100M.jsonl",
        "sha_10m_meta_key": "cleanqwen_lengthmatched_dose2p64x_10M.jsonl",
        "label": "roberta_cleanqwen_lengthmatched_dose2p64x_rowholdout",
    },
}


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


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


def count_jsonl(path: pathlib.Path) -> dict[str, Any]:
    rows = 0
    words = 0
    source_words: dict[str, int] = {}
    first_rows: list[dict[str, Any]] = []
    last_rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            text = str(obj.get("text") or "")
            field_words = int(obj.get("words", len(text.split())))
            actual_words = len(text.split())
            if field_words != actual_words:
                raise RuntimeError(f"word-count mismatch in {path} row {rows}: field={field_words} actual={actual_words}")
            rows += 1
            words += field_words
            src = str(obj.get("source") or "")
            source_words[src] = source_words.get(src, 0) + field_words
            meta = {"row": rows - 1, "words": field_words, "example_id": obj.get("example_id"), "source": src, "text_prefix": text[:120]}
            if len(first_rows) < 4:
                first_rows.append(meta)
            last_rows.append(meta)
            if len(last_rows) > 4:
                last_rows.pop(0)
    return {
        "path": str(path),
        "rows": rows,
        "words": words,
        "sha256": sha256_file(path),
        "first_rows": first_rows,
        "last_rows": last_rows,
        "source_word_types": len(source_words),
        "source_words_top20": sorted(source_words.items(), key=lambda kv: (-kv[1], kv[0]))[:20],
        "expected_update_steps_at_batch256": math.ceil(rows / RECIPE["batch_size"]),
    }


def materialize_clean100(overwrite: bool = False) -> dict[str, Any]:
    clean10 = POOL_DIR / STREAM_KEYS["clean"]["pool_10m_name"]
    clean100 = POOL_DIR / STREAM_KEYS["clean"]["pool_100m_name"]
    sidecar = clean100.with_suffix(".materialization.json")
    if not clean10.exists():
        raise FileNotFoundError(clean10)
    if clean100.exists() and not overwrite:
        return {
            "created": False,
            "reason": "existing clean 100M stream retained",
            "path": str(clean100),
            "sha256": sha256_file(clean100),
            "sidecar": str(sidecar),
            "sidecar_exists": sidecar.exists(),
        }
    tmp = clean100.with_suffix(".jsonl.tmp")
    with tmp.open("wb") as out:
        for _ in range(10):
            with clean10.open("rb") as src:
                shutil.copyfileobj(src, out, length=1 << 20)
    os.replace(tmp, clean100)
    payload = {
        "status": "CLEAN_ROBERTA_100M_STREAM_MATERIALIZED",
        "created_utc": now(),
        "source_10m": str(clean10),
        "source_10m_sha256": sha256_file(clean10),
        "repeats": 10,
        "output_100m": str(clean100),
        "output_100m_sha256": sha256_file(clean100),
        "purpose": "RoBERTa MAX-dose view-versus-clean generality leg if the DeBERTa decomposition shows the fixed-budget source/freed-budget component carries the dose effect.",
    }
    write_json(sidecar, payload)
    return {"created": True, "path": str(clean100), "sha256": payload["output_100m_sha256"], "sidecar": str(sidecar)}


def load_meta() -> dict[str, Any]:
    if not META_PATH.exists():
        raise FileNotFoundError(META_PATH)
    meta = read_json(META_PATH)
    if meta.get("status") != "MATCHED_MAX_ROWHOLDOUT_POOLS_MATERIALIZED":
        raise RuntimeError(f"unexpected MAX pool metadata status: {meta.get('status')}")
    audit = meta.get("audit") or {}
    if not (audit.get("all_exact_10M") and audit.get("row_length_sequence_identical_all_arms") and audit.get("view_repeat_suffix_identical_after_changed_block")):
        raise RuntimeError(f"research pool integrity fields are not clean: {audit}")
    return meta


def stream_path(data_arm: str, meta: dict[str, Any]) -> pathlib.Path:
    if data_arm == "clean":
        return POOL_DIR / STREAM_KEYS["clean"]["pool_100m_name"]
    spec = STREAM_KEYS[data_arm]
    p = pathlib.Path(meta["files"]["training"][spec["training_meta_key"]])
    return p if p.is_absolute() else USER_ROOT / p


def expected_sha(data_arm: str, meta: dict[str, Any]) -> str | None:
    if data_arm == "clean":
        sidecar = (POOL_DIR / STREAM_KEYS["clean"]["pool_100m_name"]).with_suffix(".materialization.json")
        if sidecar.exists():
            return str(read_json(sidecar).get("output_100m_sha256"))
        return None
    return str(meta["sha256"][STREAM_KEYS[data_arm]["sha_meta_key"]])


def model_variant() -> dict[str, Any]:
    tok = AutoTokenizer.from_pretrained(str(TOKENIZER), use_fast=True)
    cfg = RobertaConfig(
        vocab_size=len(tok),
        hidden_size=RECIPE["hidden_size"],
        num_hidden_layers=RECIPE["n_layer"],
        num_attention_heads=RECIPE["n_head"],
        intermediate_size=RECIPE["hidden_size"] * RECIPE["ffn_mult"],
        max_position_embeddings=512,
        type_vocab_size=1,
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        pad_token_id=tok.pad_token_id,
        bos_token_id=tok.bos_token_id,
        eos_token_id=tok.eos_token_id,
        layer_norm_eps=1e-5,
    )
    model = RobertaForMaskedLM(cfg)
    return {"parameter_count": sum(p.numel() for p in model.parameters()), "vocab_size": len(tok), "config": cfg.to_dict()}


def default_run_dir(data_arm: str) -> pathlib.Path:
    return RUNS_DIR / f"roberta_{data_arm}_dose2p64x_matched_rowholdout_100M_seed43022"


def build_command(data_arm: str, run_dir: pathlib.Path, stream: pathlib.Path) -> list[str]:
    return [
        sys.executable, "-B", str(TRAINER),
        "--tokenizer_path", str(TOKENIZER),
        "--tokenizer_label", "compliant16k_reinvest10M",
        "--model_family", "roberta",
        "--hidden_size", str(RECIPE["hidden_size"]),
        "--n_layer", str(RECIPE["n_layer"]),
        "--n_head", str(RECIPE["n_head"]),
        "--ffn_mult", str(RECIPE["ffn_mult"]),
        "--seed", str(RECIPE["seed"]),
        "--extra_init_seed", str(RECIPE["extra_init_seed"]),
        "--train_rng_seed", str(RECIPE["train_rng_seed"]),
        "--batch_size", str(RECIPE["batch_size"]),
        "--seq_length", str(RECIPE["seq_length"]),
        "--max_seq_length", str(RECIPE["max_seq_length"]),
        "--learning_rate", str(RECIPE["learning_rate"]),
        "--weight_decay", str(RECIPE["weight_decay"]),
        "--warmup_fraction", str(RECIPE["warmup_fraction"]),
        "--mask_prob", str(RECIPE["mask_prob"]),
        "--max_word_exposure", str(RECIPE["max_word_exposure"]),
        "--checkpoint_words", str(RECIPE["checkpoint_words"]),
        "--lr_total_steps", str(RECIPE["lr_total_steps"]),
        "--num_workers", str(RECIPE["num_workers"]),
        "--log_every", str(RECIPE["log_every"]),
        "--example_jsonl", str(stream),
        "--example_jsonl_label", STREAM_KEYS[data_arm]["label"],
        "--output_dir", str(run_dir),
    ]


def run_smoke(data_arm: str, stream: pathlib.Path, out_dir: pathlib.Path) -> dict[str, Any]:
    smoke_dir = out_dir / "smoke_forward" / data_arm
    smoke_dir.mkdir(parents=True, exist_ok=True)
    cmd = build_command(data_arm, smoke_dir, stream) + ["--smoke-forward-only", "--smoke-rows", "4", "--smoke-batch-size", "2"]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["TOKENIZERS_PARALLELISM"] = "false"
    env.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    proc = subprocess.run(cmd, cwd=str(USER_ROOT), env=env, capture_output=True, text=True, timeout=300)
    (smoke_dir / "smoke_stdout.log").write_text(proc.stdout, encoding="utf-8")
    (smoke_dir / "smoke_stderr.log").write_text(proc.stderr, encoding="utf-8")
    payload_path = smoke_dir / "smoke_forward.json"
    payload = read_json(payload_path) if payload_path.exists() else None
    return {
        "returncode": proc.returncode,
        "cmd": cmd,
        "stdout_tail": proc.stdout[-1200:],
        "stderr_tail": proc.stderr[-1200:],
        "payload_path": str(payload_path),
        "payload": payload,
        "finite_loss": payload.get("finite_loss") if isinstance(payload, dict) else None,
        "loss": payload.get("loss") if isinstance(payload, dict) else None,
        "masked_tokens": payload.get("mask_stats", {}).get("masked_tokens") if isinstance(payload, dict) else None,
    }


def preflight_one(data_arm: str, args: argparse.Namespace, mv: dict[str, Any] | None = None) -> dict[str, Any]:
    if data_arm not in ["view", "repeat", "clean"]:
        raise ValueError(data_arm)
    if not TRAINER.exists():
        raise FileNotFoundError(TRAINER)
    if not TOKENIZER.exists():
        raise FileNotFoundError(TOKENIZER)
    meta = load_meta()
    if data_arm == "clean" or args.materialize_clean:
        clean_materialization = materialize_clean100(overwrite=args.overwrite_clean100)
    else:
        clean_materialization = None
    tok_sha = sha256_file(TOKENIZER / "tokenizer.json")
    if tok_sha != EXPECTED_TOKENIZER_SHA:
        raise RuntimeError(f"tokenizer SHA mismatch: {tok_sha}")
    stream = stream_path(data_arm, meta)
    if not stream.exists():
        raise FileNotFoundError(stream)
    count = count_jsonl(stream) if args.count_words else {"path": str(stream), "count_skipped": True, "sha256": sha256_file(stream)}
    if not count.get("count_skipped"):
        if int(count["words"]) != TOTAL_WORDS:
            raise RuntimeError(f"{data_arm} stream words {count['words']} != {TOTAL_WORDS}")
        if int(count["rows"]) != 653_130:
            raise RuntimeError(f"{data_arm} stream rows {count['rows']} != 653130")
    exp_sha = expected_sha(data_arm, meta)
    if exp_sha is not None and count.get("sha256") != exp_sha:
        raise RuntimeError(f"{data_arm} stream SHA mismatch: {count.get('sha256')} != {exp_sha}")
    if mv is None:
        mv = model_variant()
    if mv["parameter_count"] != RECIPE["parameter_count_expected"] or mv["vocab_size"] != RECIPE["vocab_size_expected"]:
        raise RuntimeError(f"RoBERTa model coordinate mismatch: {mv['parameter_count']} params, vocab {mv['vocab_size']}")
    run_dir = pathlib.Path(args.run_dir) if args.run_dir and data_arm != "all" else default_run_dir(data_arm)
    if not run_dir.is_absolute():
        run_dir = USER_ROOT / run_dir
    preflight = {
        "status": "ROBERTA_MAXDOSE_TRANSFER_PREFLIGHT_OK",
        "created_utc": now(),
        "data_arm": data_arm,
        "scientific_purpose": "CPU dry-run for a future RoBERTa MAX-dose generality test using the fixed research tokenizer and research matched 2.64x row-holdout pools.",
        "stream": str(stream),
        "stream_accounting": count,
        "expected_stream_sha256": exp_sha,
        "clean100_materialization": clean_materialization,
        "tokenizer_dir": str(TOKENIZER),
        "tokenizer_json_sha256": tok_sha,
        "model_variant": {k: v for k, v in mv.items() if k != "config"},
        "recipe": RECIPE,
        "actual_updates_at_batch256_from_rows": count.get("expected_update_steps_at_batch256"),
        "lr_total_steps_kept_from_step211_and_step256": RECIPE["lr_total_steps"],
        "run_dir_if_launched": str(run_dir),
        "command_if_launched": build_command(data_arm, run_dir, stream),
        "launch_policy": {
            "semantic_leg_load_bearing": "launch view + repeat RoBERTa MAX-dose pair",
            "source_or_freed_budget_leg_load_bearing": "launch view + clean RoBERTa MAX-dose pair as the total fixed-budget treatment transfer test",
            "do_not_launch_before": "research/258 DeBERTa budget-decomposition readout identifies which leg carries the MAX-minus-1x dose growth",
        },
        "no_training_started": bool(args.dry_run),
        "no_evaluation_superglue_aoa_upload_or_leaderboard": True,
    }
    if args.smoke_forward:
        preflight["cpu_smoke_forward"] = run_smoke(data_arm, stream, PREFLIGHT_DIR)
        smoke = preflight["cpu_smoke_forward"]
        if smoke.get("returncode") != 0 or not smoke.get("finite_loss") or not smoke.get("masked_tokens"):
            raise RuntimeError(f"CPU smoke-forward failed for {data_arm}: {smoke}")
    write_json(PREFLIGHT_DIR / f"{data_arm}_preflight.json", preflight)
    return preflight


def summarize_metrics(metrics_path: pathlib.Path) -> dict[str, Any]:
    if not metrics_path.exists():
        return {"metrics_exists": False}
    m = read_json(metrics_path)
    saved = [x.get("name") for x in m.get("saved_checkpoints", []) if isinstance(x, dict)]
    return {
        "metrics_exists": True,
        "status": m.get("status"),
        "word_exposure": m.get("word_exposure"),
        "actual_training_steps": m.get("actual_training_steps"),
        "parameter_count": m.get("parameter_count"),
        "vocab_size": m.get("vocab_size"),
        "tokenizer_label": m.get("tokenizer_label"),
        "loss_first": m.get("loss_first"),
        "loss_last": m.get("loss_last"),
        "saved_checkpoints": saved,
    }


def launch_one(data_arm: str, args: argparse.Namespace) -> None:
    args2 = argparse.Namespace(**vars(args))
    args2.dry_run = False
    args2.count_words = True
    pre = preflight_one(data_arm, args2)
    run_dir = pathlib.Path(pre["run_dir_if_launched"])
    if run_dir.exists() and any(run_dir.iterdir()) and not args.allow_nonempty:
        raise RuntimeError(f"run_dir exists and is non-empty: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    write_json(run_dir / "launcher_preflight.json", pre)
    stdout_path = run_dir / "train_stdout.log"
    stderr_path = run_dir / "train_stderr.log"
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    env["TOKENIZERS_PARALLELISM"] = "false"
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    env.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    env["TMPDIR"] = f"/tmp/q_frontier_consolidation_step257_roberta_{data_arm}_{int(time.time())}"
    pathlib.Path(env["TMPDIR"]).mkdir(parents=True, exist_ok=True)
    started = now()
    with stdout_path.open("w", encoding="utf-8") as out, stderr_path.open("w", encoding="utf-8") as err:
        out.write(json.dumps({"event": "roberta_launcher_start", "utc": started, "data_arm": data_arm, "gpu": args.gpu}) + "\n")
        out.flush()
        proc = subprocess.run(pre["command_if_launched"], stdout=out, stderr=err, text=True, env=env, cwd=str(USER_ROOT))
    result = {
        "status": "ROBERTA_MAXDOSE_TRANSFER_TRAIN_FINISHED" if proc.returncode == 0 else "ROBERTA_MAXDOSE_TRANSFER_TRAIN_FAILED",
        "returncode": proc.returncode,
        "data_arm": data_arm,
        "gpu": args.gpu,
        "started_utc": started,
        "finished_utc": now(),
        "run_dir": str(run_dir),
        "stdout_log": str(stdout_path),
        "stderr_log": str(stderr_path),
        "metrics_path": str(run_dir / "scientific_metrics.json"),
        "metrics": summarize_metrics(run_dir / "scientific_metrics.json"),
        "no_evaluation_superglue_aoa_upload_or_leaderboard": True,
    }
    if proc.returncode != 0 and stderr_path.exists():
        result["stderr_tail"] = stderr_path.read_text(encoding="utf-8", errors="replace")[-4000:]
    write_json(run_dir / "launcher_result.json", result)
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)
    raise SystemExit(proc.returncode)


def write_scaffold_report(preflights: list[dict[str, Any]]) -> None:
    PREFLIGHT_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "status": "ROBERTA_MAXDOSE_TRANSFER_SCAFFOLD_READY",
        "created_utc": now(),
        "scientific_question": "Does the amplified 2.64x compact-dose effect transfer to a second bidirectional absolute-position MLM coordinate under the same fixed tokenizer and text pools?",
        "deberta_dependency": "Do not launch GPU RoBERTa MAX-dose training until the DeBERTa budget-decomposition readout identifies whether semantic re-expression or source/freed-budget admission carries MAX-minus-1x growth.",
        "arms_preflighted": [p["data_arm"] for p in preflights],
        "preflights": {p["data_arm"]: str(PREFLIGHT_DIR / f"{p['data_arm']}_preflight.json") for p in preflights},
        "launch_choices_after_deberta": {
            "semantic_growth": ["view", "repeat"],
            "source_or_freed_budget_growth": ["view", "clean"],
        },
        "common_recipe": RECIPE,
        "stream_rows_words": {p["data_arm"]: {"rows": p["stream_accounting"].get("rows"), "words": p["stream_accounting"].get("words"), "sha256": p["stream_accounting"].get("sha256"), "updates_at_batch256": p.get("actual_updates_at_batch256_from_rows")} for p in preflights},
        "no_training_started": True,
        "no_evaluation_superglue_aoa_upload_or_leaderboard": True,
    }
    write_json(PREFLIGHT_DIR / "roberta_maxdose_transfer_scaffold.json", payload)
    lines = [
        "# research RoBERTa MAX-dose transfer scaffold",
        "",
        "This is a CPU dry-run scaffold; no RoBERTa MAX-dose GPU training has been launched.",
        "",
        f"Question: {payload['scientific_question']}",
        "",
        "## Launch rule after DeBERTa dose decomposition",
        "- If MAX-minus-1x growth is carried by `view-repeat`, launch RoBERTa `view` + `repeat`.",
        "- If growth is carried by source/freed-budget admission, launch RoBERTa `view` + `clean` for the total fixed-budget treatment transfer.",
        "",
        "## Preflighted streams",
    ]
    for p in preflights:
        c = p["stream_accounting"]
        lines.append(f"- {p['data_arm']}: rows {c.get('rows')} words {c.get('words')} sha `{c.get('sha256')}` updates/batch256 {p.get('actual_updates_at_batch256_from_rows')}; preflight `{PREFLIGHT_DIR / (p['data_arm'] + '_preflight.json')}`")
        smoke = p.get("cpu_smoke_forward")
        if smoke:
            lines.append(f"  - CPU smoke: returncode {smoke.get('returncode')}, finite_loss {smoke.get('finite_loss')}, loss {smoke.get('loss')}, masked_tokens {smoke.get('masked_tokens')}")
    lines += ["", f"JSON: `{PREFLIGHT_DIR / 'roberta_maxdose_transfer_scaffold.json'}`"]
    (PREFLIGHT_DIR / "roberta_maxdose_transfer_scaffold.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-arm", choices=["view", "repeat", "clean", "all"], required=True)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--run-dir", default="")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--count-words", action="store_true")
    ap.add_argument("--materialize-clean", action="store_true")
    ap.add_argument("--overwrite-clean100", action="store_true")
    ap.add_argument("--smoke-forward", action="store_true")
    ap.add_argument("--allow-nonempty", action="store_true")
    args = ap.parse_args()

    if args.data_arm == "all" and not args.dry_run:
        raise SystemExit("--data-arm all is only supported with --dry-run")
    PREFLIGHT_DIR.mkdir(parents=True, exist_ok=True)
    if args.dry_run:
        arms = ["view", "repeat", "clean"] if args.data_arm == "all" else [args.data_arm]
        mv = model_variant()
        outs = [preflight_one(a, args, mv=mv) for a in arms]
        write_scaffold_report(outs)
        print(json.dumps({
            "status": "ROBERTA_MAXDOSE_TRANSFER_DRYRUN_OK",
            "arms": arms,
            "preflight_dir": str(PREFLIGHT_DIR),
            "scaffold_json": str(PREFLIGHT_DIR / "roberta_maxdose_transfer_scaffold.json"),
            "no_training_started": True,
        }, indent=2, ensure_ascii=False), flush=True)
        return
    launch_one(args.data_arm, args)


if __name__ == "__main__":
    main()
