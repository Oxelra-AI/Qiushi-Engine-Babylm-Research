#!/usr/bin/env python3
"""research: isolated Strict-Small-compliant tokenizer training and retrain launcher.

Tokenizer provenance correction: the inherited baseline16k tokenizer used by compact_view_reinvest
was trained on Strict 100M text, not only the Strict-Small 10M training corpus. This
script builds the replacement tokenizer from the exact compact_view_reinvest 10M pool
and optionally launches retraining with all original model/data/training seeds and
recipe fixed except tokenizer_path/tokenizer_label.

Modes:
  --mode tokenizer     Train and audit tokenizer only (CPU)
  --mode preflight     Verify tokenizer, data, and command manifests (CPU)
  --mode train         Launch one training seed (GPU, long)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import random
import shutil
import subprocess
import sys
import time
from statistics import mean
from typing import Iterable

from transformers import AutoTokenizer

STUDY = pathlib.Path("experiments/archive/representation_and_objectives")
A02_DENSITY_DIR = pathlib.Path("experiments/archive/frontier_consolidation/data/density_cleanqwen_overlay_medium_riskhard")
TRAIN_10M = A02_DENSITY_DIR / "cleanqwen_fineweb_compact_view_reinvest_10M.jsonl"
TRAIN_100M = A02_DENSITY_DIR / "cleanqwen_fineweb_compact_view_reinvest_100M.jsonl"
META = A02_DENSITY_DIR / "density_cleanqwen_rowholdout_overlay_metadata.json"
OLD_TOKENIZER = pathlib.Path("experiments/archive/initial_model_studies/training/runs/fullcycle_official_wwm_debertav2_8x480_seed43_100M_b256/hf_model")
TRAINER = pathlib.Path("experiments/archive/compact_experience/scripts/masking_curriculum_trainer.py")
OUT_DIR = STUDY / "data/strictsmall_tokenizer_retrain"
TOKENIZER_DIR = OUT_DIR / "strictsmall_compact_reinvest_16k_tokenizer"
AI_RUN_ROOT = STUDY / "training/runs"

EXPECTED_100M_SHA = "3dd19f09deeca44d6b07340e70cd15baa4b5c7bffe0bd462ff37536e470e7691"
VOCAB_SIZE = 16384
SPECIAL_TOKENS = ["<unk>", "<s>", "</s>", "<pad>", "<mask>"]
SEEDS = {
    "43022": {"seed": 43, "extra_init_seed": 43022, "train_rng_seed": 43023, "gpu": 0},
    "43122": {"seed": 43, "extra_init_seed": 43122, "train_rng_seed": 43123, "gpu": 1},
}


def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def jsonl_text_batches(path: pathlib.Path, batch_size: int = 1000) -> Iterable[list[str]]:
    batch: list[str] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            batch.append(str(obj["text"]))
            if len(batch) >= batch_size:
                yield batch
                batch = []
    if batch:
        yield batch


def jsonl_summary(path: pathlib.Path, sample_n: int = 2000) -> dict:
    rows = 0
    words = 0
    samples: list[str] = []
    source_counts: dict[str, int] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            rows += 1
            text = str(obj["text"])
            w = int(obj.get("words", len(text.split())))
            actual = len(text.split())
            if w != actual:
                raise RuntimeError(f"word mismatch in {path} row {rows}: field {w} actual {actual}")
            words += w
            source = str(obj.get("source", ""))
            source_counts[source] = source_counts.get(source, 0) + 1
            if len(samples) < sample_n:
                samples.append(text)
    return {
        "path": str(path),
        "exists": path.exists(),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "rows": rows,
        "whitespace_words": words,
        "sample_texts": samples,
        "source_counts_top": sorted(source_counts.items(), key=lambda kv: (-kv[1], kv[0]))[:20],
    }


def token_summary(tok, texts: list[str]) -> dict:
    lens = []
    word_counts = []
    word_start_tokens = 0
    total_tokens = 0
    for t in texts:
        ids = tok(t, add_special_tokens=False, truncation=False)["input_ids"]
        toks = tok.convert_ids_to_tokens(ids)
        lens.append(len(ids))
        wc = len(t.split())
        word_counts.append(wc)
        total_tokens += len(ids)
        word_start_tokens += sum(1 for s in toks if str(s).startswith("Ġ") or str(s).startswith("▁"))
    total_words = max(1, sum(word_counts))
    return {
        "num_texts": len(texts),
        "total_words": sum(word_counts),
        "total_tokens": total_tokens,
        "tokens_per_whitespace_word": total_tokens / total_words,
        "mean_tokens_per_text": mean(lens) if lens else 0.0,
        "max_tokens_per_text": max(lens) if lens else 0,
        "rows_over_256_tokens": sum(1 for x in lens if x > 256),
        "word_start_token_fraction": word_start_tokens / max(1, total_tokens),
    }


def compare_tokenizers(samples: list[str]) -> dict:
    old_tok = AutoTokenizer.from_pretrained(str(OLD_TOKENIZER), use_fast=True)
    new_tok = AutoTokenizer.from_pretrained(str(TOKENIZER_DIR), use_fast=True)
    probes = [
        "The young child remembered the beautiful garden yesterday.",
        "The Home Energy Rater uses software to analyze the home's design based on its plans.",
        "Polar regions become much colder than now, with large temperature differences from the equator.",
    ]
    out = {
        "old_tokenizer_path": str(OLD_TOKENIZER),
        "new_tokenizer_path": str(TOKENIZER_DIR),
        "old_len": len(old_tok),
        "new_len": len(new_tok),
        "old_special_tokens_map": old_tok.special_tokens_map,
        "new_special_tokens_map": new_tok.special_tokens_map,
        "old_special_ids": old_tok.all_special_ids,
        "new_special_ids": new_tok.all_special_ids,
        "sample_comparison_first_rows": {
            "old_baseline16k": token_summary(old_tok, samples),
            "new_strictsmall16k": token_summary(new_tok, samples),
        },
        "probe_tokens": {},
    }
    for text in probes:
        out["probe_tokens"][text] = {
            "old": old_tok.convert_ids_to_tokens(old_tok(text, add_special_tokens=False)["input_ids"]),
            "new": new_tok.convert_ids_to_tokens(new_tok(text, add_special_tokens=False)["input_ids"]),
        }
    return out


def train_tokenizer(force: bool = False) -> dict:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if TOKENIZER_DIR.exists() and force:
        shutil.rmtree(TOKENIZER_DIR)
    if TOKENIZER_DIR.exists() and (TOKENIZER_DIR / "tokenizer.json").exists() and not force:
        print(f"Tokenizer already exists at {TOKENIZER_DIR}; use --force-tokenizer to retrain", flush=True)
    else:
        TOKENIZER_DIR.mkdir(parents=True, exist_ok=True)
        base = AutoTokenizer.from_pretrained(str(OLD_TOKENIZER), use_fast=True)
        t0 = time.time()
        # This retrains the BPE model with the same tokenizer pipeline as the inherited
        # GPT2-like byte-level baseline, preserving special-token strings/ids and vocab size.
        new_tok = base.train_new_from_iterator(
            jsonl_text_batches(TRAIN_10M, batch_size=1000),
            vocab_size=VOCAB_SIZE,
            length=jsonl_summary(TRAIN_10M, sample_n=0)["rows"],
            new_special_tokens=[],
            min_frequency=2,
            show_progress=True,
        )
        new_tok.save_pretrained(str(TOKENIZER_DIR))
        tok_cfg = TOKENIZER_DIR / "tokenizer_config.json"
        if tok_cfg.exists():
            cfg = json.loads(tok_cfg.read_text(encoding="utf-8"))
        else:
            cfg = {}
        cfg["tokenizer_class"] = "PreTrainedTokenizerFast"
        cfg["model_max_length"] = 1024
        cfg["strict_small_tokenizer_training_source"] = str(TRAIN_10M)
        cfg["strict_small_tokenizer_training_words"] = 10_000_000
        tok_cfg.write_text(json.dumps(cfg, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        elapsed = time.time() - t0
        print(json.dumps({"event": "tokenizer_trained", "elapsed_sec": elapsed, "path": str(TOKENIZER_DIR)}, ensure_ascii=False), flush=True)

    train10 = jsonl_summary(TRAIN_10M, sample_n=2000)
    train100 = jsonl_summary(TRAIN_100M, sample_n=0)
    if train10["whitespace_words"] != 10_000_000:
        raise RuntimeError(f"10M tokenizer file has {train10['whitespace_words']} words")
    if train100["whitespace_words"] != 100_000_000:
        raise RuntimeError(f"100M training file has {train100['whitespace_words']} words")
    if train100["sha256"] != EXPECTED_100M_SHA:
        raise RuntimeError(f"100M sha mismatch: {train100['sha256']} != {EXPECTED_100M_SHA}")

    loaded = AutoTokenizer.from_pretrained(str(TOKENIZER_DIR), use_fast=True)
    tokenizer_hashes = {p.name: sha256_file(p) for p in sorted(TOKENIZER_DIR.glob("*")) if p.is_file()}
    manifest = {
        "status": "STRICTSMALL_TOKENIZER_READY",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "human_correction": "Inherited baseline16k tokenizer was trained outside Strict-Small 10M budget; this tokenizer is trained only on cleanqwen_fineweb_compact_view_reinvest_10M.jsonl.",
        "tokenizer_training_source": train10 | {"sample_texts": "omitted_from_manifest"},
        "model_training_source": train100 | {"sample_texts": "omitted_from_manifest"},
        "metadata_path": str(META),
        "old_tokenizer_path": str(OLD_TOKENIZER),
        "tokenizer_dir": str(TOKENIZER_DIR),
        "tokenizer_vocab_size_requested": VOCAB_SIZE,
        "tokenizer_vocab_size_actual": len(loaded),
        "special_tokens": SPECIAL_TOKENS,
        "special_tokens_map": loaded.special_tokens_map,
        "special_ids": loaded.all_special_ids,
        "tokenizer_hashes": tokenizer_hashes,
        "comparison_to_inherited_tokenizer_on_first_2000_rows": compare_tokenizers(train10["sample_texts"]),
        "strict_small_accounting": {
            "tokenizer_language_training_words": train10["whitespace_words"],
            "model_training_corpus_unique_words": 10_000_000,
            "model_training_exposure_words": train100["whitespace_words"],
            "same_10M_pool_for_tokenizer_and_model": True,
            "same_100M_stream_hash_as_previous_reinvest": train100["sha256"] == EXPECTED_100M_SHA,
        },
    }
    (TOKENIZER_DIR / "strictsmall_tokenizer_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (OUT_DIR / "strictsmall_tokenizer_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": manifest["status"],
        "tokenizer_dir": str(TOKENIZER_DIR),
        "vocab_size": len(loaded),
        "tokenizer_json_sha256": tokenizer_hashes.get("tokenizer.json"),
        "old_tpw": manifest["comparison_to_inherited_tokenizer_on_first_2000_rows"]["sample_comparison_first_rows"]["old_baseline16k"]["tokens_per_whitespace_word"],
        "new_tpw": manifest["comparison_to_inherited_tokenizer_on_first_2000_rows"]["sample_comparison_first_rows"]["new_strictsmall16k"]["tokens_per_whitespace_word"],
        "manifest": str(OUT_DIR / "strictsmall_tokenizer_manifest.json"),
    }, indent=2, ensure_ascii=False), flush=True)
    return manifest


def train_command(seed_key: str) -> tuple[list[str], pathlib.Path, dict]:
    s = SEEDS[seed_key]
    run_dir = AI_RUN_ROOT / f"strictsmalltok_compact_view_reinvest_seed{seed_key}"
    cmd = [
        sys.executable,
        str(TRAINER),
        "--example_jsonl", str(TRAIN_100M),
        "--example_jsonl_label", "cleanqwen_fineweb_compact_view_reinvest_strictsmalltok",
        "--example_jsonl_meta", str(META),
        "--output_dir", str(run_dir),
        "--tokenizer_path", str(TOKENIZER_DIR),
        "--tokenizer_label", "strictsmall_compact_reinvest_16k",
        "--hidden_size", "480",
        "--n_layer", "8",
        "--n_head", "8",
        "--ffn_mult", "4",
        "--seed", str(s["seed"]),
        "--extra_init_seed", str(s["extra_init_seed"]),
        "--train_rng_seed", str(s["train_rng_seed"]),
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
    manifest = {
        "status": "STRICTSMALLTOKENIZER_RETRAIN_COMMAND",
        "seed_key": seed_key,
        "run_dir": str(run_dir),
        "train_file": str(TRAIN_100M),
        "train_file_sha256": sha256_file(TRAIN_100M),
        "expected_sha256": EXPECTED_100M_SHA,
        "hash_ok": sha256_file(TRAIN_100M) == EXPECTED_100M_SHA,
        "tokenizer_path": str(TOKENIZER_DIR),
        "tokenizer_manifest": str(TOKENIZER_DIR / "strictsmall_tokenizer_manifest.json"),
        "tokenizer_sha256": sha256_file(TOKENIZER_DIR / "tokenizer.json") if (TOKENIZER_DIR / "tokenizer.json").exists() else None,
        "changed_factor": "tokenizer only: trained from the same 10M pool instead of inherited Strict-100M tokenizer",
        "fixed_recipe": {
            "model_family": "DeBERTa-v2 masked LM",
            "hidden_size": 480,
            "n_layer": 8,
            "n_head": 8,
            "ffn_mult": 4,
            "seed": s["seed"],
            "extra_init_seed": s["extra_init_seed"],
            "train_rng_seed": s["train_rng_seed"],
            "batch_size": 256,
            "seq_length": 256,
            "max_seq_length": 256,
            "learning_rate": 0.001,
            "warmup_fraction": 0.06,
            "weight_decay": 0.01,
            "masking_curriculum": "wwm_fixed",
            "mask_prob_start": 0.15,
            "mask_prob_end": 0.15,
            "checkpoint_words": 1_000_000,
            "max_word_exposure": 100_000_000,
            "num_workers": 0,
            "log_every": 50,
            "dynamics_trace_every": 200,
        },
        "command": cmd,
        "cuda_visible_devices": str(s["gpu"]),
    }
    return cmd, run_dir, manifest


def preflight() -> dict:
    if not (TOKENIZER_DIR / "tokenizer.json").exists():
        raise FileNotFoundError(f"Tokenizer missing: {TOKENIZER_DIR}; run --mode tokenizer first")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(str(TOKENIZER_DIR), use_fast=True)
    old = AutoTokenizer.from_pretrained(str(OLD_TOKENIZER), use_fast=True)
    ten = jsonl_summary(TRAIN_10M, sample_n=2000)
    hundred = jsonl_summary(TRAIN_100M, sample_n=0)
    payload = {
        "status": "STRICTSMALLTOKENIZER_PREFLIGHT_OK",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "tokenizer_dir": str(TOKENIZER_DIR),
        "tokenizer_len": len(tok),
        "old_tokenizer_len": len(old),
        "tokenizer_hashes": {p.name: sha256_file(p) for p in sorted(TOKENIZER_DIR.glob("*")) if p.is_file()},
        "tokenizer_training_file": ten | {"sample_texts": "omitted_from_preflight"},
        "model_training_file": hundred | {"sample_texts": "omitted_from_preflight"},
        "commands": {},
    }
    if len(tok) != VOCAB_SIZE:
        raise RuntimeError(f"new tokenizer length {len(tok)} != {VOCAB_SIZE}")
    if tok.all_special_ids != old.all_special_ids:
        raise RuntimeError(f"special ids changed: new {tok.all_special_ids}, old {old.all_special_ids}")
    if tok.special_tokens_map != old.special_tokens_map:
        raise RuntimeError(f"special map changed: new {tok.special_tokens_map}, old {old.special_tokens_map}")
    if ten["whitespace_words"] != 10_000_000 or hundred["whitespace_words"] != 100_000_000:
        raise RuntimeError("word accounting failed")
    if hundred["sha256"] != EXPECTED_100M_SHA:
        raise RuntimeError("100M stream hash failed")
    for seed_key in SEEDS:
        cmd, run_dir, man = train_command(seed_key)
        payload["commands"][seed_key] = man
        (OUT_DIR / f"train_command_seed{seed_key}.json").write_text(json.dumps(man, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (OUT_DIR / "strictsmall_retrain_preflight.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": payload["status"],
        "tokenizer_len": len(tok),
        "special_ids": tok.all_special_ids,
        "tokenizer_sha256": payload["tokenizer_hashes"].get("tokenizer.json"),
        "train100_sha_ok": hundred["sha256"] == EXPECTED_100M_SHA,
        "commands": [str(OUT_DIR / f"train_command_seed{k}.json") for k in SEEDS],
        "preflight": str(OUT_DIR / "strictsmall_retrain_preflight.json"),
    }, indent=2, ensure_ascii=False), flush=True)
    return payload


def launch_train(seed_key: str, force: bool = False) -> None:
    if seed_key not in SEEDS:
        raise KeyError(seed_key)
    pf = preflight()
    cmd, run_dir, man = train_command(seed_key)
    if run_dir.exists():
        if force:
            shutil.rmtree(run_dir)
        else:
            raise FileExistsError(f"run_dir exists; use --force-train to remove: {run_dir}")
    run_dir.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(SEEDS[seed_key]["gpu"])
    env["TOKENIZERS_PARALLELISM"] = "false"
    (OUT_DIR / f"launched_train_command_seed{seed_key}.json").write_text(json.dumps(man, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"event": "launch_train", "seed_key": seed_key, "run_dir": str(run_dir), "cuda": env["CUDA_VISIBLE_DEVICES"]}, ensure_ascii=False), flush=True)
    proc = subprocess.run(cmd, env=env, text=True)
    raise SystemExit(proc.returncode)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["tokenizer", "preflight", "train"], required=True)
    ap.add_argument("--seed-key", choices=sorted(SEEDS), default="43022")
    ap.add_argument("--force-tokenizer", action="store_true")
    ap.add_argument("--force-train", action="store_true")
    args = ap.parse_args()
    if args.mode == "tokenizer":
        train_tokenizer(force=args.force_tokenizer)
    elif args.mode == "preflight":
        preflight()
    elif args.mode == "train":
        launch_train(args.seed_key, force=args.force_train)


if __name__ == "__main__":
    main()
