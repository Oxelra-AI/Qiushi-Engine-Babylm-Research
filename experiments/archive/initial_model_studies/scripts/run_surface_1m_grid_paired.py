#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys

from training_process import run_training
from datetime import datetime

ROOT = pathlib.Path("experiments/archive/initial_model_studies")
TRAIN = ROOT / "training/scripts/babylm_compare_train.py"
VERIFY = ROOT / "scripts/verify_babylm_checkpoint.py"
RUN_ROOT = ROOT / "training/runs"
OUT = ROOT / "data/surface_1m_training_summary.json"
LOG = (ROOT.parents[2] / 'research/notes/initial_model_studies/surface_1m_train.log')

SPECS = [
    ("char_surface_causal", "char_surface"),
    ("lookup_adapter_causal", "lookup_adapter"),
]
SEEDS = [42, 43]
COMMON = [
    "--max_word_exposure", "1000000",
    "--example_pool_words", "10000000",
    "--order_mode", "random",
    "--checkpoint_words", "1000000",
    "--lr_total_steps", "98",
    "--seq_length", "256",
    "--words_per_example", "160",
    "--batch_size", "64",
    "--n_layer", "4",
    "--n_embd", "256",
    "--n_head", "4",
    "--surface_dim", "64",
    "--ngram_vocab_size", "1024",
    "--max_ngrams_per_token", "16",
    "--lookup_rank", "5",
    "--log_every", "50",
]


def run(cmd: list[str], env: dict[str, str], logf) -> subprocess.CompletedProcess:
    logf.write("\n$ " + " ".join(cmd) + "\n")
    logf.flush()
    p = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env)
    logf.write(p.stdout)
    logf.write(f"\n[returncode={p.returncode}]\n")
    logf.flush()
    print(p.stdout[-3000:], flush=True)
    if p.returncode != 0:
        raise SystemExit(p.returncode)
    return p


def load_json(path: pathlib.Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def args_map_from_config(run_dir: pathlib.Path) -> dict[str, str]:
    cfg = load_json(run_dir / "config.json")
    args = cfg.get("script_args", [])
    out: dict[str, str] = {}
    i = 0
    while i < len(args):
        if isinstance(args[i], str) and args[i].startswith("--") and i + 1 < len(args) and not str(args[i + 1]).startswith("--"):
            out[args[i][2:]] = str(args[i + 1])
            i += 2
        else:
            i += 1
    return out


def main() -> None:
    env = os.environ.copy()
    cache = ROOT / "training/hf_modules_cache"
    cache.mkdir(parents=True, exist_ok=True)
    env["HF_MODULES_CACHE"] = str(cache.resolve())
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    LOG.parent.mkdir(parents=True, exist_ok=True)
    summaries = []
    with LOG.open("a", encoding="utf-8") as logf:
        logf.write(f"\n=== research surface 1M paired grid start {datetime.utcnow().isoformat()}Z ===\n")
        for seed in SEEDS:
            baseline_run = f"babylm_step26_ctrl_dense_pool10M_seed{seed}_1M"
            baseline_dir = RUN_ROOT / baseline_run
            baseline_manifest = load_json(baseline_dir / "example_order_manifest.json")
            baseline_metrics = load_json(baseline_dir / "scientific_metrics.json")
            baseline_args = args_map_from_config(baseline_dir)
            baseline_order = baseline_manifest["consumed_example_ids_in_order"]
            baseline_selected = baseline_manifest.get("selected_example_ids_before_order", baseline_order)
            baseline_source = baseline_manifest["source_words_consumed"]
            required = {
                "seed": str(seed),
                "shared_core_seed": baseline_args.get("shared_core_seed"),
                "extra_init_seed": baseline_args.get("extra_init_seed"),
                "train_rng_seed": baseline_args.get("train_rng_seed"),
            }
            assert required["shared_core_seed"] == str(baseline_metrics["shared_core_seed"]), required
            assert required["extra_init_seed"] is not None and required["train_rng_seed"] is not None, required
            for variant, short in SPECS:
                run_id = f"babylm_step37_{short}_pool10M_seed{seed}_1M"
                cmd = [
                    sys.executable, str(TRAIN),
                    "--output_dir", str(RUN_ROOT / run_id),
                    "--variant", variant,
                    "--seed", required["seed"],
                    "--shared_core_seed", required["shared_core_seed"],
                    "--extra_init_seed", required["extra_init_seed"],
                    "--train_rng_seed", required["train_rng_seed"],
                    *COMMON,
                ]
                run_training(cmd, output_dir=RUN_ROOT / run_id, timeout=1800, env=env, logf=logf)
                run_dir = RUN_ROOT / run_id
                vp = run(["python", str(VERIFY), str(run_dir)], env, logf)
                (run_dir / "checkpoint_verify.json").write_text(vp.stdout, encoding="utf-8")
                verify = json.loads(vp.stdout)
                metrics = load_json(run_dir / "scientific_metrics.json")
                manifest = load_json(run_dir / "example_order_manifest.json")
                root_ok = verify["loads"][0]["ok"]
                ckpt_ok = verify["loads"][1]["ok"]
                revision_ok = verify["revision_load"]["ok"]
                assert root_ok and ckpt_ok and revision_ok, run_id
                assert metrics["word_exposure"] == 1_000_000, run_id
                assert metrics["example_pool_words_actual"] == 10_000_000, run_id
                assert metrics["lr_schedule_total_steps"] == 98, run_id
                assert metrics.get("shared_core_init_applied") is True, run_id
                assert str(metrics.get("shared_core_seed")) == required["shared_core_seed"], run_id
                assert manifest["selected_example_ids_before_order"] == baseline_selected, (run_id, "selected IDs differ")
                assert manifest["consumed_example_ids_in_order"] == baseline_order, (run_id, "order differs")
                assert manifest["source_words_consumed"] == baseline_source, (run_id, "source mix differs")
                if variant == "char_surface_causal":
                    assert metrics.get("surface_mode") == "char_ngram", run_id
                    assert metrics.get("surface_summary_last", {}).get("surface_mode_is_char_ngram") == 1.0, run_id
                    assert (run_dir / "surface_char_ngram_manifest.json").exists(), run_id
                if variant == "lookup_adapter_causal":
                    assert metrics.get("surface_mode") == "lookup", run_id
                    assert metrics.get("surface_summary_last", {}).get("surface_mode_is_char_ngram") == 0.0, run_id
                    assert (run_dir / "surface_lookup_manifest.json").exists(), run_id
                summaries.append({
                    "run_id": run_id,
                    "variant": variant,
                    "seed": seed,
                    "baseline_run_id": baseline_run,
                    "baseline_pairing_args": required,
                    "parameter_count": metrics["parameter_count"],
                    "loss_first": metrics["loss_first"],
                    "loss_last": metrics["loss_last"],
                    "surface_mode": metrics.get("surface_mode"),
                    "surface_summary_last": metrics.get("surface_summary_last"),
                    "shared_core_target": metrics.get("shared_core_target"),
                    "root_ok": root_ok,
                    "ckpt_ok": ckpt_ok,
                    "revision_ok": revision_ok,
                    "selected_order_matches_baseline": True,
                    "source_mix_matches_baseline": True,
                    "source_words": manifest["source_words_consumed"],
                    "surface_manifest_file": "surface_char_ngram_manifest.json" if variant == "char_surface_causal" else "surface_lookup_manifest.json",
                })
                print("TRAINED_SURFACE_PAIRED", run_id, "paired", required, "params", metrics["parameter_count"], "loss", round(metrics["loss_first"], 4), "->", round(metrics["loss_last"], 4), "surface", metrics.get("surface_summary_last"))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(summaries, indent=2, ensure_ascii=False), encoding="utf-8")
    print("WROTE", OUT)


if __name__ == "__main__":
    main()
