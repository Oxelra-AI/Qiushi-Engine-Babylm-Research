#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys

from training_process import run_training

ROOT = pathlib.Path("experiments/archive/initial_model_studies")
TRAIN = ROOT / "training/scripts/babylm_compare_train.py"
VERIFY = ROOT / "scripts/verify_babylm_checkpoint.py"
RUN_ROOT = ROOT / "training/runs"
OUT = ROOT / "data/surface_smoke_summary.json"

SPECS = [
    ("dense_untied_causal", "babylm_dense_untied_surface_smoke"),
    ("char_surface_causal", "babylm_char_surface_smoke"),
    ("lookup_adapter_causal", "babylm_lookup_adapter_smoke"),
]

COMMON = [
    "--max_word_exposure", "10000",
    "--example_pool_words", "20000",
    "--order_mode", "random",
    "--checkpoint_words", "10000",
    "--lr_total_steps", "16",
    "--shared_core_seed", "123",
    "--extra_init_seed", "456",
    "--train_rng_seed", "789",
    "--seed", "42",
    "--seq_length", "128",
    "--words_per_example", "80",
    "--batch_size", "16",
    "--n_layer", "2",
    "--n_embd", "128",
    "--n_head", "4",
    "--surface_dim", "64",
    "--ngram_vocab_size", "1024",
    "--max_ngrams_per_token", "16",
    "--lookup_rank", "5",
    "--log_every", "4",
]


def run(cmd: list[str], env: dict[str, str]) -> subprocess.CompletedProcess:
    print("$", " ".join(cmd), flush=True)
    p = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env)
    print(p.stdout[-5000:], flush=True)
    if p.returncode != 0:
        raise SystemExit(p.returncode)
    return p


def main() -> None:
    env = os.environ.copy()
    cache = ROOT / "training/hf_modules_cache"
    cache.mkdir(parents=True, exist_ok=True)
    env["HF_MODULES_CACHE"] = str(cache.resolve())
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    rows = []
    manifests = {}
    for variant, run_id in SPECS:
        run_training([
            sys.executable, str(TRAIN),
            "--output_dir", str(RUN_ROOT / run_id),
            "--variant", variant,
            *COMMON,
        ], output_dir=RUN_ROOT / run_id, timeout=900, env=env)
        run_dir = RUN_ROOT / run_id
        vp = run(["python", str(VERIFY), str(run_dir)], env)
        (run_dir / "checkpoint_verify.json").write_text(vp.stdout, encoding="utf-8")
        verify = json.loads(vp.stdout)
        metrics = json.loads((run_dir / "scientific_metrics.json").read_text(encoding="utf-8"))
        manifest = json.loads((run_dir / "example_order_manifest.json").read_text(encoding="utf-8"))
        root_ok, ckpt_ok = verify["loads"][0]["ok"], verify["loads"][1]["ok"]
        revision_ok = verify["revision_load"]["ok"]
        assert root_ok and ckpt_ok and revision_ok, run_id
        assert metrics["word_exposure"] == 10000, run_id
        assert metrics["example_pool_words_actual"] == 20000, run_id
        assert metrics["lr_schedule_total_steps"] == 16, run_id
        assert metrics.get("shared_core_init_applied") is True, run_id
        if variant == "char_surface_causal":
            assert metrics.get("surface_mode") == "char_ngram"
            assert metrics.get("surface_summary_last") is not None
            assert metrics["surface_summary_last"].get("surface_mode_is_char_ngram") == 1.0
            assert (run_dir / "surface_char_ngram_manifest.json").exists()
        if variant == "lookup_adapter_causal":
            assert metrics.get("surface_mode") == "lookup"
            assert metrics.get("surface_summary_last") is not None
            assert metrics["surface_summary_last"].get("surface_mode_is_char_ngram") == 0.0
            assert (run_dir / "surface_lookup_manifest.json").exists()
        manifests[variant] = manifest
        rows.append({
            "variant": variant,
            "run_id": run_id,
            "parameter_count": metrics["parameter_count"],
            "loss_first": metrics["loss_first"],
            "loss_last": metrics["loss_last"],
            "surface_mode": metrics.get("surface_mode"),
            "surface_summary_last": metrics.get("surface_summary_last"),
            "shared_core_target": metrics.get("shared_core_target"),
            "root_ok": root_ok,
            "ckpt_ok": ckpt_ok,
            "revision_ok": revision_ok,
            "first12_examples": manifest["consumed_example_ids_in_order"][:12],
            "source_words": manifest["source_words_consumed"],
        })
        print("SMOKE_OK", variant, "params", metrics["parameter_count"], "loss", metrics["loss_first"], "->", metrics["loss_last"], "surface", metrics.get("surface_summary_last"))
    base_order = manifests["dense_untied_causal"]["consumed_example_ids_in_order"]
    base_selected = manifests["dense_untied_causal"]["selected_example_ids_before_order"]
    base_source = manifests["dense_untied_causal"]["source_words_consumed"]
    for variant in ["char_surface_causal", "lookup_adapter_causal"]:
        assert manifests[variant]["consumed_example_ids_in_order"] == base_order, variant
        assert manifests[variant]["selected_example_ids_before_order"] == base_selected, variant
        assert manifests[variant]["source_words_consumed"] == base_source, variant
    summary = {
        "status": "SURFACE_SMOKE_OK",
        "shared_pool_order_identical": True,
        "rows": rows,
        "parameter_deltas_vs_dense": {
            r["variant"]: r["parameter_count"] - rows[0]["parameter_count"] for r in rows
        },
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("WROTE", OUT)


if __name__ == "__main__":
    main()
