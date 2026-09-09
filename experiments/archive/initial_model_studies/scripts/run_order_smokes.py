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
MODES = ["random", "source_stage", "readability_interleave"]

COMMON = [
    "--variant", "dense_untied_causal",
    "--max_word_exposure", "10000",
    "--example_pool_words", "20000",
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
    "--log_every", "4",
]


def run(cmd: list[str], env: dict[str, str]) -> subprocess.CompletedProcess:
    print("$", " ".join(cmd), flush=True)
    p = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env)
    print(p.stdout, flush=True)
    if p.returncode != 0:
        raise SystemExit(p.returncode)
    return p


def main() -> None:
    env = os.environ.copy()
    cache = ROOT / "training/hf_modules_cache"
    cache.mkdir(parents=True, exist_ok=True)
    env["HF_MODULES_CACHE"] = str(cache.resolve())
    env.setdefault("TOKENIZERS_PARALLELISM", "false")

    manifests = {}
    for mode in MODES:
        run_id = f"babylm_step30_order_{mode}_pool20k_train10k_smoke"
        run_training([
            sys.executable, str(TRAIN),
            "--output_dir", str(RUN_ROOT / run_id),
            "--order_mode", mode,
            *COMMON,
        ], output_dir=RUN_ROOT / run_id, timeout=900, env=env)
        run_dir = RUN_ROOT / run_id
        vp = run(["python", str(VERIFY), str(run_dir)], env)
        (run_dir / "checkpoint_verify.json").write_text(vp.stdout, encoding="utf-8")
        d = json.loads(vp.stdout)
        m = json.loads((run_dir / "scientific_metrics.json").read_text())
        e = json.loads((run_dir / "example_order_manifest.json").read_text())
        root, ckpt = d["loads"]
        print("SMOKE", mode, "root", root["ok"], "ckpt", ckpt["ok"], "revision", d["revision_load"]["ok"], "params", m["parameter_count"], "order", m.get("order_mode"), "first", e["consumed_example_ids_in_order"][:10])
        assert root["ok"] and ckpt["ok"] and d["revision_load"]["ok"]
        assert m.get("order_mode") == mode
        assert e.get("order_mode") == mode
        assert m.get("example_pool_words_actual") == 20000
        assert m.get("lr_schedule_total_steps") == 16
        assert m.get("word_exposure") == 10000
        manifests[mode] = e

    selected = {mode: manifests[mode]["selected_example_ids_before_order"] for mode in MODES}
    sorted_ids = {mode: manifests[mode]["consumed_example_ids_sorted"] for mode in MODES}
    source_words = {mode: manifests[mode]["source_words_consumed"] for mode in MODES}
    final_order = {mode: manifests[mode]["consumed_example_ids_in_order"] for mode in MODES}
    assert selected["random"] == selected["source_stage"] == selected["readability_interleave"]
    assert sorted_ids["random"] == sorted_ids["source_stage"] == sorted_ids["readability_interleave"]
    assert source_words["random"] == source_words["source_stage"] == source_words["readability_interleave"]
    assert final_order["random"] != final_order["source_stage"]
    assert final_order["random"] != final_order["readability_interleave"]
    summary = {
        "modes": MODES,
        "selected_exact_equal": True,
        "sorted_set_equal": True,
        "source_mix_equal": True,
        "random_vs_source_order_differs": True,
        "random_vs_readability_order_differs": True,
        "source_words": source_words["random"],
        "first20": {mode: final_order[mode][:20] for mode in MODES},
        "order_info": {mode: manifests[mode].get("order_info") for mode in MODES},
    }
    out = ROOT / "data/order_smoke_summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print("ORDER_SMOKE_OK", json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
