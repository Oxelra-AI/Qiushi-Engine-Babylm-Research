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
OUT = ROOT / "data/pure_order_1m_training_summary.json"
LOG = (ROOT.parents[2] / 'research/notes/initial_model_studies/pure_order_1m_train.log')

ORDER_MODES = ["source_stage", "readability_interleave"]
SEEDS = [42, 43]
COMMON = [
    "--variant", "dense_untied_causal",
    "--max_word_exposure", "1000000",
    "--example_pool_words", "10000000",
    "--checkpoint_words", "1000000",
    "--lr_total_steps", "98",
    "--seq_length", "256",
    "--words_per_example", "160",
    "--batch_size", "64",
    "--n_layer", "4",
    "--n_embd", "256",
    "--n_head", "4",
    "--log_every", "50",
]


def run(cmd: list[str], env: dict[str, str], logf) -> subprocess.CompletedProcess:
    logf.write("\n$ " + " ".join(cmd) + "\n")
    logf.flush()
    p = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env)
    logf.write(p.stdout)
    logf.write(f"\n[returncode={p.returncode}]\n")
    logf.flush()
    print(p.stdout[-2000:], flush=True)
    if p.returncode != 0:
        raise SystemExit(p.returncode)
    return p


def load_manifest(run_id: str) -> dict:
    return json.loads((RUN_ROOT / run_id / "example_order_manifest.json").read_text(encoding="utf-8"))


def main() -> None:
    env = os.environ.copy()
    cache = ROOT / "training/hf_modules_cache"
    cache.mkdir(parents=True, exist_ok=True)
    env["HF_MODULES_CACHE"] = str(cache.resolve())
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    LOG.parent.mkdir(parents=True, exist_ok=True)
    summaries = []
    with LOG.open("a", encoding="utf-8") as logf:
        logf.write(f"\n=== research pure order 1M grid start {datetime.utcnow().isoformat()}Z ===\n")
        for seed in SEEDS:
            random_run = f"babylm_step26_ctrl_dense_pool10M_seed{seed}_1M"
            random_manifest = load_manifest(random_run)
            random_order = random_manifest["consumed_example_ids_in_order"]
            random_set = sorted(random_order)
            random_source = random_manifest["source_words_consumed"]
            for mode in ORDER_MODES:
                run_id = f"babylm_step30_order_{mode}_pool10M_seed{seed}_1M"
                cmd = [
                    sys.executable, str(TRAIN),
                    "--output_dir", str(RUN_ROOT / run_id),
                    "--order_mode", mode,
                    "--seed", str(seed),
                    "--shared_core_seed", str(seed),
                    "--extra_init_seed", str(1000 + seed),
                    "--train_rng_seed", str(2000 + seed),
                    *COMMON,
                ]
                run_training(cmd, output_dir=RUN_ROOT / run_id, timeout=1800, env=env, logf=logf)
                run_dir = RUN_ROOT / run_id
                vp = run(["python", str(VERIFY), str(run_dir)], env, logf)
                (run_dir / "checkpoint_verify.json").write_text(vp.stdout, encoding="utf-8")
                verify = json.loads(vp.stdout)
                metrics = json.loads((run_dir / "scientific_metrics.json").read_text(encoding="utf-8"))
                manifest = json.loads((run_dir / "example_order_manifest.json").read_text(encoding="utf-8"))
                root_ok = verify["loads"][0]["ok"]
                ckpt_ok = verify["loads"][1]["ok"]
                revision_ok = verify["revision_load"]["ok"]
                selected_before = manifest["selected_example_ids_before_order"]
                final_order = manifest["consumed_example_ids_in_order"]
                selected_matches_random_order = selected_before == random_order
                set_matches_random = sorted(final_order) == random_set
                source_matches_random = manifest["source_words_consumed"] == random_source
                final_order_differs = final_order != random_order
                assert root_ok and ckpt_ok and revision_ok, run_id
                assert metrics["word_exposure"] == 1_000_000, run_id
                assert metrics["example_pool_words_actual"] == 10_000_000, run_id
                assert metrics["lr_schedule_total_steps"] == 98, run_id
                assert metrics.get("shared_core_init_applied") is True, run_id
                assert selected_matches_random_order, (run_id, "selected IDs differ from random baseline")
                assert set_matches_random and source_matches_random, run_id
                assert final_order_differs, (run_id, "order did not change")
                summaries.append({
                    "run_id": run_id,
                    "seed": seed,
                    "order_mode": mode,
                    "random_baseline_run_id": random_run,
                    "parameter_count": metrics["parameter_count"],
                    "loss_first": metrics["loss_first"],
                    "loss_last": metrics["loss_last"],
                    "root_ok": root_ok,
                    "ckpt_ok": ckpt_ok,
                    "revision_ok": revision_ok,
                    "selected_matches_random_order": selected_matches_random_order,
                    "selected_set_matches_random": set_matches_random,
                    "source_mix_matches_random": source_matches_random,
                    "final_order_differs_from_random": final_order_differs,
                    "source_words": manifest["source_words_consumed"],
                    "first20_final_order": final_order[:20],
                    "first20_random_order": random_order[:20],
                    "order_info": manifest.get("order_info"),
                })
                print("TRAINED_ORDER", run_id, "loss", round(metrics["loss_first"], 4), "->", round(metrics["loss_last"], 4), "source_same", source_matches_random, "selected_same", selected_matches_random_order)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(summaries, indent=2, ensure_ascii=False), encoding="utf-8")
    print("WROTE", OUT)


if __name__ == "__main__":
    main()
