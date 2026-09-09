#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import pathlib
import subprocess

from training_process import run_training
import sys
from datetime import datetime

ROOT = pathlib.Path("experiments/archive/initial_model_studies")
RUN_ROOT = ROOT / "training/runs"
LOG = (ROOT.parents[2] / 'research/notes/initial_model_studies/controlled_grid_train.log')
VERIFY = ROOT / "scripts/verify_babylm_checkpoint.py"
TRAIN = ROOT / "training/scripts/babylm_compare_train.py"

COMMON = [
    "--max_word_exposure", "1000000",
    "--checkpoint_words", "1000000",
    "--lr_total_steps", "98",
    "--seq_length", "256",
    "--words_per_example", "160",
    "--batch_size", "64",
    "--n_layer", "4",
    "--n_embd", "256",
    "--n_head", "4",
    "--memory_dim", "64",
    "--memory_dropout", "0.0",
    "--log_every", "50",
]

CELLS = []
for pool_name, pool_words in [("pool1M", 1_000_000), ("pool10M", 10_000_000)]:
    for seed in [42, 43]:
        for variant, short in [("dense_untied_causal", "dense"), ("memory_causal", "memory")]:
            run_id = f"babylm_step26_ctrl_{short}_{pool_name}_seed{seed}_1M"
            CELLS.append({"pool_name": pool_name, "pool_words": pool_words, "seed": seed, "variant": variant, "short": short, "run_id": run_id})


def run_cmd(cmd: list[str], env: dict[str, str], logf) -> subprocess.CompletedProcess:
    logf.write("\n$ " + " ".join(cmd) + "\n")
    logf.flush()
    p = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env)
    logf.write(p.stdout)
    logf.write(f"\n[returncode={p.returncode}]\n")
    logf.flush()
    if p.returncode != 0:
        print(p.stdout[-4000:])
        raise SystemExit(p.returncode)
    return p


def main() -> None:
    LOG.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    cache = ROOT / "training/hf_modules_cache"
    cache.mkdir(parents=True, exist_ok=True)
    env["HF_MODULES_CACHE"] = str(cache.resolve())
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    summaries = []
    with LOG.open("a", encoding="utf-8") as logf:
        logf.write(f"\n=== research controlled grid start {datetime.utcnow().isoformat()}Z ===\n")
        for cell in CELLS:
            run_id = cell["run_id"]
            run_dir = RUN_ROOT / run_id
            cmd = [
                sys.executable, str(TRAIN),
                "--output_dir", str(RUN_ROOT / run_id),
                "--variant", cell["variant"],
                "--example_pool_words", str(cell["pool_words"]),
                "--seed", str(cell["seed"]),
                "--shared_core_seed", str(cell["seed"]),
                "--extra_init_seed", str(1000 + cell["seed"]),
                "--train_rng_seed", str(2000 + cell["seed"]),
                *COMMON,
            ]
            run_training(cmd, output_dir=RUN_ROOT / run_id, timeout=1800, env=env, logf=logf)
            verify_cmd = ["python", str(VERIFY), str(run_dir)]
            vp = run_cmd(verify_cmd, env, logf)
            (run_dir / "checkpoint_verify.json").write_text(vp.stdout, encoding="utf-8")
            d = json.loads(vp.stdout)
            m = json.loads((run_dir / "scientific_metrics.json").read_text())
            e = json.loads((run_dir / "example_order_manifest.json").read_text())
            assert all(x["ok"] for x in d["loads"]), run_id
            assert d["revision_load"]["ok"], run_id
            assert m["word_exposure"] == 1_000_000, run_id
            assert m["lr_schedule_total_steps"] == 98, run_id
            assert m["example_pool_words_actual"] == cell["pool_words"], run_id
            assert m.get("shared_core_init_applied") is True, run_id
            summaries.append({
                **cell,
                "parameter_count": m["parameter_count"],
                "loss_first": m["loss_first"],
                "loss_last": m["loss_last"],
                "example_ids": e["consumed_example_ids_in_order"],
                "source_words": e["source_words_consumed"],
                "verify_root_ok": d["loads"][0]["ok"],
                "verify_ckpt_ok": d["loads"][1]["ok"],
                "verify_revision_ok": d["revision_load"]["ok"],
                "memory_summary_last": m.get("memory_summary_last"),
            })
            print("TRAINED", run_id, "params", m["parameter_count"], "loss", round(m["loss_first"],4), "->", round(m["loss_last"],4), "pool", cell["pool_words"])
        # Paired order checks: dense and memory must see identical example IDs for each pool/seed.
        for pool_name in ["pool1M", "pool10M"]:
            for seed in [42, 43]:
                dense = next(s for s in summaries if s["pool_name"] == pool_name and s["seed"] == seed and s["short"] == "dense")
                memory = next(s for s in summaries if s["pool_name"] == pool_name and s["seed"] == seed and s["short"] == "memory")
                assert dense["example_ids"] == memory["example_ids"], (pool_name, seed)
                print("ORDER_PAIR_OK", pool_name, seed, "n", len(dense["example_ids"]), "first", dense["example_ids"][:8])
        out = ROOT / "data/controlled_grid_training_summary.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        serializable = []
        for s in summaries:
            t = dict(s)
            t["example_ids_sha_preview"] = t["example_ids"][:20]
            t.pop("example_ids")
            serializable.append(t)
        out.write_text(json.dumps(serializable, indent=2, ensure_ascii=False), encoding="utf-8")
        print("WROTE", out)


if __name__ == "__main__":
    main()
