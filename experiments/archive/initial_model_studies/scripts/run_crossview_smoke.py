#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import pathlib
import re
import subprocess

from training_process import run_training
import sys

from transformers import AutoModelForMaskedLM, AutoTokenizer

ROOT = pathlib.Path("experiments/archive/initial_model_studies")
TRAIN = ROOT / "training/scripts/babylm_crossview_mask_train.py"
STRICT_DIR = ROOT / "repos/babylm-eval/strict"
META = ROOT / "data/crossview_revision_64/crossview_materialization_revision_64.json"
RUN_ROOT = ROOT / "training/runs"
OUT = ROOT / "data/crossview_smoke_summary.json"
NOTE = (ROOT.parents[2] / 'research/notes/initial_model_studies/crossview_smoke.md')
SEED = 42
RUNS = {
    "adjacent": "babylm_crossview_adjacent_smoke_seed42",
    "shuffled": "babylm_crossview_shuffled_smoke_seed42",
}


def setup_env() -> dict[str, str]:
    env = os.environ.copy()
    hf_home = ROOT / "training/hf_home"
    env["HF_HOME"] = str(hf_home.resolve())
    env["HF_HUB_CACHE"] = str((hf_home / "hub").resolve())
    env["TRANSFORMERS_CACHE"] = str((hf_home / "transformers").resolve())
    env["HF_MODULES_CACHE"] = str((ROOT / "training/hf_modules_cache").resolve())
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    for key in ["HF_HOME", "HF_HUB_CACHE", "TRANSFORMERS_CACHE", "HF_MODULES_CACHE"]:
        pathlib.Path(env[key]).mkdir(parents=True, exist_ok=True)
    return env


def run(cmd: list[str], env: dict[str, str], cwd: pathlib.Path | None = None) -> subprocess.CompletedProcess:
    print("$", " ".join(cmd), flush=True)
    p = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env, cwd=str(cwd) if cwd else None)
    print(p.stdout[-5000:], flush=True)
    if p.returncode != 0:
        raise RuntimeError(f"command failed with {p.returncode}: {' '.join(cmd)}\n{p.stdout[-10000:]}")
    return p


def read_avg(report: pathlib.Path) -> float:
    txt = report.read_text(encoding="utf-8", errors="replace")
    m = re.search(r"AVERAGE ACCURACY\s*\n([0-9.\-]+)", txt)
    if not m:
        raise RuntimeError(f"could not parse {report}\n{txt[:500]}")
    return float(m.group(1))


def train_arm(arm: str, jsonl: str, words: int, env: dict[str, str]) -> dict:
    run_id = RUNS[arm]
    run_dir = RUN_ROOT / run_id
    if run_dir.exists():
        import shutil
        shutil.rmtree(run_dir)
    run_training([
        sys.executable, str(TRAIN),
        "--output_dir", str(run_dir),
        "--example_jsonl", jsonl,
        "--example_jsonl_label", f"crossview_smoke_{arm}",
        "--example_jsonl_meta", str(META),
        "--max_word_exposure", str(words),
        "--example_pool_words", str(words),
        "--checkpoint_words", str(words),
        "--tokenizer_label", "baseline16k",
        "--mask_prob", "0.15",
        "--seq_length", "256",
        "--max_seq_length", "256",
        "--max_position_embeddings", "512",
        "--batch_size", "8",
        "--lr_total_steps", "9",
        "--hidden_size", "64",
        "--n_layer", "1",
        "--n_head", "4",
        "--ffn_mult", "4",
        "--learning_rate", "0.001",
        "--seed", "42",
        "--extra_init_seed", "456",
        "--train_rng_seed", "789",
        "--log_every", "1",
    ], output_dir=run_dir, timeout=1200, env=env)
    tok = AutoTokenizer.from_pretrained(run_dir / "hf_model")
    model = AutoModelForMaskedLM.from_pretrained(run_dir / "hf_model")
    AutoTokenizer.from_pretrained(run_dir / "hf_model" / "chck_1M")
    AutoModelForMaskedLM.from_pretrained(run_dir / "hf_model" / "chck_1M")
    metrics = json.loads((run_dir / "scientific_metrics.json").read_text(encoding="utf-8"))
    summary = json.loads((run_dir / "crossview_supervision_summary.json").read_text(encoding="utf-8"))
    manifest = json.loads((run_dir / "crossview_supervision_manifest.json").read_text(encoding="utf-8"))
    return {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "model_class": model.__class__.__name__,
        "tokenizer_class": tok.__class__.__name__,
        "metrics": metrics,
        "supervision_summary": summary,
        "supervision_manifest": manifest,
    }


def compare_manifests(adj: list[dict], shuf: list[dict]) -> dict:
    if len(adj) != len(shuf):
        return {"ok": False, "reason": "length mismatch", "len_adj": len(adj), "len_shuf": len(shuf)}
    mismatches = []
    pair_count = 0
    for a, s in zip(adj, shuf):
        if a["kind"] == "pair_crossview" or s["kind"] == "pair_crossview":
            pair_count += 1
            fields = ["row_index", "example_id", "source_pair_id", "anchor_group_count", "anchor_token_count", "anchor_source_word_indices", "anchor_norms", "words"]
            bad = {f: (a.get(f), s.get(f)) for f in fields if a.get(f) != s.get(f)}
            if bad:
                mismatches.append({"row_index": a.get("row_index"), "bad": bad})
    return {"ok": len(mismatches) == 0, "pair_count": pair_count, "mismatches_first10": mismatches[:10]}


def eval_fast(run_dir: pathlib.Path, env: dict[str, str]) -> dict:
    outdir = (run_dir / "eval_results_smoke").resolve()
    model_path = (run_dir / "hf_model").resolve()
    scores = {}
    for task_name, task, data_path in [
        ("blimp_fast", "blimp", "evaluation_data/fast_eval/blimp_fast"),
        ("entity_tracking_fast", "entity_tracking", "evaluation_data/fast_eval/entity_tracking_fast"),
    ]:
        run([
            sys.executable, "-m", "evaluation_pipeline.sentence_zero_shot.run",
            "--model_path_or_name", str(model_path), "--backend", "mlm", "--task", task,
            "--data_path", data_path, "--save_predictions", "--revision_name", "chck_1M",
            "--batch_size", "64", "--output_dir", str(outdir),
        ], env, cwd=STRICT_DIR)
        report = outdir / "hf_model" / "chck_1M" / "zero_shot" / "mlm" / task / task_name / "best_temperature_report.txt"
        scores[task_name] = read_avg(report)
    return scores


def main() -> None:
    env = setup_env()
    meta = json.loads(META.read_text(encoding="utf-8"))
    sm = meta["seeds"][str(SEED)]
    adj_path = sm["smoke_adjacent_path"]
    shuf_path = sm["smoke_shuffled_path"]
    words = int(sm["smoke_adjacent_summary"]["total_words"])
    assert words == int(sm["smoke_shuffled_summary"]["total_words"])
    assert sm["per_pair_supervision_proxy_identical_smoke"] is True
    adj = train_arm("adjacent", adj_path, words, env)
    shuf = train_arm("shuffled", shuf_path, words, env)
    cmp = compare_manifests(adj["supervision_manifest"], shuf["supervision_manifest"])
    if not cmp["ok"]:
        raise RuntimeError(f"supervision manifest mismatch: {cmp}")
    eval_scores = {arm: eval_fast(pathlib.Path(row["run_dir"]), env) for arm, row in [("adjacent", adj), ("shuffled", shuf)]}
    payload = {
        "status": "CROSSVIEW_SMOKE_OK",
        "materialization_meta": str(META),
        "smoke_words": words,
        "materializer_seed_summary": sm,
        "adjacent": {k: v for k, v in adj.items() if k != "supervision_manifest"},
        "shuffled": {k: v for k, v in shuf.items() if k != "supervision_manifest"},
        "per_sample_supervision_comparison": cmp,
        "eval_scores": eval_scores,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    NOTE.parent.mkdir(parents=True, exist_ok=True)
    NOTE.write_text(
        "# research — cross-view masking smoke\n\n"
        f"Summary JSON: `{OUT}`\n\n"
        f"Smoke words: {words}. Adjacent and shuffled pair examples have identical per-sample anchor groups/tokens: {cmp['ok']} across {cmp['pair_count']} pair examples.\n\n"
        f"Adjacent eval smoke: BLiMP {eval_scores['adjacent']['blimp_fast']:.2f}, Entity {eval_scores['adjacent']['entity_tracking_fast']:.2f}.\n"
        f"Shuffled eval smoke: BLiMP {eval_scores['shuffled']['blimp_fast']:.2f}, Entity {eval_scores['shuffled']['entity_tracking_fast']:.2f}.\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": payload["status"], "summary": str(OUT), "comparison": cmp, "eval_scores": eval_scores}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
