#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import json
import os
import pathlib
import re
import subprocess

from training_process import run_training
import sys

from transformers import AutoModelForMaskedLM, AutoTokenizer

ROOT = pathlib.Path("experiments/archive/initial_model_studies")
TRAIN = ROOT / "training/scripts/babylm_masked_train.py"
STRICT_DIR = ROOT / "repos/babylm-eval/strict"
RUN_ROOT = ROOT / "training/runs"
TOK40K = ROOT / "training/tokenizers/official40k"
OUT_JSON = ROOT / "data/wwm_official40k_1m_profile.json"
OUT_TRAIN_JSON = ROOT / "data/wwm_official40k_1m_training_summary.json"
OUT_NOTE = (ROOT.parents[2] / 'research/notes/initial_model_studies/wwm_official40k_1m_profile.md')
LOG = (ROOT.parents[2] / 'research/notes/initial_model_studies/wwm_official40k_1m_train_and_profile.log')

SEED_SPECS = [
    {
        "seed": 42,
        "extra_init_seed": 456,
        "train_rng_seed": 789,
        "run_id": "babylm_masked_wwm_official40k_seed42_1M",
        "baseline_run_id": "babylm_masked_wwm_pos512_1M",
        "baseline_profile": ROOT / "data/masked_1m_grid_pos512_profile.json",
    },
    {
        "seed": 43,
        "extra_init_seed": 457,
        "train_rng_seed": 790,
        "run_id": "babylm_masked_wwm_official40k_seed43_1M",
        "baseline_run_id": "babylm_masked_wwm_pos512_seed43_1M",
        "baseline_profile": ROOT / "data/masked_1m_seed43_profile.json",
    },
]

TASKS = [
    ("blimp_fast", "blimp", "evaluation_data/fast_eval/blimp_fast"),
    ("supplement_fast", "blimp", "evaluation_data/fast_eval/supplement_fast"),
    ("ewok_fast", "ewok", "evaluation_data/fast_eval/evaluation_data/fast_eval/ewok_fast"),
    ("entity_tracking_fast", "entity_tracking", "evaluation_data/fast_eval/entity_tracking_fast"),
    ("comps", "comps", "evaluation_data/full_eval/comps"),
]
COLS = ["blimp_fast", "supplement_fast", "ewok_fast", "entity_tracking_fast", "comps", "reading_eye_tracking", "reading_self_paced"]
COMMON = [
    "--max_word_exposure", "1000000",
    "--example_pool_words", "10000000",
    "--checkpoint_words", "1000000",
    "--words_per_example", "160",
    "--tokenizer_path", str(TOK40K),
    "--tokenizer_label", "official40k_bytelevel_bpe",
    "--tokenization_summary_limit", "0",
    "--mask_mode", "wwm",
    "--mask_prob", "0.15",
    "--seq_length", "256",
    "--max_seq_length", "256",
    "--max_position_embeddings", "512",
    "--batch_size", "64",
    "--lr_total_steps", "98",
    "--hidden_size", "256",
    "--n_layer", "8",
    "--n_head", "8",
    "--learning_rate", "0.001",
    "--log_every", "50",
]


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
    # In-process tokenizer loads and dataset downloads in this runner must also
    # Use writable local caches, not the read-only global HF cache.
    os.environ.update({k: env[k] for k in ["HF_HOME", "HF_HUB_CACHE", "TRANSFORMERS_CACHE", "HF_MODULES_CACHE", "TOKENIZERS_PARALLELISM"] if k in env})
    return env


def run(cmd: list[str], env: dict[str, str], cwd: pathlib.Path | None = None, logf=None) -> subprocess.CompletedProcess:
    line = "$ " + " ".join(cmd)
    print(line, flush=True)
    if logf:
        logf.write("\n" + line + "\n"); logf.flush()
    p = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env, cwd=str(cwd) if cwd else None)
    print(p.stdout[-6000:], flush=True)
    if logf:
        logf.write(p.stdout + f"\n[returncode={p.returncode}]\n"); logf.flush()
    if p.returncode != 0:
        raise RuntimeError(f"command failed with {p.returncode}: {' '.join(cmd)}\n{p.stdout[-12000:]}")
    return p


def load_json(path: pathlib.Path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_avg(report: pathlib.Path) -> float:
    txt = report.read_text(encoding="utf-8", errors="replace")
    m = re.search(r"AVERAGE ACCURACY\s*\n([0-9.\-]+)", txt)
    if not m:
        raise RuntimeError(f"Could not parse {report}\n{txt[:400]}")
    return float(m.group(1))


def read_reading(report: pathlib.Path) -> dict[str, float]:
    txt = report.read_text(encoding="utf-8", errors="replace")
    out = {}
    for label, key in [("EYE TRACKING SCORE", "reading_eye_tracking"), ("SELF-PACED READING SCORE", "reading_self_paced")]:
        m = re.search(re.escape(label) + r":\s*([0-9.\-]+)", txt)
        if not m:
            raise RuntimeError(f"Could not parse {label} from {report}\n{txt}")
        out[key] = float(m.group(1))
    return out


def baseline_scores(profile_path: pathlib.Path, baseline_run_id: str) -> dict[str, float]:
    data = load_json(profile_path)
    for row in data["profile_rows"]:
        if row["run_id"] == baseline_run_id:
            return row["scores"]
    raise KeyError((profile_path, baseline_run_id))


def load_trainer_module():
    spec = importlib.util.spec_from_file_location("babylm_masked_train_mod", TRAIN)
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def reconstruct_examples_for_summary(seed: int, max_words: int = 1_000_000, pool_words: int = 10_000_000):
    mod = load_trainer_module()
    class Args:
        dataset_id = "BabyLM-community/BabyLM-2026-Strict-Small"
        dataset_revision = "c92ab16b4f08858304b0815706065b3354d8fc0a"
    raw_dir, _ = mod.download_dataset(Args, ROOT / "data/reconstruct_tmp")
    files = [raw_dir / n for n in mod.TRAIN_FILES]
    pool_examples = list(mod.iter_examples(files, pool_words, 160))
    for i, ex in enumerate(pool_examples):
        ex.example_id = i
    import random
    rng = random.Random(seed)
    rng.shuffle(pool_examples)
    examples = []
    actual = 0
    for ex in pool_examples:
        if actual >= max_words:
            break
        if actual + ex.words <= max_words:
            examples.append(ex)
            actual += ex.words
        else:
            take = max_words - actual
            if take > 0:
                examples.append(mod.Example(" ".join(ex.text.split()[:take]), take, example_id=ex.example_id, source=ex.source))
                actual += take
            break
    assert actual == max_words
    return mod, examples


def compute_baseline16k_coupling(seed: int, baseline_run_id: str) -> dict:
    mod, examples = reconstruct_examples_for_summary(seed)
    # Load the exact portable 16k tokenizer from the matched local baseline run
    # instead of querying the remote baseline repo. This avoids read-only global
    # HF cache writes and makes the comparison tied to the actual baseline artifact.
    baseline_tok_path = RUN_ROOT / baseline_run_id / "hf_model"
    tok = mod.make_portable_tokenizer(str(baseline_tok_path))
    return mod.summarize_tokenization_coupling(examples, tok, 256, 0)


def train_and_check(spec: dict, env: dict[str, str], logf) -> dict:
    run_id = spec["run_id"]
    run_dir = RUN_ROOT / run_id
    cmd = [
        sys.executable, str(TRAIN),
        "--output_dir", str(run_dir),
        "--seed", str(spec["seed"]), "--extra_init_seed", str(spec["extra_init_seed"]), "--train_rng_seed", str(spec["train_rng_seed"]),
        *COMMON,
    ]
    metrics_path = run_dir / "scientific_metrics.json"
    if metrics_path.exists() and (run_dir / "hf_model" / "config.json").exists() and (run_dir / "hf_model" / "chck_1M" / "config.json").exists():
        print(f"REUSE_COMPLETED_TRAINING {run_id}", flush=True)
        if logf:
            logf.write(f"\nREUSE_COMPLETED_TRAINING {run_id}\n"); logf.flush()
    else:
        run_training(cmd, output_dir=run_dir, timeout=1800, env=env, logf=logf)
    checks = []
    for rel in ["hf_model", "hf_model/chck_1M"]:
        p = run_dir / rel
        tok = AutoTokenizer.from_pretrained(p)
        model = AutoModelForMaskedLM.from_pretrained(p)
        checks.append({
            "rel": rel,
            "ok": True,
            "model_class": model.__class__.__name__,
            "tokenizer_class": tok.__class__.__name__,
            "vocab_size": len(tok),
            "params": sum(x.numel() for x in model.parameters()),
            "max_position_embeddings": int(model.config.max_position_embeddings),
        })
        assert len(tok) == 40000
        assert int(model.config.max_position_embeddings) >= 512
    metrics = load_json(run_dir / "scientific_metrics.json")
    manifest = load_json(run_dir / "example_order_manifest.json")
    baseline_manifest = load_json(RUN_ROOT / spec["baseline_run_id"] / "example_order_manifest.json")
    assert metrics["word_exposure"] == 1_000_000
    assert metrics["mask_mode"] == "wwm"
    assert metrics["vocab_size"] == 40000
    assert metrics["tokenizer_label"] == "official40k_bytelevel_bpe"
    assert metrics["max_seq_length"] == 256 and metrics["max_position_embeddings"] >= 512
    assert manifest["consumed_example_ids_in_order"] == baseline_manifest["consumed_example_ids_in_order"]
    assert manifest["source_words_consumed"] == baseline_manifest["source_words_consumed"]
    coupling40 = load_json(run_dir / "tokenization_coupling_summary.json")
    coupling16 = compute_baseline16k_coupling(spec["seed"], spec["baseline_run_id"])
    assert coupling40["total_words_summarized"] == coupling16["total_words_summarized"] == 1_000_000
    return {"run_id": run_id, "seed": spec["seed"], "baseline_run_id": spec["baseline_run_id"], "metrics": metrics, "manifest": manifest, "load_checks": checks, "coupling40k": coupling40, "coupling16k_recomputed": coupling16}


def profile(run_id: str, env: dict[str, str], logf) -> dict:
    run_dir = RUN_ROOT / run_id
    model_path = (run_dir / "hf_model").resolve()
    outdir = (run_dir / "eval_results_wwm_official40k_1m").resolve()
    scores = {}
    reports = {}
    for task_name, task, data_path in TASKS:
        run([
            sys.executable, "-m", "evaluation_pipeline.sentence_zero_shot.run",
            "--model_path_or_name", str(model_path), "--backend", "mlm", "--task", task,
            "--data_path", data_path, "--save_predictions", "--revision_name", "chck_1M",
            "--batch_size", "64", "--output_dir", str(outdir),
        ], env, cwd=STRICT_DIR, logf=logf)
        report = outdir / "hf_model" / "chck_1M" / "zero_shot" / "mlm" / task / task_name / "best_temperature_report.txt"
        assert report.exists(), report
        scores[task_name] = read_avg(report)
        reports[task_name] = str(report)
    run([
        sys.executable, "-m", "evaluation_pipeline.reading.run",
        "--model_path_or_name", str(model_path), "--backend", "mlm",
        "--data_path", "evaluation_data/fast_eval/reading/reading_data.csv",
        "--revision_name", "chck_1M", "--output_dir", str(outdir),
    ], env, cwd=STRICT_DIR, logf=logf)
    rreport = outdir / "hf_model" / "chck_1M" / "zero_shot" / "mlm" / "reading" / "report.txt"
    assert rreport.exists(), rreport
    scores.update(read_reading(rreport))
    reports["reading"] = str(rreport)
    return {"run_id": run_id, "scores": scores, "reports": reports}


def diff(a: float, b: float) -> float:
    return round(a - b, 4)


def pct(a: float, b: float) -> float:
    return round(100.0 * (a - b) / b, 4) if b else 0.0


def fmt(x):
    return f"{x:.2f}" if isinstance(x, float) else str(x)


def coupling_delta(c40: dict, c16: dict) -> dict:
    keys = [
        "untruncated_tokens_per_whitespace_word",
        "kept_tokens_per_whitespace_word",
        "word_groups_kept_per_whitespace_word",
        "truncated_example_fraction",
        "total_tokens_lost_to_truncation",
        "max_untruncated_tokens",
        "total_untruncated_tokens",
        "total_kept_tokens_at_max_seq_length",
        "expected_wwm_predicted_tokens_at_mask_prob_0p15_if_selected_groups_cover_tokens",
    ]
    out = {}
    for k in keys:
        out[k + "_16k"] = c16[k]
        out[k + "_40k"] = c40[k]
        if isinstance(c16[k], (int, float)) and isinstance(c40[k], (int, float)):
            out[k + "_delta"] = c40[k] - c16[k]
            out[k + "_pct_change"] = pct(c40[k], c16[k])
    return out


def main() -> None:
    env = setup_env()
    LOG.parent.mkdir(parents=True, exist_ok=True)
    train_rows = []
    profile_rows = []
    with LOG.open("a", encoding="utf-8") as logf:
        for spec in SEED_SPECS:
            train_rows.append(train_and_check(spec, env, logf))
        for row in train_rows:
            profile_rows.append(profile(row["run_id"], env, logf))

    train_summary = []
    rows = []
    deltas = {}
    coupling_rows = []
    for spec, tr, pr in zip(SEED_SPECS, train_rows, profile_rows):
        m = tr["metrics"]
        bscore = baseline_scores(spec["baseline_profile"], spec["baseline_run_id"])
        d = {c: diff(pr["scores"][c], bscore[c]) for c in COLS}
        deltas[str(spec["seed"])] = d
        cd = coupling_delta(tr["coupling40k"], tr["coupling16k_recomputed"])
        coupling_rows.append({"seed": spec["seed"], "run_id": tr["run_id"], "baseline_run_id": spec["baseline_run_id"], **cd})
        train_summary.append({
            "run_id": tr["run_id"], "seed": spec["seed"], "baseline_run_id": spec["baseline_run_id"],
            "parameter_count": m["parameter_count"], "embedding_parameter_count": m["embedding_parameter_count"], "vocab_size": m["vocab_size"],
            "loss_first": m["loss_first"], "loss_last": m["loss_last"], "masked_tokens_per_whitespace_word": m["masked_tokens_per_whitespace_word"],
            "word_exposure": m["word_exposure"], "steps": m["actual_training_steps"], "tokenizer_label": m["tokenizer_label"],
            "source_words": tr["manifest"]["source_words_consumed"], "first12_examples": tr["manifest"]["consumed_example_ids_in_order"][:12],
            "load_checks": tr["load_checks"], "coupling40k": tr["coupling40k"], "coupling16k_recomputed": tr["coupling16k_recomputed"],
        })
        rows.append({
            "run_id": tr["run_id"], "seed": spec["seed"], "baseline_run_id": spec["baseline_run_id"],
            "scores": pr["scores"], "baseline_scores_16k_wwm": bscore, "official40k_minus_baseline16k_wwm": d, "reports": pr["reports"],
        })
    mean_delta = {c: round(sum(deltas[str(s["seed"])][c] for s in SEED_SPECS) / len(SEED_SPECS), 4) for c in COLS}
    payload = {"train_summary": train_summary, "profile_rows": rows, "official40k_minus_baseline16k_wwm_by_seed": deltas, "mean_official40k_minus_baseline16k_wwm": mean_delta, "coupling_rows": coupling_rows}
    OUT_TRAIN_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_TRAIN_JSON.write_text(json.dumps(train_summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    OUT_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    lines = [
        "# research — WWM fixed base: official-corpus 40k tokenizer vs baseline 16k at 1M",
        "",
        f"Evidence JSON: `{OUT_JSON}`",
        "",
        "Only the tokenizer representation changes. The raw official-corpus examples/order/source mix, WWM objective, fixed length 256, model depth/width/heads, optimizer schedule, word exposure, and seeds are matched to each seed's existing WWM 16k baseline. Vocabulary size changes embedding capacity and tokenization/WWM density; those linked changes are recorded below.",
        "",
        "| seed | tok | BLiMP | Supp | EWoK | Entity | COMPS | Read eye | Read SPR | loss_last | params | emb params | mask tok/word |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row, tr in zip(rows, train_summary):
        s = row["scores"]
        lines.append(f"| {row['seed']} | 40k | {fmt(s['blimp_fast'])} | {fmt(s['supplement_fast'])} | {fmt(s['ewok_fast'])} | {fmt(s['entity_tracking_fast'])} | {fmt(s['comps'])} | {fmt(s['reading_eye_tracking'])} | {fmt(s['reading_self_paced'])} | {fmt(tr['loss_last'])} | {tr['parameter_count']} | {tr['embedding_parameter_count']} | {tr['masked_tokens_per_whitespace_word']:.4f} |")
    lines += ["", "## 40k minus matched 16k WWM baseline", "", "| seed | BLiMP | Supp | EWoK | Entity | COMPS | Read eye | Read SPR |", "|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for spec in SEED_SPECS:
        d = deltas[str(spec["seed"])]
        lines.append(f"| {spec['seed']} | {fmt(d['blimp_fast'])} | {fmt(d['supplement_fast'])} | {fmt(d['ewok_fast'])} | {fmt(d['entity_tracking_fast'])} | {fmt(d['comps'])} | {fmt(d['reading_eye_tracking'])} | {fmt(d['reading_self_paced'])} |")
    lines += ["", "## Mean score delta", "", "| BLiMP | Supp | EWoK | Entity | COMPS | Read eye | Read SPR |", "|---:|---:|---:|---:|---:|---:|---:|", f"| {fmt(mean_delta['blimp_fast'])} | {fmt(mean_delta['supplement_fast'])} | {fmt(mean_delta['ewok_fast'])} | {fmt(mean_delta['entity_tracking_fast'])} | {fmt(mean_delta['comps'])} | {fmt(mean_delta['reading_eye_tracking'])} | {fmt(mean_delta['reading_self_paced'])} |"]
    lines += ["", "## Coupled tokenizer changes", "", "| seed | 16k tokens/word | 40k tokens/word | % change | 16k expected WWM pred tokens | 40k expected WWM pred tokens | % change | 16k trunc frac | 40k trunc frac |", "|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for c in coupling_rows:
        lines.append(f"| {c['seed']} | {c['untruncated_tokens_per_whitespace_word_16k']:.4f} | {c['untruncated_tokens_per_whitespace_word_40k']:.4f} | {c['untruncated_tokens_per_whitespace_word_pct_change']:.2f}% | {c['expected_wwm_predicted_tokens_at_mask_prob_0p15_if_selected_groups_cover_tokens_16k']:.1f} | {c['expected_wwm_predicted_tokens_at_mask_prob_0p15_if_selected_groups_cover_tokens_40k']:.1f} | {c['expected_wwm_predicted_tokens_at_mask_prob_0p15_if_selected_groups_cover_tokens_pct_change']:.2f}% | {c['truncated_example_fraction_16k']:.4f} | {c['truncated_example_fraction_40k']:.4f} |")
    OUT_NOTE.parent.mkdir(parents=True, exist_ok=True)
    OUT_NOTE.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"mean_official40k_minus_baseline16k_wwm": mean_delta, "by_seed": deltas, "out": str(OUT_JSON)}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
