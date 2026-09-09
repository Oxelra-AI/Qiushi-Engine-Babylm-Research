#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import pathlib
import re
import subprocess
import sys
import time
from collections import defaultdict

ROOT = pathlib.Path("experiments/archive/initial_model_studies")
STRICT = ROOT / "repos/babylm-eval/strict"
PARQUET = ROOT / "data/ewok-core-1.0/data/test/ewok-core-1.0.parquet"
VOCAB = STRICT / "evaluation_pipeline/ewok/vocab.txt"
NLTK_DATA = (ROOT / "data/nltk_data").resolve()
EWOK_OUT = STRICT / "evaluation_data/full_eval/ewok_filtered_word_tokenize"
MODEL = ROOT / "training/runs/babylm_fullcycle_debertav2_8x480_wwm_seed42_100M_b256/hf_model"
EVAL_OUT = ROOT / "training/runs/babylm_fullcycle_debertav2_8x480_wwm_seed42_100M_b256/eval_results_full_ewok_word_tokenize"
LOG = (ROOT.parents[2] / 'research/notes/initial_model_studies/local_ewok_word_tokenize_filter_and_eval.log')
SUMMARY = ROOT / "data/local_ewok_word_tokenize_filter_summary.json"
SCORE_JSON = ROOT / "data/debertav2_b256_full_ewok_word_tokenize_score.json"
NOTE = (ROOT.parents[2] / 'research/notes/initial_model_studies/debertav2_b256_full_ewok_word_tokenize_score.md')
FALLBACK_SUMMARY = ROOT / "data/local_ewok_filter_summary.json"
FALLBACK_SCORE = ROOT / "data/debertav2_b256_full_ewok_score.json"


def read_parquet_rows(path: pathlib.Path) -> list[dict]:
    try:
        import pandas as pd
        return pd.read_parquet(path).to_dict(orient="records")
    except Exception:
        import pyarrow.parquet as pq
        return pq.read_table(path).to_pylist()


def parse_avg(report: pathlib.Path) -> float:
    txt = report.read_text(encoding="utf-8", errors="replace")
    m = re.search(r"AVERAGE ACCURACY\s*\n([0-9.\-]+)", txt)
    if not m:
        raise RuntimeError(f"Could not parse average from {report}\n{txt[:1000]}")
    return float(m.group(1))


def setup_env() -> dict:
    env = os.environ.copy()
    env["NLTK_DATA"] = str(NLTK_DATA)
    hf_home = ROOT / "training/hf_home"
    env["HF_HOME"] = str(hf_home.resolve())
    env["HF_HUB_CACHE"] = str((hf_home / "hub").resolve())
    env["TRANSFORMERS_CACHE"] = str((hf_home / "transformers").resolve())
    env["HF_MODULES_CACHE"] = str((ROOT / "training/hf_modules_cache").resolve())
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    for k in ["HF_HOME", "HF_HUB_CACHE", "TRANSFORMERS_CACHE", "HF_MODULES_CACHE"]:
        pathlib.Path(env[k]).mkdir(parents=True, exist_ok=True)
    return env


def main() -> None:
    t0 = time.time()
    LOG.parent.mkdir(parents=True, exist_ok=True)
    logf = LOG.open("w", encoding="utf-8")
    def log(msg: str):
        print(msg, flush=True)
        logf.write(msg + "\n")
        logf.flush()

    os.environ["NLTK_DATA"] = str(NLTK_DATA)
    import nltk
    nltk.data.path.insert(0, str(NLTK_DATA))
    from nltk.tokenize import word_tokenize
    test_tokens = word_tokenize("This is a test.")
    if test_tokens != ["This", "is", "a", "test", "."]:
        raise RuntimeError(f"Unexpected word_tokenize test output: {test_tokens}")

    rows = read_parquet_rows(PARQUET)
    vocab = {line.strip() for line in VOCAB.read_text(encoding="utf-8").splitlines() if line.strip()}
    EWOK_OUT.mkdir(parents=True, exist_ok=True)
    for old in EWOK_OUT.glob("*.jsonl"):
        old.unlink()

    items_per_domain: dict[str, list[dict]] = defaultdict(list)
    skipped = 0
    skip_by_domain: dict[str, int] = defaultdict(int)
    required = ["Domain", "Context1", "Context2", "Target1", "Target2", "ContextType", "ContextDiff", "TargetDiff"]
    for r in rows:
        missing = [k for k in required if k not in r]
        if missing:
            raise RuntimeError(f"Missing EWoK columns {missing}; available={sorted(r.keys())}")
        domain = str(r["Domain"])
        skip = False
        for key in ("Context1", "Context2", "Target1", "Target2"):
            for word in word_tokenize(str(r[key]).lower()):
                if word not in vocab:
                    skip = True
                    break
            if skip:
                break
        if skip:
            skipped += 1
            skip_by_domain[domain] += 1
            continue
        clean = {k: (v.item() if hasattr(v, "item") else v) for k, v in r.items()}
        items_per_domain[domain].append(clean)

    written_by_domain = {}
    for domain, items in sorted(items_per_domain.items()):
        path = EWOK_OUT / f"{domain}.jsonl"
        with path.open("w", encoding="utf-8") as out:
            for item in items:
                out.write(json.dumps(item, ensure_ascii=False) + "\n")
                swapped = dict(item)
                swapped["Context1"], swapped["Context2"] = swapped["Context2"], swapped["Context1"]
                swapped["Target1"], swapped["Target2"] = swapped["Target2"], swapped["Target1"]
                out.write(json.dumps(swapped, ensure_ascii=False) + "\n")
        written_by_domain[domain] = {"filtered_items": len(items), "jsonl_lines_with_swaps": len(items) * 2, "path": str(path)}

    total_filtered = sum(v["filtered_items"] for v in written_by_domain.values())
    total_lines = sum(v["jsonl_lines_with_swaps"] for v in written_by_domain.values())
    fallback_summary = json.loads(FALLBACK_SUMMARY.read_text(encoding="utf-8")) if FALLBACK_SUMMARY.exists() else None
    fallback_score = json.loads(FALLBACK_SCORE.read_text(encoding="utf-8")) if FALLBACK_SCORE.exists() else None
    count_delta = None
    if fallback_summary:
        count_delta = {
            "filtered_items_word_tokenize_minus_fallback": total_filtered - int(fallback_summary["filtered_items"]),
            "jsonl_lines_word_tokenize_minus_fallback": total_lines - int(fallback_summary["jsonl_lines_with_swaps"]),
            "domain_filtered_item_deltas": {d: written_by_domain.get(d, {"filtered_items": 0})["filtered_items"] - fallback_summary.get("domains", {}).get(d, {"filtered_items": 0})["filtered_items"] for d in sorted(set(written_by_domain) | set(fallback_summary.get("domains", {})))}
        }
    summary = {
        "source_parquet": str(PARQUET),
        "nltk_data": str(NLTK_DATA),
        "tokenizer_used_for_filter": "nltk.word_tokenize",
        "word_tokenize_test": test_tokens,
        "raw_rows": len(rows),
        "filtered_items": total_filtered,
        "skipped_items": skipped,
        "jsonl_lines_with_swaps": total_lines,
        "domains": written_by_domain,
        "skipped_by_domain": dict(sorted(skip_by_domain.items())),
        "output_dir": str(EWOK_OUT),
        "fallback_comparison": count_delta,
        "note": "No EWoK example text is printed here; filtered JSONLs are internal evaluator data from the gated local dataset.",
    }
    SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    log(json.dumps({"event": "ewok_filtered_word_tokenize", "raw_rows": len(rows), "filtered_items": total_filtered, "lines": total_lines, "domains": len(written_by_domain), "fallback_delta": count_delta}))

    env = setup_env()
    cmd = [
        sys.executable, "-m", "evaluation_pipeline.sentence_zero_shot.run",
        "--model_path_or_name", str((MODEL / "chck_100M").resolve()),
        "--backend", "mlm",
        "--task", "ewok",
        "--data_path", str(EWOK_OUT.resolve()),
        "--save_predictions",
        "--batch_size", "64",
        "--output_dir", str(EVAL_OUT.resolve()),
    ]
    log("$ " + " ".join(cmd))
    p = subprocess.run(cmd, cwd=str(STRICT), env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    log(p.stdout[-6000:])
    if p.returncode != 0:
        raise RuntimeError(f"EWoK eval failed with return code {p.returncode}\n{p.stdout[-10000:]}")
    # Because model_path_or_name is actual chck_100M path and revision omitted, output stem is chck_100M/main.
    hits = sorted(EVAL_OUT.rglob("best_temperature_report.txt"))
    if not hits:
        raise FileNotFoundError(f"no best_temperature_report under {EVAL_OUT}")
    report = hits[0]
    pred = report.parent / "predictions.json"
    score = parse_avg(report)
    score_delta = None
    if fallback_score:
        score_delta = score - float(fallback_score["ewok_full_score"])
    payload = {
        "model": "baseline16k DeBERTa-v2 8x480 WWM",
        "model_path": str(MODEL / "chck_100M"),
        "backend": "mlm",
        "ewok_full_score": score,
        "filter_summary": str(SUMMARY),
        "report": str(report),
        "predictions": str(pred),
        "eval_output_dir": str(EVAL_OUT),
        "fallback_score": fallback_score,
        "fallback_score_delta": score_delta,
        "elapsed_sec": time.time() - t0,
        "warning": "Full local EWoK generated from provided gated parquet with official nltk.word_tokenize resources and official vocab filter.",
    }
    SCORE_JSON.parent.mkdir(parents=True, exist_ok=True)
    SCORE_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    NOTE.write_text("\n".join([
        "# research — baseline16k DeBERTa-v2 full EWoK with official word_tokenize",
        "",
        f"Evidence JSON: `{SCORE_JSON}`",
        f"Filter summary: `{SUMMARY}`",
        f"Report: `{report}`",
        "",
        f"Full EWoK score: **{score:.2f}**",
        f"Fallback-score delta: {score_delta}",
        f"Filtered items: {total_filtered}; swapped JSONL lines: {total_lines}",
        "",
        "This uses the local EWoK parquet, official BabyLM vocab filter, and NLTK `word_tokenize` with the local `data/nltk_data` resources.",
    ]) + "\n", encoding="utf-8")
    log(json.dumps({"status": "FULL_EWOK_WORD_TOKENIZE_DONE", "score": score, "score_json": str(SCORE_JSON), "score_delta_vs_fallback": score_delta, "elapsed_sec": time.time() - t0}, indent=2))
    logf.close()

if __name__ == "__main__":
    main()
