#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import pathlib
import re
import shutil
import subprocess
import sys
import time
from collections import defaultdict

ROOT = pathlib.Path("experiments/archive/initial_model_studies")
STRICT = ROOT / "repos/babylm-eval/strict"
PARQUET = ROOT / "data/ewok-core-1.0/data/test/ewok-core-1.0.parquet"
VOCAB = STRICT / "evaluation_pipeline/ewok/vocab.txt"
EWOK_OUT = STRICT / "evaluation_data/full_eval/ewok_filtered"
MODEL = ROOT / "training/runs/babylm_fullcycle_debertav2_8x480_wwm_seed42_100M_b256/hf_model"
EVAL_OUT = ROOT / "training/runs/babylm_fullcycle_debertav2_8x480_wwm_seed42_100M_b256/eval_results_full_ewok"
LOG = (ROOT.parents[2] / 'research/notes/initial_model_studies/local_ewok_filter_and_eval.log')
SUMMARY = ROOT / "data/local_ewok_filter_summary.json"
SCORE_JSON = ROOT / "data/debertav2_b256_full_ewok_score.json"
NOTE = (ROOT.parents[2] / 'research/notes/initial_model_studies/debertav2_b256_full_ewok_score.md')


def get_tokenizer():
    try:
        from nltk.tokenize import word_tokenize
        word_tokenize("This is a test.")
        return "nltk.word_tokenize", lambda s: word_tokenize(s)
    except Exception:
        from nltk.tokenize import TreebankWordTokenizer
        tok = TreebankWordTokenizer()
        return "nltk.TreebankWordTokenizer_fallback", lambda s: tok.tokenize(s)


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

    if not PARQUET.exists():
        raise FileNotFoundError(PARQUET)
    rows = read_parquet_rows(PARQUET)
    with VOCAB.open("r", encoding="utf-8") as f:
        vocab = {line.strip() for line in f if line.strip()}
    tokenizer_name, tokenize = get_tokenizer()

    if EWOK_OUT.exists():
        for old in EWOK_OUT.glob("*.jsonl"):
            old.unlink()
    else:
        EWOK_OUT.mkdir(parents=True, exist_ok=True)

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
            for word in tokenize(str(r[key]).lower()):
                if word not in vocab:
                    skip = True
                    break
            if skip:
                break
        if skip:
            skipped += 1
            skip_by_domain[domain] += 1
            continue
        # Convert numpy/pandas scalars to JSON-safe basic types without printing plain text.
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
    summary = {
        "source_parquet": str(PARQUET),
        "tokenizer_used_for_filter": tokenizer_name,
        "raw_rows": len(rows),
        "filtered_items": total_filtered,
        "skipped_items": skipped,
        "jsonl_lines_with_swaps": total_lines,
        "domains": written_by_domain,
        "skipped_by_domain": dict(sorted(skip_by_domain.items())),
        "output_dir": str(EWOK_OUT),
        "note": "No EWoK example text is printed here; filtered JSONLs are internal evaluator data from the gated local dataset.",
    }
    SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    log(json.dumps({"event": "ewok_filtered", "raw_rows": len(rows), "filtered_items": total_filtered, "lines": total_lines, "domains": len(written_by_domain), "tokenizer": tokenizer_name}))

    env = setup_env()
    cmd = [
        sys.executable, "-m", "evaluation_pipeline.sentence_zero_shot.run",
        "--model_path_or_name", str(MODEL.resolve()),
        "--backend", "mlm",
        "--task", "ewok",
        "--data_path", str(EWOK_OUT.resolve()),
        "--save_predictions",
        "--revision_name", "chck_100M",
        "--batch_size", "64",
        "--output_dir", str(EVAL_OUT.resolve()),
    ]
    log("$ " + " ".join(cmd))
    p = subprocess.run(cmd, cwd=str(STRICT), env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    log(p.stdout[-6000:])
    if p.returncode != 0:
        raise RuntimeError(f"EWoK eval failed with return code {p.returncode}\n{p.stdout[-10000:]}")
    report = EVAL_OUT / "hf_model" / "chck_100M" / "zero_shot" / "mlm" / "ewok" / "ewok_filtered" / "best_temperature_report.txt"
    pred = EVAL_OUT / "hf_model" / "chck_100M" / "zero_shot" / "mlm" / "ewok" / "ewok_filtered" / "predictions.json"
    score = parse_avg(report)
    payload = {
        "model": "baseline16k DeBERTa-v2 8x480 WWM",
        "model_path": str(MODEL),
        "revision": "chck_100M",
        "backend": "mlm",
        "ewok_full_score": score,
        "filter_summary": str(SUMMARY),
        "report": str(report),
        "predictions": str(pred),
        "eval_output_dir": str(EVAL_OUT),
        "elapsed_sec": time.time() - t0,
        "warning": "This is full local EWoK generated from provided gated parquet with official vocab filter; not fast EWoK.",
    }
    SCORE_JSON.parent.mkdir(parents=True, exist_ok=True)
    SCORE_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    NOTE.write_text("\n".join([
        "# research — baseline16k DeBERTa-v2 full EWoK score",
        "",
        f"Evidence JSON: `{SCORE_JSON}`",
        f"Filter summary: `{SUMMARY}`",
        f"Report: `{report}`",
        "",
        f"Full EWoK score: **{score:.2f}**",
        "",
        "This uses the local EWoK parquet and the official BabyLM vocab-filter/sentence-zero-shot evaluation flow. It is not the earlier fast-EWoK interim score.",
    ]) + "\n", encoding="utf-8")
    log(json.dumps({"status": "FULL_EWOK_DONE", "score": score, "score_json": str(SCORE_JSON), "elapsed_sec": time.time() - t0}, indent=2))
    logf.close()


if __name__ == "__main__":
    main()
