#!/usr/bin/env python3
"""research: audit tokenization as an isolated factor for clean-Qwen data.

This does not train or evaluate a model. It compares the inherited strong 16k
tokenizer with a clean official-only 40k SentencePiece tokenizer on the exact
research clean-Qwen pools, focusing on whether 40k changes sequence packing,
truncation risk, pair-row token density, and lexical fragmentation. These are
mechanistic reasons to decide whether a tokenizer-isolation training screen is
worth H100 time after the MNTP auxiliary failed.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse
import hashlib
import json
import math
import statistics as stats
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List

from transformers import AutoTokenizer, DebertaV2Tokenizer

ROOT = _public_path('experiments/archive/compact_experience')
DEFAULT_16K = _public_path('experiments/archive/initial_model_studies/training/runs/fullcycle_official_wwm_debertav2_8x480_seed43_100M_b256/hf_model')
DEFAULT_40K = _public_path('experiments/archive/compact_experience/data/shared_tokenizer/hf_tokenizer_40k_shared')
QWEN_10M = _public_path('experiments/archive/compact_experience/data/qwen_clean_aligned/training_corpora/qwen_aligned_10M.jsonl')
OFF_10M = _public_path('experiments/archive/compact_experience/data/qwen_clean_aligned/training_corpora/official_only_10M.jsonl')
OUT = _public_path('experiments/archive/compact_experience/data/tokenizer_factor_audit.json')


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def pct(vals: List[float], q: float) -> float:
    if not vals:
        return float("nan")
    xs = sorted(vals)
    k = (len(xs) - 1) * q
    lo = math.floor(k); hi = math.ceil(k)
    if lo == hi:
        return float(xs[lo])
    return float(xs[lo] * (hi - k) + xs[hi] * (k - lo))


def summarize(vals: List[float]) -> Dict[str, Any]:
    return {
        "n": len(vals),
        "mean": stats.fmean(vals) if vals else None,
        "median": pct(vals, 0.50) if vals else None,
        "p05": pct(vals, 0.05) if vals else None,
        "p95": pct(vals, 0.95) if vals else None,
        "p99": pct(vals, 0.99) if vals else None,
        "min": min(vals) if vals else None,
        "max": max(vals) if vals else None,
    }


def load_tokenizer(path: Path):
    # DebertaV2Tokenizer handles SentencePiece tokenizers saved by prior scripts;
    # AutoTokenizer handles the inherited HF model tokenizer.
    try:
        return AutoTokenizer.from_pretrained(str(path), use_fast=False)
    except Exception:
        return DebertaV2Tokenizer.from_pretrained(str(path))


def iter_rows(path: Path):
    with path.open(encoding="utf-8") as f:
        for i, line in enumerate(f):
            row = json.loads(line)
            text = row.get("text", "")
            words = int(row.get("words", len(text.split())))
            source = row.get("source", "")
            yield i, text, words, source


def audit_one(tok, path: Path, max_rows: int) -> Dict[str, Any]:
    toks_all: List[int] = []
    t_per_word_all: List[float] = []
    pair_toks: List[int] = []
    pair_tpw: List[float] = []
    source_counts = Counter()
    over = Counter()
    source_stats = defaultdict(lambda: {"rows": 0, "words": 0, "tokens": 0, "over256": 0})
    examples = []
    for i, text, words, source in iter_rows(path):
        if i >= max_rows:
            break
        ids = tok(text, add_special_tokens=False)["input_ids"]
        nt = len(ids)
        tpw = nt / max(1, words)
        toks_all.append(nt)
        t_per_word_all.append(tpw)
        source_counts[source] += 1
        if nt > 256:
            over[">256"] += 1
        if nt > 384:
            over[">384"] += 1
        if nt > 512:
            over[">512"] += 1
        ss = source_stats[source]
        ss["rows"] += 1; ss["words"] += words; ss["tokens"] += nt; ss["over256"] += int(nt > 256)
        if source == "qwen_pair_packed":
            pair_toks.append(nt); pair_tpw.append(tpw)
            if len(examples) < 8:
                examples.append({"row": i, "words": words, "tokens": nt, "tokens_per_word": tpw, "text_head": text[:220]})
    by_source = {}
    for s, d in source_stats.items():
        by_source[s] = {
            **d,
            "tokens_per_word": d["tokens"] / max(1, d["words"]),
            "over256_rate": d["over256"] / max(1, d["rows"]),
        }
    return {
        "rows_seen": len(toks_all),
        "token_lengths": summarize(toks_all),
        "tokens_per_word": summarize(t_per_word_all),
        "over_counts": dict(over),
        "over256_rate": over[">256"] / max(1, len(toks_all)),
        "qwen_pair_rows": {
            "token_lengths": summarize(pair_toks),
            "tokens_per_word": summarize(pair_tpw),
            "over256_rate": sum(1 for x in pair_toks if x > 256) / max(1, len(pair_toks)),
            "examples": examples,
        },
        "by_source": by_source,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tok16", default=str(DEFAULT_16K))
    ap.add_argument("--tok40", default=str(DEFAULT_40K))
    ap.add_argument("--max_rows", type=int, default=0, help="0 = all rows")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    max_rows = args.max_rows if args.max_rows and args.max_rows > 0 else 10**12
    tok16_path = Path(args.tok16)
    tok40_path = Path(args.tok40)
    tok16 = load_tokenizer(tok16_path)
    tok40 = load_tokenizer(tok40_path)
    report = {
        "status": "TOKENIZER_FACTOR_AUDIT",
        "purpose": "Audit whether 40k vocabulary is a plausible isolated next factor after MNTP failed, without using eval/AoA/CDI signals.",
        "tokenizers": {
            "baseline16k": {"path": str(tok16_path), "vocab_size": len(tok16)},
            "official_shared40k": {"path": str(tok40_path), "vocab_size": len(tok40)},
        },
        "inputs": {
            "qwen_10M": {"path": str(QWEN_10M), "sha256": sha256_file(QWEN_10M)},
            "official_10M": {"path": str(OFF_10M), "sha256": sha256_file(OFF_10M)},
        },
        "max_rows": None if max_rows >= 10**12 else max_rows,
        "audits": {},
    }
    for name, tok in [("baseline16k", tok16), ("official_shared40k", tok40)]:
        report["audits"][name] = {
            "qwen_aligned_10M": audit_one(tok, QWEN_10M, max_rows),
            "official_only_10M": audit_one(tok, OFF_10M, max_rows),
        }
    # Direct ratio comparisons for qwen pair rows.
    bq = report["audits"]["baseline16k"]["qwen_aligned_10M"]
    fq = report["audits"]["official_shared40k"]["qwen_aligned_10M"]
    bo = report["audits"]["baseline16k"]["official_only_10M"]
    fo = report["audits"]["official_shared40k"]["official_only_10M"]
    def ratio(a,b):
        return None if b in (None, 0) else a/b
    report["direct_comparison"] = {
        "qwen_all_mean_tokens_per_word_40k_over_16k": ratio(fq["tokens_per_word"]["mean"], bq["tokens_per_word"]["mean"]),
        "official_all_mean_tokens_per_word_40k_over_16k": ratio(fo["tokens_per_word"]["mean"], bo["tokens_per_word"]["mean"]),
        "qwen_pair_mean_tokens_per_word_40k_over_16k": ratio(fq["qwen_pair_rows"]["tokens_per_word"]["mean"], bq["qwen_pair_rows"]["tokens_per_word"]["mean"]),
        "qwen_pair_over256_rate_16k": bq["qwen_pair_rows"]["over256_rate"],
        "qwen_pair_over256_rate_40k": fq["qwen_pair_rows"]["over256_rate"],
        "all_qwen_over256_rate_16k": bq["over256_rate"],
        "all_qwen_over256_rate_40k": fq["over256_rate"],
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"status": report["status"], "out": str(out), "direct_comparison": report["direct_comparison"]}, indent=2))


if __name__ == "__main__":
    main()
