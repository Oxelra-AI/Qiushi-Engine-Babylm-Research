#!/usr/bin/env python3
"""Audit official-only geometry corpora and trained runs.

Compares original 160-word official geometry against cap120 official geometry under the
trainer-visible tokenizer rule (add_special_tokens=False, right truncation at 256). It
also records actual optimizer-step counts and losses from scientific_metrics.json when
runs exist. It reads no evaluation outputs, AoA/CDI words, child curves, predictions, or
scores.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import collections
import json
import pathlib
import statistics
from typing import Any, Dict, Iterable, List, Optional

from transformers import AutoTokenizer

ROOT = _public_path('experiments/archive/compact_experience')
TOKENIZER = pathlib.Path("experiments/archive/initial_model_studies/training/runs/fullcycle_official_wwm_debertav2_8x480_seed43_100M_b256/hf_model")
SEQ_LEN = 256
OUT = _public_path('experiments/archive/compact_experience/data/official_geometry_audit/official_geometry_exposure_and_steps.json')
NOTE = _public_path('research/notes/compact_experience/official_geometry_exposure_and_steps.md')

CORPORA = {
    "official160_b256": {
        "meta": _public_path('experiments/archive/compact_experience/data/official_geometry/official160/official160_geometry_metadata.json'),
        "train": _public_path('experiments/archive/compact_experience/data/official_geometry/official160/training_corpora/official160_100M.jsonl'),
        "pool": _public_path('experiments/archive/compact_experience/data/official_geometry/official160/training_corpora/official160_10M.jsonl'),
        "run": _public_path('experiments/archive/compact_experience/training/runs/official160_b256_16k_seed43022'),
    },
    "official_cap120geom_b256": {
        "meta": _public_path('experiments/archive/compact_experience/data/contextual_one_pair/cap120/contextual_one_pair_cap120_metadata.json'),
        "train": _public_path('experiments/archive/compact_experience/data/contextual_one_pair/cap120/training_corpora/official_context_lengthmatched_cap120_100M.jsonl'),
        "pool": _public_path('experiments/archive/compact_experience/data/contextual_one_pair/cap120/training_corpora/official_context_lengthmatched_cap120_10M.jsonl'),
        "run": _public_path('experiments/archive/compact_experience/training/runs/official_cap120geom_b256_16k_seed43022'),
    },
    "official_cap120geom_b286_lr2515": {
        "meta": _public_path('experiments/archive/compact_experience/data/contextual_one_pair/cap120/contextual_one_pair_cap120_metadata.json'),
        "train": _public_path('experiments/archive/compact_experience/data/contextual_one_pair/cap120/training_corpora/official_context_lengthmatched_cap120_100M.jsonl'),
        "pool": _public_path('experiments/archive/compact_experience/data/contextual_one_pair/cap120/training_corpora/official_context_lengthmatched_cap120_10M.jsonl'),
        "run": _public_path('experiments/archive/compact_experience/training/runs/official_cap120geom_b286_lr2515_16k_seed43022'),
    },
    "official_cap120geom_b288_lr2515": {
        "meta": _public_path('experiments/archive/compact_experience/data/contextual_one_pair/cap120/contextual_one_pair_cap120_metadata.json'),
        "train": _public_path('experiments/archive/compact_experience/data/contextual_one_pair/cap120/training_corpora/official_context_lengthmatched_cap120_100M.jsonl'),
        "pool": _public_path('experiments/archive/compact_experience/data/contextual_one_pair/cap120/training_corpora/official_context_lengthmatched_cap120_10M.jsonl'),
        "run": _public_path('experiments/archive/compact_experience/training/runs/official_cap120geom_b288_lr2515_16k_seed43022'),
    },
}


def read_jsonl(path: pathlib.Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def read_json(path: pathlib.Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def stats(xs: List[float]) -> Dict[str, Any]:
    if not xs:
        return {"n": 0}
    ys = sorted(float(x) for x in xs)
    def q(p: float) -> float:
        if len(ys) == 1:
            return ys[0]
        pos = p * (len(ys) - 1)
        lo = int(pos)
        hi = min(lo + 1, len(ys) - 1)
        frac = pos - lo
        return ys[lo] * (1 - frac) + ys[hi] * frac
    return {"n": len(ys), "mean": statistics.fmean(ys), "min": ys[0], "p05": q(0.05), "p25": q(0.25), "median": q(0.5), "p75": q(0.75), "p95": q(0.95), "max": ys[-1], "sum": sum(ys)}


def encode_len(tok, text: str) -> int:
    return len(tok(text, add_special_tokens=False, truncation=False)["input_ids"])


def audit_pool(tok, path: pathlib.Path) -> Dict[str, Any]:
    n = 0
    words = 0
    toks: List[int] = []
    visible: List[int] = []
    source_words = collections.Counter()
    source_rows = collections.Counter()
    over = 0
    row_words: List[int] = []
    for r in read_jsonl(path):
        text = str(r["text"])
        w = int(r.get("words", len(text.split())))
        if w != len(text.split()):
            raise RuntimeError(f"word mismatch in {path}: declared={w} actual={len(text.split())}")
        L = encode_len(tok, text)
        toks.append(L)
        visible.append(min(L, SEQ_LEN))
        if L > SEQ_LEN:
            over += 1
        src = str(r.get("source", ""))
        source_words[src] += w
        source_rows[src] += 1
        row_words.append(w)
        words += w
        n += 1
    return {
        "file": str(path),
        "rows": n,
        "words": words,
        "row_word_stats": stats([float(x) for x in row_words]),
        "token_length_stats": stats([float(x) for x in toks]),
        "visible_token_stats": stats([float(x) for x in visible]),
        "total_visible_tokens": int(sum(visible)),
        "total_untruncated_tokens": int(sum(toks)),
        "truncated_tokens": int(sum(max(0, x - SEQ_LEN) for x in toks)),
        "rows_over_seq_len": over,
        "fraction_rows_over_seq_len": over / n if n else None,
        "source_words": dict(sorted(source_words.items())),
        "source_rows": dict(sorted(source_rows.items())),
        "tokenizer_rule": {"add_special_tokens": False, "truncation_side": "right", "seq_len": SEQ_LEN},
    }


def run_metrics(path: pathlib.Path) -> Dict[str, Any]:
    p = path / "scientific_metrics.json"
    if not p.exists():
        return {"run_dir": str(path), "exists": path.exists(), "scientific_metrics_exists": False}
    m = read_json(p)
    return {
        "run_dir": str(path),
        "exists": True,
        "scientific_metrics_exists": True,
        "word_exposure": m.get("word_exposure"),
        "actual_training_steps": m.get("actual_training_steps"),
        "loss_first": m.get("loss_first"),
        "loss_last": m.get("loss_last"),
        "saved_checkpoints": len(m.get("saved_checkpoints", [])),
        "checkpoint_names_tail": [c.get("name") for c in m.get("saved_checkpoints", [])[-5:]],
    }


def main() -> None:
    tok = AutoTokenizer.from_pretrained(str(TOKENIZER), use_fast=True)
    corpora: Dict[str, Any] = {}
    for name, paths in CORPORA.items():
        rec: Dict[str, Any] = {"paths": {k: str(v) for k, v in paths.items()}}
        if paths["meta"].exists():
            meta = read_json(paths["meta"])
            rec["metadata_status"] = meta.get("status")
            rec["metadata_words_10M"] = meta.get("words_10M") or meta.get("words_per_pool") or meta.get("total_words")
            rec["metadata_rows_10M"] = meta.get("rows_10M") or meta.get("treatment_rows") or meta.get("rows_per_pool")
        else:
            rec["metadata_missing"] = True
        if paths["pool"].exists():
            rec["pool_audit"] = audit_pool(tok, paths["pool"])
        else:
            rec["pool_missing"] = True
        if paths["train"].exists():
            # Train files are 10x larger; avoid expensive full tokenization. Word/row counts are still important.
            rows = 0; words = 0
            for r in read_jsonl(paths["train"]):
                rows += 1; words += int(r.get("words", len(str(r.get("text", "")).split())))
            rec["train_file_counts"] = {"rows": rows, "words": words}
        else:
            rec["train_missing"] = True
        rec["run_metrics"] = run_metrics(paths["run"])
        corpora[name] = rec

    payload = {
        "status": "OFFICIAL_GEOMETRY_EXPOSURE_AND_STEPS_AUDIT",
        "non_leakage_statement": "Reads only training corpora, metadata, tokenizer-visible length statistics, and scientific_metrics from training runs. No evaluation outputs, AoA/CDI words, child curves, AoA predictions, or AoA scores are used.",
        "corpora": corpora,
    }
    _public_path('experiments/archive/compact_experience/data/official_geometry_audit').mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = ["# research official-geometry exposure and step audit", "", payload["non_leakage_statement"], ""]
    for name, rec in corpora.items():
        p = rec.get("pool_audit") or {}
        rm = rec.get("run_metrics") or {}
        lines.append(
            f"- `{name}` rows10M={p.get('rows')} words10M={p.get('words')} visible_tokens10M={p.get('total_visible_tokens')} "
            f"rows_over256={p.get('rows_over_seq_len')} frac_over256={p.get('fraction_rows_over_seq_len')} "
            f"run_steps={rm.get('actual_training_steps')} exposure={rm.get('word_exposure')} loss_last={rm.get('loss_last')} ckpts={rm.get('saved_checkpoints')}"
        )
    _public_path('research/notes/compact_experience').mkdir(parents=True, exist_ok=True)
    NOTE.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(OUT), "note": str(NOTE), "names": list(corpora)}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
