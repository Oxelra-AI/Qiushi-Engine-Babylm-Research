#!/usr/bin/env python3
"""Audit seq256 tokenizer visibility for candidate training pools."""
from __future__ import annotations

import argparse
import json
import pathlib
import statistics
from typing import Any

from transformers import AutoTokenizer

TOKENIZER_DEFAULT = "experiments/archive/initial_model_studies/training/runs/fullcycle_official_wwm_debertav2_8x480_seed43_100M_b256/hf_model"
CANDIDATES = {
    "cached_fineweb_raw3m_treatment": pathlib.Path("experiments/archive/representation_and_objectives/training/data/cached_fineweb_broad_source_candidate/cleanqwen_cached_fineweb_broad_10M.jsonl"),
    "cached_fineweb_raw3m_control": pathlib.Path("experiments/archive/representation_and_objectives/training/data/cached_fineweb_broad_source_candidate/cleanqwen_official_lengthmatched_control_10M.jsonl"),
    "cached_fineweb_single_doc_treatment": pathlib.Path("experiments/archive/representation_and_objectives/training/data/cached_fineweb_single_doc_candidate/cleanqwen_cached_fineweb_single_doc_10M.jsonl"),
    "cached_fineweb_single_doc_control": pathlib.Path("experiments/archive/representation_and_objectives/training/data/cached_fineweb_single_doc_candidate/cleanqwen_official_lengthmatched_single_doc_control_10M.jsonl"),
}
OUT_DEFAULT = pathlib.Path("experiments/archive/representation_and_objectives/training/data/candidate_seq256_visibility/candidate_seq256_visibility.json")
NOTE_DEFAULT = pathlib.Path("research/notes/representation_and_objectives/candidate_seq256_visibility.md")


def stat(vals: list[float]) -> dict[str, Any]:
    if not vals:
        return {"n": 0}
    xs = sorted(vals)
    def q(p: float):
        return xs[min(len(xs)-1, max(0, round((len(xs)-1)*p)))]
    return {"n": len(xs), "min": xs[0], "p05": q(0.05), "mean": statistics.mean(xs), "median": statistics.median(xs), "p95": q(0.95), "p99": q(0.99), "max": xs[-1]}


def audit_file(path: pathlib.Path, tok, seq_len: int, focus_prefixes: list[str]) -> dict[str, Any]:
    rows = 0
    total_words = 0
    truncated_rows = 0
    token_lens: list[int] = []
    word_lens: list[int] = []
    focus: dict[str, dict[str, Any]] = {p: {"rows": 0, "words": 0, "truncated_rows": 0, "token_lens": [], "examples": []} for p in focus_prefixes}
    with path.open(encoding="utf-8") as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue
            o = json.loads(line)
            text = o["text"]
            words = int(o.get("words", len(text.split())))
            source = str(o.get("source", ""))
            ids = tok(text, add_special_tokens=False, truncation=False)["input_ids"]
            tl = len(ids)
            is_trunc = tl > seq_len
            rows += 1
            total_words += words
            token_lens.append(tl)
            word_lens.append(words)
            if is_trunc:
                truncated_rows += 1
            for pfx in focus_prefixes:
                if source.startswith(pfx):
                    rec = focus[pfx]
                    rec["rows"] += 1
                    rec["words"] += words
                    rec["token_lens"].append(tl)
                    if is_trunc:
                        rec["truncated_rows"] += 1
                        if len(rec["examples"]) < 5:
                            rec["examples"].append({"row": i, "source": source, "words": words, "tokens": tl, "text_excerpt": text[:500]})
    focus_out = {}
    for pfx, rec in focus.items():
        lens = rec.pop("token_lens")
        rec["token_length_stats"] = stat([float(x) for x in lens])
        rec["truncated_fraction"] = rec["truncated_rows"] / rec["rows"] if rec["rows"] else None
        focus_out[pfx] = rec
    return {
        "path": str(path),
        "rows": rows,
        "words": total_words,
        "seq_len": seq_len,
        "truncated_rows": truncated_rows,
        "truncated_fraction": truncated_rows / rows if rows else None,
        "token_length_stats": stat([float(x) for x in token_lens]),
        "word_length_stats": stat([float(x) for x in word_lens]),
        "focus_by_source_prefix": focus_out,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", default=TOKENIZER_DEFAULT)
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--out", default=str(OUT_DEFAULT))
    ap.add_argument("--note", default=str(NOTE_DEFAULT))
    args = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
    focus = [
        "fineweb_edu_random_quality_cached_initial_model_studies",
        "fineweb_edu_random_quality_single_doc_cached_initial_model_studies",
        "official_lengthmatched_to_cached_fineweb",
        "official_lengthmatched_to_cached_fineweb_single_doc",
        "qwen_pair_packed",
    ]
    results = {name: audit_file(path, tok, args.seq_len, focus) for name, path in CANDIDATES.items()}
    payload = {
        "status": "CANDIDATE_SEQ256_VISIBILITY_AUDIT",
        "tokenizer": args.tokenizer,
        "seq_len": args.seq_len,
        "candidates": results,
        "interpretation": "Rows above 256 baseline16k tokens are truncated exactly as in the protected trainer. A future broad-source run should prefer candidates where replacement-block truncation is limited or explicitly accepted as part of the source-distribution test.",
    }
    out = pathlib.Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    note = pathlib.Path(args.note)
    lines = ["# research candidate seq256 visibility audit\n\n", f"Tokenizer: `{args.tokenizer}`; seq_len={args.seq_len}.\n\n"]
    lines.append("| candidate | rows | trunc frac | focus source | focus rows | focus trunc frac | focus token p95 | focus token max |\n")
    lines.append("|---|---:|---:|---|---:|---:|---:|---:|\n")
    for name, res in results.items():
        base_tr = res["truncated_fraction"]
        focus_items = [(k,v) for k,v in res["focus_by_source_prefix"].items() if v.get("rows")]
        if not focus_items:
            lines.append(f"| {name} | {res['rows']} | {base_tr:.4f} |  |  |  |  |  |\n")
        else:
            for j,(k,v) in enumerate(focus_items):
                p95 = v["token_length_stats"].get("p95")
                mx = v["token_length_stats"].get("max")
                lines.append(f"| {name if j==0 else ''} | {res['rows'] if j==0 else ''} | {base_tr:.4f} | {k} | {v['rows']} | {v['truncated_fraction']:.4f} | {p95:.1f} | {mx:.1f} |\n")
    lines.append(f"\nJSON: `{out}`\n")
    note.parent.mkdir(parents=True, exist_ok=True)
    note.write_text("".join(lines), encoding="utf-8")
    print(json.dumps({
        "status": payload["status"],
        "out": str(out),
        "note": str(note),
        "candidate_truncated_fractions": {k: v["truncated_fraction"] for k,v in results.items()},
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
