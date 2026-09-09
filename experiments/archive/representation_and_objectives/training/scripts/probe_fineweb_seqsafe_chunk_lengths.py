#!/usr/bin/env python3
"""Probe seq256-safe chunking for cached FineWeb fallback candidates.

research showed that 160-word cached FineWeb/control rows have substantial baseline16k
truncation, and the official length-matched control block truncates more often than
the FineWeb block. This script tests shorter row chunks for the cached single-doc
FineWeb block and its official length-matched control, using the same official
filler stream as the research materializer.
"""
from __future__ import annotations

import argparse
import collections
import json
import pathlib
import statistics
from dataclasses import dataclass
from typing import Any, Iterable

from transformers import AutoTokenizer

DEFAULT_QWEN_POOL = pathlib.Path("experiments/archive/compact_experience/data/qwen_clean_aligned/training_corpora/qwen_aligned_10M.jsonl")
DEFAULT_FINEWEB = pathlib.Path("experiments/archive/initial_model_studies/data/fineweb_relation_matched_3M/fineweb_random_quality_3000000w.jsonl")
TOKENIZER_DEFAULT = "experiments/archive/initial_model_studies/training/runs/fullcycle_official_wwm_debertav2_8x480_seed43_100M_b256/hf_model"
OUT_DEFAULT = pathlib.Path("experiments/archive/representation_and_objectives/training/data/fineweb_seqsafe/seqsafe_chunk_probe.json")
NOTE_DEFAULT = pathlib.Path("research/notes/representation_and_objectives/fineweb_seqsafe_chunk_probe.md")


@dataclass
class Row:
    text: str
    words: int
    source: str
    example_id: int
    meta: dict[str, Any]


def norm_text(text: str) -> str:
    return " ".join(str(text).replace("\u00a0", " ").split())


def stats(vals: Iterable[int | float]) -> dict[str, Any]:
    xs = list(vals)
    if not xs:
        return {"n": 0}
    ys = sorted(xs)
    def q(p: float):
        return ys[min(len(ys) - 1, max(0, round((len(ys) - 1) * p)))]
    return {
        "n": len(xs),
        "min": ys[0],
        "p05": q(0.05),
        "mean": statistics.mean(xs),
        "median": statistics.median(xs),
        "p95": q(0.95),
        "p99": q(0.99),
        "max": ys[-1],
        "sum": sum(xs),
    }


def load_qwen_pool(path: pathlib.Path) -> tuple[list[Row], list[Row]]:
    pair, filler = [], []
    with path.open(encoding="utf-8") as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue
            o = json.loads(line)
            text = norm_text(o["text"])
            words = int(o.get("words", len(text.split())))
            if words != len(text.split()):
                raise RuntimeError(f"qwen word mismatch row {i}: {words}!={len(text.split())}")
            r = Row(text=text, words=words, source=str(o.get("source", "")), example_id=int(o.get("example_id", i)), meta={})
            (pair if r.source == "qwen_pair_packed" else filler).append(r)
    return pair, filler


def load_fineweb_single_doc(path: pathlib.Path) -> list[Row]:
    rows: list[Row] = []
    with path.open(encoding="utf-8") as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue
            o = json.loads(line)
            ids = o.get("doc_ids") or []
            if len(ids) != 1:
                continue
            text = norm_text(o["text"])
            words = int(o.get("words", len(text.split())))
            if words != len(text.split()):
                raise RuntimeError(f"fineweb word mismatch row {i}: {words}!={len(text.split())}")
            rows.append(Row(text=text, words=words, source="fineweb_single_doc", example_id=i, meta={"doc_id": str(ids[0]), "source_row": i}))
    return rows


def split_rows(rows: list[Row], chunk_words: int) -> list[Row]:
    out: list[Row] = []
    for r in rows:
        ws = r.text.split()
        for start in range(0, len(ws), chunk_words):
            part = ws[start:start + chunk_words]
            if not part:
                continue
            meta = dict(r.meta)
            meta.update({"parent_example_id": r.example_id, "parent_start_word": start, "parent_words": r.words})
            out.append(Row(text=" ".join(part), words=len(part), source=r.source, example_id=r.example_id * 10_000 + start, meta=meta))
    return out


@dataclass
class StreamState:
    rows: list[Row]
    row_i: int = 0
    word_i: int = 0

    def take_words(self, n: int) -> Row:
        out: list[str] = []
        comp: collections.Counter[str] = collections.Counter()
        segs: list[dict[str, Any]] = []
        while len(out) < n:
            if self.row_i >= len(self.rows):
                raise RuntimeError("official filler exhausted")
            r = self.rows[self.row_i]
            ws = r.text.split()
            rem = len(ws) - self.word_i
            need = n - len(out)
            take = min(rem, need)
            if take <= 0:
                self.row_i += 1
                self.word_i = 0
                continue
            out.extend(ws[self.word_i:self.word_i + take])
            comp[r.source] += take
            segs.append({"source": r.source, "example_id": r.example_id, "start_word": self.word_i, "words": take})
            self.word_i += take
            if self.word_i >= len(ws):
                self.row_i += 1
                self.word_i = 0
        return Row(text=" ".join(out), words=n, source="official_lengthmatched_to_cached_fineweb_seqsafe", example_id=0, meta={"component_sources": dict(comp), "segments": segs})


def official_control_rows(filler: list[Row], lengths: list[int]) -> list[Row]:
    st = StreamState(filler)
    out = []
    for i, n in enumerate(lengths):
        r = st.take_words(n)
        r.example_id = 8_700_000 + i
        out.append(r)
    return out


def token_len(tok, text: str) -> int:
    return len(tok(text, add_special_tokens=False, truncation=False)["input_ids"])


def visible_words(tok, words: list[str], seq_len: int) -> int:
    # Exact enough for audit: binary search the largest word-prefix whose tokenized length fits.
    if not words:
        return 0
    if token_len(tok, " ".join(words)) <= seq_len:
        return len(words)
    lo, hi = 0, len(words)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if token_len(tok, " ".join(words[:mid])) <= seq_len:
            lo = mid
        else:
            hi = mid - 1
    return lo


def audit_rows(rows: list[Row], tok, seq_len: int, exact_visible: bool) -> dict[str, Any]:
    token_lens = []
    word_lens = []
    trunc = 0
    visible = 0
    example_trunc = []
    source_counter = collections.Counter()
    multi_source = 0
    for i, r in enumerate(rows):
        tl = token_len(tok, r.text)
        token_lens.append(tl)
        word_lens.append(r.words)
        source_counter[r.source] += r.words
        if len((r.meta or {}).get("component_sources", {})) > 1:
            multi_source += 1
        if tl > seq_len:
            trunc += 1
            if len(example_trunc) < 5:
                example_trunc.append({"row": i, "words": r.words, "tokens": tl, "text_excerpt": r.text[:400], "meta": r.meta})
            if exact_visible:
                visible += visible_words(tok, r.text.split(), seq_len)
            else:
                # Conservative lower-bound approximation if exact computation is disabled.
                visible += min(r.words, int(r.words * seq_len / max(1, tl)))
        else:
            visible += r.words
    total_words = sum(word_lens)
    return {
        "rows": len(rows),
        "words": total_words,
        "truncated_rows": trunc,
        "truncated_fraction": trunc / len(rows) if rows else None,
        "visible_words_prefix_exact": visible,
        "visible_word_fraction_prefix_exact": visible / total_words if total_words else None,
        "hidden_words_prefix_exact": total_words - visible,
        "token_length_stats": stats(token_lens),
        "word_length_stats": stats(word_lens),
        "source_word_counts": dict(source_counter),
        "multi_source_rows": multi_source,
        "multi_source_row_fraction": multi_source / len(rows) if rows else None,
        "truncated_examples": example_trunc,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--qwen-pool", default=str(DEFAULT_QWEN_POOL))
    ap.add_argument("--fineweb", default=str(DEFAULT_FINEWEB))
    ap.add_argument("--tokenizer", default=TOKENIZER_DEFAULT)
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--chunks", default="64,80,96,112,128,144,160")
    ap.add_argument("--out", default=str(OUT_DEFAULT))
    ap.add_argument("--note", default=str(NOTE_DEFAULT))
    ap.add_argument("--fast-visible", action="store_true", help="Use token-ratio approximation for visible words instead of exact prefix binary search.")
    args = ap.parse_args()

    chunk_sizes = [int(x) for x in args.chunks.split(",") if x.strip()]
    qwen_pair, filler = load_qwen_pool(pathlib.Path(args.qwen_pool))
    fineweb = load_fineweb_single_doc(pathlib.Path(args.fineweb))
    tok = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
    results = {}
    for cw in chunk_sizes:
        fw_rows = split_rows(fineweb, cw)
        ctrl_rows = official_control_rows(filler, [r.words for r in fw_rows])
        fw_a = audit_rows(fw_rows, tok, args.seq_len, exact_visible=not args.fast_visible)
        ctrl_a = audit_rows(ctrl_rows, tok, args.seq_len, exact_visible=not args.fast_visible)
        results[str(cw)] = {
            "chunk_words": cw,
            "fineweb": fw_a,
            "official_control": ctrl_a,
            "treatment_minus_control_truncated_fraction": fw_a["truncated_fraction"] - ctrl_a["truncated_fraction"],
            "treatment_minus_control_visible_word_fraction": fw_a["visible_word_fraction_prefix_exact"] - ctrl_a["visible_word_fraction_prefix_exact"],
        }

    payload = {
        "status": "FINEWEB_SEQSAFE_CHUNK_PROBE",
        "tokenizer": args.tokenizer,
        "seq_len": args.seq_len,
        "qwen_pair_rows": len(qwen_pair),
        "qwen_pair_words": sum(r.words for r in qwen_pair),
        "fineweb_single_doc_input_rows": len(fineweb),
        "fineweb_single_doc_words": sum(r.words for r in fineweb),
        "results": results,
        "interpretation": "Shorter chunks reduce baseline16k seq256 truncation in the cached FineWeb replacement block and the official length-matched control. A fallback broad-source training run should avoid the 160-word block if its treatment/control visibility mismatch remains large.",
    }
    out = pathlib.Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    lines = ["# research FineWeb seq256-safe chunk probe\n\n"]
    lines.append(f"Tokenizer: `{args.tokenizer}`; seq_len={args.seq_len}. Source: cached INITIAL_MODEL_STUDIES FineWeb-Edu single-document rows.\n\n")
    lines.append("| chunk words | FineWeb rows | FineWeb trunc frac | FineWeb visible word frac | control trunc frac | control visible word frac | Δvisible word frac (FW-control) | FW token p95 | control token p95 |\n")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
    for cw in chunk_sizes:
        r = results[str(cw)]
        fw = r["fineweb"]; ct = r["official_control"]
        lines.append(
            f"| {cw} | {fw['rows']} | {fw['truncated_fraction']:.4f} | {fw['visible_word_fraction_prefix_exact']:.4f} | "
            f"{ct['truncated_fraction']:.4f} | {ct['visible_word_fraction_prefix_exact']:.4f} | {r['treatment_minus_control_visible_word_fraction']:+.4f} | "
            f"{fw['token_length_stats']['p95']:.1f} | {ct['token_length_stats']['p95']:.1f} |\n"
        )
    lines.append("\nInterpretation: the 160-word cached fallback imported an avoidable visibility mismatch. A seqsafe variant should use a shorter split such as 96 or 112 words if it preserves enough context while bringing both blocks near-full visibility.\n\n")
    lines.append(f"JSON: `{out}`\n")
    note = pathlib.Path(args.note); note.parent.mkdir(parents=True, exist_ok=True)
    note.write_text("".join(lines), encoding="utf-8")
    print(json.dumps({"status": payload["status"], "out": str(out), "note": str(note), "chunks": chunk_sizes}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
