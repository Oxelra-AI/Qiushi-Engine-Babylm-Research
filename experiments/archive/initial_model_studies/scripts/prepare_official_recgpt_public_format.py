#!/usr/bin/env python3
"""research: prepare official BabyLM text in public RecGPT JSONL/parquet format.

The public RecGPT training code expects:
  data/<dataset>.jsonl with {"text": ...} documents
  data/tokenized/<dataset>.parquet with a list<int32> input_ids column

This script converts the official 10M-word text to sentence-like JSONL documents,
tokenizes with the research RecGPT-style ByteLevel BPE tokenizer, writes parquet,
and records both word exposure and public count_dataset_tokens() token clock.
"""
from __future__ import annotations
import json, pathlib, re, sys, time
import pyarrow as pa
import pyarrow.parquet as pq
from transformers import AutoTokenizer

ROOT = pathlib.Path("experiments/archive/initial_model_studies")
SRC = ROOT / "data/custom_corpus/official_only_10M.txt"
TOK = ROOT / "data/recgpt_official_tokenizer"
OUT_DIR = ROOT / "data/recgpt_official_public_format"
JSONL = OUT_DIR / "official_10M_sentence_docs.jsonl"
PARQUET = OUT_DIR / "official_10M_sentence_docs.parquet"
SUMMARY = OUT_DIR / "summary.json"


def sentence_docs(text: str):
    # Sentence-ish split while preserving punctuation; keep chunks large enough to avoid tiny-doc dominance.
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    buf = []
    wc = 0
    for part in parts:
        part = part.strip()
        if not part:
            continue
        w = len(part.split())
        if wc + w > 120 and buf:
            yield " ".join(buf)
            buf = []
            wc = 0
        buf.append(part)
        wc += w
        if wc >= 40 and part.endswith((".", "!", "?")):
            yield " ".join(buf)
            buf = []
            wc = 0
    if buf:
        yield " ".join(buf)


def main():
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    text = SRC.read_text(encoding="utf-8")
    raw_words = len(text.split())
    docs = list(sentence_docs(text))
    doc_words = sum(len(d.split()) for d in docs)
    with JSONL.open("w", encoding="utf-8") as f:
        for d in docs:
            print(json.dumps({"text": d}, ensure_ascii=False), file=f)
    tok = AutoTokenizer.from_pretrained(str(TOK), use_fast=True)
    schema = pa.schema([("input_ids", pa.list_(pa.int32()))])
    writer = pq.ParquetWriter(PARQUET, schema)
    rows = []
    token_sum = 0
    training_token_sum = 0
    for i, d in enumerate(docs, 1):
        ids = tok.encode(d, add_special_tokens=False)
        if len(ids) >= 2:
            rows.append(ids)
            token_sum += len(ids)
            training_token_sum += len(ids) - 1
        if len(rows) >= 1024:
            writer.write_table(pa.table({"input_ids": rows}, schema=schema))
            rows.clear()
    if rows:
        writer.write_table(pa.table({"input_ids": rows}, schema=schema))
    writer.close()
    summary = {
        "status": "OFFICIAL_RECGPT_PUBLIC_FORMAT_READY",
        "source": str(SRC),
        "tokenizer": str(TOK),
        "jsonl": str(JSONL),
        "parquet": str(PARQUET),
        "raw_words": raw_words,
        "doc_count": len(docs),
        "doc_words": doc_words,
        "token_sum": token_sum,
        "public_training_token_count_sum_len_minus_1": training_token_sum,
        "words_per_training_token": raw_words / training_token_sum,
        "training_tokens_per_1M_words": training_token_sum / 10.0,
        "elapsed_sec": round(time.time() - t0, 1),
        "note": "Checkpoint marks for true word exposure should use raw word counts; public code's strict-small marks use epoch_tokens/10 as proxy when dataset has exactly 10M words.",
    }
    SUMMARY.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
