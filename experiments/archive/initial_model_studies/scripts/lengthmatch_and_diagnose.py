#!/usr/bin/env python3
"""research: use the structured arm's final per-example word-length sequence to cut official text.

Outputs:
- official_lengthmatched_9999969w_62720ex.jsonl: official corpus cut to exactly
  the structured symmetric corpus's 62,720 word-length sequence.
- lengthmatch_tokenization_diagnostics.json: tokenizer/truncation comparison under
  baseline16k and max_seq_length=256 before launching the four-arm screen.
"""
from __future__ import annotations
import json, pathlib, statistics, sys

ROOT = pathlib.Path("experiments/archive/initial_model_studies")
RAW_DIR = ROOT / "data/reconstruct_tmp/raw_dataset"
SCRIPTS = ROOT / "training/scripts"
sys.path.insert(0, str(SCRIPTS))
from babylm_masked_train_fullcycle import Example, make_portable_tokenizer, summarize_tokenization_coupling

STRUCT = ROOT / "data/symmetric_batch_corpora/structured_symmetric_9999969w_62720ex.jsonl"
OUT_DIR = ROOT / "data/lengthmatched_corpora"
OFF = OUT_DIR / "official_lengthmatched_9999969w_62720ex.jsonl"
DIAG = OUT_DIR / "lengthmatch_tokenization_diagnostics.json"
ALL_SOURCES = ["bnc_spoken.train.txt", "childes.train.txt", "gutenberg.train.txt", "open_subtitles.train.txt", "simple_wiki.train.txt", "switchboard.train.txt"]
TARGET_WORDS = 9_999_969
MAX_SEQ = 256


def read_jsonl_examples(path: pathlib.Path, source_default: str) -> list[Example]:
    exs=[]
    with path.open(encoding="utf-8") as f:
        for i,line in enumerate(f):
            if not line.strip(): continue
            o=json.loads(line); text=str(o["text"]); words=int(o.get("words", len(text.split())))
            if words != len(text.split()): raise RuntimeError(f"word mismatch {path}:{i}")
            exs.append(Example(text=text, words=words, example_id=i, source=str(o.get("source", source_default))))
    return exs


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    structured = read_jsonl_examples(STRUCT, "structured")
    lens = [e.words for e in structured]
    if sum(lens) != TARGET_WORDS or len(lens) != 62720:
        raise RuntimeError(f"bad structured geometry words={sum(lens)} examples={len(lens)}")
    words=[]
    for fn in ALL_SOURCES:
        for line in (RAW_DIR/fn).read_text(encoding="utf-8", errors="replace").splitlines():
            words.extend(line.split())
    words = words[:TARGET_WORDS]
    if len(words) != TARGET_WORDS: raise RuntimeError(f"official words {len(words)}")
    off_examples=[]; pos=0
    with OFF.open("w", encoding="utf-8") as f:
        for i,L in enumerate(lens):
            chunk=words[pos:pos+L]; pos += L
            text=" ".join(chunk)
            o={"text": text, "words": L, "source": "official_lengthmatched", "example_id": i}
            f.write(json.dumps(o, ensure_ascii=False)+"\n")
            off_examples.append(Example(text=text, words=L, example_id=i, source="official_lengthmatched"))
    if pos != TARGET_WORDS: raise RuntimeError(f"consumed {pos}")
    tok = make_portable_tokenizer("")
    off_sum = summarize_tokenization_coupling(off_examples, tok, MAX_SEQ, sample_limit=0)
    st_sum = summarize_tokenization_coupling(structured, tok, MAX_SEQ, sample_limit=0)
    def lens_summary(xs):
        return {"n": len(xs), "sum": sum(xs), "min": min(xs), "max": max(xs), "mean": statistics.mean(xs), "median": statistics.median(xs), "p95": sorted(xs)[int(0.95*(len(xs)-1))]}
    diag={
        "status": "LENGTHMATCHED_OFFICIAL_BUILT",
        "official_lengthmatched_output": str(OFF),
        "structured_input": str(STRUCT),
        "same_word_length_sequence": [e.words for e in off_examples] == lens,
        "word_length_summary": lens_summary(lens),
        "official_tokenization": off_sum,
        "structured_tokenization": st_sum,
        "differences_structured_minus_official": {
            "untruncated_tokens_per_whitespace_word": st_sum["untruncated_tokens_per_whitespace_word"] - off_sum["untruncated_tokens_per_whitespace_word"],
            "kept_tokens_per_whitespace_word": st_sum["kept_tokens_per_whitespace_word"] - off_sum["kept_tokens_per_whitespace_word"],
            "word_groups_kept_per_whitespace_word": st_sum["word_groups_kept_per_whitespace_word"] - off_sum["word_groups_kept_per_whitespace_word"],
            "truncated_example_fraction": st_sum["truncated_example_fraction"] - off_sum["truncated_example_fraction"],
            "total_tokens_lost_to_truncation": st_sum["total_tokens_lost_to_truncation"] - off_sum["total_tokens_lost_to_truncation"],
        },
    }
    DIAG.write_text(json.dumps(diag, indent=2, ensure_ascii=False)+"\n")
    print(json.dumps({"official_out": str(OFF), "diag": str(DIAG), "same_lengths": diag["same_word_length_sequence"], "official_trunc_frac": off_sum["truncated_example_fraction"], "structured_trunc_frac": st_sum["truncated_example_fraction"], "delta_kept_tokens_per_word": diag["differences_structured_minus_official"]["kept_tokens_per_whitespace_word"]}, indent=2))

if __name__ == "__main__":
    main()
