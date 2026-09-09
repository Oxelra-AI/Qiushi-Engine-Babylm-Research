#!/usr/bin/env python3
"""Train a local 40k tokenizer on the official BabyLM Strict-Small corpus.

This is for the research tokenizer-granularity experiment. The tokenizer is trained
only on the official 10M-word corpus and writes a manifest documenting that text
exposure, file hashes, special tokens, vocab size, and a baseline-vs-40k
coupling sample. It saves as a portable Hugging Face fast tokenizer so
`babylm_masked_train.py --tokenizer_path` can load it.
"""
from __future__ import annotations

import hashlib
import json
import os
import pathlib
from statistics import mean

from huggingface_hub import snapshot_download
from tokenizers import ByteLevelBPETokenizer
from transformers import AutoTokenizer, PreTrainedTokenizerFast

STUDY = pathlib.Path("experiments/archive/initial_model_studies")
OUT = STUDY / "training/tokenizers/official40k"
RAW = OUT / "raw_dataset"
DATASET_ID = "BabyLM-community/BabyLM-2026-Strict-Small"
DATASET_REVISION = "c92ab16b4f08858304b0815706065b3354d8fc0a"
BASELINE_TOKENIZER_REPO = "BabyLM-community/BabyLM-2026-Baseline-GPT2-Strict"
TRAIN_FILES = [
    "bnc_spoken.train.txt",
    "childes.train.txt",
    "gutenberg.train.txt",
    "open_subtitles.train.txt",
    "simple_wiki.train.txt",
    "switchboard.train.txt",
]
SPECIAL_TOKENS = ["<s>", "<pad>", "</s>", "<unk>", "<mask>"]


def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def count_words(path: pathlib.Path) -> int:
    n = 0
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            n += len(line.split())
    return n


def sample_lines(files: list[pathlib.Path], max_lines: int = 2000) -> list[str]:
    rows: list[str] = []
    for fp in files:
        with fp.open("r", encoding="utf-8", errors="replace") as f:
            for line in f:
                s = line.strip()
                if s:
                    rows.append(s)
                    if len(rows) >= max_lines:
                        return rows
    return rows


def token_summary(tok, texts: list[str]) -> dict:
    lens = []
    word_counts = []
    for t in texts:
        word_counts.append(len(t.split()))
        lens.append(len(tok(t, add_special_tokens=False)["input_ids"]))
    total_words = max(1, sum(word_counts))
    return {
        "num_texts": len(texts),
        "total_words": sum(word_counts),
        "total_tokens": sum(lens),
        "tokens_per_whitespace_word": sum(lens) / total_words,
        "mean_tokens_per_text": mean(lens) if lens else 0.0,
        "max_tokens_per_text": max(lens) if lens else 0,
        "hist": {
            "<=64": sum(1 for x in lens if x <= 64),
            "65-128": sum(1 for x in lens if 64 < x <= 128),
            "129-256": sum(1 for x in lens if 128 < x <= 256),
            ">256": sum(1 for x in lens if x > 256),
        },
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    RAW.mkdir(parents=True, exist_ok=True)
    local = pathlib.Path(snapshot_download(
        repo_id=DATASET_ID,
        repo_type="dataset",
        revision=DATASET_REVISION,
        allow_patterns=TRAIN_FILES + ["README.md"],
        local_dir=RAW,
        local_dir_use_symlinks=False,
    ))
    files = [local / n for n in TRAIN_FILES]
    manifest_files = []
    for fp in files:
        if not fp.exists():
            raise FileNotFoundError(fp)
        manifest_files.append({
            "name": fp.name,
            "path": str(fp),
            "bytes": fp.stat().st_size,
            "sha256": sha256_file(fp),
            "whitespace_words": count_words(fp),
        })
    total_words = sum(x["whitespace_words"] for x in manifest_files)

    # Train a GPT2-like byte-level BPE so WWM word-start detection via Ġ remains meaningful.
    tokenizer = ByteLevelBPETokenizer(add_prefix_space=True)
    tokenizer.train(
        files=[str(p) for p in files],
        vocab_size=40000,
        min_frequency=2,
        special_tokens=SPECIAL_TOKENS,
        show_progress=True,
    )
    tokenizer.save_model(str(OUT), prefix="official40k")
    tokenizer.save(str(OUT / "tokenizer.json"))
    fast = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer._tokenizer,
        bos_token="<s>",
        pad_token="<pad>",
        eos_token="</s>",
        unk_token="<unk>",
        mask_token="<mask>",
        model_max_length=1024,
    )
    fast.save_pretrained(OUT)
    tok_cfg = OUT / "tokenizer_config.json"
    cfg = json.loads(tok_cfg.read_text(encoding="utf-8"))
    cfg["tokenizer_class"] = "PreTrainedTokenizerFast"
    cfg["add_prefix_space"] = True
    tok_cfg.write_text(json.dumps(cfg, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    # Quick comparative summary on deterministic sample, not the full expensive 1M run.
    baseline = AutoTokenizer.from_pretrained(BASELINE_TOKENIZER_REPO, revision="main", use_fast=True)
    loaded = AutoTokenizer.from_pretrained(OUT, use_fast=True)
    texts = sample_lines(files, 2000)
    example = "The young child remembered the beautiful garden yesterday."
    manifest = {
        "tokenizer_label": "official40k_bytelevel_bpe",
        "tokenizer_type": "ByteLevelBPE(add_prefix_space=True)",
        "vocab_size_requested": 40000,
        "vocab_size_actual": len(loaded),
        "special_tokens": SPECIAL_TOKENS,
        "dataset_id": DATASET_ID,
        "dataset_revision": DATASET_REVISION,
        "files": manifest_files,
        "total_whitespace_words_charged_to_tokenizer_training": total_words,
        "training_notes": "Tokenizer trained only on official Strict-Small corpus; tokenizer training text must be counted as part of allowed language exposure when judging final submissions.",
        "sample_comparison_2000_nonblank_lines": {
            "baseline16k": token_summary(baseline, texts),
            "official40k": token_summary(loaded, texts),
        },
        "example_tokens": {
            "text": example,
            "baseline16k": baseline.convert_ids_to_tokens(baseline(example, add_special_tokens=False)["input_ids"]),
            "official40k": loaded.convert_ids_to_tokens(loaded(example, add_special_tokens=False)["input_ids"]),
        },
        "paths": {
            "tokenizer_dir": str(OUT),
            "tokenizer_json": str(OUT / "tokenizer.json"),
            "vocab_file": str(OUT / "official40k-vocab.json"),
            "merges_file": str(OUT / "official40k-merges.txt"),
        },
    }
    (OUT / "tokenizer_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": "OFFICIAL40K_TOKENIZER_READY",
        "out": str(OUT),
        "vocab_size_actual": len(loaded),
        "total_words": total_words,
        "baseline_sample_tokens_per_word": manifest["sample_comparison_2000_nonblank_lines"]["baseline16k"]["tokens_per_whitespace_word"],
        "official40k_sample_tokens_per_word": manifest["sample_comparison_2000_nonblank_lines"]["official40k"]["tokens_per_whitespace_word"],
    }, indent=2))


if __name__ == "__main__":
    main()
