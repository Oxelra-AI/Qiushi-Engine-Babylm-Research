#!/usr/bin/env python3
"""research: train a shared 40k SentencePiece BPE tokenizer on the 7.5M official
subset that BOTH Phase-2 arms (official-only and mix_25pct) contain.

Tokenizer control rationale: if the tokenizer is trained on
official+aligned text, the official-only arm would inherit aligned subword
statistics through the shared tokenizer, confounding the data comparison. The
7.5M official subset is common to both pools and is a strict subset of the
official 10M budget, so it is a fair, budget-legal shared tokenizer basis.
"""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path

ROOT = Path("experiments/archive/compact_experience")
OUT_DIR = ROOT / "data/shared_tokenizer"


def train_tokenizer(text_path: Path, vocab_size: int, tag: str) -> str:
    import sentencepiece as spm
    model_prefix = str(OUT_DIR / f"sp_bpe_{vocab_size}_{tag}")
    spm.SentencePieceTrainer.train(
        input=str(text_path),
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        model_type="bpe",
        character_coverage=1.0,
        pad_id=0, unk_id=1, bos_id=2, eos_id=3,
        user_defined_symbols=["<mask>"],
        byte_fallback=True,
        split_digits=True,
        num_threads=4,
        train_extremely_large_corpus=False,
    )
    sp = spm.SentencePieceProcessor(); sp.load(f"{model_prefix}.model")
    print("vocab", sp.get_piece_size())
    for s in ["The children played happily in the garden.",
              "unbelievably extraordinary circumstances"]:
        print("  ", len(sp.encode(s, out_type=str)), sp.encode(s, out_type=str)[:8])
    return model_prefix


def convert_to_hf(model_prefix: str, out_name: str) -> Path:
    from transformers import DebertaV2Tokenizer
    out_dir = OUT_DIR / out_name
    out_dir.mkdir(parents=True, exist_ok=True)
    tok = DebertaV2Tokenizer(
        vocab_file=f"{model_prefix}.model", do_lower_case=False,
        bos_token="<s>", eos_token="</s>", unk_token="<unk>",
        sep_token="</s>", pad_token="<pad>", cls_token="<s>", mask_token="<mask>",
    )
    tok.save_pretrained(str(out_dir))
    print("saved HF tokenizer", out_dir, "vocab", tok.vocab_size)
    return out_dir


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--text", default=str(OUT_DIR / "official_subset_7p5M.txt"))
    p.add_argument("--vocab_size", type=int, default=40000)
    p.add_argument("--tag", default="official7p5M")
    p.add_argument("--hf_name", default="hf_tokenizer_40k_shared")
    args = p.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    prefix = train_tokenizer(Path(args.text), args.vocab_size, args.tag)
    hf = convert_to_hf(prefix, args.hf_name)
    meta = {"status": "SHARED_40K_TOKENIZER_TRAINED",
            "trained_on": args.text, "trained_on_words": 7500000,
            "vocab_size": args.vocab_size, "hf_tokenizer_dir": str(hf),
            "elapsed_sec": round(time.time() - t0, 1)}
    (OUT_DIR / "shared_tokenizer_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
