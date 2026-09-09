#!/usr/bin/env python3
"""research: Train a 40k SentencePiece BPE tokenizer on mixture data.

The leader uses a 40k SentencePiece BPE tokenizer trained on its specific
paired data. This script trains a 40k tokenizer on our best mixture pool,
which should give similar morphological coverage benefits.

Key design decisions:
- 40,000 vocabulary size (matching leader)
- BPE algorithm (matching leader's SentencePiece BPE)
- Trained on the mixture pool text (not just official or just paired)
- Includes special tokens: <unk>, <s>, </s>, <pad>, <mask>
- character_coverage=1.0 for English text
- No pre-tokenization splitting (let SentencePiece handle everything)

Usage:
  python train_40k_tokenizer.py --data_source mix_50pct
  python train_40k_tokenizer.py --data_source official+aligned
"""
from __future__ import annotations
import argparse
import json
import tempfile
import time
from pathlib import Path

# Will use sentencepiece library
# import sentencepiece as spm

ROOT = Path("experiments/archive/compact_experience")
MIXTURE_DIR = ROOT / "data/mixture"
OUT_DIR = ROOT / "data/tokenizer"


def extract_training_text(data_source: str) -> Path:
    """Extract training text from the specified source."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    text_path = OUT_DIR / f"tokenizer_train_text_{data_source}.txt"
    
    if text_path.exists():
        print(f"  Training text already exists: {text_path}")
        return text_path
    
    # Combine text from specified pools
    if data_source.startswith("mix_"):
        pool_path = MIXTURE_DIR / f"{data_source}_pool.jsonl"
        sources = [pool_path]
    elif data_source == "official+aligned":
        sources = [
            MIXTURE_DIR / "official_pool.jsonl",
            ROOT / "data/paired_alignment/aligned_pool.jsonl",
        ]
    elif data_source == "official":
        sources = [MIXTURE_DIR / "official_pool.jsonl"]
    elif data_source == "aligned":
        sources = [ROOT / "data/paired_alignment/aligned_pool.jsonl"]
    else:
        raise ValueError(f"Unknown data source: {data_source}")
    
    print(f"  Extracting text from: {[str(s) for s in sources]}")
    n_lines = 0
    n_words = 0
    with text_path.open("w", encoding="utf-8") as out:
        for src in sources:
            with src.open() as f:
                for line in f:
                    row = json.loads(line)
                    text = row["text"]
                    out.write(text + "\n")
                    n_lines += 1
                    n_words += len(text.split())
    
    print(f"  Written: {text_path} ({n_lines} lines, {n_words} words)")
    return text_path


def train_tokenizer(text_path: Path, vocab_size: int = 40000, data_source: str = "mix_50pct"):
    """Train a SentencePiece BPE tokenizer."""
    import sentencepiece as spm
    
    model_prefix = str(OUT_DIR / f"sp_bpe_{vocab_size}_{data_source}")
    
    print(f"\n  Training SentencePiece BPE tokenizer...")
    print(f"    Vocab size: {vocab_size}")
    print(f"    Input: {text_path}")
    print(f"    Output prefix: {model_prefix}")
    
    spm.SentencePieceTrainer.train(
        input=str(text_path),
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        model_type="bpe",
        character_coverage=1.0,
        # Standard special tokens matching the leader
        pad_id=0,
        unk_id=1,  # will be remapped
        bos_id=2,
        eos_id=3,
        # Add mask token
        user_defined_symbols=["<mask>"],
        # BPE-specific settings
        byte_fallback=True,
        split_digits=True,
        # Training parameters
        num_threads=4,
        train_extremely_large_corpus=False,
    )
    
    # Verify
    sp = spm.SentencePieceProcessor()
    sp.load(f"{model_prefix}.model")
    
    print(f"\n  Tokenizer trained successfully!")
    print(f"    Vocab size: {sp.get_piece_size()}")
    
    # Test tokenization
    test_sentences = [
        "The children played happily in the garden.",
        "unbelievably extraordinary circumstances",
        "A young boy offers water to a sea turtle.",
        "He painted his eyes with a red band and was bald.",
    ]
    
    for sent in test_sentences:
        pieces = sp.encode(sent, out_type=str)
        print(f"    '{sent[:40]}...' → {len(pieces)} pieces: {pieces[:8]}...")
    
    return model_prefix


def convert_to_hf_tokenizer(model_prefix: str, vocab_size: int = 40000):
    """Convert SentencePiece model to HuggingFace tokenizer format."""
    from transformers import DebertaV2Tokenizer, PreTrainedTokenizerFast
    
    model_path = f"{model_prefix}.model"
    out_dir = OUT_DIR / f"hf_tokenizer_40k"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # DebertaV2Tokenizer natively supports SentencePiece
    tok = DebertaV2Tokenizer(
        vocab_file=model_path,
        do_lower_case=False,
        bos_token="<s>",
        eos_token="</s>",
        unk_token="<unk>",
        sep_token="</s>",
        pad_token="<pad>",
        cls_token="<s>",
        mask_token="<mask>",
    )
    
    tok.save_pretrained(str(out_dir))
    print(f"\n  HuggingFace tokenizer saved: {out_dir}")
    print(f"    Vocab size: {tok.vocab_size}")
    
    # Test
    test = "The children played happily in the garden."
    ids = tok(test)["input_ids"]
    tokens = tok.convert_ids_to_tokens(ids)
    print(f"    Test: {tokens[:10]}...")
    
    return out_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_source", default="official+aligned",
                        help="Data source for tokenizer training")
    parser.add_argument("--vocab_size", type=int, default=40000)
    args = parser.parse_args()
    
    start = time.time()
    print("=" * 60)
    print(f"Training {args.vocab_size} SentencePiece BPE tokenizer")
    print(f"Data source: {args.data_source}")
    print("=" * 60)
    
    # 1. Extract text
    print("\n1. Extracting training text:")
    text_path = extract_training_text(args.data_source)
    
    # 2. Train tokenizer
    print("\n2. Training tokenizer:")
    model_prefix = train_tokenizer(text_path, args.vocab_size, args.data_source)
    
    # 3. Convert to HF format
    print("\n3. Converting to HuggingFace format:")
    hf_dir = convert_to_hf_tokenizer(model_prefix, args.vocab_size)
    
    elapsed = time.time() - start
    print(f"\n{'=' * 60}")
    print(f"Done in {elapsed:.1f}s")
    print(f"Tokenizer: {hf_dir}")
    
    # Save metadata
    meta = {
        "status": "reference_40K_TOKENIZER_TRAINED",
        "data_source": args.data_source,
        "vocab_size": args.vocab_size,
        "model_type": "bpe",
        "sentencepiece_model": f"{model_prefix}.model",
        "sentencepiece_vocab": f"{model_prefix}.vocab",
        "hf_tokenizer_dir": str(hf_dir),
        "elapsed_sec": round(elapsed, 1),
    }
    meta_path = OUT_DIR / "tokenizer_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Metadata: {meta_path}")


if __name__ == "__main__":
    main()
