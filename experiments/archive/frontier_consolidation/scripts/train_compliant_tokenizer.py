#!/usr/bin/env python3
"""research: Train a compliant 16k BPE tokenizer from the 10M Strict-Small pool.

The current baseline16k tokenizer was trained on the 100M Strict corpus.
BabyLM Strict-Small rules require tokenizer training data counts toward
the 10M word budget. This script trains an equivalent 16k BPE tokenizer
using ONLY the compact_view_reinvest 10M pool.

Preserves exact same tokenizer architecture:
  - BPE byte-level model
  - 16,384 vocab size (including 5 special tokens)
  - Same normalizer (Prepend + NFKC + newline handling)
  - Same pre-tokenizer (regex split + ByteLevel + 24-char chunk)
  - Same post-processor (TemplateProcessing with <s>/<\s>)
  - Same decoder (ByteLevel + Strip + newline)
  - Same special tokens: <unk>(0), <s>(1), </s>(2), <pad>(3), <mask>(4)
"""
import json, hashlib, os, sys, time
from pathlib import Path

ROOT = Path("experiments/archive/frontier_consolidation")
POOL_10M = ROOT / "data/density_cleanqwen_overlay_medium_riskhard/cleanqwen_fineweb_compact_view_reinvest_10M.jsonl"
TEMPLATE_TOK = Path("experiments/archive/initial_model_studies/training/runs/fullcycle_official_wwm_debertav2_8x480_seed43_100M_b256/hf_model/tokenizer.json")
OUT = ROOT / "data/compliant_tokenizer"
OUT.mkdir(parents=True, exist_ok=True)

VOCAB_SIZE = 16384
SPECIAL_TOKENS = ["<unk>", "<s>", "</s>", "<pad>", "<mask>"]


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def text_iterator(jsonl_path: Path):
    """Yield text fields from the JSONL pool."""
    with open(jsonl_path) as f:
        for line in f:
            row = json.loads(line)
            text = row.get("text", "")
            if text:
                yield text


def main():
    print(f"Training compliant 16k tokenizer from 10M pool")
    print(f"  Pool: {POOL_10M}")
    print(f"  Template: {TEMPLATE_TOK}")
    print(f"  Output: {OUT}")
    
    # Verify source exists
    assert POOL_10M.exists(), f"Pool not found: {POOL_10M}"
    assert TEMPLATE_TOK.exists(), f"Template not found: {TEMPLATE_TOK}"
    
    pool_sha = sha256_file(POOL_10M)
    print(f"  Pool SHA256: {pool_sha}")
    
    # Count pool stats
    n_rows = 0
    n_words = 0
    with open(POOL_10M) as f:
        for line in f:
            row = json.loads(line)
            n_rows += 1
            n_words += row.get("words", 0)
    print(f"  Pool: {n_rows} rows, {n_words} words")
    
    # Load the template tokenizer to extract configuration
    from tokenizers import Tokenizer
    from tokenizers.models import BPE
    from tokenizers.trainers import BpeTrainer
    
    template = Tokenizer.from_file(str(TEMPLATE_TOK))
    
    # Create new tokenizer with same configuration but empty BPE
    new_tok = Tokenizer(BPE(unk_token="<unk>"))
    new_tok.normalizer = template.normalizer
    new_tok.pre_tokenizer = template.pre_tokenizer
    new_tok.post_processor = template.post_processor
    new_tok.decoder = template.decoder
    
    # Train BPE on the 10M pool text
    trainer = BpeTrainer(
        vocab_size=VOCAB_SIZE,
        special_tokens=SPECIAL_TOKENS,
        min_frequency=2,
        show_progress=True,
    )
    
    print(f"\nTraining tokenizer...")
    t0 = time.time()
    new_tok.train_from_iterator(text_iterator(POOL_10M), trainer=trainer)
    elapsed = time.time() - t0
    print(f"  Training took {elapsed:.1f}s")
    
    # Verify
    actual_vocab = new_tok.get_vocab_size()
    print(f"  Vocab size: {actual_vocab}")
    
    # Verify special tokens
    for st in SPECIAL_TOKENS:
        tid = new_tok.token_to_id(st)
        print(f"  {st}: id={tid}")
    
    # Save the tokenizer
    tok_file = OUT / "tokenizer.json"
    new_tok.save(str(tok_file))
    
    # Also save the HuggingFace tokenizer config and special_tokens_map
    # Copy tokenizer_config.json from template (same structure)
    template_cfg_path = TEMPLATE_TOK.parent / "tokenizer_config.json"
    if template_cfg_path.exists():
        import shutil
        shutil.copy2(template_cfg_path, OUT / "tokenizer_config.json")
    
    template_spm_path = TEMPLATE_TOK.parent / "special_tokens_map.json"
    if template_spm_path.exists():
        import shutil
        shutil.copy2(template_spm_path, OUT / "special_tokens_map.json")
    
    # Test: encode a sample sentence
    test_text = "The cat sat on the mat."
    encoded = new_tok.encode(test_text)
    print(f"\n  Test encode: '{test_text}' -> {encoded.ids[:20]}")
    print(f"  Tokens: {encoded.tokens[:20]}")
    decoded = new_tok.decode(encoded.ids)
    print(f"  Decoded: '{decoded}'")
    
    # Compare with template tokenizer
    template_enc = template.encode(test_text)
    print(f"  Template encode: {template_enc.ids[:20]}")
    print(f"  Template tokens: {template_enc.tokens[:20]}")
    
    # Verify with HuggingFace PreTrainedTokenizerFast
    from transformers import PreTrainedTokenizerFast
    hf_tok = PreTrainedTokenizerFast(
        tokenizer_file=str(tok_file),
        unk_token="<unk>",
        bos_token="<s>",
        eos_token="</s>",
        pad_token="<pad>",
        mask_token="<mask>",
    )
    hf_enc = hf_tok(test_text)
    print(f"  HF encode: {hf_enc['input_ids'][:20]}")
    print(f"  HF vocab size: {hf_tok.vocab_size}")
    
    # Save HF tokenizer
    hf_tok.save_pretrained(str(OUT))
    
    tok_sha = sha256_file(tok_file)
    
    metadata = {
        "status": "COMPLIANT_TOKENIZER_TRAINED",
        "scientific_purpose": "Train a compliant 16k BPE tokenizer from the 10M Strict-Small pool only, replacing the 100M-trained baseline16k.",
        "training_data": {
            "pool": str(POOL_10M),
            "pool_sha256": pool_sha,
            "pool_rows": n_rows,
            "pool_words": n_words,
        },
        "tokenizer": {
            "vocab_size": actual_vocab,
            "type": "BPE byte-level",
            "special_tokens": SPECIAL_TOKENS,
            "output_dir": str(OUT),
            "tokenizer_json_sha256": tok_sha,
        },
        "template_source": str(TEMPLATE_TOK),
        "elapsed_sec": round(elapsed, 1),
    }
    
    (OUT / "tokenizer_metadata.json").write_text(json.dumps(metadata, indent=2))
    print(f"\nDone. Tokenizer saved to {OUT}")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
