#!/usr/bin/env python3
"""research: train a legal same-pool tokenizer with explicit ByteLevel alphabet.

Why this exists
---------------
The research compliant tokenizer was trained on the correct 10M pool, but the trainer
was not given an explicit byte-level initial alphabet. It consequently maps newline
and some UTF-8 byte symbols to <unk> when scoring official Supplement dialogue/QA
strings. This script trains the same 16k byte-level BPE on exactly the same allowed
10M pool, copying the same normalizer/pre-tokenizer/post-processor/decoder template,
but passes ByteLevel.alphabet() to BpeTrainer to preserve byte-level coverage.

This does not use evaluation text to choose vocabulary. Evaluation text is used only
by the separate audit scripts to test coverage of the resulting tokenizer.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import hashlib
import json
import pathlib
import time

USER_ROOT = _public_path('.')
WS = _public_path('experiments/archive/frontier_consolidation')
POOL_10M = _public_path('experiments/archive/frontier_consolidation/data/density_cleanqwen_overlay_medium_riskhard/cleanqwen_fineweb_compact_view_reinvest_10M.jsonl')
TEMPLATE_TOK = _public_path('experiments/archive/initial_model_studies/training/runs/fullcycle_official_wwm_debertav2_8x480_seed43_100M_b256/hf_model/tokenizer.json')
OUT = _public_path('experiments/archive/frontier_consolidation/data/compliant_tokenizer_bytealphabet')
OUT.mkdir(parents=True, exist_ok=True)
VOCAB_SIZE = 16384
SPECIAL_TOKENS = ["<unk>", "<s>", "</s>", "<pad>", "<mask>"]


def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def text_iterator(path: pathlib.Path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            text = row.get("text", "")
            if text:
                yield text


def count_pool(path: pathlib.Path):
    rows = 0
    words = 0
    chars = 0
    newlines = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            text = row.get("text", "")
            rows += 1
            words += int(row.get("words", 0))
            chars += len(text)
            newlines += text.count("\n")
    return {"rows": rows, "words": words, "chars": chars, "literal_newlines_in_text_fields": newlines}


def main():
    from tokenizers import Tokenizer
    from tokenizers.models import BPE
    from tokenizers.trainers import BpeTrainer
    from tokenizers.pre_tokenizers import ByteLevel
    from transformers import PreTrainedTokenizerFast
    import shutil

    t0 = time.time()
    if not POOL_10M.exists():
        raise FileNotFoundError(POOL_10M)
    if not TEMPLATE_TOK.exists():
        raise FileNotFoundError(TEMPLATE_TOK)

    template = Tokenizer.from_file(str(TEMPLATE_TOK))
    new_tok = Tokenizer(BPE(unk_token="<unk>"))
    new_tok.normalizer = template.normalizer
    new_tok.pre_tokenizer = template.pre_tokenizer
    new_tok.post_processor = template.post_processor
    new_tok.decoder = template.decoder

    trainer = BpeTrainer(
        vocab_size=VOCAB_SIZE,
        special_tokens=SPECIAL_TOKENS,
        min_frequency=2,
        initial_alphabet=ByteLevel.alphabet(),
        show_progress=True,
    )
    print(json.dumps({
        "event": "train_start",
        "pool": str(POOL_10M),
        "pool_sha256": sha256_file(POOL_10M),
        "out": str(OUT),
        "vocab_size": VOCAB_SIZE,
        "initial_alphabet_size": len(ByteLevel.alphabet()),
    }), flush=True)
    new_tok.train_from_iterator(text_iterator(POOL_10M), trainer=trainer)
    tok_file = _public_path('experiments/archive/frontier_consolidation/data/compliant_tokenizer_bytealphabet/tokenizer.json')
    new_tok.save(str(tok_file))

    for name in ["tokenizer_config.json", "special_tokens_map.json"]:
        src = _public_path('experiments/archive/initial_model_studies/training/runs/fullcycle_official_wwm_debertav2_8x480_seed43_100M_b256/hf_model') / name
        if src.exists():
            shutil.copy2(src, OUT / name)

    hf_tok = PreTrainedTokenizerFast(
        tokenizer_file=str(tok_file),
        unk_token="<unk>", bos_token="<s>", eos_token="</s>", pad_token="<pad>", mask_token="<mask>",
    )
    hf_tok.save_pretrained(str(OUT))

    from tokenizers.pre_tokenizers import ByteLevel as BL
    alphabet = set(BL.alphabet())
    vocab = set(new_tok.get_vocab().keys())
    sample = "Who cleaned?\nDavid cleaned. Care of sheets: – machine at 40˚C or Persian فارسی."
    enc = new_tok.encode(sample)
    metadata = {
        "status": "BYTEALPHABET_COMPLIANT_TOKENIZER_TRAINED",
        "scientific_purpose": "Legal 16k byte-level BPE tokenizer trained only on the same 10M Strict-Small reinvest pool, with explicit ByteLevel.alphabet() coverage to avoid unseen-byte <unk> at evaluation.",
        "training_data": {"pool": str(POOL_10M), "pool_sha256": sha256_file(POOL_10M), **count_pool(POOL_10M)},
        "tokenizer": {
            "output_dir": str(OUT),
            "tokenizer_json_sha256": sha256_file(tok_file),
            "vocab_size": new_tok.get_vocab_size(),
            "initial_alphabet_size": len(BL.alphabet()),
            "missing_bytelevel_alphabet": sorted(alphabet - vocab),
            "special_token_ids": {s: new_tok.token_to_id(s) for s in SPECIAL_TOKENS},
        },
        "template_source": str(TEMPLATE_TOK),
        "sample_encode": {"text": sample, "ids": enc.ids, "tokens": enc.tokens, "unk_count": sum(1 for x in enc.ids if x == new_tok.token_to_id("<unk>"))},
        "elapsed_sec": round(time.time() - t0, 3),
    }
    (_public_path('experiments/archive/frontier_consolidation/data/compliant_tokenizer_bytealphabet/tokenizer_metadata.json')).write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({
        "status": metadata["status"],
        "out": str(OUT),
        "tokenizer_json_sha256": metadata["tokenizer"]["tokenizer_json_sha256"],
        "vocab_size": metadata["tokenizer"]["vocab_size"],
        "missing_bytelevel_alphabet": len(metadata["tokenizer"]["missing_bytelevel_alphabet"]),
        "sample_unk_count": metadata["sample_encode"]["unk_count"],
        "elapsed_sec": metadata["elapsed_sec"],
    }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
