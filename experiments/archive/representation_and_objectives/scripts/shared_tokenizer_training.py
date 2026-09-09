#!/usr/bin/env python3
"""research: Build shared compliant 16k BPE tokenizer for the FW mechanism-scale comparison.

The tokenizer is trained on the ~9.68M-word common text pool:
- All official BabyLM rows (8,343,200 words)
- Retained Qwen rows (843,775 words)  
- FineWeb source texts only — NOT compact rewrites or source repeats (494,154 words)
- Neutral topup (20 words)

This ensures the tokenizer does not learn from text that differs between
compact_view and source_repeat arms (i.e., the companion-specific text).

Uses the same GPT2-like byte-level BPE approach as the legal16k tokenizer
from research, with vocab_size=16384 and min_frequency=2.
"""
import json, pathlib, hashlib, time, sys, collections

WORKSPACE = pathlib.Path("experiments/archive/representation_and_objectives")
ARM_FILE = WORKSPACE / "data/fw_full_arms/fw_preserved_compact_view_10M.jsonl"
USABLE_PAIRS = WORKSPACE / "data/fw_full_preservation/full26k_usable_pairs_for_materializer.jsonl"
BASE_TOKENIZER = WORKSPACE / "data/strictsmall_tokenizer_retrain/strictsmall_compact_reinvest_16k_tokenizer"
OUT_DIR = WORKSPACE / "data/shared_tokenizer"
POOL_FILE = OUT_DIR / "shared_tokenizer_pool.jsonl"
TOK_DIR = OUT_DIR / "shared_16k_tokenizer"
NOTE_PATH = (WORKSPACE.parents[2] / 'research/notes/representation_and_objectives/shared_tokenizer.md')

VOCAB_SIZE = 16384
MIN_FREQ = 2

FW_SOURCES = frozenset(["fw_preserved_compact_view", "fw_preserved_source_repeat"])

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1<<20), b""):
            h.update(chunk)
    return h.hexdigest()

def wc(text):
    return len(text.split())

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    TOK_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    
    # ── Phase 1: Build shared tokenizer pool ──
    print("Phase 1: Building shared tokenizer pool...")
    
    # 1a: Extract non-FineWeb rows from the arm file (official + Qwen + neutral)
    pool_rows = []
    pool_words = 0
    source_counts = collections.Counter()
    
    with open(ARM_FILE) as f:
        for line in f:
            r = json.loads(line)
            src = r.get("source", "")
            if src not in FW_SOURCES:
                pool_rows.append({"text": r["text"], "words": r["words"], "source": src})
                pool_words += r["words"]
                source_counts[src] += r["words"]
    
    print(f"  Non-FineWeb rows: {len(pool_rows)}, {pool_words} words")
    for src, w in source_counts.most_common():
        print(f"    {src}: {w} words")
    
    # 1b: Extract FineWeb source texts (NOT rewrites) from usable pairs
    fw_source_words = 0
    with open(USABLE_PAIRS) as f:
        for line in f:
            r = json.loads(line)
            src_text = r.get("source_text", "")
            if src_text:
                sw = wc(src_text)
                pool_rows.append({"text": src_text, "words": sw, "source": "fw_source_only"})
                fw_source_words += sw
    
    print(f"  FineWeb source texts: {fw_source_words} words")
    pool_words += fw_source_words
    print(f"  Total shared pool: {len(pool_rows)} rows, {pool_words} words")
    
    # Save the pool
    with open(POOL_FILE, "w") as f:
        for r in pool_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    pool_sha = sha256_file(POOL_FILE)
    print(f"  Saved: {POOL_FILE} (SHA: {pool_sha[:16]}...)")
    
    # ── Phase 2: Train the tokenizer ──
    print("\nPhase 2: Training 16k BPE tokenizer...")
    
    if not BASE_TOKENIZER.exists():
        print(f"ERROR: Base tokenizer not found at {BASE_TOKENIZER}")
        sys.exit(1)
    
    from transformers import AutoTokenizer
    
    base = AutoTokenizer.from_pretrained(str(BASE_TOKENIZER), use_fast=True)
    print(f"  Base tokenizer loaded: vocab={len(base)}, type={type(base).__name__}")
    
    def text_iterator(batch_size=1000):
        batch = []
        with open(POOL_FILE) as f:
            for line in f:
                r = json.loads(line)
                batch.append(r["text"])
                if len(batch) >= batch_size:
                    yield batch
                    batch = []
        if batch:
            yield batch
    
    train_t0 = time.time()
    new_tok = base.train_new_from_iterator(
        text_iterator(batch_size=1000),
        vocab_size=VOCAB_SIZE,
        length=len(pool_rows),
        new_special_tokens=[],
        min_frequency=MIN_FREQ,
        show_progress=True,
    )
    new_tok.save_pretrained(str(TOK_DIR))
    train_elapsed = time.time() - train_t0
    print(f"  Tokenizer trained in {train_elapsed:.1f}s, saved to {TOK_DIR}")
    
    # Update config
    tok_cfg_path = TOK_DIR / "tokenizer_config.json"
    if tok_cfg_path.exists():
        cfg = json.loads(tok_cfg_path.read_text())
    else:
        cfg = {}
    cfg["tokenizer_class"] = "PreTrainedTokenizerFast"
    cfg["model_max_length"] = 1024
    cfg["strict_small_shared_tokenizer_pool"] = str(POOL_FILE)
    cfg["strict_small_shared_tokenizer_pool_words"] = pool_words
    cfg["strict_small_shared_tokenizer_purpose"] = "shared across compact_view and source_repeat arms"
    tok_cfg_path.write_text(json.dumps(cfg, indent=2) + "\n")
    
    # ── Phase 3: Validate ──
    print("\nPhase 3: Validating...")
    loaded = AutoTokenizer.from_pretrained(str(TOK_DIR), use_fast=True)
    
    # Check vocab size
    actual_vocab = len(loaded)
    print(f"  Actual vocab size: {actual_vocab}")
    
    # Check special tokens
    print(f"  Special tokens: {loaded.special_tokens_map}")
    print(f"  Special IDs: {loaded.all_special_ids}")
    
    # Check no UNK on sample pool text
    unk_count = 0
    total_tokens = 0
    total_words_checked = 0
    n_checked = 0
    with open(POOL_FILE) as f:
        for line in f:
            if n_checked >= 5000:
                break
            r = json.loads(line)
            ids = loaded(r["text"], add_special_tokens=False)["input_ids"]
            total_tokens += len(ids)
            total_words_checked += r["words"]
            unk_id = loaded.unk_token_id
            if unk_id is not None:
                unk_count += ids.count(unk_id)
            n_checked += 1
    
    tpw = total_tokens / total_words_checked if total_words_checked else 0
    print(f"  Sample tokens/word: {tpw:.4f} (from {n_checked} rows, {total_words_checked} words)")
    print(f"  UNK tokens in sample: {unk_count}")
    
    # Check ByteLevel coverage
    byte_tokens = set()
    for i in range(256):
        byte_tokens.add(chr(i))
    missing_bytes = 0
    for b in range(256):
        text = bytes([b]).decode("latin1")
        ids = loaded(text, add_special_tokens=False)["input_ids"]
        if not ids:
            missing_bytes += 1
    print(f"  Missing byte-level entries: {missing_bytes}")
    
    # Compare with base tokenizer on same sample
    base_tpw_total = 0
    base_words_total = 0
    with open(POOL_FILE) as f:
        for i, line in enumerate(f):
            if i >= 2000:
                break
            r = json.loads(line)
            base_ids = base(r["text"], add_special_tokens=False)["input_ids"]
            base_tpw_total += len(base_ids)
            base_words_total += r["words"]
    base_tpw = base_tpw_total / base_words_total if base_words_total else 0
    new_tpw_2k = 0
    new_words_2k = 0
    with open(POOL_FILE) as f:
        for i, line in enumerate(f):
            if i >= 2000:
                break
            r = json.loads(line)
            new_ids = loaded(r["text"], add_special_tokens=False)["input_ids"]
            new_tpw_2k += len(new_ids)
            new_words_2k += r["words"]
    new_tpw = new_tpw_2k / new_words_2k if new_words_2k else 0
    print(f"  Base legal16k tokens/word (2k rows): {base_tpw:.4f}")
    print(f"  New shared16k tokens/word (2k rows): {new_tpw:.4f}")
    
    # Save tokenizer hashes
    tok_hashes = {}
    for p in sorted(TOK_DIR.glob("*")):
        if p.is_file():
            tok_hashes[p.name] = sha256_file(p)
    
    # ── Save manifest ──
    manifest = {
        "status": "SHARED_TOKENIZER_READY",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "purpose": "shared across compact_view/source_repeat for FW mechanism-scale comparison",
        "pool": {
            "file": str(POOL_FILE),
            "sha256": pool_sha,
            "rows": len(pool_rows),
            "words": pool_words,
            "source_breakdown": dict(source_counts) | {"fw_source_only": fw_source_words},
        },
        "tokenizer": {
            "dir": str(TOK_DIR),
            "vocab_size_requested": VOCAB_SIZE,
            "vocab_size_actual": actual_vocab,
            "min_frequency": MIN_FREQ,
            "base_tokenizer": str(BASE_TOKENIZER),
            "special_tokens_map": loaded.special_tokens_map,
            "special_ids": loaded.all_special_ids,
            "tokens_per_word_sample": round(tpw, 4),
            "unk_count_in_sample": unk_count,
            "missing_bytes": missing_bytes,
            "base_tokens_per_word": round(base_tpw, 4),
            "new_tokens_per_word": round(new_tpw, 4),
            "training_seconds": round(train_elapsed, 1),
            "hashes": tok_hashes,
        },
        "compliance": {
            "tokenizer_pool_words": pool_words,
            "within_10M_budget": pool_words <= 10_000_000,
            "no_external_text": True,
            "shared_across_arms": True,
        },
        "elapsed_sec": round(time.time() - t0, 1),
    }
    
    manifest_path = OUT_DIR / "shared_tokenizer_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    
    # Save note
    NOTE_PATH.parent.mkdir(parents=True, exist_ok=True)
    NOTE_PATH.write_text(
        "# research — Shared 16k BPE tokenizer for FW mechanism comparison\n\n"
        f"- Pool: {pool_words} words from {len(pool_rows)} rows (official+Qwen+FW source)\n"
        f"- Vocab: {actual_vocab} (requested {VOCAB_SIZE})\n"
        f"- Tokens/word: {tpw:.4f} (base legal16k: {base_tpw:.4f})\n"
        f"- UNK: {unk_count}, missing bytes: {missing_bytes}\n"
        f"- Training: {train_elapsed:.1f}s\n"
        f"- Compliant: pool within 10M budget, no external text\n\n"
        f"Tokenizer dir: `{TOK_DIR}`\n"
        f"Manifest: `{manifest_path}`\n"
    )
    
    print(json.dumps({
        "status": manifest["status"],
        "vocab_size": actual_vocab,
        "pool_words": pool_words,
        "tokens_per_word": round(tpw, 4),
        "base_tpw": round(base_tpw, 4),
        "unk_count": unk_count,
        "missing_bytes": missing_bytes,
        "training_sec": round(train_elapsed, 1),
        "tokenizer_dir": str(TOK_DIR),
        "manifest": str(manifest_path),
    }, indent=2), flush=True)

if __name__ == "__main__":
    main()
