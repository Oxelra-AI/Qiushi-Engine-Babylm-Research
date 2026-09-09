#!/usr/bin/env python3
"""research: Neutral tokenizer and order-balanced causal pools.

Implements two corrections for orthogonal architecture-transfer test:
1. Tokenizer neutrality: BPE 16k trained ONLY on common filler text (9.58M words),
   identical between compact-view and repeat arms → neither arm gets tokenizer advantage.
2. Order balance: 50% source→view, 50% view→source per pair (fixed seed 171043),
   applied identically to both arms → causal model cannot reduce to continuation/copying.

Produces in OUT_DIR:
  neutral_tokenizer/          HF-compatible BPE tokenizer
  causal_compact_10M.jsonl    order-balanced compact-view pool (10M words)
  causal_repeat_10M.jsonl     order-balanced repeat pool (10M words)
  filler_rows.jsonl           extracted filler (identical across arms)
  manifest.json               provenance, hashes, verification
"""
import json, hashlib, pathlib, random, sys, time, os

os.environ["TOKENIZERS_PARALLELISM"] = "true"

ROOT = pathlib.Path("experiments/archive/frontier_consolidation")
BASE = ROOT / "data/density_cleanqwen_overlay_medium_riskhard"
PAIRS_FILE = ROOT / "data/density_core_reinvestment_medium_riskhard/selected_compact_reinvest_pairs.jsonl"
CV_POOL = BASE / "cleanqwen_fineweb_compact_view_reinvest_10M.jsonl"
OUT = ROOT / "data/causal_transfer_scaffold"

SEED = 171043
VOCAB_SIZE = 16384
TARGET_TOTAL_WORDS = 10_000_000

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def wc(text):
    return len(text.split())

def main():
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)

    # ── research: Separate filler from pair rows ──────────────────────────
    print("research: Separating filler and pair rows...", flush=True)
    filler_rows = []
    pair_row_count = 0
    filler_words = 0
    pair_words_from_pool = 0

    with open(CV_POOL) as f:
        for line in f:
            r = json.loads(line)
            src = r.get("source", "")
            if "compact_view" in str(src):
                pair_row_count += 1
                pair_words_from_pool += r.get("words", 0)
            else:
                filler_rows.append(r)
                filler_words += r.get("words", 0)

    print(f"  Filler: {len(filler_rows)} rows, {filler_words:,} words")
    print(f"  Pair rows: {pair_row_count}, {pair_words_from_pool:,} words")

    # Save filler rows (for reference and tokenizer)
    filler_jsonl = OUT / "filler_rows.jsonl"
    filler_txt = OUT / "filler_text_for_tokenizer.txt"
    with open(filler_jsonl, "w") as fj, open(filler_txt, "w") as ft:
        for r in filler_rows:
            fj.write(json.dumps(r, ensure_ascii=False) + "\n")
            ft.write(r["text"] + "\n")
    print(f"  Saved filler: {filler_jsonl.name}, {filler_txt.name}")

    # ── research: Train neutral tokenizer on filler only ──────────────────
    print("\nStep 2: Training neutral BPE tokenizer on filler...", flush=True)
    from tokenizers import Tokenizer
    from tokenizers.models import BPE
    from tokenizers.trainers import BpeTrainer
    from tokenizers.pre_tokenizers import ByteLevel
    from tokenizers.decoders import ByteLevel as ByteLevelDecoder

    raw_tok = Tokenizer(BPE(unk_token="<unk>"))
    raw_tok.pre_tokenizer = ByteLevel(add_prefix_space=False)
    raw_tok.decoder = ByteLevelDecoder()

    trainer = BpeTrainer(
        vocab_size=VOCAB_SIZE,
        special_tokens=["<unk>", "<s>", "</s>", "<pad>", "<mask>"],
        min_frequency=2,
    )
    raw_tok.train([str(filler_txt)], trainer)
    actual_vocab = raw_tok.get_vocab_size()
    print(f"  Vocab size: {actual_vocab}")

    # Save HF-compatible tokenizer
    tok_dir = OUT / "neutral_tokenizer"
    tok_dir.mkdir(exist_ok=True)
    raw_tok.save(str(tok_dir / "tokenizer.json"))

    from transformers import PreTrainedTokenizerFast
    hf_tok = PreTrainedTokenizerFast(
        tokenizer_object=raw_tok,
        bos_token="<s>",
        eos_token="</s>",
        unk_token="<unk>",
        pad_token="<pad>",
        mask_token="<mask>",
        model_max_length=256,
    )
    hf_tok.save_pretrained(str(tok_dir))
    tok_json_sha = sha256_file(tok_dir / "tokenizer.json")
    print(f"  Tokenizer saved to {tok_dir}")
    print(f"  tokenizer.json SHA256: {tok_json_sha}")

    # Quick tokenizer smoke
    test_text = "The cat sat on the mat. Palm leaves have a spiritual significance."
    ids = hf_tok.encode(test_text)
    decoded = hf_tok.decode(ids)
    print(f"  Smoke: '{test_text[:50]}...' → {len(ids)} tokens → '{decoded[:50]}...'")

    # ── research: Read individual pairs ───────────────────────────────────
    print("\nStep 3: Reading individual pairs...", flush=True)
    pairs = []
    with open(PAIRS_FILE) as f:
        for line in f:
            pairs.append(json.loads(line))

    total_pair_words = sum(p["pair_words"] for p in pairs)
    print(f"  {len(pairs)} pairs, {total_pair_words:,} pair words")
    assert total_pair_words + filler_words == TARGET_TOTAL_WORDS, \
        f"Word budget mismatch: {total_pair_words} + {filler_words} = {total_pair_words + filler_words} != {TARGET_TOTAL_WORDS}"

    # ── research: Assign order with fixed seed ────────────────────────────
    print("\nStep 4: Assigning pair order (50/50 balance)...", flush=True)
    rng = random.Random(SEED)
    orders = ["source_first" if rng.random() < 0.5 else "view_first" for _ in pairs]
    n_sf = sum(1 for o in orders if o == "source_first")
    n_vf = len(orders) - n_sf
    print(f"  source_first: {n_sf}, view_first: {n_vf}")

    # ── research: Build pair items for both arms ──────────────────────────
    print("\nStep 5: Building pair items...", flush=True)
    compact_items = []
    repeat_items = []

    for idx, (p, order) in enumerate(zip(pairs, orders)):
        src_text = p["source_text"]
        view_text = p["rewrite_text"]
        target_rw = p["rewrite_words"]

        # Construct repeat: first target_rw words of source
        src_words_list = src_text.split()
        if target_rw <= len(src_words_list):
            repeat_text = " ".join(src_words_list[:target_rw])
        else:
            # Edge case: repeat entire source (shouldn't happen for compact regime)
            repeat_text = " ".join((src_words_list * ((target_rw // len(src_words_list)) + 2))[:target_rw])

        # Verify word counts
        assert wc(view_text) == target_rw, f"Pair {idx}: view wc {wc(view_text)} != {target_rw}"
        assert wc(repeat_text) == target_rw, f"Pair {idx}: repeat wc {wc(repeat_text)} != {target_rw}"

        # Apply order
        if order == "source_first":
            cv_text = src_text + " " + view_text
            rp_text = src_text + " " + repeat_text
        else:
            cv_text = view_text + " " + src_text
            rp_text = repeat_text + " " + src_text

        cv_wc = wc(cv_text)
        rp_wc = wc(rp_text)
        assert cv_wc == p["pair_words"], f"Pair {idx}: cv wc {cv_wc} != {p['pair_words']}"
        assert rp_wc == p["pair_words"], f"Pair {idx}: rp wc {rp_wc} != {p['pair_words']}"

        compact_items.append({"text": cv_text, "words": cv_wc, "pair_idx": idx, "order": order})
        repeat_items.append({"text": rp_text, "words": rp_wc, "pair_idx": idx, "order": order})

    cv_pair_words = sum(it["words"] for it in compact_items)
    rp_pair_words = sum(it["words"] for it in repeat_items)
    print(f"  Compact pair words: {cv_pair_words:,}")
    print(f"  Repeat pair words:  {rp_pair_words:,}")

    # ── research: Build and save JSONL pools ──────────────────────────────
    # Interleave: filler rows in original order, pair items inserted at evenly
    # spaced positions to approximate original stream structure.
    print("\nStep 6: Building JSONL pools...", flush=True)

    n_filler = len(filler_rows)
    n_pairs = len(compact_items)

    # Compute insertion points: spread pair items evenly across filler
    # Every ~(n_filler / n_pairs) filler rows, insert one pair item
    pair_interval = n_filler / n_pairs if n_pairs > 0 else float("inf")

    def build_pool(filler, items, source_label):
        rows = []
        item_idx = 0
        next_insert = pair_interval / 2  # Start inserting at first half-interval
        for fi, fr in enumerate(filler):
            rows.append({"text": fr["text"], "words": fr.get("words", wc(fr["text"])),
                          "source": fr.get("source", "filler")})
            # Insert pair items at evenly spaced positions
            while item_idx < len(items) and fi >= next_insert:
                it = items[item_idx]
                rows.append({"text": it["text"], "words": it["words"],
                              "source": source_label})
                item_idx += 1
                next_insert += pair_interval
        # Append any remaining pair items
        while item_idx < len(items):
            it = items[item_idx]
            rows.append({"text": it["text"], "words": it["words"],
                          "source": source_label})
            item_idx += 1
        return rows

    cv_pool = build_pool(filler_rows, compact_items, "causal_compact_view")
    rp_pool = build_pool(filler_rows, repeat_items, "causal_repeat")

    cv_path = OUT / "causal_compact_10M.jsonl"
    rp_path = OUT / "causal_repeat_10M.jsonl"

    cv_total_w = 0
    with open(cv_path, "w") as f:
        for r in cv_pool:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            cv_total_w += r["words"]

    rp_total_w = 0
    with open(rp_path, "w") as f:
        for r in rp_pool:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            rp_total_w += r["words"]

    print(f"  Compact pool: {len(cv_pool)} rows, {cv_total_w:,} words → {cv_path.name}")
    print(f"  Repeat pool:  {len(rp_pool)} rows, {rp_total_w:,} words → {rp_path.name}")
    assert cv_total_w == TARGET_TOTAL_WORDS, f"CV words {cv_total_w} != {TARGET_TOTAL_WORDS}"
    assert rp_total_w == TARGET_TOTAL_WORDS, f"RP words {rp_total_w} != {TARGET_TOTAL_WORDS}"

    # ── research: Tokenization smoke test ─────────────────────────────────
    print("\nStep 7: Tokenization smoke...", flush=True)
    cv_tokens_sample = 0
    rp_tokens_sample = 0
    n_sample = min(1000, len(cv_pool))
    for i in range(n_sample):
        cv_tokens_sample += len(hf_tok.encode(cv_pool[i]["text"]))
        rp_tokens_sample += len(hf_tok.encode(rp_pool[i]["text"]))

    cv_tok_ratio = cv_tokens_sample / sum(cv_pool[i]["words"] for i in range(n_sample))
    rp_tok_ratio = rp_tokens_sample / sum(rp_pool[i]["words"] for i in range(n_sample))
    print(f"  Compact tok/word ratio (first {n_sample}): {cv_tok_ratio:.3f}")
    print(f"  Repeat tok/word ratio  (first {n_sample}): {rp_tok_ratio:.3f}")
    est_total_tokens = int(TARGET_TOTAL_WORDS * (cv_tok_ratio + rp_tok_ratio) / 2)
    est_seqs = est_total_tokens // 256
    print(f"  Est total tokens per pool: ~{est_total_tokens:,}")
    print(f"  Est 256-token sequences per epoch: ~{est_seqs:,}")

    # ── research: Manifest ────────────────────────────────────────────────
    elapsed = time.time() - t0
    manifest = {
        "status": "CAUSAL_TRANSFER_SCAFFOLD_BUILT",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_sec": round(elapsed, 1),
        "strategist_corrections": {
            "tokenizer_neutrality": "BPE trained on common filler only (identical between arms)",
            "order_balance": f"50/50 source-first/view-first (seed {SEED}): {n_sf}/{n_vf}",
        },
        "tokenizer": {
            "vocab_size": actual_vocab,
            "training_data": "filler_text_for_tokenizer.txt",
            "training_words": filler_words,
            "dir": str(tok_dir.relative_to(ROOT)),
            "tokenizer_json_sha256": tok_json_sha,
        },
        "pools": {
            "compact_view": {
                "path": str(cv_path.relative_to(ROOT)),
                "rows": len(cv_pool),
                "words": cv_total_w,
                "sha256": sha256_file(cv_path),
                "pair_items": len(compact_items),
                "est_tok_word_ratio": round(cv_tok_ratio, 4),
            },
            "repeat": {
                "path": str(rp_path.relative_to(ROOT)),
                "rows": len(rp_pool),
                "words": rp_total_w,
                "sha256": sha256_file(rp_path),
                "pair_items": len(repeat_items),
                "est_tok_word_ratio": round(rp_tok_ratio, 4),
            },
        },
        "filler": {
            "rows": len(filler_rows),
            "words": filler_words,
            "sha256": sha256_file(filler_jsonl),
        },
        "pairs": {
            "total": len(pairs),
            "total_pair_words": total_pair_words,
            "source_file": str(PAIRS_FILE.relative_to(ROOT)),
        },
        "source_pool_sha256": sha256_file(CV_POOL),
        "order_seed": SEED,
        "est_total_tokens": est_total_tokens,
        "est_seqs_per_epoch": est_seqs,
    }

    manifest_path = OUT / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print(f"Done in {elapsed:.1f}s.")
    print(json.dumps(manifest, indent=2))

if __name__ == "__main__":
    main()
