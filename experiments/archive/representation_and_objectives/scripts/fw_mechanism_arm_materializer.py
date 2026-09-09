#!/usr/bin/env python3
"""research: FW1.5M mechanism-family arm materialization.

CPU-only. No training, no evaluation. Builds three matched 10M corpora
from COMPACT_EXPERIENCE base pool + frozen FineWeb sources + compact rewrites.

Arms (differ only in FineWeb companion text):
  compact_view:    source + compact rewrite
  source_repeat:   source + repeated source prefix
  (source_diversity: deferred until compact/repeat comparison)

All arms share:
  - Same official BabyLM rows
  - Same retained Qwen paraphrase rows
  - Same FineWeb source sentences
  - One shared compliant tokenizer from ~9.38M common words
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import collections
import hashlib
import json
import math
import pathlib
import random
import re
import statistics
import time
from typing import Any

_SCRIPT_DIR = _public_path('experiments/archive/representation_and_objectives/scripts')
_WORKSPACE  = _public_path('experiments/archive/representation_and_objectives')

# ── Inputs ─────────────────────────────────────────────────────────
BASE_POOL = pathlib.Path("experiments/archive/compact_experience/data/qwen_clean_aligned/training_corpora/qwen_aligned_10M.jsonl")
FROZEN_SOURCES = _public_path('experiments/archive/representation_and_objectives/data/fw_mechanism_source_selection/fw_mechanism_frozen_sources.jsonl')
NEW_REWRITES   = _public_path('experiments/archive/representation_and_objectives/data/fw_compact_rewrite_generation/compact_rewrites_accepted.jsonl')

# ── Outputs ────────────────────────────────────────────────────────
OUT_DIR   = _public_path('experiments/archive/representation_and_objectives/data/fw_mechanism_arms')
NOTE_PATH = _public_path('research/notes/representation_and_objectives/fw_mechanism_arms.md')

# ── Constants ──────────────────────────────────────────────────────
TOTAL_WORDS = 10_000_000
PASSES = 10
NEUTRAL_TOPUP_WORDS = 9
QWEN_SOURCE = "qwen_pair_packed"
MAX_ROW_WORDS = 160
RNG_SEED = 98291301


def read_jsonl(path: pathlib.Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def wc(text: str) -> int:
    return len((text or "").split())


def norm_hash(text: str) -> str:
    t = " ".join(text.lower().split())
    return hashlib.sha256(t.encode("utf-8")).hexdigest()[:32]


def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ── Pool construction helpers ──────────────────────────────────────

def pack_pairs_into_rows(pairs: list[dict], use_rewrite: bool,
                         source_label: str, base_eid: int) -> list[dict]:
    """Pack source+companion pairs into ≤160-word rows.
    
    For compact_view arm: text = source + \\n + rewrite
    For source_repeat arm: text = source + \\n + source_prefix (matched words)
    """
    rows = []
    current_words = []
    current_pair_ids = []
    current_word_count = 0
    eid = base_eid

    for p in pairs:
        src = p["text"]
        sw = p["words"]
        if use_rewrite:
            companion = p.get("rewrite_text", "")
            cw = wc(companion)
        else:
            # Repeat: fill with source words to match rewrite word count
            rw_count = p.get("rewrite_words", wc(p.get("rewrite_text", "")))
            src_words = src.split()
            companion = " ".join(src_words[:rw_count]) if rw_count > 0 else ""
            cw = wc(companion)

        pair_text = src + " " + companion if companion else src
        pair_wc = sw + cw

        if current_word_count + pair_wc > MAX_ROW_WORDS and current_words:
            # Flush current row
            rows.append({
                "text": " ".join(current_words),
                "words": current_word_count,
                "example_id": eid,
                "source": source_label,
                "pair_ids": current_pair_ids,
            })
            eid += 1
            current_words = []
            current_pair_ids = []
            current_word_count = 0

        current_words.extend(pair_text.split())
        current_pair_ids.append(p.get("norm_hash", ""))
        current_word_count += pair_wc

    if current_words:
        rows.append({
            "text": " ".join(current_words),
            "words": current_word_count,
            "example_id": eid,
            "source": source_label,
            "pair_ids": current_pair_ids,
        })

    return rows


def select_qwen_rows_to_retain(qwen_rows: list[dict],
                                target_words: int,
                                seed: int) -> list[dict]:
    """Select Qwen rows to retain up to target word count.
    Shuffles then greedily selects until budget is reached."""
    rng = random.Random(seed)
    shuffled = list(qwen_rows)
    rng.shuffle(shuffled)
    retained = []
    total = 0
    for r in shuffled:
        w = r.get("words", 0)
        if total + w <= target_words:
            retained.append(r)
            total += w
    return retained


def build_neutral_topup(official_rows: list[dict], words_needed: int,
                        seed: int, base_eid: int) -> list[dict]:
    """Build neutral topup from official pool words."""
    if words_needed <= 0:
        return []
    # Collect enough official text
    rng = random.Random(seed)
    shuffled = list(official_rows)
    rng.shuffle(shuffled)
    all_words = []
    for r in shuffled:
        all_words.extend(r.get("text", "").split())
        if len(all_words) >= words_needed * 2:
            break
    text = " ".join(all_words[:words_needed])
    return [{
        "text": text,
        "words": wc(text),
        "example_id": base_eid,
        "source": "neutral_topup",
    }]


def build_shared_tokenizer_pool(official_rows: list[dict],
                                 retained_qwen: list[dict],
                                 source_sentences: list[str],
                                 neutral: list[dict]) -> str:
    """Build the shared tokenizer training text from common words."""
    parts = []
    for r in official_rows:
        parts.append(r.get("text", ""))
    for r in retained_qwen:
        parts.append(r.get("text", ""))
    for s in source_sentences:
        parts.append(s)
    for r in neutral:
        parts.append(r.get("text", ""))
    return "\n".join(parts)


def main() -> None:
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load COMPACT_EXPERIENCE base pool ──────────────────────────────────────
    print("Loading COMPACT_EXPERIENCE base pool...", flush=True)
    base_rows = read_jsonl(BASE_POOL)
    official_rows = [r for r in base_rows if r.get("source") != QWEN_SOURCE]
    qwen_rows = [r for r in base_rows if r.get("source") == QWEN_SOURCE]
    official_words = sum(r.get("words", 0) for r in official_rows)
    qwen_words = sum(r.get("words", 0) for r in qwen_rows)
    print(f"  Official: {len(official_rows)} rows, {official_words:,} words")
    print(f"  Qwen: {len(qwen_rows)} rows, {qwen_words:,} words")

    # ── Load frozen FineWeb sources ───────────────────────────────
    print("Loading frozen FineWeb sources...", flush=True)
    sources = read_jsonl(FROZEN_SOURCES)
    print(f"  Sources: {len(sources)}, {sum(s['words'] for s in sources):,} words")

    # Load compact rewrites (existing + new if available).
    n_with_rw = sum(1 for s in sources if s.get("has_rewrite"))
    n_without_rw = sum(1 for s in sources if not s.get("has_rewrite"))
    print(f"  With existing rewrite: {n_with_rw}")
    print(f"  Without rewrite: {n_without_rw}")

    # Try to load new rewrites
    new_rw_map = {}
    if NEW_REWRITES.exists():
        new_rows = read_jsonl(NEW_REWRITES)
        for r in new_rows:
            nh = r.get("norm_hash", "")
            if nh and r.get("accepted"):
                new_rw_map[nh] = r
        print(f"  New accepted rewrites loaded: {len(new_rw_map)}")
    else:
        print(f"  New rewrites not yet available at {NEW_REWRITES}")

    # Merge rewrites into frozen sources
    pairs_ready = []
    pairs_missing = []
    for s in sources:
        nh = s.get("norm_hash", "")
        if s.get("has_rewrite") and s.get("rewrite_text"):
            pairs_ready.append(s)
        elif nh in new_rw_map:
            merged = dict(s)
            merged["rewrite_text"] = new_rw_map[nh]["rewrite_text"]
            merged["rewrite_words"] = new_rw_map[nh]["rewrite_words"]
            merged["content_recall"] = new_rw_map[nh].get("content_recall")
            merged["entity_recall"] = new_rw_map[nh].get("entity_recall")
            merged["number_recall"] = new_rw_map[nh].get("number_recall")
            merged["has_rewrite"] = True
            pairs_ready.append(merged)
        else:
            pairs_missing.append(s)

    print(f"  Pairs ready: {len(pairs_ready)}")
    print(f"  Pairs missing rewrite: {len(pairs_missing)}")

    if len(pairs_missing) > 0:
        print(f"\n  WARNING: {len(pairs_missing)} sources still lack rewrites.")
        print(f"  Building partial arms with {len(pairs_ready)} available pairs.")
        print(f"  Rerun after generation task completes for full arms.")

    # ── Compute exact budgets ─────────────────────────────────────
    pair_source_words = sum(p["words"] for p in pairs_ready)
    pair_rewrite_words = sum(p.get("rewrite_words", 0) for p in pairs_ready)
    pair_total_words = pair_source_words + pair_rewrite_words
    changed_block_words = pair_total_words + NEUTRAL_TOPUP_WORDS
    filler_words = TOTAL_WORDS - changed_block_words
    retained_qwen_target = filler_words - official_words

    print(f"\n  Budget:")
    print(f"    Official (all kept): {official_words:,}")
    print(f"    FineWeb pairs: {pair_total_words:,} ({pair_source_words:,} src + {pair_rewrite_words:,} rw)")
    print(f"    Neutral topup: {NEUTRAL_TOPUP_WORDS}")
    print(f"    Changed block: {changed_block_words:,}")
    print(f"    Filler target: {filler_words:,}")
    print(f"    Retained Qwen target: {retained_qwen_target:,}")

    if retained_qwen_target < 0:
        print(f"\n  ERROR: FineWeb block too large, would need to displace official rows.")
        print(f"  Trim {-retained_qwen_target:,} words of FineWeb pairs or use smaller scale.")
        # Trim pairs to fit
        while pair_total_words + NEUTRAL_TOPUP_WORDS + official_words > TOTAL_WORDS and pairs_ready:
            removed = pairs_ready.pop()
            pair_source_words -= removed["words"]
            pair_rewrite_words -= removed.get("rewrite_words", 0)
            pair_total_words = pair_source_words + pair_rewrite_words
        changed_block_words = pair_total_words + NEUTRAL_TOPUP_WORDS
        filler_words = TOTAL_WORDS - changed_block_words
        retained_qwen_target = filler_words - official_words
        print(f"  After trimming: {len(pairs_ready)} pairs, retained Qwen {retained_qwen_target:,}")

    # ── Select retained Qwen rows ─────────────────────────────────
    retained_qwen = select_qwen_rows_to_retain(
        qwen_rows, retained_qwen_target, RNG_SEED)
    actual_retained_qwen_words = sum(r["words"] for r in retained_qwen)

    # Adjust for rounding: the remaining words go to a tiny neutral topup
    actual_filler = official_words + actual_retained_qwen_words
    adjusted_neutral = TOTAL_WORDS - actual_filler - pair_total_words
    if adjusted_neutral < 0:
        print(f"  ERROR: negative neutral topup {adjusted_neutral}")
        return

    print(f"  Retained Qwen: {len(retained_qwen)} rows, {actual_retained_qwen_words:,} words")
    print(f"  Adjusted neutral topup: {adjusted_neutral}")

    # ── Build filler (shared across all arms) ─────────────────────
    filler = list(official_rows) + list(retained_qwen)
    filler_word_total = sum(r["words"] for r in filler)
    print(f"\n  Filler: {len(filler)} rows, {filler_word_total:,} words")

    # ── Pack FineWeb pairs into rows ──────────────────────────────
    print("Packing FineWeb pairs...", flush=True)
    compact_rows = pack_pairs_into_rows(pairs_ready, use_rewrite=True,
                                         source_label="fw_compact_view",
                                         base_eid=950000)
    repeat_rows = pack_pairs_into_rows(pairs_ready, use_rewrite=False,
                                        source_label="fw_source_repeat",
                                        base_eid=950000)

    # Build neutral topup
    neutral_rows = build_neutral_topup(official_rows, adjusted_neutral,
                                        RNG_SEED + 7, 990000)

    # ── Build complete arms ───────────────────────────────────────
    arms = {}
    for arm_name, fw_rows in [("compact_view", compact_rows),
                               ("source_repeat", repeat_rows)]:
        pool = list(fw_rows) + list(neutral_rows) + list(filler)
        total_w = sum(r["words"] for r in pool)
        arms[arm_name] = {
            "rows": pool,
            "total_words": total_w,
            "n_rows": len(pool),
            "fw_rows": len(fw_rows),
            "fw_words": sum(r["words"] for r in fw_rows),
        }
        print(f"  Arm {arm_name}: {len(pool)} rows, {total_w:,} words")

    # ── Verify word totals ────────────────────────────────────────
    for arm_name, arm in arms.items():
        if arm["total_words"] != TOTAL_WORDS:
            print(f"  WARNING: {arm_name} total {arm['total_words']} != {TOTAL_WORDS}")

    # ── Build shared tokenizer pool ───────────────────────────────
    source_texts = [p["text"] for p in pairs_ready]
    tok_pool_text = build_shared_tokenizer_pool(
        official_rows, retained_qwen, source_texts, neutral_rows)
    tok_pool_words = wc(tok_pool_text)
    print(f"\n  Shared tokenizer pool: {tok_pool_words:,} words")

    # ── Save outputs ──────────────────────────────────────────────
    # Save 10M pools
    for arm_name, arm in arms.items():
        pool_path = OUT_DIR / f"fw_mechanism_{arm_name}_10M.jsonl"
        with pool_path.open("w", encoding="utf-8") as f:
            for r in arm["rows"]:
                rec = {"text": r["text"], "words": r["words"],
                       "example_id": r.get("example_id", 0),
                       "source": r.get("source", "")}
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"  Saved {pool_path.name}: {arm['n_rows']} rows")

    # Save shared tokenizer pool
    tok_path = _public_path('experiments/archive/representation_and_objectives/data/fw_mechanism_arms/shared_tokenizer_pool.txt')
    tok_path.write_text(tok_pool_text, encoding="utf-8")
    print(f"  Saved shared tokenizer pool: {tok_pool_words:,} words")

    # ── Save manifest ─────────────────────────────────────────────
    manifest = {
        "status": "FW_MECHANISM_ARMS",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "complete": len(pairs_missing) == 0,
        "pairs_ready": len(pairs_ready),
        "pairs_missing": len(pairs_missing),
        "base_pool": str(BASE_POOL),
        "base_pool_sha256": sha256_file(BASE_POOL),
        "budgets": {
            "official_words": official_words,
            "retained_qwen_words": actual_retained_qwen_words,
            "pair_source_words": pair_source_words,
            "pair_rewrite_words": pair_rewrite_words,
            "pair_total_words": pair_total_words,
            "neutral_topup_words": adjusted_neutral,
            "total": TOTAL_WORDS,
        },
        "arms": {name: {k: v for k, v in info.items() if k != "rows"}
                 for name, info in arms.items()},
        "shared_tokenizer_pool_words": tok_pool_words,
        "retained_qwen_rows": len(retained_qwen),
        "displaced_qwen_words": qwen_words - actual_retained_qwen_words,
        "elapsed_sec": round(time.time() - t0, 1),
    }

    manifest_path = _public_path('experiments/archive/representation_and_objectives/data/fw_mechanism_arms/fw_mechanism_arms_manifest.json')
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8")

    # ── Save note ─────────────────────────────────────────────────
    _public_path('research/notes/representation_and_objectives').mkdir(parents=True, exist_ok=True)
    note = [
        "# research — FW mechanism arm materialization\n\n",
        f"Complete: {'YES' if len(pairs_missing) == 0 else 'PARTIAL'}\n",
        f"Pairs ready: {len(pairs_ready)}, missing: {len(pairs_missing)}\n\n",
        "## Budget\n",
        f"- Official BabyLM: {official_words:,} words (all kept)\n",
        f"- Retained Qwen: {actual_retained_qwen_words:,} words "
        f"({len(retained_qwen)} rows)\n",
        f"- FineWeb pairs: {pair_total_words:,} words\n",
        f"- Neutral: {adjusted_neutral} words\n",
        f"- Shared tokenizer pool: {tok_pool_words:,} words\n\n",
        "## Arms\n",
    ]
    for arm_name, arm in arms.items():
        note.append(f"- {arm_name}: {arm['n_rows']} rows, "
                     f"{arm['total_words']:,} words\n")
    note.append(f"\n## Artifacts\n")
    note.append(f"- Manifest: `{manifest_path}`\n")
    for arm_name in arms:
        note.append(f"- {arm_name}: `{OUT_DIR}/fw_mechanism_{arm_name}_10M.jsonl`\n")
    note.append(f"- Tokenizer pool: `{OUT_DIR}/shared_tokenizer_pool.txt`\n")
    NOTE_PATH.write_text("".join(note), encoding="utf-8")

    print(json.dumps({
        "status": manifest["status"],
        "complete": manifest["complete"],
        "pairs_ready": len(pairs_ready),
        "pairs_missing": len(pairs_missing),
        "official_words": official_words,
        "retained_qwen_words": actual_retained_qwen_words,
        "pair_total_words": pair_total_words,
        "shared_tok_words": tok_pool_words,
        "elapsed_sec": manifest["elapsed_sec"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
