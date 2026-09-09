#!/usr/bin/env python3
"""research: Materialize HS/LS/HD/LD factorial streams for contextual-anchor ordinary-WWM experiment.

Constructs 4 × 10M JSONL pools and 4 × 100M streams from:
  - 12,155 compact candidate pairs (pair-level source/compact texts)
  - 3,006 changed-block row meta (maps packed rows → pair_ids)
  - Historical 10M compact-view pool (for filler rows 3006-64739)
  - Derangement mapping (length-preserving, zero self-pairs)

Arms:
  HS: source_i + compact_i               (high-diversity, same anchors)
  LS: source_i + repeat_i                (low-diversity, same anchors)
  HD: source_i + compact_deranged(i)     (high-diversity, different anchors)
  LD: source_i + repeat_deranged(i)      (low-diversity, different anchors)

All four share identical filler rows (9,576,480 words). Changed rows differ only
in the second-context slot of each packed pair. Repeat views use the historical
hash-rotated cyclic source repetition matching compact view_words.

No training is launched. This is a CPU/file construction only.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse
import collections
import hashlib
import json
import math
import os
import time
from pathlib import Path
from typing import Any

try:
    from transformers import AutoTokenizer
except Exception:
    AutoTokenizer = None


def find_user_root() -> Path:
    return _PUBLIC_ROOT


USER_ROOT = find_user_root()

# --- Source paths ---
PAIRS_JSONL = USER_ROOT / "experiments/archive/frontier_consolidation/data/factorial_view_candidate_audit/compact_candidate_pairs.jsonl"
ROW_META = USER_ROOT / "experiments/archive/frontier_consolidation/data/density_cleanqwen_overlay_medium_riskhard/cleanqwen_fineweb_compact_view_reinvest_changed_block_rows_meta.jsonl"
POOL_COMPACT = USER_ROOT / "experiments/archive/frontier_consolidation/data/density_cleanqwen_overlay_medium_riskhard/cleanqwen_fineweb_compact_view_reinvest_10M.jsonl"
POOL_REPEAT = USER_ROOT / "experiments/archive/frontier_consolidation/data/density_cleanqwen_overlay_medium_riskhard/cleanqwen_fineweb_repeat_compact_reinvest_10M.jsonl"
TOKENIZER_DIR = USER_ROOT / "experiments/archive/frontier_consolidation/data/compliant_tokenizer"
DEFAULT_OUT = USER_ROOT / "experiments/archive/representation_and_objectives/data/factorial_streams"

CHANGED_ROWS = 3006
FILLER_START = CHANGED_ROWS
EXPECTED_ROWS = 64740
EXPECTED_WORDS = 10_000_000
EPOCHS = 10
EXPECTED_100M_WORDS = EXPECTED_WORDS * EPOCHS


def now_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def wc(text: str) -> int:
    return len(text.split())


def repeat_source_words(source_text: str, n_words: int, salt: str) -> str:
    """Historical hash-rotated cyclic source repetition, matching research builder."""
    toks = source_text.split()
    if not toks or n_words <= 0:
        return ""
    h = int(hashlib.sha1(salt.encode("utf-8")).hexdigest()[:8], 16)
    start = h % len(toks)
    rot = toks[start:] + toks[:start]
    out: list[str] = []
    while len(out) < n_words:
        out.extend(rot[: n_words - len(out)])
    return " ".join(out)


def length_derangement(pairs: list[dict[str, Any]]) -> dict[int, int]:
    """Exact copy of research derangement: within view-word-count bins, rotate one step.
    Singleton bins are deranged among singletons sorted by view_words."""
    by_len: dict[int, list[int]] = collections.defaultdict(list)
    for i, p in enumerate(pairs):
        by_len[int(p["view_words"])].append(i)
    mapping: dict[int, int] = {}
    singletons: list[int] = []
    for L, idxs in sorted(by_len.items()):
        if len(idxs) == 1:
            singletons.extend(idxs)
            continue
        idxs = sorted(idxs, key=lambda i: hashlib.sha1(str(pairs[i]["pair_id"]).encode()).hexdigest())
        for a, b in zip(idxs, idxs[1:] + idxs[:1]):
            mapping[a] = b
    if singletons:
        if len(singletons) == 1:
            i = singletons[0]
            best = min((j for j in range(len(pairs)) if j != i),
                       key=lambda j: abs(int(pairs[j]["view_words"]) - int(pairs[i]["view_words"])))
            mapping[i] = best
        else:
            singletons = sorted(singletons, key=lambda i: (int(pairs[i]["view_words"]), str(pairs[i]["pair_id"])))
            for a, b in zip(singletons, singletons[1:] + singletons[:1]):
                mapping[a] = b
    return mapping


def load_pairs(path: Path) -> list[dict[str, Any]]:
    pairs: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            obj["source_text"] = str(obj["source_text"])
            obj["view_text"] = str(obj.get("view_text") or obj.get("rewrite_text") or obj.get("original_compact_text"))
            obj["source_words"] = int(obj.get("source_words", wc(obj["source_text"])))
            obj["view_words"] = int(obj.get("view_words", wc(obj["view_text"])))
            pairs.append(obj)
    return pairs


def load_row_meta(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rows.append(json.loads(line))
    return rows


def build_changed_row(
    row_meta: dict[str, Any],
    pair_map: dict[str, dict[str, Any]],
    derangement: dict[int, int],
    pair_index_map: dict[str, int],
    arm: str,
    all_pairs: list[dict[str, Any]],
) -> tuple[str, int]:
    """Build text for one changed row in the specified arm."""
    parts: list[str] = []
    for pid in row_meta["pair_ids"]:
        pair = pair_map[pid]
        src = pair["source_text"]
        pi = pair_index_map[pid]

        if arm == "HS":
            second = pair["view_text"]
        elif arm == "LS":
            second = repeat_source_words(src, pair["view_words"], pid)
        elif arm == "HD":
            donor_i = derangement[pi]
            donor = all_pairs[donor_i]
            second = donor["view_text"]
        elif arm == "LD":
            donor_i = derangement[pi]
            donor = all_pairs[donor_i]
            second = repeat_source_words(donor["source_text"], donor["view_words"], donor["pair_id"])
        else:
            raise ValueError(f"unknown arm {arm}")

        parts.append(src)
        parts.append(second)

    text = " ".join(parts)
    return text, wc(text)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default=str(DEFAULT_OUT))
    ap.add_argument("--dry_run", action="store_true", help="Only produce the construction report, not files")
    ap.add_argument("--skip_100m", action="store_true", help="Only produce 10M pools")
    ap.add_argument("--skip_bpe", action="store_true", help="Skip BPE audit")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    # --- Load pair data ---
    print(json.dumps({"event": "loading_pairs", "path": str(PAIRS_JSONL)}), flush=True)
    pairs = load_pairs(PAIRS_JSONL)
    pair_map: dict[str, dict[str, Any]] = {p["pair_id"]: p for p in pairs}
    pair_index_map: dict[str, int] = {p["pair_id"]: i for i, p in enumerate(pairs)}
    print(json.dumps({"event": "pairs_loaded", "n": len(pairs)}), flush=True)

    # --- Load row meta ---
    print(json.dumps({"event": "loading_meta", "path": str(ROW_META)}), flush=True)
    row_meta = load_row_meta(ROW_META)
    assert len(row_meta) == CHANGED_ROWS, f"Expected {CHANGED_ROWS} changed rows, got {len(row_meta)}"
    # Verify all pair_ids exist
    missing_pairs = []
    all_meta_pair_ids = []
    for rm in row_meta:
        for pid in rm["pair_ids"]:
            all_meta_pair_ids.append(pid)
            if pid not in pair_map:
                missing_pairs.append(pid)
    assert not missing_pairs, f"Missing pair_ids in pair_map: {missing_pairs[:5]}"
    print(json.dumps({"event": "meta_loaded", "rows": len(row_meta), "total_pairs_in_meta": len(all_meta_pair_ids), "unique_pairs": len(set(all_meta_pair_ids))}), flush=True)

    # --- Build derangement ---
    derangement = length_derangement(pairs)
    assert len(derangement) == len(pairs), f"Derangement incomplete: {len(derangement)} vs {len(pairs)}"
    self_pairs = sum(1 for i, j in derangement.items() if i == j)
    assert self_pairs == 0, f"Self-pairs in derangement: {self_pairs}"
    print(json.dumps({"event": "derangement_built", "n": len(derangement), "self_pairs": self_pairs}), flush=True)

    # --- Load filler rows ---
    print(json.dumps({"event": "loading_pool", "path": str(POOL_COMPACT)}), flush=True)
    filler_rows: list[str] = []  # raw JSON lines
    filler_words = 0
    with POOL_COMPACT.open("r", encoding="utf-8") as f:
        for row_i, line in enumerate(f):
            if row_i < CHANGED_ROWS:
                continue
            if not line.strip():
                continue
            filler_rows.append(line.rstrip("\n"))
            obj = json.loads(line)
            filler_words += int(obj["words"])
    assert len(filler_rows) == EXPECTED_ROWS - CHANGED_ROWS, f"Filler rows: {len(filler_rows)} vs expected {EXPECTED_ROWS - CHANGED_ROWS}"
    print(json.dumps({"event": "filler_loaded", "rows": len(filler_rows), "words": filler_words}), flush=True)

    # --- Build changed rows for each arm ---
    ARMS = ["HS", "LS", "HD", "LD"]
    arm_changed_words: dict[str, int] = {}
    arm_changed_texts: dict[str, list[tuple[str, int, int]]] = {a: [] for a in ARMS}  # (text, words, example_id)

    for arm in ARMS:
        total_w = 0
        for rm in row_meta:
            text, words = build_changed_row(rm, pair_map, derangement, pair_index_map, arm, pairs)
            arm_changed_texts[arm].append((text, words, rm["example_id"]))
            total_w += words
        arm_changed_words[arm] = total_w

    # --- Report per-arm statistics ---
    arm_total_words: dict[str, int] = {}
    arm_row_counts: dict[str, int] = {}
    for arm in ARMS:
        arm_total_words[arm] = arm_changed_words[arm] + filler_words
        arm_row_counts[arm] = CHANGED_ROWS + len(filler_rows)

    print(json.dumps({"event": "changed_rows_built", "arm_changed_words": arm_changed_words, "arm_total_words": arm_total_words}), flush=True)

    # --- Verify word count symmetry ---
    hs_ls_word_delta = arm_total_words["HS"] - arm_total_words["LS"]
    hd_ld_word_delta = arm_total_words["HD"] - arm_total_words["LD"]
    hs_hd_word_delta = arm_total_words["HS"] - arm_total_words["HD"]
    ls_ld_word_delta = arm_total_words["LS"] - arm_total_words["LD"]
    interaction_word_delta = (arm_total_words["HS"] - arm_total_words["LS"]) - (arm_total_words["HD"] - arm_total_words["LD"])

    if args.dry_run:
        print(json.dumps({
            "status": "FACTORIAL_DRY_RUN",
            "arm_total_words": arm_total_words,
            "word_deltas": {
                "HS_minus_LS": hs_ls_word_delta,
                "HD_minus_LD": hd_ld_word_delta,
                "HS_minus_HD": hs_hd_word_delta,
                "LS_minus_LD": ls_ld_word_delta,
                "interaction": interaction_word_delta,
            },
        }, indent=2), flush=True)
        return

    # --- Write 10M pools ---
    pool_paths: dict[str, Path] = {}
    pool_shas: dict[str, str] = {}
    for arm in ARMS:
        pool_path = out_dir / f"factorial_{arm.lower()}_10M.jsonl"
        pool_paths[arm] = pool_path
        with pool_path.open("w", encoding="utf-8") as f:
            # Write changed rows
            for text, words, example_id in arm_changed_texts[arm]:
                row_obj = {
                    "text": text,
                    "words": words,
                    "example_id": example_id,
                    "source": f"factorial_{arm.lower()}_compact_reinvest",
                }
                f.write(json.dumps(row_obj, ensure_ascii=False) + "\n")
            # Write filler rows (identical)
            for fline in filler_rows:
                f.write(fline + "\n")
        pool_shas[arm] = sha256_file(pool_path)
        print(json.dumps({"event": "pool_written", "arm": arm, "path": str(pool_path), "sha256": pool_shas[arm]}), flush=True)

    # --- Verify pool integrity ---
    pool_verification: dict[str, dict[str, Any]] = {}
    for arm in ARMS:
        total_w = 0
        n_rows = 0
        with pool_paths[arm].open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                obj = json.loads(line)
                total_w += int(obj["words"])
                n_rows += 1
        pool_verification[arm] = {"rows": n_rows, "words": total_w, "sha256": pool_shas[arm]}
    print(json.dumps({"event": "pool_verification", "pools": pool_verification}), flush=True)

    # --- Write 100M streams (10 epochs) ---
    stream_paths: dict[str, Path] = {}
    stream_shas: dict[str, str] = {}
    if not args.skip_100m:
        for arm in ARMS:
            stream_path = out_dir / f"factorial_{arm.lower()}_100M.jsonl"
            stream_paths[arm] = stream_path
            with stream_path.open("w", encoding="utf-8") as fout:
                for epoch in range(EPOCHS):
                    with pool_paths[arm].open("r", encoding="utf-8") as fin:
                        for line in fin:
                            fout.write(line)
            stream_shas[arm] = sha256_file(stream_path)
            # Verify 100M stream
            total_w = 0
            n_rows = 0
            with stream_path.open("r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    obj = json.loads(line)
                    total_w += int(obj["words"])
                    n_rows += 1
            stream_paths[arm] = stream_path
            print(json.dumps({"event": "stream_written", "arm": arm, "rows": n_rows, "words": total_w, "sha256": stream_shas[arm]}), flush=True)

    # --- BPE audit (optional) ---
    bpe_audit: dict[str, Any] = {"skipped": True}
    if not args.skip_bpe and AutoTokenizer is not None and TOKENIZER_DIR.exists():
        tok = AutoTokenizer.from_pretrained(str(TOKENIZER_DIR), use_fast=True)
        bpe_audit = {"skipped": False, "tokenizer_vocab_size": len(tok)}
        # Sample first 500 changed rows per arm
        sample_n = min(500, CHANGED_ROWS)
        arm_bpe_totals: dict[str, int] = {}
        arm_bpe_changed: dict[str, int] = {}
        for arm in ARMS:
            total_bpe = 0
            changed_bpe = 0
            with pool_paths[arm].open("r", encoding="utf-8") as f:
                for ri, line in enumerate(f):
                    if ri >= sample_n:
                        break
                    obj = json.loads(line)
                    enc = tok(obj["text"], add_special_tokens=False, truncation=False)["input_ids"]
                    changed_bpe += len(enc)
            arm_bpe_changed[arm] = changed_bpe
        bpe_audit["sample_changed_rows"] = sample_n
        bpe_audit["changed_bpe_totals"] = arm_bpe_changed
        bpe_audit["HS_minus_LS_bpe"] = arm_bpe_changed["HS"] - arm_bpe_changed["LS"]
        bpe_audit["HD_minus_LD_bpe"] = arm_bpe_changed["HD"] - arm_bpe_changed["LD"]
        bpe_audit["HS_minus_HD_bpe"] = arm_bpe_changed["HS"] - arm_bpe_changed["HD"]
        bpe_audit["LS_minus_LD_bpe"] = arm_bpe_changed["LS"] - arm_bpe_changed["LD"]
        bpe_audit["interaction_bpe"] = (arm_bpe_changed["HS"] - arm_bpe_changed["LS"]) - (arm_bpe_changed["HD"] - arm_bpe_changed["LD"])
        print(json.dumps({"event": "bpe_audit", **bpe_audit}), flush=True)

    # --- Construction report ---
    elapsed = time.time() - t0
    report = {
        "status": "FACTORIAL_STREAMS_MATERIALIZED",
        "created_utc": now_utc(),
        "meaning": "Four-arm HS/LS/HD/LD factorial 10M pools and 100M streams for contextual-anchor ordinary-WWM experiment. No training launched.",
        "inputs": {
            "pairs_jsonl": str(PAIRS_JSONL),
            "pairs_jsonl_sha256": sha256_file(PAIRS_JSONL),
            "row_meta": str(ROW_META),
            "pool_compact": str(POOL_COMPACT),
            "pool_compact_sha256": sha256_file(POOL_COMPACT),
            "pool_repeat": str(POOL_REPEAT),
            "tokenizer": str(TOKENIZER_DIR),
            "n_pairs": len(pairs),
            "changed_rows": CHANGED_ROWS,
            "filler_rows": len(filler_rows),
            "filler_words": filler_words,
        },
        "derangement": {
            "n_mapped": len(derangement),
            "self_pairs": self_pairs,
        },
        "pool_10M": pool_verification,
        "pool_10M_word_deltas": {
            "HS_minus_LS": hs_ls_word_delta,
            "HD_minus_LD": hd_ld_word_delta,
            "HS_minus_HD": hs_hd_word_delta,
            "LS_minus_LD": ls_ld_word_delta,
            "interaction": interaction_word_delta,
        },
        "stream_100M": {arm: {"sha256": stream_shas.get(arm), "path": str(stream_paths.get(arm, ""))} for arm in ARMS} if not args.skip_100m else "skipped",
        "bpe_audit": bpe_audit,
        "elapsed_sec": round(elapsed, 1),
    }
    (out_dir / "factorial_stream_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    # Write markdown summary
    md_lines = [
        "# research factorial stream materialization",
        f"\nJSON: `{out_dir / 'factorial_stream_report.json'}`\n",
        "## 10M pool word counts",
    ]
    for arm in ARMS:
        pv = pool_verification[arm]
        md_lines.append(f"- {arm}: {pv['words']} words, {pv['rows']} rows, SHA {pv['sha256'][:16]}...")
    md_lines.append(f"\n## Word deltas")
    md_lines.append(f"- HS-LS: {hs_ls_word_delta}")
    md_lines.append(f"- HD-LD: {hd_ld_word_delta}")
    md_lines.append(f"- HS-HD: {hs_hd_word_delta}")
    md_lines.append(f"- LS-LD: {ls_ld_word_delta}")
    md_lines.append(f"- Interaction (HS-LS)-(HD-LD): {interaction_word_delta}")
    if not bpe_audit.get("skipped"):
        md_lines.append(f"\n## BPE audit (first {bpe_audit.get('sample_changed_rows', '?')} changed rows)")
        md_lines.append(f"- HS-LS BPE: {bpe_audit.get('HS_minus_LS_bpe')}")
        md_lines.append(f"- HD-LD BPE: {bpe_audit.get('HD_minus_LD_bpe')}")
        md_lines.append(f"- Interaction BPE: {bpe_audit.get('interaction_bpe')}")
    (out_dir / "factorial_stream_report.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    print(json.dumps({"status": report["status"], "out_json": str(out_dir / "factorial_stream_report.json"), "elapsed_sec": report["elapsed_sec"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
