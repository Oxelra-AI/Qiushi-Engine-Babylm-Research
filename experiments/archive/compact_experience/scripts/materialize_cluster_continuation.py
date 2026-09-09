#!/usr/bin/env python3
"""research: Materialize four-arm continuation data for cluster mechanism test.

Takes quality-filtered clusters and the original clean-Qwen 10M pool, then builds
four MATCHED 10M pools for 80M→100M continuation training:

  E1 (true_cluster): Selected cluster sentences packed adjacently in rows, padded
                     with official filler to reach exactly 160 words per replaced row
  E2 (anchor_shuffle): Same anchors but sentences from different blocks, same padding
  E3 (anchor_repeat): One sentence repeated to match anchor/word exposure, same padding
  E4 (untouched_tail): Original clean-Qwen 10M pool unchanged (continuation baseline)

All arms have EXACTLY the same total word count (10M) and row count (64381).
The only difference is the content within the cluster-intervention rows.

Does NOT use official AoA/CDI items, evaluation outputs, or downstream labels.
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
import pathlib
import random
import time
from typing import Any

ROOT = _public_path('experiments/archive/compact_experience')
DEFAULT_CLUSTERS = _public_path('experiments/archive/compact_experience/data/cluster_mechanism_test/filtered_clusters.jsonl')
DEFAULT_POOL = _public_path('experiments/archive/compact_experience/data/qwen_clean_aligned/training_corpora/qwen_aligned_10M.jsonl')
DEFAULT_OUT = _public_path('experiments/archive/compact_experience/data/cluster_mechanism_test')
WORDS_PER_ROW = 160


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def word_count(text: str) -> int:
    return len(text.split())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clusters", type=str, default=str(DEFAULT_CLUSTERS))
    ap.add_argument("--base_pool", type=str, default=str(DEFAULT_POOL))
    ap.add_argument("--output_dir", type=str, default=str(DEFAULT_OUT))
    ap.add_argument("--seed", type=int, default=43043)
    args = ap.parse_args()

    out_dir = pathlib.Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    # === Load filtered clusters ===
    clusters: list[dict] = []
    with open(args.clusters, "r") as f:
        for line in f:
            if line.strip():
                clusters.append(json.loads(line))
    print(f"Loaded {len(clusters)} filtered clusters")

    # === Load base pool ===
    pool_rows: list[dict] = []
    with open(args.base_pool, "r") as f:
        for line in f:
            if line.strip():
                pool_rows.append(json.loads(line))
    pool_total_words = sum(r['words'] for r in pool_rows)
    print(f"Loaded {len(pool_rows)} base pool rows ({pool_total_words} words)")

    # === Group clusters into 160-word packed rows ===
    # Strategy: pack consecutive clusters until we reach ~160 words, then
    # pad with official filler text to reach exactly 160.
    # This ensures replaced rows match original word count perfectly.
    
    # First, build cluster packs: groups of clusters that fit ~120-150 words
    # (leaving 10-40 words for filler padding)
    cluster_packs: list[list[int]] = []  # indices into clusters
    current_pack: list[int] = []
    current_words = 0
    MAX_CLUSTER_FILL = 145  # leave room for padding
    
    for ci, c in enumerate(clusters):
        cwords = c.get("words", 0)
        if current_words + cwords > MAX_CLUSTER_FILL and current_pack:
            cluster_packs.append(current_pack)
            current_pack = [ci]
            current_words = cwords
        else:
            current_pack.append(ci)
            current_words += cwords
    if current_pack:
        cluster_packs.append(current_pack)
    
    print(f"Packed into {len(cluster_packs)} row-groups")

    # === Identify filler rows for replacement ===
    filler_indices: list[int] = []
    for i, row in enumerate(pool_rows):
        source = row.get("source", "")
        if not source.startswith("qwen") and not source.startswith("pair") and not source.startswith("cluster"):
            filler_indices.append(i)
    
    n_needed = len(cluster_packs)
    if n_needed > len(filler_indices):
        print(f"WARNING: need {n_needed} slots but only {len(filler_indices)} available. Truncating.")
        cluster_packs = cluster_packs[:len(filler_indices)]
        n_needed = len(cluster_packs)
    
    # Select replacement positions (deterministic)
    rng_pos = random.Random(args.seed + 1)
    selected_positions = sorted(rng_pos.sample(filler_indices, n_needed))
    
    # Collect filler text snippets for padding (from non-selected filler rows)
    non_selected_filler = [i for i in filler_indices if i not in set(selected_positions)]
    filler_word_bank: list[str] = []
    for idx in non_selected_filler[:500]:  # use first 500 non-selected rows as padding source
        filler_word_bank.extend(pool_rows[idx]["text"].split())
    filler_bank_pos = 0

    def get_padding_words(n: int) -> str:
        """Get n filler words for padding."""
        nonlocal filler_bank_pos
        if filler_bank_pos + n > len(filler_word_bank):
            filler_bank_pos = 0  # wrap around
        words = filler_word_bank[filler_bank_pos:filler_bank_pos + n]
        filler_bank_pos += n
        return " ".join(words)

    # === Build row replacements for each arm ===
    # Group clusters by anchor for E2 cross-block lookup
    by_anchor: dict[str, list[dict]] = collections.defaultdict(list)
    for c in clusters:
        by_anchor[c.get("anchor", "")].append(c)

    def build_arm_rows(arm_name: str) -> list[dict]:
        """Build replacement rows for one arm."""
        nonlocal filler_bank_pos
        filler_bank_pos = 0  # reset for each arm to keep padding consistent size
        
        arm_rows: list[dict] = []
        for pack_idx, pack_cindices in enumerate(cluster_packs):
            target_words = pool_rows[selected_positions[pack_idx]]["words"]
            
            if arm_name == "E1_true_cluster":
                # Pack cluster sentences adjacently
                texts_list = []
                for ci in pack_cindices:
                    texts_list.extend(clusters[ci]["texts"])
                core_text = " ".join(texts_list)
            
            elif arm_name == "E2_anchor_shuffle":
                # Same anchors but cross-block sentences
                texts_list = []
                for ci in pack_cindices:
                    c = clusters[ci]
                    anchor = c.get("anchor", "")
                    block_id = c.get("block_id", "")
                    orig_texts = c.get("texts", [])
                    
                    # Find cross-block alternatives
                    alternatives = []
                    for other in by_anchor.get(anchor, []):
                        if other["block_id"] != block_id:
                            alternatives.extend(other.get("texts", []))
                    
                    if alternatives:
                        # Replace one sentence with cross-block alternative
                        shuffled = list(orig_texts)
                        replace_idx = rng.randint(0, len(shuffled) - 1)
                        shuffled[replace_idx] = rng.choice(alternatives)
                        texts_list.extend(shuffled)
                    else:
                        texts_list.extend(reversed(orig_texts))
                
                core_text = " ".join(texts_list)
            
            elif arm_name == "E3_anchor_repeat":
                # Repeat first sentence of each cluster
                texts_list = []
                for ci in pack_cindices:
                    c = clusters[ci]
                    orig_texts = c.get("texts", [])
                    if orig_texts:
                        first = orig_texts[0]
                        # Repeat to approximately match original cluster word count
                        target_cw = sum(word_count(t) for t in orig_texts)
                        first_wc = word_count(first)
                        repeats = max(2, round(target_cw / max(first_wc, 1)))
                        repeated = " ".join([first] * repeats)
                        # Trim to target
                        texts_list.append(" ".join(repeated.split()[:target_cw + 3]))
                
                core_text = " ".join(texts_list)
            
            else:
                raise ValueError(f"Unknown arm: {arm_name}")
            
            # Pad or trim to exactly target_words
            core_words = core_text.split()
            if len(core_words) >= target_words:
                final_text = " ".join(core_words[:target_words])
            else:
                pad_needed = target_words - len(core_words)
                padding = get_padding_words(pad_needed)
                final_text = core_text + " " + padding
            
            # Verify exact word count
            final_wc = word_count(final_text)
            assert final_wc == target_words, f"Row {pack_idx}: expected {target_words}, got {final_wc}"
            
            arm_rows.append({
                "text": final_text,
                "words": target_words,
                "example_id": pool_rows[selected_positions[pack_idx]].get("example_id", 0),
                "source": f"{arm_name}_{clusters[pack_cindices[0]].get('source', 'unknown')}",
            })
        
        return arm_rows

    # === Generate all arms ===
    arms_data: dict[str, list[dict]] = {}
    for arm in ["E1_true_cluster", "E2_anchor_shuffle", "E3_anchor_repeat"]:
        arms_data[arm] = build_arm_rows(arm)
        total_arm_words = sum(r["words"] for r in arms_data[arm])
        print(f"  {arm}: {len(arms_data[arm])} rows, {total_arm_words} replacement words")

    # === Write four pools ===
    pool_paths = {}
    pool_shas = {}

    for arm_name, arm_rows in arms_data.items():
        arm_pool = list(pool_rows)  # copy
        for pos_idx, row_idx in enumerate(selected_positions):
            if pos_idx < len(arm_rows):
                arm_pool[row_idx] = arm_rows[pos_idx]
        
        out_path = out_dir / f"continuation_10M_{arm_name}.jsonl"
        total_w = 0
        with out_path.open("w") as f:
            for row in arm_pool:
                out_row = {
                    "text": row["text"],
                    "words": row["words"],
                    "example_id": row.get("example_id", 0),
                    "source": row.get("source", ""),
                }
                f.write(json.dumps(out_row, ensure_ascii=False) + "\n")
                total_w += row["words"]
        
        pool_shas[arm_name] = sha256_bytes(out_path.read_bytes())
        pool_paths[arm_name] = str(out_path)
        print(f"  {arm_name}: {total_w} total words -> {out_path.name}")

    # E4: untouched tail (identical to base pool)
    e4_path = out_dir / "continuation_10M_E4_untouched_tail.jsonl"
    e4_total = 0
    with e4_path.open("w") as f:
        for row in pool_rows:
            out_row = {
                "text": row["text"],
                "words": row["words"],
                "example_id": row.get("example_id", 0),
                "source": row.get("source", ""),
            }
            f.write(json.dumps(out_row, ensure_ascii=False) + "\n")
            e4_total += row["words"]
    
    pool_shas["E4_untouched_tail"] = sha256_bytes(e4_path.read_bytes())
    pool_paths["E4_untouched_tail"] = str(e4_path)
    print(f"  E4_untouched_tail: {e4_total} total words -> {e4_path.name}")

    # === Verify all pools have identical word counts ===
    print("\n=== Word count verification ===")
    for arm_name in ["E1_true_cluster", "E2_anchor_shuffle", "E3_anchor_repeat", "E4_untouched_tail"]:
        p = out_dir / f"continuation_10M_{arm_name}.jsonl"
        total = 0
        with p.open("r") as f:
            for line in f:
                if line.strip():
                    total += json.loads(line)["words"]
        print(f"  {arm_name}: {total} words")

    # === Summary metadata ===
    summary = {
        "status": "CLUSTER_CONTINUATION_MATERIALIZED",
        "created_utc": now(),
        "design": "80M->100M continuation from clean-Qwen chck_80M with matched 10M pools",
        "arms": {
            "E1_true_cluster": "Cluster sentences packed adjacently (complementary predicate evidence), padded to 160w",
            "E2_anchor_shuffle": "Same anchors, cross-block sentences (anchor-exposure control), padded to 160w",
            "E3_anchor_repeat": "First sentence repeated (repetition/exposure control), padded to 160w",
            "E4_untouched_tail": "Original clean-Qwen pool unchanged (pure continuation baseline)",
        },
        "clusters_used": len(clusters),
        "cluster_packs_inserted": len(cluster_packs),
        "rows_replaced_per_arm": n_needed,
        "total_pool_rows": len(pool_rows),
        "word_matching": "All arms have identical total word counts (rows padded to original filler row length)",
        "pool_sha256": pool_shas,
        "pool_paths": pool_paths,
        "continuation_spec": {
            "init_checkpoint": "training/runs/qwen_clean_aligned_16k_seed43022/hf_model/chck_80M",
            "exposure_budget": 20_000_000,
            "passes_over_pool": 2,
            "checkpoint_interval_words": 5_000_000,
            "expected_checkpoints": ["chck_85M", "chck_90M", "chck_95M", "chck_100M"],
        },
        "non_leakage_statement": "No official AoA/CDI words, child curves, or downstream evaluation outputs used.",
    }

    summary_path = out_dir / "continuation_materialization_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")

    print(json.dumps({
        "summary": str(summary_path),
        "arms": list(pool_paths.keys()),
        "cluster_packs": len(cluster_packs),
        "pool_paths": pool_paths,
    }, indent=2))


if __name__ == "__main__":
    main()
