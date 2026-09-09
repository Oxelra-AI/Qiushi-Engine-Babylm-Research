#!/usr/bin/env python3
"""research: Build S and G 10M pools and 100M training streams.

Uses the same filler rows as the historical compact_view_reinvest pool.
Replaces each compact rewrite in the pair-block rows with S_text or G_text
from the v2 marginal corpora, preserving source text and row geometry.

Output: 10M pool and 100M stream for each of S and G arms.
"""
import json, hashlib, pathlib, time

POOL_PATH = "experiments/archive/frontier_consolidation/data/density_core_reinvestment_medium_riskhard/fineweb_compact_view_reinvest_10M.jsonl"
META_PATH = "experiments/archive/frontier_consolidation/data/density_core_reinvestment_medium_riskhard/fineweb_compact_view_reinvest_changed_block_rows_meta.jsonl"
V2_PATH   = "experiments/archive/representation_and_objectives/data/marginal_corpora_v2/marginal_corpora_csg_v2.jsonl"
OUT_DIR   = pathlib.Path("experiments/archive/representation_and_objectives/data/training_streams")

def main():
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load v2 corpora indexed by pair_id
    print("Loading v2 marginal corpora...", flush=True)
    v2_by_id = {}
    with open(V2_PATH) as f:
        for line in f:
            r = json.loads(line)
            v2_by_id[r["pair_id"]] = r
    print(f"  {len(v2_by_id)} pairs loaded", flush=True)

    # Load changed-block metadata
    print("Loading block metadata...", flush=True)
    block_meta = []
    with open(META_PATH) as f:
        for line in f:
            block_meta.append(json.loads(line))
    block_row_indices = {m["row_index"] for m in block_meta}
    print(f"  {len(block_meta)} block rows, indices {min(block_row_indices)}-{max(block_row_indices)}", flush=True)

    # Load pair data to reconstruct source texts
    print("Loading original pairs for source text...", flush=True)
    pairs_path = "experiments/archive/frontier_consolidation/data/density_core_reinvestment_medium_riskhard/selected_compact_reinvest_pairs.jsonl"
    orig_pairs = {}
    with open(pairs_path) as f:
        for line in f:
            p = json.loads(line)
            orig_pairs[p["pair_id"]] = p
    print(f"  {len(orig_pairs)} original pairs", flush=True)

    # Build block-row metadata index: row_index -> list of pair_ids
    block_pairs = {}
    for m in block_meta:
        block_pairs[m["row_index"]] = m["pair_ids"]

    # Read the full original pool
    print("Reading original 10M pool...", flush=True)
    pool_rows = []
    with open(POOL_PATH) as f:
        for line in f:
            pool_rows.append(json.loads(line))
    print(f"  {len(pool_rows)} rows, {sum(r['words'] for r in pool_rows)} words", flush=True)

    # For each arm (S and G), build replacement rows
    for arm_name, text_key in [("S", "S_text"), ("G", "G_text")]:
        print(f"\nBuilding {arm_name} pool...", flush=True)
        new_rows = []
        total_words = 0
        pair_words = 0
        filler_words = 0
        mismatches = 0

        for i, orig_row in enumerate(pool_rows):
            if i in block_pairs:
                # Reconstruct this row with the new arm's side text
                pair_ids = block_pairs[i]
                parts = []
                for pid in pair_ids:
                    orig = orig_pairs[pid]
                    v2 = v2_by_id[pid]
                    # source_text stays the same
                    source = orig["source_text"]
                    side = v2[text_key]
                    parts.append(source)
                    parts.append(side)
                
                new_text = " ".join(parts)
                new_words = len(new_text.split())
                
                # Check word count matches original
                if new_words != orig_row["words"]:
                    mismatches += 1
                    if mismatches <= 5:
                        print(f"  WARN row {i}: new={new_words} vs orig={orig_row['words']}", flush=True)
                
                new_row = {
                    "text": new_text,
                    "words": new_words,
                    "example_id": orig_row["example_id"],
                    "source": f"fineweb_{arm_name.lower()}_extract_reinvest",
                }
                new_rows.append(new_row)
                pair_words += new_words
            else:
                # Filler row: keep unchanged
                new_rows.append(orig_row)
                filler_words += orig_row["words"]
            
            total_words += new_rows[-1]["words"]

        # Save 10M pool
        pool_out = OUT_DIR / f"arm_{arm_name.lower()}_10M_pool.jsonl"
        with open(pool_out, "w") as f:
            for r in new_rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        pool_sha = hashlib.sha256(open(pool_out, "rb").read()).hexdigest()

        # Build 100M stream (10 epochs)
        stream_out = OUT_DIR / f"arm_{arm_name.lower()}_100M_stream.jsonl"
        stream_words = 0
        with open(stream_out, "w") as f:
            for epoch in range(10):
                for r in new_rows:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
                    stream_words += r["words"]
        stream_sha = hashlib.sha256(open(stream_out, "rb").read()).hexdigest()

        print(f"  {arm_name}: pool_rows={len(new_rows)} pool_words={total_words} "
              f"pair_words={pair_words} filler_words={filler_words} "
              f"mismatches={mismatches}")
        print(f"  Stream: {stream_words} words ({stream_words//1000000}M)")
        print(f"  Pool SHA: {pool_sha}")
        print(f"  Stream SHA: {stream_sha}")

    # Save manifest
    manifest = {
        "status": "TRAINING_STREAMS_BUILT",
        "arms": ["S", "G"],
        "pool_rows": len(pool_rows),
        "filler_rows": len(pool_rows) - len(block_meta),
        "pair_block_rows": len(block_meta),
        "original_pool_sha": hashlib.sha256(open(POOL_PATH, "rb").read()).hexdigest(),
        "v2_corpora_sha": "d206742ab59154f2f41787a4c005cb19fd33b420b69c755b65cddcd3d7ca63f6",
        "elapsed_sec": round(time.time() - t0, 1),
    }
    
    manifest_out = OUT_DIR / "stream_manifest.json"
    with open(manifest_out, "w") as f:
        json.dump(manifest, f, indent=2)
    
    print(f"\nManifest: {manifest_out}")
    print(f"Elapsed: {manifest['elapsed_sec']}s")


if __name__ == "__main__":
    main()
