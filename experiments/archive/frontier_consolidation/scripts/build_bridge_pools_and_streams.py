#!/usr/bin/env python3
"""research: build bridge_compactgap and bridge_mid 10M pools + 100M streams.

Reads the research source-attested continuity pairs, replaces compact view text
in the existing pool/stream structure, and writes verified pools and streams.

CPU/file-only. No training, evaluation, upload, or leaderboard action.
"""

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative

import json, hashlib, time, sys
from pathlib import Path

def words(t): return [w for w in t.split() if w]
def sha256f(p):
    h=hashlib.sha256()
    with open(p,"rb") as f:
        for c in iter(lambda:f.read(1<<20),b""): h.update(c)
    return h.hexdigest()
def now(): return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

def main():
    ROOT = _public_path('experiments/archive/frontier_consolidation/scripts/build_bridge_pools_and_streams.py')
    for p in [ROOT]+list(ROOT.parents):
        if (p/ "experiments").exists() and (p/ "CITATION.cff").exists(): ROOT=p; break
    WS = ROOT/"experiments/archive/frontier_consolidation"

    # Load bridge pair views
    bridge_path = WS/"data/source_attested_continuity_bridge/source_attested_continuity_pairs.jsonl"
    print(f"Loading bridge pairs from {bridge_path} ...", flush=True)
    bridge_views = {"bridge_compactgap": {}, "bridge_mid": {}}
    with open(bridge_path) as f:
        for line in f:
            rec = json.loads(line)
            pid = rec["pair_id"]
            for vname in bridge_views:
                bridge_views[vname][pid] = rec["variants"][vname]["text"]
    print(f"  Loaded {len(bridge_views['bridge_compactgap'])} pairs for each variant", flush=True)

    # Load pair source texts
    pairs_path = WS/"data/factorial_view_candidate_audit/compact_candidate_pairs.jsonl"
    pair_sources = {}
    with open(pairs_path) as f:
        for line in f:
            p = json.loads(line)
            pair_sources[p["pair_id"]] = p["source_text"]
    print(f"  Loaded {len(pair_sources)} pair sources", flush=True)

    # Load changed-block row metadata
    meta_path = WS/"data/density_cleanqwen_overlay_medium_riskhard/cleanqwen_fineweb_compact_view_reinvest_changed_block_rows_meta.jsonl"
    row_metas = [json.loads(x) for x in open(meta_path)]
    print(f"  {len(row_metas)} changed-block rows", flush=True)

    # Load compact 10M pool (for filler and topup rows)
    compact_pool_path = WS/"data/density_cleanqwen_overlay_medium_riskhard/cleanqwen_fineweb_compact_view_reinvest_10M.jsonl"
    compact_rows = [json.loads(x) for x in open(compact_pool_path)]
    n_changed = len(row_metas)
    filler_rows = compact_rows[n_changed:]
    print(f"  {len(compact_rows)} total pool rows, {len(filler_rows)} filler", flush=True)

    # Build 10M pools
    OUT = WS/"data/bridge_pools_and_streams"
    OUT.mkdir(parents=True, exist_ok=True)

    pool_results = {}
    for vname in bridge_views:
        pool_path = OUT / f"{vname}_10M.jsonl"
        print(f"\nBuilding {vname} 10M pool -> {pool_path} ...", flush=True)
        total_words = 0; row_count = 0; mismatches = 0
        with open(pool_path, "w") as fout:
            for ri, rmeta in enumerate(row_metas):
                if not rmeta["pair_ids"]:
                    # Topup row: keep compact pool text
                    row_data = compact_rows[ri]
                    json.dump(row_data, fout, ensure_ascii=False); fout.write("\n")
                    total_words += row_data["words"]; row_count += 1
                    continue
                # Rebuild from source + bridge view
                parts = []
                for pid in rmeta["pair_ids"]:
                    parts.append(pair_sources[pid])
                    parts.append(bridge_views[vname][pid])
                row_text = " ".join(parts)
                actual_words = len(words(row_text))
                if actual_words != rmeta["words"]:
                    mismatches += 1
                    if mismatches <= 5:
                        print(f"  WARNING row {ri}: expected {rmeta['words']}, got {actual_words}", flush=True)
                row_data = {"text": row_text, "words": actual_words,
                           "example_id": rmeta["example_id"], "source": f"bridge_{vname}"}
                json.dump(row_data, fout, ensure_ascii=False); fout.write("\n")
                total_words += actual_words; row_count += 1
            # Filler
            for frow in filler_rows:
                json.dump(frow, fout, ensure_ascii=False); fout.write("\n")
                total_words += frow["words"]; row_count += 1
        pool_sha = sha256f(pool_path)
        pool_results[vname] = {"pool_path": str(pool_path), "pool_sha256": pool_sha,
                               "rows": row_count, "total_words": total_words,
                               "word_mismatches": mismatches}
        print(f"  {vname}: {row_count} rows, {total_words} words, mismatches={mismatches}, SHA={pool_sha[:16]}...", flush=True)

    # Build 100M streams by replacing changed-block rows in compact 100M stream
    compact_stream_path = WS/"data/density_cleanqwen_overlay_medium_riskhard/cleanqwen_fineweb_compact_view_reinvest_100M.jsonl"
    # Build changed-row lookup
    changed_ids = set()
    bridge_row_texts = {"bridge_compactgap": {}, "bridge_mid": {}}
    for ri, rmeta in enumerate(row_metas):
        if not rmeta["pair_ids"]:
            continue
        eid = rmeta["example_id"]
        changed_ids.add(eid)
        for vname in bridge_views:
            parts = []
            for pid in rmeta["pair_ids"]:
                parts.append(pair_sources[pid])
                parts.append(bridge_views[vname][pid])
            bridge_row_texts[vname][eid] = " ".join(parts)

    stream_results = {}
    for vname in bridge_views:
        stream_path = OUT / f"{vname}_100M.jsonl"
        print(f"\nBuilding {vname} 100M stream -> {stream_path} ...", flush=True)
        total_words = 0; total_rows = 0; replaced = 0; changed_eids = set()
        with open(compact_stream_path) as fin, open(stream_path, "w") as fout:
            for line in fin:
                row = json.loads(line)
                eid = row["example_id"]
                if eid in changed_ids:
                    new_text = bridge_row_texts[vname][eid]
                    new_words = len(words(new_text))
                    row_out = {"text": new_text, "words": new_words,
                              "example_id": eid, "source": f"bridge_{vname}"}
                    json.dump(row_out, fout, ensure_ascii=False); fout.write("\n")
                    total_words += new_words; replaced += 1; changed_eids.add(eid)
                else:
                    fout.write(line)
                    total_words += row["words"]
                total_rows += 1
        stream_sha = sha256f(stream_path)
        stream_results[vname] = {"stream_path": str(stream_path), "stream_sha256": stream_sha,
                                 "total_rows": total_rows, "total_words": total_words,
                                 "replaced_rows": replaced, "changed_ids": len(changed_eids),
                                 "filler_rows": total_rows - replaced}
        print(f"  {vname}: {total_rows} rows, {total_words} words, {replaced} replaced, SHA={stream_sha[:16]}...", flush=True)

    # Write manifest
    manifest = {"status": "BRIDGE_POOLS_AND_STREAMS", "created_utc": now(),
                "pool_results": pool_results, "stream_results": stream_results,
                "compact_stream_sha256": sha256f(compact_stream_path),
                "no_training_eval_upload_aoa_or_leaderboard": True}
    mpath = OUT/"bridge_pools_and_streams_manifest.json"
    with open(mpath, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nManifest: {mpath}", flush=True)
    print(json.dumps({"status": manifest["status"], "out": str(OUT),
                      "pools": {k: v["total_words"] for k,v in pool_results.items()},
                      "streams": {k: v["total_words"] for k,v in stream_results.items()}}, indent=2))

if __name__ == "__main__":
    main()
