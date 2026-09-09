#!/usr/bin/env python3
"""research: build source-conditioned auxiliary pair data for corrected dual-view training.

For each of the 3,005 legal pair rows (eid 950000-953004) build, per constituent pair:
  - source token ids (true source)
  - rewrite token ids + rewrite word groups
  - the char span of the rewrite inside the FULL packed row, and the full-row token
    indices covering it, so the main-view WWM mask on rewrite word groups can be
    transferred to the auxiliary rewrite-only segment identically.

Also build a deterministic length-matched (same source_words) different-doc SHUFFLE map at
the pair level, guaranteeing the shuffled arm changes only source correspondence, not source
length or token budget.

Output: data/aux_pair_data/aux_pair_data.json
"""
import json, os, time, hashlib, collections, random
from pathlib import Path

STUDY = Path("experiments/archive/frontier_consolidation")
WORKSPACE = STUDY
POOL_10M = WORKSPACE / "data/density_cleanqwen_overlay_medium_riskhard/cleanqwen_fineweb_compact_view_reinvest_10M.jsonl"
PAIR_JSONL = WORKSPACE / "data/density_core_reinvestment_medium_riskhard/selected_compact_reinvest_pairs.jsonl"
META_JSONL = WORKSPACE / "data/density_cleanqwen_overlay_medium_riskhard/cleanqwen_fineweb_compact_view_reinvest_changed_block_rows_meta.jsonl"
TOKENIZER_DIR = WORKSPACE / "data/compliant_tokenizer"
OUT_DIR = WORKSPACE / "data/aux_pair_data"
SEQ_LENGTH = 256
SHUFFLE_SEED = 43022


def is_word_start(s: str) -> bool:
    return s.startswith("Ġ") or s.startswith("▁")


def word_groups(input_ids, special_ids, tok):
    group = [-1] * len(input_ids)
    gid = -1
    cache = {}
    for i, tid in enumerate(input_ids):
        if tid in special_ids:
            continue
        if tid not in cache:
            s = tok.convert_ids_to_tokens(int(tid))
            cache[tid] = bool(s is not None and is_word_start(str(s)))
        if gid < 0 or cache[tid] or i == 0:
            gid += 1
        group[i] = gid
    return group


def main():
    start = time.time()
    hf_cache = str(OUT_DIR / "hf_cache")
    os.makedirs(hf_cache, exist_ok=True)
    os.environ.setdefault("HF_HOME", hf_cache)
    os.environ.setdefault("TRANSFORMERS_CACHE", hf_cache)
    os.environ.setdefault("HF_MODULES_CACHE", str(Path(hf_cache) / "modules"))
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(TOKENIZER_DIR))
    special_ids = set(tok.all_special_ids)

    pair_by_id = {}
    for line in open(PAIR_JSONL):
        p = json.loads(line)
        pair_by_id[p["pair_id"]] = p

    meta_by_eid = {}
    for line in open(META_JSONL):
        m = json.loads(line)
        meta_by_eid[int(m["example_id"])] = m

    pair_rows = {}
    for line in open(POOL_10M):
        row = json.loads(line)
        eid = int(row["example_id"])
        if eid in meta_by_eid:
            pair_rows[eid] = row

    # ── Build a global pair->source-ids table and length-matched shuffle map ──
    # Each pair id: source ids (true). Group by source_words count for derangement.
    all_pair_ids = []
    src_words_of = {}
    src_ids_of = {}
    for pid, p in pair_by_id.items():
        s_ids = tok(p["source_text"], add_special_tokens=False)["input_ids"]
        src_ids_of[pid] = list(s_ids)
        src_words_of[pid] = int(p["source_words"])
        all_pair_ids.append(pid)

    by_count = collections.defaultdict(list)
    for pid in all_pair_ids:
        by_count[src_words_of[pid]].append(pid)

    rng = random.Random(SHUFFLE_SEED)
    shuffle_map = {}  # pid -> decoy pid (same source_words, different pid, prefer diff doc)
    for cnt, group in by_count.items():
        if len(group) < 2:
            # cannot derange a singleton group; map to itself flagged
            for pid in group:
                shuffle_map[pid] = None
            continue
        g = list(group)
        rng.shuffle(g)
        # rotation derangement guarantees no fixed point for len>=2
        for i, pid in enumerate(g):
            decoy = g[(i + 1) % len(g)]
            shuffle_map[pid] = decoy

    n_singleton = sum(1 for v in shuffle_map.values() if v is None)

    results = {}
    n_ok = 0
    n_fail = 0
    n_used_pairs = 0
    n_shuffle_diff_doc = 0
    n_shuffle_same_doc = 0
    n_shuffle_missing = 0

    for eid in sorted(pair_rows.keys()):
        row = pair_rows[eid]
        meta = meta_by_eid[eid]
        full_text = row["text"]
        pair_ids = meta["pair_ids"]

        full_enc = tok(full_text, add_special_tokens=False, truncation=True,
                       max_length=SEQ_LENGTH, return_offsets_mapping=True)
        full_ids = list(full_enc["input_ids"])
        full_offs = full_enc["offset_mapping"]
        full_wg = word_groups(full_ids, special_ids, tok)

        cursor = 0
        pair_records = []
        ok = True
        for pid in pair_ids:
            if pid not in pair_by_id:
                ok = False
                break
            p = pair_by_id[pid]
            src = p["source_text"]
            rw = p["rewrite_text"]
            src_idx = full_text.find(src, cursor)
            rw_idx = full_text.find(rw, cursor)
            if src_idx < 0 or rw_idx < 0:
                ok = False
                break
            rw_start = rw_idx
            rw_end = rw_idx + len(rw)
            # full-row token indices covering the rewrite
            rw_full_tok_idx = [i for i, (a, b) in enumerate(full_offs)
                               if b > a and a >= rw_start and b <= rw_end]
            rw_full_wgs = sorted(set(full_wg[i] for i in rw_full_tok_idx if full_wg[i] >= 0))

            # rewrite-only tokenization + word groups
            rw_enc = tok(rw, add_special_tokens=False, truncation=True,
                         max_length=SEQ_LENGTH, return_offsets_mapping=True)
            rw_ids = list(rw_enc["input_ids"])
            rw_offs = rw_enc["offset_mapping"]
            rw_wg = word_groups(rw_ids, special_ids, tok)

            # map full-rewrite word-group -> rewrite-only word-group by char alignment
            rw_char_to_tok = {}
            for j, (a, b) in enumerate(rw_offs):
                if b <= a:
                    continue
                for c in range(a, b):
                    rw_char_to_tok[c] = j
            full_wg_to_rw_wg = {}
            for i in rw_full_tok_idx:
                a, b = full_offs[i]
                rw_char = a - rw_start
                j = rw_char_to_tok.get(rw_char)
                if j is None:
                    continue
                fwg = full_wg[i]
                rwg = rw_wg[j]
                if fwg >= 0 and rwg >= 0:
                    full_wg_to_rw_wg[fwg] = rwg

            decoy = shuffle_map.get(pid)
            if decoy is None:
                decoy_src_ids = src_ids_of[pid]  # fallback: same (flagged)
                n_shuffle_missing += 1
            else:
                decoy_src_ids = src_ids_of[decoy]
                if str(pair_by_id[decoy].get("doc_id")) != str(p.get("doc_id")):
                    n_shuffle_diff_doc += 1
                else:
                    n_shuffle_same_doc += 1

            pair_records.append({
                "pair_id": pid,
                "source_ids": src_ids_of[pid],
                "decoy_pair_id": decoy,
                "decoy_source_ids": decoy_src_ids,
                "source_words": src_words_of[pid],
                "rewrite_words": int(p["rewrite_words"]),
                "rw_ids": rw_ids,
                "rw_word_group": rw_wg,
                "full_rw_word_groups": rw_full_wgs,
                "full_wg_to_rw_wg": {str(k): v for k, v in full_wg_to_rw_wg.items()},
            })
            cursor = max(src_idx + len(src), rw_end)
            n_used_pairs += 1

        if not ok or not pair_records:
            n_fail += 1
            continue

        results[str(eid)] = {
            "example_id": eid,
            "n_pairs": len(pair_records),
            "pairs": pair_records,
        }
        n_ok += 1

    summary = {
        "status": "AUX_PAIR_DATA",
        "n_pair_rows": len(pair_rows),
        "n_ok": n_ok,
        "n_fail": n_fail,
        "n_used_pairs": n_used_pairs,
        "shuffle_seed": SHUFFLE_SEED,
        "shuffle_singleton_groups_flagged": n_singleton,
        "shuffle_diff_doc": n_shuffle_diff_doc,
        "shuffle_same_doc": n_shuffle_same_doc,
        "shuffle_missing_fallback_selfsource": n_shuffle_missing,
        "tokenizer_sha": hashlib.sha256((TOKENIZER_DIR / "tokenizer.json").read_bytes()).hexdigest(),
        "pool_sha": hashlib.sha256(POOL_10M.read_bytes()).hexdigest(),
        "seq_length": SEQ_LENGTH,
        "elapsed_sec": round(time.time() - start, 2),
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "aux_pair_data.json").write_text(
        json.dumps({"summary": summary, "pair_data": results}, ensure_ascii=False))
    (OUT_DIR / "aux_pair_data_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
