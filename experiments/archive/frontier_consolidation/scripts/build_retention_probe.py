#!/usr/bin/env python3
"""research: build a fixed, label-free, corpus-internal retention probe set.

The probe set is drawn from the legal 10M compact-view reinvest pool, stratified by
the natural source labels. Each probe row gets a single FIXED deterministic WWM mask
so that per-checkpoint masked-token NLL differences are pure checkpoint effects (the
mask never changes across checkpoints). This is INFERENCE-ONLY: the probe is never
used to update weights. It is a benchmark-independent (no official-task labels)
retention measurement, stratified by source and by target-token structure
(function-word vs content-word).

Output: data/retention_probe/retention_probe.json
  { "rows": [ {source, text, word_groups, masked_word_group_idx, target_word,
               is_function_word} ... ],
    "summary": {...} }
"""
from __future__ import annotations
import json, random, hashlib
from pathlib import Path
from collections import defaultdict

ROOT = Path("experiments/archive/frontier_consolidation")
POOL = ROOT / "data/density_cleanqwen_overlay_medium_riskhard/cleanqwen_fineweb_compact_view_reinvest_10M.jsonl"
TOKJSON = ROOT / "data/compliant_tokenizer/tokenizer.json"
OUT_DIR = ROOT / "data/retention_probe"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Fixed English function-word list (closed-class): determiners, pronouns, aux,
# prepositions, conjunctions, particles. Structure stratum, not a benchmark label.
FUNCTION_WORDS = set("""
a an the this that these those my your his her its our their whose
i you he she it we they me him us them mine yours hers ours theirs
am is are was were be been being do does did have has had having
will would shall should can could may might must ought need dare
of in on at by for with about against between into through during
before after above below to from up down over under again further
then once here there when where why how all any both each few more
most other some such no nor not only own same so than too very
and but or yet because as until while if unless although though
whether either neither about across along around behind beneath beside
besides beyond inside outside toward towards upon within without
what which who whom whose
""".split())

def load_wordlevel_vocab():
    obj = json.loads(TOKJSON.read_text())
    return obj

def main(rows_per_source=320, seed=135, max_words=260, masks_per_row=4):
    rng = random.Random(seed)
    by_source = defaultdict(list)
    with open(POOL) as f:
        for line in f:
            o = json.loads(line)
            s = o["source"]
            # skip the tiny neutral topup singleton
            if s.startswith("neutral_"):
                continue
            by_source[s].append(o)

    probe_rows = []
    summary_counts = {}
    for s, rows in sorted(by_source.items()):
        rng.shuffle(rows)
        picked = 0
        for o in rows:
            if picked >= rows_per_source:
                break
            text = o["text"]
            words = text.split()
            if len(words) < 8 or len(words) > max_words:
                continue
            # Deterministic masked word-groups: pick `masks_per_row` distinct
            # content-bearing positions deterministically from a hash of the text
            # so the mask set is fixed forever. Each becomes one probe target.
            cand = [i for i, w in enumerate(words) if 0 < i < len(words) - 1]
            if len(cand) < masks_per_row:
                continue
            h = int(hashlib.sha256((str(seed) + "|" + text).encode()).hexdigest(), 16)
            chosen = []
            pool_idx = list(cand)
            hh = h
            for _ in range(masks_per_row):
                if not pool_idx:
                    break
                j = hh % len(pool_idx)
                chosen.append(pool_idx.pop(j))
                hh //= max(1, len(pool_idx) + 1)
            for idx in chosen:
                target_word = words[idx]
                tw = target_word.strip(".,!?;:'\"()[]").lower()
                is_fw = tw in FUNCTION_WORDS
                probe_rows.append({
                    "source": s,
                    "text": text,
                    "words": words,
                    "masked_idx": idx,
                    "target_word": target_word,
                    "is_function_word": bool(is_fw),
                })
            picked += 1
        summary_counts[s] = picked

    # structure stratum tally
    fw = sum(1 for r in probe_rows if r["is_function_word"])
    cw = len(probe_rows) - fw
    summary = {
        "probe_targets_total": len(probe_rows),
        "sampled_rows_per_source": summary_counts,
        "function_word_targets": fw,
        "content_word_targets": cw,
        "seed": seed,
        "rows_per_source_requested": rows_per_source,
        "masks_per_row": masks_per_row,
        "pool": str(POOL),
        "tokenizer": str(TOKJSON),
    }
    out = {"rows": probe_rows, "summary": summary}
    (OUT_DIR / "retention_probe.json").write_text(json.dumps(out))
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
