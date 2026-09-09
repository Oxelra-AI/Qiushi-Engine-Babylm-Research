#!/usr/bin/env python3
"""research: Build consolidated cross-view data for the DeBERTa MLM causal separation study.

Creates a single JSONL with pair rows (source, own rewrite, wrong rewrite, word origins)
and filler rows, totaling exactly 10M words per epoch. The trainer selects the active
partner (own/wrong) and visibility (visible/blocked) at training time.
"""
from __future__ import annotations
import json, pathlib, hashlib, sys, time

FUNCTION_WORDS = {
    "a","an","the","this","that","these","those","is","am","are","was","were",
    "be","been","being","have","has","had","having","do","does","did","will",
    "would","shall","should","can","could","may","might","must","not","no",
    "nor","and","but","or","so","yet","for","in","on","at","to","from","by",
    "with","of","about","as","if","than","because","while","when","where",
    "how","who","whom","whose","which","what","he","she","it","they","we",
    "you","i","me","him","her","us","them","my","your","his","its","our",
    "their","some","any","many","much","few","little","all","both","each",
    "every","there","here","up","out","just","also","very","too","then",
    "now","only","still","even","more","most","s","t","re","ve","ll","d",
    "n","m","into","over","under","between","through","during","before",
    "after","above","below","other","another","such","own",
}

def classify_word(w: str) -> str:
    low = w.lower().strip(".,!?;:\"'()-[]{}…–—/\\")
    if not low:
        return "function"
    try:
        float(low.replace(",", ""))
        return "number"
    except ValueError:
        pass
    if low in FUNCTION_WORDS:
        return "function"
    return "content"

def annotate_origins(source_words_list: list[str], rewrite_words_list: list[str]) -> list[str]:
    """Classify each rewrite word as copied_X or absent_X relative to paired source."""
    src_set = {w.lower().strip(".,!?;:\"'()-[]{}…–—/\\") for w in source_words_list}
    origins = []
    for w in rewrite_words_list:
        wclass = classify_word(w)
        low = w.lower().strip(".,!?;:\"'()-[]{}…–—/\\")
        origin = "copied" if low in src_set else "absent"
        origins.append(f"{origin}_{wclass}")
    return origins

def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def main():
    t0 = time.time()
    pair_path = pathlib.Path("experiments/archive/frontier_consolidation/data"
                             "density_core_reinvestment_medium_riskhard/"
                             "selected_compact_reinvest_pairs.jsonl")
    wrong_path = pathlib.Path("experiments/archive/representation_and_objectives/data"
                              "wrong_partner_matching_preflight/"
                              "wrong_partner_derangement.jsonl")
    filler_path = pathlib.Path("experiments/archive/frontier_consolidation/data"
                               "causal_transfer_scaffold/filler_rows.jsonl")
    out_dir = pathlib.Path("experiments/archive/representation_and_objectives/data/crossview_data")
    out_dir.mkdir(parents=True, exist_ok=True)
    TARGET = 10_000_000

    # --- Load pairs ---
    print("Loading pairs...", flush=True)
    pairs = {}
    with pair_path.open() as f:
        for line in f:
            if not line.strip(): continue
            obj = json.loads(line)
            pairs[obj["pair_id"]] = obj
    pair_ids = sorted(pairs.keys())
    print(f"  {len(pairs)} pairs", flush=True)

    # --- Load wrong-partner mapping ---
    print("Loading derangement...", flush=True)
    wrong_map = {}
    with wrong_path.open() as f:
        for line in f:
            if not line.strip(): continue
            obj = json.loads(line)
            wrong_map[obj["source_pair_id"]] = obj["donor_rewrite_pair_id"]
    assert len(wrong_map) == len(pairs), f"{len(wrong_map)} != {len(pairs)}"
    print(f"  {len(wrong_map)} mappings", flush=True)

    # --- Build pair rows ---
    print("Building pair rows...", flush=True)
    pair_rows = []
    total_pair_words = 0
    own_origin_counts, wrong_origin_counts = {}, {}

    for pid in pair_ids:
        p = pairs[pid]
        src_text = p["source_text"]
        own_rw = p["rewrite_text"]
        wrong_pid = wrong_map[pid]
        wrong_rw = pairs[wrong_pid]["rewrite_text"]

        src_wl = src_text.split()
        own_wl = own_rw.split()
        wrong_wl = wrong_rw.split()

        own_origins = annotate_origins(src_wl, own_wl)
        wrong_origins = annotate_origins(src_wl, wrong_wl)

        for o in own_origins: own_origin_counts[o] = own_origin_counts.get(o, 0) + 1
        for o in wrong_origins: wrong_origin_counts[o] = wrong_origin_counts.get(o, 0) + 1

        sw, ow, ww = len(src_wl), len(own_wl), len(wrong_wl)
        pair_rows.append({
            "type": "pair", "pair_id": pid,
            "source_text": src_text, "own_rewrite": own_rw, "wrong_rewrite": wrong_rw,
            "source_words": sw, "own_rewrite_words": ow, "wrong_rewrite_words": ww,
            "own_total_words": sw + ow, "wrong_total_words": sw + ww,
            "own_rewrite_origins": own_origins, "wrong_rewrite_origins": wrong_origins,
        })
        total_pair_words += sw + ow

    print(f"  {len(pair_rows)} pair rows, {total_pair_words} pair words", flush=True)

    # --- Load filler (exact match: 9,576,489 words) ---
    print("Loading filler...", flush=True)
    filler_rows = []
    filler_words = 0
    filler_needed = TARGET - total_pair_words

    with filler_path.open() as f:
        for line in f:
            if not line.strip(): continue
            obj = json.loads(line)
            text = str(obj["text"])
            words = len(text.split())
            if filler_words + words > filler_needed:
                take = filler_needed - filler_words
                if take > 0:
                    filler_rows.append({"type": "filler",
                                        "text": " ".join(text.split()[:take]), "words": take})
                    filler_words += take
                break
            filler_rows.append({"type": "filler", "text": text, "words": words})
            filler_words += words

    actual = total_pair_words + filler_words
    print(f"  {len(filler_rows)} filler rows, {filler_words} filler words", flush=True)
    print(f"  Total epoch words: {actual} (target {TARGET})", flush=True)
    assert actual == TARGET, f"Word mismatch: {actual} != {TARGET}"

    # --- Write consolidated JSONL ---
    stream = pair_rows + filler_rows
    out_path = out_dir / "crossview_consolidated.jsonl"
    print(f"Writing {len(stream)} rows...", flush=True)
    with out_path.open("w", encoding="utf-8") as f:
        for row in stream:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # --- Origin summary ---
    own_copied = sum(v for k, v in own_origin_counts.items() if k.startswith("copied"))
    own_absent = sum(v for k, v in own_origin_counts.items() if k.startswith("absent"))
    wrong_copied = sum(v for k, v in wrong_origin_counts.items() if k.startswith("copied"))
    wrong_absent = sum(v for k, v in wrong_origin_counts.items() if k.startswith("absent"))
    own_total = own_copied + own_absent
    wrong_total = wrong_copied + wrong_absent

    manifest = {
        "status": "CROSSVIEW_DATA_BUILT",
        "pair_path": str(pair_path), "pair_sha256": sha256_file(pair_path),
        "wrong_path": str(wrong_path), "wrong_sha256": sha256_file(wrong_path),
        "filler_path": str(filler_path),
        "n_pairs": len(pair_rows), "n_filler": len(filler_rows), "n_total": len(stream),
        "total_pair_words": total_pair_words, "total_filler_words": filler_words,
        "epoch_words": actual,
        "own_origin_counts": dict(sorted(own_origin_counts.items(), key=lambda x: -x[1])),
        "own_copied_frac": own_copied / own_total if own_total else 0,
        "own_absent_content_count": own_origin_counts.get("absent_content", 0),
        "wrong_origin_counts": dict(sorted(wrong_origin_counts.items(), key=lambda x: -x[1])),
        "wrong_copied_frac": wrong_copied / wrong_total if wrong_total else 0,
        "wrong_absent_content_count": wrong_origin_counts.get("absent_content", 0),
        "out_path": str(out_path), "out_sha256": sha256_file(out_path),
        "elapsed_sec": round(time.time() - t0, 1),
    }
    (out_dir / "crossview_data_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps({
        "status": manifest["status"],
        "n_pairs": manifest["n_pairs"], "n_filler": manifest["n_filler"],
        "epoch_words": manifest["epoch_words"],
        "own_copied_frac": round(manifest["own_copied_frac"], 4),
        "own_absent_content": manifest["own_absent_content_count"],
        "wrong_copied_frac": round(manifest["wrong_copied_frac"], 4),
        "elapsed": manifest["elapsed_sec"],
    }, indent=2), flush=True)

if __name__ == "__main__":
    main()
