#!/usr/bin/env python3
"""research: corrected consolidated data for MLM cross-view causal separation.

This supersedes research's partner-relative wrong-rewrite labels.  Each rewrite
carries origin labels computed against its own true source before derangement, so
own and wrong arms train the same intrinsic rewrite target strata.  The wrong arm
still pairs source_i with rewrite_pi(i); only the target labels follow the rewrite
identity rather than the receiving source.
"""
from __future__ import annotations

import hashlib
import json
import pathlib
import time
from collections import Counter

FUNCTION_WORDS = {
    "a", "an", "the", "this", "that", "these", "those", "is", "am", "are", "was", "were",
    "be", "been", "being", "have", "has", "had", "having", "do", "does", "did", "will",
    "would", "shall", "should", "can", "could", "may", "might", "must", "not", "no",
    "nor", "and", "but", "or", "so", "yet", "for", "in", "on", "at", "to", "from", "by",
    "with", "of", "about", "as", "if", "than", "because", "while", "when", "where",
    "how", "who", "whom", "whose", "which", "what", "he", "she", "it", "they", "we",
    "you", "i", "me", "him", "her", "us", "them", "my", "your", "his", "its", "our",
    "their", "some", "any", "many", "much", "few", "little", "all", "both", "each",
    "every", "there", "here", "up", "out", "just", "also", "very", "too", "then",
    "now", "only", "still", "even", "more", "most", "s", "t", "re", "ve", "ll", "d",
    "n", "m", "into", "over", "under", "between", "through", "during", "before",
    "after", "above", "below", "other", "another", "such", "own",
}


def norm_word(w: str) -> str:
    return w.lower().strip(".,!?;:\"'()-[]{}…–—/\\")


def classify_word(w: str) -> str:
    low = norm_word(w)
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
    """Classify each rewrite whitespace word relative to its true source."""
    src_set = {norm_word(w) for w in source_words_list if norm_word(w)}
    origins: list[str] = []
    for w in rewrite_words_list:
        low = norm_word(w)
        origin = "copied" if low and low in src_set else "absent"
        origins.append(f"{origin}_{classify_word(w)}")
    return origins


def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    t0 = time.time()
    pair_path = pathlib.Path(
        "experiments/archive/frontier_consolidation/data/density_core_reinvestment_medium_riskhard"
        "selected_compact_reinvest_pairs.jsonl"
    )
    wrong_path = pathlib.Path(
        "experiments/archive/representation_and_objectives/data/wrong_partner_matching_preflight"
        "wrong_partner_derangement.jsonl"
    )
    filler_path = pathlib.Path(
        "experiments/archive/frontier_consolidation/data/causal_transfer_scaffold/filler_rows.jsonl"
    )
    out_dir = pathlib.Path("experiments/archive/representation_and_objectives/data/crossview_data_v2")
    out_dir.mkdir(parents=True, exist_ok=True)
    target_words = 10_000_000

    print("Loading compact pairs...", flush=True)
    pairs: dict[str, dict] = {}
    with pair_path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                obj = json.loads(line)
                pairs[obj["pair_id"]] = obj
    pair_ids = sorted(pairs)
    print(f"  {len(pair_ids)} pairs", flush=True)

    print("Precomputing true rewrite-origin labels...", flush=True)
    true_origins: dict[str, list[str]] = {}
    origin_counts = Counter()
    for pid in pair_ids:
        p = pairs[pid]
        origins = annotate_origins(p["source_text"].split(), p["rewrite_text"].split())
        assert len(origins) == len(p["rewrite_text"].split())
        true_origins[pid] = origins
        origin_counts.update(origins)

    print("Loading derangement...", flush=True)
    wrong_map: dict[str, str] = {}
    with wrong_path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                obj = json.loads(line)
                wrong_map[obj["source_pair_id"]] = obj["donor_rewrite_pair_id"]
    assert set(wrong_map) == set(pair_ids), "Derangement source ids do not match compact pairs"
    assert set(wrong_map.values()) == set(pair_ids), "Derangement does not use each rewrite exactly once"
    assert all(wrong_map[pid] != pid for pid in pair_ids), "Self-pair found in wrong map"
    print("  bijective derangement with zero self-pairs", flush=True)

    pair_rows: list[dict] = []
    total_pair_words = 0
    total_wrong_pair_words = 0
    exact_rewrite_len = 0
    len_abs_delta = 0
    for pid in pair_ids:
        p = pairs[pid]
        donor_pid = wrong_map[pid]
        donor = pairs[donor_pid]
        sw = len(p["source_text"].split())
        own_rw_words = len(p["rewrite_text"].split())
        wrong_rw_words = len(donor["rewrite_text"].split())
        if own_rw_words == wrong_rw_words:
            exact_rewrite_len += 1
        len_abs_delta += abs(own_rw_words - wrong_rw_words)
        row = {
            "type": "pair",
            "source_pair_id": pid,
            "pair_id": pid,
            "source_text": p["source_text"],
            "source_words": sw,
            "own_rewrite_pair_id": pid,
            "own_rewrite": p["rewrite_text"],
            "own_rewrite_words": own_rw_words,
            "own_rewrite_origins_true": true_origins[pid],
            "wrong_rewrite_pair_id": donor_pid,
            "wrong_rewrite": donor["rewrite_text"],
            "wrong_rewrite_words": wrong_rw_words,
            "wrong_rewrite_origins_true": true_origins[donor_pid],
            "own_total_words": sw + own_rw_words,
            "wrong_total_words": sw + wrong_rw_words,
        }
        pair_rows.append(row)
        total_pair_words += row["own_total_words"]
        total_wrong_pair_words += row["wrong_total_words"]

    assert total_pair_words == total_wrong_pair_words, (total_pair_words, total_wrong_pair_words)
    print(f"Pair rows: {len(pair_rows)}; pair words/epoch {total_pair_words}", flush=True)

    filler_needed = target_words - total_pair_words
    print(f"Loading filler: need {filler_needed} words", flush=True)
    filler_rows: list[dict] = []
    filler_words = 0
    with filler_path.open(encoding="utf-8") as f:
        for k, line in enumerate(f):
            if not line.strip():
                continue
            obj = json.loads(line)
            text = str(obj["text"])
            words = text.split()
            n = len(words)
            if filler_words + n > filler_needed:
                take = filler_needed - filler_words
                if take > 0:
                    filler_rows.append({
                        "type": "filler",
                        "filler_id": f"filler:{k}:prefix:{take}",
                        "text": " ".join(words[:take]),
                        "words": take,
                    })
                    filler_words += take
                break
            filler_rows.append({
                "type": "filler",
                "filler_id": f"filler:{k}",
                "text": text,
                "words": n,
            })
            filler_words += n
    assert total_pair_words + filler_words == target_words
    print(f"Filler rows: {len(filler_rows)}; filler words {filler_words}", flush=True)

    stream = pair_rows + filler_rows
    out_path = out_dir / "crossview_consolidated_v2.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        for row in stream:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    copied = sum(v for k, v in origin_counts.items() if k.startswith("copied"))
    absent = sum(v for k, v in origin_counts.items() if k.startswith("absent"))
    manifest = {
        "status": "CROSSVIEW_DATA_V2_BUILT",
        "supersedes": "experiments/archive/representation_and_objectives/data/crossview_data/crossview_consolidated.jsonl",
        "scientific_repair": (
            "Rewrite-origin labels are intrinsic to each rewrite's own true source and are carried through "
            "the wrong-partner derangement. Thus own and wrong arms expose the same rewrite target strata; "
            "wrong-visible no longer reclassifies donor rewrite words relative to the receiving source."
        ),
        "pair_path": str(pair_path),
        "pair_sha256": sha256_file(pair_path),
        "wrong_path": str(wrong_path),
        "wrong_sha256": sha256_file(wrong_path),
        "filler_path": str(filler_path),
        "n_pairs": len(pair_rows),
        "n_filler": len(filler_rows),
        "n_total": len(stream),
        "total_pair_words": total_pair_words,
        "total_wrong_pair_words": total_wrong_pair_words,
        "total_filler_words": filler_words,
        "epoch_words_each_arm": target_words,
        "exact_rewrite_length_matches": exact_rewrite_len,
        "exact_rewrite_length_frac": exact_rewrite_len / len(pair_rows),
        "rewrite_length_total_abs_delta": len_abs_delta,
        "true_rewrite_origin_counts": dict(sorted(origin_counts.items(), key=lambda x: (-x[1], x[0]))),
        "true_rewrite_copied_frac": copied / (copied + absent),
        "true_rewrite_absent_content_count": origin_counts.get("absent_content", 0),
        "out_path": str(out_path),
        "out_sha256": sha256_file(out_path),
        "elapsed_sec": round(time.time() - t0, 2),
    }
    (out_dir / "crossview_data_v2_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "status": manifest["status"],
        "n_pairs": manifest["n_pairs"],
        "n_filler": manifest["n_filler"],
        "epoch_words_each_arm": manifest["epoch_words_each_arm"],
        "true_rewrite_copied_frac": round(manifest["true_rewrite_copied_frac"], 4),
        "true_absent_content": manifest["true_rewrite_absent_content_count"],
        "out_sha256": manifest["out_sha256"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
