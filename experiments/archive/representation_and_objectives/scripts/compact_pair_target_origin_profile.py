#!/usr/bin/env python3
"""research compact pair target-origin profile.

Profiles the actual 12,155 source--compact pairs that would underlie an MLM
cross-view causal-separation study. The profile is word-level (not a tokenizer
replacement): rewrite words are classified as copied from the source or
source-absent; copied words are split by source position and multiplicity. This
sets the expected target strata for trainer-side loss readouts and masking-dose
accounting.
"""
from __future__ import annotations

import collections
import hashlib
import json
import pathlib
import re

PAIR_PATH = pathlib.Path("experiments/archive/frontier_consolidation/data/density_core_reinvestment_medium_riskhard/selected_compact_reinvest_pairs.jsonl")
OUT_DIR = pathlib.Path("experiments/archive/representation_and_objectives/data/compact_pair_target_origin_profile")
WORD_RE = re.compile(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)?")
FUNCTION = set("""
a an and are as at be been being but by can could did do does for from had has have he her hers him his i if in into is it its me my no not of on or our ours she so than that the their theirs them there they this to was we were what when where which who whom whose will with would you your yours
""".split())


def norm_words(text: str):
    return [m.group(0).lower() for m in WORD_RE.finditer(text)]


def word_class(w: str) -> str:
    if any(c.isdigit() for c in w):
        return "number"
    if w in FUNCTION:
        return "function"
    if len(w) <= 2:
        return "short_other"
    return "content"


def source_position_zone(pos: int, n: int) -> str:
    if n <= 0:
        return "absent"
    frac = (pos + 0.5) / n
    if frac < 1/3:
        return "head_third"
    if frac < 2/3:
        return "middle_third"
    return "tail_third"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for line in PAIR_PATH.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        sw = norm_words(r["source_text"])
        rw = norm_words(r["rewrite_text"])
        pos = collections.defaultdict(list)
        for i, w in enumerate(sw):
            pos[w].append(i)
        word_recs = []
        for k, w in enumerate(rw):
            p = pos.get(w, [])
            copied = bool(p)
            if copied:
                first = p[0]
                mult = len(p)
                zone = source_position_zone(first, len(sw))
            else:
                first = None
                mult = 0
                zone = "absent"
            word_recs.append({
                "rewrite_index": k,
                "word": w,
                "class": word_class(w),
                "copied_from_source": copied,
                "source_match_count": mult,
                "first_source_index": first,
                "source_position_zone": zone,
            })
        dom = (r.get("domain_hits") or ["no_domain"])[0]
        rows.append({
            "pair_id": r["pair_id"],
            "source_words_field": int(r["source_words"]),
            "rewrite_words_field": int(r["rewrite_words"]),
            "source_norm_words": len(sw),
            "rewrite_norm_words": len(rw),
            "primary_domain": dom,
            "word_records": word_recs,
        })

    counters = collections.Counter()
    domain = collections.defaultdict(collections.Counter)
    class_zone = collections.Counter()
    pair_flags = collections.Counter()
    for r in rows:
        d = r["primary_domain"]
        has_absent_content = False
        has_tail_copied_content = False
        for wr in r["word_records"]:
            cls = wr["class"]
            zone = wr["source_position_zone"]
            copied = wr["copied_from_source"]
            counters["rewrite_words_profiled"] += 1
            counters[f"class::{cls}"] += 1
            domain[d]["rewrite_words_profiled"] += 1
            domain[d][f"class::{cls}"] += 1
            if copied:
                counters["copied_words"] += 1
                domain[d]["copied_words"] += 1
                if wr["source_match_count"] == 1:
                    counters["copied_unique_words"] += 1
                    domain[d]["copied_unique_words"] += 1
                else:
                    counters["copied_multi_words"] += 1
                    domain[d]["copied_multi_words"] += 1
            else:
                counters["source_absent_words"] += 1
                domain[d]["source_absent_words"] += 1
                if cls == "content":
                    counters["source_absent_content_words"] += 1
                    domain[d]["source_absent_content_words"] += 1
                    has_absent_content = True
            counters[f"zone::{zone}"] += 1
            domain[d][f"zone::{zone}"] += 1
            counters[f"class_zone::{cls}::{zone}"] += 1
            class_zone[(cls, zone)] += 1
            if copied and cls == "content" and zone == "tail_third":
                has_tail_copied_content = True
        pair_flags["pairs"] += 1
        pair_flags["pairs_with_absent_content"] += int(has_absent_content)
        pair_flags["pairs_with_tail_copied_content"] += int(has_tail_copied_content)

    total = counters["rewrite_words_profiled"]
    summary = {
        "status": "COMPACT_PAIR_TARGET_ORIGIN_PROFILE",
        "pair_path": str(PAIR_PATH),
        "pair_path_sha256": hashlib.sha256(PAIR_PATH.read_bytes()).hexdigest(),
        "n_pairs": len(rows),
        "rewrite_words_profiled": total,
        "counter": dict(counters),
        "fractions": {
            "copied_words": counters["copied_words"] / total,
            "source_absent_words": counters["source_absent_words"] / total,
            "source_absent_content_words": counters["source_absent_content_words"] / total,
            "copied_unique_words": counters["copied_unique_words"] / total,
            "copied_tail_third_words": counters["zone::tail_third"] / total,
            "content_words": counters["class::content"] / total,
        },
        "pair_flags": dict(pair_flags),
        "pair_flag_fractions": {
            k: v / max(1, pair_flags["pairs"]) for k, v in pair_flags.items() if k != "pairs"
        },
        "domain_summary": {
            d: {**dict(c), "copied_frac": c["copied_words"] / max(1, c["rewrite_words_profiled"]), "source_absent_frac": c["source_absent_words"] / max(1, c["rewrite_words_profiled"]), "source_absent_content_frac": c["source_absent_content_words"] / max(1, c["rewrite_words_profiled"])}
            for d, c in sorted(domain.items())
        },
        "training_readout_use": "Use these strata for MLM loss accounting: copied_unique, copied_multi, copied_tail, source_absent_content, source_absent_function/other. Source-absent content is the decisive compact-only target mass; copied-target loss is a retrieval/null channel rather than the presumed mechanism.",
    }
    (OUT_DIR / "compact_pair_target_origin_profile.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    # Write a compact row-level JSONL without text for later trainer annotations.
    rec_path = OUT_DIR / "compact_pair_target_origin_records.jsonl"
    with rec_path.open("w", encoding="utf-8") as f:
        for r in rows:
            slim = {k: r[k] for k in ["pair_id", "source_words_field", "rewrite_words_field", "source_norm_words", "rewrite_norm_words", "primary_domain"]}
            slim["word_records"] = r["word_records"]
            f.write(json.dumps(slim, ensure_ascii=False) + "\n")
    md = (OUT_DIR.parents[4] / 'research/documents/representation_and_objectives/data/compact_pair_target_origin_profile/compact_pair_target_origin_profile.md')
    md.write_text(
        "# research compact pair target-origin profile\n\n"
        f"Pairs: {len(rows)}; rewrite words profiled: {total}.\n\n"
        f"Copied rewrite words: {counters['copied_words']} ({summary['fractions']['copied_words']:.2%}); source-absent rewrite words: {counters['source_absent_words']} ({summary['fractions']['source_absent_words']:.2%}); source-absent content words: {counters['source_absent_content_words']} ({summary['fractions']['source_absent_content_words']:.2%}).\n\n"
        f"Pairs with at least one source-absent content word: {pair_flags['pairs_with_absent_content']} ({summary['pair_flag_fractions']['pairs_with_absent_content']:.2%}).\n\n"
        f"Unique copied rewrite words: {counters['copied_unique_words']} ({summary['fractions']['copied_unique_words']:.2%}); tail-third rewrite words: {counters['zone::tail_third']} ({summary['fractions']['copied_tail_third_words']:.2%}).\n\n"
        "These counts are the population for target-stratified MLM loss readouts in the source-partner visibility intervention.\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": summary["status"], "n_pairs": len(rows), "rewrite_words": total, "source_absent_frac": summary["fractions"]["source_absent_words"], "source_absent_content_frac": summary["fractions"]["source_absent_content_words"], "pairs_with_absent_content_frac": summary["pair_flag_fractions"]["pairs_with_absent_content"], "json": str(OUT_DIR / "compact_pair_target_origin_profile.json")}, indent=2))


if __name__ == "__main__":
    main()
