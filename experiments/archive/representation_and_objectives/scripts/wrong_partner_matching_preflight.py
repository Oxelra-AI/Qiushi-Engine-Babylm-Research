#!/usr/bin/env python3
"""research wrong-partner matching preflight for MLM cross-view causal separation.

Constructs a deterministic derangement of compact rewrites across sources while
preserving every source text and every rewrite text exactly once. The primary
constraint is exact rewrite word length, so each source row keeps the same total
word count (source_i + rewrite_j with rewrite_words_j == rewrite_words_i) whenever
possible. Within exact length groups it first rotates inside primary-domain cells;
singletons are relaxed to same rewrite length across domains. The output is a
mapping and quantitative mismatch report, not a training dataset.
"""
from __future__ import annotations

import collections
import hashlib
import json
import pathlib
import random
import re
from typing import Iterable

PAIR_PATH = pathlib.Path("experiments/archive/frontier_consolidation/data/density_core_reinvestment_medium_riskhard/selected_compact_reinvest_pairs.jsonl")
OUT_DIR = pathlib.Path("experiments/archive/representation_and_objectives/data/wrong_partner_matching_preflight")
WORD_RE = re.compile(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)?")


def norm_words(text: str) -> list[str]:
    return [m.group(0).lower() for m in WORD_RE.finditer(text)]


def jaccard(a: Iterable[str], b: Iterable[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def recall(a_source: Iterable[str], b_rewrite: Iterable[str]) -> float:
    sa, rb = set(a_source), set(b_rewrite)
    if not rb:
        return 0.0
    return len(sa & rb) / len(rb)


def rotate(ids: list[int], rng: random.Random) -> dict[int, int]:
    ids = list(ids)
    if len(ids) == 1:
        raise ValueError("cannot derange singleton")
    ids_sorted = sorted(ids)
    shift = rng.randrange(1, len(ids_sorted))
    return {src: ids_sorted[(k + shift) % len(ids_sorted)] for k, src in enumerate(ids_sorted)}


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for line in PAIR_PATH.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        r["primary_domain"] = (r.get("domain_hits") or ["no_domain"])[0]
        r["source_norm"] = norm_words(r["source_text"])
        r["rewrite_norm"] = norm_words(r["rewrite_text"])
        rows.append(r)
    rng = random.Random(219013)

    # First pass: exact (rewrite_words, primary_domain) groups with >=2 members.
    by_len_domain = collections.defaultdict(list)
    for i, r in enumerate(rows):
        by_len_domain[(int(r["rewrite_words"]), r["primary_domain"])].append(i)
    donor = {}
    relaxed = []
    for key in sorted(by_len_domain):
        ids = by_len_domain[key]
        if len(ids) >= 2:
            donor.update(rotate(ids, rng))
        else:
            relaxed.extend(ids)

    # Second pass: exact rewrite length across domains for leftovers. Include only leftovers,
    # so donor reuse does not occur.
    by_len = collections.defaultdict(list)
    for i in relaxed:
        by_len[int(rows[i]["rewrite_words"])].append(i)
    impossible_singletons = []
    for length, ids in sorted(by_len.items()):
        if len(ids) >= 2:
            donor.update(rotate(ids, rng))
        else:
            impossible_singletons.extend(ids)

    # Final repair for rare exact-length singletons: swap donors with nearest nonself row.
    # This preserves one-to-one use but may introduce a small total-word-count delta for
    # those rows. Record exactly if it happens.
    if impossible_singletons:
        all_ids = set(range(len(rows)))
        used_donors = set(donor.values())
        free_donors = sorted(all_ids - used_donors)
        # free_donors should equal impossible_singletons before repair.
        for i in impossible_singletons:
            # choose a target row k already assigned, not itself, whose donor can be swapped
            # to i without self-pairing and with minimal donor rewrite length mismatch for k/i.
            best = None
            for k, d in donor.items():
                if k == i or d == i:
                    continue
                # swap: i gets d; k gets i, requiring k != i and i != k, d != i.
                cost = abs(rows[i]["rewrite_words"] - rows[d]["rewrite_words"]) + abs(rows[k]["rewrite_words"] - rows[i]["rewrite_words"])
                if best is None or cost < best[0]:
                    best = (cost, k, d)
            if best is None:
                raise RuntimeError(f"could not repair singleton {i}")
            _, k, d = best
            donor[i] = d
            donor[k] = i

    if len(donor) != len(rows) or len(set(donor.values())) != len(rows):
        raise RuntimeError(f"not a bijection: assigned {len(donor)} unique donors {len(set(donor.values()))} rows {len(rows)}")
    if any(i == j for i, j in donor.items()):
        raise RuntimeError("self-pair found")

    records = []
    exact_len = 0
    same_domain = 0
    total_word_delta_abs = 0
    own_recalls = []
    wrong_recalls = []
    own_jaccards = []
    wrong_jaccards = []
    wrong_ge_own = 0
    for i, j in sorted(donor.items()):
        src = rows[i]
        dr = rows[j]
        own_r = recall(src["source_norm"], src["rewrite_norm"])
        wrong_r = recall(src["source_norm"], dr["rewrite_norm"])
        own_j = jaccard(src["source_norm"], src["rewrite_norm"])
        wrong_j = jaccard(src["source_norm"], dr["rewrite_norm"])
        own_recalls.append(own_r); wrong_recalls.append(wrong_r)
        own_jaccards.append(own_j); wrong_jaccards.append(wrong_j)
        wrong_ge_own += int(wrong_r >= own_r)
        len_delta = int(dr["rewrite_words"]) - int(src["rewrite_words"])
        total_word_delta_abs += abs(len_delta)
        exact_len += int(len_delta == 0)
        same_domain += int(src["primary_domain"] == dr["primary_domain"])
        records.append({
            "source_pair_id": src["pair_id"],
            "donor_rewrite_pair_id": dr["pair_id"],
            "source_words": int(src["source_words"]),
            "own_rewrite_words": int(src["rewrite_words"]),
            "donor_rewrite_words": int(dr["rewrite_words"]),
            "total_words_wrong": int(src["source_words"]) + int(dr["rewrite_words"]),
            "total_words_own": int(src["source_words"]) + int(src["rewrite_words"]),
            "rewrite_len_delta": len_delta,
            "source_domain": src["primary_domain"],
            "donor_domain": dr["primary_domain"],
            "same_primary_domain": src["primary_domain"] == dr["primary_domain"],
            "own_source_rewrite_recall": own_r,
            "wrong_source_rewrite_recall": wrong_r,
            "own_source_rewrite_jaccard": own_j,
            "wrong_source_rewrite_jaccard": wrong_j,
        })

    def stats(xs):
        xs = sorted(float(x) for x in xs)
        n = len(xs)
        def q(p):
            if n == 0:
                return None
            idx = min(n-1, max(0, int(round(p*(n-1)))))
            return xs[idx]
        return {"mean": sum(xs)/max(n,1), "p10": q(0.1), "p50": q(0.5), "p90": q(0.9), "min": xs[0] if xs else None, "max": xs[-1] if xs else None}

    map_path = OUT_DIR / "wrong_partner_derangement.jsonl"
    with map_path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    h = hashlib.sha256(map_path.read_bytes()).hexdigest()
    summary = {
        "status": "WRONG_PARTNER_MATCHING_PREFLIGHT",
        "pair_path": str(PAIR_PATH),
        "n_pairs": len(rows),
        "mapping_path": str(map_path),
        "mapping_sha256": h,
        "self_pairs": 0,
        "bijection": True,
        "exact_rewrite_length_pairs": exact_len,
        "exact_rewrite_length_frac": exact_len / len(rows),
        "total_abs_rewrite_word_delta": total_word_delta_abs,
        "same_primary_domain_pairs": same_domain,
        "same_primary_domain_frac": same_domain / len(rows),
        "own_rewrite_source_recall": stats(own_recalls),
        "wrong_rewrite_source_recall": stats(wrong_recalls),
        "own_rewrite_source_jaccard": stats(own_jaccards),
        "wrong_rewrite_source_jaccard": stats(wrong_jaccards),
        "wrong_recall_ge_own_count": wrong_ge_own,
        "wrong_recall_ge_own_frac": wrong_ge_own / len(rows),
        "interpretation": "Wrong partner can preserve every source/rewrite and exact row word counts, but lexical overlap with source necessarily collapses. Therefore copied-target visible benefits are not a clean semantic-causality readout; the decisive target stratum is source-absent/noncopied compact words and downstream Supplement/relational-EWoK transitions, with wrong-visible minus wrong-blocked subtracting generic cross-boundary context effects.",
        "random_seed": 219013,
    }
    (OUT_DIR / "wrong_partner_matching_preflight.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    md = (OUT_DIR.parents[4] / 'research/documents/representation_and_objectives/data/wrong_partner_matching_preflight/wrong_partner_matching_preflight.md')
    md.write_text(
        "# research wrong-partner matching preflight\n\n"
        f"Pairs: {len(rows)}; bijective derangement; self-pairs: 0.\n\n"
        f"Exact rewrite-length preservation: {exact_len}/{len(rows)} ({exact_len/len(rows):.3%}); total abs rewrite-word delta {total_word_delta_abs}.\n\n"
        f"Same primary-domain donor: {same_domain}/{len(rows)} ({same_domain/len(rows):.3%}).\n\n"
        f"Own source→rewrite word recall mean {summary['own_rewrite_source_recall']['mean']:.3f}; wrong source→rewrite recall mean {summary['wrong_rewrite_source_recall']['mean']:.3f}.\n\n"
        f"Own Jaccard mean {summary['own_rewrite_source_jaccard']['mean']:.3f}; wrong Jaccard mean {summary['wrong_rewrite_source_jaccard']['mean']:.3f}.\n\n"
        "Interpretation: wrong-visible is feasible as a dose/internal-context control, but copied-token contrasts must be read separately because lexical-copy opportunity is intentionally broken. Source-absent/noncopied target loss and Supplement/relational-EWoK transitions carry the causal mechanism test.\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": summary["status"], "n_pairs": len(rows), "exact_length_frac": summary["exact_rewrite_length_frac"], "same_domain_frac": summary["same_primary_domain_frac"], "own_recall_mean": summary["own_rewrite_source_recall"]["mean"], "wrong_recall_mean": summary["wrong_rewrite_source_recall"]["mean"], "json": str(OUT_DIR / "wrong_partner_matching_preflight.json")}, indent=2))


if __name__ == "__main__":
    main()
