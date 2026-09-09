#!/usr/bin/env python3
"""research: source novelty x semantic content target-set construction profile.

CPU-only preparation for a possible post-endpoint separation.  It constructs matched
whole-word deletion selections crossing source-novelty (source-absent vs copied) with
semantic class (relational/event, ordinary nonrelational nonentity, name/number), using
exact BPE-piece matching and the same packed-pool annotation geometry.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import pathlib
import random
import statistics
import sys
import time
from typing import Any

USER_ROOT = pathlib.Path(".").resolve()
SCRIPT_DIR = USER_ROOT / "experiments/archive/representation_and_objectives/scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from annotate_packed_pool import (  # noqa: E402
    CAT_RW_ABS_CONTENT,
    CAT_RW_COPIED,
    norm_word,
    word_class,
)
from source_absent_lexical_profile import flags as lexical_flags  # noqa: E402

POOL = pathlib.Path("experiments/archive/frontier_consolidation/data/density_cleanqwen_overlay_medium_riskhard/cleanqwen_fineweb_compact_view_reinvest_10M.jsonl")
ANNOTATION = pathlib.Path("experiments/archive/representation_and_objectives/data/packed_pool_annotation/packed_pool_annotations.jsonl")
TOKENIZER = pathlib.Path("experiments/archive/representation_and_objectives/training/runs/gc_compact_repeat_reinvest_16k_seed43022_r2/hf_model/chck_1M")
OUT_DIR = pathlib.Path("experiments/archive/representation_and_objectives/data/novelty_semantics_selection_profile")


def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def qstats(xs: list[float]) -> dict[str, Any]:
    if not xs:
        return {"n": 0}
    s = sorted(float(x) for x in xs)
    n = len(s)
    def q(p: float) -> float:
        return s[min(n - 1, max(0, int(round(p * (n - 1)))))]
    return {
        "n": n,
        "mean": round(statistics.mean(s), 6),
        "median": round(statistics.median(s), 6),
        "p05": round(q(0.05), 6),
        "p25": round(q(0.25), 6),
        "p75": round(q(0.75), 6),
        "p95": round(q(0.95), 6),
        "min": round(s[0], 6),
        "max": round(s[-1], 6),
    }


def semantic_class(surface: str) -> str | None:
    nw = norm_word(surface)
    fs = set(lexical_flags(surface, nw))
    rel_event = bool({"relational_or_abstracting_word", "event_or_state_word"} & fs)
    capnum = bool({"capitalized", "number_or_year"} & fs)
    generic = "generic_entity_word" in fs
    if rel_event:
        return "rel_event"
    if capnum:
        return "capitalized_or_number"
    if not generic:
        return "ordinary_nonrel_nonentity"
    return "generic_entity_nonrel"


def load_annotations(path: pathlib.Path) -> list[dict[str, Any]]:
    out = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line))
    return out


def load_pool_rows(path: pathlib.Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue
            obj = json.loads(line)
            text = str(obj["text"])
            rows.append({"row_index": i, "example_id": int(obj.get("example_id", i)), "text": text, "words": text.split()})
    return rows


def bpe_per_word(tok, text: str, words: list[str]) -> collections.Counter[int]:
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    offsets = enc["offset_mapping"]
    char_to_word = {}
    pos = 0
    for wi, w in enumerate(words):
        start = text.index(w, pos)
        for ci in range(start, start + len(w)):
            char_to_word[ci] = wi
        pos = start + len(w)
    bpe = collections.Counter()
    for cs, ce in offsets:
        if cs == ce:
            continue
        wi = None
        for ci in range(cs, ce):
            wi = char_to_word.get(ci)
            if wi is not None:
                break
        if wi is not None:
            bpe[wi] += 1
    return bpe


def collect_groups(pool_rows: list[dict[str, Any]], annotations: list[dict[str, Any]], tok) -> list[dict[str, Any]]:
    pool_freq = collections.Counter()
    for r in pool_rows:
        pool_freq.update(norm_word(w) for w in r["words"])
    by_row = {int(a["row_index"]): a for a in annotations}
    groups = []
    for row in pool_rows:
        a = by_row.get(int(row["row_index"]))
        if not a:
            continue
        cats = a["word_categories"]
        if not any(c in (CAT_RW_ABS_CONTENT, CAT_RW_COPIED) for c in cats):
            continue
        bpe = bpe_per_word(tok, row["text"], row["words"])
        n_words = len(row["words"])
        for wi, cat in enumerate(cats):
            if wi >= n_words or cat not in (CAT_RW_ABS_CONTENT, CAT_RW_COPIED):
                continue
            surf = row["words"][wi]
            wc = word_class(surf)
            if wc not in {"content", "number"}:
                continue
            pieces = int(bpe.get(wi, 0))
            if pieces <= 0:
                continue
            nw = norm_word(surf)
            sem = semantic_class(surf)
            groups.append({
                "row_index": int(row["row_index"]),
                "example_id": int(row["example_id"]),
                "word_index": wi,
                "bpe_pieces": pieces,
                "norm_word": nw,
                "surface": surf,
                "word_class": wc,
                "source_category": "source_absent" if cat == CAT_RW_ABS_CONTENT else "copied",
                "semantic_class": sem,
                "flags": lexical_flags(surf, nw),
                "pool_support": int(pool_freq[nw]),
                "support_log": round(math.log1p(pool_freq[nw]), 6),
                "rel_word_pos": round(wi / max(1, n_words - 1), 6),
                "n_words_row": n_words,
            })
    return groups


def summarize_group(gs: list[dict[str, Any]]) -> dict[str, Any]:
    types = collections.Counter(g["norm_word"] for g in gs)
    flags = collections.Counter(f for g in gs for f in g["flags"])
    pieces = [int(g["bpe_pieces"]) for g in gs]
    support = [float(g["support_log"]) for g in gs]
    pos = [float(g["rel_word_pos"]) for g in gs]
    return {
        "n_groups": len(gs),
        "total_bpe_pieces": int(sum(pieces)),
        "n_types": len(types),
        "top_norms": [{"norm": w, "count": int(c)} for w, c in types.most_common(20)],
        "bpe_pieces": qstats(pieces),
        "support_log": qstats(support),
        "rel_word_pos": qstats(pos),
        "flag_fraction": {k: round(v / max(1, len(gs)), 6) for k, v in sorted(flags.items())},
    }


def match_groups(target: list[dict[str, Any]], candidates: list[dict[str, Any]], seed: int, max_targets: int | None = None) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rng = random.Random(seed)
    cand_by_bpe: dict[int, list[dict[str, Any]]] = collections.defaultdict(list)
    for c in candidates:
        cand_by_bpe[int(c["bpe_pieces"])].append(c)
    for bucket in cand_by_bpe.values():
        rng.shuffle(bucket)
        bucket.sort(key=lambda g: (g["support_log"], g["rel_word_pos"], rng.random()))
    targ = list(target)
    rng.shuffle(targ)
    targ.sort(key=lambda g: (int(g["bpe_pieces"]), g["support_log"], g["rel_word_pos"], rng.random()))
    if max_targets is not None:
        targ = targ[:max_targets]
    matched_t = []
    matched_c = []
    for t in targ:
        bucket = cand_by_bpe.get(int(t["bpe_pieces"]), [])
        if not bucket:
            continue
        # Minimize a simple feature distance among a small same-BPE bucket sample.
        best_i = min(range(len(bucket)), key=lambda i: abs(bucket[i]["support_log"] - t["support_log"]) + 0.5 * abs(bucket[i]["rel_word_pos"] - t["rel_word_pos"]))
        c = bucket.pop(best_i)
        matched_t.append(t)
        matched_c.append(c)
    return matched_t, matched_c


def match_summary(name: str, t: list[dict[str, Any]], c: list[dict[str, Any]], target_total: int, candidate_total: int) -> dict[str, Any]:
    def diffs(key: str) -> list[float]:
        return [float(cc[key]) - float(tt[key]) for tt, cc in zip(t, c)]
    return {
        "name": name,
        "target_total_groups": target_total,
        "candidate_total_groups": candidate_total,
        "matched_pairs": len(t),
        "fraction_target_matched": round(len(t) / max(1, target_total), 6),
        "target_total_bpe_pieces_matched": int(sum(g["bpe_pieces"] for g in t)),
        "candidate_total_bpe_pieces_matched": int(sum(g["bpe_pieces"] for g in c)),
        "exact_total_bpe_match": int(sum(g["bpe_pieces"] for g in t)) == int(sum(g["bpe_pieces"] for g in c)),
        "support_log_candidate_minus_target": qstats(diffs("support_log")),
        "rel_word_pos_candidate_minus_target": qstats(diffs("rel_word_pos")),
        "target_summary": summarize_group(t),
        "candidate_summary": summarize_group(c),
    }


def write_selection(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for g in rows:
            f.write(json.dumps(g, ensure_ascii=False) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default=str(POOL))
    ap.add_argument("--annotation", default=str(ANNOTATION))
    ap.add_argument("--tokenizer", default=str(TOKENIZER))
    ap.add_argument("--output_dir", default=str(OUT_DIR))
    args = ap.parse_args()
    t0 = time.time()
    out = pathlib.Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
    pool_rows = load_pool_rows(pathlib.Path(args.pool))
    anns = load_annotations(pathlib.Path(args.annotation))
    groups = collect_groups(pool_rows, anns, tok)

    by = collections.defaultdict(list)
    for g in groups:
        by[(g["source_category"], g["semantic_class"])].append(g)
    group_summaries = {f"{k[0]}|{k[1]}": summarize_group(v) for k, v in sorted(by.items())}

    specs = [
        ("novel_rel_vs_copied_rel", ("source_absent", "rel_event"), ("copied", "rel_event")),
        ("novel_ord_vs_copied_ord", ("source_absent", "ordinary_nonrel_nonentity"), ("copied", "ordinary_nonrel_nonentity")),
        ("novel_capnum_vs_copied_capnum", ("source_absent", "capitalized_or_number"), ("copied", "capitalized_or_number")),
        ("novel_rel_vs_novel_ord", ("source_absent", "rel_event"), ("source_absent", "ordinary_nonrel_nonentity")),
        ("copied_rel_vs_copied_ord", ("copied", "rel_event"), ("copied", "ordinary_nonrel_nonentity")),
    ]
    matches = {}
    for i, (name, tk, ck) in enumerate(specs):
        target = by.get(tk, [])
        cand = by.get(ck, [])
        mt, mc = match_groups(target, cand, seed=234000 + i)
        matches[name] = match_summary(name, mt, mc, len(target), len(cand))
        write_selection(out / f"{name}.target_selection.jsonl", mt)
        write_selection(out / f"{name}.matched_selection.jsonl", mc)

    payload = {
        "status": "NOVELTY_SEMANTICS_SELECTION_PROFILE",
        "meaning": "Matched target-set feasibility for separating source novelty from semantic content after the pending 100M endpoint/A02 transfer result. CPU-only; no model endpoint read.",
        "inputs": {
            "pool": args.pool,
            "pool_sha256": sha256_file(pathlib.Path(args.pool)),
            "annotation": args.annotation,
            "annotation_sha256": sha256_file(pathlib.Path(args.annotation)),
            "tokenizer": args.tokenizer,
        },
        "n_groups_total": len(groups),
        "group_summaries": group_summaries,
        "matches": matches,
        "future_readout_logic": {
            "source_novelty_holding_rel_event_semantics": "Compare deletion of novel_rel against matched copied_rel.",
            "semantic_content_holding_source_novelty": "Compare deletion of novel_rel against matched novel_ord.",
            "harmful_or_noisy_novel_name_number_check": "Compare deletion of novel_capnum against matched copied_capnum, because current losses show the sign can reverse for names/dates.",
            "endpoint_use": "Only consider 100M follow-up if pending packed endpoint and A02 transfer show that the local channel reaches task competence; otherwise preserve as local mechanism evidence.",
        },
        "elapsed_sec": round(time.time() - t0, 1),
    }
    (out / "novelty_semantics_selection_profile.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    md = [
        "# research novelty × semantic target-set profile",
        "",
        "Matched selections are written as JSONL target/matched files under this directory. They are not training results.",
        "",
        "## Group counts",
        "",
    ]
    for k, s in sorted(group_summaries.items()):
        md.append(f"- {k}: groups={s['n_groups']}, pieces={s['total_bpe_pieces']}, types={s['n_types']}")
    md += ["", "## Pairwise match summaries", ""]
    for name, s in matches.items():
        md.append(f"- {name}: matched {s['matched_pairs']}/{s['target_total_groups']} target groups; pieces {s['target_total_bpe_pieces_matched']} vs {s['candidate_total_bpe_pieces_matched']}; support_log diff mean {s['support_log_candidate_minus_target'].get('mean')}; position diff mean {s['rel_word_pos_candidate_minus_target'].get('mean')}")
    md += ["", "## Reading", "", "These selections define the next possible separation: source novelty at fixed relational/event semantics and relational/event semantics at fixed source novelty. Use them only after the pending 100M endpoint and A02 transfer result show the local compact channel reaches task competence.", ""]
    (out / "novelty_semantics_selection_profile.md").write_text("\n".join(md), encoding="utf-8")
    print(json.dumps({"status": payload["status"], "out_json": str(out / "novelty_semantics_selection_profile.json"), "n_groups_total": len(groups), "group_keys": sorted(group_summaries), "matches": {k: {kk: vv for kk, vv in v.items() if kk in ("matched_pairs", "fraction_target_matched", "target_total_bpe_pieces_matched", "candidate_total_bpe_pieces_matched", "exact_total_bpe_match")} for k, v in matches.items()}, "elapsed_sec": payload["elapsed_sec"]}, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
