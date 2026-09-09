#!/usr/bin/env python3
"""research: clean matched-subset design for source-attested bridge tests.

This is CPU/file-only. It joins the compact changed-block row packing with the
existing research/237 bridge candidates and quantifies what a clean same-row
bridge-vs-compact-vs-extractive comparison would require.

The key design correction is: do NOT fill missing bridge pairs with natural
compact rows and then compare to compact. Instead choose a matched subset of
whole row units for which every pair has an accepted transformation-like bridge
candidate. Then construct matched 10M/100M streams where all arm-specific rows
are drawn only from this subset; every non-subset slot is arm-identical filler.
This tests the bridge realization on the same rows rather than a compact/bridge
blend.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse
import csv
import hashlib
import json
import math
import re
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

WORD_RE = re.compile(r"[A-Za-z0-9]+(?:['\u2019-][A-Za-z0-9]+)?")
STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "then", "else", "so", "because", "as", "than",
    "to", "of", "in", "on", "for", "with", "without", "by", "from", "at", "into", "onto", "over",
    "under", "about", "between", "among", "through", "during", "before", "after", "above", "below",
    "is", "am", "are", "was", "were", "be", "been", "being", "do", "does", "did", "done", "doing",
    "have", "has", "had", "having", "can", "could", "may", "might", "must", "shall", "should",
    "will", "would", "this", "that", "these", "those", "there", "here", "it", "its", "they", "them",
    "their", "theirs", "he", "him", "his", "she", "her", "hers", "we", "us", "our", "ours", "you",
    "your", "yours", "i", "me", "my", "mine", "who", "whom", "whose", "which", "what", "where",
    "when", "why", "how", "not", "no", "nor", "only", "just", "also", "very", "more", "most", "less",
    "least", "much", "many", "some", "any", "all", "each", "every", "other", "another", "such", "own",
    "same", "too", "again", "still", "already", "yet", "up", "down", "out", "off", "back", "away",
    "within", "across", "per", "via", "using", "used", "use", "uses", "become", "became",
}
TRANSFORM_CLASSES = {"light_restructure", "substantive_restructure"}
CHANGED_MIN = 950000
CHANGED_MAX = 953004
EXPECTED_CHANGED_ROWS = 3005
EXPECTED_ROWS_10M = 64740
EXPECTED_WORDS_10M = 10_000_000


def find_user_root() -> Path:
    return _PUBLIC_ROOT


ROOT = find_user_root()
WS = ROOT / "experiments/archive/frontier_consolidation"
PAIR_PATH = WS / "data/factorial_view_candidate_audit/compact_candidate_pairs.jsonl"
ROW_META = WS / "data/density_cleanqwen_overlay_medium_riskhard/cleanqwen_fineweb_compact_view_reinvest_changed_block_rows_meta.jsonl"
COMPACT_10M = WS / "data/density_cleanqwen_overlay_medium_riskhard/cleanqwen_fineweb_compact_view_reinvest_10M.jsonl"
COMPACT_100M = WS / "data/density_cleanqwen_overlay_medium_riskhard/cleanqwen_fineweb_compact_view_reinvest_100M.jsonl"
ACCEPTED = WS / "data/improved_fluent_bridge/final_accepted.jsonl"
STRUCTURAL = WS / "data/bridge_structural_transformation/per_pair_structural.csv"
EXTRACTIVE_DIR = WS / "data/extractive_view_pool_preflight"
EXTRACTIVE_PREFLIGHT = EXTRACTIVE_DIR / "extractive_view_pool_preflight.json"
OUT_DIR_DEFAULT = WS / "data/bridge_subset_clean_design"
TOKENIZER_PATH = WS / "data/compliant_tokenizer"


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def split_words(text: str) -> list[str]:
    return [w for w in text.split() if w]


def norm_word(w: str) -> str:
    parts = WORD_RE.findall(w)
    return "".join(parts).lower() if parts else ""


def is_content(norm: str) -> bool:
    if not norm:
        return False
    if norm in STOPWORDS:
        return False
    if norm.isdigit():
        return True
    return len(norm) >= 4


def monotone_align(src_norms: list[str], view_norms: list[str]) -> list[int | None]:
    # Exact spirit of research atlas: greedy monotone alignment over all normalized words.
    pos_by_norm: dict[str, list[int]] = defaultdict(list)
    for i, n in enumerate(src_norms):
        if n:
            pos_by_norm[n].append(i)
    cursors: dict[str, int] = defaultdict(int)
    last = -1
    aligned: list[int | None] = []
    for n in view_norms:
        if not n or n not in pos_by_norm:
            aligned.append(None)
            continue
        poss = pos_by_norm[n]
        cur = cursors[n]
        chosen = None
        # first occurrence after last, preserving monotone order
        for j in range(cur, len(poss)):
            if poss[j] > last:
                chosen = poss[j]
                cursors[n] = j + 1
                break
        if chosen is None:
            aligned.append(None)
        else:
            aligned.append(chosen)
            last = chosen
    return aligned


def gap_metrics(source_text: str, view_text: str) -> dict[str, Any]:
    src = split_words(source_text)
    vw = split_words(view_text)
    src_norms = [norm_word(w) for w in src]
    vw_norms = [norm_word(w) for w in vw]
    aligned = monotone_align(src_norms, vw_norms)
    adjacent = 0
    gap1 = 0
    skip = 0
    pos = [a for a in aligned if a is not None]
    for a, b in zip(aligned, aligned[1:]):
        if a is None or b is None:
            continue
        adjacent += 1
        if b == a + 1:
            gap1 += 1
        else:
            skip += 1
    source_span = None
    if pos and len(src) > 0:
        source_span = (max(pos) - min(pos) + 1) / len(src)
    content_norms = [n for n in vw_norms if is_content(n)]
    src_bag = Counter(n for n in src_norms if is_content(n))
    absent = 0
    retained = 0
    bag = dict(src_bag)
    for n in content_norms:
        if bag.get(n, 0) > 0:
            bag[n] -= 1
            retained += 1
        else:
            absent += 1
    return {
        "view_words": len(vw),
        "source_words": len(src),
        "compression": len(vw) / len(src) if src else None,
        "aligned_view_fraction": sum(a is not None for a in aligned) / len(vw) if vw else None,
        "source_span": source_span,
        "adjacent_aligned_pairs": adjacent,
        "gap1_adjacent_pairs": gap1,
        "skip_adjacent_pairs": skip,
        "gap1_frac": gap1 / adjacent if adjacent else None,
        "skip_frac": skip / adjacent if adjacent else None,
        "view_content_words": len(content_norms),
        "source_absent_content_words": absent,
        "source_absent_content_frac": absent / len(content_norms) if content_norms else 0.0,
        "source_retained_content_words": retained,
    }


def build_extractive_wide(source_words: list[str], target_count: int) -> str:
    N = len(source_words)
    V = target_count
    if V >= N:
        return " ".join(source_words[:V])
    norms = [norm_word(w) for w in source_words]
    priorities = []
    for i in range(N):
        p = 2.0 if is_content(norms[i]) else 0.0
        p += (i / max(N - 1, 1)) * 0.15
        p += i * 1e-8
        priorities.append(p)
    top = sorted(range(N), key=lambda i: priorities[i], reverse=True)[:V]
    return " ".join(source_words[i] for i in sorted(top))


def build_extractive_balanced(source_words: list[str], target_count: int, target_content_frac: float = 0.6474) -> str:
    N = len(source_words)
    V = target_count
    if V >= N:
        return " ".join(source_words[:V])
    norms = [norm_word(w) for w in source_words]
    cont = [is_content(n) for n in norms]
    cidx = [i for i, c in enumerate(cont) if c]
    fidx = [i for i, c in enumerate(cont) if not c]
    target_c = min(len(cidx), max(1, round(target_content_frac * V)))
    target_f = V - target_c
    if target_f > len(fidx):
        target_f = len(fidx)
        target_c = V - target_f
    if target_c > len(cidx):
        target_c = len(cidx)
        target_f = V - target_c

    def evenly(indices: list[int], k: int) -> list[int]:
        if k <= 0:
            return []
        if k >= len(indices):
            return list(indices)
        step = len(indices) / k
        sel = []
        for j in range(k):
            idx = min(int(j * step + step / 2), len(indices) - 1)
            sel.append(indices[idx])
        sel = sorted(set(sel))
        rem = [i for i in indices if i not in set(sel)]
        while len(sel) < k and rem:
            sel.append(rem.pop(0))
            sel.sort()
        return sel[:k]

    kept = sorted(set(evenly(cidx, target_c) + evenly(fidx, target_f)))
    while len(kept) > V:
        for i in list(kept):
            if not cont[i]:
                kept.remove(i)
                break
        else:
            kept.pop(0)
    while len(kept) < V:
        candidates = [i for i in range(N) if i not in set(kept)]
        if not candidates:
            break
        kept.append(max(candidates))
        kept.sort()
    return " ".join(source_words[i] for i in kept)


def stat(vals: list[float | int | None]) -> dict[str, Any]:
    xs = [float(v) for v in vals if v is not None and math.isfinite(float(v))]
    if not xs:
        return {"n": 0}
    xs_sorted = sorted(xs)
    def pct(p: float) -> float:
        return xs_sorted[min(len(xs_sorted)-1, max(0, int(round((len(xs_sorted)-1)*p))))]
    return {
        "n": len(xs),
        "mean": statistics.mean(xs),
        "median": statistics.median(xs),
        "p10": pct(0.10),
        "p25": pct(0.25),
        "p75": pct(0.75),
        "p90": pct(0.90),
        "min": min(xs),
        "max": max(xs),
    }


def summarize_geoms(geoms: list[dict[str, Any]]) -> dict[str, Any]:
    totals = {
        "adjacent_aligned_pairs": sum(int(g["adjacent_aligned_pairs"]) for g in geoms),
        "gap1_adjacent_pairs": sum(int(g["gap1_adjacent_pairs"]) for g in geoms),
        "skip_adjacent_pairs": sum(int(g["skip_adjacent_pairs"]) for g in geoms),
        "view_content_words": sum(int(g["view_content_words"]) for g in geoms),
        "source_absent_content_words": sum(int(g["source_absent_content_words"]) for g in geoms),
        "view_words": sum(int(g["view_words"]) for g in geoms),
        "source_words": sum(int(g["source_words"]) for g in geoms),
    }
    totals["pooled_gap1"] = totals["gap1_adjacent_pairs"] / totals["adjacent_aligned_pairs"] if totals["adjacent_aligned_pairs"] else None
    totals["pooled_skip"] = totals["skip_adjacent_pairs"] / totals["adjacent_aligned_pairs"] if totals["adjacent_aligned_pairs"] else None
    totals["pooled_absent_content_frac"] = totals["source_absent_content_words"] / totals["view_content_words"] if totals["view_content_words"] else 0.0
    totals["pooled_compression"] = totals["view_words"] / totals["source_words"] if totals["source_words"] else None
    return {
        "n": len(geoms),
        "stats": {
            "gap1_frac": stat([g["gap1_frac"] for g in geoms]),
            "skip_frac": stat([g["skip_frac"] for g in geoms]),
            "source_span": stat([g["source_span"] for g in geoms]),
            "compression": stat([g["compression"] for g in geoms]),
            "aligned_view_fraction": stat([g["aligned_view_fraction"] for g in geoms]),
            "source_absent_content_frac": stat([g["source_absent_content_frac"] for g in geoms]),
        },
        "totals": totals,
    }


def choose_best_by_pair(accepted_rows: list[dict[str, Any]], structural_by_pair: dict[str, dict[str, Any]], tiers_by_pair: dict[str, str]) -> dict[str, dict[str, Any]]:
    # Keep only accepted outputs that structural analysis calls transformation-like.
    out: dict[str, dict[str, Any]] = {}
    rank_tier = {"A_seed_transform": 0, "B_adjudicate_transform": 1, "unknown": 2}
    rank_class = {"substantive_restructure": 0, "light_restructure": 1}
    for r in accepted_rows:
        pid = r["pair_id"]
        s = structural_by_pair.get(pid)
        if not s or s.get("edit_class") not in TRANSFORM_CLASSES:
            continue
        rec = dict(r)
        rec["edit_class"] = s.get("edit_class")
        rec["tier"] = tiers_by_pair.get(pid, "unknown")
        rec["longest_run_frac"] = float(s.get("longest_run_frac", "nan"))
        rec["order_agreement"] = float(s.get("order_agreement", "nan"))
        rec["function_word_edit_rate"] = float(s.get("function_word_edit_rate", "nan"))
        rec["novel_word_fraction"] = float(s.get("novel_word_fraction", "nan"))
        # deterministic quality: seed/adjudicate, substantive, lower copied run, higher function edit, closer to target length.
        word_ratio = float(rec.get("word_ratio", 999))
        score = (
            rank_tier.get(rec["tier"], 3),
            rank_class.get(rec["edit_class"], 2),
            rec["longest_run_frac"],
            -rec["function_word_edit_rate"],
            abs(word_ratio - 1.0),
            str(rec.get("regime", "")),
        )
        if pid not in out or score < out[pid]["_rank_score"]:
            rec["_rank_score"] = score
            out[pid] = rec
    for rec in out.values():
        rec.pop("_rank_score", None)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR_DEFAULT)
    ap.add_argument("--materialize-prototype", action="store_true", help="write prototype subset-only 10M pools using existing candidates")
    args = ap.parse_args()

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    for p in [PAIR_PATH, ROW_META, COMPACT_10M, COMPACT_100M, ACCEPTED, STRUCTURAL, EXTRACTIVE_PREFLIGHT]:
        if not p.exists():
            raise FileNotFoundError(p)

    pairs = load_jsonl(PAIR_PATH)
    pair_by_id = {p["pair_id"]: p for p in pairs}
    row_metas_all = load_jsonl(ROW_META)
    row_metas = [r for r in row_metas_all if CHANGED_MIN <= int(r.get("example_id", -1)) <= CHANGED_MAX]
    topup_or_extra = [r for r in row_metas_all if not (CHANGED_MIN <= int(r.get("example_id", -1)) <= CHANGED_MAX)]

    accepted = load_jsonl(ACCEPTED)
    structural_by_pair: dict[str, dict[str, Any]] = {}
    with STRUCTURAL.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            structural_by_pair[row["pair_id"]] = row
    tiers_by_pair: dict[str, str] = {}
    cand_json = out_dir.parent / "bridge_high_fidelity_candidates/bridge_high_fidelity_candidates.json"
    # Path above is not robust; use known location.
    cand_json = WS / "data/bridge_high_fidelity_candidates/bridge_high_fidelity_candidates.json"
    if cand_json.exists():
        cand = json.loads(cand_json.read_text(encoding="utf-8"))
        for r in cand.get("records", []):
            tiers_by_pair[r["pair_id"]] = r.get("tier", "unknown")
    bridge_by_pair = choose_best_by_pair(accepted, structural_by_pair, tiers_by_pair)

    # Whole-row coverage: a clean row is one for which every pair in that row has bridge.
    whole_rows: list[dict[str, Any]] = []
    partial_rows: list[dict[str, Any]] = []
    for r in row_metas:
        pids = list(r.get("pair_ids") or [])
        have = [pid for pid in pids if pid in bridge_by_pair]
        rec = {
            "row_index": int(r["row_index"]),
            "example_id": int(r["example_id"]),
            "words": int(r["words"]),
            "n_pairs": len(pids),
            "n_bridge_pairs": len(have),
            "pair_ids": pids,
            "bridge_pair_ids": have,
            "missing_pair_ids": [pid for pid in pids if pid not in bridge_by_pair],
            "component_sources": r.get("component_sources", {}),
        }
        if pids and len(have) == len(pids):
            whole_rows.append(rec)
        elif have:
            partial_rows.append(rec)

    # Compute alternative row text for clean rows.
    def build_row_text(row: dict[str, Any], variant: str) -> str:
        parts: list[str] = []
        for pid in row["pair_ids"]:
            pair = pair_by_id[pid]
            source = pair["source_text"]
            if variant == "compact":
                view = pair["view_text"]
            elif variant == "bridge":
                view = bridge_by_pair[pid]["generated_text"]
            elif variant == "extractive_balanced":
                view = build_extractive_balanced(split_words(source), int(pair["view_words"]))
            elif variant == "extractive_wide":
                view = build_extractive_wide(split_words(source), int(pair["view_words"]))
            else:
                raise ValueError(variant)
            parts.extend([source, view])
        return " ".join(parts)

    variants = ["compact", "bridge", "extractive_balanced", "extractive_wide"]
    clean_row_records: list[dict[str, Any]] = []
    for row in whole_rows:
        rec = dict(row)
        rec["variant_words"] = {}
        rec["variant_token_counts"] = {}
        rec["word_mismatch_vs_meta"] = {}
        for v in variants:
            text = build_row_text(row, v)
            rec["variant_words"][v] = len(split_words(text))
            rec["word_mismatch_vs_meta"][v] = len(split_words(text)) - row["words"]
        clean_row_records.append(rec)

    # Pair-level geometry for all existing bridge pairs and the clean whole-row subset.
    pair_geoms = {v: [] for v in variants}
    for pid, br in bridge_by_pair.items():
        pair = pair_by_id[pid]
        source = pair["source_text"]
        views = {
            "compact": pair["view_text"],
            "bridge": br["generated_text"],
            "extractive_balanced": build_extractive_balanced(split_words(source), int(pair["view_words"])),
            "extractive_wide": build_extractive_wide(split_words(source), int(pair["view_words"])),
        }
        for v, text in views.items():
            g = gap_metrics(source, text)
            g["pair_id"] = pid
            g["variant"] = v
            pair_geoms[v].append(g)

    whole_pair_ids = sorted({pid for r in whole_rows for pid in r["pair_ids"]})
    whole_pair_geoms = {v: [g for g in pair_geoms[v] if g["pair_id"] in set(whole_pair_ids)] for v in variants}

    # Existing compact-fallback dilution estimate: if all current bridge pairs were substituted in compact rows, how much of changed words differ?
    bridge_pair_word_sum = 0
    compact_pair_word_sum = 0
    for pid in bridge_by_pair:
        pair = pair_by_id[pid]
        bridge_pair_word_sum += int(pair["source_words"]) + len(split_words(bridge_by_pair[pid]["generated_text"]))
        compact_pair_word_sum += int(pair["source_words"]) + int(pair["view_words"])
    changed_words_per_10m = sum(int(r["words"]) for r in row_metas)
    whole_row_words = sum(r["words"] for r in whole_rows)
    whole_row_bridge_words = sum(rec["variant_words"]["bridge"] for rec in clean_row_records)

    # Row-selection possibilities for future full-pool generation.
    row_pair_count_dist = Counter(len(r.get("pair_ids") or []) for r in row_metas)
    clean_row_pair_count_dist = Counter(r["n_pairs"] for r in whole_rows)
    transform_pairs_by_bucket = Counter(str(bridge_by_pair[pid].get("prototype_bucket", "unknown")) for pid in bridge_by_pair)
    transform_rows_by_bucket_presence = Counter()
    for r in whole_rows:
        buckets = sorted(set(str(bridge_by_pair[pid].get("prototype_bucket", "unknown")) for pid in r["pair_ids"]))
        for b in buckets:
            transform_rows_by_bucket_presence[b] += 1

    # Prototype 10M materialization with identical neutral filler can be used as a dry-run only.
    prototype_paths: dict[str, Any] = {}
    if args.materialize_prototype and whole_rows:
        compact_rows = load_jsonl(COMPACT_10M)
        selected_examples = {r["example_id"] for r in whole_rows}
        # Replace non-selected changed rows by their existing compact text in every arm? That would be a blend and is not accepted for training.
        # Instead mark those rows arm-identical neutral by copying the COMPACT row identically in all arms under source 'bridge_subset_neutral'.
        # This prototype demonstrates mechanics only; a final training design should decide whether to top up with legal filler or resample selected rows.
        for v in variants:
            path = out_dir / f"prototype_existing_candidates_{v}_10M.jsonl"
            total_words = 0
            rows_written = 0
            arm_specific_rows = 0
            with path.open("w", encoding="utf-8") as fout:
                for obj in compact_rows:
                    eid = int(obj.get("example_id", -1))
                    if eid in selected_examples:
                        row = next(r for r in whole_rows if r["example_id"] == eid)
                        text = build_row_text(row, v)
                        out_obj = dict(obj)
                        out_obj["text"] = text
                        out_obj["words"] = len(split_words(text))
                        out_obj["source"] = f"clean_subset_{v}"
                        arm_specific_rows += 1
                    else:
                        out_obj = dict(obj)
                        if CHANGED_MIN <= eid <= CHANGED_MAX:
                            out_obj["source"] = "neutral_identical_compact_row_not_for_factor_training"
                    json.dump(out_obj, fout, ensure_ascii=False)
                    fout.write("\n")
                    total_words += int(out_obj["words"])
                    rows_written += 1
            prototype_paths[v] = {"path": str(path), "sha256": sha256_file(path), "rows": rows_written, "words": total_words, "arm_specific_rows": arm_specific_rows}

    payload = {
        "status": "BRIDGE_SUBSET_CLEAN_DESIGN",
        "created_utc": now(),
        "inputs": {
            "pair_path": str(PAIR_PATH),
            "row_meta": str(ROW_META),
            "accepted_bridge": str(ACCEPTED),
            "structural_csv": str(STRUCTURAL),
            "compact_10m_sha256": sha256_file(COMPACT_10M),
            "compact_100m_sha256": sha256_file(COMPACT_100M),
        },
        "counts": {
            "candidate_pairs_total": len(pairs),
            "row_meta_all_lines": len(row_metas_all),
            "changed_rows_in_range": len(row_metas),
            "topup_or_extra_rows_outside_range": len(topup_or_extra),
            "accepted_bridge_rows": len(accepted),
            "transformation_like_pairs_existing_best": len(bridge_by_pair),
            "whole_rows_fully_bridge_fillable_existing": len(whole_rows),
            "partial_rows_with_some_bridge_existing": len(partial_rows),
            "whole_row_pairs_existing": len(whole_pair_ids),
            "changed_words_per_10m": changed_words_per_10m,
            "whole_row_words_per_10m_existing": whole_row_words,
            "whole_row_bridge_words_per_10m_existing": whole_row_bridge_words,
            "whole_row_changed_word_fraction_existing": whole_row_words / changed_words_per_10m if changed_words_per_10m else None,
            "whole_row_total_word_fraction_existing": whole_row_words / EXPECTED_WORDS_10M,
            "fallback_blend_pair_word_fraction_changed_block_existing_upper": bridge_pair_word_sum / changed_words_per_10m if changed_words_per_10m else None,
            "fallback_blend_compact_pair_word_fraction_changed_block_existing": compact_pair_word_sum / changed_words_per_10m if changed_words_per_10m else None,
        },
        "row_pair_count_distribution": dict(sorted(row_pair_count_dist.items())),
        "clean_row_pair_count_distribution_existing": dict(sorted(clean_row_pair_count_dist.items())),
        "transform_pairs_by_bucket_existing": dict(transform_pairs_by_bucket),
        "clean_rows_by_bucket_presence_existing": dict(transform_rows_by_bucket_presence),
        "geometry_existing_all_transform_pairs": {v: summarize_geoms(pair_geoms[v]) for v in variants},
        "geometry_existing_whole_row_subset": {v: summarize_geoms(whole_pair_geoms[v]) for v in variants},
        "clean_design": {
            "why_step238_handoff_is_not_clean": "Filling bridge where available and keeping natural compact elsewhere makes a compact/bridge blend; a whole-stream score difference would mostly measure unchanged compact plus changed-pair selection and cannot identify the bridge realization.",
            "clean_unit": "whole changed-block row: every source+view pair in a row must have a bridge candidate before that row can enter the arm-specific subset; rows with partial bridge coverage are excluded or regenerated, not compact-filled.",
            "matched_arms": ["bridge_subset", "natural_compact_same_subset", "extractive_balanced_same_subset", "extractive_wide_same_subset"],
            "required_future_scale_gate": "Count whole rows, not only pairs. A later generation run should report fully fillable row words and pair/row geometry. A 100M pretraining run is only worth considering if enough whole-row bridge material exists that arm-specific words are not a tiny perturbation; if not, use local/frozen probes or redesign generation instead.",
            "interpretation_boundaries": {
                "bridge_positive_vs_compact": "Would imply that source-attested structural re-expression with compact-like order/continuity can support the selected DeBERTa competence at least as well as natural compact on those rows; because bridge also has different compression and retained-source fraction, it would not prove source-absent lexical novelty is useless.",
                "bridge_negative_vs_compact": "Would show that the natural compact realization has value not reproduced by the current source-attested bridge; the implicated bundle includes source-absent content, stronger compression, lower retained-source fraction, natural generation distribution, and any remaining quality differences, not novel vocabulary alone.",
                "bridge_between_compact_and_extractive": "Would support a continuum where compact-like structural geometry/fluent re-expression recovers part of the extractive deficit, while the remaining gap is a coupled natural-compact factor.",
                "bridge_approximately_extractive": "Would mean the apparent geometry match is not sufficient for downstream learning; source-attested bridge construction or quality still resembles source-only selection in the learning coordinate.",
                "bridge_local_only": "If local NLL/MLM improves but stable selected families do not, treat it as another local-learning/downstream split, not a route to endpoint training.",
            },
        },
        "clean_rows_existing_sample": clean_row_records[:20],
        "partial_rows_existing_sample": partial_rows[:20],
        "prototype_paths": prototype_paths,
        "no_training_generation_official_eval_upload_aoa_or_leaderboard": True,
    }
    out_json = out_dir / "bridge_subset_clean_design.json"
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    out_csv = out_dir / "existing_clean_whole_rows.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        fields = ["row_index", "example_id", "words", "n_pairs", "variant_words_compact", "variant_words_bridge", "variant_words_extractive_balanced", "variant_words_extractive_wide", "pair_ids"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in clean_row_records:
            w.writerow({
                "row_index": r["row_index"],
                "example_id": r["example_id"],
                "words": r["words"],
                "n_pairs": r["n_pairs"],
                "variant_words_compact": r["variant_words"]["compact"],
                "variant_words_bridge": r["variant_words"]["bridge"],
                "variant_words_extractive_balanced": r["variant_words"]["extractive_balanced"],
                "variant_words_extractive_wide": r["variant_words"]["extractive_wide"],
                "pair_ids": " ".join(r["pair_ids"]),
            })

    md = [
        "# research bridge matched-subset clean design",
        "",
        "## Immediate correction",
        "A bridge/compact fallback stream would be mostly natural compact and cannot measure the bridge factor. The clean unit is a whole changed-block row: every pair in the row must have a transformation-like bridge candidate; otherwise the row is excluded or must be regenerated.",
        "",
        "## Existing research/237 material",
        f"- Transformation-like bridge pairs available: {len(bridge_by_pair)} / {len(pairs)}",
        f"- Whole changed-block rows fully fillable now: {len(whole_rows)} / {len(row_metas)}",
        f"- Whole-row pair count now: {len(whole_pair_ids)} pairs",
        f"- Existing fillable row words per 10M: {whole_row_words} ({whole_row_words/EXPECTED_WORDS_10M:.4%} of total; {whole_row_words/changed_words_per_10m:.4%} of changed-block words)",
        f"- Partial rows with at least one bridge pair: {len(partial_rows)}",
        "",
        "## Geometry on existing whole-row subset",
        "| arm | n pairs | pooled gap1 | pooled skip | pooled absent content | pooled compression |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for v in variants:
        summ = payload["geometry_existing_whole_row_subset"][v]
        totals = summ["totals"]
        md.append(f"| {v} | {summ['n']} | {totals.get('pooled_gap1')} | {totals.get('pooled_skip')} | {totals.get('pooled_absent_content_frac')} | {totals.get('pooled_compression')} |")
    md += [
        "",
        "## Interpretation boundaries",
        "- A positive bridge-vs-compact result would not isolate novel vocabulary alone because bridge also changes compression, retained-source fraction, and generation distribution.",
        "- A negative bridge-vs-compact result would implicate the natural-compact bundle: source-absent content, stronger compression, lower source retention, surface quality/distribution, and any remaining row-selection differences.",
        "- Bridge-vs-extractive on the same rows is the test of whether compact-like structural re-expression recovers any of the extractive stable-family deficit.",
        "- Do not run 100M pretraining from a compact/bridge blend; the scale gate must count whole fillable rows and produce matched bridge/compact/extractive subset arms.",
        "",
        f"JSON: `{out_json}`",
        f"CSV: `{out_csv}`",
    ]
    out_md = out_dir / "bridge_subset_clean_design.md"
    out_md.write_text("\n".join(md) + "\n", encoding="utf-8")

    print(json.dumps({
        "status": payload["status"],
        "out_json": str(out_json),
        "out_md": str(out_md),
        "transformation_like_pairs": len(bridge_by_pair),
        "whole_rows": len(whole_rows),
        "whole_row_pairs": len(whole_pair_ids),
        "whole_row_words": whole_row_words,
        "whole_row_total_word_fraction": whole_row_words / EXPECTED_WORDS_10M,
        "no_training_generation_official_eval_upload_aoa_or_leaderboard": True,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
