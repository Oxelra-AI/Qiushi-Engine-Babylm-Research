#!/usr/bin/env python3
"""research: Materialize position-matched direct register-displacement pools.

This refines the direct register-displacement instrument by reducing the row-order
confound. For each row word count in the quarter_1x FineWeb block, it selects
matched pairs of clean rows: one child/subtitle-preferential, one
Gutenberg/SimpleWiki-preferential, with small row-index distance. The same
quarter_1x FineWeb text multiset is inserted into the selected positions of each
arm by exact word count. Unselected rows remain original clean text.

No generation, streaming, model loading, training, official evaluation, upload,
or leaderboard action.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative

import collections, hashlib, json, pathlib, statistics, time, math
from typing import Any

try:
    from scipy.optimize import linear_sum_assignment
except Exception as e:  # pragma: no cover
    linear_sum_assignment = None


def find_user_root() -> pathlib.Path:
    return _PUBLIC_ROOT

ROOT = find_user_root()
WS = ROOT / "experiments/archive/frontier_consolidation"
POOL256 = WS / "data/dose_2p64x_rowholdout_pools"
POOL280 = WS / "data/subdose_ladder_maxgeom_pools"
OUT_DIR = WS / "data/register_position_matched_direct_pools"

CLEAN_10M = POOL256 / "cleanqwen_lengthmatched_dose2p64x_10M.jsonl"
VIEW_10M = POOL256 / "compact_view_dose2p64x_10M.jsonl"
QUARTER_10M = POOL280 / "subdose_quarter_1x_view_10M.jsonl"
CLEAN_META = POOL256 / "cleanqwen_lengthmatched_dose2p64x_changed_block_rows_meta.jsonl"
VIEW_META = POOL256 / "compact_view_dose2p64x_changed_block_rows_meta.jsonl"
SUBDOSE_META = POOL280 / "subdose_ladder_metadata.json"

PAIR_ROWS = 7923
META_ROWS = 7924
TOTAL_LINES = 65313
BUDGET_WORDS = 10_000_000
REPEATS_100M = 10
CHILD_SUB = {"childes", "open_subtitles"}
ADULT = {"gutenberg", "simple_wiki"}

# Cost weights: position dominates within a strong-purity candidate band; purity
# tie-break prevents using weak rows when nearby pure rows exist.
POSITION_WEIGHT = 1.0
PURITY_WEIGHT = 35.0
TOP_MULT = 5
TOP_PAD = 40


def now():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def rel(p: pathlib.Path) -> str:
    try:
        return str(p.relative_to(ROOT))
    except Exception:
        return str(p)


def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def write_jsonl(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, separators=(",", ":")) + "\n")


def repeat_jsonl(src: pathlib.Path, dst: pathlib.Path, repeats: int=REPEATS_100M) -> None:
    with open(dst, "w", encoding="utf-8") as out:
        for _ in range(repeats):
            with open(src, encoding="utf-8") as inp:
                for line in inp:
                    out.write(line)


def row_words(r: dict[str, Any]) -> int:
    return int(r.get("words", len(str(r.get("text", "")).split())))


def comp_words(meta: dict[str, Any], sources: set[str]) -> int:
    comps = meta.get("component_sources", {}) or {}
    return sum(int(comps.get(k, 0)) for k in sources)


def purity(meta: dict[str, Any], desired: set[str], opposed: set[str]) -> float:
    w = max(1, int(meta.get("words", 0)))
    return (comp_words(meta, desired) - comp_words(meta, opposed)) / w


def summarise_meta(rows: list[dict[str, Any]]) -> dict[str, Any]:
    lengths = [int(r.get("words", 0)) for r in rows]
    comps = collections.Counter()
    idxs = []
    for r in rows:
        comps.update({k:int(v) for k,v in (r.get("component_sources", {}) or {}).items()})
        idxs.append(int(r.get("row_index", -1)))
    words = sum(lengths)
    child = sum(comps.get(k,0) for k in CHILD_SUB)
    adult = sum(comps.get(k,0) for k in ADULT)
    bins = collections.Counter(f"{(i//500)*500:04d}-{(i//500)*500+499:04d}" for i in idxs)
    return {
        "n_rows": len(rows), "words": words,
        "mean_words": statistics.mean(lengths) if lengths else None,
        "min_words": min(lengths) if lengths else None,
        "max_words": max(lengths) if lengths else None,
        "row_index_min": min(idxs) if idxs else None,
        "row_index_max": max(idxs) if idxs else None,
        "row_index_mean": statistics.mean(idxs) if idxs else None,
        "row_index_sd": statistics.pstdev(idxs) if len(idxs) > 1 else 0.0,
        "row_index_bins_500": dict(sorted(bins.items())),
        "component_words": dict(comps),
        "child_sub_words": child,
        "adult_gut_simple_words": adult,
        "other_words": words - child - adult,
        "child_sub_fraction": child / words if words else 0.0,
        "adult_gut_simple_fraction": adult / words if words else 0.0,
    }


def identify_quarter_active(clean_rows, view_rows, quarter_rows) -> list[int]:
    active, bad = [], []
    for i in range(PAIR_ROWS):
        if quarter_rows[i]["text"] == view_rows[i]["text"]:
            active.append(i)
        elif quarter_rows[i]["text"] != clean_rows[i]["text"]:
            bad.append(i)
            if len(bad) >= 10:
                break
    if bad:
        raise AssertionError(f"Quarter mismatch at {bad}")
    return active


def choose_position_matched(clean_meta_pair_rows: list[dict[str, Any]], target_hist: collections.Counter) -> tuple[set[int], set[int], list[dict[str, Any]]]:
    if linear_sum_assignment is None:
        raise RuntimeError("scipy.optimize.linear_sum_assignment unavailable")
    child_sel: set[int] = set()
    adult_sel: set[int] = set()
    pair_records: list[dict[str, Any]] = []
    for w, need in sorted(target_hist.items()):
        idxs = [i for i,r in enumerate(clean_meta_pair_rows) if int(r["words"]) == w]
        child_rank = sorted(idxs, key=lambda i: purity(clean_meta_pair_rows[i], CHILD_SUB, ADULT), reverse=True)
        adult_rank = sorted(idxs, key=lambda i: purity(clean_meta_pair_rows[i], ADULT, CHILD_SUB), reverse=True)
        k = min(len(idxs), max(need * TOP_MULT, need + TOP_PAD))
        C = child_rank[:k]
        A = adult_rank[:k]
        # If k is too small to allow enough disjoint pairs, expand to all rows of that length.
        if len(C) < need or len(A) < need:
            C, A = child_rank, adult_rank
        # rectangular assignment over top candidate bands, then choose the best `need` assigned pairs.
        import numpy as np
        cost = np.zeros((len(C), len(A)), dtype=float)
        for ci, c in enumerate(C):
            pc = purity(clean_meta_pair_rows[c], CHILD_SUB, ADULT)
            for ai, a in enumerate(A):
                pa = purity(clean_meta_pair_rows[a], ADULT, CHILD_SUB)
                # Same clean row in both arms is allowed only when exact word-length
                # availability forces overlap. The penalty makes distinct matched
                # rows preferred whenever they are available.
                same_penalty = 1e5 if c == a else 0.0
                cost[ci, ai] = POSITION_WEIGHT * abs(c - a) - PURITY_WEIGHT * (pc + pa) + same_penalty
        rr, cc = linear_sum_assignment(cost)
        pairs = []
        for ci, ai in zip(rr, cc):
            c, a = C[ci], A[ai]
            pc = purity(clean_meta_pair_rows[c], CHILD_SUB, ADULT)
            pa = purity(clean_meta_pair_rows[a], ADULT, CHILD_SUB)
            pairs.append((cost[ci,ai], abs(c-a), -(pc+pa), c, a, pc, pa))
        pairs.sort()
        take = pairs[:need]
        if len(take) < need:
            raise AssertionError(f"Could only choose {len(take)} pairs of length {w}, need {need}")
        for _, dist, negpur, c, a, pc, pa in take:
            child_sel.add(c); adult_sel.add(a)
            pair_records.append({"words": w, "child_row": c, "adult_row": a, "same_clean_row_both_arms": c == a, "row_distance": dist, "child_purity": pc, "adult_purity": pa})
    if len(child_sel) != sum(target_hist.values()) or len(adult_sel) != sum(target_hist.values()):
        raise AssertionError({"child": len(child_sel), "adult": len(adult_sel), "target": sum(target_hist.values())})
    if collections.Counter(int(clean_meta_pair_rows[i]["words"]) for i in child_sel) != target_hist:
        raise AssertionError("child length hist mismatch")
    if collections.Counter(int(clean_meta_pair_rows[i]["words"]) for i in adult_sel) != target_hist:
        raise AssertionError("adult length hist mismatch")
    return child_sel, adult_sel, pair_records


def assign_fineweb(active_rows: list[int], target_positions: set[int], view_meta_pair_rows, clean_meta_pair_rows) -> dict[int, int]:
    fine_by_len = collections.defaultdict(list)
    pos_by_len = collections.defaultdict(list)
    for i in active_rows:
        fine_by_len[int(view_meta_pair_rows[i]["words"])].append(i)
    for j in sorted(target_positions):
        pos_by_len[int(clean_meta_pair_rows[j]["words"])].append(j)
    if {k:len(v) for k,v in fine_by_len.items()} != {k:len(v) for k,v in pos_by_len.items()}:
        raise AssertionError("length count mismatch")
    assign = {}
    for w in sorted(fine_by_len):
        for pos, src in zip(pos_by_len[w], fine_by_len[w]):
            assign[pos] = src
    return assign


def text_multiset_hash(rows: list[dict[str, Any]]) -> str:
    h = hashlib.sha256()
    for text in sorted(str(r.get("text", "")) for r in rows):
        h.update(text.encode("utf-8")); h.update(b"\0")
    return h.hexdigest()


def build_arm(arm: str, target_positions: set[int], active_rows: list[int], clean_rows, view_rows, clean_meta_pair_rows, view_meta_pair_rows, original_fw_hash: str) -> dict[str, Any]:
    assign = assign_fineweb(active_rows, target_positions, view_meta_pair_rows, clean_meta_pair_rows)
    out_rows = []
    repl_records = []
    for i in range(TOTAL_LINES):
        if i in assign:
            src = assign[i]
            r = dict(view_rows[src])
            r["regpos_arm"] = arm
            r["regpos_row_kind"] = "position_matched_direct_fineweb_replacement"
            r["regpos_output_row_index"] = i
            r["regpos_original_fineweb_row_index"] = src
            r["regpos_replaced_clean_row_index"] = i
            out_rows.append(r)
            repl_records.append({"output_row_index": i, "original_fineweb_row_index": src, "words": row_words(r), "replaced_clean_component_sources": clean_meta_pair_rows[i].get("component_sources", {})})
        else:
            r = dict(clean_rows[i])
            r["regpos_arm"] = arm
            r["regpos_row_kind"] = "original_clean_unselected"
            r["regpos_output_row_index"] = i
            out_rows.append(r)
    for i,r in enumerate(out_rows):
        if row_words(r) != row_words(clean_rows[i]):
            raise AssertionError(f"row words mismatch at {i}")
    n, words = len(out_rows), sum(row_words(r) for r in out_rows)
    assert n == TOTAL_LINES and words == BUDGET_WORDS
    inserted_fw_hash = text_multiset_hash([out_rows[i] for i in sorted(target_positions)])
    assert inserted_fw_hash == original_fw_hash
    tenm = OUT_DIR / f"regpos_{arm}_samefw_quarter_10M.jsonl"
    hundredm = OUT_DIR / f"regpos_{arm}_samefw_quarter_100M.jsonl"
    write_jsonl(tenm, out_rows)
    repeat_jsonl(tenm, hundredm)
    displaced_rows = [clean_meta_pair_rows[i] for i in sorted(target_positions)]
    return {
        "arm_key": arm,
        "file_10m": rel(tenm), "file_100m": rel(hundredm),
        "sha_10m": sha256_file(tenm), "sha_100m": sha256_file(hundredm),
        "rows_10m": n, "words_10m": words,
        "rows_100m_by_construction": n*REPEATS_100M, "words_100m_by_construction": words*REPEATS_100M,
        "rho": sum(int(view_meta_pair_rows[i]["words"]) for i in active_rows) / BUDGET_WORDS,
        "active_fineweb_rows": len(active_rows),
        "active_fineweb_words": sum(int(view_meta_pair_rows[i]["words"]) for i in active_rows),
        "fineweb_text_multiset_hash": inserted_fw_hash,
        "displaced_clean_summary": summarise_meta(displaced_rows),
        "direct_replacement_row_records": repl_records,
        "no_clean_row_relocation": True,
        "unselected_rows_equal_original_clean_text": True,
        "row_word_count_sequence_identical_to_max_clean": True,
    }


def main() -> None:
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    clean_rows = load_jsonl(CLEAN_10M); view_rows = load_jsonl(VIEW_10M); quarter_rows = load_jsonl(QUARTER_10M)
    clean_meta = load_jsonl(CLEAN_META); view_meta = load_jsonl(VIEW_META)
    submeta = json.loads(SUBDOSE_META.read_text())
    assert len(clean_rows) == len(view_rows) == len(quarter_rows) == TOTAL_LINES
    assert len(clean_meta) == len(view_meta) == META_ROWS
    clean_meta_pair_rows = clean_meta[:PAIR_ROWS]; view_meta_pair_rows = view_meta[:PAIR_ROWS]
    for i in range(TOTAL_LINES):
        if not (row_words(clean_rows[i]) == row_words(view_rows[i]) == row_words(quarter_rows[i])):
            raise AssertionError(f"base geometry mismatch at {i}")
    active_rows = identify_quarter_active(clean_rows, view_rows, quarter_rows)
    active_words = sum(int(view_meta_pair_rows[i]["words"]) for i in active_rows)
    assert len(active_rows) == int(submeta["doses"]["quarter_1x"]["active_rows"])
    assert active_words == int(submeta["doses"]["quarter_1x"]["active_total_pw"])
    target_hist = collections.Counter(int(view_meta_pair_rows[i]["words"]) for i in active_rows)
    child_pos, adult_pos, pair_records = choose_position_matched(clean_meta_pair_rows, target_hist)
    original_fw_hash = text_multiset_hash([view_rows[i] for i in active_rows])
    child_result = build_arm("childsub_posmatched", child_pos, active_rows, clean_rows, view_rows, clean_meta_pair_rows, view_meta_pair_rows, original_fw_hash)
    adult_result = build_arm("adult_posmatched", adult_pos, active_rows, clean_rows, view_rows, clean_meta_pair_rows, view_meta_pair_rows, original_fw_hash)

    dists = [r["row_distance"] for r in pair_records]
    pos_summary = {
        "n_pairs": len(pair_records),
        "mean_abs_row_distance": statistics.mean(dists),
        "median_abs_row_distance": statistics.median(dists),
        "p90_abs_row_distance": sorted(dists)[int(0.9*(len(dists)-1))],
        "max_abs_row_distance": max(dists),
        "same_clean_row_both_arms_count": sum(1 for r in pair_records if r.get("same_clean_row_both_arms")),
        "mean_child_purity": statistics.mean(r["child_purity"] for r in pair_records),
        "mean_adult_purity": statistics.mean(r["adult_purity"] for r in pair_records),
    }
    child_D = set(r["output_row_index"] for r in child_result["direct_replacement_row_records"])
    adult_D = set(r["output_row_index"] for r in adult_result["direct_replacement_row_records"])
    summary = {
        "status": "REGISTER_POSITION_MATCHED_DIRECT_POOLS_MATERIALIZED",
        "created_utc": now(),
        "purpose": "Same-rho register-displacement pair with position-matched child/subtitle and adult-prose replacement rows, identical admitted quarter_1x FineWeb text multiset, and no clean row relocation.",
        "base_files": {"clean_10m": rel(CLEAN_10M), "view_10m": rel(VIEW_10M), "quarter_10m": rel(QUARTER_10M), "clean_meta": rel(CLEAN_META), "view_meta": rel(VIEW_META)},
        "geometry": {"total_rows_10m": TOTAL_LINES, "budget_words_10m": BUDGET_WORDS, "active_fineweb_rows": len(active_rows), "active_fineweb_words": active_words, "rho": active_words/BUDGET_WORDS, "same_row_word_count_sequence": True, "same_fineweb_text_multiset_across_arms": True},
        "position_matching_summary": pos_summary,
        "arm_results": [child_result, adult_result],
        "contrast_summary": {
            "displaced_set_intersection_rows": len(child_D & adult_D),
            "displaced_set_union_rows": len(child_D | adult_D),
            "displaced_set_jaccard": len(child_D & adult_D)/len(child_D | adult_D),
            "child_row_index_mean": child_result["displaced_clean_summary"]["row_index_mean"],
            "adult_row_index_mean": adult_result["displaced_clean_summary"]["row_index_mean"],
            "row_index_mean_difference_child_minus_adult": child_result["displaced_clean_summary"]["row_index_mean"] - adult_result["displaced_clean_summary"]["row_index_mean"],
            "childsub_displaced_child_sub_fraction": child_result["displaced_clean_summary"]["child_sub_fraction"],
            "childsub_displaced_adult_fraction": child_result["displaced_clean_summary"]["adult_gut_simple_fraction"],
            "adult_displaced_child_sub_fraction": adult_result["displaced_clean_summary"]["child_sub_fraction"],
            "adult_displaced_adult_fraction": adult_result["displaced_clean_summary"]["adult_gut_simple_fraction"],
        },
        "elapsed_sec": round(time.time()-t0,3),
        "no_generation_streaming_model_loading_training_eval_upload_or_leaderboard": True,
    }
    (OUT_DIR / "register_position_matched_direct_pools_metadata.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")
    (OUT_DIR / "position_matched_pair_records.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False)+"\n" for r in pair_records), encoding="utf-8")
    md = [
        "# research position-matched direct register-displacement pools", "",
        "CPU/file-only materialization. No generation, streaming, model loading, training, official evaluation, upload, or leaderboard action.", "",
        "## Scientific contrast", 
        "Both arms admit the same quarter_1x FineWeb text multiset (758 rows, 105,962 words, rho=0.0105962) by exact word count. The replaced clean rows are selected as child/subtitle vs Gutenberg/SimpleWiki pairs with the same word-count histogram and reduced row-index mismatch. Unselected rows remain original clean text at original positions.", "",
        "## Position matching", "```json", json.dumps(pos_summary, indent=2, ensure_ascii=False), "```", "",
        "## Arm summaries",
    ]
    for ar in [child_result, adult_result]:
        ds = ar["displaced_clean_summary"]
        md += [f"### {ar['arm_key']}", f"10M: `{ar['file_10m']}`", f"100M: `{ar['file_100m']}`", f"SHA10: `{ar['sha_10m']}`", f"SHA100: `{ar['sha_100m']}`", f"Displaced rows/words: {ds['n_rows']} / {ds['words']}", f"Displaced composition: child+subtitle {ds['child_sub_words']} ({ds['child_sub_fraction']:.6f}), Gutenberg+SimpleWiki {ds['adult_gut_simple_words']} ({ds['adult_gut_simple_fraction']:.6f}), other {ds['other_words']}", f"Row index mean/sd/span: {ds['row_index_mean']:.1f} / {ds['row_index_sd']:.1f} / {ds['row_index_min']}..{ds['row_index_max']}", ""]
    md += ["## Interpretation use", "Use this position-matched direct pair as the primary register-removal H100 discriminator if the sub-dose curve makes rho≈0.011 worth resolving. The simpler direct pair remains a sensitivity check; the fixed-position relocation pair remains a different control on active FineWeb timing.", "", "Metadata: `experiments/archive/frontier_consolidation/data/register_position_matched_direct_pools/register_position_matched_direct_pools_metadata.json`"]
    (OUT_DIR / "register_position_matched_direct_pools_summary.md").write_text("\n".join(md)+"\n", encoding="utf-8")
    print(json.dumps({"status": summary["status"], "summary": rel(OUT_DIR / "register_position_matched_direct_pools_summary.md"), "metadata": rel(OUT_DIR / "register_position_matched_direct_pools_metadata.json"), "position_matching": pos_summary, "arms": [{"arm": ar["arm_key"], "file_100m": ar["file_100m"], "sha_100m": ar["sha_100m"], "displaced_child_frac": ar["displaced_clean_summary"]["child_sub_fraction"], "displaced_adult_frac": ar["displaced_clean_summary"]["adult_gut_simple_fraction"], "row_index_mean": ar["displaced_clean_summary"]["row_index_mean"]} for ar in [child_result, adult_result]], "elapsed_sec": summary["elapsed_sec"]}, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
