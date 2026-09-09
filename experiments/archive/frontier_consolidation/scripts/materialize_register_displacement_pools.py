#!/usr/bin/env python3
"""research: Materialize register-contrast displacement pools.

Scientific purpose
------------------
The current fixed-budget admission leg is no longer only a coverage/amount
question. The sub-dose ladder replaces clean BabyLM-register rows with a small
FineWeb source+compact-view block. This script builds, from existing pools only,
a same-rho and same-row-geometry pair that admits the identical FineWeb block
as the existing quarter_1x arm but chooses which clean rows are absent:

  * childsub_removed: omit rows maximally composed of CHILDES + OpenSubtitles;
  * adult_removed:    omit rows maximally composed of Gutenberg + SimpleWiki.

Both arms:
  * use the identical active FineWeb row positions and text as quarter_1x;
  * preserve the exact 65,313-row / 10,000,000-word row-length sequence;
  * use no generation, streaming, model loading, training, evaluation, upload.

Implementation detail
---------------------
Let A be the quarter_1x active row positions in the first MAX changed block.
Let H be the multiset of row word-counts in A. For each target register, choose
D with exactly length multiset H from all clean changed-block rows, maximizing
register dominance within each word length. The pool keeps FineWeb rows at A.
Rows not in A or D stay at their original clean positions. Clean rows in A\D
are moved into same-length holes D\A. This controls the timing/position of the
admitted FineWeb block and makes the displaced clean-register set D the intended
contrast, with only the necessary length-preserving clean-row relocations.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative

import collections
import hashlib
import json
import pathlib
import statistics
import time
from typing import Any


def find_user_root() -> pathlib.Path:
    return _PUBLIC_ROOT

ROOT = find_user_root()
WS = ROOT / "experiments/archive/frontier_consolidation"
POOL256 = WS / "data/dose_2p64x_rowholdout_pools"
POOL280 = WS / "data/subdose_ladder_maxgeom_pools"
OUT_DIR = WS / "data/register_displacement_pools"

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

ARMS = collections.OrderedDict([
    ("childsub_removed", {
        "label": "quarter_rho_samefw_childes_opensubtitles_removed",
        "target_register": "CHILDES + OpenSubtitles clean rows removed",
        "desired_sources": sorted(CHILD_SUB),
        "opposed_sources": sorted(ADULT),
    }),
    ("adult_removed", {
        "label": "quarter_rho_samefw_gutenberg_simplewiki_removed",
        "target_register": "Gutenberg + SimpleWiki clean rows removed",
        "desired_sources": sorted(ADULT),
        "opposed_sources": sorted(CHILD_SUB),
    }),
])


def now() -> str:
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
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            out.append(json.loads(line))
    return out


def write_jsonl(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, separators=(",", ":")) + "\n")


def repeat_jsonl(src: pathlib.Path, dst: pathlib.Path, repeats: int = REPEATS_100M) -> None:
    with open(dst, "w", encoding="utf-8") as out:
        for _ in range(repeats):
            with open(src, encoding="utf-8") as inp:
                for line in inp:
                    out.write(line)


def word_count_rows(rows: list[dict[str, Any]]) -> tuple[int, int]:
    words = 0
    for r in rows:
        words += int(r.get("words", len(str(r.get("text", "")).split())))
    return len(rows), words


def comp_words(meta: dict[str, Any], sources: set[str]) -> int:
    comps = meta.get("component_sources", {}) or {}
    return sum(int(comps.get(k, 0)) for k in sources)


def score_for(meta: dict[str, Any], desired: set[str], opposed: set[str]) -> tuple[float, int, int, int]:
    w = int(meta.get("words", 0))
    d = comp_words(meta, desired)
    o = comp_words(meta, opposed)
    # Primary: dominance fraction. Secondary: desired words. Tertiary: fewer opposed words.
    return ((d - o) / max(1, w), d, -o, -int(meta.get("row_index", 0)))


def summarize_meta_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    lengths = [int(r.get("words", 0)) for r in rows]
    comps = collections.Counter()
    for r in rows:
        comps.update({k: int(v) for k, v in (r.get("component_sources", {}) or {}).items()})
    words = sum(lengths)
    child = sum(comps.get(k, 0) for k in CHILD_SUB)
    adult = sum(comps.get(k, 0) for k in ADULT)
    return {
        "n_rows": len(rows),
        "words": words,
        "mean_words": statistics.mean(lengths) if lengths else None,
        "min_words": min(lengths) if lengths else None,
        "max_words": max(lengths) if lengths else None,
        "component_words": dict(comps),
        "child_sub_words": child,
        "adult_gut_simple_words": adult,
        "other_words": words - child - adult,
        "child_sub_fraction": child / words if words else 0.0,
        "adult_gut_simple_fraction": adult / words if words else 0.0,
    }


def source_hist(rows: list[dict[str, Any]]) -> collections.Counter:
    c = collections.Counter()
    for r in rows:
        src = str(r.get("source", ""))
        c[src] += int(r.get("words", len(str(r.get("text", "")).split())))
    return c


def identify_active_rows(clean_rows: list[dict[str, Any]], view_rows: list[dict[str, Any]], quarter_rows: list[dict[str, Any]]) -> list[int]:
    active = []
    bad = []
    for i in range(PAIR_ROWS):
        q = quarter_rows[i]["text"]
        if q == view_rows[i]["text"]:
            active.append(i)
        elif q != clean_rows[i]["text"]:
            bad.append(i)
    if bad:
        raise AssertionError(f"Quarter row is neither view nor clean at first bad rows: {bad[:10]}")
    return active


def choose_displaced_set(clean_meta_pair_rows: list[dict[str, Any]], target_hist: collections.Counter, desired: set[str], opposed: set[str]) -> set[int]:
    by_len: dict[int, list[int]] = collections.defaultdict(list)
    for i, r in enumerate(clean_meta_pair_rows):
        by_len[int(r["words"])].append(i)
    selected: set[int] = set()
    deficits = []
    for w, need in sorted(target_hist.items()):
        cand = list(by_len[w])
        cand.sort(key=lambda i: score_for(clean_meta_pair_rows[i], desired, opposed), reverse=True)
        if len(cand) < need:
            deficits.append((w, need, len(cand)))
        selected.update(cand[:need])
    if deficits:
        raise AssertionError(f"Unexpected length deficits: {deficits[:10]}")
    if len(selected) != sum(target_hist.values()):
        raise AssertionError(f"Selected {len(selected)} rows, target {sum(target_hist.values())}")
    sel_hist = collections.Counter(int(clean_meta_pair_rows[i]["words"]) for i in selected)
    if sel_hist != target_hist:
        raise AssertionError("Selected length histogram does not equal target active histogram")
    return selected


def make_assignment(active: set[int], displaced: set[int], clean_meta_pair_rows: list[dict[str, Any]]) -> dict[int, int]:
    """Map inactive position -> clean source row index. Identity except holes D\A."""
    donors_by_len: dict[int, list[int]] = collections.defaultdict(list)
    holes_by_len: dict[int, list[int]] = collections.defaultdict(list)
    for i in sorted(active - displaced):
        donors_by_len[int(clean_meta_pair_rows[i]["words"])].append(i)
    for i in sorted(displaced - active):
        holes_by_len[int(clean_meta_pair_rows[i]["words"])].append(i)
    if {k: len(v) for k, v in donors_by_len.items()} != {k: len(v) for k, v in holes_by_len.items()}:
        raise AssertionError({"donors": {k: len(v) for k, v in donors_by_len.items()}, "holes": {k: len(v) for k, v in holes_by_len.items()}})
    assign = {}
    for i in range(PAIR_ROWS):
        if i in active:
            continue
        if i not in displaced:
            assign[i] = i
    relocations = {}
    for w in sorted(holes_by_len):
        donors = donors_by_len[w]
        holes = holes_by_len[w]
        for h, d in zip(holes, donors):
            relocations[h] = d
    assign.update(relocations)
    # All inactive pair positions assigned exactly once, and no displaced clean source is used.
    assert set(assign.keys()) == (set(range(PAIR_ROWS)) - active)
    used_clean = set(assign.values())
    if used_clean & displaced:
        raise AssertionError("Displaced clean row was still used")
    expected_used = set(range(PAIR_ROWS)) - displaced
    if used_clean != expected_used:
        raise AssertionError({"missing_used": sorted(expected_used-used_clean)[:10], "extra_used": sorted(used_clean-expected_used)[:10]})
    return assign


def build_arm_rows(arm_key: str, arm_info: dict[str, Any], active_list: list[int], clean_rows: list[dict[str, Any]], view_rows: list[dict[str, Any]], quarter_rows: list[dict[str, Any]], clean_meta_pair_rows: list[dict[str, Any]], view_meta_pair_rows: list[dict[str, Any]]) -> dict[str, Any]:
    active = set(active_list)
    target_hist = collections.Counter(int(view_meta_pair_rows[i]["words"]) for i in active)
    desired = set(arm_info["desired_sources"])
    opposed = set(arm_info["opposed_sources"])
    displaced = choose_displaced_set(clean_meta_pair_rows, target_hist, desired, opposed)
    assign = make_assignment(active, displaced, clean_meta_pair_rows)

    out_rows: list[dict[str, Any]] = []
    row_records = []
    for i in range(TOTAL_LINES):
        if i < PAIR_ROWS:
            if i in active:
                r = dict(view_rows[i])
                kind = "active_identical_fineweb_view"
                clean_source_index = None
            else:
                src_i = assign[i]
                r = dict(clean_rows[src_i])
                kind = "clean_identity" if src_i == i else "clean_relocated_same_length"
                clean_source_index = src_i
            # Keep training text/words/source content from original record, but add non-training metadata.
            r["regdisp_arm"] = arm_key
            r["regdisp_row_kind"] = kind
            r["regdisp_output_row_index"] = i
            if clean_source_index is not None:
                r["regdisp_clean_source_row_index"] = clean_source_index
            out_rows.append(r)
            if kind != "clean_identity":
                row_records.append({
                    "output_row_index": i,
                    "kind": kind,
                    "words": int(r.get("words", len(str(r.get("text", "")).split()))),
                    "clean_source_row_index": clean_source_index,
                    "omitted_clean_row_index_if_hole": i if i in displaced and i not in active else None,
                })
        else:
            # Use existing quarter tail to preserve its topup/filler convention exactly.
            r = dict(quarter_rows[i])
            r["regdisp_arm"] = arm_key
            r["regdisp_row_kind"] = "quarter_tail_or_shared_filler"
            r["regdisp_output_row_index"] = i
            out_rows.append(r)

    # Verify row lengths equal clean/view/quarter geometry at every line.
    mismatches = []
    for i, r in enumerate(out_rows):
        rw = int(r.get("words", len(str(r.get("text", "")).split())))
        gw = int(clean_rows[i].get("words", len(str(clean_rows[i].get("text", "")).split())))
        if rw != gw:
            mismatches.append((i, rw, gw))
            if len(mismatches) >= 10:
                break
    if mismatches:
        raise AssertionError(f"Row length mismatches: {mismatches}")
    rows_n, words_n = word_count_rows(out_rows)
    assert rows_n == TOTAL_LINES and words_n == BUDGET_WORDS, (rows_n, words_n)

    # Identity checks against quarter arm.
    active_text_bad = [i for i in active if out_rows[i]["text"] != quarter_rows[i]["text"] or out_rows[i]["text"] != view_rows[i]["text"]]
    tail_bad = []
    for i in range(PAIR_ROWS, TOTAL_LINES):
        if out_rows[i]["text"] != quarter_rows[i]["text"]:
            tail_bad.append(i)
            if len(tail_bad) >= 10:
                break
    if active_text_bad or tail_bad:
        raise AssertionError({"active_text_bad": active_text_bad[:10], "tail_bad": tail_bad[:10]})

    tenm = OUT_DIR / f"regdisp_{arm_key}_samefw_quarter_10M.jsonl"
    hundredm = OUT_DIR / f"regdisp_{arm_key}_samefw_quarter_100M.jsonl"
    write_jsonl(tenm, out_rows)
    repeat_jsonl(tenm, hundredm)

    # Count 100M cheaply while repeating by construction; still verify rows/words from 10M.
    sha10 = sha256_file(tenm)
    sha100 = sha256_file(hundredm)

    displaced_rows = [clean_meta_pair_rows[i] for i in sorted(displaced)]
    retained_clean_indices = set(range(PAIR_ROWS)) - displaced
    active_clean_reused = sorted(active - displaced)
    hole_positions = sorted(displaced - active)
    moved_words = sum(int(clean_meta_pair_rows[i]["words"]) for i in active_clean_reused)
    out = {
        "arm_key": arm_key,
        "label": arm_info["label"],
        "target_register": arm_info["target_register"],
        "desired_sources": arm_info["desired_sources"],
        "opposed_sources": arm_info["opposed_sources"],
        "file_10m": rel(tenm),
        "file_100m": rel(hundredm),
        "sha_10m": sha10,
        "sha_100m": sha100,
        "rows_10m": rows_n,
        "words_10m": words_n,
        "rows_100m_by_construction": rows_n * REPEATS_100M,
        "words_100m_by_construction": words_n * REPEATS_100M,
        "rho": sum(int(view_meta_pair_rows[i]["words"]) for i in active) / BUDGET_WORDS,
        "active_fineweb_rows": len(active),
        "active_fineweb_words": sum(int(view_meta_pair_rows[i]["words"]) for i in active),
        "displaced_clean_summary": summarize_meta_rows(displaced_rows),
        "retained_clean_changed_block_summary": summarize_meta_rows([clean_meta_pair_rows[i] for i in sorted(retained_clean_indices)]),
        "original_quarter_displaced_clean_summary": summarize_meta_rows([clean_meta_pair_rows[i] for i in sorted(active)]),
        "displaced_overlap_with_quarter_active_rows": len(active & displaced),
        "displaced_not_original_active_rows": len(displaced - active),
        "original_active_clean_rows_relocated": len(active - displaced),
        "relocated_clean_words": moved_words,
        "row_records_for_nonidentity_pair_rows": row_records,
        "row_length_sequence_identical_to_clean": True,
        "fineweb_active_text_identical_to_quarter": True,
        "tail_from_pair_boundary_identical_to_quarter": True,
    }
    (OUT_DIR / f"regdisp_{arm_key}_metadata.json").write_text(json.dumps(out, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")
    return out


def main() -> None:
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    clean_rows = load_jsonl(CLEAN_10M)
    view_rows = load_jsonl(VIEW_10M)
    quarter_rows = load_jsonl(QUARTER_10M)
    clean_meta = load_jsonl(CLEAN_META)
    view_meta = load_jsonl(VIEW_META)
    subdose_meta = json.loads(SUBDOSE_META.read_text())
    assert len(clean_rows) == len(view_rows) == len(quarter_rows) == TOTAL_LINES
    assert len(clean_meta) == len(view_meta) == META_ROWS
    clean_meta_pair_rows = clean_meta[:PAIR_ROWS]
    view_meta_pair_rows = view_meta[:PAIR_ROWS]

    # Verify the base row-length geometry exactly.
    geom_bad = []
    for i in range(TOTAL_LINES):
        cw = int(clean_rows[i].get("words", len(str(clean_rows[i].get("text", "")).split())))
        vw = int(view_rows[i].get("words", len(str(view_rows[i].get("text", "")).split())))
        qw = int(quarter_rows[i].get("words", len(str(quarter_rows[i].get("text", "")).split())))
        if not (cw == vw == qw):
            geom_bad.append((i, cw, vw, qw))
            if len(geom_bad) >= 10:
                break
    if geom_bad:
        raise AssertionError(f"Base row-word sequence mismatch: {geom_bad}")

    active_list = identify_active_rows(clean_rows, view_rows, quarter_rows)
    if len(active_list) != int(subdose_meta["doses"]["quarter_1x"]["active_rows"]):
        raise AssertionError((len(active_list), subdose_meta["doses"]["quarter_1x"]["active_rows"]))
    active_words = sum(int(view_meta_pair_rows[i]["words"]) for i in active_list)
    if active_words != int(subdose_meta["doses"]["quarter_1x"]["active_total_pw"]):
        raise AssertionError((active_words, subdose_meta["doses"]["quarter_1x"]["active_total_pw"]))

    arm_results = []
    for arm_key, arm_info in ARMS.items():
        print(f"Materializing {arm_key}...", flush=True)
        arm_results.append(build_arm_rows(arm_key, arm_info, active_list, clean_rows, view_rows, quarter_rows, clean_meta_pair_rows, view_meta_pair_rows))

    # Cross-arm identity: the active FineWeb block must be identical row-by-row.
    child_rows = load_jsonl(OUT_DIR / "regdisp_childsub_removed_samefw_quarter_10M.jsonl")
    adult_rows = load_jsonl(OUT_DIR / "regdisp_adult_removed_samefw_quarter_10M.jsonl")
    active_mismatch = []
    for i in active_list:
        if child_rows[i]["text"] != adult_rows[i]["text"] or child_rows[i]["text"] != quarter_rows[i]["text"]:
            active_mismatch.append(i)
            if len(active_mismatch) >= 10:
                break
    if active_mismatch:
        raise AssertionError(f"Active FineWeb mismatch: {active_mismatch}")

    summary = {
        "status": "REGISTER_DISPLACEMENT_POOLS_MATERIALIZED",
        "created_utc": now(),
        "purpose": "Same-rho, same-row-geometry pair admitting the identical quarter_1x FineWeb source+compact-view block while displacing contrasting clean registers.",
        "base_files": {
            "clean_10m": rel(CLEAN_10M),
            "view_10m": rel(VIEW_10M),
            "quarter_10m": rel(QUARTER_10M),
            "clean_meta": rel(CLEAN_META),
            "view_meta": rel(VIEW_META),
            "subdose_meta": rel(SUBDOSE_META),
        },
        "geometry": {
            "total_rows_10m": TOTAL_LINES,
            "budget_words_10m": BUDGET_WORDS,
            "pair_rows": PAIR_ROWS,
            "active_fineweb_rows": len(active_list),
            "active_fineweb_words": active_words,
            "rho": active_words / BUDGET_WORDS,
            "same_active_positions_as_subdose_quarter_1x": True,
            "same_active_fineweb_text_across_arms": True,
            "same_row_word_count_sequence_as_MAX_clean_view_quarter": True,
        },
        "arm_results": arm_results,
        "contrast_summary": {
            "childsub_removed_displaced_child_sub_fraction": arm_results[0]["displaced_clean_summary"]["child_sub_fraction"],
            "childsub_removed_displaced_adult_fraction": arm_results[0]["displaced_clean_summary"]["adult_gut_simple_fraction"],
            "adult_removed_displaced_child_sub_fraction": arm_results[1]["displaced_clean_summary"]["child_sub_fraction"],
            "adult_removed_displaced_adult_fraction": arm_results[1]["displaced_clean_summary"]["adult_gut_simple_fraction"],
            "displaced_set_intersection_rows": len(set(json.loads((OUT_DIR / "regdisp_childsub_removed_metadata.json").read_text())["row_records_for_nonidentity_pair_rows"][i].get("omitted_clean_row_index_if_hole") for i in range(0)) if False else []),
        },
        "elapsed_sec": round(time.time() - t0, 3),
        "no_generation_streaming_model_loading_training_eval_upload_or_leaderboard": True,
    }
    # Easier direct intersection from metadata source rows requires recomputation.
    # Recompute displaced sets for summary fields.
    target_hist = collections.Counter(int(view_meta_pair_rows[i]["words"]) for i in active_list)
    child_D = choose_displaced_set(clean_meta_pair_rows, target_hist, CHILD_SUB, ADULT)
    adult_D = choose_displaced_set(clean_meta_pair_rows, target_hist, ADULT, CHILD_SUB)
    summary["contrast_summary"]["displaced_set_intersection_rows"] = len(child_D & adult_D)
    summary["contrast_summary"]["displaced_set_union_rows"] = len(child_D | adult_D)
    summary["contrast_summary"]["displaced_set_jaccard"] = len(child_D & adult_D) / len(child_D | adult_D)

    (OUT_DIR / "register_displacement_pools_metadata.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")

    md = [
        "# research register-contrast displacement pools",
        "",
        "CPU/file-only materialization. No generation, streaming, model loading, training, official evaluation, upload, or leaderboard action.",
        "",
        "## Scientific contrast",
        "Both arms admit exactly the same quarter_1x FineWeb source+compact-view rows at the same row positions (758 rows, 105,962 words, rho=0.0105962). They preserve the MAX-row 65,313-row / 10M-word geometry and differ in which clean rows are absent.",
        "",
        "## Arm summaries",
    ]
    for ar in arm_results:
        ds = ar["displaced_clean_summary"]
        md += [
            f"### {ar['arm_key']}",
            f"10M: `{ar['file_10m']}`",
            f"100M: `{ar['file_100m']}`",
            f"SHA10: `{ar['sha_10m']}`",
            f"SHA100: `{ar['sha_100m']}`",
            f"Displaced rows/words: {ds['n_rows']} / {ds['words']}",
            f"Displaced composition: child+subtitle {ds['child_sub_words']} ({ds['child_sub_fraction']:.6f}), Gutenberg+SimpleWiki {ds['adult_gut_simple_words']} ({ds['adult_gut_simple_fraction']:.6f}), other {ds['other_words']}",
            f"Overlap with original quarter active clean rows: {ar['displaced_overlap_with_quarter_active_rows']}; necessary relocated clean rows: {ar['original_active_clean_rows_relocated']} ({ar['relocated_clean_words']} words)",
            "",
        ]
    md += [
        "## Interpretation use",
        "Train these two arms only after the current sub-dose/clean-anchor curve indicates that rho≈0.011 is a meaningful region. If their stable-family V-C curves land together, the evidence supports an admission-of-second-register mixture effect rather than value-of-removed-register; if they diverge, the clean material removed is itself part of the principle.",
        "",
        "Metadata: `experiments/archive/frontier_consolidation/data/register_displacement_pools/register_displacement_pools_metadata.json`",
    ]
    (OUT_DIR / "register_displacement_pools_summary.md").write_text("\n".join(md)+"\n", encoding="utf-8")
    print(json.dumps({
        "status": summary["status"],
        "summary": rel(OUT_DIR / "register_displacement_pools_summary.md"),
        "metadata": rel(OUT_DIR / "register_displacement_pools_metadata.json"),
        "arms": [{"arm": ar["arm_key"], "file_100m": ar["file_100m"], "words_100m": ar["words_100m_by_construction"], "sha_100m": ar["sha_100m"], "displaced_child_frac": ar["displaced_clean_summary"]["child_sub_fraction"], "displaced_adult_frac": ar["displaced_clean_summary"]["adult_gut_simple_fraction"]} for ar in arm_results],
        "elapsed_sec": summary["elapsed_sec"],
    }, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
