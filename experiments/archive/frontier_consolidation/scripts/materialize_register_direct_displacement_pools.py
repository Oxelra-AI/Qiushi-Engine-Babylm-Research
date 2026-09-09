#!/usr/bin/env python3
"""research: Materialize direct register-displacement pools.

This is the primary same-rho register instrument for the next H100 pair.
It admits the identical quarter_1x FineWeb source+compact-view row multiset but
places those FineWeb rows directly onto clean row positions selected to remove
contrasting registers:

  * childsub_direct: replace CHILDES/OpenSubtitles-dominant clean rows;
  * adult_direct:    replace Gutenberg/SimpleWiki-dominant clean rows.

Controls preserved:
  * exact 65,313-row MAX geometry and 10,000,000 words per 10M pool;
  * exact row word-count sequence at every line;
  * identical admitted FineWeb text multiset and word-count multiset across arms;
  * no clean row relocation: unselected clean rows remain at their original row.

This differs from materialize_register_displacement_pools.py, which held
FineWeb positions fixed and relocated clean rows. The direct version is the
cleaner discriminator for whether the effect depends on the register removed.
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
OUT_DIR = WS / "data/register_direct_displacement_pools"

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
    ("childsub_direct", {
        "label": "quarter_rho_samefw_direct_childes_opensubtitles_removed",
        "target_register": "directly replaced CHILDES + OpenSubtitles clean rows",
        "desired_sources": sorted(CHILD_SUB),
        "opposed_sources": sorted(ADULT),
    }),
    ("adult_direct", {
        "label": "quarter_rho_samefw_direct_gutenberg_simplewiki_removed",
        "target_register": "directly replaced Gutenberg + SimpleWiki clean rows",
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
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


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


def row_words(r: dict[str, Any]) -> int:
    return int(r.get("words", len(str(r.get("text", "")).split())))


def word_count_rows(rows: list[dict[str, Any]]) -> tuple[int, int]:
    return len(rows), sum(row_words(r) for r in rows)


def comp_words(meta: dict[str, Any], sources: set[str]) -> int:
    comps = meta.get("component_sources", {}) or {}
    return sum(int(comps.get(k, 0)) for k in sources)


def score_for(meta: dict[str, Any], desired: set[str], opposed: set[str]) -> tuple[float, int, int, int]:
    w = int(meta.get("words", 0))
    d = comp_words(meta, desired)
    o = comp_words(meta, opposed)
    return ((d - o) / max(1, w), d, -o, -int(meta.get("row_index", 0)))


def summarize_meta_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    lengths = [int(r.get("words", 0)) for r in rows]
    comps = collections.Counter()
    for r in rows:
        comps.update({k: int(v) for k, v in (r.get("component_sources", {}) or {}).items()})
    words = sum(lengths)
    child = sum(comps.get(k, 0) for k in CHILD_SUB)
    adult = sum(comps.get(k, 0) for k in ADULT)
    idxs = [int(r.get("row_index", -1)) for r in rows]
    bins = collections.Counter()
    for i in idxs:
        bins[f"{(i//1000)*1000:04d}-{(i//1000)*1000+999:04d}"] += 1
    return {
        "n_rows": len(rows),
        "words": words,
        "mean_words": statistics.mean(lengths) if lengths else None,
        "min_words": min(lengths) if lengths else None,
        "max_words": max(lengths) if lengths else None,
        "row_index_min": min(idxs) if idxs else None,
        "row_index_max": max(idxs) if idxs else None,
        "row_index_mean": statistics.mean(idxs) if idxs else None,
        "row_index_bins_1000": dict(sorted(bins.items())),
        "component_words": dict(comps),
        "child_sub_words": child,
        "adult_gut_simple_words": adult,
        "other_words": words - child - adult,
        "child_sub_fraction": child / words if words else 0.0,
        "adult_gut_simple_fraction": adult / words if words else 0.0,
    }


def identify_quarter_active_rows(clean_rows: list[dict[str, Any]], view_rows: list[dict[str, Any]], quarter_rows: list[dict[str, Any]]) -> list[int]:
    active = []
    bad = []
    for i in range(PAIR_ROWS):
        if quarter_rows[i]["text"] == view_rows[i]["text"]:
            active.append(i)
        elif quarter_rows[i]["text"] != clean_rows[i]["text"]:
            bad.append(i)
            if len(bad) >= 10:
                break
    if bad:
        raise AssertionError(f"Quarter row is neither clean nor view at {bad}")
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
    if collections.Counter(int(clean_meta_pair_rows[i]["words"]) for i in selected) != target_hist:
        raise AssertionError("Selected length multiset mismatch")
    return selected


def assign_fineweb_to_positions(active_rows: list[int], target_positions: set[int], view_meta_pair_rows: list[dict[str, Any]], clean_meta_pair_rows: list[dict[str, Any]]) -> dict[int, int]:
    """Return position -> original quarter FineWeb row index, matched by exact word length."""
    fine_by_len: dict[int, list[int]] = collections.defaultdict(list)
    pos_by_len: dict[int, list[int]] = collections.defaultdict(list)
    for i in active_rows:
        fine_by_len[int(view_meta_pair_rows[i]["words"])].append(i)
    for j in sorted(target_positions):
        pos_by_len[int(clean_meta_pair_rows[j]["words"])].append(j)
    if {k: len(v) for k, v in fine_by_len.items()} != {k: len(v) for k, v in pos_by_len.items()}:
        raise AssertionError({"fine": {k: len(v) for k, v in fine_by_len.items()}, "pos": {k: len(v) for k, v in pos_by_len.items()}})
    assign: dict[int, int] = {}
    for w in sorted(fine_by_len):
        # Preserve the original FineWeb order within each row length; positions are row-order sorted.
        for pos, src in zip(pos_by_len[w], fine_by_len[w]):
            assign[pos] = src
    return assign


def multiset_hash_texts(rows: list[dict[str, Any]]) -> str:
    h = hashlib.sha256()
    for txt in sorted(str(r.get("text", "")) for r in rows):
        h.update(txt.encode("utf-8"))
        h.update(b"\0")
    return h.hexdigest()


def build_arm(arm_key: str, arm_info: dict[str, Any], active_rows: list[int], clean_rows: list[dict[str, Any]], view_rows: list[dict[str, Any]], quarter_rows: list[dict[str, Any]], clean_meta_pair_rows: list[dict[str, Any]], view_meta_pair_rows: list[dict[str, Any]]) -> dict[str, Any]:
    target_hist = collections.Counter(int(view_meta_pair_rows[i]["words"]) for i in active_rows)
    desired = set(arm_info["desired_sources"])
    opposed = set(arm_info["opposed_sources"])
    displaced = choose_displaced_set(clean_meta_pair_rows, target_hist, desired, opposed)
    assign = assign_fineweb_to_positions(active_rows, displaced, view_meta_pair_rows, clean_meta_pair_rows)

    out_rows: list[dict[str, Any]] = []
    row_records = []
    for i in range(TOTAL_LINES):
        if i in assign:
            src = assign[i]
            r = dict(view_rows[src])
            r["regdisp_arm"] = arm_key
            r["regdisp_row_kind"] = "direct_fineweb_replaces_selected_clean_row"
            r["regdisp_output_row_index"] = i
            r["regdisp_original_fineweb_row_index"] = src
            r["regdisp_replaced_clean_row_index"] = i
            out_rows.append(r)
            row_records.append({
                "output_row_index": i,
                "original_fineweb_row_index": src,
                "words": row_words(r),
                "replaced_clean_component_sources": clean_meta_pair_rows[i].get("component_sources", {}),
            })
        else:
            r = dict(clean_rows[i])
            r["regdisp_arm"] = arm_key
            r["regdisp_row_kind"] = "original_clean_or_shared_tail"
            r["regdisp_output_row_index"] = i
            out_rows.append(r)

    # Row length geometry.
    bad = []
    for i, r in enumerate(out_rows):
        rw = row_words(r)
        cw = row_words(clean_rows[i])
        if rw != cw:
            bad.append((i, rw, cw))
            if len(bad) >= 10:
                break
    if bad:
        raise AssertionError(f"Row length mismatches: {bad}")
    n, words = word_count_rows(out_rows)
    assert n == TOTAL_LINES and words == BUDGET_WORDS, (n, words)

    active_inserted_rows = [out_rows[i] for i in sorted(displaced)]
    original_fineweb_rows = [view_rows[i] for i in active_rows]
    if multiset_hash_texts(active_inserted_rows) != multiset_hash_texts(original_fineweb_rows):
        raise AssertionError("FineWeb text multiset differs from quarter active block")

    tenm = OUT_DIR / f"regdisp_direct_{arm_key}_samefw_quarter_10M.jsonl"
    hundredm = OUT_DIR / f"regdisp_direct_{arm_key}_samefw_quarter_100M.jsonl"
    write_jsonl(tenm, out_rows)
    repeat_jsonl(tenm, hundredm)
    sha10 = sha256_file(tenm)
    sha100 = sha256_file(hundredm)

    displaced_rows = [clean_meta_pair_rows[i] for i in sorted(displaced)]
    displaced_summary = summarize_meta_rows(displaced_rows)
    result = {
        "arm_key": arm_key,
        "label": arm_info["label"],
        "target_register": arm_info["target_register"],
        "desired_sources": arm_info["desired_sources"],
        "opposed_sources": arm_info["opposed_sources"],
        "file_10m": rel(tenm),
        "file_100m": rel(hundredm),
        "sha_10m": sha10,
        "sha_100m": sha100,
        "rows_10m": n,
        "words_10m": words,
        "rows_100m_by_construction": n * REPEATS_100M,
        "words_100m_by_construction": words * REPEATS_100M,
        "rho": sum(int(view_meta_pair_rows[i]["words"]) for i in active_rows) / BUDGET_WORDS,
        "active_fineweb_rows": len(active_rows),
        "active_fineweb_words": sum(int(view_meta_pair_rows[i]["words"]) for i in active_rows),
        "active_fineweb_text_multiset_hash": multiset_hash_texts(original_fineweb_rows),
        "inserted_fineweb_text_multiset_hash": multiset_hash_texts(active_inserted_rows),
        "displaced_clean_summary": displaced_summary,
        "original_quarter_displaced_clean_summary": summarize_meta_rows([clean_meta_pair_rows[i] for i in active_rows]),
        "direct_replacement_row_records": row_records,
        "no_clean_row_relocation": True,
        "unselected_rows_equal_original_clean_text": True,
        "row_word_count_sequence_identical_to_max_clean": True,
    }
    (OUT_DIR / f"regdisp_direct_{arm_key}_metadata.json").write_text(json.dumps(result, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")
    return result


def main() -> None:
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    clean_rows = load_jsonl(CLEAN_10M)
    view_rows = load_jsonl(VIEW_10M)
    quarter_rows = load_jsonl(QUARTER_10M)
    clean_meta = load_jsonl(CLEAN_META)
    view_meta = load_jsonl(VIEW_META)
    submeta = json.loads(SUBDOSE_META.read_text())
    assert len(clean_rows) == len(view_rows) == len(quarter_rows) == TOTAL_LINES
    assert len(clean_meta) == len(view_meta) == META_ROWS
    clean_meta_pair_rows = clean_meta[:PAIR_ROWS]
    view_meta_pair_rows = view_meta[:PAIR_ROWS]

    # Base geometry and quarter active block.
    for i in range(TOTAL_LINES):
        cw, vw, qw = row_words(clean_rows[i]), row_words(view_rows[i]), row_words(quarter_rows[i])
        if not (cw == vw == qw):
            raise AssertionError(f"Base row words differ at {i}: clean={cw}, view={vw}, quarter={qw}")
    active_rows = identify_quarter_active_rows(clean_rows, view_rows, quarter_rows)
    expected_active = int(submeta["doses"]["quarter_1x"]["active_rows"])
    expected_words = int(submeta["doses"]["quarter_1x"]["active_total_pw"])
    active_words = sum(int(view_meta_pair_rows[i]["words"]) for i in active_rows)
    assert len(active_rows) == expected_active and active_words == expected_words

    results = []
    for arm_key, arm_info in ARMS.items():
        print(f"Materializing direct {arm_key}...", flush=True)
        results.append(build_arm(arm_key, arm_info, active_rows, clean_rows, view_rows, quarter_rows, clean_meta_pair_rows, view_meta_pair_rows))

    # Cross-arm checks.
    child_D = set(rr["output_row_index"] for rr in results[0]["direct_replacement_row_records"])
    adult_D = set(rr["output_row_index"] for rr in results[1]["direct_replacement_row_records"])
    if results[0]["active_fineweb_text_multiset_hash"] != results[1]["active_fineweb_text_multiset_hash"]:
        raise AssertionError("FineWeb active text multiset hashes differ across arms")
    summary = {
        "status": "REGISTER_DIRECT_DISPLACEMENT_POOLS_MATERIALIZED",
        "created_utc": now(),
        "purpose": "Primary same-rho clean-register displacement pair: identical quarter_1x FineWeb block, direct replacement of contrasting clean-register rows, no clean row relocation.",
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
            "active_fineweb_rows": len(active_rows),
            "active_fineweb_words": active_words,
            "rho": active_words / BUDGET_WORDS,
            "same_row_word_count_sequence_as_MAX_clean_view_quarter": True,
            "same_fineweb_text_multiset_across_arms": True,
            "fineweb_source_multiset": "quarter_1x active rows from research subdose ladder",
        },
        "arm_results": results,
        "contrast_summary": {
            "displaced_set_intersection_rows": len(child_D & adult_D),
            "displaced_set_union_rows": len(child_D | adult_D),
            "displaced_set_jaccard": len(child_D & adult_D) / len(child_D | adult_D),
            "childsub_direct_displaced_child_sub_fraction": results[0]["displaced_clean_summary"]["child_sub_fraction"],
            "childsub_direct_displaced_adult_fraction": results[0]["displaced_clean_summary"]["adult_gut_simple_fraction"],
            "adult_direct_displaced_child_sub_fraction": results[1]["displaced_clean_summary"]["child_sub_fraction"],
            "adult_direct_displaced_adult_fraction": results[1]["displaced_clean_summary"]["adult_gut_simple_fraction"],
        },
        "elapsed_sec": round(time.time() - t0, 3),
        "no_generation_streaming_model_loading_training_eval_upload_or_leaderboard": True,
    }
    (OUT_DIR / "register_direct_displacement_pools_metadata.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")

    md = [
        "# research direct register-displacement pools",
        "",
        "CPU/file-only materialization. No generation, streaming, model loading, training, official evaluation, upload, or leaderboard action.",
        "",
        "## Scientific contrast",
        "Both arms admit the same quarter_1x FineWeb source+compact-view text multiset (758 rows, 105,962 words, rho=0.0105962). Each FineWeb row is placed onto a selected clean row of exactly the same word count. Unselected rows keep their original clean text at their original positions. Thus the intended changing variable is the register of clean material directly replaced.",
        "",
        "## Arm summaries",
    ]
    for ar in results:
        ds = ar["displaced_clean_summary"]
        md += [
            f"### {ar['arm_key']}",
            f"10M: `{ar['file_10m']}`",
            f"100M: `{ar['file_100m']}`",
            f"SHA10: `{ar['sha_10m']}`",
            f"SHA100: `{ar['sha_100m']}`",
            f"Displaced rows/words: {ds['n_rows']} / {ds['words']}",
            f"Displaced composition: child+subtitle {ds['child_sub_words']} ({ds['child_sub_fraction']:.6f}), Gutenberg+SimpleWiki {ds['adult_gut_simple_words']} ({ds['adult_gut_simple_fraction']:.6f}), other {ds['other_words']}",
            f"Displaced row-index span/mean: {ds['row_index_min']}..{ds['row_index_max']} / {ds['row_index_mean']:.1f}",
            "",
        ]
    md += [
        "## Relationship to fixed-position auxiliary pair",
        "The earlier `register_displacement_pools/` pair holds FineWeb positions fixed and relocates clean rows to alter the omitted set. This direct pair should be the primary H100 discriminator because it directly replaces the selected clean register and does not relocate unselected clean rows.",
        "",
        "## Interpretation use",
        "Train only when the sub-dose/clean-anchor curve makes rho≈0.011 worth resolving. If childsub_direct and adult_direct land together against the same clean anchor, the effect is tied to admitting a second register rather than to the register removed. If they diverge, the sacrificed clean register is part of the data-efficient learning law at this scale.",
        "",
        "Metadata: `experiments/archive/frontier_consolidation/data/register_direct_displacement_pools/register_direct_displacement_pools_metadata.json`",
    ]
    (OUT_DIR / "register_direct_displacement_pools_summary.md").write_text("\n".join(md)+"\n", encoding="utf-8")
    print(json.dumps({
        "status": summary["status"],
        "summary": rel(OUT_DIR / "register_direct_displacement_pools_summary.md"),
        "metadata": rel(OUT_DIR / "register_direct_displacement_pools_metadata.json"),
        "arms": [{
            "arm": ar["arm_key"],
            "file_100m": ar["file_100m"],
            "words_100m": ar["words_100m_by_construction"],
            "sha_100m": ar["sha_100m"],
            "displaced_child_frac": ar["displaced_clean_summary"]["child_sub_fraction"],
            "displaced_adult_frac": ar["displaced_clean_summary"]["adult_gut_simple_fraction"],
            "row_index_mean": ar["displaced_clean_summary"]["row_index_mean"],
        } for ar in results],
        "elapsed_sec": summary["elapsed_sec"],
    }, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
