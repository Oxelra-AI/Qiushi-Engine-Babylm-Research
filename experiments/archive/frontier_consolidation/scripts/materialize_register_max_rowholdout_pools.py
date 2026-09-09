#!/usr/bin/env python3
"""research: materialize MAX-dose register-skewed row-holdout arms.

Scientific role
---------------
The quarter-rho register contrast prepared in research has weak resolution unless
subdose evidence shows sharp saturation below 1% of the budget. This script
builds the high-power MAX-rho counterpart from existing data only: both arms
admit the identical research MAX compact-view FineWeb block, but the clean rows
omitted from the 10M budget are chosen from contrasting registers.

Design
------
* changed block geometry is the existing MAX view geometry: 7,923 FineWeb rows
  plus one 133-word neutral clean topup row, followed by the same suffix length
  sequence as research MAX.
* childspeech arm omits 6,992 non-qwen 160-word rows drawn only from
  CHILDES/OpenSubtitles/BNC/Switchboard.
* adultprose arm omits 6,992 non-qwen 160-word rows drawn only from
  Gutenberg/SimpleWiki.
* selections are nearest to the original research proportional MAX heldout row
  indices, so the new skewed arms bracket the trained proportional MAX view arm
  in the same base-row coordinate.
* qwen_pair_packed rows are never selected for omission and are kept at the same
  suffix positions as the research common filler template.

No generation, no streaming, no model loading, no training, no evaluation, no
upload, and no leaderboard action.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative

import bisect, collections, hashlib, json, pathlib, random, statistics, time
from typing import Any


def find_user_root() -> pathlib.Path:
    return _PUBLIC_ROOT

ROOT = find_user_root()
WS = ROOT / "experiments/archive/frontier_consolidation"
BASE_POOL = ROOT / "experiments/archive/compact_experience/data/qwen_clean_aligned/training_corpora/qwen_aligned_10M.jsonl"
POOL256 = WS / "data/dose_2p64x_rowholdout_pools"
OUT = WS / "data/register_max_rowholdout_pools"
VIEW10 = POOL256 / "compact_view_dose2p64x_10M.jsonl"
VIEW100 = POOL256 / "compact_view_dose2p64x_100M.jsonl"
ORIG_HELDOUT = POOL256 / "heldout_cleanqwen_rows.jsonl"
COMMON_FILLER_TEMPLATE = POOL256 / "common_filler_rows.jsonl"
META256 = POOL256 / "dose2p64x_rowholdout_metadata.json"
PAIR_ROWS = 7923
TOPUP_WORDS = 133
TOTAL_WORDS = 10_000_000
TOTAL_ROWS = 65313
HELDOUT_ROWS = 6992
MAX_PACKET_WORDS = 160
PASSES = 10
RNG_SEED = 82914124
DEV_SPEECH = {"childes", "open_subtitles", "bnc_spoken", "switchboard"}
ADULT_PROSE = {"gutenberg", "simple_wiki"}
PROTECTED = "qwen_pair_packed"


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def rel(p: pathlib.Path) -> str:
    try:
        return str(p.relative_to(ROOT))
    except Exception:
        return str(p)


def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def iter_jsonl(path: pathlib.Path):
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def load_jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
    return list(iter_jsonl(path))


def write_jsonl(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, separators=(",", ":")) + "\n")


def repeat_jsonl(src: pathlib.Path, dst: pathlib.Path, repeats: int = PASSES) -> None:
    with dst.open("w", encoding="utf-8") as out:
        for _ in range(repeats):
            with src.open(encoding="utf-8") as inp:
                for line in inp:
                    out.write(line)


def nwords(row: dict[str, Any]) -> int:
    return int(row.get("words", len(str(row.get("text", "")).split())))


def norm_source(row: dict[str, Any]) -> str:
    return str(row.get("source", row.get("source_name", "unknown"))).split("::")[-1]


def base_index(rows: list[dict[str, Any]]) -> dict[int, int]:
    out = {}
    for i, r in enumerate(rows):
        ex = int(r.get("example_id") if r.get("example_id") is not None else i)
        if ex in out:
            raise AssertionError(f"duplicate example_id {ex}")
        out[ex] = i
    return out


def row_record_from_base(row: dict[str, Any], *, arm: str, kind: str, output_i: int | None = None) -> dict[str, Any]:
    out = {
        "text": " ".join(str(row.get("text", "")).split()),
        "words": nwords(row),
        "example_id": row.get("example_id"),
        "source": row.get("source", row.get("source_name", "unknown")),
        "regmax_arm": arm,
        "regmax_row_kind": kind,
    }
    if output_i is not None:
        out["regmax_output_row_index"] = output_i
    return out


def choose_nearest(targets: list[int], candidates: list[int]) -> tuple[list[int], list[dict[str, Any]]]:
    """Choose unique candidate indices nearest to each target base index."""
    pool = sorted(candidates)
    chosen: list[int] = []
    records: list[dict[str, Any]] = []
    for t in sorted(targets):
        if not pool:
            raise RuntimeError("candidate pool exhausted")
        pos = bisect.bisect_left(pool, t)
        best_j = None
        best_key = None
        for j in [pos - 2, pos - 1, pos, pos + 1, pos + 2]:
            if 0 <= j < len(pool):
                cand = pool[j]
                key = (abs(cand - t), cand)
                if best_key is None or key < best_key:
                    best_key = key
                    best_j = j
        assert best_j is not None and best_key is not None
        cand = pool.pop(best_j)
        chosen.append(cand)
        records.append({"target_original_heldout_base_index": t, "chosen_base_index": cand, "abs_distance": abs(cand - t)})
    return chosen, records


def stream_words_from_rows(rows: list[dict[str, Any]]) -> list[tuple[str, str, int]]:
    stream: list[tuple[str, str, int]] = []
    for r in rows:
        src = norm_source(r)
        ex = int(r.get("example_id") if r.get("example_id") is not None else -1)
        for w in str(r.get("text", "")).split():
            stream.append((w, src, ex))
    return stream


def make_topup(chosen_rows: list[dict[str, Any]], arm: str) -> dict[str, Any]:
    # Match the research construction: the small clean topup is drawn from the
    # selected clean holdout stream. It is only 133 words, is audited separately,
    # and is not part of the admitted FineWeb block.
    rng = random.Random(RNG_SEED)
    order = list(chosen_rows)
    rng.shuffle(order)
    stream = stream_words_from_rows(order)
    if len(stream) < TOPUP_WORDS:
        raise RuntimeError("holdout stream too short for topup")
    seg = stream[:TOPUP_WORDS]
    comps = collections.Counter(src for _, src, _ in seg)
    return {
        "text": " ".join(w for w, _, _ in seg),
        "words": TOPUP_WORDS,
        "example_id": 2_840_000 + (0 if arm == "childspeech" else 1),
        "source": f"regmax_{arm}_clean_topup::{comps.most_common(1)[0][0]}",
        "component_sources": dict(comps),
        "regmax_arm": arm,
        "regmax_row_kind": "clean_topup_from_selected_holdout",
        "regmax_output_row_index": PAIR_ROWS,
    }


def text_hash(row: dict[str, Any]) -> str:
    h = hashlib.sha1()
    h.update(str(row.get("text", "")).encode("utf-8"))
    h.update(b"\0")
    h.update(str(row.get("source", "")).encode("utf-8"))
    h.update(b"\0")
    h.update(str(row.get("example_id", "")).encode("utf-8"))
    return h.hexdigest()


def fill_suffix_from_template(base_rows: list[dict[str, Any]], selected_set: set[int], template: list[dict[str, Any]], arm: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Use research suffix as a length/qwen-position template.

    Qwen rows are kept exact at their template positions. Every non-qwen suffix
    position has length 160 and is filled with one remaining non-qwen base row.
    This uses all remaining base rows exactly once and preserves the suffix row
    word-count sequence across both register arms and the trained MAX arms.
    """
    ex_to_base = {int(r.get("example_id") if r.get("example_id") is not None else i): i for i, r in enumerate(base_rows)}
    remaining_nonqwen: dict[int, dict[str, Any]] = {}
    remaining_qwen: dict[int, dict[str, Any]] = {}
    for i, r in enumerate(base_rows):
        src = norm_source(r)
        if i in selected_set:
            continue
        ex = int(r.get("example_id") if r.get("example_id") is not None else i)
        if src == PROTECTED:
            remaining_qwen[ex] = r
        else:
            if nwords(r) != MAX_PACKET_WORDS:
                raise AssertionError(("unexpected nonqwen length", i, src, nwords(r)))
            remaining_nonqwen[ex] = r
    unused_nonqwen = dict(remaining_nonqwen)
    # Deterministic backfill order close to research common filler: first use rows
    # as they appear in the template if they remain; then use remaining rows in
    # a stable hash order. All non-qwen rows are length 160, so any fill preserves
    # the row-length sequence.
    stable_extra = sorted(unused_nonqwen, key=lambda ex: hashlib.sha1(f"{arm}|{ex}".encode()).hexdigest())
    extra_cursor = 0
    suffix: list[dict[str, Any]] = []
    qwen_positions = []
    template_kept_nonqwen = 0
    backfilled_nonqwen = 0
    for j, tr in enumerate(template):
        out_i = PAIR_ROWS + 1 + j
        src = norm_source(tr)
        ex = int(tr.get("example_id") if tr.get("example_id") is not None else -999999)
        if src == PROTECTED:
            if ex not in remaining_qwen:
                raise AssertionError(f"protected qwen template row missing exid={ex}")
            rec = row_record_from_base(remaining_qwen[ex], arm=arm, kind="protected_qwen_template_position", output_i=out_i)
            qwen_positions.append(out_i)
            suffix.append(rec)
        else:
            if ex in unused_nonqwen:
                br = unused_nonqwen.pop(ex)
                template_kept_nonqwen += 1
            else:
                while extra_cursor < len(stable_extra) and stable_extra[extra_cursor] not in unused_nonqwen:
                    extra_cursor += 1
                if extra_cursor >= len(stable_extra):
                    raise RuntimeError("no nonqwen row left for backfill")
                bx = stable_extra[extra_cursor]
                br = unused_nonqwen.pop(bx)
                backfilled_nonqwen += 1
            rec = row_record_from_base(br, arm=arm, kind="remaining_nonqwen_suffix", output_i=out_i)
            suffix.append(rec)
    if unused_nonqwen:
        raise AssertionError(f"unused nonqwen rows after suffix fill: {len(unused_nonqwen)}")
    if len(qwen_positions) != len(remaining_qwen):
        raise AssertionError(("qwen position count mismatch", len(qwen_positions), len(remaining_qwen)))
    return suffix, {
        "suffix_rows": len(suffix),
        "qwen_template_positions": len(qwen_positions),
        "template_kept_nonqwen_rows": template_kept_nonqwen,
        "backfilled_nonqwen_rows": backfilled_nonqwen,
        "all_remaining_nonqwen_used": True,
        "all_qwen_rows_kept_at_template_positions": True,
    }


def summarize_indices(indices: list[int], base_rows: list[dict[str, Any]]) -> dict[str, Any]:
    comps = collections.Counter()
    for i in indices:
        comps[norm_source(base_rows[i])] += nwords(base_rows[i])
    total = sum(comps.values())
    dev = sum(comps.get(s, 0) for s in DEV_SPEECH)
    adult = sum(comps.get(s, 0) for s in ADULT_PROSE)
    return {
        "rows": len(indices),
        "words": total,
        "source_words": dict(comps.most_common()),
        "dev_speech_words": dev,
        "adult_prose_words": adult,
        "other_words": total - dev - adult,
        "dev_speech_fraction": dev / total if total else None,
        "adult_prose_fraction": adult / total if total else None,
        "row_index_min": min(indices) if indices else None,
        "row_index_max": max(indices) if indices else None,
        "row_index_mean": statistics.mean(indices) if indices else None,
        "row_index_sd": statistics.pstdev(indices) if len(indices) > 1 else 0.0,
    }


def summarize_pool(rows: list[dict[str, Any]]) -> dict[str, Any]:
    comps = collections.Counter()
    kinds = collections.Counter()
    for r in rows:
        comps[norm_source(r)] += nwords(r)
        kinds[str(r.get("regmax_row_kind", "unmarked"))] += 1
    return {"rows": len(rows), "words": sum(nwords(r) for r in rows), "source_words": dict(comps.most_common()), "row_kinds": dict(kinds.most_common())}


def make_arm(arm: str, selected_indices: list[int], base_rows: list[dict[str, Any]], view_rows: list[dict[str, Any]], template: list[dict[str, Any]]) -> dict[str, Any]:
    selected_set = set(selected_indices)
    chosen_rows = [base_rows[i] for i in selected_indices]
    topup = make_topup(chosen_rows, arm)
    fine_rows = []
    for i in range(PAIR_ROWS):
        r = dict(view_rows[i])
        r["regmax_arm"] = arm
        r["regmax_row_kind"] = "identical_max_fineweb_compact_view"
        r["regmax_output_row_index"] = i
        fine_rows.append(r)
    suffix, suffix_summary = fill_suffix_from_template(base_rows, selected_set, template, arm)
    out_rows = fine_rows + [topup] + suffix
    assert len(out_rows) == TOTAL_ROWS
    assert sum(nwords(r) for r in out_rows) == TOTAL_WORDS
    # Same row word-count sequence as the existing MAX view pool.
    if [nwords(r) for r in out_rows] != [nwords(r) for r in view_rows]:
        raise AssertionError("new arm row word-count sequence differs from MAX view")
    tenm = OUT / f"regmax_{arm}_samefw_10M.jsonl"
    hundredm = OUT / f"regmax_{arm}_samefw_100M.jsonl"
    write_jsonl(tenm, out_rows)
    repeat_jsonl(tenm, hundredm)
    return {
        "arm": arm,
        "file_10m": rel(tenm),
        "file_100m": rel(hundredm),
        "sha_10m": sha256_file(tenm),
        "sha_100m": sha256_file(hundredm),
        "selected_holdout_summary": summarize_indices(selected_indices, base_rows),
        "topup_summary": {"words": TOPUP_WORDS, "component_sources": topup.get("component_sources", {}), "source": topup.get("source")},
        "suffix_summary": suffix_summary,
        "pool_summary": summarize_pool(out_rows),
    }


def main() -> None:
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    base_rows = load_jsonl(BASE_POOL)
    view_rows = load_jsonl(VIEW10)
    orig_heldout = load_jsonl(ORIG_HELDOUT)
    template = load_jsonl(COMMON_FILLER_TEMPLATE)
    assert len(view_rows) == TOTAL_ROWS
    assert len(template) == TOTAL_ROWS - PAIR_ROWS - 1
    assert sum(nwords(r) for r in base_rows) == TOTAL_WORDS
    assert sum(nwords(r) for r in view_rows) == TOTAL_WORDS
    meta256 = json.loads(META256.read_text(encoding="utf-8"))
    selected_pair_words = int(meta256["dose"]["selected_pair_words"])
    changed_block_budget = int(meta256["dose"]["changed_block_budget_words"])
    assert selected_pair_words == sum(nwords(view_rows[i]) for i in range(PAIR_ROWS))
    assert changed_block_budget == selected_pair_words + TOPUP_WORDS
    assert changed_block_budget // MAX_PACKET_WORDS == HELDOUT_ROWS
    idx_by_ex = base_index(base_rows)
    targets = []
    for r in orig_heldout:
        ex = int(r["example_id"])
        targets.append(idx_by_ex[ex])
    assert len(targets) == HELDOUT_ROWS
    dev_candidates = [i for i, r in enumerate(base_rows) if norm_source(r) in DEV_SPEECH and nwords(r) == MAX_PACKET_WORDS]
    adult_candidates = [i for i, r in enumerate(base_rows) if norm_source(r) in ADULT_PROSE and nwords(r) == MAX_PACKET_WORDS]
    if len(dev_candidates) < HELDOUT_ROWS or len(adult_candidates) < HELDOUT_ROWS:
        raise RuntimeError({"dev_candidates": len(dev_candidates), "adult_candidates": len(adult_candidates), "need": HELDOUT_ROWS})
    child_idx, child_match = choose_nearest(targets, dev_candidates)
    adult_idx, adult_match = choose_nearest(targets, adult_candidates)
    # Check selections are pure, non-overlapping, and qwen-free.
    for name, idxs, group in [("childspeech", child_idx, DEV_SPEECH), ("adultprose", adult_idx, ADULT_PROSE)]:
        if len(idxs) != len(set(idxs)) or len(idxs) != HELDOUT_ROWS:
            raise AssertionError((name, len(idxs), len(set(idxs))))
        bad = [i for i in idxs if norm_source(base_rows[i]) not in group or nwords(base_rows[i]) != MAX_PACKET_WORDS]
        if bad:
            raise AssertionError((name, bad[:5]))
    child_arm = make_arm("childspeech", child_idx, base_rows, view_rows, template)
    adult_arm = make_arm("adultprose", adult_idx, base_rows, view_rows, template)
    d_child = [r["abs_distance"] for r in child_match]
    d_adult = [r["abs_distance"] for r in adult_match]
    pair_dist = [abs(c - a) for c, a in zip(sorted(child_idx), sorted(adult_idx))]
    geom = {
        "total_rows_10m": TOTAL_ROWS,
        "total_words_10m": TOTAL_WORDS,
        "pair_rows": PAIR_ROWS,
        "selected_pair_words": selected_pair_words,
        "topup_words": TOPUP_WORDS,
        "rho_pair_words": selected_pair_words / TOTAL_WORDS,
        "heldout_rows_per_arm": HELDOUT_ROWS,
        "heldout_budget_words_per_arm": HELDOUT_ROWS * MAX_PACKET_WORDS,
        "row_word_count_sequence_identical_to_step256_max_view": True,
        "fineweb_pair_rows_identical_to_step256_max_view": True,
        "qwen_pair_packed_rows_never_selected": True,
    }
    match_summary = {
        "target_basis": "nearest to original research proportional MAX heldout base-row indices",
        "child_to_original_distance": {"mean": statistics.mean(d_child), "median": statistics.median(d_child), "p90": sorted(d_child)[int(0.9*(len(d_child)-1))], "max": max(d_child)},
        "adult_to_original_distance": {"mean": statistics.mean(d_adult), "median": statistics.median(d_adult), "p90": sorted(d_adult)[int(0.9*(len(d_adult)-1))], "max": max(d_adult)},
        "child_adult_sorted_pair_distance": {"mean": statistics.mean(pair_dist), "median": statistics.median(pair_dist), "p90": sorted(pair_dist)[int(0.9*(len(pair_dist)-1))], "max": max(pair_dist)},
    }
    # Write selection maps for future audits.
    with (OUT / "regmax_selection_records.jsonl").open("w", encoding="utf-8") as f:
        for rec_c, rec_a in zip(child_match, adult_match):
            c = rec_c["chosen_base_index"]; a = rec_a["chosen_base_index"]
            f.write(json.dumps({
                "target_original_heldout_base_index": rec_c["target_original_heldout_base_index"],
                "childspeech_base_index": c,
                "adultprose_base_index": a,
                "childspeech_source": norm_source(base_rows[c]),
                "adultprose_source": norm_source(base_rows[a]),
                "childspeech_distance_to_target": rec_c["abs_distance"],
                "adultprose_distance_to_target": rec_a["abs_distance"],
                "child_adult_abs_distance": abs(c - a),
            }, ensure_ascii=False) + "\n")
    summary = {
        "status": "REGISTER_MAX_ROWHOLDOUT_POOLS_MATERIALIZED",
        "created_utc": now(),
        "purpose": "High-power MAX-rho register-displacement pair: identical admitted MAX FineWeb compact-view block, contrasting clean-register rows omitted from the fixed 10M budget.",
        "base_files": {"base_pool": rel(BASE_POOL), "view10": rel(VIEW10), "view100": rel(VIEW100), "original_step256_heldout": rel(ORIG_HELDOUT), "filler_template": rel(COMMON_FILLER_TEMPLATE), "metadata256": rel(META256)},
        "geometry": geom,
        "position_matching": match_summary,
        "arm_results": [child_arm, adult_arm],
        "sha256_step256_view100_reference": sha256_file(VIEW100),
        "selection_records": rel(OUT / "regmax_selection_records.jsonl"),
        "no_generation_streaming_model_loading_training_eval_upload_or_leaderboard": True,
        "elapsed_sec": round(time.time() - t0, 3),
    }
    (OUT / "register_max_rowholdout_metadata.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    md = [
        "# research MAX register-skewed row-holdout pools", "",
        "File-only materialization from existing base and MAX compact-view files. No generation, streaming, model loading, training, evaluation, upload, or leaderboard action.", "",
        "## Scientific contrast", 
        "Both arms admit the identical research MAX FineWeb compact-view block (7,923 rows, 1,118,587 words; rho=0.111859). The childspeech arm removes only CHILDES/OpenSubtitles/BNC/Switchboard 160-word rows from the original clean base; the adultprose arm removes only Gutenberg/SimpleWiki 160-word rows. The original trained MAX view arm is the proportional row-holdout midpoint.", "",
        "## Geometry", "```json", json.dumps(geom, indent=2), "```", "",
        "## Position matching", "```json", json.dumps(match_summary, indent=2), "```", "",
        "## Arm summaries", "```json", json.dumps({a['arm']: a['selected_holdout_summary'] for a in [child_arm, adult_arm]}, indent=2, ensure_ascii=False), "```", "",
        "Topup rows are only 133 clean words per arm and are reported separately in metadata; the admitted FineWeb rows are identical.", "",
        f"Metadata: `{rel(OUT / 'register_max_rowholdout_metadata.json')}`",
        f"Selection records: `{rel(OUT / 'regmax_selection_records.jsonl')}`",
    ]
    (OUT / "register_max_rowholdout_summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": summary["status"],
        "summary_md": rel(OUT / "register_max_rowholdout_summary.md"),
        "geometry": geom,
        "position_matching": match_summary,
        "arms": [{"arm": a["arm"], "file_100m": a["file_100m"], "sha_100m": a["sha_100m"], "selected": a["selected_holdout_summary"], "topup": a["topup_summary"]} for a in [child_arm, adult_arm]],
    }, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
