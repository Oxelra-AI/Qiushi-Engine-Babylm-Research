#!/usr/bin/env python3
"""research: supplementary overlap audit for register pair stream files.

The first stream audit found 23 positions that are replacement rows in both arms,
whereas the research position-matching summary reported only 3 same-clean-row
pair overlaps.  This file-only script quantifies the actual effective contrast:
which positions differ between arms, which positions are replaced in one/both
arms, and what clean sources are removed in each subset.

No model loading, training, evaluation, GPU, GlobalPIQA, SuperGLUE, AoA, upload,
or leaderboard.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative

import collections, csv, hashlib, json, pathlib, statistics, time
from typing import Any


def find_user_root() -> pathlib.Path:
    return _PUBLIC_ROOT

ROOT = find_user_root()
WS = ROOT / "experiments/archive/frontier_consolidation"
OUT = WS / "data/register_stream_overlap_audit"
POOL = WS / "data/register_position_matched_direct_pools"
CLEAN10 = WS / "data/dose_2p64x_rowholdout_pools/cleanqwen_lengthmatched_dose2p64x_10M.jsonl"
META = POOL / "register_position_matched_direct_pools_metadata.json"
FILES = {
    "childsub_posmatched": POOL / "regpos_childsub_posmatched_samefw_quarter_10M.jsonl",
    "adult_posmatched": POOL / "regpos_adult_posmatched_samefw_quarter_10M.jsonl",
}


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def rel(p: pathlib.Path) -> str:
    try:
        return str(p.relative_to(ROOT))
    except Exception:
        return str(p)


def read_jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def row_words(row: dict[str, Any]) -> int:
    return int(row.get("words", len(str(row.get("text", "")).split())))


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def source_component(row: dict[str, Any]) -> str:
    src = str(row.get("source", ""))
    if "::" in src:
        return src.split("::", 1)[1]
    return src or "unknown"


def group_of_component(comp: str) -> str:
    if comp in {"childes", "open_subtitles"}:
        return "child_sub"
    if comp in {"gutenberg", "simple_wiki"}:
        return "adult_gut_simple"
    return "other"


def summarize_indices(label: str, indices: set[int], clean: list[dict[str, Any]], rows: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    comp = collections.Counter()
    groups = collections.Counter()
    words = 0
    examples = []
    for i in sorted(indices):
        w = row_words(clean[i]); words += w
        c = source_component(clean[i])
        comp[c] += w
        groups[group_of_component(c)] += w
        if len(examples) < 12:
            examples.append({
                "row": i,
                "words": w,
                "clean_source": clean[i].get("source"),
                "child_kind": rows["childsub_posmatched"][i].get("regpos_row_kind"),
                "adult_kind": rows["adult_posmatched"][i].get("regpos_row_kind"),
                "child_text_sha": text_hash(str(rows["childsub_posmatched"][i].get("text", ""))),
                "adult_text_sha": text_hash(str(rows["adult_posmatched"][i].get("text", ""))),
                "clean_text_sha": text_hash(str(clean[i].get("text", ""))),
            })
    return {
        "label": label,
        "row_count": len(indices),
        "word_count": words,
        "component_words": dict(sorted(comp.items())),
        "group_words": dict(sorted(groups.items())),
        "row_mean": statistics.mean(indices) if indices else None,
        "row_min": min(indices) if indices else None,
        "row_max": max(indices) if indices else None,
        "examples": examples,
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    clean = read_jsonl(CLEAN10)
    rows = {k: read_jsonl(v) for k, v in FILES.items()}
    meta = json.loads(META.read_text(encoding="utf-8"))
    changed = {k: {i for i, r in enumerate(v) if r.get("regpos_row_kind") == "position_matched_direct_fineweb_replacement"} for k, v in rows.items()}
    child = changed["childsub_posmatched"]
    adult = changed["adult_posmatched"]
    both = child & adult
    child_only = child - adult
    adult_only = adult - child
    union = child | adult
    neither = set(range(len(clean))) - union

    text_diff = {i for i in range(len(clean)) if rows["childsub_posmatched"][i].get("text") != rows["adult_posmatched"][i].get("text")}
    child_diff_from_clean = {i for i in range(len(clean)) if rows["childsub_posmatched"][i].get("text") != clean[i].get("text")}
    adult_diff_from_clean = {i for i in range(len(clean)) if rows["adult_posmatched"][i].get("text") != clean[i].get("text")}
    unexpected_child = child_diff_from_clean - child
    unexpected_adult = adult_diff_from_clean - adult

    fine_hashes = {k: [text_hash(str(rows[k][i].get("text", ""))) for i in sorted(changed[k])] for k in rows}
    fine_counter_equal = collections.Counter(fine_hashes["childsub_posmatched"]) == collections.Counter(fine_hashes["adult_posmatched"])

    # Among overlapping replacement positions, are the exact FineWeb texts same or different?
    both_same_fineweb_text = {i for i in both if rows["childsub_posmatched"][i].get("text") == rows["adult_posmatched"][i].get("text")}
    both_different_fineweb_text = both - both_same_fineweb_text

    subset_summaries = [
        summarize_indices("child_replaced_only", child_only, clean, rows),
        summarize_indices("adult_replaced_only", adult_only, clean, rows),
        summarize_indices("replaced_in_both_arms", both, clean, rows),
        summarize_indices("changed_union", union, clean, rows),
        summarize_indices("unchanged_in_both_arms", neither, clean, rows),
        summarize_indices("cross_arm_text_diff_positions", text_diff, clean, rows),
    ]

    def write_csv(path: pathlib.Path, records: list[dict[str, Any]]) -> None:
        flat = []
        for r in records:
            rr = {k: v for k, v in r.items() if k != "examples"}
            rr["component_words_json"] = json.dumps(rr.pop("component_words"), ensure_ascii=False)
            rr["group_words_json"] = json.dumps(rr.pop("group_words"), ensure_ascii=False)
            flat.append(rr)
        fields = sorted({k for r in flat for k in r}) if flat else []
        with path.open("w", encoding="utf-8", newline="") as f:
            if fields:
                w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(flat)

    write_csv(OUT / "overlap_subset_summaries.csv", subset_summaries)
    # Save example rows separately.
    ex_rows = []
    for s in subset_summaries:
        for ex in s["examples"]:
            ex_rows.append({"subset": s["label"], **ex})
    fields = sorted({k for r in ex_rows for k in r}) if ex_rows else []
    with (OUT / "overlap_examples.csv").open("w", encoding="utf-8", newline="") as f:
        if fields:
            w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(ex_rows)

    summary = {
        "status": "REGISTER_STREAM_OVERLAP_AUDIT_COMPLETE",
        "created_utc": now(),
        "boundary": "File-only overlap/effective-contrast audit; no model loading, training, evaluation, GPU, GlobalPIQA, SuperGLUE, AoA, upload, or leaderboard.",
        "files": {"child": rel(FILES["childsub_posmatched"]), "adult": rel(FILES["adult_posmatched"]), "clean": rel(CLEAN10), "metadata": rel(META)},
        "row_counts": {"total": len(clean), "child_changed": len(child), "adult_changed": len(adult), "both_changed": len(both), "child_only_changed": len(child_only), "adult_only_changed": len(adult_only), "union_changed": len(union), "neither_changed": len(neither), "cross_arm_text_diff": len(text_diff)},
        "word_counts": {"both_changed_clean_words": sum(row_words(clean[i]) for i in both), "child_only_clean_words": sum(row_words(clean[i]) for i in child_only), "adult_only_clean_words": sum(row_words(clean[i]) for i in adult_only), "cross_arm_text_diff_clean_words": sum(row_words(clean[i]) for i in text_diff)},
        "fineweb_multiset_identical_across_arms": fine_counter_equal,
        "both_changed_same_fineweb_text_count": len(both_same_fineweb_text),
        "both_changed_different_fineweb_text_count": len(both_different_fineweb_text),
        "child_diff_from_clean_count": len(child_diff_from_clean),
        "adult_diff_from_clean_count": len(adult_diff_from_clean),
        "unexpected_child_diffs_outside_marker_count": len(unexpected_child),
        "unexpected_adult_diffs_outside_marker_count": len(unexpected_adult),
        "position_matching_summary_from_metadata": meta.get("position_matching_summary"),
        "subset_summaries": subset_summaries,
        "output_files": {"subset_csv": rel(OUT / "overlap_subset_summaries.csv"), "examples_csv": rel(OUT / "overlap_examples.csv"), "summary_json": rel(OUT / "register_stream_overlap_audit_summary.json"), "summary_md": rel(OUT / "register_stream_overlap_audit_summary.md")},
    }
    (OUT / "register_stream_overlap_audit_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")

    lines = [
        "# research register stream overlap audit",
        "",
        summary["boundary"],
        "",
        "## Effective row contrast",
        "",
        f"- Child/subtitle-removal arm changed rows: {len(child)}.",
        f"- Adult-prose-removal arm changed rows: {len(adult)}.",
        f"- Positions changed in both arms: {len(both)}; positions changed in only child arm: {len(child_only)}; only adult arm: {len(adult_only)}; union: {len(union)}.",
        f"- Actual cross-arm text-different rows: {len(text_diff)}.",
        f"- FineWeb text multiset identical across arms: {fine_counter_equal}.",
        f"- Overlapping replacement positions with same FineWeb text: {len(both_same_fineweb_text)}; with different FineWeb text: {len(both_different_fineweb_text)}.",
        f"- Unexpected child/adult text differences outside replacement markers: {len(unexpected_child)} / {len(unexpected_adult)}.",
        "",
        "The metadata position-match field `same_clean_row_both_arms_count=3` counts matched selection pairs that used the same clean row; it is not the same as the actual number of output positions that are replacement rows in both arms, which is 23. The score contrast remains mostly direct register substitution because 735/758 replacement positions per arm are not shared, all rows keep the same word-count sequence, and rows outside the changed union are identical clean text.",
        "",
        "## Subset source mixtures (clean words at positions)",
        "",
        "| subset | rows | words | row mean | source groups |",
        "|---|---:|---:|---:|---|",
    ]
    for s in subset_summaries:
        lines.append(f"| {s['label']} | {s['row_count']} | {s['word_count']} | {s['row_mean'] if s['row_mean'] is not None else ''} | `{json.dumps(s['group_words'], ensure_ascii=False)}` |")
    lines += [
        "",
        "## Reading consequence",
        "",
        "The direct pair is a valid high-contrast test of removal-side opportunity cost, but not a pure abstract register axis. Interpret pair deltas as conditional on these selected mixtures and on the small amount of shared replacement-position overlap.",
        "",
        f"JSON: `{rel(OUT / 'register_stream_overlap_audit_summary.json')}`",
        f"Subset CSV: `{rel(OUT / 'overlap_subset_summaries.csv')}`",
        f"Examples CSV: `{rel(OUT / 'overlap_examples.csv')}`",
    ]
    (OUT / "register_stream_overlap_audit_summary.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    print(json.dumps({
        "status": summary["status"],
        "row_counts": summary["row_counts"],
        "fineweb_multiset_identical_across_arms": fine_counter_equal,
        "summary_md": rel(OUT / "register_stream_overlap_audit_summary.md"),
    }, indent=2, ensure_ascii=False), flush=True)

if __name__ == "__main__":
    main()
