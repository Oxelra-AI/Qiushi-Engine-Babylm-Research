#!/usr/bin/env python3
"""research: split compact-view base ordinary-fit anchors by register-exposure subset.

The research scorer produced base losses on the full research 2,647-row set and
on the 1,743-row subset that contains no inherited COMPACT_EXPERIENCE ALN source/rewrite
text.  For the up-dose interpretation we also need the complementary 904 rows:
these rows contain inherited ALN pair text in the *base* family but are not row
identities in the current base/dose streams, so within-family dose deltas on
this subset test hard/selectable-register transfer without dose re-admission.

This script derives exact base anchors from the row-level research score file,
without rescoring models.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import csv
import json
import pathlib
import statistics
import time
from collections import Counter, defaultdict
from typing import Any

ROOT0 = _public_path('experiments/archive/relation_learning/scripts/summarize_base_fit_register_subsets.py')
ROOT = _PUBLIC_ROOT

WS = ROOT / "experiments/archive/relation_learning"
FULL_PATH = ROOT / "experiments/archive/frontier_consolidation/data/density_cleanqwen_overlay_medium_riskhard/heldout_cleanqwen_rows.jsonl"
CLEAN_PATH = WS / "data/current_dose_stream_heldout2647_exposure/heldout2647_no_inherited_aln_pair_text_hits.jsonl"
HIT_PATH = WS / "data/current_dose_stream_heldout2647_exposure/heldout2647_inherited_aln_pair_text_hit_rows.jsonl"
ROWS_CSV = WS / "data/compactview_ordinary_fit_subsets/ordinary_fit_rows.csv"
OUT = WS / "data/register_subset_fit_anchors"


def rel(p: pathlib.Path) -> str:
    try:
        return str(p.relative_to(ROOT))
    except Exception:
        return str(p)


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def read_jsonl(path: pathlib.Path):
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def norm_source(x: Any) -> str:
    return str(x or "")


def row_key(obj: dict[str, Any]) -> tuple[str, int]:
    return (norm_source(obj.get("source", "")), int(obj["example_id"]))


def load_rowset(path: pathlib.Path) -> dict[tuple[str, int], dict[str, Any]]:
    rows: dict[tuple[str, int], dict[str, Any]] = {}
    for obj in read_jsonl(path):
        rows[row_key(obj)] = obj
    return rows


def write_csv(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("\n", encoding="utf-8")
        return
    keys: list[str] = []
    seen: set[str] = set()
    for r in rows:
        for k in r:
            if k not in seen:
                keys.append(k); seen.add(k)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader(); w.writerows(rows)


def mean(xs: list[float]) -> float:
    return float(statistics.mean(xs)) if xs else float("nan")


def se(xs: list[float]) -> float:
    if len(xs) < 2:
        return float("nan")
    return float(statistics.stdev(xs) / (len(xs) ** 0.5))


def summarize_rowset(name: str, rows: dict[tuple[str, int], dict[str, Any]]) -> dict[str, Any]:
    source_counts = Counter(norm_source(r.get("source", "")) for r in rows.values())
    word_values = [int(r.get("words", len(str(r.get("text", "")).split()))) for r in rows.values()]
    text_len_values = [len(str(r.get("text", ""))) for r in rows.values()]
    return {
        "subset": name,
        "rows": len(rows),
        "mean_words": mean([float(x) for x in word_values]),
        "median_words": float(statistics.median(word_values)) if word_values else float("nan"),
        "mean_chars": mean([float(x) for x in text_len_values]),
        "source_counts": dict(sorted(source_counts.items())),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    full = load_rowset(FULL_PATH)
    clean = load_rowset(CLEAN_PATH)
    hit = load_rowset(HIT_PATH)
    derived_hit = {k: v for k, v in full.items() if k not in clean}
    if set(hit) != set(derived_hit):
        raise RuntimeError({
            "message": "explicit hit rows differ from full-clean complement",
            "hit_only": len(set(hit) - set(derived_hit)),
            "derived_only": len(set(derived_hit) - set(hit)),
        })
    if set(full) != (set(clean) | set(hit)) or (set(clean) & set(hit)):
        raise RuntimeError("rowset partition invariant failed")

    membership: dict[tuple[str, int], str] = {}
    for k in clean:
        membership[k] = "heldout2647_no_inherited_aln_text"
    for k in hit:
        membership[k] = "heldout2647_inherited_aln_text_hit"

    scored_full: list[dict[str, Any]] = []
    with ROWS_CSV.open(newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["subset"] != "heldout2647_full":
                continue
            k = (norm_source(r.get("source", "")), int(r["example_id"]))
            if k not in membership:
                raise RuntimeError(f"scored row key not in rowset partition: {k}")
            q = dict(r)
            q["register_subset"] = membership[k]
            q["loss"] = float(q["loss"])
            q["n_masked"] = int(q["n_masked"])
            q["example_id"] = int(q["example_id"])
            q["seed"] = int(q["seed"])
            scored_full.append(q)

    by_checkpoint: list[dict[str, Any]] = []
    grouped: dict[tuple[str, str, int, str], list[dict[str, Any]]] = defaultdict(list)
    # subset, arm, seed, checkpoint
    for r in scored_full:
        grouped[(r["register_subset"], r["arm"], int(r["seed"]), r["checkpoint"])].append(r)
    for (subset, arm, seed, ckpt), rs in sorted(grouped.items()):
        vals = [float(x["loss"]) for x in rs]
        by_checkpoint.append({
            "subset": subset,
            "arm": arm,
            "dose": rs[0]["dose"],
            "seed": seed,
            "checkpoint": ckpt,
            "n": len(vals),
            "mean_loss": mean(vals),
            "se_loss": se(vals),
            "mean_masked": mean([float(x["n_masked"]) for x in rs]),
        })

    late_group: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for r in by_checkpoint:
        late_group[(r["subset"], r["arm"], int(r["seed"]))].append(r)
    late_summary: list[dict[str, Any]] = []
    for (subset, arm, seed), rs in sorted(late_group.items()):
        late_summary.append({
            "subset": subset,
            "arm": arm,
            "dose": rs[0]["dose"],
            "seed": seed,
            "checkpoints": ",".join(r["checkpoint"] for r in sorted(rs, key=lambda x: x["checkpoint"])),
            "n_min": min(int(r["n"]) for r in rs),
            "mean_loss": mean([float(r["mean_loss"]) for r in rs]),
            "mean_masked": mean([float(r["mean_masked"]) for r in rs]),
        })

    # Verify weighted reconstruction of the full research late anchors from the partition.
    full_late = []
    full_group: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for r in scored_full:
        full_group[(r["arm"], int(r["seed"]))].append(r)
    for (arm, seed), rs in sorted(full_group.items()):
        ck_means = []
        for ck in sorted({x["checkpoint"] for x in rs}):
            vals = [float(x["loss"]) for x in rs if x["checkpoint"] == ck]
            ck_means.append(mean(vals))
        full_late.append({"subset": "heldout2647_full_recomputed", "arm": arm, "dose": rs[0]["dose"], "seed": seed, "n_min": len({(x["source"], x["example_id"]) for x in rs}), "mean_loss": mean(ck_means)})

    write_csv(OUT / "base_register_subset_by_checkpoint.csv", by_checkpoint)
    write_csv(OUT / "base_register_subset_late_summary.csv", late_summary)
    write_csv(OUT / "base_full_recomputed_late_summary.csv", full_late)

    payload = {
        "status": "BASE_REGISTER_SUBSET_FIT_ANCHORS_DONE",
        "created_utc": now(),
        "inputs": {
            "full_rows": rel(FULL_PATH),
            "no_inherited_aln_text_rows": rel(CLEAN_PATH),
            "inherited_aln_text_hit_rows": rel(HIT_PATH),
            "base_row_scores": rel(ROWS_CSV),
        },
        "partition": {
            "full": summarize_rowset("heldout2647_full", full),
            "no_inherited_aln_text": summarize_rowset("heldout2647_no_inherited_aln_text", clean),
            "inherited_aln_text_hit": summarize_rowset("heldout2647_inherited_aln_text_hit", hit),
        },
        "late_summary": late_summary,
        "full_recomputed_late_summary": full_late,
        "outputs": {
            "by_checkpoint_csv": rel(OUT / "base_register_subset_by_checkpoint.csv"),
            "late_summary_csv": rel(OUT / "base_register_subset_late_summary.csv"),
            "full_recomputed_csv": rel(OUT / "base_full_recomputed_late_summary.csv"),
            "summary_json": rel(OUT / "summary.json"),
        },
        "scientific_interpretation": "The 904 inherited-ALN-text rows are the hard selectable-register complement of the 1,743-row no-inherited-ALN-text subset. Future dose deltas should be read separately on these subsets within seed: the 1,743 rows test outside-register clean transfer, while the 904 rows test fit movement on difficult restatement-selectable text not re-admitted by the repaired dose pairs.",
    }
    (OUT / "summary.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
