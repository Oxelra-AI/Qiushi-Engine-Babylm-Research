#!/usr/bin/env python3
"""research: epoch-phase/source-exposure atlas for the scale1.75 late peak.

This CPU/file-only analysis maps legal 10M pool row order and exact checkpoint
exposure overshoot onto the 70M--100M selected score trajectory.  It asks whether
the newly resolved 84M cheap7 peak is obviously tied to a particular deterministic
source phase of the repeated 10M corpus, or instead looks like vector competence
allocation not explained by a single recent source slice.

No training, model evaluation, upload, or leaderboard submission is performed.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse
import csv
import json
import math
import pathlib
import time
from collections import defaultdict
from statistics import mean
from typing import Any

POOL_WORDS = 10_000_000
CHEAP_COLUMNS = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA", "Reading"]
EXPECTED_ENDPOINTS = [
    "chck_70M", "chck_72M", "chck_74M", "chck_76M", "chck_78M", "chck_80M",
    "chck_82M", "chck_84M", "chck_86M", "chck_88M", "chck_90M", "chck_92M",
    "chck_94M", "chck_96M", "chck_98M", "chck_100M",
]


def find_user_root() -> pathlib.Path:
    return _PUBLIC_ROOT


USER_ROOT = find_user_root()
STUDY = USER_ROOT / "experiments/archive/frontier_consolidation"
WORKSPACE = STUDY
DEFAULT_POOL = WORKSPACE / "data/density_cleanqwen_overlay_medium_riskhard/cleanqwen_fineweb_compact_view_reinvest_10M.jsonl"
DEFAULT_META = WORKSPACE / "data/density_cleanqwen_overlay_medium_riskhard/density_cleanqwen_rowholdout_overlay_metadata.json"
DEFAULT_METRICS = WORKSPACE / "training/runs/adapter128_scale1p75_h100M100M_seed43022_official_ladder/scientific_metrics.json"
DEFAULT_TRAJ = WORKSPACE / "data/selected_trajectory_eval_scale1p75_seed43022_reference_common2M/selected_trajectory.json"


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def rel(path: pathlib.Path | str) -> str:
    p = pathlib.Path(path)
    try:
        return str(p.resolve().relative_to(USER_ROOT))
    except Exception:
        return str(p)


def read_json(path: pathlib.Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: pathlib.Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def fnum(x: Any) -> float | None:
    if x is None or x == "":
        return None
    try:
        y = float(x)
    except Exception:
        return None
    if math.isnan(y):
        return None
    return y


def source_family(source: str) -> str:
    s = source or ""
    if s == "cleanqwen_fineweb_compact_view_reinvest":
        return "compact_pair_block"
    if s.startswith("neutral_cleanqwen_topup_compact_reinvest"):
        return "compact_topup"
    if s == "qwen_pair_packed":
        return "inherited_qwen_pair"
    if s in {"childes", "open_subtitles", "bnc_spoken", "gutenberg", "simple_wiki", "switchboard"}:
        return "official_base"
    return "other"


def load_pool(pool_path: pathlib.Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    pos = 0
    source_words: dict[str, int] = defaultdict(int)
    family_words: dict[str, int] = defaultdict(int)
    source_rows: dict[str, int] = defaultdict(int)
    family_rows: dict[str, int] = defaultdict(int)
    with pool_path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            obj = json.loads(line)
            words = int(obj.get("words", len(str(obj.get("text", "")).split())))
            text = str(obj.get("text", ""))
            source = str(obj.get("source", ""))
            fam = source_family(source)
            rec = {
                "row_index": idx,
                "start": pos,
                "end": pos + words,
                "words": words,
                "source": source,
                "family": fam,
                "example_id": obj.get("example_id"),
                "char_len": len(text),
            }
            rows.append(rec)
            pos += words
            source_words[source] += words
            family_words[fam] += words
            source_rows[source] += 1
            family_rows[fam] += 1
    summary = {
        "pool_path": rel(pool_path),
        "n_rows": len(rows),
        "total_words": pos,
        "source_words": dict(sorted(source_words.items(), key=lambda kv: (-kv[1], kv[0]))),
        "source_rows": dict(sorted(source_rows.items(), key=lambda kv: (-kv[1], kv[0]))),
        "family_words": dict(sorted(family_words.items(), key=lambda kv: (-kv[1], kv[0]))),
        "family_rows": dict(sorted(family_rows.items(), key=lambda kv: (-kv[1], kv[0]))),
        "first_rows": rows[:5],
        "last_rows": rows[-5:],
    }
    return rows, summary


def add_interval_comp(comp: dict[str, Any], row: dict[str, Any], amount: int) -> None:
    if amount <= 0:
        return
    comp["total_words"] += amount
    comp.setdefault("source_words", defaultdict(int))[row["source"]] += amount
    comp.setdefault("family_words", defaultdict(int))[row["family"]] += amount
    comp.setdefault("rows_touched", set()).add(row["row_index"])
    comp.setdefault("source_rows_touched", defaultdict(set))[row["source"]].add(row["row_index"])
    comp.setdefault("family_rows_touched", defaultdict(set))[row["family"]].add(row["row_index"])


def composition_for_cycle_segment(rows: list[dict[str, Any]], start: int, end: int) -> dict[str, Any]:
    """Composition for [start,end) within one cycle, no wrap."""
    comp: dict[str, Any] = {"total_words": 0}
    # 64k rows and ~20 segments only; linear scan is fine and more transparent.
    for row in rows:
        ov = min(end, int(row["end"])) - max(start, int(row["start"]))
        if ov > 0:
            add_interval_comp(comp, row, int(ov))
    return finalize_comp(comp)


def finalize_comp(comp: dict[str, Any]) -> dict[str, Any]:
    total = int(comp.get("total_words", 0))
    src = comp.get("source_words", {})
    fam = comp.get("family_words", {})
    src_rows = comp.get("source_rows_touched", {})
    fam_rows = comp.get("family_rows_touched", {})
    out = {
        "total_words": total,
        "source_words": dict(sorted(((k, int(v)) for k, v in src.items()), key=lambda kv: (-kv[1], kv[0]))),
        "source_fraction": {k: (float(v) / total if total else None) for k, v in sorted(src.items(), key=lambda kv: (-kv[1], kv[0]))},
        "family_words": dict(sorted(((k, int(v)) for k, v in fam.items()), key=lambda kv: (-kv[1], kv[0]))),
        "family_fraction": {k: (float(v) / total if total else None) for k, v in sorted(fam.items(), key=lambda kv: (-kv[1], kv[0]))},
        "n_rows_touched": len(comp.get("rows_touched", set())),
        "source_rows_touched": {k: len(v) for k, v in sorted(src_rows.items())},
        "family_rows_touched": {k: len(v) for k, v in sorted(fam_rows.items())},
    }
    return out


def composition_for_stream_interval(rows: list[dict[str, Any]], start_abs: int, end_abs: int, pool_words: int = POOL_WORDS) -> dict[str, Any]:
    if end_abs <= start_abs:
        return finalize_comp({"total_words": 0})
    comp: dict[str, Any] = {"total_words": 0}
    cur = start_abs
    while cur < end_abs:
        cyc_pos = cur % pool_words
        take = min(end_abs - cur, pool_words - cyc_pos)
        seg = composition_for_cycle_segment(rows, int(cyc_pos), int(cyc_pos + take))
        # merge finalized seg back
        for source, words in seg["source_words"].items():
            comp.setdefault("source_words", defaultdict(int))[source] += int(words)
        for fam, words in seg["family_words"].items():
            comp.setdefault("family_words", defaultdict(int))[fam] += int(words)
        comp["total_words"] += int(seg["total_words"])
        # row touched counts across cycles are less meaningful; keep union by source/family only within cycle not absolute.
        cur += take
    return finalize_comp(comp)


def bin_rows(rows: list[dict[str, Any]], bin_words: int) -> list[dict[str, Any]]:
    bins = []
    for start in range(0, POOL_WORDS, bin_words):
        end = min(POOL_WORDS, start + bin_words)
        comp = composition_for_cycle_segment(rows, start, end)
        flat = {
            "bin_start": start,
            "bin_end": end,
            "bin_m_start": start / 1_000_000.0,
            "bin_m_end": end / 1_000_000.0,
            "total_words": comp["total_words"],
        }
        for source, words in comp["source_words"].items():
            flat[f"source_words__{source}"] = words
            flat[f"source_frac__{source}"] = comp["source_fraction"][source]
        for fam, words in comp["family_words"].items():
            flat[f"family_words__{fam}"] = words
            flat[f"family_frac__{fam}"] = comp["family_fraction"][fam]
        bins.append(flat)
    return bins


def load_checkpoint_actuals(metrics_path: pathlib.Path, endpoints: list[str]) -> dict[str, int]:
    metrics = read_json(metrics_path)
    out = {}
    for rec in metrics.get("saved_checkpoints", []):
        if isinstance(rec, dict) and rec.get("name") in endpoints:
            out[str(rec["name"])] = int(rec["actual_cumulative_word_exposure"])
    return out


def load_scores(traj_path: pathlib.Path) -> dict[str, dict[str, Any]]:
    if not traj_path.exists():
        return {}
    data = read_json(traj_path)
    if isinstance(data, dict) and isinstance(data.get("rows"), list):
        data = data["rows"]
    out = {}
    if isinstance(data, list):
        for row in data:
            if not isinstance(row, dict) or not row.get("endpoint"):
                continue
            if row.get("error"):
                continue
            if fnum(row.get("cheap7")) is None:
                continue
            rec = {k: row.get(k) for k in ["endpoint", "words", "cheap7"] + CHEAP_COLUMNS + ["payload_path"]}
            for k in ["cheap7"] + CHEAP_COLUMNS:
                rec[k] = fnum(rec.get(k))
            out[str(row["endpoint"])] = rec
    return out


def flatten_interval_record(ep_prev: str, ep: str, actual_prev: int, actual: int, comp: dict[str, Any], scores: dict[str, dict[str, Any]]) -> dict[str, Any]:
    row: dict[str, Any] = {
        "from_endpoint": ep_prev,
        "to_endpoint": ep,
        "actual_start": actual_prev,
        "actual_end": actual,
        "delta_words": actual - actual_prev,
        "cycle_start": actual_prev % POOL_WORDS,
        "cycle_end": actual % POOL_WORDS,
        "wraps_cycle": (actual % POOL_WORDS) < (actual_prev % POOL_WORDS),
        "total_composition_words": comp["total_words"],
    }
    for fam, words in comp["family_words"].items():
        row[f"family_words__{fam}"] = words
        row[f"family_frac__{fam}"] = comp["family_fraction"][fam]
    for src, words in comp["source_words"].items():
        row[f"source_words__{src}"] = words
        row[f"source_frac__{src}"] = comp["source_fraction"][src]
    a = scores.get(ep_prev)
    b = scores.get(ep)
    if a and b:
        for key in ["cheap7"] + CHEAP_COLUMNS:
            if fnum(a.get(key)) is not None and fnum(b.get(key)) is not None:
                row[f"delta_score__{key}"] = float(b[key] - a[key])
        # additional interpretation coordinates
        nogp = [c for c in CHEAP_COLUMNS if c != "GlobalPIQA"]
        core = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS"]
        rel = ["EWoK", "Entity"]
        for name, cols in [("cheap6_no_GlobalPIQA", nogp), ("cheap5_no_GP_Reading", core), ("relation_state", rel)]:
            av = [fnum(a.get(c)) for c in cols]
            bv = [fnum(b.get(c)) for c in cols]
            if all(v is not None for v in av + bv):
                row[f"delta_score__{name}"] = float(mean(bv) - mean(av))  # type: ignore[arg-type]
    return row


def write_csv(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    seen = set()
    for r in rows:
        for k in r.keys():
            if k not in seen:
                seen.add(k)
                keys.append(k)
    with path.open("w", encoding="utf-8", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=keys)
        wr.writeheader()
        for r in rows:
            wr.writerow({k: r.get(k, "") for k in keys})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", type=pathlib.Path, default=DEFAULT_POOL)
    ap.add_argument("--metadata", type=pathlib.Path, default=DEFAULT_META)
    ap.add_argument("--metrics", type=pathlib.Path, default=DEFAULT_METRICS)
    ap.add_argument("--trajectory", type=pathlib.Path, default=DEFAULT_TRAJ)
    ap.add_argument("--out-dir", type=pathlib.Path, required=True)
    ap.add_argument("--bin-words", type=int, default=1_000_000)
    args = ap.parse_args()
    t0 = time.time()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    rows, pool_summary = load_pool(args.pool)
    if pool_summary["total_words"] != POOL_WORDS:
        raise RuntimeError(f"Pool words {pool_summary['total_words']} != {POOL_WORDS}")
    meta = read_json(args.metadata) if args.metadata.exists() else {}
    actuals = load_checkpoint_actuals(args.metrics, EXPECTED_ENDPOINTS)
    scores = load_scores(args.trajectory)

    bins = bin_rows(rows, args.bin_words)
    intervals = []
    prev_ep = None
    prev_actual = None
    for ep in EXPECTED_ENDPOINTS:
        actual = actuals.get(ep)
        if actual is None:
            continue
        if prev_ep is not None and prev_actual is not None:
            comp = composition_for_stream_interval(rows, prev_actual, actual)
            intervals.append(flatten_interval_record(prev_ep, ep, prev_actual, actual, comp, scores))
        prev_ep = ep
        prev_actual = actual

    # Endpoint phase rows: where each checkpoint falls in the repeated 10M cycle, plus score if present.
    endpoint_phase = []
    for ep in EXPECTED_ENDPOINTS:
        actual = actuals.get(ep)
        if actual is None:
            continue
        rec: dict[str, Any] = {
            "endpoint": ep,
            "target_words": int(ep[len("chck_"):-1]) * 1_000_000,
            "actual_words": actual,
            "actual_minus_target": actual - int(ep[len("chck_"):-1]) * 1_000_000,
            "epoch_index_zero_based": actual // POOL_WORDS,
            "cycle_pos": actual % POOL_WORDS,
            "cycle_pos_m": (actual % POOL_WORDS) / 1_000_000.0,
        }
        if ep in scores:
            rec.update({f"score__{k}": scores[ep].get(k) for k in ["cheap7"] + CHEAP_COLUMNS})
        endpoint_phase.append(rec)

    # Compact pair block positional summaries.
    compact_rows = [r for r in rows if r["family"] == "compact_pair_block"]
    compact_positions = {
        "n_rows": len(compact_rows),
        "words": int(sum(r["words"] for r in compact_rows)),
        "first_start": min((r["start"] for r in compact_rows), default=None),
        "last_end": max((r["end"] for r in compact_rows), default=None),
        "cycle_start_m": (min((r["start"] for r in compact_rows), default=0) / 1_000_000.0) if compact_rows else None,
        "cycle_last_end_m": (max((r["end"] for r in compact_rows), default=0) / 1_000_000.0) if compact_rows else None,
        "n_contiguous_runs": None,
        "contiguous_runs_top10": [],
    }
    runs = []
    if compact_rows:
        cur = {"start": compact_rows[0]["start"], "end": compact_rows[0]["end"], "rows": 1, "words": compact_rows[0]["words"]}
        prev_idx = compact_rows[0]["row_index"]
        for r in compact_rows[1:]:
            if r["row_index"] == prev_idx + 1 and r["start"] == cur["end"]:
                cur["end"] = r["end"]
                cur["rows"] += 1
                cur["words"] += r["words"]
            else:
                runs.append(dict(cur))
                cur = {"start": r["start"], "end": r["end"], "rows": 1, "words": r["words"]}
            prev_idx = r["row_index"]
        runs.append(dict(cur))
    compact_positions["n_contiguous_runs"] = len(runs)
    compact_positions["contiguous_runs_top10"] = sorted(runs, key=lambda x: -x["words"])[:10]

    summary = {
        "status": "EPOCH_PHASE_EXPOSURE_ATLAS",
        "created_utc": now(),
        "purpose": "Map the deterministic repeated 10M corpus phase and source composition around the reference scale1.75 84M cheap7 peak.",
        "pool_summary": pool_summary,
        "metadata_core": {
            "status": meta.get("status"),
            "total_words_per_pool": meta.get("total_words_per_pool"),
            "passes": meta.get("passes"),
            "compact_reinvest_family": meta.get("families", {}).get("compact_reinvest", {}),
        },
        "metrics": {"path": rel(args.metrics), "n_actual_endpoints": len(actuals)},
        "trajectory": {"path": rel(args.trajectory), "n_score_endpoints": len(scores), "scored_endpoints": sorted(scores)},
        "compact_pair_block_positions": compact_positions,
        "endpoint_phase_rows": endpoint_phase,
        "interval_rows": intervals,
        "interpretation": {},
        "output_files": {},
        "elapsed_sec": round(time.time() - t0, 2),
    }

    # Lightweight interpretation, intentionally cautious.
    by_to = {r["to_endpoint"]: r for r in intervals}
    key_intervals = {k: by_to.get(k) for k in ["chck_82M", "chck_84M", "chck_86M", "chck_100M"] if by_to.get(k)}
    summary["interpretation"] = {
        "key_intervals": key_intervals,
        "caution": "Recent interval source composition is a temporal clue only: checkpoint scores reflect accumulated optimization, and no single interval proves causality without matched ordering or replay controls.",
        "main_question_for_pending_scores": "If 84M remains best after resumed late rows and SuperGLUE, decide endpoint packaging separately from any transferable mechanism claim; seed43122 scoring is still required for robustness of the late phase.",
    }

    bins_path = args.out_dir / "pool_phase_bins.csv"
    intervals_path = args.out_dir / "checkpoint_interval_source_score_atlas.csv"
    endpoint_path = args.out_dir / "checkpoint_epoch_phase.csv"
    write_csv(bins_path, bins)
    write_csv(intervals_path, intervals)
    write_csv(endpoint_path, endpoint_phase)
    summary["output_files"] = {
        "pool_phase_bins_csv": rel(bins_path),
        "checkpoint_interval_source_score_atlas_csv": rel(intervals_path),
        "checkpoint_epoch_phase_csv": rel(endpoint_path),
        "summary_json": rel(args.out_dir / "epoch_phase_exposure_atlas.json"),
        "summary_md": rel(args.out_dir / "epoch_phase_exposure_atlas.md"),
    }
    write_json(args.out_dir / "epoch_phase_exposure_atlas.json", summary)

    def fmt_frac(row: dict[str, Any], key: str) -> str:
        v = fnum(row.get(key))
        return f"{100*v:.2f}%" if v is not None else ""

    lines = ["# research epoch-phase/source-exposure atlas\n\n"]
    lines.append(f"Pool: `{rel(args.pool)}`; rows {pool_summary['n_rows']:,}, words {pool_summary['total_words']:,}.\n\n")
    lines.append("## 10M pool family composition\n\n")
    for fam, words in pool_summary["family_words"].items():
        lines.append(f"- {fam}: {words:,} words ({100*words/POOL_WORDS:.3f}%).\n")
    lines.append("\n## Compact pair-block position\n\n")
    lines.append(f"Compact pair rows: {compact_positions['n_rows']:,}, words {compact_positions['words']:,}, first {compact_positions['cycle_start_m']:.3f}M, last end {compact_positions['cycle_last_end_m']:.3f}M, contiguous runs {compact_positions['n_contiguous_runs']}.\n\n")
    lines.append("## Checkpoint phase and cheap7\n\n")
    lines.append("| endpoint | actual words | cycle pos (M) | cheap7 | BLiMP | Supplement | EWoK | Entity | COMPS | GlobalPIQA | Reading |\n")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
    for r in endpoint_phase:
        lines.append("| " + str(r["endpoint"]) + f" | {int(r['actual_words']):,} | {r['cycle_pos_m']:.3f} | " + " | ".join(
            f"{float(r.get('score__'+c)):.6f}" if r.get("score__"+c) is not None else "" for c in ["cheap7"] + CHEAP_COLUMNS
        ) + " |\n")
    lines.append("\n## Recent interval composition with score movement\n\n")
    lines.append("| interval | Δwords | cycle start→end (M) | official_base | inherited_qwen_pair | compact_pair_block | Δcheap7 | Δcheap6_noGP | Δrelation_state |\n")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|\n")
    for r in intervals:
        lines.append(
            f"| {r['from_endpoint']}→{r['to_endpoint']} | {int(r['delta_words']):,} | {r['cycle_start']/1_000_000:.3f}→{r['cycle_end']/1_000_000:.3f}{' wrap' if r['wraps_cycle'] else ''} | "
            f"{fmt_frac(r, 'family_frac__official_base')} | {fmt_frac(r, 'family_frac__inherited_qwen_pair')} | {fmt_frac(r, 'family_frac__compact_pair_block')} | "
            f"{(fnum(r.get('delta_score__cheap7')) if fnum(r.get('delta_score__cheap7')) is not None else '')} | "
            f"{(fnum(r.get('delta_score__cheap6_no_GlobalPIQA')) if fnum(r.get('delta_score__cheap6_no_GlobalPIQA')) is not None else '')} | "
            f"{(fnum(r.get('delta_score__relation_state')) if fnum(r.get('delta_score__relation_state')) is not None else '')} |\n"
        )
    lines.append("\nInterpretation: these interval/source phases are clues about deterministic replay geometry only. They do not by themselves prove that the 84M movement is caused by the immediately preceding source slice. The robust next evidence remains the pending SuperGLUE/grid completion and then seed43122 scoring.\n\n")
    lines.append(f"JSON: `{summary['output_files']['summary_json']}`\n")
    (args.out_dir / "epoch_phase_exposure_atlas.md").write_text("".join(lines), encoding="utf-8")

    print(json.dumps({
        "status": summary["status"],
        "pool_words": pool_summary["total_words"],
        "compact_pair_block_words": compact_positions["words"],
        "compact_first_m": compact_positions["cycle_start_m"],
        "compact_last_end_m": compact_positions["cycle_last_end_m"],
        "n_intervals": len(intervals),
        "out_json": summary["output_files"]["summary_json"],
        "out_md": summary["output_files"]["summary_md"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
