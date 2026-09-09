#!/usr/bin/env python3
"""research: small-sample phase-lag analysis for the reference late grid.

Uses the research epoch-phase/source-exposure atlas plus recovered selected cheap7
scores to test whether the 84M endpoint can be explained by a simple recent source
phase (especially compact-pair-block exposure at the start of each 10M cycle).
This is descriptive CPU/file analysis only; it is not a causal training result.
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
from statistics import mean, pstdev
from typing import Any

CHEAP_COLUMNS = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA", "Reading"]
SCORE_KEYS = ["cheap7", "cheap6_no_GlobalPIQA", "cheap5_no_GP_Reading", "relation_state"]
FEATURES = [
    "family_frac__compact_pair_block",
    "family_frac__inherited_qwen_pair",
    "family_frac__official_base",
]


def find_user_root() -> pathlib.Path:
    return _PUBLIC_ROOT


USER_ROOT = find_user_root()
WORKSPACE = USER_ROOT / "experiments/archive/frontier_consolidation"
DEFAULT_ATLAS = WORKSPACE / "data/epoch_phase_exposure_atlas/epoch_phase_exposure_atlas.json"


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def rel(path: pathlib.Path | str) -> str:
    p = pathlib.Path(path)
    try:
        return str(p.resolve().relative_to(USER_ROOT))
    except Exception:
        return str(p)


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


def pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 3 or len(xs) != len(ys):
        return None
    mx, my = mean(xs), mean(ys)
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx <= 0 or vy <= 0:
        return None
    return float(sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / math.sqrt(vx * vy))


def read_json(path: pathlib.Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: pathlib.Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_csv(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    keys: list[str] = []
    seen = set()
    for r in rows:
        for k in r:
            if k not in seen:
                keys.append(k); seen.add(k)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=keys)
        wr.writeheader()
        for r in rows:
            wr.writerow({k: r.get(k, "") for k in keys})


def interval_records(atlas: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for r in atlas.get("interval_rows", []):
        rec = dict(r)
        for f in FEATURES:
            rec[f] = fnum(rec.get(f)) or 0.0
        for key in SCORE_KEYS:
            rec[f"delta_score__{key}"] = fnum(rec.get(f"delta_score__{key}"))
        rows.append(rec)
    return rows


def scored_intervals(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [r for r in rows if fnum(r.get("delta_score__cheap7")) is not None]


def summarize_by_feature_threshold(rows: list[dict[str, Any]], feature: str, threshold: float) -> dict[str, Any]:
    hi = [r for r in rows if (fnum(r.get(feature)) or 0.0) >= threshold]
    lo = [r for r in rows if (fnum(r.get(feature)) or 0.0) < threshold]
    def block(rs: list[dict[str, Any]]) -> dict[str, Any]:
        out: dict[str, Any] = {"n": len(rs), "intervals": [f"{r['from_endpoint']}→{r['to_endpoint']}" for r in rs]}
        for key in SCORE_KEYS:
            vals = [fnum(r.get(f"delta_score__{key}")) for r in rs]
            vals = [float(v) for v in vals if v is not None]
            out[f"mean_delta_{key}"] = float(mean(vals)) if vals else None
            out[f"std_delta_{key}"] = float(pstdev(vals)) if len(vals) > 1 else (0.0 if vals else None)
            out[f"positive_delta_{key}"] = sum(1 for v in vals if v > 0)
        return out
    return {"feature": feature, "threshold": threshold, "high": block(hi), "low": block(lo)}


def lagged_feature_correlations(rows: list[dict[str, Any]], max_lag: int = 2) -> list[dict[str, Any]]:
    out = []
    for lag in range(0, max_lag + 1):
        for feat in FEATURES:
            for skey in SCORE_KEYS:
                xs, ys = [], []
                for i, r in enumerate(rows):
                    j = i - lag
                    if j < 0:
                        continue
                    y = fnum(r.get(f"delta_score__{skey}"))
                    x = fnum(rows[j].get(feat))
                    if x is not None and y is not None:
                        xs.append(float(x)); ys.append(float(y))
                out.append({
                    "lag_intervals": lag,
                    "feature": feat,
                    "score_delta": skey,
                    "n": len(xs),
                    "pearson_r": pearson(xs, ys),
                    "feature_mean": float(mean(xs)) if xs else None,
                    "score_delta_mean": float(mean(ys)) if ys else None,
                })
    return out


def make_plot(atlas: dict[str, Any], out_png: pathlib.Path) -> dict[str, Any]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        return {"ok": False, "error": repr(e)}
    eps = []
    for r in atlas.get("endpoint_phase_rows", []):
        c7 = fnum(r.get("score__cheap7"))
        if c7 is not None:
            eps.append((float(r["actual_words"]) / 1_000_000.0, str(r["endpoint"]), float(r["cycle_pos_m"]), c7))
    intervals = scored_intervals(interval_records(atlas))
    fig, ax1 = plt.subplots(figsize=(10.5, 4.8))
    xs = [e[0] for e in eps]
    ys = [e[3] for e in eps]
    ax1.plot(xs, ys, marker="o", color="#1f77b4", lw=2, label="cheap7")
    for x, ep, cyc, y in eps:
        if ep in {"chck_82M", "chck_84M", "chck_100M"}:
            ax1.annotate(ep.replace("chck_", ""), (x, y), textcoords="offset points", xytext=(4, 7), fontsize=8)
    for r in intervals:
        cf = fnum(r.get("family_frac__compact_pair_block")) or 0.0
        if cf >= 0.10:
            ax1.axvspan(float(r["actual_start"]) / 1_000_000.0, float(r["actual_end"]) / 1_000_000.0, color="#2ca02c", alpha=0.12)
    ax1.set_xlabel("actual cumulative words (M)")
    ax1.set_ylabel("cheap7")
    ax1.set_title("Reference scale1.75 late trajectory: compact-rich source phases shaded")
    ax1.grid(alpha=0.25)
    ax2 = ax1.twinx()
    ax2.plot(xs, [e[2] for e in eps], marker=".", color="#d62728", alpha=0.65, label="cycle position")
    ax2.set_ylabel("position within repeated 10M cycle (M)")
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc="lower left")
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=170)
    plt.close(fig)
    return {"ok": True, "path": rel(out_png), "size_bytes": out_png.stat().st_size}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--atlas", type=pathlib.Path, default=DEFAULT_ATLAS)
    ap.add_argument("--out-dir", type=pathlib.Path, required=True)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    atlas = read_json(args.atlas)
    rows = interval_records(atlas)
    scored = scored_intervals(rows)

    threshold_summaries = [
        summarize_by_feature_threshold(scored, "family_frac__compact_pair_block", 0.10),
        summarize_by_feature_threshold(scored, "family_frac__compact_pair_block", 0.01),
        summarize_by_feature_threshold(scored, "family_frac__inherited_qwen_pair", 0.17),
    ]
    lags = lagged_feature_correlations(scored, max_lag=2)
    plot = make_plot(atlas, args.out_dir / "reference_late_phase_cheap7.png")

    # Directly compare the local sequence around the 84M peak.
    local = []
    for r in scored:
        if r.get("from_endpoint") in {"chck_78M", "chck_80M", "chck_82M", "chck_84M", "chck_86M", "chck_88M", "chck_90M"}:
            local.append({
                "interval": f"{r['from_endpoint']}→{r['to_endpoint']}",
                "compact_frac": r.get("family_frac__compact_pair_block", 0.0),
                "qwen_frac": r.get("family_frac__inherited_qwen_pair", 0.0),
                "official_frac": r.get("family_frac__official_base", 0.0),
                "delta_cheap7": r.get("delta_score__cheap7"),
                "delta_cheap6_no_GlobalPIQA": r.get("delta_score__cheap6_no_GlobalPIQA"),
                "delta_relation_state": r.get("delta_score__relation_state"),
            })

    summary = {
        "status": "PHASE_LAG_SCORE_ANALYSIS",
        "created_utc": now(),
        "atlas": rel(args.atlas),
        "n_scored_intervals": len(scored),
        "threshold_summaries": threshold_summaries,
        "lagged_feature_correlations": lags,
        "local_78_to_92M_sequence": local,
        "plot": plot,
        "scientific_reading": (
            "The compact block is deterministically at the start of each 10M cycle, and compact-rich intervals sometimes precede gains, but this small late-window sequence does not support a simple recent-source-slice explanation of the 84M peak: the 82M→84M gain itself occurs in a no-compact interval, other no-compact intervals can be positive or negative, and compact-rich intervals are not uniformly broad-positive. Treat source phase as geometry context, not causal evidence."
        ),
        "output_files": {
            "summary_json": rel(args.out_dir / "phase_lag_score_analysis.json"),
            "summary_md": rel(args.out_dir / "phase_lag_score_analysis.md"),
            "correlations_csv": rel(args.out_dir / "lagged_correlations.csv"),
            "local_sequence_csv": rel(args.out_dir / "local_78_92M_sequence.csv"),
            "plot_png": plot.get("path") if isinstance(plot, dict) else None,
        },
    }
    write_json(args.out_dir / "phase_lag_score_analysis.json", summary)
    write_csv(args.out_dir / "lagged_correlations.csv", lags)
    write_csv(args.out_dir / "local_78_92M_sequence.csv", local)

    lines = ["# research phase-lag score analysis\n\n"]
    lines.append(f"Scored intervals: {len(scored)}. Source atlas: `{rel(args.atlas)}`.\n\n")
    lines.append("## Compact-rich versus compact-poor intervals\n\n")
    for block in threshold_summaries[:2]:
        lines.append(f"Threshold `{block['feature']} >= {block['threshold']}`:\n")
        for name in ["high", "low"]:
            b = block[name]
            lines.append(f"- {name}: n={b['n']}, intervals={b['intervals']}, mean Δcheap7={b.get('mean_delta_cheap7')}, mean Δcheap6_noGP={b.get('mean_delta_cheap6_no_GlobalPIQA')}, mean Δrelation_state={b.get('mean_delta_relation_state')}\n")
        lines.append("\n")
    lines.append("## Local 78M--92M sequence\n\n")
    lines.append("| interval | compact frac | qwen frac | official frac | Δcheap7 | Δcheap6_noGP | Δrelation_state |\n")
    lines.append("|---|---:|---:|---:|---:|---:|---:|\n")
    for r in local:
        lines.append(f"| {r['interval']} | {100*float(r['compact_frac']):.2f}% | {100*float(r['qwen_frac']):.2f}% | {100*float(r['official_frac']):.2f}% | {r['delta_cheap7']} | {r['delta_cheap6_no_GlobalPIQA']} | {r['delta_relation_state']} |\n")
    lines.append("\n## Reading\n\n")
    lines.append(summary["scientific_reading"] + "\n\n")
    if plot.get("ok"):
        lines.append(f"Figure: `{plot['path']}`\n\n")
    lines.append(f"JSON: `{summary['output_files']['summary_json']}`\n")
    (args.out_dir / "phase_lag_score_analysis.md").write_text("".join(lines), encoding="utf-8")

    print(json.dumps({"status": summary["status"], "n_scored_intervals": len(scored), "plot_ok": plot.get("ok"), "out_json": summary["output_files"]["summary_json"], "out_md": summary["output_files"]["summary_md"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
