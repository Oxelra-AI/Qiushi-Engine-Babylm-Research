#!/usr/bin/env python3
"""research: file-only interpretation scale for incoming register pair.

Uses completed seed-spread, vector-noise, breadth/reference, RoBERTa-transfer, and
register-pool metadata to define how the coming register-displacement results
should be read.  No model loading, training, evaluation, GPU, or leaderboard.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative

import csv, json, math, pathlib, statistics, time
from typing import Any


def find_user_root() -> pathlib.Path:
    return _PUBLIC_ROOT

ROOT = find_user_root()
WS = ROOT / "experiments/archive/frontier_consolidation"
OUT = WS / "data/register_interpretation_scale"
SEED_SPREAD = WS / "data/full_deberta_seed_ladder_stable_eval/full_deberta_seed_ladder_seed_spread.csv"
SEED_DELTAS = WS / "data/full_deberta_seed_ladder_stable_eval/full_deberta_seed_ladder_treatment_deltas.csv"
VECTOR = WS / "data/axis_noise_floor_readout/seed_1x_vector_reproducibility.csv"
REF_SUMM = WS / "data/reference_decomposition_readout/contrast_window_summaries.csv"
REG_META = WS / "data/register_position_matched_direct_pools/register_position_matched_direct_pools_metadata.json"
ROBERTA_SUMMARY = WS / "data/roberta_transfer_readout/roberta_transfer_summary_rows.csv"

PRIMARY_METRICS = ["cheap6_no_GlobalPIQA", "cheap5_no_GlobalPIQA_Reading", "exEntity5", "Entity"]


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def rel(p: pathlib.Path) -> str:
    try:
        return str(p.relative_to(ROOT))
    except Exception:
        return str(p)


def fnum(x: Any) -> float | None:
    try:
        y = float(x)
        return y if math.isfinite(y) else None
    except Exception:
        return None


def read_csv(path: pathlib.Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({k for r in rows for k in r}) if rows else []
    with path.open("w", encoding="utf-8", newline="") as f:
        if fields:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader(); w.writerows(rows)


def seed_spread_rows() -> list[dict[str, Any]]:
    rows = read_csv(SEED_SPREAD)
    out = []
    ex_cols = ["BLiMP", "Supplement", "EWoK", "COMPS", "Reading"]

    def metric_spread_value(r: dict[str, str], metric: str) -> float | None:
        direct = fnum(r.get(f"treatment_delta_seed43122_minus_seed43022_{metric}"))
        if direct is not None:
            return direct
        if metric == "exEntity5":
            vals = [fnum(r.get(f"treatment_delta_seed43122_minus_seed43022_{c}")) for c in ex_cols]
            vals2 = [v for v in vals if v is not None]
            return statistics.mean(vals2) if len(vals2) == len(ex_cols) else None
        return None

    for metric in PRIMARY_METRICS:
        vals = []
        late_vals = []
        for r in rows:
            v = metric_spread_value(r, metric)
            if v is None:
                continue
            vals.append(v)
            w = int(float(r.get("words") or 0))
            if 70_000_000 <= w <= 100_000_000:
                late_vals.append(v)
        for window, xs in [("full10_100", vals), ("mature70_100", late_vals)]:
            if not xs:
                continue
            out.append({
                "kind": "same_data_seed_spread_of_compact_minus_repeat",
                "metric": metric,
                "window": window,
                "n": len(xs),
                "mean_abs_spread": statistics.mean(abs(x) for x in xs),
                "max_abs_spread": max(abs(x) for x in xs),
                "signed_mean_spread": statistics.mean(xs),
                "range_min": min(xs),
                "range_max": max(xs),
            })
    return out


def vector_rows() -> list[dict[str, Any]]:
    out = []
    for r in read_csv(VECTOR):
        if r.get("window") in {"common10_80", "mature70_100", "late80_100"} and r.get("family_set") in {"stable6", "stable5_exEntity"}:
            out.append({
                "kind": "same_data_seed_vector_noise",
                "window": r.get("window"),
                "family_set": r.get("family_set"),
                "n_checkpoints": int(float(r.get("n_checkpoints") or 0)),
                "seed43022_net": fnum(r.get("seed43022_net")),
                "seed43122_net": fnum(r.get("seed43122_net")),
                "seed43022_rms": fnum(r.get("seed43022_rms")),
                "seed43122_rms": fnum(r.get("seed43122_rms")),
                "difference_l2": fnum(r.get("difference_l2")),
                "difference_over_mean_seed_l2": fnum(r.get("difference_over_mean_seed_l2")),
                "cosine": fnum(r.get("cosine")),
            })
    return out


def ref_rows() -> list[dict[str, Any]]:
    wanted = {
        ("D1_VminusB", "exEntity5", "common10_80"),
        ("D1_VminusB", "exEntity5", "late80_100"),
        ("D1_VminusCold", "exEntity5", "common10_80"),
        ("D1_VminusCmax", "exEntity5", "late80_100"),
        ("D1_BminusCold", "exEntity5", "common10_80"),
        ("D1_BminusCmax", "exEntity5", "late80_100"),
        ("D1_VminusCold", "cheap6_no_GlobalPIQA", "common10_80"),
        ("D1_VminusCmax", "cheap6_no_GlobalPIQA", "late80_100"),
        ("Rbt_VminusC", "exEntity5", "late80_100"),
        ("Rbt_VminusC", "cheap6_no_GlobalPIQA", "late80_100"),
    }
    out = []
    for r in read_csv(REF_SUMM):
        key = (r.get("contrast"), r.get("quantity"), r.get("window"))
        if key in wanted:
            out.append({
                "kind": "reference_effect_scale",
                "contrast": r.get("contrast"),
                "metric": r.get("quantity"),
                "window": r.get("window"),
                "n": int(float(r.get("n") or 0)),
                "mean": fnum(r.get("mean")),
                "median": fnum(r.get("median")),
                "min": fnum(r.get("min")),
                "max": fnum(r.get("max")),
                "checkpoints": r.get("checkpoints"),
            })
    return out


def roberta_rows() -> list[dict[str, Any]]:
    out = []
    for r in read_csv(ROBERTA_SUMMARY):
        if r.get("contrast") == "view_minus_clean" and r.get("metric") in {"cheap6_no_GlobalPIQA", "cheap5_no_GlobalPIQA_Reading", "Entity", "EWoK"} and r.get("window") in {"common_10M_80M", "available_all"}:
            out.append({
                "kind": "roberta_transfer_scale_legacy_summary",
                "contrast": r.get("contrast"),
                "metric": r.get("metric"),
                "window": r.get("window"),
                "n": int(float(r.get("n") or 0)),
                "mean": fnum(r.get("mean")),
                "min": fnum(r.get("min")),
                "max": fnum(r.get("max")),
            })
    return out


def register_meta_rows() -> list[dict[str, Any]]:
    meta = json.loads(REG_META.read_text(encoding="utf-8"))
    out = []
    for arm in meta.get("arm_results", []):
        ds = arm.get("displaced_clean_summary") or {}
        out.append({
            "kind": "incoming_register_arm_design",
            "arm": arm.get("arm_key"),
            "rho": arm.get("rho"),
            "active_fineweb_words": arm.get("active_fineweb_words"),
            "displaced_words": ds.get("words"),
            "child_sub_fraction": ds.get("child_sub_fraction"),
            "adult_gut_simple_fraction": ds.get("adult_gut_simple_fraction"),
            "row_index_mean": ds.get("row_index_mean"),
            "row_index_sd": ds.get("row_index_sd"),
            "sha_100m": arm.get("sha_100m"),
        })
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    seed_rows = seed_spread_rows()
    vec = vector_rows()
    ref = ref_rows()
    rbt = roberta_rows()
    reg = register_meta_rows()

    # Useful scalar anchors for prose
    def get_ref(contrast: str, metric: str, window: str) -> float | None:
        for r in ref:
            if r.get("contrast") == contrast and r.get("metric") == metric and r.get("window") == window:
                return r.get("mean")
        return None
    def get_spread(metric: str, window: str) -> float | None:
        for r in seed_rows:
            if r.get("metric") == metric and r.get("window") == window:
                return r.get("mean_abs_spread")
        return None

    anchors = {
        "old_D1_VminusB_exEntity5_common10_80": get_ref("D1_VminusB", "exEntity5", "common10_80"),
        "old_D1_VminusB_exEntity5_late80_100": get_ref("D1_VminusB", "exEntity5", "late80_100"),
        "old_D1_BminusCold_exEntity5_common10_80": get_ref("D1_BminusCold", "exEntity5", "common10_80"),
        "old_D1_VminusCold_exEntity5_common10_80": get_ref("D1_VminusCold", "exEntity5", "common10_80"),
        "matched_D1_VminusCmax_exEntity5_late80_100": get_ref("D1_VminusCmax", "exEntity5", "late80_100"),
        "roberta_Rbt_VminusC_exEntity5_late80_100": get_ref("Rbt_VminusC", "exEntity5", "late80_100"),
        "roberta_Rbt_VminusC_cheap6_late80_100": get_ref("Rbt_VminusC", "cheap6_no_GlobalPIQA", "late80_100"),
        "seed_spread_cheap6_full10_100_mean_abs": get_spread("cheap6_no_GlobalPIQA", "full10_100"),
        "seed_spread_cheap6_mature70_100_mean_abs": get_spread("cheap6_no_GlobalPIQA", "mature70_100"),
        "seed_spread_exEntity5_full10_100_mean_abs": get_spread("exEntity5", "full10_100"),
        "seed_spread_exEntity5_mature70_100_mean_abs": get_spread("exEntity5", "mature70_100"),
    }

    all_rows = seed_rows + vec + ref + rbt + reg
    write_csv(OUT / "interpretation_scale_rows.csv", all_rows)
    result = {
        "status": "REGISTER_INTERPRETATION_SCALE_COMPLETE",
        "created_utc": now(),
        "boundary": "File-only synthesis from existing tables; no model loading, training, evaluation, GPU, GlobalPIQA, SuperGLUE, AoA, upload, or leaderboard.",
        "anchors": anchors,
        "files": {
            "scale_rows": rel(OUT / "interpretation_scale_rows.csv"),
            "summary_json": rel(OUT / "register_interpretation_scale_summary.json"),
            "summary_md": rel(OUT / "register_interpretation_scale_summary.md"),
        },
        "source_files": [rel(SEED_SPREAD), rel(SEED_DELTAS), rel(VECTOR), rel(REF_SUMM), rel(REG_META), rel(ROBERTA_SUMMARY)],
        "scientific_reading": [
            "The incoming pair should be read primarily as child/subtitle-removed minus adult-prose-removed at identical admitted FineWeb text and rho, not as another mixed-dose point.",
            "A mixed quarter_1x null does not cancel the register pair; it increases its value because the mixed arm can hide opposite removal-side effects.",
            "Old broad companion specificity is weak: V-B exEntity5 is near zero or negative; therefore a register result should be interpreted as fixed-budget stream/register substitution, not semantic companion correspondence.",
            "Any DeBERTa-positive admission result is coordinate-conditional because RoBERTa MAX view-clean is negative late outside Entity and cheap6.",
            "Use common-window means and family vectors; do not promote a single checkpoint or Entity-only movement to a general principle.",
        ],
    }
    (OUT / "register_interpretation_scale_summary.json").write_text(json.dumps(result, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")

    def fmt(x: Any) -> str:
        y = fnum(x)
        return "" if y is None else f"{y:+.4f}"
    lines = [
        "# research register interpretation scale",
        "",
        result["boundary"],
        "",
        "## Scalar anchors",
        "",
        f"- Old DeBERTa MAX companion-form specificity, V-B exEntity5 common10_80: {fmt(anchors['old_D1_VminusB_exEntity5_common10_80'])}; late80_100: {fmt(anchors['old_D1_VminusB_exEntity5_late80_100'])}.",
        f"- Old DeBERTa broad stream/composition leg, B-C_old exEntity5 common10_80: {fmt(anchors['old_D1_BminusCold_exEntity5_common10_80'])}.",
        f"- Old DeBERTa total V-C_old exEntity5 common10_80: {fmt(anchors['old_D1_VminusCold_exEntity5_common10_80'])}; matched-clean MAX late V-Cmax exEntity5: {fmt(anchors['matched_D1_VminusCmax_exEntity5_late80_100'])}.",
        f"- RoBERTa MAX view-clean late exEntity5: {fmt(anchors['roberta_Rbt_VminusC_exEntity5_late80_100'])}; late cheap6: {fmt(anchors['roberta_Rbt_VminusC_cheap6_late80_100'])}.",
        f"- Two-seed same-data treatment-delta mean absolute spread: cheap6 full10_100 {fmt(anchors['seed_spread_cheap6_full10_100_mean_abs'])}, mature70_100 {fmt(anchors['seed_spread_cheap6_mature70_100_mean_abs'])}; exEntity5 full10_100 {fmt(anchors['seed_spread_exEntity5_full10_100_mean_abs'])}, mature70_100 {fmt(anchors['seed_spread_exEntity5_mature70_100_mean_abs'])}.",
        "",
        "## Register design",
        "",
        "| arm | rho | FineWeb words | displaced child/sub frac | displaced adult frac | row mean |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for r in reg:
        lines.append(f"| {r['arm']} | {r['rho']:.6f} | {int(r['active_fineweb_words'])} | {r['child_sub_fraction']:.4f} | {r['adult_gut_simple_fraction']:.4f} | {r['row_index_mean']:.1f} |")
    lines += [
        "",
        "## Reading rules for incoming scores",
        "",
        "The primary contrast is adult-prose-removal versus child/subtitle-removal at identical admitted FineWeb. A mixed quarter_1x null should be treated as potentially cancelled signal, not as an automatic stop.",
        "",
        "If both arms agree and both beat the two-clean anchor on cheap6/exEntity5 over the common window, the DeBERTa result supports a small admixture/register-presence effect. If they agree near zero or below clean, the old admission leg is not stable at rho≈0.011 under this control. If they diverge, the principle must include the value of the removed clean register; direct child-minus-adult sign tells which register is costly to sacrifice.",
        "",
        "Entity must remain separate because previous positive Entity movement was operation-skewed. RoBERTa negative late V-C keeps any positive DeBERTa register law coordinate-conditional until transfer is tested or explained.",
        "",
        f"CSV: `{rel(OUT / 'interpretation_scale_rows.csv')}`",
        f"JSON: `{rel(OUT / 'register_interpretation_scale_summary.json')}`",
    ]
    (OUT / "register_interpretation_scale_summary.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "anchors": anchors, "summary_md": rel(OUT / "register_interpretation_scale_summary.md")}, indent=2, ensure_ascii=False), flush=True)

if __name__ == "__main__":
    main()
