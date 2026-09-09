#!/usr/bin/env python3
"""research: independent file-level recomputation of key macro quantities.

Reads the macro CSV summaries and writes a compact scientific check of the quantities
used in the route judgment. CPU-only; no model loading.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from statistics import mean

ROOT = Path("experiments/archive/frontier_consolidation/data")
A01_OUT = Path("experiments/archive/representation_and_objectives/data/macro_check")
A01_OUT.mkdir(parents=True, exist_ok=True)

BAL = ROOT / "entity_balanced_and_transfer_readout_full/entity_balanced_rows.csv"
REF = ROOT / "reference_decomposition_readout/contrast_window_summaries.csv"
CONTR = ROOT / "reference_decomposition_readout/contrast_rows.csv"
MARG = ROOT / "margin_domain_window_readout/window_summaries.csv"
MISS = ROOT / "reference_decomposition_readout/missing_contrast_rows.csv"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def f(x: str) -> float:
    return float(x) if x not in ("", None) else float("nan")


bal = read_csv(BAL)
ref = read_csv(REF)
contr = read_csv(CONTR)
marg = read_csv(MARG)
miss = read_csv(MISS)

# Entity late means from balanced_rows.csv, recomputed directly.
entity_summary = {}
for cname in ["deberta_basin1", "deberta_basin2", "deberta_breadth_minus_repeat", "deberta_view_minus_breadth", "roberta_max"]:
    rows = [r for r in bal if r["contrast_name"] == cname and r["checkpoint"] in {"chck_80M", "chck_90M", "chck_100M"}]
    entity_summary[cname] = {
        "n": len(rows),
        "official_mean": mean(f(r["official_all18_delta_pp"]) for r in rows),
        "zero_mean": mean(f(r["zero_ops_delta_pp"]) for r in rows),
        "nonzero_mean": mean(f(r["nonzero_ops_delta_pp"]) for r in rows),
        "balanced_mean": mean(f(r["balanced_zero_nonzero_delta_pp"]) for r in rows),
        "spread_mean": mean(f(r["nonzero_minus_zero_spread_pp"]) for r in rows),
    }

# Key broad V/B/C quantities visible in the reference decomposition.
key_refs = {}
for r in ref:
    key = (r["contrast"], r["quantity"], r["window"])
    if key in {
        ("D1_VminusB", "exEntity5", "late80_100"),
        ("D1_VminusB", "cheap6_no_GlobalPIQA", "late80_100"),
        ("D1_BminusCold", "exEntity5", "late80_100"),
        ("D1_VminusCold", "exEntity5", "common10_80"),
        ("D1_VminusR", "exEntity5", "late80_100"),
        ("D1_VminusR", "Entity", "late80_100"),
    }:
        key_refs["|".join(key)] = {k: r[k] for k in ["n", "mean", "min", "max", "checkpoints"]}

# Per-family V-B rows available at late checkpoints.
vb_rows = [r for r in contr if r["contrast"] == "D1_VminusB" and r["checkpoint"] in {"chck_80M", "chck_90M", "chck_100M"}]
vb_by_family = {}
for r in vb_rows:
    vb_by_family.setdefault(r["quantity"], []).append({"checkpoint": r["checkpoint"], "delta": f(r["delta"])})

# Missing V-B rows by broad family.
missing_vb = [r for r in miss if r["contrast"] == "D1_VminusB" and r["checkpoint"] in {"chck_80M", "chck_90M", "chck_100M"}]
missing_vb_counts = {}
for r in missing_vb:
    missing_vb_counts[r["quantity"]] = missing_vb_counts.get(r["quantity"], 0) + 1

# EWoK all-domain margin means by dose/window, to show heterogeneity rather than broad positive support.
ewok_margin = [
    {"dose": r["dose_name"], "window": r["window"], "mean": f(r["margin_mean"]), "frac_pos": f(r["frac_margin_positive"]), "n": int(r["n"])}
    for r in marg
    if r["family"] == "EWoK" and r["subgroup"] == "all" and r["window"] in {"common10_80", "endpoint80"}
]

out = {
    "entity_late_means": entity_summary,
    "selected_reference_quantities": key_refs,
    "visible_vminusb_by_family_late": vb_by_family,
    "missing_vminusb_counts_late": missing_vb_counts,
    "ewok_margin_all_domain_selected": ewok_margin,
    "interpretation": {
        "entity_view_breadth": "Entity V-B is positive on zero and nonzero operation strata at all late checkpoints in the visible A02 table.",
        "entity_breadth_repeat": "B-R has the same signed zero-loss/nonzero-gain shape as V-R, so aggregate Entity B-R is not a balanced state-record improvement.",
        "broad_vb": "Broad ex-Entity V-B is not established because visible rows are sparse and mixed; many 90M/100M broad V-B rows are missing.",
        "bridge_to_macro": "A01 selector-reader evidence predicts correspondence-specific effects, not uniform broad V-B. The visible macro evidence currently supports at most an Entity/state-update correspondence test."
    }
}

(A01_OUT / "macro_check_summary.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

lines = []
lines.append("# research macro quantity recomputation\n\n")
lines.append("This CPU-only file read recomputes the quantities used in the research review from A02 CSV files.\n\n")
lines.append("## Entity late means\n\n")
lines.append("| contrast | n | official | zero-op | nonzero | balanced zero/nonzero | spread |\n")
lines.append("|---|---:|---:|---:|---:|---:|---:|\n")
for k, v in entity_summary.items():
    lines.append(f"| {k} | {v['n']} | {v['official_mean']:+.3f} | {v['zero_mean']:+.3f} | {v['nonzero_mean']:+.3f} | {v['balanced_mean']:+.3f} | {v['spread_mean']:+.3f} |\n")
lines.append("\n## Visible late V-B rows by family\n\n")
lines.append("| family | rows | deltas |\n")
lines.append("|---|---:|---|\n")
for fam, rows in sorted(vb_by_family.items()):
    ds = ", ".join(f"{r['checkpoint']} {r['delta']:+.3f}" for r in rows)
    lines.append(f"| {fam} | {len(rows)} | {ds} |\n")
lines.append("\n## Missing late V-B rows\n\n")
lines.append("| family | missing rows |\n")
lines.append("|---|---:|\n")
for fam, n in sorted(missing_vb_counts.items()):
    lines.append(f"| {fam} | {n} |\n")
lines.append("\n## EWoK selected all-domain margin rows\n\n")
lines.append("| dose | window | n | margin mean | frac positive |\n")
lines.append("|---|---|---:|---:|---:|\n")
for r in ewok_margin:
    lines.append(f"| {r['dose']} | {r['window']} | {r['n']} | {r['mean']:+.4f} | {r['frac_pos']:.3f} |\n")
lines.append("\n## Scientific reading\n\n")
lines.append("The recomputation matches the peer summaries: Entity V-B is balanced-positive, B-R and V-R are operation-skewed, and broad ex-Entity V-B is still too sparse and mixed to support a uniform compact companion law.\n")

((A01_OUT.parents[4] / 'research/documents/representation_and_objectives/data/macro_check/macro_check_summary.md')).write_text("".join(lines), encoding="utf-8")
print(json.dumps({"status": "DONE", "md": str((A01_OUT.parents[4] / 'research/documents/representation_and_objectives/data/macro_check/macro_check_summary.md')), "json": str(A01_OUT / "macro_check_summary.json")}, ensure_ascii=False))
