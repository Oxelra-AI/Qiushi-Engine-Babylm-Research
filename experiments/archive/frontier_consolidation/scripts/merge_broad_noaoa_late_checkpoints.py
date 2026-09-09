#!/usr/bin/env python3
"""Merge research broad no-AoA late-checkpoint evaluations.

This CPU-only analyzer waits for no model loading.  It combines the two split
output roots (seed43022 and seed43122) produced by
`training/scripts/broad_noaoa_late_checkpoints.py`, compares
selected checkpoints with known 100M fast references, and writes a scientific
summary for deciding whether earlier stopping or checkpoint averaging is worth
further evaluation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Any, Dict, Optional

ROOT = Path.cwd()
STUDY = ROOT / "experiments/archive/frontier_consolidation"
DEFAULT_ROOTS = [
    STUDY / "data/broad_noaoa_seed43022",
    STUDY / "data/broad_noaoa_seed43122",
]
OUT_DIR_DEFAULT = STUDY / "data/broad_noaoa_late_checkpoint_merge"

TABLE_KEYS = [
    "BLiMP", "Supplement", "EWoK", "Entity", "Entity_full", "COMPS",
    "GlobalPIQA_parallel", "GlobalPIQA_nonparallel", "GlobalPIQA_mean",
    "Reading", "Reading_eye", "Reading_self_paced", "equal7_mean", "equal7_full_entity",
]

KNOWN_100M = {
    "r43022_100M_fast": {
        "family": "compact_view_reinvest", "seed": "43022", "exposure_m": 100,
        "source": "experiments/archive/frontier_consolidation/data/density_noaoa_eval_reinvest/per_target/compact_view_reinvest.json",
        "BLiMP": 66.63, "Supplement": 66.4, "EWoK": 53.09, "Entity": 28.07,
        "Entity_full": 27.75, "COMPS": 51.97, "GlobalPIQA_parallel": 25.24,
        "GlobalPIQA_nonparallel": 46.0, "GlobalPIQA_mean": 35.62,
        "Reading": 8.24, "Reading_eye": 11.15, "Reading_self_paced": 5.33,
    },
    "r43122_100M_fast": {
        "family": "compact_view_reinvest", "seed": "43122", "exposure_m": 100,
        "source": "experiments/archive/representation_and_objectives/data/compact_reinvest_seed43122_fast/compact_view_reinvest_seed43122_fast_summary.json",
        "BLiMP": 65.75, "Supplement": 63.2, "EWoK": 49.36, "Entity": 26.68,
        "Entity_full": 26.29, "COMPS": 51.54, "GlobalPIQA_parallel": 24.27,
        "GlobalPIQA_nonparallel": 46.0, "GlobalPIQA_mean": 35.135,
        "Reading": 8.865, "Reading_eye": 12.87, "Reading_self_paced": 4.86,
    },
}

TARGET_META = {
    "r43022_80M": {"seed": "43022", "exposure_m": 80},
    "r43022_90M": {"seed": "43022", "exposure_m": 90},
    "r43122_45M": {"seed": "43122", "exposure_m": 45},
    "r43122_80M": {"seed": "43122", "exposure_m": 80},
}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", nargs="*", type=Path, default=DEFAULT_ROOTS)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR_DEFAULT)
    ap.add_argument("--allow-partial", action="store_true")
    return ap.parse_args()


def scores_from_payload(raw: Dict[str, Any]) -> Dict[str, Optional[float]]:
    out: Dict[str, Optional[float]] = {k: None for k in TABLE_KEYS}
    tasks = raw.get("tasks", {})
    for col in ["BLiMP", "Supplement", "EWoK", "Entity", "Entity_full", "COMPS", "GlobalPIQA_parallel", "GlobalPIQA_nonparallel"]:
        if col in tasks:
            out[col] = tasks[col].get("score")
    if "Reading" in tasks:
        for k, v in tasks["Reading"].get("scores", {}).items():
            out[k] = v
    if out.get("GlobalPIQA_parallel") is not None and out.get("GlobalPIQA_nonparallel") is not None:
        out["GlobalPIQA_mean"] = (out["GlobalPIQA_parallel"] + out["GlobalPIQA_nonparallel"]) / 2.0  # type: ignore[operator]
    eq_keys = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA_mean", "Reading"]
    if all(out.get(k) is not None for k in eq_keys):
        out["equal7_mean"] = sum(float(out[k]) for k in eq_keys) / len(eq_keys)  # type: ignore[arg-type]
    eq_full_keys = ["BLiMP", "Supplement", "EWoK", "Entity_full", "COMPS", "GlobalPIQA_mean", "Reading"]
    if all(out.get(k) is not None for k in eq_full_keys):
        out["equal7_full_entity"] = sum(float(out[k]) for k in eq_full_keys) / len(eq_full_keys)  # type: ignore[arg-type]
    return out


def add_composites(scores: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(scores)
    if "GlobalPIQA_mean" not in out and out.get("GlobalPIQA_parallel") is not None and out.get("GlobalPIQA_nonparallel") is not None:
        out["GlobalPIQA_mean"] = (float(out["GlobalPIQA_parallel"]) + float(out["GlobalPIQA_nonparallel"])) / 2.0
    eq_keys = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA_mean", "Reading"]
    if all(k in out and out[k] is not None for k in eq_keys):
        out["equal7_mean"] = sum(float(out[k]) for k in eq_keys) / len(eq_keys)
    eq_full_keys = ["BLiMP", "Supplement", "EWoK", "Entity_full", "COMPS", "GlobalPIQA_mean", "Reading"]
    if all(k in out and out[k] is not None for k in eq_full_keys):
        out["equal7_full_entity"] = sum(float(out[k]) for k in eq_full_keys) / len(eq_full_keys)
    return out


def diff(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, float]:
    out = {}
    for k in TABLE_KEYS:
        if a.get(k) is not None and b.get(k) is not None:
            out[k] = round(float(a[k]) - float(b[k]), 4)
    return out


def load_all(roots: list[Path]) -> tuple[Dict[str, Dict[str, Any]], list[str]]:
    raw: Dict[str, Dict[str, Any]] = {}
    missing = []
    for root in roots:
        per_dir = root / "per_target"
        if not per_dir.exists():
            missing.append(f"missing per_target directory: {per_dir}")
            continue
        for p in sorted(per_dir.glob("*.json")):
            try:
                raw[p.stem] = json.loads(p.read_text(encoding="utf-8"))
            except Exception as e:
                missing.append(f"failed to read {p}: {e!r}")
    for t in TARGET_META:
        if t not in raw:
            missing.append(f"missing target payload: {t}")
    return raw, missing


def build_scientific_read(table: Dict[str, Dict[str, Any]], comparisons: Dict[str, Dict[str, float]], missing: list[str]) -> Dict[str, str]:
    lines: Dict[str, str] = {}
    required = ["r43022_80M", "r43122_80M", "r43022_90M", "r43122_45M"]
    incomplete = []
    for t in required:
        if t not in table:
            incomplete.append(f"missing target table: {t}")
        elif table[t].get("equal7_full_entity") is None:
            incomplete.append(f"incomplete scores for {t}: equal7_full_entity is absent")
    if missing or incomplete:
        lines["partial"] = "Some target payloads or scores are absent, so no stopping or averaging decision should be made from this merge alone."
        if incomplete:
            lines["incomplete_scores"] = "; ".join(incomplete)
        return lines
    # Candidate 80M common stop: compare each seed to its own 100M and compare across seeds.
    r430_80 = table["r43022_80M"]
    r431_80 = table["r43122_80M"]
    r430_100 = add_composites(KNOWN_100M["r43022_100M_fast"])
    r431_100 = add_composites(KNOWN_100M["r43122_100M_fast"])
    d430 = diff(r430_80, r430_100)
    d431 = diff(r431_80, r431_100)
    min80 = min(float(r430_80["equal7_full_entity"]), float(r431_80["equal7_full_entity"]))
    min100 = min(float(r430_100["equal7_full_entity"]), float(r431_100["equal7_full_entity"]))
    mean80 = (float(r430_80["equal7_full_entity"]) + float(r431_80["equal7_full_entity"])) / 2
    mean100 = (float(r430_100["equal7_full_entity"]) + float(r431_100["equal7_full_entity"])) / 2
    lines["common_80M"] = (
        f"At 80M, equal7_full_entity seed-min {min80:.3f} vs known 100M seed-min {min100:.3f}, "
        f"seed-mean {mean80:.3f} vs {mean100:.3f}; seed43022 delta {d430.get('equal7_full_entity')} and "
        f"seed43122 delta {d431.get('equal7_full_entity')}."
    )
    # Explain if Supplement-only improvement bought losses elsewhere.
    for target, ref in [("r43022_80M", "r43022_100M_fast"), ("r43122_80M", "r43122_100M_fast"), ("r43122_45M", "r43122_100M_fast"), ("r43022_90M", "r43022_100M_fast")]:
        comp = comparisons.get(f"{target}_minus_{ref}", {})
        bad = [k for k in ["EWoK", "Entity_full", "COMPS", "GlobalPIQA_mean", "Reading", "BLiMP"] if comp.get(k, 0.0) <= -0.5]
        good_supp = comp.get("Supplement", 0.0) > 0.5
        lines[target] = (
            f"{target} vs {ref}: Supplement delta {comp.get('Supplement')}, equal7_full_entity delta {comp.get('equal7_full_entity')}, "
            f"notable broad losses {bad if bad else 'none >=0.5'}; Supplement-only gain {'exists' if good_supp else 'does not exist'} in this fast surface."
        )
    # Decision wording kept direct and scientific, not procedural.
    if d430.get("equal7_full_entity", -999) >= -0.2 and d431.get("equal7_full_entity", -999) >= 0.3 and d431.get("EWoK", -999) >= -0.5:
        lines["route_read"] = "The 80M stopping candidate preserves the broad fast surface well enough to test late checkpoint averaging or official-coordinate collation next."
    else:
        lines["route_read"] = "The 80M stopping candidate does not preserve the broad fast surface strongly enough by itself; checkpoint averaging should only proceed if it directly addresses the measured losses, otherwise return to mechanism construction rather than a broad stopping/full-eval route."
    return lines


def fmt(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, float):
        return f"{v:.3f}"
    return str(v)


def write_note(payload: Dict[str, Any], out_md: Path) -> None:
    lines = [
        "# research — merged broad no-AoA late-checkpoint screen\n\n",
        "This merge combines the two split GPU evaluation roots for selected existing compact_view_reinvest checkpoints. It is not a full official score because SuperGLUE, AoA, and full current official collation are absent.\n\n",
        f"Summary JSON: `{out_md.parent / 'broad_noaoa_late_checkpoint_merge.json'}`\n\n",
    ]
    if payload.get("missing"):
        lines.append("## Missing or incomplete inputs\n")
        for m in payload["missing"]:
            lines.append(f"- {m}\n")
    lines.append("\n## Candidate checkpoint scores\n")
    lines.append("| target | BLiMP | Supp | EWoK-fast | Entity full | COMPS | GPIQA | Reading | equal7 fullEnt |\n")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|\n")
    for t, scores in payload.get("table", {}).items():
        lines.append("| " + t + " | " + " | ".join(fmt(scores.get(k)) for k in [
            "BLiMP", "Supplement", "EWoK", "Entity_full", "COMPS", "GlobalPIQA_mean", "Reading", "equal7_full_entity",
        ]) + " |\n")
    lines.append("\n## Known 100M fast references\n")
    lines.append("| reference | BLiMP | Supp | EWoK-fast | Entity full | COMPS | GPIQA | Reading | equal7 fullEnt |\n")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|\n")
    for t, scores in payload.get("known_100M_fast", {}).items():
        lines.append("| " + t + " | " + " | ".join(fmt(scores.get(k)) for k in [
            "BLiMP", "Supplement", "EWoK", "Entity_full", "COMPS", "GlobalPIQA_mean", "Reading", "equal7_full_entity",
        ]) + " |\n")
    lines.append("\n## Contrasts vs own 100M fast reference\n")
    keep = ["equal7_full_entity", "BLiMP", "Supplement", "EWoK", "Entity_full", "COMPS", "GlobalPIQA_mean", "Reading"]
    for name, delta in payload.get("comparisons", {}).items():
        vals = ", ".join(f"{k} {delta[k]:+.3f}" for k in keep if k in delta)
        lines.append(f"- **{name}**: {vals}\n")
    lines.append("\n## Scientific read\n")
    for msg in payload.get("scientific_read", {}).values():
        lines.append(f"- {msg}\n")
    out_md.write_text("".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    raw, missing = load_all(args.roots)
    table = {t: scores_from_payload(raw[t]) for t in sorted(raw) if t in TARGET_META}
    known = {k: add_composites(v) for k, v in KNOWN_100M.items()}
    comparisons: Dict[str, Dict[str, float]] = {}
    for t, s in table.items():
        seed = TARGET_META[t]["seed"]
        ref = f"r{seed}_100M_fast"
        comparisons[f"{t}_minus_{ref}"] = diff(s, known[ref])
    if "r43022_80M" in table and "r43122_80M" in table:
        comparisons["r43122_80M_minus_r43022_80M"] = diff(table["r43122_80M"], table["r43022_80M"])
    if "r43022_90M" in table and "r43022_80M" in table:
        comparisons["r43022_90M_minus_r43022_80M"] = diff(table["r43022_90M"], table["r43022_80M"])
    if "r43122_45M" in table and "r43122_80M" in table:
        comparisons["r43122_45M_minus_r43122_80M"] = diff(table["r43122_45M"], table["r43122_80M"])
    scientific = build_scientific_read(table, comparisons, missing)
    payload = {
        "status": "BROAD_NOAOA_LATE_CHECKPOINT_MERGE_PARTIAL" if missing and not args.allow_partial else "BROAD_NOAOA_LATE_CHECKPOINT_MERGE_COMPLETE",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "input_roots": [str(p) for p in args.roots],
        "missing": missing,
        "table": table,
        "known_100M_fast": known,
        "comparisons": comparisons,
        "scientific_read": scientific,
    }
    out_json = args.out_dir / "broad_noaoa_late_checkpoint_merge.json"
    payload["out_json"] = str(out_json)
    payload["out_md"] = str(args.out_dir / "broad_noaoa_late_checkpoint_merge.md")
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_note(payload, Path(payload["out_md"]))
    print(json.dumps({
        "status": payload["status"],
        "missing": missing,
        "out_json": payload["out_json"],
        "out_md": payload["out_md"],
        "scientific_read": scientific,
    }, indent=2, ensure_ascii=False))
    if missing and not args.allow_partial:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
