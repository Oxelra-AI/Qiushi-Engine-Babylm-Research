#!/usr/bin/env python3
"""Merge research broad no-AoA screens for checkpoint-averaged models.

The averaged models are arithmetic averages of existing compact_view_reinvest
checkpoints.  This analyzer compares each average with (i) its own 100M fast
reference and (ii) the raw checkpoint that motivated the average, to decide
whether averaging actually repairs the measured tradeoff before any official full
evaluation.
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
    STUDY / "data/broad_noaoa_avg43022",
    STUDY / "data/broad_noaoa_avg43122",
]
BROAD_SELECTED = STUDY / "data/broad_noaoa_late_checkpoint_merge/broad_noaoa_late_checkpoint_merge.json"
OUT_DIR_DEFAULT = STUDY / "data/checkpoint_average_noaoa_merge"

TABLE_KEYS = [
    "BLiMP", "Supplement", "EWoK", "Entity", "Entity_full", "COMPS",
    "GlobalPIQA_parallel", "GlobalPIQA_nonparallel", "GlobalPIQA_mean",
    "Reading", "Reading_eye", "Reading_self_paced", "equal7_mean", "equal7_full_entity",
]
KNOWN_100M = {
    "avg43022_90_100": "r43022_100M_fast",
    "avg43122_80_100": "r43122_100M_fast",
}
MOTIVATING_RAW = {
    "avg43022_90_100": "r43022_90M",
    "avg43122_80_100": "r43122_80M",
}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", nargs="*", type=Path, default=DEFAULT_ROOTS)
    ap.add_argument("--selected-merge", type=Path, default=BROAD_SELECTED)
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


def diff(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, float]:
    out = {}
    for k in TABLE_KEYS:
        if a.get(k) is not None and b.get(k) is not None:
            out[k] = round(float(a[k]) - float(b[k]), 4)
    return out


def load_avg_payloads(roots: list[Path]) -> tuple[Dict[str, Dict[str, Any]], list[str]]:
    raw = {}
    missing = []
    for root in roots:
        per = root / "per_target"
        if not per.exists():
            missing.append(f"missing per_target directory: {per}")
            continue
        for p in sorted(per.glob("*.json")):
            try:
                raw[p.stem] = json.loads(p.read_text(encoding="utf-8"))
            except Exception as e:
                missing.append(f"failed to read {p}: {e!r}")
    for t in sorted(KNOWN_100M):
        if t not in raw:
            missing.append(f"missing averaged target payload: {t}")
    return raw, missing


def fmt(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, float):
        return f"{v:.3f}"
    return str(v)


def scientific_read(table: Dict[str, Dict[str, Any]], selected: Dict[str, Any], comparisons: Dict[str, Dict[str, float]], missing: list[str]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    if missing:
        out["partial"] = "Averaged-model screen is incomplete; do not choose an official follow-up from this merge."
        return out
    known = selected.get("known_100M_fast", {})
    raw_table = selected.get("table", {})
    for avg, ref in KNOWN_100M.items():
        comp100 = comparisons.get(f"{avg}_minus_{ref}", {})
        raw = MOTIVATING_RAW[avg]
        compraw = comparisons.get(f"{avg}_minus_{raw}", {})
        out[avg] = (
            f"{avg}: vs own 100M fast reference equal7_full_entity {comp100.get('equal7_full_entity')}, "
            f"Supplement {comp100.get('Supplement')}, EWoK {comp100.get('EWoK')}, Entity_full {comp100.get('Entity_full')}, "
            f"GlobalPIQA_mean {comp100.get('GlobalPIQA_mean')}; vs motivating raw {raw} equal7_full_entity {compraw.get('equal7_full_entity')}, "
            f"Supplement {compraw.get('Supplement')}, EWoK {compraw.get('EWoK')}, GlobalPIQA_mean {compraw.get('GlobalPIQA_mean')}."
        )
    # Simple route interpretation by measured quantities.
    a430 = comparisons.get("avg43022_90_100_minus_r43022_100M_fast", {})
    a431 = comparisons.get("avg43122_80_100_minus_r43122_100M_fast", {})
    if a430.get("equal7_full_entity", -999) > 0 and a431.get("equal7_full_entity", -999) > 0 and a430.get("EWoK", -999) >= -0.3 and a431.get("EWoK", -999) >= -0.3:
        out["route"] = "Checkpoint averaging improved the measured broad fast surface in both seeds without an EWoK loss large enough to reject the route; next evidence should be a narrow official-coordinate check on the better average or raw checkpoint, starting with full Supplement/EWoK and then SuperGLUE/AoA only if the no-AoA surface remains competitive."
    else:
        out["route"] = "Checkpoint averaging did not produce a robust broad fast improvement across seeds; avoid escalating this route to full official evaluation unless a specific component result still gives a stronger, cheaper discriminating test."
    return out


def write_note(payload: Dict[str, Any], out_md: Path) -> None:
    lines = [
        "# research — checkpoint-average broad no-AoA merge\n\n",
        "Arithmetic averaged weights from existing compact_view_reinvest checkpoints; no training or corpus modification. Not a full official score because SuperGLUE, AoA, and current official collation are absent.\n\n",
        f"Summary JSON: `{out_md.parent / 'checkpoint_average_noaoa_merge.json'}`\n\n",
    ]
    if payload.get("missing"):
        lines.append("## Missing inputs\n")
        for m in payload["missing"]:
            lines.append(f"- {m}\n")
    lines.append("\n## Averaged-model scores\n")
    lines.append("| target | BLiMP | Supp | EWoK-fast | Entity full | COMPS | GPIQA | Reading | equal7 fullEnt |\n")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|\n")
    for t, scores in payload.get("table", {}).items():
        lines.append("| " + t + " | " + " | ".join(fmt(scores.get(k)) for k in [
            "BLiMP", "Supplement", "EWoK", "Entity_full", "COMPS", "GlobalPIQA_mean", "Reading", "equal7_full_entity",
        ]) + " |\n")
    lines.append("\n## Contrasts\n")
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
    raw, missing = load_avg_payloads(args.roots)
    table = {t: scores_from_payload(raw[t]) for t in sorted(raw) if t in KNOWN_100M}
    if not args.selected_merge.exists():
        missing.append(f"missing selected checkpoint merge: {args.selected_merge}")
        selected = {"known_100M_fast": {}, "table": {}}
    else:
        selected = json.loads(args.selected_merge.read_text(encoding="utf-8"))
    comparisons: Dict[str, Dict[str, float]] = {}
    for avg, ref in KNOWN_100M.items():
        if avg in table and ref in selected.get("known_100M_fast", {}):
            comparisons[f"{avg}_minus_{ref}"] = diff(table[avg], selected["known_100M_fast"][ref])
        raw_key = MOTIVATING_RAW[avg]
        if avg in table and raw_key in selected.get("table", {}):
            comparisons[f"{avg}_minus_{raw_key}"] = diff(table[avg], selected["table"][raw_key])
    if "avg43022_90_100" in table and "avg43122_80_100" in table:
        comparisons["avg43122_80_100_minus_avg43022_90_100"] = diff(table["avg43122_80_100"], table["avg43022_90_100"])
    # Mark incomplete scores.
    for t, s in table.items():
        if s.get("equal7_full_entity") is None:
            missing.append(f"incomplete scores for {t}")
    sci = scientific_read(table, selected, comparisons, missing)
    payload = {
        "status": "CHECKPOINT_AVERAGE_NOAOA_MERGE_PARTIAL" if missing and not args.allow_partial else "CHECKPOINT_AVERAGE_NOAOA_MERGE_COMPLETE",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "input_roots": [str(p) for p in args.roots],
        "selected_merge": str(args.selected_merge),
        "missing": missing,
        "table": table,
        "comparisons": comparisons,
        "scientific_read": sci,
    }
    out_json = args.out_dir / "checkpoint_average_noaoa_merge.json"
    out_md = args.out_dir / "checkpoint_average_noaoa_merge.md"
    payload["out_json"] = str(out_json)
    payload["out_md"] = str(out_md)
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_note(payload, out_md)
    print(json.dumps({
        "status": payload["status"],
        "missing": missing,
        "out_json": str(out_json),
        "out_md": str(out_md),
        "scientific_read": sci,
    }, indent=2, ensure_ascii=False))
    if missing and not args.allow_partial:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
