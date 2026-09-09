#!/usr/bin/env python3
"""research: split existing Entity continuous-margin deltas by numops.

File-only readout from research margin CSVs.  It answers the cheap alternative
posed in research: is compact-view margin movement concentrated in rows with
state-changing operations, with zero-operation rows flat or negative?  Such a
pattern would make the official Entity carrier look like an operation/change
bias rather than cross-surface record re-identification.

No model inference, no training, no GPU, no official scoring job, no upload.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse
import csv
import json
import pathlib
import statistics
import time
from collections import defaultdict
from typing import Any

ROOT = _public_path('.')
WS = _public_path('experiments/archive/frontier_consolidation')
DEFAULT_INPUTS = [_public_path('experiments/archive/frontier_consolidation/data/ewok_entity_margin_pilot80/view_minus_repeat_margin_deltas.csv')]
OUT_DIR = _public_path('experiments/archive/frontier_consolidation/data/entity_margin_numops_readout')


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def rel(p: pathlib.Path) -> str:
    try:
        return str(p.relative_to(ROOT))
    except Exception:
        return str(p)


def fnum(x: Any) -> float | None:
    if x is None:
        return None
    s = str(x).strip()
    if s == "":
        return None
    try:
        return float(s)
    except Exception:
        return None


def read_rows(paths: list[pathlib.Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for p in paths:
        if not p.exists():
            continue
        with p.open("r", encoding="utf-8", newline="") as f:
            for r in csv.DictReader(f):
                rr = dict(r)
                rr["source_csv"] = rel(p)
                if rr.get("family") != "Entity":
                    continue
                margin = fnum(rr.get("margin_delta_view_minus_repeat"))
                if margin is None:
                    continue
                rr["margin_delta_view_minus_repeat"] = margin
                bd = fnum(rr.get("binary_delta"))
                rr["binary_delta"] = bd
                try:
                    rr["numops"] = int(str(rr.get("numops", "")).strip())
                except Exception:
                    continue
                rows.append(rr)
    return rows


def summarize_group(rows: list[dict[str, Any]], fields: dict[str, Any]) -> dict[str, Any]:
    vals = [float(r["margin_delta_view_minus_repeat"]) for r in rows]
    bvals = [float(r["binary_delta"]) for r in rows if r.get("binary_delta") is not None]
    return {
        **fields,
        "n": len(vals),
        "margin_delta_mean": sum(vals) / len(vals) if vals else None,
        "margin_delta_median": statistics.median(vals) if vals else None,
        "margin_delta_stdev": statistics.pstdev(vals) if len(vals) > 1 else 0.0 if vals else None,
        "frac_margin_positive": sum(1 for v in vals if v > 0) / len(vals) if vals else None,
        "binary_delta_mean": sum(bvals) / len(bvals) if bvals else None,
        "frac_binary_positive": sum(1 for v in bvals if v > 0) / len(bvals) if bvals else None,
    }


def aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        dose = r.get("dose_name") or r.get("dose") or "unknown"
        ck = r.get("checkpoint") or "unknown"
        split = r.get("split") or "unknown"
        groups[(dose, ck, "all", "all")].append(r)
        groups[(dose, ck, f"numops_{r['numops']}", str(r["numops"]))].append(r)
        groups[(dose, ck, "zero_ops", "zero")].append(r) if r["numops"] == 0 else groups[(dose, ck, "nonzero_ops", "nonzero")].append(r)
        groups[(dose, ck, f"split_{split}", f"split:{split}")].append(r)
        if r["numops"] == 0:
            groups[(dose, ck, f"{split}_zero_ops", f"split:{split}:zero")].append(r)
        else:
            groups[(dose, ck, f"{split}_nonzero_ops", f"split:{split}:nonzero")].append(r)
    out: list[dict[str, Any]] = []
    for (dose, ck, group, numops_group), rs in sorted(groups.items(), key=lambda x: (str(x[0][0]), str(x[0][1]), str(x[0][2]))):
        out.append(summarize_group(rs, {"dose_name": dose, "checkpoint": ck, "group": group, "numops_group": numops_group}))
    return out


def build_key_table(agg: list[dict[str, Any]]) -> list[dict[str, Any]]:
    idx = {(r["dose_name"], r["checkpoint"], r["group"]): r for r in agg}
    out: list[dict[str, Any]] = []
    for dose in sorted({r["dose_name"] for r in agg}):
        for ck in sorted({r["checkpoint"] for r in agg if r["dose_name"] == dose}, key=lambda s: int(str(s).replace("chck_", "").replace("M", "")) if str(s).startswith("chck_") else 0):
            allr = idx.get((dose, ck, "all"), {})
            z = idx.get((dose, ck, "zero_ops"), {})
            nz = idx.get((dose, ck, "nonzero_ops"), {})
            out.append({
                "dose_name": dose,
                "checkpoint": ck,
                "all_n": allr.get("n"),
                "all_margin_delta_mean": allr.get("margin_delta_mean"),
                "zero_n": z.get("n"),
                "zero_margin_delta_mean": z.get("margin_delta_mean"),
                "nonzero_n": nz.get("n"),
                "nonzero_margin_delta_mean": nz.get("margin_delta_mean"),
                "nonzero_minus_zero_margin_delta": None if z.get("margin_delta_mean") is None or nz.get("margin_delta_mean") is None else float(nz["margin_delta_mean"]) - float(z["margin_delta_mean"]),
                "all_frac_margin_positive": allr.get("frac_margin_positive"),
                "zero_frac_margin_positive": z.get("frac_margin_positive"),
                "nonzero_frac_margin_positive": nz.get("frac_margin_positive"),
                "binary_delta_mean_all": allr.get("binary_delta_mean"),
                "binary_delta_mean_zero": z.get("binary_delta_mean"),
                "binary_delta_mean_nonzero": nz.get("binary_delta_mean"),
            })
    return out


def write_csv(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for k in row.keys():
            if k not in fields:
                fields.append(k)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def write_md(payload: dict[str, Any], path: pathlib.Path) -> None:
    lines: list[str] = []
    lines.append("# research Entity margin numops readout\n\n")
    lines.append("This file-only readout groups existing sampled Entity continuous margin deltas by numops. Positive values mean MAX compact view has a larger correct-vs-distractor margin than repeat on the sampled item.\n\n")
    lines.append("| dose | checkpoint | all mean | zero-op mean | nonzero mean | nonzero minus zero | all n | zero n | nonzero n |\n")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|\n")
    for r in payload["key_table"]:
        def fmt(x: Any) -> str:
            return "NA" if x is None else f"{float(x):+.4f}"
        lines.append(f"| {r['dose_name']} | {r['checkpoint']} | {fmt(r['all_margin_delta_mean'])} | {fmt(r['zero_margin_delta_mean'])} | {fmt(r['nonzero_margin_delta_mean'])} | {fmt(r['nonzero_minus_zero_margin_delta'])} | {r['all_n']} | {r['zero_n']} | {r['nonzero_n']} |\n")
    lines.append("\n## Interpretation\n\n")
    for k, v in payload.get("interpretation", {}).items():
        lines.append(f"- **{k}**: {v}\n")
    path.write_text("".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--inputs", nargs="*", default=[str(p) for p in DEFAULT_INPUTS])
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    args = ap.parse_args()
    out_dir = pathlib.Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = [pathlib.Path(p) if pathlib.Path(p).is_absolute() else ROOT / p for p in args.inputs]
    rows = read_rows(paths)
    agg = aggregate(rows)
    key = build_key_table(agg)
    interp = {
        "operation_split": "If nonzero_ops is much larger than zero_ops, sampled margins align with the changed-state-bias alternative; positive zero_ops would weaken the pure bias explanation.",
        "sample_scope": "The current default input is the research sampled 80M margin pilot, not the full Entity dataset and not the pending common-window ladder. Re-run with the delivered research ladder CSV when that task finishes.",
        "relation_to_official_accuracy_split": "This margin split is read beside the file-only official Entity numops accuracy split in entity_numops_bias_readout.",
    }
    payload = {
        "status": "ENTITY_MARGIN_NUMOPS_READOUT_DONE",
        "created_utc": now(),
        "inputs": [rel(p) for p in paths],
        "input_rows_entity": len(rows),
        "aggregate_rows": agg,
        "key_table": key,
        "interpretation": interp,
        "no_model_inference_training_gpu_upload_or_leaderboard": True,
    }
    write_csv(out_dir / "entity_margin_numops_records.csv", rows)
    write_csv(out_dir / "entity_margin_numops_aggregate.csv", agg)
    write_csv(out_dir / "entity_margin_numops_key_table.csv", key)
    (out_dir / "entity_margin_numops_summary.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_md(payload, out_dir / "entity_margin_numops_summary.md")
    print(json.dumps({
        "status": payload["status"],
        "out_dir": rel(out_dir),
        "input_rows_entity": len(rows),
        "key_table": key,
        "no_model_inference_training_gpu_upload_or_leaderboard": True,
    }, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
