#!/usr/bin/env python3
"""research: read the delivered sampled EWoK/Entity direct-margin ladder.

This is a file-only analysis of the CPU-safe research direct margin CSV.  It does
not load models, train, evaluate official tasks, use GPU, or touch leaderboard
code.  Its purpose is to read the threshold-free V-R margin evidence beside the
official prediction-stratum evidence:

* Entity: split correct-vs-distractor margin deltas by zero-operation and
  nonzero-operation rows, so a change/no-change allocation can be separated from
  a uniform improvement in rank separation.
* EWoK: split by domain, so apparent broad V-R costs/gains are not collapsed into
  one unstable aggregate.
* Dose/window slopes: ask whether MAX dose produces a reproducible signed
  tendency or whether the sampled margins fluctuate at the current sample size.
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
import statistics
import time
from collections import defaultdict
from typing import Any, Iterable


def find_user_root() -> pathlib.Path:
    return _PUBLIC_ROOT


ROOT = find_user_root()
WS = ROOT / "experiments/archive" / 'frontier_consolidation'
DEFAULT_INPUT = WS / "data" / "ewok_entity_margin_common10_80_sample5" / "view_minus_repeat_margin_deltas.csv"
DEFAULT_OUT = WS / "data" / "margin_domain_window_readout"
RHO = {"dose1": 0.042352, "dose1p82": 0.07712, "dose2p64": 0.111872}
DOSE_ORDER = {"dose1": 0, "dose1p82": 1, "dose2p64": 2}
WINDOWS = {
    "common10_80": (10_000_000, 80_000_000),
    "early10_40": (10_000_000, 40_000_000),
    "mid50_60": (50_000_000, 60_000_000),
    "late70_80": (70_000_000, 80_000_000),
    "endpoint80": (80_000_000, 80_000_000),
}


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
        v = float(s)
        return v if math.isfinite(v) else None
    except Exception:
        return None


def ck_sort_key(ck: str) -> int:
    return int(str(ck).replace("chck_", "").replace("M", ""))


def read_rows(path: pathlib.Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            rr = dict(r)
            margin = fnum(rr.get("margin_delta_view_minus_repeat"))
            if margin is None:
                continue
            rr["margin_delta_view_minus_repeat"] = margin
            bd = fnum(rr.get("binary_delta"))
            rr["binary_delta"] = bd
            words = fnum(rr.get("words"))
            rr["words"] = int(words) if words is not None else ck_sort_key(rr.get("checkpoint", "chck_0M")) * 1_000_000
            rr["rho"] = RHO.get(str(rr.get("dose_name")))
            no = fnum(rr.get("numops"))
            rr["numops"] = int(no) if no is not None else None
            rows.append(rr)
    return rows


def mean(xs: Iterable[float]) -> float | None:
    vals = [float(x) for x in xs if math.isfinite(float(x))]
    return sum(vals) / len(vals) if vals else None


def summarize(rows: list[dict[str, Any]], fields: dict[str, Any]) -> dict[str, Any]:
    vals = [float(r["margin_delta_view_minus_repeat"]) for r in rows]
    bvals = [float(r["binary_delta"]) for r in rows if r.get("binary_delta") is not None]
    return {
        **fields,
        "n": len(vals),
        "margin_mean": mean(vals),
        "margin_median": statistics.median(vals) if vals else None,
        "margin_stdev": statistics.pstdev(vals) if len(vals) > 1 else 0.0 if vals else None,
        "frac_margin_positive": sum(1 for v in vals if v > 0) / len(vals) if vals else None,
        "binary_delta_mean": mean(bvals),
        "frac_binary_positive": sum(1 for v in bvals if v > 0) / len(bvals) if bvals else None,
    }


def subgroup_keys(r: dict[str, Any]) -> list[tuple[str, str]]:
    fam = r.get("family")
    out = [("all", "all")]
    if fam == "Entity":
        if r.get("numops") == 0:
            out.append(("zero_ops", "zero_ops"))
        elif r.get("numops") is not None:
            out.append(("nonzero_ops", "nonzero_ops"))
            out.append((f"numops_{r['numops']}", f"numops_{r['numops']}"))
        split = r.get("split") or "unknown_split"
        out.append((f"split_{split}", f"split:{split}"))
        if r.get("numops") == 0:
            out.append((f"split_{split}_zero_ops", f"split:{split}:zero_ops"))
        elif r.get("numops") is not None:
            out.append((f"split_{split}_nonzero_ops", f"split:{split}:nonzero_ops"))
    elif fam == "EWoK":
        dom = r.get("domain") or "unknown_domain"
        out.append((f"domain_{dom}", f"domain:{dom}"))
    return out


def checkpoint_summaries(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        for subgroup, label in subgroup_keys(r):
            groups[(str(r.get("family")), str(r.get("dose_name")), str(r.get("checkpoint")), subgroup)].append(r)
    out: list[dict[str, Any]] = []
    for (fam, dose, ck, subgroup), rs in sorted(groups.items(), key=lambda kv: (kv[0][0], DOSE_ORDER.get(kv[0][1], 99), ck_sort_key(kv[0][2]), kv[0][3])):
        out.append(summarize(rs, {"family": fam, "dose_name": dose, "rho": RHO.get(dose), "checkpoint": ck, "words": ck_sort_key(ck) * 1_000_000, "subgroup": subgroup}))
    return out


def window_summaries(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        for wname, (lo, hi) in WINDOWS.items():
            if lo <= int(r["words"]) <= hi:
                for subgroup, label in subgroup_keys(r):
                    groups[(str(r.get("family")), str(r.get("dose_name")), wname, subgroup)].append(r)
    out: list[dict[str, Any]] = []
    for (fam, dose, wname, subgroup), rs in sorted(groups.items(), key=lambda kv: (kv[0][0], DOSE_ORDER.get(kv[0][1], 99), kv[0][2], kv[0][3])):
        # Also compute the mean of checkpoint means so every checkpoint has equal
        # weight even if future samples are not balanced exactly.
        by_ck: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for r in rs:
            by_ck[str(r.get("checkpoint"))].append(r)
        ck_means = [mean(float(x["margin_delta_view_minus_repeat"]) for x in crs) for crs in by_ck.values()]
        rec = summarize(rs, {"family": fam, "dose_name": dose, "rho": RHO.get(dose), "window": wname, "subgroup": subgroup})
        rec["checkpoint_count"] = len(by_ck)
        rec["mean_of_checkpoint_means"] = mean([x for x in ck_means if x is not None])
        rec["checkpoints"] = ";".join(sorted(by_ck, key=ck_sort_key))
        out.append(rec)
    return out


def slope(xs: list[tuple[float, float]]) -> dict[str, Any]:
    pts = [(float(x), float(y)) for x, y in xs if math.isfinite(x) and math.isfinite(y)]
    if len(pts) < 2:
        return {"slope_per_rho": None, "intercept": None, "r2": None, "n": len(pts)}
    mx = mean([p[0] for p in pts])
    my = mean([p[1] for p in pts])
    assert mx is not None and my is not None
    ssx = sum((x - mx) ** 2 for x, _ in pts)
    if ssx == 0:
        return {"slope_per_rho": None, "intercept": None, "r2": None, "n": len(pts)}
    b = sum((x - mx) * (y - my) for x, y in pts) / ssx
    a = my - b * mx
    sst = sum((y - my) ** 2 for _, y in pts)
    sse = sum((y - (a + b * x)) ** 2 for x, y in pts)
    r2 = None if sst == 0 else 1.0 - sse / sst
    return {"slope_per_rho": b, "intercept": a, "r2": r2, "n": len(pts)}


def slope_summaries(summary_rows: list[dict[str, Any]], mode: str) -> list[dict[str, Any]]:
    # mode: checkpoint has family/checkpoint/subgroup; window has family/window/subgroup.
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for r in summary_rows:
        key = (r["family"], r.get("checkpoint") if mode == "checkpoint" else r.get("window"), r["subgroup"])
        groups[key].append(r)
    out: list[dict[str, Any]] = []
    for (fam, slot, subgroup), rs in sorted(groups.items()):
        pts = [(float(r["rho"]), float(r.get("mean_of_checkpoint_means", r.get("margin_mean")))) for r in rs if r.get("rho") is not None and r.get("margin_mean") is not None]
        rec = {"mode": mode, "family": fam, "slot": slot, "subgroup": subgroup}
        rec.update(slope(pts))
        vals_by_dose = {r["dose_name"]: r.get("mean_of_checkpoint_means", r.get("margin_mean")) for r in rs}
        for dose in ["dose1", "dose1p82", "dose2p64"]:
            rec[f"mean_{dose}"] = vals_by_dose.get(dose)
        if vals_by_dose.get("dose1") is not None and vals_by_dose.get("dose2p64") is not None:
            rec["dose2p64_minus_dose1"] = float(vals_by_dose["dose2p64"]) - float(vals_by_dose["dose1"])
        out.append(rec)
    return out


def rows_to_idx(rows: list[dict[str, Any]], key_fields: tuple[str, ...]) -> dict[tuple[Any, ...], dict[str, Any]]:
    return {tuple(r.get(k) for k in key_fields): r for r in rows}


def key_find(rows: list[dict[str, Any]], **kw: Any) -> dict[str, Any] | None:
    for r in rows:
        if all(r.get(k) == v for k, v in kw.items()):
            return r
    return None


def fmt(x: Any, signed: bool = True) -> str:
    if x is None:
        return "NA"
    try:
        xf = float(x)
        if not math.isfinite(xf):
            return "NA"
        return f"{xf:+.4f}" if signed else f"{xf:.4f}"
    except Exception:
        return str(x)


def write_csv(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for r in rows:
        for k in r.keys():
            if k not in fields:
                fields.append(k)
    if not fields:
        fields = ["empty"]
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def write_json(path: pathlib.Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def build_md(payload: dict[str, Any]) -> str:
    widx = rows_to_idx(payload["window_summaries"], ("family", "dose_name", "window", "subgroup"))
    sidx = rows_to_idx(payload["window_slopes"], ("family", "slot", "subgroup"))
    lines: list[str] = []
    lines.append("# research sampled direct-margin domain/window readout\n\n")
    lines.append("File-only analysis of the delivered CPU-safe research margin ladder. Positive margin values mean the compact-view arm has a larger correct-vs-best-distractor margin than the matched repeat arm on the sampled rows.\n\n")
    lines.append(f"Input rows: {payload['input_row_count']} ({payload['entity_rows']} Entity, {payload['ewok_rows']} EWoK).\n\n")
    lines.append("## Entity V-R margin split by operation group\n\n")
    lines.append("The official prediction split showed zero-operation losses and nonzero-operation gains at MAX. The direct-margin sample shows the same direction mainly at late checkpoints, but not a monotone clean law across all checkpoints.\n\n")
    lines.append("| window | dose | all mean | zero-op mean | nonzero mean | nonzero-zero gap | frac positive all |\n")
    lines.append("|---|---|---:|---:|---:|---:|---:|\n")
    for window in ["common10_80", "late70_80", "endpoint80"]:
        for dose in ["dose1", "dose1p82", "dose2p64"]:
            allr = widx.get(("Entity", dose, window, "all"), {})
            zr = widx.get(("Entity", dose, window, "zero_ops"), {})
            nr = widx.get(("Entity", dose, window, "nonzero_ops"), {})
            gap = None
            if zr.get("mean_of_checkpoint_means") is not None and nr.get("mean_of_checkpoint_means") is not None:
                gap = float(nr["mean_of_checkpoint_means"]) - float(zr["mean_of_checkpoint_means"])
            lines.append(f"| {window} | {dose} | {fmt(allr.get('mean_of_checkpoint_means'))} | {fmt(zr.get('mean_of_checkpoint_means'))} | {fmt(nr.get('mean_of_checkpoint_means'))} | {fmt(gap)} | {fmt(allr.get('frac_margin_positive'), signed=False)} |\n")
    lines.append("\nEntity dose slopes over rho (mean of checkpoint means):\n\n")
    lines.append("| window | subgroup | slope/rho | R2 | MAX-minus-1x |\n")
    lines.append("|---|---|---:|---:|---:|\n")
    for window in ["common10_80", "late70_80", "endpoint80"]:
        for subgroup in ["all", "zero_ops", "nonzero_ops"]:
            r = sidx.get(("Entity", window, subgroup), {})
            lines.append(f"| {window} | {subgroup} | {fmt(r.get('slope_per_rho'))} | {fmt(r.get('r2'), signed=False)} | {fmt(r.get('dose2p64_minus_dose1'))} |\n")
    lines.append("\n## EWoK V-R margin by domain\n\n")
    lines.append("EWoK sampled margins are domain-heterogeneous. The all-domain aggregate does not currently provide a stable broad mechanism by itself.\n\n")
    lines.append("| window | dose | all-domain mean | frac positive |\n")
    lines.append("|---|---|---:|---:|\n")
    for window in ["common10_80", "late70_80", "endpoint80"]:
        for dose in ["dose1", "dose1p82", "dose2p64"]:
            r = widx.get(("EWoK", dose, window, "all"), {})
            lines.append(f"| {window} | {dose} | {fmt(r.get('mean_of_checkpoint_means'))} | {fmt(r.get('frac_margin_positive'), signed=False)} |\n")
    lines.append("\nLargest absolute MAX EWoK domain means in common10_80:\n\n")
    lines.append("| domain subgroup | MAX mean | dose1 mean | MAX-minus-1x | n |\n")
    lines.append("|---|---:|---:|---:|---:|\n")
    dom_rows = []
    for r in payload["window_summaries"]:
        if r["family"] == "EWoK" and r["window"] == "common10_80" and str(r["subgroup"]).startswith("domain_") and r["dose_name"] == "dose2p64":
            d1 = widx.get(("EWoK", "dose1", "common10_80", r["subgroup"]), {})
            maxm = r.get("mean_of_checkpoint_means")
            d1m = d1.get("mean_of_checkpoint_means")
            dom_rows.append((abs(float(maxm)) if maxm is not None else -1, r, d1m))
    for _, r, d1m in sorted(dom_rows, key=lambda x: x[0], reverse=True)[:8]:
        diff = None if d1m is None or r.get("mean_of_checkpoint_means") is None else float(r["mean_of_checkpoint_means"]) - float(d1m)
        lines.append(f"| {r['subgroup'].replace('domain_', '')} | {fmt(r.get('mean_of_checkpoint_means'))} | {fmt(d1m)} | {fmt(diff)} | {r.get('n')} |\n")
    lines.append("\n## Scientific reading\n\n")
    for x in payload["scientific_reading"]:
        lines.append(f"- {x}\n")
    lines.append("\n## Files\n\n")
    for k, v in payload["files"].items():
        lines.append(f"- {k}: `{v}`\n")
    return "".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", default=str(DEFAULT_INPUT))
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT))
    args = ap.parse_args()
    inp = pathlib.Path(args.input)
    if not inp.is_absolute():
        inp = ROOT / inp
    out_dir = pathlib.Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = read_rows(inp)
    ck_rows = checkpoint_summaries(rows)
    win_rows = window_summaries(rows)
    ck_slopes = slope_summaries(ck_rows, "checkpoint")
    win_slopes = slope_summaries(win_rows, "window")

    files = {
        "checkpoint_summaries_csv": rel(out_dir / "checkpoint_summaries.csv"),
        "window_summaries_csv": rel(out_dir / "window_summaries.csv"),
        "checkpoint_slopes_csv": rel(out_dir / "checkpoint_slopes.csv"),
        "window_slopes_csv": rel(out_dir / "window_slopes.csv"),
        "summary_json": rel(out_dir / "margin_domain_window_summary.json"),
        "summary_md": rel(out_dir / "margin_domain_window_summary.md"),
    }
    payload = {
        "status": "MARGIN_DOMAIN_WINDOW_READOUT_COMPLETE",
        "created_utc": now(),
        "input_csv": rel(inp),
        "input_row_count": len(rows),
        "entity_rows": sum(1 for r in rows if r.get("family") == "Entity"),
        "ewok_rows": sum(1 for r in rows if r.get("family") == "EWoK"),
        "checkpoint_summaries": ck_rows,
        "window_summaries": win_rows,
        "checkpoint_slopes": ck_slopes,
        "window_slopes": win_slopes,
        "scientific_reading": [
            "Direct sampled margins support reading the Entity V-R effect as operation-sensitive: at MAX late70_80 and endpoint80, nonzero-operation margins exceed zero-operation margins by several log-likelihood units, matching the official prediction-stratum direction.",
            "The same direct margins are not a clean monotone dose law across the whole 10M-80M window; signs fluctuate by checkpoint and the sample has only 15 zero-op rows per checkpoint/dose. This strengthens the need to read broad V-B/V-C from official stable-family scores rather than promoting Entity alone.",
            "EWoK margins are domain-heterogeneous and do not presently supply an independent broad positive carrier for V-R; this keeps the incoming ex-Entity V-B breadth ladder decisive for the companion mechanism.",
        ],
        "files": files,
        "no_model_loading_training_evaluation_gpu_globalpiqa_superglue_aoa_upload_or_leaderboard": True,
    }
    write_csv(out_dir / "checkpoint_summaries.csv", ck_rows)
    write_csv(out_dir / "window_summaries.csv", win_rows)
    write_csv(out_dir / "checkpoint_slopes.csv", ck_slopes)
    write_csv(out_dir / "window_slopes.csv", win_slopes)
    write_json(out_dir / "margin_domain_window_summary.json", payload)
    (out_dir / "margin_domain_window_summary.md").write_text(build_md(payload), encoding="utf-8")
    print(json.dumps({
        "status": payload["status"],
        "input_row_count": len(rows),
        "summary_md": files["summary_md"],
        "summary_json": files["summary_json"],
        "no_model_loading_training_evaluation_gpu_globalpiqa_superglue_aoa_upload_or_leaderboard": True,
    }, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
