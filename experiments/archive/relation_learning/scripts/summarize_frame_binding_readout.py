#!/usr/bin/env python3
"""research: frame-wise summary for local binding readouts.

Input can be either a research-style pair_outcomes.csv or a scientific_metrics.json
whose binding_eval_final contains pair_margins.  The script groups by deterministic
query frame, frame split (train-seen vs eval-unseen when the research manifest is
available), and context/alpha when present.  This is the local readout needed to test
whether frame-varied practice transfers to unseen query phrasings rather than only
raising aggregate in-format assignment.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import pathlib
import re
import time
from collections import defaultdict
from typing import Any

ROOT = pathlib.Path(".").resolve()
MANIFEST = ROOT / "experiments/archive/relation_learning/data/frame_varied_recombination_rows/manifest.json"


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def rel(p: pathlib.Path | str) -> str:
    pp = pathlib.Path(p)
    try:
        return str(pp.resolve().relative_to(ROOT))
    except Exception:
        return str(pp)


def norm_path(p: str | pathlib.Path) -> pathlib.Path:
    q = pathlib.Path(p)
    return q if q.is_absolute() else ROOT / q


def frame_map() -> dict[str, str]:
    if not MANIFEST.exists():
        return {}
    m = json.loads(MANIFEST.read_text(encoding="utf-8"))
    return {f["frame_id"]: f["frame_split"] for f in m.get("frames", [])}


def extract_frame(pair_id: str) -> str:
    if "::" in str(pair_id):
        return str(pair_id).rsplit("::", 1)[1]
    return "single_frame"


def read_csv_rows(path: pathlib.Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def rows_from_metrics(path: pathlib.Path) -> list[dict[str, Any]]:
    js = json.loads(path.read_text(encoding="utf-8"))
    ev = js.get("binding_eval_final") or {}
    rows = []
    for r in ev.get("pair_margins", []):
        rows.append({
            "pair_id": r.get("pair_id", ""),
            "context_mode": "full_context",
            "alpha": js.get("private_adapter_scale", "saved"),
            "a_correct": r.get("a_correct", 0),
            "b_correct": r.get("b_correct", 0),
            "joint": r.get("joint_correct", 0),
            "both_wrong": int((not int(r.get("a_correct", 0))) and (not int(r.get("b_correct", 0)))),
            "joint_min_margin": r.get("joint_min_margin", float("nan")),
        })
    return rows


def load_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    out = []
    for p in args.input:
        path = norm_path(p)
        if path.suffix == ".csv":
            out.extend(read_csv_rows(path))
        else:
            out.extend(rows_from_metrics(path))
    return out


def fnum(x: Any) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def inum(x: Any) -> int:
    try:
        return int(float(x))
    except Exception:
        return 0


def write_csv(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    keys: list[str] = []
    seen = set()
    for r in rows:
        for k in r:
            if k not in seen:
                keys.append(k); seen.add(k)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader(); w.writerows(rows)


def summarize(rows: list[dict[str, Any]], fmap: dict[str, str]) -> list[dict[str, Any]]:
    buckets: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        pid = str(r.get("pair_id", ""))
        frame = str(r.get("frame_id") or extract_frame(pid))
        split = fmap.get(frame, "single_frame" if frame == "single_frame" else "unknown")
        alpha = str(r.get("alpha", "saved"))
        mode = str(r.get("context_mode", "full_context"))
        r["frame_id"] = frame
        r["frame_split"] = split
        buckets[(alpha, mode, "ALL")].append(r)
        buckets[(alpha, mode, f"frame_split:{split}")].append(r)
        buckets[(alpha, mode, f"frame:{frame}")].append(r)
    out = []
    for (alpha, mode, group), vals in sorted(buckets.items(), key=lambda x: (x[0][0], x[0][1], x[0][2])):
        n = len(vals)
        j = sum(inum(v.get("joint", 0)) for v in vals)
        a = sum(inum(v.get("a_correct", 0)) for v in vals)
        b = sum(inum(v.get("b_correct", 0)) for v in vals)
        bw = sum(inum(v.get("both_wrong", 0)) for v in vals)
        margins = [fnum(v.get("joint_min_margin")) for v in vals if math.isfinite(fnum(v.get("joint_min_margin")))]
        out.append({
            "alpha": alpha,
            "context_mode": mode,
            "group": group,
            "n": n,
            "joint": j,
            "joint_frac": j / n if n else float("nan"),
            "a_correct": a,
            "a_frac": a / n if n else float("nan"),
            "b_correct": b,
            "b_frac": b / n if n else float("nan"),
            "both_wrong": bw,
            "both_wrong_frac": bw / n if n else float("nan"),
            "mean_joint_min_margin": sum(margins) / len(margins) if margins else float("nan"),
        })
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", action="append", required=True, help="pair_outcomes.csv or scientific_metrics.json")
    ap.add_argument("--out-dir", default="experiments/archive/relation_learning/data/frame_binding_readout")
    args = ap.parse_args()
    out = norm_path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    fmap = frame_map()
    rows = load_rows(args)
    summary = summarize(rows, fmap)
    write_csv(out / "frame_pair_rows.csv", rows)
    write_csv(out / "frame_summary.csv", summary)
    obj = {"status": "FRAME_BINDING_READOUT_SUMMARY", "created_utc": now(), "inputs": [rel(norm_path(p)) for p in args.input], "n_pair_rows": len(rows), "summary_rows": len(summary), "frame_summary": rel(out / "frame_summary.csv")}
    (out / "summary.json").write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = ["# research frame binding readout summary", "", "| alpha | mode | group | n | joint | frac | A | B | both_wrong |", "|---|---|---|---:|---:|---:|---:|---:|---:|"]
    for r in summary:
        group = str(r["group"])
        if group == "ALL" or group.startswith("frame_split:") or re.match(r"frame:f0[0-7]", group):
            lines.append(f"| {r['alpha']} | {r['context_mode']} | {group} | {r['n']} | {r['joint']} | {float(r['joint_frac']):.3f} | {r['a_correct']} | {r['b_correct']} | {r['both_wrong']} |")
    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(obj, indent=2), flush=True)


if __name__ == "__main__":
    main()
