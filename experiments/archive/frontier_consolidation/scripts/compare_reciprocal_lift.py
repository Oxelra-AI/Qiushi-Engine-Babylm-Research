#!/usr/bin/env python3
"""Compare reciprocal view-lift probes across checkpoints.

Reads research reciprocal_view_lift_probe.json outputs and reports the groups most
relevant to the objective-compatible reciprocal multi-view hypothesis:
compact vs repeat, source vs compact target side, copy vs non-copy, and late
82M->100M changes.
"""
from __future__ import annotations
import argparse, json, pathlib, math
from typing import Any

KEY_FIELDS = ["model_type", "pair_type", "target_segment", "copy_by_text"]


def load(path: pathlib.Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def flatten_groups(obj: dict[str, Any], group_name: str = "model_type/pair_type/target_segment/copy_by_text") -> dict[tuple[Any, ...], dict[str, Any]]:
    rows = obj.get("summary", {}).get("groups", {}).get(group_name, [])
    out = {}
    fields = group_name.split("/")
    for row in rows:
        key = tuple(row.get("key", {}).get(f) for f in fields)
        out[key] = row
    return out


def fmt(x: Any) -> str:
    if x is None:
        return ""
    try:
        xf = float(x)
    except Exception:
        return str(x)
    if not math.isfinite(xf):
        return "nan"
    return f"{xf:.6f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--first", required=True)
    ap.add_argument("--second", required=True)
    ap.add_argument("--first-label", default="first")
    ap.add_argument("--second-label", default="second")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    first = load(pathlib.Path(args.first))
    second = load(pathlib.Path(args.second))
    g1 = flatten_groups(first)
    g2 = flatten_groups(second)
    keys = sorted(set(g1) | set(g2), key=str)
    rows = []
    for k in keys:
        r1 = g1.get(k, {})
        r2 = g2.get(k, {})
        m1 = r1.get("mean_lift")
        m2 = r2.get("mean_lift")
        delta = None
        if m1 is not None and m2 is not None:
            delta = float(m2) - float(m1)
        rows.append({
            "key": dict(zip("model_type/pair_type/target_segment/copy_by_text".split("/"), k)),
            "n_first": r1.get("n"),
            "n_second": r2.get("n"),
            "mean_lift_first": m1,
            "mean_lift_second": m2,
            "delta_second_minus_first": delta,
            "pos_frac_first": r1.get("positive_lift_frac"),
            "pos_frac_second": r2.get("positive_lift_frac"),
            "mean_paired_nll_first": r1.get("mean_paired_nll"),
            "mean_paired_nll_second": r2.get("mean_paired_nll"),
            "mean_sideonly_nll_first": r1.get("mean_sideonly_nll"),
            "mean_sideonly_nll_second": r2.get("mean_sideonly_nll"),
        })
    # Key scientific aggregates.
    def find(pair_type: str, target: str, copy: bool) -> dict[str, Any] | None:
        key = ("mlm", pair_type, target, copy)
        return next((r for r in rows if tuple(r["key"].get(f) for f in ["model_type","pair_type","target_segment","copy_by_text"]) == key), None)
    highlights = {
        "compact_noncopy_other": find("compact", "other", False),
        "compact_noncopy_source": find("compact", "source", False),
        "compact_copy_other": find("compact", "other", True),
        "compact_copy_source": find("compact", "source", True),
        "repeat_noncopy_source": find("repeat", "source", False),
        "repeat_copy_other": find("repeat", "other", True),
    }
    result = {
        "status": "RECIPROCAL_VIEW_LIFT_COMPARISON",
        "first": args.first,
        "second": args.second,
        "first_label": args.first_label,
        "second_label": args.second_label,
        "n_records_first": first.get("target_records"),
        "n_records_second": second.get("target_records"),
        "rows": rows,
        "highlights": highlights,
    }
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / "reciprocal_view_lift_comparison.json"
    out_json.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = ["# research reciprocal view-lift comparison", ""]
    lines.append(f"First: `{args.first_label}` `{args.first}`")
    lines.append(f"Second: `{args.second_label}` `{args.second}`")
    lines.append(f"Records: {first.get('target_records')} -> {second.get('target_records')}")
    lines.append("")
    lines.append("## Copy/text strata")
    lines.append("| pair_type | target | copy_by_text | n first | n second | mean lift first | mean lift second | Δ second-first | pos first | pos second |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        k = r["key"]
        lines.append(f"| {k.get('pair_type')} | {k.get('target_segment')} | {k.get('copy_by_text')} | {r.get('n_first')} | {r.get('n_second')} | {fmt(r.get('mean_lift_first'))} | {fmt(r.get('mean_lift_second'))} | {fmt(r.get('delta_second_minus_first'))} | {fmt(r.get('pos_frac_first'))} | {fmt(r.get('pos_frac_second'))} |")
    lines.append("")
    lines.append("## Mechanistic reading")
    lines.append("Large copy-stratum lift is expected and mostly reflects visible lexical repetition. The load-bearing signal for semantic-view invariance is compact non-copy lift in both target directions. If this non-copy lift is small or falls from 82M to 100M while official relation/state scores also fall, it is a plausible diagnostic of reciprocal semantic support but not yet a trainable objective. Any training proposal must avoid merely increasing exact-copy targets.")
    lines.append("")
    lines.append(f"JSON: `{out_json}`")
    (out_dir / "reciprocal_view_lift_comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "out_json": str(out_json), "out_md": str(out_dir / "reciprocal_view_lift_comparison.md")}, indent=2), flush=True)

if __name__ == "__main__":
    main()
