#!/usr/bin/env python3
"""Summarize selected repaired FineWeb seqsafe96 full official-style evaluations."""
from __future__ import annotations

import argparse
import json
import pathlib
from typing import Any

DEFAULT_OUT = pathlib.Path("experiments/archive/representation_and_objectives/data/fineweb_seqsafe96_full_eval")
DEFAULT_NOTE = pathlib.Path("research/notes/representation_and_objectives/fineweb_selected_full_eval_summary.md")
KEYS = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA", "SuperGLUE", "Reading", "AoA"]
LEADER = {"name": "wwm_curriculum_simplification_40k", "Overall": 41.80}
SOURCE_PAIR = ("fineweb_seqsafe96_treatment_repairseq", "fineweb_seqsafe96_control_repairseq")


def r(x: Any, nd: int = 4):
    if x is None:
        return None
    try:
        return round(float(x), nd)
    except Exception:
        return x


def load_rows(per: pathlib.Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    if not per.exists():
        return rows
    for p in sorted(per.glob("*.json")):
        d = json.loads(p.read_text(encoding="utf-8"))
        oo = d.get("official_overall") or {}
        scores = oo.get("scores") or {}
        row = {k: scores.get(k) for k in KEYS}
        row.update({
            "Overall": oo.get("Overall"),
            "NLP_average": oo.get("NLP_average"),
            "Human_like_average": oo.get("Human_like_average"),
            "submit_ready_overall": oo.get("submit_ready_overall"),
            "submit_ready_aoa": oo.get("submit_ready_aoa"),
            "aoa_status": oo.get("aoa_status"),
            "aoa_raw_correlation": oo.get("aoa_raw_correlation"),
            "aoa_leaderboard_score": oo.get("aoa_leaderboard_score"),
            "endpoint": d.get("endpoint"),
            "source_target": d.get("source_target") or (d.get("target", "").rsplit("__", 1)[0] if "__" in d.get("target", "") else None),
            "run_dir": d.get("run_dir"),
            "model_path": d.get("model_path"),
            "per_target_json": str(p),
            "prefill_source": d.get("prefill_source"),
        })
        rows[p.stem] = row
    return rows


def pair_deltas(rows: dict[str, dict[str, Any]]) -> dict[str, Any]:
    deltas: dict[str, Any] = {}
    for t_name, t in rows.items():
        if not t_name.startswith(SOURCE_PAIR[0] + "__"):
            continue
        endpoint = t.get("endpoint")
        c_name = f"{SOURCE_PAIR[1]}__{endpoint}"
        if c_name not in rows:
            continue
        c = rows[c_name]
        d: dict[str, Any] = {}
        for k in KEYS + ["Overall", "NLP_average", "Human_like_average"]:
            tv, cv = t.get(k), c.get(k)
            d[k] = None if tv is None or cv is None else round(float(tv) - float(cv), 6)
        d["knowledge_sum_EWoK_Entity_COMPS_GlobalPIQA"] = None
        if all(d.get(k) is not None for k in ["EWoK", "Entity", "COMPS", "GlobalPIQA"]):
            d["knowledge_sum_EWoK_Entity_COMPS_GlobalPIQA"] = round(sum(float(d[k]) for k in ["EWoK", "Entity", "COMPS", "GlobalPIQA"]), 6)
        deltas[f"{t_name}_minus_{c_name}"] = d
    return deltas


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", default=str(DEFAULT_OUT))
    ap.add_argument("--note", default=str(DEFAULT_NOTE))
    args = ap.parse_args()
    out = pathlib.Path(args.out_root)
    per = out / "per_target"
    note = pathlib.Path(args.note)
    rows = load_rows(per)
    deltas = pair_deltas(rows)
    complete = {k: v for k, v in rows.items() if v.get("Overall") is not None}
    best = max(complete.items(), key=lambda kv: float(kv[1]["Overall"])) if complete else None
    payload = {
        "status": "FINEWEB_SELECTED_FULL_EVAL_SUMMARY",
        "leader_reference": LEADER,
        "targets": rows,
        "same_endpoint_treatment_minus_control": deltas,
        "best_by_overall": {"target": best[0], "row": best[1]} if best else None,
        "interpretation": "Selected endpoint full completion after repaired no-AoA trajectory. Same-endpoint deltas test cached FineWeb source breadth versus official lengthmatched control; SOTA judgment still requires comparison to the public 41.8 leader and submit-ready AoA.",
    }
    out.mkdir(parents=True, exist_ok=True)
    summary = out / "fineweb_selected_full_eval_summary.json"
    summary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    lines = ["# research FineWeb selected full-evaluation summary\n\n", f"Summary JSON: `{summary}`\n\n"]
    lines.append("| target | endpoint | Overall | BLiMP | Supp | EWoK | Entity | COMPS | GPIQA | SuperGLUE | Reading | AoA leaderboard | AoA raw | submit-ready |\n")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|\n")
    for name, row in sorted(rows.items()):
        lines.append("| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |\n".format(
            name, row.get("endpoint"), r(row.get("Overall")), r(row.get("BLiMP"), 2), r(row.get("Supplement"), 2),
            r(row.get("EWoK"), 2), r(row.get("Entity"), 2), r(row.get("COMPS"), 2), r(row.get("GlobalPIQA"), 2),
            r(row.get("SuperGLUE"), 4), r(row.get("Reading"), 2), r(row.get("AoA"), 4), r(row.get("aoa_raw_correlation"), 6),
            row.get("submit_ready_overall"),
        ))
    if deltas:
        lines.append("\n## Same-endpoint treatment minus control deltas\n\n")
        lines.append("```json\n" + json.dumps(deltas, indent=2, ensure_ascii=False) + "\n```\n")
    if best:
        lines.append(f"\nBest selected full-eval target: `{best[0]}` Overall {best[1].get('Overall')}. Public reference: {LEADER['Overall']}.\n")
    note.parent.mkdir(parents=True, exist_ok=True)
    note.write_text("".join(lines), encoding="utf-8")
    print(json.dumps({"summary": str(summary), "note": str(note), "best_by_overall": payload["best_by_overall"], "deltas": deltas}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
