#!/usr/bin/env python3
"""research: three-arm GlobalPIQA row/margin synthesis for the FW family.

Reads saved research/115 GlobalPIQA all-option margin rows for compact, row-block
breadth, and interleaved breadth. It does not run models. It asks whether a single
arm or a stable row pattern combines compact broad strength with breadth-like
hard-row movement.
"""
from __future__ import annotations

import csv
import json
import statistics
import time
from collections import Counter
from pathlib import Path
from typing import Any

WS = Path("experiments/archive/representation_and_objectives")
MARGIN = WS / "data/fw_globalpiqa_margin_reader"
OUT = WS / "data/fw_globalpiqa_threearm_synthesis"
NOTE = (WS.parents[2] / 'research/notes/representation_and_objectives/fw_globalpiqa_threearm_synthesis.md')

KEYS = {
    "compact": "fw_compact_fullbatch_seed43022",
    "rowblock": "fw_breadth_rowblock_fullbatch_seed43022",
    "interleaved": "fw_breadth_interleaved_fullbatch_seed43022",
}


def load_rows(key: str, mode: str) -> dict[str, dict[str, Any]]:
    p = MARGIN / f"{key}_{mode}_rows.csv"
    if not p.exists():
        raise FileNotFoundError(p)
    rows: dict[str, dict[str, Any]] = {}
    with p.open(newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            eid = row["example_id"]
            row["correct_bool"] = str(row.get("correct", "")).lower() in {"true", "1"}
            row["rank_int"] = int(float(row["correct_rank"]))
            row["top_minus_correct_float"] = float(row["top_minus_correct"])
            rows[eid] = row
    return rows


def summarize_set(rows_by_arm: dict[str, dict[str, dict[str, Any]]], mode: str) -> dict[str, Any]:
    ids = sorted(set.intersection(*(set(x) for x in rows_by_arm.values())))
    out: dict[str, Any] = {"mode": mode, "n_common": len(ids), "per_arm": {}}
    for arm, rows in rows_by_arm.items():
        corr = [rows[i]["correct_bool"] for i in ids]
        ranks = [rows[i]["rank_int"] for i in ids]
        margins = [rows[i]["top_minus_correct_float"] for i in ids]
        out["per_arm"][arm] = {
            "accuracy": 100 * sum(corr) / len(ids) if ids else None,
            "correct": sum(corr),
            "rank_counts": dict(Counter(str(x) for x in ranks)),
            "mean_top_minus_correct": statistics.fmean(margins) if margins else None,
            "median_top_minus_correct": statistics.median(margins) if margins else None,
        }
    # All exact correctness patterns.
    patt = Counter()
    for i in ids:
        patt["".join("1" if rows_by_arm[a][i]["correct_bool"] else "0" for a in ["compact", "rowblock", "interleaved"])] += 1
    out["correctness_patterns_order_compact_rowblock_interleaved"] = dict(patt)
    out["oracle_union_accuracy"] = 100 * sum(any(rows_by_arm[a][i]["correct_bool"] for a in rows_by_arm) for i in ids) / len(ids) if ids else None
    out["all_wrong"] = sum(not any(rows_by_arm[a][i]["correct_bool"] for a in rows_by_arm) for i in ids)
    out["all_correct"] = sum(all(rows_by_arm[a][i]["correct_bool"] for a in rows_by_arm) for i in ids)
    # Pairwise rank/margin movements relative to compact.
    for arm in ["rowblock", "interleaved"]:
        out[f"{arm}_vs_compact"] = {
            "better_rank": sum(rows_by_arm[arm][i]["rank_int"] < rows_by_arm["compact"][i]["rank_int"] for i in ids),
            "same_rank": sum(rows_by_arm[arm][i]["rank_int"] == rows_by_arm["compact"][i]["rank_int"] for i in ids),
            "worse_rank": sum(rows_by_arm[arm][i]["rank_int"] > rows_by_arm["compact"][i]["rank_int"] for i in ids),
            "lower_margin": sum(rows_by_arm[arm][i]["top_minus_correct_float"] < rows_by_arm["compact"][i]["top_minus_correct_float"] for i in ids),
            "higher_margin": sum(rows_by_arm[arm][i]["top_minus_correct_float"] > rows_by_arm["compact"][i]["top_minus_correct_float"] for i in ids),
            "mean_margin_delta": statistics.fmean(rows_by_arm[arm][i]["top_minus_correct_float"] - rows_by_arm["compact"][i]["top_minus_correct_float"] for i in ids) if ids else None,
        }
    # Rows where rowblock improves over compact but interleaved does not; and vice versa.
    out["layout_disagreement"] = {
        "rowblock_better_rank_interleaved_not": sum((rows_by_arm["rowblock"][i]["rank_int"] < rows_by_arm["compact"][i]["rank_int"]) and not (rows_by_arm["interleaved"][i]["rank_int"] < rows_by_arm["compact"][i]["rank_int"]) for i in ids),
        "interleaved_better_rank_rowblock_not": sum((rows_by_arm["interleaved"][i]["rank_int"] < rows_by_arm["compact"][i]["rank_int"]) and not (rows_by_arm["rowblock"][i]["rank_int"] < rows_by_arm["compact"][i]["rank_int"]) for i in ids),
        "both_better_rank_than_compact": sum((rows_by_arm["rowblock"][i]["rank_int"] < rows_by_arm["compact"][i]["rank_int"]) and (rows_by_arm["interleaved"][i]["rank_int"] < rows_by_arm["compact"][i]["rank_int"]) for i in ids),
    }
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    result = {"status": "FW_GLOBALPIQA_THREEARM_SYNTHESIS", "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "modes": {}}
    for mode in ["parallel", "nonparallel"]:
        rows_by_arm = {arm: load_rows(key, mode) for arm, key in KEYS.items()}
        result["modes"][mode] = summarize_set(rows_by_arm, mode)
    out_json = OUT / "fw_globalpiqa_threearm_synthesis.json"
    out_json.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    lines = ["# research — FW three-arm GlobalPIQA row/margin synthesis\n\n"]
    for mode in ["parallel", "nonparallel"]:
        s = result["modes"][mode]
        lines.append(f"## {mode}\n\n")
        lines.append(f"Common rows: {s['n_common']}; oracle-union accuracy across three arms: {s['oracle_union_accuracy']:.2f}; all-wrong rows: {s['all_wrong']}; all-correct rows: {s['all_correct']}.\n\n")
        for arm, a in s["per_arm"].items():
            lines.append(f"- {arm}: acc={a['accuracy']:.2f}, correct={a['correct']}, rank_counts={a['rank_counts']}, mean top-minus-correct={a['mean_top_minus_correct']:.3f}.\n")
        lines.append(f"- correctness patterns compact/rowblock/interleaved: {s['correctness_patterns_order_compact_rowblock_interleaved']}\n")
        for arm in ["rowblock", "interleaved"]:
            v = s[f"{arm}_vs_compact"]
            lines.append(f"- {arm} vs compact: better/same/worse rank={v['better_rank']}/{v['same_rank']}/{v['worse_rank']}; lower/higher margin={v['lower_margin']}/{v['higher_margin']}; mean margin delta={v['mean_margin_delta']:.3f}.\n")
        lines.append(f"- layout disagreement: {s['layout_disagreement']}\n\n")
    lines.append(f"Files: `{out_json}`\n")
    NOTE.write_text("".join(lines), encoding="utf-8")
    print(json.dumps({"status": result["status"], "parallel": result["modes"]["parallel"], "nonparallel": result["modes"]["nonparallel"], "note": str(NOTE)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
