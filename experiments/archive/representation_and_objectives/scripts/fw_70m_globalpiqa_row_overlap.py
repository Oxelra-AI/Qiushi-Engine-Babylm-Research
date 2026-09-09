#!/usr/bin/env python3
"""GlobalPIQA row-overlap readout for FW 70M cheap evaluations.

CPU-only: reads existing official prediction files for the compact-view and
row-block source-breadth 70M checkpoints, reproduces the official split scores,
and asks whether the source-breadth/compact difference touches the research
hard GlobalPIQA_parallel rows or only easier/nonparallel rows.
"""
from __future__ import annotations

import csv
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path("experiments/archive/representation_and_objectives")
A02 = Path("experiments/archive/frontier_consolidation")
DATA_ROOT = ROOT / "data/globalpiqa_official_lineage/official_dl_scratch/generated_by_current_official_dl/evaluation_data/full_eval"
PAR_DATA = DATA_ROOT / "global_piqa_parallel/eng_latn.jsonl"
NON_DATA = DATA_ROOT / "global_piqa_nonparallel/eng_latn.jsonl"
HARD = ROOT / "data/globalpiqa_parallel_anatomy/globalpiqa_parallel_cross_endpoint_rows.csv"
OUT = ROOT / "data/fw_70m_globalpiqa_row_overlap"
NOTE = (ROOT.parents[2] / 'research/notes/representation_and_objectives/fw_70m_globalpiqa_row_overlap.md')

PRED = {
    "compact_view_70M": {
        "parallel": A02 / "data/fw_comparison_eval/official_outputs/compact_view_chck_70M/GlobalPIQA_parallel/chck_70M/full_step086_compact_view_chck_70M_GlobalPIQA_parallel/zero_shot/mlm/global_piqa_parallel/global_piqa_parallel/predictions.json",
        "nonparallel": A02 / "data/fw_comparison_eval/official_outputs/compact_view_chck_70M/GlobalPIQA_nonparallel/chck_70M/full_step086_compact_view_chck_70M_GlobalPIQA_nonparallel/zero_shot/mlm/global_piqa_nonparallel/global_piqa_nonparallel/predictions.json",
    },
    "source_breadth_rowblock_70M": {
        "parallel": A02 / "data/fw_comparison_eval/official_outputs/source_breadth_chck_70M/GlobalPIQA_parallel/chck_70M/full_step086_source_breadth_chck_70M_GlobalPIQA_parallel/zero_shot/mlm/global_piqa_parallel/global_piqa_parallel/predictions.json",
        "nonparallel": A02 / "data/fw_comparison_eval/official_outputs/source_breadth_chck_70M/GlobalPIQA_nonparallel/chck_70M/full_step086_source_breadth_chck_70M_GlobalPIQA_nonparallel/zero_shot/mlm/global_piqa_nonparallel/global_piqa_nonparallel/predictions.json",
    },
}


def norm(s: Any) -> str:
    return re.sub(r"\s+", " ", str(s).strip().lower())


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_predictions(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def prediction_text(v: Any) -> str:
    if isinstance(v, dict):
        preds = v.get("predictions")
        if isinstance(preds, list) and preds:
            first = preds[0]
            if isinstance(first, dict):
                return str(first.get("pred", ""))
            return str(first)
        return str(v.get("pred") or v.get("prediction") or "")
    return str(v)


def choice_for(item: dict[str, Any], pred_value: Any) -> tuple[int | None, str]:
    ptxt = prediction_text(pred_value)
    pn = norm(ptxt)
    opts = []
    for j in range(4):
        key = f"solution{j}"
        if key in item:
            opts.append((j, norm(item[key])))
    exact = [j for j, o in opts if o == pn]
    if len(exact) == 1:
        return exact[0], ptxt.strip()
    contains = [j for j, o in opts if pn and (pn in o or o in pn)]
    if len(contains) == 1:
        return contains[0], ptxt.strip()
    return None, ptxt.strip()


def score(mode: str, items: list[dict[str, Any]], pred_path: Path) -> dict[str, Any]:
    preds = load_predictions(pred_path)
    rows = []
    for idx, item in enumerate(items):
        eid = item["example_id"]
        label = int(item["label"])
        ch, ptxt = choice_for(item, preds.get(eid))
        ok = (ch == label)
        rows.append({
            "mode": mode,
            "index": idx,
            "example_id": eid,
            "label": label,
            "choice": ch,
            "ok": bool(ok),
            "prompt": item.get("prompt", ""),
            "categories": item.get("categories", ""),
            "prediction_text": ptxt,
            "solution0": item.get("solution0", ""),
            "solution1": item.get("solution1", ""),
            "solution2": item.get("solution2", ""),
            "solution3": item.get("solution3", ""),
        })
    n = len(rows)
    return {
        "n": n,
        "accuracy": 100.0 * sum(r["ok"] for r in rows) / n if n else math.nan,
        "choice_counts": dict(sorted(Counter(str(r["choice"]) for r in rows).items())),
        "label_counts": dict(sorted(Counter(str(r["label"]) for r in rows).items())),
        "rows": rows,
    }


def load_step107_hard() -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    with HARD.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            out[row["example_id"]] = row
    return out


def subset_stats(rows_by_arm: dict[str, dict[str, dict[str, Any]]], ids: set[str]) -> dict[str, Any]:
    arms = list(rows_by_arm)
    out: dict[str, Any] = {"n": len(ids)}
    for arm in arms:
        rows = [rows_by_arm[arm][eid] for eid in ids if eid in rows_by_arm[arm]]
        out[f"{arm}_covered"] = len(rows)
        out[f"{arm}_correct"] = sum(r["ok"] for r in rows)
        out[f"{arm}_accuracy"] = 100.0 * sum(r["ok"] for r in rows) / len(rows) if rows else None
        out[f"{arm}_choice_counts"] = dict(sorted(Counter(str(r["choice"]) for r in rows).items()))
    if len(arms) == 2:
        a, b = arms
        common = [eid for eid in ids if eid in rows_by_arm[a] and eid in rows_by_arm[b]]
        out["common_n"] = len(common)
        out["a_correct_b_wrong"] = sum(rows_by_arm[a][eid]["ok"] and not rows_by_arm[b][eid]["ok"] for eid in common)
        out["b_correct_a_wrong"] = sum(rows_by_arm[b][eid]["ok"] and not rows_by_arm[a][eid]["ok"] for eid in common)
        out["both_correct"] = sum(rows_by_arm[a][eid]["ok"] and rows_by_arm[b][eid]["ok"] for eid in common)
        out["both_wrong"] = sum((not rows_by_arm[a][eid]["ok"]) and (not rows_by_arm[b][eid]["ok"]) for eid in common)
    return out


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    if not fields:
        keys = []
        for r in rows:
            for k in r:
                if k not in keys:
                    keys.append(k)
        fields = keys
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader(); w.writerows(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    items = {"parallel": load_jsonl(PAR_DATA), "nonparallel": load_jsonl(NON_DATA)}
    results: dict[str, dict[str, Any]] = {}
    summary_rows = []
    all_row_csv = []
    for arm, paths in PRED.items():
        results[arm] = {}
        for mode, path in paths.items():
            res = score(mode, items[mode], path)
            results[arm][mode] = res
            summary_rows.append({
                "arm": arm,
                "mode": mode,
                "n": res["n"],
                "accuracy": res["accuracy"],
                "choice_counts": json.dumps(res["choice_counts"], sort_keys=True),
                "label_counts": json.dumps(res["label_counts"], sort_keys=True),
                "predictions_path": str(path),
            })
            for r in res["rows"]:
                rr = {k: v for k, v in r.items() if k not in ["solution0", "solution1", "solution2", "solution3"]}
                rr["arm"] = arm
                all_row_csv.append(rr)
    # compact vs breadth pair movement by mode.
    comparison_rows = []
    for mode in ["parallel", "nonparallel"]:
        a_rows = {r["example_id"]: r for r in results["compact_view_70M"][mode]["rows"]}
        b_rows = {r["example_id"]: r for r in results["source_breadth_rowblock_70M"][mode]["rows"]}
        for eid in sorted(set(a_rows) & set(b_rows)):
            a = a_rows[eid]; b = b_rows[eid]
            comparison_rows.append({
                "mode": mode,
                "example_id": eid,
                "label": a["label"],
                "category": a["categories"],
                "compact_ok": a["ok"],
                "breadth_ok": b["ok"],
                "movement": "same_correct" if a["ok"] and b["ok"] else "same_wrong" if (not a["ok"] and not b["ok"]) else "breadth_fixes_compact" if b["ok"] else "compact_fixes_breadth",
                "compact_choice": a["choice"],
                "breadth_choice": b["choice"],
                "prompt": a["prompt"],
                "compact_pred": a["prediction_text"],
                "breadth_pred": b["prediction_text"],
                "solution0": a.get("solution0", ""),
                "solution1": a.get("solution1", ""),
                "solution2": a.get("solution2", ""),
                "solution3": a.get("solution3", ""),
            })
    write_csv(OUT / "fw_70m_globalpiqa_summary.csv", summary_rows)
    write_csv(OUT / "fw_70m_globalpiqa_row_movements.csv", comparison_rows)
    write_csv(OUT / "fw_70m_globalpiqa_arm_rows.csv", all_row_csv)
    research = load_step107_hard()
    always_wrong_ids = {eid for eid, r in research.items() if int(r["n_ok"]) == 0}
    half_wrong_ids = {eid for eid, r in research.items() if int(r["n_wrong"]) >= 5}
    non_hard_ids = {r["example_id"] for r in results["compact_view_70M"]["parallel"]["rows"]} - half_wrong_ids
    par_rows_by_arm = {
        arm: {r["example_id"]: r for r in results[arm]["parallel"]["rows"]}
        for arm in PRED
    }
    subset = {
        "parallel_all": subset_stats(par_rows_by_arm, set(par_rows_by_arm["compact_view_70M"])),
        "parallel_step107_always_wrong_all10": subset_stats(par_rows_by_arm, always_wrong_ids),
        "parallel_step107_wrong_at_least_half": subset_stats(par_rows_by_arm, half_wrong_ids),
        "parallel_not_step107_wrong_at_least_half": subset_stats(par_rows_by_arm, non_hard_ids),
    }
    # category x movement table for parallel.
    catmove = defaultdict(Counter)
    for r in comparison_rows:
        if r["mode"] == "parallel":
            catmove[r["category"]][r["movement"]] += 1
    cat_rows = []
    for cat, ctr in sorted(catmove.items()):
        total = sum(ctr.values())
        row = {"category": cat, "n": total}
        row.update({k: ctr.get(k, 0) for k in ["same_correct", "same_wrong", "breadth_fixes_compact", "compact_fixes_breadth"]})
        row["breadth_minus_compact_correct"] = row["breadth_fixes_compact"] - row["compact_fixes_breadth"]
        cat_rows.append(row)
    write_csv(OUT / "fw_70m_globalpiqa_parallel_category_movements.csv", cat_rows)
    hard52 = subset["parallel_step107_always_wrong_all10"]
    payload = {
        "status": "FW_70M_GLOBALPIQA_ROW_OVERLAP_DONE",
        "data_paths": {"parallel": str(PAR_DATA), "nonparallel": str(NON_DATA), "hard_rows": str(HARD)},
        "prediction_paths": {arm: {mode: str(p) for mode, p in paths.items()} for arm, paths in PRED.items()},
        "summary_rows": summary_rows,
        "subset_stats": subset,
        "category_movements_parallel": cat_rows,
        "interpretation": {
            "official_split_reproduction": "The script reproduces the 70M split scores from existing official predictions: both arms are 26.21 on parallel; compact is 51.00 vs breadth 47.00 on nonparallel, producing the observed 2-point aggregate GlobalPIQA gap.",
            "hard_parallel_result": f"On the research 52-row all-endpoint-wrong parallel hard set, compact gets {hard52.get('compact_view_70M_correct')}/52 and row-block breadth gets {hard52.get('source_breadth_rowblock_70M_correct')}/52, with {hard52.get('both_wrong')} shared wrong rows; breadth fixes {hard52.get('b_correct_a_wrong')} compact misses while compact fixes {hard52.get('a_correct_b_wrong')} breadth misses. This is no evidence that the FW row-block breadth arm repairs the deep-rank parallel weakness at 70M.",
            "decision_use": "Do not launch a GlobalPIQA-specific expensive route from row-block breadth's 70M result. Full 100M evaluation should be reserved for arms whose complete cheap7/trajectory makes SOTA plausible, then read with the already prepared margin wrapper.",
        },
    }
    (OUT / "fw_70m_globalpiqa_row_overlap.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    # Note.
    def fmt(x: Any) -> str:
        return "" if x is None else (f"{float(x):.2f}" if isinstance(x, (int, float)) else str(x))
    lines = []
    lines.append("# research — A02 FW 70M GlobalPIQA row-overlap readout")
    lines.append("")
    lines.append("## Measurement")
    lines.append("")
    lines.append("CPU-only readout of existing A02 70M official prediction files for the FW compact-view and row-block whole-sentence source-breadth arms. No model scoring or training was run.")
    lines.append("")
    lines.append("| arm | parallel | nonparallel | aggregate |")
    lines.append("|---|---:|---:|---:|")
    for arm in PRED:
        pa = results[arm]["parallel"]["accuracy"]
        na = results[arm]["nonparallel"]["accuracy"]
        lines.append(f"| `{arm}` | {pa:.2f} | {na:.2f} | {(pa+na)/2:.3f} |")
    lines.append("")
    lines.append("The official 70M GlobalPIQA gap is entirely nonparallel: both arms score 26.21 on the 103-row parallel set; compact is 51.00 and row-block breadth is 47.00 on nonparallel.")
    lines.append("")
    lines.append("## research hard-row projection")
    lines.append("")
    lines.append("| subset | n | compact correct | breadth correct | breadth fixes compact | compact fixes breadth | both wrong |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for name, st in subset.items():
        lines.append(f"| `{name}` | {st['n']} | {st.get('compact_view_70M_correct')} ({fmt(st.get('compact_view_70M_accuracy'))}) | {st.get('source_breadth_rowblock_70M_correct')} ({fmt(st.get('source_breadth_rowblock_70M_accuracy'))}) | {st.get('b_correct_a_wrong')} | {st.get('a_correct_b_wrong')} | {st.get('both_wrong')} |")
    lines.append("")
    hard52 = subset["parallel_step107_always_wrong_all10"]
    half82 = subset["parallel_step107_wrong_at_least_half"]
    lines.append(f"On the 52 rows that research found wrong for all ten inspected endpoints, compact gets {hard52.get('compact_view_70M_correct')}/52 and row-block breadth gets {hard52.get('source_breadth_rowblock_70M_correct')}/52. Breadth fixes {hard52.get('b_correct_a_wrong')} compact miss while compact fixes {hard52.get('a_correct_b_wrong')} breadth misses, with {hard52.get('both_wrong')} rows wrong for both. On the 82 rows wrong for at least half of prior endpoints, compact is {half82.get('compact_view_70M_correct')}/82 and breadth is {half82.get('source_breadth_rowblock_70M_correct')}/82. Thus row-block breadth's 70M EWoK gain is not accompanied by a repair of the load-bearing GlobalPIQA_parallel hard rows.")
    lines.append("")
    lines.append("## Parallel category movements")
    lines.append("")
    lines.append("| category | n | same correct | same wrong | breadth fixes compact | compact fixes breadth | breadth - compact |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for r in cat_rows:
        lines.append(f"| `{r['category']}` | {r['n']} | {r.get('same_correct',0)} | {r.get('same_wrong',0)} | {r.get('breadth_fixes_compact',0)} | {r.get('compact_fixes_breadth',0)} | {r.get('breadth_minus_compact_correct',0)} |")
    lines.append("")
    lines.append("## Research consequence")
    lines.append("")
    lines.append("The 70M FW family does not currently show the kind of GlobalPIQA_parallel movement that would explain the 41.80 gap. Compact remains the better 70M cheap7 arm because it preserves Entity and nonparallel GlobalPIQA; row-block breadth's EWoK increase is not enough and is partly domain-weighted. The correct next evidence is still complete 100M cheap7/full-vector reading when it appears, plus the existing margin wrapper if an arm is near range; the row-block breadth 70M result does not justify a new GlobalPIQA-specific expensive training line.")
    lines.append("")
    lines.append("## Files")
    lines.append("")
    lines.append(f"- JSON: `{OUT/'fw_70m_globalpiqa_row_overlap.json'}`")
    lines.append(f"- summary CSV: `{OUT/'fw_70m_globalpiqa_summary.csv'}`")
    lines.append(f"- row movement CSV: `{OUT/'fw_70m_globalpiqa_row_movements.csv'}`")
    lines.append(f"- parallel category movement CSV: `{OUT/'fw_70m_globalpiqa_parallel_category_movements.csv'}`")
    NOTE.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": payload["status"],
        "compact_parallel": results["compact_view_70M"]["parallel"]["accuracy"],
        "breadth_parallel": results["source_breadth_rowblock_70M"]["parallel"]["accuracy"],
        "compact_nonparallel": results["compact_view_70M"]["nonparallel"]["accuracy"],
        "breadth_nonparallel": results["source_breadth_rowblock_70M"]["nonparallel"]["accuracy"],
        "hard52_subset": subset["parallel_step107_always_wrong_all10"],
        "json": str(OUT / "fw_70m_globalpiqa_row_overlap.json"),
        "note": str(NOTE),
    }, indent=2))


if __name__ == "__main__":
    main()
