#!/usr/bin/env python3
"""research: summarize aligned-vs-shuffled residual slices on full EWoK records.

Reads existing research CSVs only; no model inference and no training.
"""
from __future__ import annotations

import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path("experiments/archive/representation_and_objectives/data/full_ewok_coupled_turnover")
OUT_DIR = Path("experiments/archive/representation_and_objectives/data/aligned_shuffled_full_ewok_residual")
OUT_JSON = OUT_DIR / "aligned_shuffled_full_ewok_residual_slices.json"
OUT_MD = Path("research/notes/representation_and_objectives/aligned_shuffled_full_ewok_residual_slices.md")
TARGETS = ["mlm_only_20M", "coupled_aligned_20M", "coupled_shuffled_20M"]


def parse_bool(x: object) -> bool:
    return str(x).strip().lower() in {"true", "1", "yes"}


def parse_float(row: dict[str, str], key: str) -> float:
    try:
        return float(row.get(key, 0.0))
    except Exception:
        return 0.0


def correct_flag(row: dict[str, str]) -> bool:
    if "correct" in row:
        return parse_bool(row["correct"])
    return parse_bool(row.get("saved_model_correct_flag", False))


def stable_failure_flag(row: dict[str, str]) -> bool:
    if "stable_failure" in row:
        return parse_bool(row["stable_failure"])
    return parse_bool(row.get("conditional_reversal_failure_stable", False))


def load_records() -> dict[str, dict[int, dict[str, str]]]:
    records: dict[str, dict[int, dict[str, str]]] = {}
    for name in TARGETS:
        path = ROOT / "targets" / name / "ewok_interaction_records.csv"
        if not path.exists():
            raise FileNotFoundError(path)
        cur: dict[int, dict[str, str]] = {}
        with path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                cur[int(row["global_index"])] = row
        records[name] = cur
    keys = set(records[TARGETS[0]])
    for name in TARGETS[1:]:
        if set(records[name]) != keys:
            raise RuntimeError(f"index mismatch for {name}")
    if len(keys) != 7618:
        raise RuntimeError(f"expected 7618 rows, got {len(keys)}")
    return records


def summarize_group(records: dict[str, dict[int, dict[str, str]]], field: str) -> list[dict[str, object]]:
    keys = sorted(records["mlm_only_20M"])
    groups: dict[str, list[int]] = defaultdict(list)
    for idx in keys:
        groups[records["mlm_only_20M"][idx].get(field, "")].append(idx)
    out = []
    for group, idxs in groups.items():
        n = len(idxs)
        a_corr = s_corr = m_corr = 0
        a_st = s_st = m_st = 0
        a_only = s_only = both = neither = 0
        repaired_by_aligned_not_shuffled = 0
        repaired_by_shuffled_not_aligned = 0
        broken_by_aligned_not_shuffled = 0
        broken_by_shuffled_not_aligned = 0
        delta_int = []
        delta_off_t1 = []
        delta_off_t2 = []
        for idx in idxs:
            m = records["mlm_only_20M"][idx]
            a = records["coupled_aligned_20M"][idx]
            s = records["coupled_shuffled_20M"][idx]
            mc = correct_flag(m); ac = correct_flag(a); sc = correct_flag(s)
            ms = stable_failure_flag(m); ast = stable_failure_flag(a); sst = stable_failure_flag(s)
            m_corr += mc; a_corr += ac; s_corr += sc
            m_st += ms; a_st += ast; s_st += sst
            a_only += ac and not sc
            s_only += sc and not ac
            both += ac and sc
            neither += (not ac) and (not sc)
            repaired_by_aligned_not_shuffled += (not mc) and ac and not sc
            repaired_by_shuffled_not_aligned += (not mc) and sc and not ac
            broken_by_aligned_not_shuffled += mc and (not ac) and sc
            broken_by_shuffled_not_aligned += mc and ac and (not sc)
            delta_int.append(parse_float(a, "interaction_sum") - parse_float(s, "interaction_sum"))
            delta_off_t1.append(parse_float(a, "official_margin_t1_sum") - parse_float(s, "official_margin_t1_sum"))
            delta_off_t2.append(parse_float(a, "official_margin_t2_sum") - parse_float(s, "official_margin_t2_sum"))
        out.append({
            "group": group,
            "n": n,
            "mlm_accuracy": m_corr / n,
            "aligned_accuracy": a_corr / n,
            "shuffled_accuracy": s_corr / n,
            "aligned_minus_shuffled_correct_count": a_corr - s_corr,
            "aligned_correct_shuffled_wrong": a_only,
            "shuffled_correct_aligned_wrong": s_only,
            "both_correct": both,
            "both_wrong": neither,
            "mlm_stable_failure": m_st,
            "aligned_stable_failure": a_st,
            "shuffled_stable_failure": s_st,
            "aligned_minus_shuffled_stable_failure": a_st - s_st,
            "repaired_base_wrong_aligned_not_shuffled": repaired_by_aligned_not_shuffled,
            "repaired_base_wrong_shuffled_not_aligned": repaired_by_shuffled_not_aligned,
            "broken_base_correct_aligned_not_shuffled": broken_by_aligned_not_shuffled,
            "broken_base_correct_shuffled_not_aligned": broken_by_shuffled_not_aligned,
            "interaction_sum_aligned_minus_shuffled_mean": statistics.fmean(delta_int),
            "interaction_sum_aligned_minus_shuffled_median": statistics.median(delta_int),
            "official_margin_t1_aligned_minus_shuffled_mean": statistics.fmean(delta_off_t1),
            "official_margin_t2_aligned_minus_shuffled_mean": statistics.fmean(delta_off_t2),
        })
    out.sort(key=lambda r: (abs(int(r["aligned_minus_shuffled_correct_count"])), abs(int(r["aligned_minus_shuffled_stable_failure"])), abs(float(r["interaction_sum_aligned_minus_shuffled_mean"]))), reverse=True)
    return out


def main() -> None:
    records = load_records()
    payload = {
        "status": "PASS",
        "boundary": "Existing research full EWoK CSVs only; no model inference/training.",
        "root": str(ROOT),
        "slices": {field: summarize_group(records, field) for field in ["domain", "ContextDiff", "TargetDiff"]},
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = ["# research aligned-vs-shuffled full-EWoK residual slices", "", f"Status: **{payload['status']}**", "", "Existing research CSVs only; no model inference/training.", ""]
    for field, rows in payload["slices"].items():
        lines += [f"## {field}", "", "| group | n | acc aligned-shuffled count | stable aligned-shuffled | aligned-only | shuffled-only | interaction mean | interaction median |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
        for r in rows[:12]:
            lines.append(
                f"| {r['group']} | {r['n']} | {r['aligned_minus_shuffled_correct_count']} | {r['aligned_minus_shuffled_stable_failure']} | "
                f"{r['aligned_correct_shuffled_wrong']} | {r['shuffled_correct_aligned_wrong']} | "
                f"{float(r['interaction_sum_aligned_minus_shuffled_mean']):.4f} | {float(r['interaction_sum_aligned_minus_shuffled_median']):.4f} |"
            )
        lines.append("")
    lines += [f"JSON: `{OUT_JSON}`", ""]
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"status": "PASS", "out_json": str(OUT_JSON), "out_md": str(OUT_MD)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
