#!/usr/bin/env python3
"""Row-set analysis of FW 70M EWoK predictions.

The 70M cheap files currently contain complete EWoK predictions for compact_view
and row-block source_breadth. This script reproduces the official per-domain EWoK
mean from prediction files and locates the breadth-vs-compact movement with respect
to the stable conditional-failure sets found in research.

CPU-only: no model loading, no evaluation examples are used for training decisions.
"""
from __future__ import annotations

import csv
import json
import math
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path.cwd()
A01_WS = ROOT / "experiments/archive/representation_and_objectives"
A02_WS = ROOT / "experiments/archive/frontier_consolidation"
EWOK_DIR = A01_WS / "data/pristine_official_coordinate/babylm-eval/strict/evaluation_data/full_eval/ewok_filtered"
RECORDS = A01_WS / "data/full_ewok_interaction_specificity/ewok_full_interaction_specificity_records_fullcpu.csv"
OUT = A01_WS / "data/fw_70m_ewok_prediction_overlap"
NOTE = A01_WS / "notes/fw_70m_ewok_prediction_overlap.md"

PRED_PATHS = {
    "compact_view_70M": A02_WS / "data/fw_comparison_eval/official_outputs/compact_view_chck_70M/EWoK/chck_70M/full_step086_compact_view_chck_70M_EWoK/zero_shot/mlm/ewok/ewok_filtered/predictions.json",
    "source_breadth_rowblock_70M": A02_WS / "data/fw_comparison_eval/official_outputs/source_breadth_chck_70M/EWoK/chck_70M/full_step086_source_breadth_chck_70M_EWoK/zero_shot/mlm/ewok/ewok_filtered/predictions.json",
}

MODELS = ["legal40_depth_12x384_43022", "legal40_8x480_43022"]


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def norm(s: str) -> str:
    return " ".join(str(s).split()).strip()


def as_bool(x: Any) -> bool:
    return str(x).strip().lower() in {"true", "1", "yes"}


def qstats(xs: list[float]) -> dict[str, Any]:
    vals = sorted(float(x) for x in xs if math.isfinite(float(x)))
    if not vals:
        return {"n": 0}
    def q(p: float) -> float:
        if len(vals) == 1: return vals[0]
        pos = p * (len(vals) - 1)
        lo = math.floor(pos); hi = math.ceil(pos)
        if lo == hi: return vals[lo]
        return vals[lo] * (hi - pos) + vals[hi] * (pos - lo)
    return {"n": len(vals), "min": vals[0], "p05": q(0.05), "mean": statistics.fmean(vals), "median": statistics.median(vals), "p95": q(0.95), "max": vals[-1]}


def load_gold() -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    flat: list[dict[str, Any]] = []
    by_domain: dict[str, list[dict[str, Any]]] = {}
    gi = 0
    for p in sorted(EWOK_DIR.glob("*.jsonl")):
        rows = []
        with p.open(encoding="utf-8") as f:
            for li, line in enumerate(f):
                obj = json.loads(line)
                rec = {
                    "global_index": gi,
                    "domain": p.stem,
                    "local_index": li,
                    "correct_sentence": norm(obj["Context1"] + " " + obj["Target1"]),
                    "ContextType": obj.get("ContextType"),
                    "ContextDiff": obj.get("ContextDiff"),
                    "TargetDiff": obj.get("TargetDiff"),
                    "ConceptA": obj.get("ConceptA"),
                    "ConceptB": obj.get("ConceptB"),
                }
                flat.append(rec); rows.append(rec); gi += 1
        by_domain[p.stem] = rows
    return flat, by_domain


def load_predictions(path: Path) -> dict[tuple[str, int], str]:
    pdata = json.loads(path.read_text(encoding="utf-8"))
    out: dict[tuple[str, int], str] = {}
    for domain, bundle in pdata.items():
        preds = bundle.get("predictions", []) if isinstance(bundle, dict) else []
        for j, rec in enumerate(preds):
            rid = str(rec.get("id", ""))
            try:
                idx = int(rid.rsplit("_", 1)[1])
            except Exception:
                idx = j
            out[(domain, idx)] = norm(str(rec.get("pred", "")))
    return out


def score_arm(name: str, preds: dict[tuple[str, int], str], by_domain: dict[str, list[dict[str, Any]]]) -> tuple[dict[str, Any], dict[int, bool]]:
    domain_rows = []
    flags: dict[int, bool] = {}
    for domain, rows in by_domain.items():
        correct = 0; total = 0; missing = 0
        for rec in rows:
            pred = preds.get((domain, rec["local_index"]))
            if pred is None:
                missing += 1
                continue
            ok = pred == rec["correct_sentence"]
            flags[int(rec["global_index"])] = ok
            total += 1; correct += int(ok)
        pct = 100.0 * correct / total if total else None
        domain_rows.append({"arm": name, "domain": domain, "correct": correct, "total": total, "missing": missing, "accuracy_pct": pct})
    official_mean = statistics.fmean(float(r["accuracy_pct"]) for r in domain_rows if r["accuracy_pct"] is not None)
    micro_correct = sum(r["correct"] for r in domain_rows)
    micro_total = sum(r["total"] for r in domain_rows)
    return {
        "arm": name,
        "official_domain_mean_pct": official_mean,
        "micro_accuracy_pct": 100.0 * micro_correct / micro_total,
        "micro_correct": micro_correct,
        "micro_total": micro_total,
        "missing_predictions": sum(r["missing"] for r in domain_rows),
        "domain_rows": sorted(domain_rows, key=lambda r: r["domain"]),
    }, flags


def load_step100_sets() -> dict[str, set[int]]:
    by_idx: dict[int, dict[str, dict[str, str]]] = defaultdict(dict)
    with RECORDS.open(encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            model = r.get("model", "")
            if model in MODELS:
                by_idx[int(r["global_index"])] [model] = r
    sets = {"depth_stable": set(), "legal40_8x480_stable": set(), "both_stable": set(), "either_stable": set(), "both_wrong": set(), "all_rows": set(by_idx)}
    for gi, mm in by_idx.items():
        if not all(m in mm for m in MODELS):
            continue
        d_fail = as_bool(mm[MODELS[0]].get("conditional_reversal_failure_stable"))
        s_fail = as_bool(mm[MODELS[1]].get("conditional_reversal_failure_stable"))
        if d_fail: sets["depth_stable"].add(gi)
        if s_fail: sets["legal40_8x480_stable"].add(gi)
        if d_fail and s_fail: sets["both_stable"].add(gi)
        if d_fail or s_fail: sets["either_stable"].add(gi)
        if as_bool(mm[MODELS[0]].get("saved_model_wrong_flag")) and as_bool(mm[MODELS[1]].get("saved_model_wrong_flag")):
            sets["both_wrong"].add(gi)
    return sets


def subset_summary(name: str, indices: set[int], flags_by_arm: dict[str, dict[int, bool]], flat_by_idx: dict[int, dict[str, Any]]) -> dict[str, Any]:
    arms = list(flags_by_arm)
    out: dict[str, Any] = {"subset": name, "n": len(indices)}
    for arm in arms:
        vals = [flags_by_arm[arm].get(i) for i in indices if i in flags_by_arm[arm]]
        out[f"{arm}_covered"] = len(vals)
        out[f"{arm}_correct"] = sum(bool(v) for v in vals)
        out[f"{arm}_accuracy_pct"] = 100.0 * sum(bool(v) for v in vals) / len(vals) if vals else None
    if len(arms) == 2:
        a, b = arms
        both = [i for i in indices if i in flags_by_arm[a] and i in flags_by_arm[b]]
        a_true_b_false = [i for i in both if flags_by_arm[a][i] and not flags_by_arm[b][i]]
        a_false_b_true = [i for i in both if (not flags_by_arm[a][i]) and flags_by_arm[b][i]]
        both_true = [i for i in both if flags_by_arm[a][i] and flags_by_arm[b][i]]
        both_false = [i for i in both if (not flags_by_arm[a][i]) and (not flags_by_arm[b][i])]
        out.update({
            "paired_covered": len(both),
            f"{b}_minus_{a}_correct_count": len(a_false_b_true) - len(a_true_b_false),
            f"{b}_minus_{a}_accuracy_points": 100.0 * (len(a_false_b_true) - len(a_true_b_false)) / len(both) if both else None,
            f"{a}_only_correct": len(a_true_b_false),
            f"{b}_only_correct": len(a_false_b_true),
            "both_correct": len(both_true),
            "both_wrong": len(both_false),
        })
        # Domain location of the paired changes, useful for distinguishing stable relational repairs from domain weighting.
        ctr_b_gain = Counter(flat_by_idx[i]["domain"] for i in a_false_b_true)
        ctr_a_gain = Counter(flat_by_idx[i]["domain"] for i in a_true_b_false)
        out["source_breadth_only_correct_by_domain_top"] = dict(ctr_b_gain.most_common(12))
        out["compact_only_correct_by_domain_top"] = dict(ctr_a_gain.most_common(12))
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    flat, by_domain = load_gold()
    flat_by_idx = {int(r["global_index"]): r for r in flat}
    pred_maps = {name: load_predictions(path) for name, path in PRED_PATHS.items()}
    arm_summaries = {}
    flags_by_arm = {}
    domain_rows_all = []
    for name, preds in pred_maps.items():
        summ, flags = score_arm(name, preds, by_domain)
        arm_summaries[name] = {k: v for k, v in summ.items() if k != "domain_rows"}
        flags_by_arm[name] = flags
        domain_rows_all.extend(summ["domain_rows"])
    # Domain deltas breadth minus compact.
    domain_by = defaultdict(dict)
    for r in domain_rows_all:
        domain_by[r["domain"]][r["arm"]] = r
    domain_delta_rows = []
    for domain, d in sorted(domain_by.items()):
        c = d.get("compact_view_70M", {})
        b = d.get("source_breadth_rowblock_70M", {})
        domain_delta_rows.append({
            "domain": domain,
            "compact_accuracy_pct": c.get("accuracy_pct"),
            "breadth_accuracy_pct": b.get("accuracy_pct"),
            "breadth_minus_compact_pct": None if c.get("accuracy_pct") is None or b.get("accuracy_pct") is None else b["accuracy_pct"] - c["accuracy_pct"],
            "n": c.get("total") or b.get("total"),
            "compact_correct": c.get("correct"),
            "breadth_correct": b.get("correct"),
            "breadth_minus_compact_correct_count": None if c.get("correct") is None or b.get("correct") is None else b["correct"] - c["correct"],
        })
    stable_sets = load_step100_sets()
    subset_rows = []
    complement_sets = {}
    all_rows = stable_sets["all_rows"]
    for k, s in stable_sets.items():
        subset_rows.append(subset_summary(k, s, flags_by_arm, flat_by_idx))
        if k != "all_rows":
            complement_sets[f"not_{k}"] = all_rows - s
    for k, s in complement_sets.items():
        subset_rows.append(subset_summary(k, s, flags_by_arm, flat_by_idx))
    # Examples where breadth fixes compact on the robust both_stable subset.
    c_flags = flags_by_arm["compact_view_70M"]
    b_flags = flags_by_arm["source_breadth_rowblock_70M"]
    examples = []
    for gi in sorted(stable_sets["both_stable"]):
        if gi in c_flags and gi in b_flags and (not c_flags[gi]) and b_flags[gi]:
            rec = flat_by_idx[gi]
            examples.append({k: rec.get(k) for k in ["global_index", "domain", "local_index", "ContextType", "ContextDiff", "TargetDiff", "ConceptA", "ConceptB", "correct_sentence"]})
        if len(examples) >= 30:
            break
    out = {
        "status": "FW_70M_EWOK_PREDICTION_OVERLAP",
        "created_utc": now(),
        "prediction_paths": {k: str(v) for k, v in PRED_PATHS.items()},
        "ewok_dir": str(EWOK_DIR),
        "official_reproduction": arm_summaries,
        "domain_deltas_breadth_minus_compact": domain_delta_rows,
        "stable_subset_rows": subset_rows,
        "breadth_fixes_compact_on_both_stable_first30": examples,
        "interpretation": {
            "official_mean_rule": "EWoK is mean of domain accuracies, not micro accuracy over all 7618 rows.",
            "scope": "70M EWoK-only prediction movement; not a full endpoint result and not enough to allocate new training by itself.",
        },
    }
    json_path = OUT / "fw_70m_ewok_prediction_overlap.json"
    json_path.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    with (OUT / "ewok_70m_domain_deltas.csv").open("w", newline="", encoding="utf-8") as f:
        fields = ["domain", "n", "compact_accuracy_pct", "breadth_accuracy_pct", "breadth_minus_compact_pct", "compact_correct", "breadth_correct", "breadth_minus_compact_correct_count"]
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows([{k: r.get(k) for k in fields} for r in domain_delta_rows])
    with (OUT / "ewok_70m_stable_subset_deltas.csv").open("w", newline="", encoding="utf-8") as f:
        fields = ["subset", "n", "compact_view_70M_covered", "compact_view_70M_correct", "compact_view_70M_accuracy_pct", "source_breadth_rowblock_70M_covered", "source_breadth_rowblock_70M_correct", "source_breadth_rowblock_70M_accuracy_pct", "paired_covered", "source_breadth_rowblock_70M_minus_compact_view_70M_correct_count", "source_breadth_rowblock_70M_minus_compact_view_70M_accuracy_points", "compact_view_70M_only_correct", "source_breadth_rowblock_70M_only_correct", "both_correct", "both_wrong"]
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows([{k: r.get(k) for k in fields} for r in subset_rows])
    with (OUT / "breadth_fixes_compact_on_both_stable_first30.csv").open("w", newline="", encoding="utf-8") as f:
        if examples:
            fields = list(examples[0].keys()); w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(examples)
    # Note
    lines = []
    lines.append("# research — A02 FW 70M EWoK prediction overlap\n")
    lines.append("This CPU-only readout uses the completed 70M EWoK prediction files from A02's FW compact and row-block breadth arms. It reproduces official EWoK as the mean of domain accuracies, then locates breadth-vs-compact movement with respect to research stable conditional-failure sets.\n")
    lines.append("## Score reproduction\n")
    for arm, rec in arm_summaries.items():
        lines.append(f"- {arm}: official-domain mean {rec['official_domain_mean_pct']:.4f}; micro accuracy {rec['micro_accuracy_pct']:.4f}; {rec['micro_correct']}/{rec['micro_total']} row matches.")
    lines.append("\n## Domain movement, breadth minus compact\n")
    for r in sorted(domain_delta_rows, key=lambda x: float(x.get("breadth_minus_compact_pct") or 0), reverse=True):
        lines.append(f"- {r['domain']}: {r['breadth_minus_compact_pct']:+.3f} points ({r['breadth_minus_compact_correct_count']:+} rows, n={r['n']}).")
    lines.append("\n## Stable-failure subset movement\n")
    for r in subset_rows:
        if r["subset"] in ["both_stable", "either_stable", "depth_stable", "legal40_8x480_stable", "not_both_stable", "all_rows"]:
            lines.append(f"- {r['subset']}: n={r['n']}, compact {r.get('compact_view_70M_accuracy_pct'):.3f}%, breadth {r.get('source_breadth_rowblock_70M_accuracy_pct'):.3f}%, breadth-compact {r.get('source_breadth_rowblock_70M_minus_compact_view_70M_accuracy_points'):+.3f} points ({r.get('source_breadth_rowblock_70M_minus_compact_view_70M_correct_count'):+} rows).")
    lines.append("\n## Scientific reading\n")
    lines.append("The 70M EWoK row movement is early and column-only. It can indicate whether the independent-source breadth arm touches the same robust relational-error subset, but it cannot settle the FW mechanism or Overall SOTA route without complete paired cheap7/100M results and the research four-cell/margin readouts.")
    lines.append("\n## Evidence files\n")
    lines.append(f"- JSON: `{json_path}`")
    lines.append(f"- Domain delta CSV: `{OUT / 'ewok_70m_domain_deltas.csv'}`")
    lines.append(f"- Stable subset delta CSV: `{OUT / 'ewok_70m_stable_subset_deltas.csv'}`")
    NOTE.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": out["status"],
        "official_means": {k: v["official_domain_mean_pct"] for k, v in arm_summaries.items()},
        "both_stable_delta_points": next(r.get("source_breadth_rowblock_70M_minus_compact_view_70M_accuracy_points") for r in subset_rows if r["subset"] == "both_stable"),
        "json": str(json_path),
        "note": str(NOTE),
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
