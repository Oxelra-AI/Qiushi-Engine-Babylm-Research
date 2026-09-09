#!/usr/bin/env python3
"""research: Partition the rel_eq0 error mass by option class.

The binding arms lose ~7-11 points on rel_eq0 vs chck82. Recency accounts for ~1/6.
This script diagnoses what the remaining ~5/6 of the error mass looks like:
- option word count / item count bias (trained on long multiword phrases)
- 'nothing' preference shifts
- initial-state vs operation-derived option preference
- how the prediction moves when chck82 was correct but binding is wrong

The key output: for each binding arm, the items where chck82 is correct and binding
is wrong, classified by what the wrong prediction is.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse
import collections
import csv
import json
import pathlib
import re
import sys
import time
from typing import Any

ROOT = _public_path('.')
STUDY = _public_path('experiments/archive/relation_learning')

ENTITY_DATA = _public_path('experiments/archive/representation_and_objectives/data/pristine_official_coordinate/babylm-eval/strict/evaluation_data/full_eval/entity_tracking')

RECENCY_CSV = _public_path('experiments/archive/relation_learning/data/entity_recency_diagnostic_alpha050_075_100/prediction_rows.csv')


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", str(s).strip().lower().rstrip("."))


def count_items(s: str) -> int:
    """Count the number of objects in an Entity contents string."""
    n = norm(s)
    if not n or n == "nothing":
        return 0
    return len([x for x in re.split(r"\s+and\s+", n) if x.strip()])


def word_count(s: str) -> int:
    return len(norm(s).split())


def load_entity_items() -> dict[str, dict]:
    """Load all Entity benchmark items with their options, keyed by pred_id."""
    items = {}
    for fname in ["ambiref.jsonl", "regular.jsonl", "move_contents.jsonl"]:
        etype = fname.replace(".jsonl", "")
        fp = ENTITY_DATA / fname
        if not fp.exists():
            continue
        # Items are grouped by numops in the eval harness
        by_numops: dict[int, list] = collections.defaultdict(list)
        with open(fp) as f:
            for line in f:
                d = json.loads(line)
                by_numops[d["numops"]].append(d)
        # pred_id format: {etype}_{numops}_ops_{item_index}
        for numops, group in by_numops.items():
            for idx, item in enumerate(group):
                pid = f"{etype}_{numops}_ops_{idx}"
                items[pid] = {
                    "pred_id": pid,
                    "entity_type": etype,
                    "numops": numops,
                    "options": item["options"],
                    "gold": item["options"][0],  # first option is gold
                    "input_prefix": item["input_prefix"],
                    "sample_id": item.get("sample_id"),
                }
    return items


def parse_initial_states(prefix: str) -> dict[int, str]:
    """Extract initial box contents from the input prefix."""
    states = {}
    # Pattern: Box N contains X, Box M contains Y, ...
    # The initial state is before any operations
    # Find the initial state description (before first operation)
    parts = re.split(r"\.\s+(?=[A-Z])", prefix)
    init_part = parts[0] if parts else prefix
    for m in re.finditer(r"Box\s+(\d+)\s+contains\s+(.+?)(?:,\s+Box|\.$|$)", init_part):
        box = int(m.group(1))
        contents = m.group(2).strip().rstrip(",").rstrip(".")
        states[box] = norm(contents)
    return states


def classify_option(opt: str, initial_states: dict[int, str], gold: str) -> dict:
    """Classify an option by its relationship to the scenario."""
    n_opt = norm(opt)
    n_gold = norm(gold)
    is_gold = (n_opt == n_gold)
    is_nothing = (n_opt == "nothing" or n_opt == "")
    
    # Check if option matches any initial state
    matches_initial = any(n_opt == v for v in initial_states.values())
    
    return {
        "is_gold": is_gold,
        "is_nothing": is_nothing,
        "matches_initial": matches_initial,
        "word_count": word_count(opt),
        "item_count": count_items(opt),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(_public_path('experiments/archive/relation_learning/data/rel_eq0_error_partition')))
    args = ap.parse_args()
    out = pathlib.Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Load Entity benchmark items
    items = load_entity_items()
    print(f"Loaded {len(items)} Entity benchmark items", flush=True)

    # Load prediction rows
    rows = []
    with open(RECENCY_CSV) as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    print(f"Loaded {len(rows)} prediction rows", flush=True)

    # Index predictions by (label, pred_id)
    pred_by_label: dict[str, dict[str, dict]] = collections.defaultdict(dict)
    for r in rows:
        pred_by_label[r["label"]][r["pred_id"]] = r

    labels = ["chck82", "binding_ep25_alpha0p50", "binding_ep25_alpha0p75", "binding_ep25_alpha1p00", "coherent86"]
    
    # Focus on rel_eq0 items
    chck82_preds = pred_by_label["chck82"]
    rel_eq0_pids = [pid for pid, r in chck82_preds.items() if int(r["relevant_updates"]) == 0]
    print(f"rel_eq0 items: {len(rel_eq0_pids)}", flush=True)

    # For each binding label, find items where chck82 correct -> binding wrong
    flip_analysis = {}
    option_analysis = {}

    for label in labels:
        if label == "chck82":
            continue
        preds = pred_by_label[label]
        
        # Flips: chck82 correct, label wrong
        flips = []
        for pid in rel_eq0_pids:
            chck_row = chck82_preds[pid]
            label_row = preds.get(pid)
            if not label_row:
                continue
            chck_correct = int(chck_row["correct"])
            label_correct = int(label_row["correct"])
            if chck_correct == 1 and label_correct == 0:
                flips.append(pid)
        
        # Reverse flips: chck82 wrong -> label correct
        reverse_flips = []
        for pid in rel_eq0_pids:
            chck_row = chck82_preds[pid]
            label_row = preds.get(pid)
            if not label_row:
                continue
            if int(chck_row["correct"]) == 0 and int(label_row["correct"]) == 1:
                reverse_flips.append(pid)

        # Classify what the wrong predictions look like
        flip_details = []
        wc_gold_list = []
        wc_pred_list = []
        ic_gold_list = []
        ic_pred_list = []
        nothing_count = 0
        matches_initial_count = 0
        matches_recency_count = 0
        longer_than_gold = 0
        shorter_than_gold = 0
        more_items_than_gold = 0
        fewer_items_than_gold = 0
        
        for pid in flips:
            label_row = preds[pid]
            chck_row = chck82_preds[pid]
            item = items.get(pid)
            
            pred_text = label_row["pred"]
            gold_text = chck_row["gold"]
            
            n_pred = norm(pred_text)
            n_gold = norm(gold_text)
            
            pred_wc = word_count(pred_text)
            gold_wc = word_count(gold_text)
            pred_ic = count_items(pred_text)
            gold_ic = count_items(gold_text)
            
            wc_gold_list.append(gold_wc)
            wc_pred_list.append(pred_wc)
            ic_gold_list.append(gold_ic)
            ic_pred_list.append(pred_ic)
            
            if n_pred == "nothing" or not n_pred:
                nothing_count += 1
            if pred_wc > gold_wc:
                longer_than_gold += 1
            elif pred_wc < gold_wc:
                shorter_than_gold += 1
            if pred_ic > gold_ic:
                more_items_than_gold += 1
            elif pred_ic < gold_ic:
                fewer_items_than_gold += 1
            
            # Check if prediction matches any initial state
            if item:
                initial = parse_initial_states(item["input_prefix"])
                if any(n_pred == v for v in initial.values()):
                    matches_initial_count += 1
            
            # Check recency
            if int(label_row.get("pred_is_recency", 0)):
                matches_recency_count += 1
            
            irr_ops = int(label_row.get("irrelevant_ops", 0))
            flip_details.append({
                "pid": pid,
                "pred": pred_text,
                "gold": gold_text,
                "pred_wc": pred_wc,
                "gold_wc": gold_wc,
                "pred_ic": pred_ic, 
                "gold_ic": gold_ic,
                "is_nothing": n_pred in ("nothing", ""),
                "is_recency": int(label_row.get("pred_is_recency", 0)),
                "irrelevant_ops": irr_ops,
            })
        
        n_flips = len(flips)
        summary = {
            "label": label,
            "rel_eq0_n": len(rel_eq0_pids),
            "chck82_correct_on_rel_eq0": sum(1 for pid in rel_eq0_pids if int(chck82_preds[pid]["correct"])),
            "flips_to_wrong": n_flips,
            "reverse_flips_to_correct": len(reverse_flips),
            "net_loss": n_flips - len(reverse_flips),
        }
        
        if n_flips > 0:
            summary.update({
                "nothing_picks": nothing_count,
                "nothing_pct": round(100 * nothing_count / n_flips, 1),
                "matches_initial_state": matches_initial_count,
                "matches_initial_pct": round(100 * matches_initial_count / n_flips, 1),
                "matches_recency": matches_recency_count,
                "matches_recency_pct": round(100 * matches_recency_count / n_flips, 1),
                "longer_than_gold": longer_than_gold,
                "longer_pct": round(100 * longer_than_gold / n_flips, 1),
                "shorter_than_gold": shorter_than_gold,
                "shorter_pct": round(100 * shorter_than_gold / n_flips, 1),
                "more_items_than_gold": more_items_than_gold,
                "fewer_items_than_gold": fewer_items_than_gold,
                "mean_pred_wc": round(sum(wc_pred_list) / n_flips, 2),
                "mean_gold_wc": round(sum(wc_gold_list) / n_flips, 2),
                "mean_pred_ic": round(sum(ic_pred_list) / n_flips, 2),
                "mean_gold_ic": round(sum(ic_gold_list) / n_flips, 2),
            })
        
        flip_analysis[label] = summary
        
        # By irrelevant_ops stratum
        by_irr = collections.defaultdict(lambda: {"n": 0, "nothing": 0, "initial": 0, "recency": 0, "longer": 0, "shorter": 0})
        for fd in flip_details:
            irr = fd["irrelevant_ops"]
            if irr <= 3:
                key = "1-3"
            elif irr <= 6:
                key = "4-6"
            else:
                key = "7+"
            by_irr[key]["n"] += 1
            if fd["is_nothing"]:
                by_irr[key]["nothing"] += 1
            if fd.get("is_recency"):
                by_irr[key]["recency"] += 1
            if fd["pred_wc"] > fd["gold_wc"]:
                by_irr[key]["longer"] += 1
            elif fd["pred_wc"] < fd["gold_wc"]:
                by_irr[key]["shorter"] += 1
        
        summary["by_irrelevant_ops"] = dict(by_irr)

    # Also: global option-class analysis across ALL rel_eq0 items
    # For each label, what's the average word count and item count of the prediction?
    for label in labels:
        preds = pred_by_label[label]
        wcs = []
        ics = []
        nothing_n = 0
        for pid in rel_eq0_pids:
            r = preds.get(pid)
            if not r:
                continue
            wcs.append(word_count(r["pred"]))
            ics.append(count_items(r["pred"]))
            if norm(r["pred"]) in ("nothing", ""):
                nothing_n += 1
        option_analysis[label] = {
            "mean_pred_wc": round(sum(wcs) / len(wcs), 2) if wcs else 0,
            "mean_pred_ic": round(sum(ics) / len(ics), 2) if ics else 0,
            "nothing_picks": nothing_n,
            "nothing_pct": round(100 * nothing_n / len(wcs), 1) if wcs else 0,
            "n": len(wcs),
        }
    
    # Also compute gold stats for reference
    gold_wcs = []
    gold_ics = []
    gold_nothing = 0
    for pid in rel_eq0_pids:
        g = chck82_preds[pid]["gold"]
        gold_wcs.append(word_count(g))
        gold_ics.append(count_items(g))
        if norm(g) in ("nothing", ""):
            gold_nothing += 1
    option_analysis["gold_distribution"] = {
        "mean_wc": round(sum(gold_wcs) / len(gold_wcs), 2),
        "mean_ic": round(sum(gold_ics) / len(gold_ics), 2),
        "nothing_n": gold_nothing,
        "nothing_pct": round(100 * gold_nothing / len(gold_wcs), 1),
        "n": len(gold_wcs),
    }

    # Write results
    obj = {
        "status": "REL_EQ0_ERROR_PARTITION",
        "created_utc": now(),
        "flip_analysis": flip_analysis,
        "option_analysis": option_analysis,
    }
    (out / "summary.json").write_text(json.dumps(obj, indent=2), encoding="utf-8")

    # Write readable summary
    lines = ["# research rel_eq0 error-mass partition", ""]
    lines.append("## Flip analysis: chck82 correct -> binding wrong on rel_eq0")
    lines.append("")
    for label, s in flip_analysis.items():
        lines.append(f"### {label}")
        lines.append(f"- rel_eq0 items: {s['rel_eq0_n']}")
        lines.append(f"- chck82 correct: {s['chck82_correct_on_rel_eq0']}")
        lines.append(f"- flips to wrong: {s['flips_to_wrong']}")
        lines.append(f"- reverse flips to correct: {s['reverse_flips_to_correct']}")
        lines.append(f"- net loss: {s['net_loss']}")
        if s["flips_to_wrong"] > 0:
            lines.append(f"- nothing picks: {s['nothing_picks']} ({s['nothing_pct']}%)")
            lines.append(f"- matches initial state: {s['matches_initial_state']} ({s['matches_initial_pct']}%)")
            lines.append(f"- matches recency: {s['matches_recency']} ({s['matches_recency_pct']}%)")
            lines.append(f"- longer than gold: {s['longer_than_gold']} ({s['longer_pct']}%)")
            lines.append(f"- shorter than gold: {s['shorter_than_gold']} ({s['shorter_pct']}%)")
            lines.append(f"- more items than gold: {s['more_items_than_gold']}")
            lines.append(f"- fewer items than gold: {s['fewer_items_than_gold']}")
            lines.append(f"- mean pred word count: {s['mean_pred_wc']} vs gold: {s['mean_gold_wc']}")
            lines.append(f"- mean pred item count: {s['mean_pred_ic']} vs gold: {s['mean_gold_ic']}")
            if "by_irrelevant_ops" in s:
                lines.append(f"- by irrelevant ops stratum:")
                for k, v in sorted(s["by_irrelevant_ops"].items()):
                    lines.append(f"  - {k}: n={v['n']}, nothing={v['nothing']}, recency={v['recency']}, longer={v['longer']}, shorter={v['shorter']}")
        lines.append("")
    
    lines.append("## Global prediction characteristics on rel_eq0")
    lines.append("")
    lines.append("| label | n | mean_pred_wc | mean_pred_ic | nothing_picks | nothing_pct |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for label in list(option_analysis.keys()):
        s = option_analysis[label]
        if label == "gold_distribution":
            lines.append(f"| GOLD | {s['n']} | {s['mean_wc']} | {s['mean_ic']} | {s.get('nothing_n', s.get('nothing_picks', 0))} | {s['nothing_pct']} |")
        else:
            lines.append(f"| {label} | {s['n']} | {s['mean_pred_wc']} | {s['mean_pred_ic']} | {s['nothing_picks']} | {s['nothing_pct']} |")
    lines.append("")

    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # Write flip details for deeper inspection
    all_flip_rows = []
    for label in labels:
        if label == "chck82":
            continue
        preds = pred_by_label[label]
        for pid in rel_eq0_pids:
            chck_row = chck82_preds[pid]
            label_row = preds.get(pid)
            if not label_row:
                continue
            if int(chck_row["correct"]) == 1 and int(label_row["correct"]) == 0:
                item = items.get(pid)
                initial = parse_initial_states(item["input_prefix"]) if item else {}
                n_pred = norm(label_row["pred"])
                all_flip_rows.append({
                    "label": label,
                    "pid": pid,
                    "pred": label_row["pred"],
                    "gold": chck_row["gold"],
                    "pred_wc": word_count(label_row["pred"]),
                    "gold_wc": word_count(chck_row["gold"]),
                    "pred_ic": count_items(label_row["pred"]),
                    "gold_ic": count_items(chck_row["gold"]),
                    "is_nothing": norm(label_row["pred"]) in ("nothing", ""),
                    "is_recency": int(label_row.get("pred_is_recency", 0)),
                    "matches_initial": any(n_pred == v for v in initial.values()),
                    "irrelevant_ops": int(label_row.get("irrelevant_ops", 0)),
                    "entity_type": label_row.get("entity_type", ""),
                })
    
    if all_flip_rows:
        keys = list(all_flip_rows[0].keys())
        with open(out / "flip_details.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(all_flip_rows)

    print(json.dumps(obj, indent=2), flush=True)


if __name__ == "__main__":
    main()
