#!/usr/bin/env python3
"""research: EWoK item-level four-cell flip audit for compact_view reinvestment.

This CPU-only audit uses existing official-coordinate EWoK prediction JSON files.
It computes correctness in the same way as official calculate_results_from_pred.py:
  correct iff pred.strip() == (Context1 + ' ' + Target1).strip().

It localizes the treatment x seed interaction by metadata and extracts exemplar rows
for the most negative patterns. It does not compute log-likelihood margins; the saved
official predictions only contain selected alternatives.
"""
import collections
import csv
import json
import pathlib

A01 = pathlib.Path("experiments/archive/representation_and_objectives")
A02 = pathlib.Path("experiments/archive/frontier_consolidation")
OUT_DIR = A01 / "data/ewok_item_flip_audit"
OUT_DIR.mkdir(parents=True, exist_ok=True)

GOLD_DIR = A01 / "data/pristine_official_coordinate/babylm-eval/strict/evaluation_data/full_eval/ewok_filtered"
PRED_PATHS = {
    "clean430": A02 / "data/official_ewok_clean_qwen/official_outputs/clean_qwen_seed43022/EWoK/chck_100M/official_ewok_clean_qwen_seed43022/zero_shot/mlm/ewok/ewok_filtered/predictions.json",
    "clean431": A02 / "data/official_ewok_clean_qwen/official_outputs/clean_qwen_seed43122/EWoK/chck_100M/official_ewok_clean_qwen_seed43122/zero_shot/mlm/ewok/ewok_filtered/predictions.json",
    "reinv430": A01 / "data/official_ewok_reeval/official_outputs/EWoK/chck_100M/official_ewok_reinvest/zero_shot/mlm/ewok/ewok_filtered/predictions.json",
    "reinv431": A01 / "data/official_ewok_reeval_seed43122/official_outputs/EWoK/chck_100M/official_ewok_seed43122/zero_shot/mlm/ewok/ewok_filtered/predictions.json",
}

preds = {k: json.loads(p.read_text()) for k, p in PRED_PATHS.items()}

items = []
for gold_path in sorted(GOLD_DIR.glob("*.jsonl")):
    domain = gold_path.stem
    gold_lines = [json.loads(line) for line in gold_path.read_text().splitlines()]
    pred_lists = {k: preds[k][domain]["predictions"] for k in preds}
    lens = {k: len(v) for k, v in pred_lists.items()}
    assert all(n == len(gold_lines) for n in lens.values()), (domain, len(gold_lines), lens)
    for idx, data in enumerate(gold_lines):
        correct_sentence = " ".join([data["Context1"], data["Target1"]]).strip()
        row = {
            "domain": domain,
            "idx": idx,
            "uid": f"{domain}_{idx}",
            "ConceptA": data.get("ConceptA"),
            "ConceptB": data.get("ConceptB"),
            "ContextType": data.get("ContextType"),
            "ContextDiff": data.get("ContextDiff"),
            "TargetDiff": data.get("TargetDiff"),
            "Context1": data.get("Context1"),
            "Context2": data.get("Context2"),
            "Target1": data.get("Target1"),
            "Target2": data.get("Target2"),
            "correct_sentence": correct_sentence,
        }
        for cell, plist in pred_lists.items():
            pr = plist[idx]["pred"].strip()
            row[cell + "_pred"] = pr
            row[cell + "_correct"] = int(pr == correct_sentence)
        row["TE430"] = row["reinv430_correct"] - row["clean430_correct"]
        row["TE431"] = row["reinv431_correct"] - row["clean431_correct"]
        row["DiD_item"] = row["TE431"] - row["TE430"]
        # compact pattern string helps later browsing
        row["pattern"] = "".join(str(row[c + "_correct"]) for c in ["clean430", "reinv430", "clean431", "reinv431"])
        items.append(row)

# Aggregate utility
GROUP_KEYS = [
    ("domain",),
    ("ContextType",),
    ("ContextDiff",),
    ("TargetDiff",),
    ("domain", "ContextType"),
    ("domain", "ContextDiff"),
    ("domain", "TargetDiff"),
    ("ContextType", "ContextDiff", "TargetDiff"),
]

def summarize_group(rows):
    n = len(rows)
    avg = lambda key: 100.0 * sum(r[key] for r in rows) / n
    te430 = avg("reinv430_correct") - avg("clean430_correct")
    te431 = avg("reinv431_correct") - avg("clean431_correct")
    return {
        "n": n,
        "clean430": avg("clean430_correct"),
        "clean431": avg("clean431_correct"),
        "reinv430": avg("reinv430_correct"),
        "reinv431": avg("reinv431_correct"),
        "TE430": te430,
        "TE431": te431,
        "DiD": te431 - te430,
        "mean_item_DiD": sum(r["DiD_item"] for r in rows) / n,
        "count_DiD_neg2": sum(1 for r in rows if r["DiD_item"] == -2),
        "count_DiD_neg1": sum(1 for r in rows if r["DiD_item"] == -1),
        "count_DiD_pos1": sum(1 for r in rows if r["DiD_item"] == 1),
        "count_DiD_pos2": sum(1 for r in rows if r["DiD_item"] == 2),
    }

aggregates = {}
for keys in GROUP_KEYS:
    table = collections.defaultdict(list)
    for r in items:
        table[tuple(r.get(k) for k in keys)].append(r)
    rows = []
    for key_tuple, rs in table.items():
        s = summarize_group(rs)
        s["group"] = dict(zip(keys, key_tuple))
        rows.append(s)
    rows.sort(key=lambda x: (x["DiD"], -x["n"]))
    aggregates["+".join(keys)] = rows

pattern_counts = collections.Counter(r["pattern"] for r in items)
pattern_summary = []
for pattern, n in pattern_counts.most_common():
    rs = [r for r in items if r["pattern"] == pattern]
    s = summarize_group(rs)
    s["pattern"] = pattern
    # pattern order: clean430, reinv430, clean431, reinv431
    s["meaning"] = "clean430,reinv430,clean431,reinv431 correctness bits"
    pattern_summary.append(s)
pattern_summary.sort(key=lambda x: (x["DiD"], -x["n"]))

# Examples for most negative DiD patterns, prioritizing major bad domains.
example_rows = []
for r in sorted(items, key=lambda x: (x["DiD_item"], x["domain"], x["idx"])):
    if r["DiD_item"] < 0:
        example_rows.append(r)
    if len(example_rows) >= 80:
        break

# Write compact CSVs.
item_csv = OUT_DIR / "ewok_negative_interaction_examples.csv"
with item_csv.open("w", newline="") as f:
    fields = [
        "uid", "domain", "idx", "ConceptA", "ConceptB", "ContextType", "ContextDiff", "TargetDiff",
        "clean430_correct", "reinv430_correct", "clean431_correct", "reinv431_correct", "TE430", "TE431", "DiD_item", "pattern",
        "Context1", "Target1", "Context2", "Target2", "clean430_pred", "reinv430_pred", "clean431_pred", "reinv431_pred",
    ]
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    for r in example_rows:
        w.writerow({k: r.get(k) for k in fields})

group_csv = OUT_DIR / "ewok_group_interactions.csv"
with group_csv.open("w", newline="") as f:
    fields = ["grouping", "group_json", "n", "clean430", "reinv430", "clean431", "reinv431", "TE430", "TE431", "DiD", "count_DiD_neg2", "count_DiD_neg1", "count_DiD_pos1", "count_DiD_pos2"]
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    for grouping, rows in aggregates.items():
        for s in rows:
            if s["n"] >= 20:  # suppress tiny cells in the CSV
                w.writerow({
                    "grouping": grouping,
                    "group_json": json.dumps(s["group"], sort_keys=True),
                    **{k: s[k] for k in fields if k not in ("grouping", "group_json")}
                })

payload = {
    "status": "EWOK_ITEM_FLIP_AUDIT",
    "method": "Correctness reconstructed from official EWoK gold and existing prediction JSONs; no margins/logits stored in predictions.",
    "prediction_paths": {k: str(v) for k, v in PRED_PATHS.items()},
    "total_items": len(items),
    "overall_micro_summary": summarize_group(items),
    "pattern_summary_sorted_by_DiD": pattern_summary,
    "aggregates": {k: v[:40] for k, v in aggregates.items()},
    "negative_examples_csv": str(item_csv),
    "group_interactions_csv": str(group_csv),
    "interpretation": {
        "what_this_can_say": "Counts item-level flips and metadata concentration of treatment-by-seed interaction on existing official predictions.",
        "what_this_cannot_say": "Does not measure confidence/margins; for that, rerun or instrument the official scorer around compute_results.py all_log_probs/rank_and_evaluate.",
    },
}

out_json = OUT_DIR / "ewok_item_flip_audit.json"
out_json.write_text(json.dumps(payload, indent=2))

# Human note
note = (A01.parents[2] / 'research/notes/representation_and_objectives/ewok_item_flip_audit.md')
worst_domains = aggregates["domain"][:8]
worst_meta = []
for grouping in ["domain+ContextType", "domain+ContextDiff", "domain+TargetDiff", "ContextType+ContextDiff+TargetDiff"]:
    worst_meta.extend([(grouping, x) for x in aggregates[grouping][:6] if x["n"] >= 20])
worst_meta.sort(key=lambda gx: (gx[1]["DiD"], -gx[1]["n"]))
worst_patterns = pattern_summary[:8]
note.write_text("\n".join([
    "# research — EWoK item flip audit",
    "",
    "This CPU audit reconstructs official EWoK correctness from the four existing prediction files. It does not contain log-likelihood margins; predictions store only selected alternatives.",
    "",
    f"Total EWoK items: {len(items)}",
    f"Micro treatment×seed interaction: {payload['overall_micro_summary']['DiD']:.3f} percentage points (official macro interaction from the 2×2 note is -2.462).",
    "",
    "## Worst official domains by micro interaction",
    *[f"- {s['group']['domain']}: DiD={s['DiD']:.3f}, TE430={s['TE430']:.3f}, TE431={s['TE431']:.3f}, n={s['n']}, neg2={s['count_DiD_neg2']}, neg1={s['count_DiD_neg1']}" for s in worst_domains],
    "",
    "## Worst metadata groups (n>=20)",
    *[f"- {grouping} {s['group']}: DiD={s['DiD']:.3f}, TE430={s['TE430']:.3f}, TE431={s['TE431']:.3f}, n={s['n']}" for grouping, s in worst_meta[:12]],
    "",
    "## Worst correctness-bit patterns",
    "Pattern order is clean430, reinvest430, clean431, reinvest431.",
    *[f"- {s['pattern']}: n={s['n']}, DiD={s['DiD']:.3f}, TE430={s['TE430']:.3f}, TE431={s['TE431']:.3f}" for s in worst_patterns],
    "",
    "## Next use",
    "Use the negative-example CSV as a target list for margin rescoring and for compact-row relation-feature analysis. A margin audit should instrument `sentence_zero_shot/compute_results.py`: for MLM, per-candidate summed log-probs are built in `compute_mlm_results` and passed to `rank_and_evaluate`, but only the selected sentence is saved. Export `stacked_probs` or candidate scores before argmax.",
    "",
    f"JSON: `{out_json}`",
    f"Negative examples CSV: `{item_csv}`",
    f"Group CSV: `{group_csv}`",
]))

print(json.dumps({
    "status": payload["status"],
    "out_json": str(out_json),
    "note": str(note),
    "total_items": len(items),
    "micro_DiD": round(payload["overall_micro_summary"]["DiD"], 6),
    "worst_domains": [{"domain": s["group"]["domain"], "DiD": round(s["DiD"], 3), "TE430": round(s["TE430"], 3), "TE431": round(s["TE431"], 3), "n": s["n"]} for s in worst_domains[:5]],
    "worst_patterns": [{"pattern": s["pattern"], "n": s["n"], "DiD": round(s["DiD"], 3)} for s in worst_patterns[:5]],
}, indent=2))
