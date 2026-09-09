#!/usr/bin/env python3
"""research review analysis for research SpanMatcher gauge probe.

This is a file-backed reviewer audit. It does not train models. It reads saved
state/comparison predictions and reconstructs the central claims: state-level
gauge reversal, direct-anchor reversal, unchanged stability, oracle/learned
alignment, and the mixed-comparison discrepancy.
"""
from __future__ import annotations

import json
import math
from collections import defaultdict, Counter
from pathlib import Path
from statistics import mean

ROOT = Path("experiments/archive/representation_and_objectives")
OUT = ROOT / "data/span_matcher_review"

RUNS = {
    "learned_shared_bsplus": ROOT / "data/learned_gauge/learned_shared_trunk_bs+1_seed29400",
    "learned_shared_bsminus": ROOT / "data/learned_gauge_bsminus/learned_shared_trunk_bs-1_seed29400",
    "oracle_shared_bsplus": ROOT / "data/oracle_gauge/oracle_shared_trunk_bs+1_seed29400",
    "oracle_shared_bsminus": ROOT / "data/oracle_gauge/oracle_shared_trunk_bs-1_seed29400",
    "oracle_untied_bsplus_partial": ROOT / "data/oracle_gauge/oracle_untied_bs+1_seed29400",
    "learned_untied_bsplus": ROOT / "data/learned_gauge_untied/learned_untied_bs+1_seed29400",
    "learned_untied_bsminus": ROOT / "data/learned_gauge_untied/learned_untied_bs-1_seed29400",
}

SUMMARY_FILES = {
    "learned_bsminus_summary": ROOT / "data/learned_gauge_bsminus/span_matcher_gauge_summary.json",
}


def load_jsonl(path: Path):
    rows = []
    if not path.exists():
        return rows
    with path.open(encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                rows.append(json.loads(s))
    return rows


def safe_mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else math.nan


def choice_rows(state_rows):
    by_key = defaultdict(list)
    for r in state_rows:
        by_key[(r.get("suite"), r.get("query_key"))].append(r)
    choices = []
    malformed = []
    for key, rows in by_key.items():
        if len(rows) != 2:
            malformed.append((key, len(rows)))
            continue
        rows = sorted(rows, key=lambda x: x.get("candidate_index", 0))
        pred_i = 0 if rows[0]["score"] >= rows[1]["score"] else 1
        target_i = None
        for i, r in enumerate(rows):
            if r.get("label_true"):
                target_i = i
        if target_i is None:
            malformed.append((key, "no true label"))
            continue
        r0 = rows[0]
        margin = rows[target_i]["score"] - rows[1 - target_i]["score"]
        choices.append({
            "suite": r0.get("suite"),
            "query_key": r0.get("query_key"),
            "correct": pred_i == target_i,
            "pred_i": pred_i,
            "target_i": target_i,
            "margin": margin,
            "d_e": r0.get("d_e"),
            "relation": r0.get("relation"),
            "relation_family": r0.get("relation_family"),
            "query_kind": r0.get("query_kind"),
            "initial_pattern": r0.get("initial_pattern"),
            "static_slot": r0.get("static_slot"),
            "is_direct_anchor": r0.get("is_direct_anchor"),
            "names": tuple(r0.get("names", [])),
            "object": r0.get("object"),
        })
    return choices, malformed


def central_from_choices(choices, comp_rows):
    ce = {}
    direct_changed = [c for c in choices if c["relation_family"] == "direct_anchor" and c["query_kind"] == "changed"]
    graph_changed = [c for c in choices if c["relation_family"] == "graph_transfer" and c["query_kind"] == "changed"]
    unchanged = [c for c in choices if c["query_kind"] == "unchanged"]
    same_init = [c for c in choices if c.get("initial_pattern") == "same" and c["query_kind"] == "changed"]
    def acc(items): return safe_mean(1.0 if x["correct"] else 0.0 for x in items)
    ce.update({
        "n_choices": len(choices),
        "direct_same": acc(direct_changed),
        "direct_n": len(direct_changed),
        "graph_same": acc(graph_changed),
        "graph_n": len(graph_changed),
        "unchanged": acc(unchanged),
        "unchanged_n": len(unchanged),
        "same_init_changed": acc(same_init),
        "same_init_n": len(same_init),
        "graph_mean_de": safe_mean(c["d_e"] for c in graph_changed),
        "graph_same_margin": safe_mean(c["margin"] for c in graph_changed),
    })
    # pair-both graph same-init, matching research's suite-level grouping
    graph_same_init = [c for c in graph_changed if c.get("initial_pattern") == "same"]
    by_suite = defaultdict(list)
    for c in graph_same_init:
        by_suite[c["suite"]].append(c)
    pair_total = pair_correct = 0
    for _, items in by_suite.items():
        if len(items) >= 2:
            pair_total += 1
            pair_correct += int(all(x["correct"] for x in items))
    ce["pair_both_graph_same"] = pair_correct / pair_total if pair_total else math.nan
    ce["pair_total"] = pair_total
    hh = [r for r in comp_rows if "heldheld" in r.get("suite", "")]
    mixed = [r for r in comp_rows if "mixed" in r.get("suite", "")]
    for tag, rows in [("hh", hh), ("mixed", mixed)]:
        ce[f"{tag}_n"] = len(rows)
        ce[f"{tag}_closure" if tag == "hh" else f"{tag}_acc"] = safe_mean(1.0 if r.get("correct") else 0.0 for r in rows)
        ce[f"{tag}_margin"] = safe_mean(r.get("signed_margin", math.nan) for r in rows)
        if rows:
            ce[f"{tag}_pred_true_rate"] = safe_mean(1.0 if r.get("pred") else 0.0 for r in rows)
    return ce


def comp_breakdown(comp_rows):
    out = {}
    for suite, rows in sorted(defaultdict(list, {k: [] for k in []}).items()):
        pass
    by_suite = defaultdict(list)
    by_rel = defaultdict(list)
    for r in comp_rows:
        by_suite[r.get("suite", "")].append(r)
        by_rel[(r.get("suite", ""), r.get("relation1"), r.get("relation2"))].append(r)
    out["by_suite"] = {}
    for k, rows in sorted(by_suite.items()):
        out["by_suite"][k] = summarize_comp_rows(rows)
    out["mixed_by_relation_pair"] = {}
    for k, rows in sorted(by_rel.items(), key=lambda kv: str(kv[0])):
        suite = k[0]
        if "mixed" not in suite:
            continue
        out["mixed_by_relation_pair"][str(k)] = summarize_comp_rows(rows)
    return out


def summarize_comp_rows(rows):
    if not rows:
        return {"n": 0}
    prod_acc = []
    product_abs = []
    for r in rows:
        prod = r.get("d_e1", 0.0) * r.get("d_e2", 0.0)
        pred_same_by_product = prod >= 0.0
        prod_acc.append(float(pred_same_by_product == bool(r.get("label"))))
        product_abs.append(abs(prod))
    return {
        "n": len(rows),
        "acc": safe_mean(float(r.get("correct")) for r in rows),
        "pred_true_rate": safe_mean(float(r.get("pred")) for r in rows),
        "label_true_rate": safe_mean(float(r.get("label")) for r in rows),
        "signed_margin_mean": safe_mean(r.get("signed_margin", math.nan) for r in rows),
        "logit_same_mean": safe_mean(r.get("logit_same", math.nan) for r in rows),
        "abs_logit_mean": safe_mean(abs(r.get("logit_same", 0.0)) for r in rows),
        "product_sign_acc": safe_mean(prod_acc),
        "abs_de_product_mean": safe_mean(product_abs),
    }


def paired_sign(choices_a, choices_b, filt):
    da = { (c["suite"], c["query_key"]): c for c in choices_a if filt(c) }
    db = { (c["suite"], c["query_key"]): c for c in choices_b if filt(c) }
    keys = sorted(set(da) & set(db))
    opp = same = zero = same_pred = both_correct_a = both_correct_b = 0
    products = []
    for k in keys:
        x = float(da[k].get("d_e", 0.0)); y = float(db[k].get("d_e", 0.0))
        products.append(x * y)
        if x == 0 or y == 0: zero += 1
        elif x * y < 0: opp += 1
        else: same += 1
        same_pred += int(da[k]["pred_i"] == db[k]["pred_i"])
        both_correct_a += int(da[k]["correct"])
        both_correct_b += int(db[k]["correct"])
    return {
        "paired_n": len(keys),
        "opposite_sign_n": opp,
        "opposite_sign_fraction": opp / len(keys) if keys else math.nan,
        "same_sign_n": same,
        "zero_n": zero,
        "same_pred_fraction": same_pred / len(keys) if keys else math.nan,
        "a_acc": both_correct_a / len(keys) if keys else math.nan,
        "b_acc": both_correct_b / len(keys) if keys else math.nan,
        "mean_product": safe_mean(products),
    }


def paired_learned_oracle(choices_l, choices_o, filt):
    dl = { (c["suite"], c["query_key"]): c for c in choices_l if filt(c) }
    do = { (c["suite"], c["query_key"]): c for c in choices_o if filt(c) }
    keys = sorted(set(dl) & set(do))
    same_sign = opp_sign = zero = 0
    for k in keys:
        x = float(dl[k].get("d_e", 0.0)); y = float(do[k].get("d_e", 0.0))
        if x == 0 or y == 0: zero += 1
        elif x * y > 0: same_sign += 1
        else: opp_sign += 1
    return {
        "paired_n": len(keys),
        "same_sign_n": same_sign,
        "same_sign_fraction": same_sign / len(keys) if keys else math.nan,
        "opposite_sign_n": opp_sign,
        "zero_n": zero,
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    runs = {}
    for name, d in RUNS.items():
        state = load_jsonl(d / "state_predictions.jsonl")
        comp = load_jsonl(d / "comparison_predictions.jsonl")
        choices, malformed = choice_rows(state)
        runs[name] = {
            "dir": str(d),
            "state_rows": len(state),
            "comp_rows": len(comp),
            "choices": choices,
            "malformed": malformed,
            "central": central_from_choices(choices, comp),
            "comp_breakdown": comp_breakdown(comp),
        }
    # concise metrics without raw choices in JSON
    review = {
        "runs": {
            k: {kk: vv for kk, vv in v.items() if kk != "choices"}
            for k, v in runs.items()
        },
        "paired_reversal": {},
        "learned_vs_oracle": {},
        "summary_files_read": {},
    }
    learned_plus = runs.get("learned_shared_bsplus", {}).get("choices", [])
    learned_minus = runs.get("learned_shared_bsminus", {}).get("choices", [])
    oracle_plus = runs.get("oracle_shared_bsplus", {}).get("choices", [])
    oracle_minus = runs.get("oracle_shared_bsminus", {}).get("choices", [])
    filters = {
        "graph_changed": lambda c: c.get("relation_family") == "graph_transfer" and c.get("query_kind") == "changed",
        "direct_changed": lambda c: c.get("relation_family") == "direct_anchor" and c.get("query_kind") == "changed",
        "unchanged": lambda c: c.get("query_kind") == "unchanged",
        "all_changed": lambda c: c.get("query_kind") == "changed",
    }
    for fname, filt in filters.items():
        review["paired_reversal"][f"learned_{fname}"] = paired_sign(learned_plus, learned_minus, filt)
        review["paired_reversal"][f"oracle_{fname}"] = paired_sign(oracle_plus, oracle_minus, filt)
        review["learned_vs_oracle"][f"bsplus_{fname}"] = paired_learned_oracle(learned_plus, oracle_plus, filt)
        review["learned_vs_oracle"][f"bsminus_{fname}"] = paired_learned_oracle(learned_minus, oracle_minus, filt)
    for name, path in SUMMARY_FILES.items():
        if path.exists():
            review["summary_files_read"][name] = json.loads(path.read_text())
    json_path = OUT / "span_matcher_review_metrics.json"
    json_path.write_text(json.dumps(review, indent=2, sort_keys=True, ensure_ascii=False) + "\n")

    md = []
    md.append("# research review of research SpanMatcher gauge result\n\n")
    md.append("## Reconstructed central metrics from saved predictions\n\n")
    md.append("| run | state rows | comp rows | malformed choices | graph_same | graph_margin | direct_same | unchanged | hh_closure | mixed_acc | mixed_margin | mixed pred-true |\n")
    md.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
    for name, rec in review["runs"].items():
        ce = rec["central"]
        md.append(
            f"| {name} | {rec['state_rows']} | {rec['comp_rows']} | {len(rec['malformed'])} "
            f"| {ce.get('graph_same', math.nan):.3f} | {ce.get('graph_same_margin', math.nan):.3f} "
            f"| {ce.get('direct_same', math.nan):.3f} | {ce.get('unchanged', math.nan):.3f} "
            f"| {ce.get('hh_closure', math.nan):.3f} | {ce.get('mixed_acc', math.nan):.3f} "
            f"| {ce.get('mixed_margin', math.nan):.3f} | {ce.get('mixed_pred_true_rate', math.nan):.3f} |\n"
        )
    md.append("\n## Row-paired sign checks\n\n")
    md.append("| comparison | paired_n | opposite sign | same sign | zero | same prediction | a_acc | b_acc |\n")
    md.append("|---|---:|---:|---:|---:|---:|---:|---:|\n")
    for name, rec in review["paired_reversal"].items():
        md.append(
            f"| {name} | {rec['paired_n']} | {rec['opposite_sign_fraction']:.3f} | "
            f"{rec['same_sign_n']} | {rec['zero_n']} | {rec['same_pred_fraction']:.3f} "
            f"| {rec['a_acc']:.3f} | {rec['b_acc']:.3f} |\n"
        )
    md.append("\n## Learned versus oracle sign alignment\n\n")
    md.append("| comparison | paired_n | same sign fraction | opposite sign n | zero |\n")
    md.append("|---|---:|---:|---:|---:|\n")
    for name, rec in review["learned_vs_oracle"].items():
        md.append(f"| {name} | {rec['paired_n']} | {rec['same_sign_fraction']:.3f} | {rec['opposite_sign_n']} | {rec['zero_n']} |\n")
    md.append("\n## Mixed-comparison breakdown\n\n")
    for run_name in ["learned_shared_bsplus", "learned_shared_bsminus", "oracle_shared_bsplus", "oracle_shared_bsminus"]:
        br = review["runs"][run_name]["comp_breakdown"]
        md.append(f"### {run_name}\n\n")
        md.append("By suite:\n\n")
        md.append("| suite | n | acc | pred_true | label_true | signed_margin | abs_logit | product_sign_acc |\n")
        md.append("|---|---:|---:|---:|---:|---:|---:|---:|\n")
        for suite, s in br["by_suite"].items():
            md.append(f"| {suite} | {s['n']} | {s['acc']:.3f} | {s['pred_true_rate']:.3f} | {s['label_true_rate']:.3f} | {s['signed_margin_mean']:.3f} | {s['abs_logit_mean']:.3f} | {s['product_sign_acc']:.3f} |\n")
        md.append("\nMixed relation pairs:\n\n")
        md.append("| suite/rels | n | acc | pred_true | label_true | signed_margin | abs_logit | product_sign_acc |\n")
        md.append("|---|---:|---:|---:|---:|---:|---:|---:|\n")
        for rel, s in br["mixed_by_relation_pair"].items():
            md.append(f"| {rel} | {s['n']} | {s['acc']:.3f} | {s['pred_true_rate']:.3f} | {s['label_true_rate']:.3f} | {s['signed_margin_mean']:.3f} | {s['abs_logit_mean']:.3f} | {s['product_sign_acc']:.3f} |\n")
        md.append("\n")
    md.append("## Reviewer interpretation from this analysis\n\n")
    md.append("Saved predictions support the state-level sign fingerprint: graph-transfer changed rows reverse between learned bs+ and bs- while unchanged rows remain accurate. The mixed comparison surface remains weak: held-held closure is perfect, but mixed held-seen comparison accuracy is only 0.5 in learned cells and not a full research replication. This should be retained as a boundary on the comparison pathway, not ignored.\n")
    md_path = (OUT.parents[4] / 'research/documents/representation_and_objectives/data/span_matcher_review/span_matcher_review_metrics.md')
    md_path.write_text("".join(md))
    print(json.dumps({"status": "SPAN_MATCHER_REVIEW_COMPLETE", "json": str(json_path), "md": str(md_path)}, indent=2))


if __name__ == "__main__":
    main()
