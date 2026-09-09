#!/usr/bin/env python3
"""research analysis for no-name-vocabulary span discovery.

Compares research lexical-available runs with research no-name-vocab causal
isolation. It computes state correctness from the two candidate scores in the
saved rows, rather than assuming a pre-existing `correct` field on state rows.
"""
from __future__ import annotations
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

PROJECT = Path("experiments/archive/representation_and_objectives")
OUT = PROJECT / "data/noname_span_analysis"

RUN_SPECS = {
    "lexical_shared_plus": PROJECT / "data/learned_shared/learned_shared_trunk_bs+1_seed29600",
    "lexical_shared_minus": PROJECT / "data/learned_shared/learned_shared_trunk_bs-1_seed29600",
    "lexical_untied_plus": PROJECT / "data/learned_untied/learned_untied_bs+1_seed29600",
    "lexical_untied_minus": PROJECT / "data/learned_untied/learned_untied_bs-1_seed29600",
    "noname_shared_plus": PROJECT / "data/noname_shared_bsplus_r2/learned_shared_trunk_bs+1_seed29600",
    "noname_shared_minus": PROJECT / "data/noname_shared_bsminus_r2/learned_shared_trunk_bs-1_seed29600",
    "noname_untied_plus": PROJECT / "data/noname_untied_bsplus/learned_untied_bs+1_seed29600",
    "noname_untied_minus": PROJECT / "data/noname_untied_bsminus/learned_untied_bs-1_seed29600",
    "noname_oracle_shared_plus": PROJECT / "data/noname_oracle_shared_bsplus/oracle_shared_trunk_bs+1_seed29600",
    "noname_oracle_shared_minus": PROJECT / "data/noname_oracle_shared_bsminus/oracle_shared_trunk_bs-1_seed29600",
}

PAIRS = [
    ("lexical_shared", "lexical_shared_plus", "lexical_shared_minus"),
    ("lexical_untied", "lexical_untied_plus", "lexical_untied_minus"),
    ("noname_shared", "noname_shared_plus", "noname_shared_minus"),
    ("noname_untied", "noname_untied_plus", "noname_untied_minus"),
    ("noname_oracle_shared", "noname_oracle_shared_plus", "noname_oracle_shared_minus"),
]


def load_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    return json.loads(path.read_text())


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    with path.open() as f:
        for line in f:
            s = line.strip()
            if s:
                rows.append(json.loads(s))
    return rows


def mean(xs: Iterable[float]) -> Optional[float]:
    xs = list(xs)
    return float(sum(xs) / len(xs)) if xs else None


def fmt(x: Any, nd: int = 3) -> str:
    if x is None:
        return "NA"
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return str(x)


def state_choice_breakdown(state_rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    by_q: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = defaultdict(list)
    for r in state_rows:
        by_q[(r.get("suite"), r.get("query_key"))].append(r)
    by_family: Dict[str, Dict[str, Any]] = defaultdict(lambda: {"n": 0, "correct": 0, "mean_margin": []})
    for _, rows in by_q.items():
        if len(rows) != 2:
            continue
        rows = sorted(rows, key=lambda x: int(x.get("candidate_index", 0)))
        pred = 0 if float(rows[0].get("score", 0.0)) >= float(rows[1].get("score", 0.0)) else 1
        target = 0 if bool(rows[0].get("label_true")) else 1
        r0 = rows[0]
        fam = str(r0.get("relation_family")) if bool(r0.get("is_changed")) else "unchanged"
        margin = float(rows[target].get("score", 0.0)) - float(rows[1-target].get("score", 0.0))
        d = by_family[fam]
        d["n"] += 1
        d["correct"] += int(pred == target)
        d["mean_margin"].append(margin)
    return {
        fam: {"n": d["n"], "acc": d["correct"] / d["n"] if d["n"] else None,
              "mean_margin": mean(d["mean_margin"])}
        for fam, d in sorted(by_family.items())
    }


def summarize_run(label: str, rd: Path) -> Dict[str, Any]:
    result = load_json(rd / "result.json")
    state = load_jsonl(rd / "state_predictions.jsonl")
    comp = load_jsonl(rd / "comparison_predictions.jsonl")
    out: Dict[str, Any] = {
        "label": label,
        "path": str(rd),
        "exists": rd.exists(),
        "has_result": result is not None,
        "n_state_rows": len(state),
        "n_comp_rows": len(comp),
    }
    if result:
        out.update({
            "condition": result.get("condition"),
            "bridge_sign": result.get("bridge_sign"),
            "matcher": result.get("matcher"),
            "epochs": result.get("epochs"),
            "vocab_size": result.get("vocab_size"),
            "init_hash_prefix": result.get("init_hash_prefix"),
            "train_state_acc": result.get("train_state_acc"),
            "train_cmp_acc": result.get("train_cmp_acc"),
            "central_eval": result.get("central_eval", {}),
            "matching_accuracy": result.get("matching_accuracy", {}),
        })
    if state:
        out["eval_state_choice_by_family"] = state_choice_breakdown(state)
    # Granular match probabilities for changed eval events, candidate_index=0 only.
    match_recs = []
    for r in state:
        if r.get("candidate_index") != 0 or not r.get("is_changed"):
            continue
        m = r.get("matching") or {}
        if m:
            rec = {k: m.get(k) for k in [
                "cand_p_cand", "cand_p_other", "cand_p_neither",
                "other_p_cand", "other_p_other", "other_p_neither"]}
            rec["suite"] = r.get("suite")
            rec["relation_family"] = r.get("relation_family")
            match_recs.append(rec)
    if match_recs:
        out["eval_match_prob_means"] = {
            k: mean(float(m[k]) for m in match_recs if m.get(k) is not None)
            for k in ["cand_p_cand", "cand_p_other", "cand_p_neither",
                      "other_p_cand", "other_p_other", "other_p_neither"]
        }
        out["eval_match_gt_half"] = {
            "cand": mean(1.0 if float(m.get("cand_p_cand", 0.0)) > 0.5 else 0.0 for m in match_recs),
            "other": mean(1.0 if float(m.get("other_p_other", 0.0)) > 0.5 else 0.0 for m in match_recs),
            "n": len(match_recs),
        }
        out["eval_match_rel_correct"] = {
            "cand_vs_other": mean(1.0 if float(m.get("cand_p_cand", 0.0)) > float(m.get("cand_p_other", 0.0)) else 0.0 for m in match_recs),
            "other_vs_cand": mean(1.0 if float(m.get("other_p_other", 0.0)) > float(m.get("other_p_cand", 0.0)) else 0.0 for m in match_recs),
            "cand_dominates_all": mean(1.0 if float(m.get("cand_p_cand", 0.0)) > max(float(m.get("cand_p_other", 0.0)), float(m.get("cand_p_neither", 0.0))) else 0.0 for m in match_recs),
            "other_dominates_all": mean(1.0 if float(m.get("other_p_other", 0.0)) > max(float(m.get("other_p_cand", 0.0)), float(m.get("other_p_neither", 0.0))) else 0.0 for m in match_recs),
            "n": len(match_recs),
        }
    return out


def pair_state_key(r: Dict[str, Any]) -> Tuple[Any, ...]:
    return (r.get("suite"), r.get("query_key"), r.get("candidate_index"), r.get("candidate"))


def analyze_pair(label: str, plus_rd: Path, minus_rd: Path) -> Dict[str, Any]:
    sp = load_jsonl(plus_rd / "state_predictions.jsonl")
    sm = load_jsonl(minus_rd / "state_predictions.jsonl")
    cp = load_jsonl(plus_rd / "comparison_predictions.jsonl")
    cm = load_jsonl(minus_rd / "comparison_predictions.jsonl")
    out: Dict[str, Any] = {
        "label": label,
        "plus_path": str(plus_rd),
        "minus_path": str(minus_rd),
        "available": bool(sp and sm),
        "n_plus_state": len(sp),
        "n_minus_state": len(sm),
        "n_plus_comp": len(cp),
        "n_minus_comp": len(cm),
    }
    if not (sp and sm):
        out["error"] = "missing state predictions for one or both signs"
        return out
    ip = {pair_state_key(r): r for r in sp}
    im = {pair_state_key(r): r for r in sm}
    common = sorted(set(ip) & set(im))
    fams = defaultdict(lambda: {"n": 0, "opposite": 0, "same": 0, "zero": 0,
                                "mean_abs_plus": [], "mean_abs_minus": []})
    for k in common:
        rp, rm = ip[k], im[k]
        de_p = float(rp.get("d_e", 0.0))
        de_m = float(rm.get("d_e", 0.0))
        fam = str(rp.get("relation_family")) if bool(rp.get("is_changed")) else "unchanged"
        d = fams[fam]
        d["n"] += 1
        prod = de_p * de_m
        if prod < 0:
            d["opposite"] += 1
        elif prod > 0:
            d["same"] += 1
        else:
            d["zero"] += 1
        d["mean_abs_plus"].append(abs(de_p))
        d["mean_abs_minus"].append(abs(de_m))
    out["state_sign_by_family"] = {}
    for fam, d in sorted(fams.items()):
        n = d["n"]
        out["state_sign_by_family"][fam] = {
            "n": n,
            "opposite": d["opposite"],
            "same": d["same"],
            "zero": d["zero"],
            "opposite_frac": d["opposite"] / n if n else None,
            "same_frac": d["same"] / n if n else None,
            "mean_abs_plus": mean(d["mean_abs_plus"]),
            "mean_abs_minus": mean(d["mean_abs_minus"]),
        }

    cp_i = {r.get("row_id"): r for r in cp}
    cm_i = {r.get("row_id"): r for r in cm}
    common_c = sorted(set(cp_i) & set(cm_i))
    comp_by_suite = defaultdict(lambda: {"n": 0, "d1_opposite": 0, "d1_same": 0,
                                         "d2_opposite": 0, "d2_same": 0,
                                         "product_same": 0, "product_opposite": 0,
                                         "plus_correct": 0, "minus_correct": 0})
    for rid in common_c:
        rp, rm = cp_i[rid], cm_i[rid]
        suite = str(rp.get("suite"))
        d = comp_by_suite[suite]
        d["n"] += 1
        for which, key in [("d1", "d_e1"), ("d2", "d_e2")]:
            prod = float(rp.get(key, 0.0)) * float(rm.get(key, 0.0))
            if prod < 0:
                d[f"{which}_opposite"] += 1
            elif prod > 0:
                d[f"{which}_same"] += 1
        pp = float(rp.get("d_e1", 0.0)) * float(rp.get("d_e2", 0.0))
        pm = float(rm.get("d_e1", 0.0)) * float(rm.get("d_e2", 0.0))
        prod2 = pp * pm
        if prod2 > 0:
            d["product_same"] += 1
        elif prod2 < 0:
            d["product_opposite"] += 1
        d["plus_correct"] += int(bool(rp.get("correct", False)))
        d["minus_correct"] += int(bool(rm.get("correct", False)))
    out["comparison_sign_by_suite"] = {
        suite: {**d,
                "d1_opposite_frac": d["d1_opposite"] / d["n"] if d["n"] else None,
                "d2_opposite_frac": d["d2_opposite"] / d["n"] if d["n"] else None,
                "product_same_frac": d["product_same"] / d["n"] if d["n"] else None,
                "plus_correct_frac": d["plus_correct"] / d["n"] if d["n"] else None,
                "minus_correct_frac": d["minus_correct"] / d["n"] if d["n"] else None}
        for suite, d in sorted(comp_by_suite.items())
    }
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    runs = {label: summarize_run(label, rd) for label, rd in RUN_SPECS.items()}
    pairs = {label: analyze_pair(label, RUN_SPECS[p], RUN_SPECS[m]) for label, p, m in PAIRS}
    obj = {"runs": runs, "pairs": pairs}
    (OUT / "noname_span_analysis.json").write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")

    lines = ["# research no-name-vocabulary span discovery analysis", "",
             "This analysis compares lexical-available research outputs with research no-name-vocab causal isolation. State correctness is recomputed from saved candidate-score pairs.", "",
             "## Run-level metrics", "",
             "| run | exists | vocab | train_state | train_cmp | graph_same | unchanged | hh_closure | mixed | eval cand>0.5 | eval other>0.5 | cand>other | other>cand |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for label in RUN_SPECS:
        r = runs[label]
        ce = r.get("central_eval", {}) or {}
        ma = ((r.get("matching_accuracy", {}) or {}).get("eval", {}) or {})
        mr = r.get("eval_match_rel_correct", {}) or {}
        lines.append("| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
            label, r.get("exists"), r.get("vocab_size"),
            fmt(r.get("train_state_acc")), fmt(r.get("train_cmp_acc")),
            fmt(ce.get("graph_same")), fmt(ce.get("unchanged")),
            fmt(ce.get("hh_closure")), fmt(ce.get("mixed_acc")),
            fmt(ma.get("cand_match_acc")), fmt(ma.get("other_match_acc")),
            fmt(mr.get("cand_vs_other")), fmt(mr.get("other_vs_cand"))))

    lines += ["", "## Eval state-choice accuracy by family", ""]
    for label in RUN_SPECS:
        r = runs[label]
        b = r.get("eval_state_choice_by_family") or {}
        if not b:
            continue
        lines.append(f"### {label}")
        lines.append("| family | n queries | acc | mean target margin |")
        lines.append("|---|---:|---:|---:|")
        for fam, d in b.items():
            lines.append(f"| {fam} | {d.get('n')} | {fmt(d.get('acc'))} | {fmt(d.get('mean_margin'))} |")
        lines.append("")

    lines += ["", "## Bridge-sign pair metrics", ""]
    for label, p in pairs.items():
        lines.append(f"### {label}")
        if not p.get("available"):
            lines.append(f"Missing paired predictions: {p.get('error')}")
            lines.append("")
            continue
        lines.append("| family | n rows | opposite frac | same frac | mean |d+| | mean |d-| |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        for fam, d in p.get("state_sign_by_family", {}).items():
            lines.append("| {} | {} | {} | {} | {} | {} |".format(
                fam, d.get("n"), fmt(d.get("opposite_frac")), fmt(d.get("same_frac")),
                fmt(d.get("mean_abs_plus")), fmt(d.get("mean_abs_minus"))))
        lines.append("")
        lines.append("Comparison coordinates by suite:")
        lines.append("| suite | n | d1 opposite | d2 opposite | product same | plus correct | minus correct |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|")
        for suite, d in p.get("comparison_sign_by_suite", {}).items():
            lines.append("| {} | {} | {} | {} | {} | {} | {} |".format(
                suite, d.get("n"), fmt(d.get("d1_opposite_frac")),
                fmt(d.get("d2_opposite_frac")), fmt(d.get("product_same_frac")),
                fmt(d.get("plus_correct_frac")), fmt(d.get("minus_correct_frac"))))
        lines.append("")

    ((OUT.parents[4] / 'research/documents/representation_and_objectives/data/noname_span_analysis/noname_span_analysis.md')).write_text("\n".join(lines) + "\n")
    print(json.dumps({"status": "NONAME_SPAN_ANALYSIS_COMPLETE",
                      "json": str(OUT / "noname_span_analysis.json"),
                      "md": str((OUT.parents[4] / 'research/documents/representation_and_objectives/data/noname_span_analysis/noname_span_analysis.md'))}, indent=2))


if __name__ == "__main__":
    main()
