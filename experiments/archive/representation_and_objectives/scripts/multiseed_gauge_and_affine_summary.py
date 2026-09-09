#!/usr/bin/env python3
"""research multi-seed gauge and affine synthesis.

Reads research/291 primary gauge result directories and their saved per-row
predictions, then summarizes bridge_sign causal transport across seeds.  Optionally
reads frozen-affine saved-output analyses and surfaces scalar sufficiency.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    if not path.exists():
        return rows
    with path.open(encoding="utf-8") as f:
        for line in f:
            s=line.strip()
            if s:
                rows.append(json.loads(s))
    return rows


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def mean(xs: Sequence[float]) -> float | None:
    return float(sum(xs)/len(xs)) if xs else None


def sd(xs: Sequence[float]) -> float | None:
    if len(xs) <= 1:
        return 0.0 if len(xs) == 1 else None
    m = mean(xs)
    return float((sum((x-m)**2 for x in xs)/(len(xs)-1))**0.5)


def sign(x: float, eps: float=1e-8) -> int:
    if x > eps: return 1
    if x < -eps: return -1
    return 0


def state_de_map(run_dir: Path) -> Dict[str, Dict[str, Any]]:
    rows = load_jsonl(run_dir / "eval_state_predictions.jsonl")
    by = defaultdict(list)
    for r in rows:
        by[r.get("query_key")].append(r)
    out = {}
    for k, rs in by.items():
        if len(rs) != 2:
            continue
        rs = sorted(rs, key=lambda r: int(r.get("candidate_index", 0)))
        r0 = rs[0]
        de = float(r0.get("d_e", float(rs[0]["score"]) - float(rs[1]["score"])))
        out[k] = {
            "d_e": de,
            "suite": r0.get("suite"), "is_changed": bool(r0.get("is_changed")),
            "relation_family": r0.get("relation_family"), "relation": r0.get("relation"),
            "initial_pattern": r0.get("initial_pattern"), "is_direct_anchor": bool(r0.get("is_direct_anchor")),
        }
    return out


def comp_map(run_dir: Path) -> Dict[str, Dict[str, Any]]:
    return {r.get("row_id"): r for r in load_jsonl(run_dir / "eval_comparison_predictions.jsonl")}


def pair_run_dirs(root: Path) -> Dict[Tuple[str,int], Dict[int, Path]]:
    pairs: Dict[Tuple[str,int], Dict[int, Path]] = defaultdict(dict)
    for d in sorted(root.iterdir()):
        if not d.is_dir() or not (d/"result.json").exists():
            continue
        r = load_json(d/"result.json")
        if r.get("no_comparisons") or r.get("no_bridge_anchors"):
            continue
        pairs[(str(r.get("condition")), int(r.get("seed")))][int(r.get("bridge_sign"))] = d
    return pairs


def analyze_pair(plus_dir: Path, minus_dir: Path) -> Dict[str, Any]:
    rp = load_json(plus_dir/"result.json")
    rm = load_json(minus_dir/"result.json")
    ce_p = rp.get("central_eval", {})
    ce_m = rm.get("central_eval", {})
    sp, sm = state_de_map(plus_dir), state_de_map(minus_dir)
    common = set(sp) & set(sm)
    cats = {
        "direct_changed": lambda m: m["is_changed"] and m["relation_family"] == "direct_anchor" and m["suite"] == "paired_state_conservation",
        "graph_changed": lambda m: m["is_changed"] and m["relation_family"] == "graph_transfer" and m["suite"] == "paired_state_conservation",
        "graph_changed_same": lambda m: m["is_changed"] and m["relation_family"] == "graph_transfer" and m["suite"] == "paired_state_conservation" and m.get("initial_pattern") == "same",
        "unchanged": lambda m: (not m["is_changed"]) and m["suite"] == "paired_state_conservation",
    }
    state = {}
    for name, filt in cats.items():
        ks = [k for k in common if filt(sp[k])]
        if not ks:
            state[name] = {"n": 0}
            continue
        dp = [float(sp[k]["d_e"]) for k in ks]
        dm = [float(sm[k]["d_e"]) for k in ks]
        same = sum(1 for a,b in zip(dp,dm) if sign(a) == sign(b) and sign(a) != 0)
        opp = sum(1 for a,b in zip(dp,dm) if sign(a) == -sign(b) and sign(a) != 0 and sign(b) != 0)
        corr = float(np.corrcoef(dp, dm)[0,1]) if len(dp)>1 else None
        corrneg = float(np.corrcoef(dp, [-x for x in dm])[0,1]) if len(dp)>1 else None
        state[name] = {"n": len(ks), "same": same, "opposite": opp, "same_frac": same/len(ks), "opposite_frac": opp/len(ks), "corr": corr, "corr_neg": corrneg, "mean_plus": mean(dp), "mean_minus": mean(dm)}
    cp, cm = comp_map(plus_dir), comp_map(minus_dir)
    comps = {}
    for suite in ["heldheld_unseen_edge_closure", "mixed_held_seen_orientation"]:
        ids = [rid for rid in set(cp)&set(cm) if cp[rid].get("suite") == suite]
        if not ids:
            comps[suite] = {"n": 0}
            continue
        prod_p = [float(cp[i]["d_e1"])*float(cp[i]["d_e2"]) for i in ids]
        prod_m = [float(cm[i]["d_e1"])*float(cm[i]["d_e2"]) for i in ids]
        same_prod = sum(1 for a,b in zip(prod_p,prod_m) if sign(a)==sign(b) and sign(a)!=0)
        opp_prod = sum(1 for a,b in zip(prod_p,prod_m) if sign(a)==-sign(b) and sign(a)!=0 and sign(b)!=0)
        comps[suite] = {"n": len(ids), "product_same_frac": same_prod/len(ids), "product_opp_frac": opp_prod/len(ids), "acc_plus": mean([float(cp[i]["correct"]) for i in ids]), "acc_minus": mean([float(cm[i]["correct"]) for i in ids]), "margin_plus": mean([float(cp[i]["signed_margin"]) for i in ids]), "margin_minus": mean([float(cm[i]["signed_margin"]) for i in ids])}
    return {
        "condition": rp.get("condition"), "seed": rp.get("seed"),
        "train_state_plus": rp.get("final_train_metrics",{}).get("train_state_acc"),
        "train_state_minus": rm.get("final_train_metrics",{}).get("train_state_acc"),
        "train_cmp_plus": rp.get("final_train_metrics",{}).get("train_cmp_acc"),
        "train_cmp_minus": rm.get("final_train_metrics",{}).get("train_cmp_acc"),
        "direct_same_plus": ce_p.get("direct_same"), "direct_same_minus": ce_m.get("direct_same"),
        "graph_same_plus": ce_p.get("graph_same"), "graph_same_minus": ce_m.get("graph_same"),
        "pair_both_plus": ce_p.get("pair_both_graph_same"), "pair_both_minus": ce_m.get("pair_both_graph_same"),
        "unchanged_plus": ce_p.get("unchanged"), "unchanged_minus": ce_m.get("unchanged"),
        "mixed_acc_plus": ce_p.get("mixed_held_seen_orientation_acc"), "mixed_acc_minus": ce_m.get("mixed_held_seen_orientation_acc"),
        "mixed_margin_plus": ce_p.get("mixed_held_seen_orientation_signed_margin"), "mixed_margin_minus": ce_m.get("mixed_held_seen_orientation_signed_margin"),
        "hh_closure_plus": ce_p.get("heldheld_unseen_edge_closure_acc"), "hh_closure_minus": ce_m.get("heldheld_unseen_edge_closure_acc"),
        "graph_margin_plus": ce_p.get("graph_same_margin"), "graph_margin_minus": ce_m.get("graph_same_margin"),
        "state_de_sign": state, "comparison_product": comps,
    }


def load_affine_summaries(paths: Sequence[Path]) -> List[Dict[str, Any]]:
    outs=[]
    for p in paths:
        if p.is_dir():
            cand = p/"frozen_affine_saved_outputs.json"
            if cand.exists():
                p=cand
        if p.exists():
            outs.append({"path": str(p), "payload": load_json(p)})
    return outs


def summarize_by_condition(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    by = defaultdict(list)
    for r in records:
        by[r["condition"]].append(r)
    out = {}
    numeric_keys = ["train_state_plus","train_state_minus","train_cmp_plus","train_cmp_minus","direct_same_plus","direct_same_minus","graph_same_plus","graph_same_minus","pair_both_plus","pair_both_minus","unchanged_plus","unchanged_minus","mixed_acc_plus","mixed_acc_minus","mixed_margin_plus","mixed_margin_minus","hh_closure_plus","hh_closure_minus","graph_margin_plus","graph_margin_minus"]
    for cond, rs in by.items():
        out[cond] = {"n": len(rs), "seeds": [r["seed"] for r in rs]}
        for k in numeric_keys:
            vals = [r.get(k) for r in rs if isinstance(r.get(k), (int,float)) and not (isinstance(r.get(k), float) and math.isnan(r.get(k)))]
            out[cond][k] = {"mean": mean([float(v) for v in vals]), "sd": sd([float(v) for v in vals]), "values": vals}
        for cat in ["direct_changed","graph_changed","graph_changed_same","unchanged"]:
            vals = [r["state_de_sign"].get(cat,{}).get("opposite_frac") for r in rs if r["state_de_sign"].get(cat,{}).get("opposite_frac") is not None]
            out[cond][f"{cat}_opposite_frac"] = {"mean": mean([float(v) for v in vals]), "sd": sd([float(v) for v in vals]), "values": vals}
        for suite in ["heldheld_unseen_edge_closure", "mixed_held_seen_orientation"]:
            vals = [r["comparison_product"].get(suite,{}).get("product_same_frac") for r in rs if r["comparison_product"].get(suite,{}).get("product_same_frac") is not None]
            out[cond][f"{suite}_product_same_frac"] = {"mean": mean([float(v) for v in vals]), "sd": sd([float(v) for v in vals]), "values": vals}
    return out


def fmt(x: Any) -> str:
    if x is None: return "NA"
    if isinstance(x, float): return f"{x:.3f}"
    return str(x)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--primary-roots", nargs="+", type=Path, required=True)
    ap.add_argument("--affine-jsons", nargs="*", type=Path, default=[])
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    records=[]
    for root in args.primary_roots:
        for (cond, seed), bsdirs in sorted(pair_run_dirs(root).items()):
            if 1 in bsdirs and -1 in bsdirs:
                records.append(analyze_pair(bsdirs[1], bsdirs[-1]))
    payload={"n_pairs": len(records), "pairs": records, "by_condition": summarize_by_condition(records), "affine_summaries": load_affine_summaries(args.affine_jsons)}
    write_json(args.out/"multiseed_gauge_and_affine_summary.json", payload)

    lines=["# research multi-seed gauge and affine synthesis", "", "## Per-pair causal gauge readout", "", "| cond | seed | train state +/- | train cmp +/- | graph same +/- | graph oppfrac | mixed acc +/- | mixed margin +/- | hh closure +/- | unchanged +/- |", "|---|---:|---|---|---|---:|---|---|---|---|"]
    for r in sorted(records, key=lambda x:(x['condition'], x['seed'])):
        lines.append("| {cond} | {seed} | {ts} | {tc} | {gs} | {opp} | {ma} | {mm} | {hh} | {un} |".format(
            cond=r['condition'], seed=r['seed'],
            ts=f"{fmt(r['train_state_plus'])}/{fmt(r['train_state_minus'])}",
            tc=f"{fmt(r['train_cmp_plus'])}/{fmt(r['train_cmp_minus'])}",
            gs=f"{fmt(r['graph_same_plus'])}/{fmt(r['graph_same_minus'])}",
            opp=fmt(r['state_de_sign'].get('graph_changed',{}).get('opposite_frac')),
            ma=f"{fmt(r['mixed_acc_plus'])}/{fmt(r['mixed_acc_minus'])}",
            mm=f"{fmt(r['mixed_margin_plus'])}/{fmt(r['mixed_margin_minus'])}",
            hh=f"{fmt(r['hh_closure_plus'])}/{fmt(r['hh_closure_minus'])}",
            un=f"{fmt(r['unchanged_plus'])}/{fmt(r['unchanged_minus'])}",
        ))
    lines.extend(["", "## By-condition means", ""])
    for cond, b in sorted(payload['by_condition'].items()):
        lines.append(f"### {cond} seeds={b['seeds']}")
        for k in ["graph_same_plus","graph_same_minus","pair_both_plus","pair_both_minus","mixed_acc_plus","mixed_acc_minus","hh_closure_plus","hh_closure_minus","graph_changed_opposite_frac","unchanged_opposite_frac","mixed_held_seen_orientation_product_same_frac","heldheld_unseen_edge_closure_product_same_frac"]:
            v=b.get(k,{})
            lines.append(f"- {k}: mean={fmt(v.get('mean'))} sd={fmt(v.get('sd'))} values={v.get('values')}")
        lines.append("")
    if payload['affine_summaries']:
        lines.extend(["## Frozen-affine scalar evidence", ""])
        for item in payload['affine_summaries']:
            lines.append(f"- Included affine summary JSON: `{item['path']}` with n_runs={item['payload'].get('n_runs')}")
            for run in item['payload'].get('runs', []):
                best = [f for f in run.get('fits', []) if f.get('kind')=='least_squares' and f.get('arm_sign')==1]
                if best:
                    ev=best[0]['eval']
                    lines.append(f"  - {Path(run['run_dir']).name}: train_cmp={fmt(run.get('train_cmp_acc'))}, hh_closure={fmt(run.get('heldheld_closure_acc'))}, affine graph_same={fmt(ev['graph_psc_same'].get('canonical_acc'))}, graph_all={fmt(ev['graph_psc_all'].get('canonical_acc'))}")
        lines.append("")
    lines.append("## Scientific reading")
    lines.append("Tied/shared_trunk stability across seeds supports causal gauge transport by shared representation. Untied stability near partial/local behavior supports that fitting local state anchors and comparisons separately is insufficient. Frozen-affine success should be read as scalar representational sufficiency only: it shows a one-dimensional coordinate can carry the sign once calibrated, not that the unanchored model learned to use it for state updating during training.")
    (args.out/"multiseed_gauge_and_affine_summary.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    print(json.dumps({"status":"MULTI_SEED_GAUGE_AFFINE_SUMMARY_COMPLETE","n_pairs":len(records),"summary":str(args.out/"multiseed_gauge_and_affine_summary.md"),"json":str(args.out/"multiseed_gauge_and_affine_summary.json")}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
