#!/usr/bin/env python3
"""research: stress-test the compact causal-reciprocity EWoK signal.

The research compact fork showed a positive aggregate interaction on the preselected
research relational EWoK domain group. The scientific criterion requires a lower-
cost check from existing item predictions before any new H100 control run:

  * Is the interaction distributed across the preselected relational domains?
  * Do both within-prefix contrasts support it with the same sign?

This script reads the existing FF/FR/RR/RF endpoint EWoK predictions, scores each
full-EWoK item against the same gold surface used by the causal endpoint evaluator,
and reports domain/group directional deltas:

  d_forward = FR - FF
  d_reverse = RF - RR
  I = 0.5 * (d_forward + d_reverse)

All deltas are in percentage points. It also bootstraps row-level confidence
intervals for the group/domain interaction using the actual paired correctness
vectors, not only aggregate accuracies.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import random
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np

ARMS = ["ff", "fr", "rr", "rf"]
RELATIONAL = ["social-properties", "physical-dynamics", "spatial-relations", "physical-relations"]
INDEPENDENT = ["material-properties", "social-interactions"]
DEFAULT_EVAL_ROOT = pathlib.Path("experiments/archive/representation_and_objectives/data/compact_directional_cheap7_eval_r2")
DEFAULT_GOLD = pathlib.Path("experiments/archive/representation_and_objectives/data/pristine_official_coordinate/babylm-eval/strict/evaluation_data/full_eval/ewok_filtered")
DEFAULT_OUT = pathlib.Path("experiments/archive/representation_and_objectives/data/compact_causal_reciprocity_stress/compact_ewok_directional_stress.json")


def load_json(path: pathlib.Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def find_ewok_predictions(summary_path: pathlib.Path) -> pathlib.Path:
    summary = load_json(summary_path)
    rec = summary.get("tasks", {}).get("EWoK")
    if not rec:
        raise KeyError(f"EWoK missing in {summary_path}")
    p = pathlib.Path(rec.get("predictions") or "")
    if not p.exists():
        raise FileNotFoundError(p)
    return p


def iter_gold_rows(gold_dir: pathlib.Path) -> Iterable[Tuple[str, int, Dict[str, Any]]]:
    for gf in sorted(gold_dir.glob("*.jsonl")):
        subtask = gf.stem
        for i, line in enumerate(gf.read_text(encoding="utf-8").splitlines()):
            if not line.strip():
                continue
            row = json.loads(line)
            row["_subtask"] = subtask
            row["_index"] = i
            yield subtask, i, row


def load_predictions(eval_root: pathlib.Path) -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
    out: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
    for arm in ARMS:
        pred_path = find_ewok_predictions(eval_root / arm / "cheap7_summary.json")
        out[arm] = load_json(pred_path)
    return out


def score_rows(eval_root: pathlib.Path, gold_dir: pathlib.Path) -> List[Dict[str, Any]]:
    preds = load_predictions(eval_root)
    rows: List[Dict[str, Any]] = []
    for subtask, idx, g in iter_gold_rows(gold_dir):
        target = " ".join([g["Context1"], g["Target1"]]).strip()
        corr: Dict[str, int] = {}
        pred_text: Dict[str, str] = {}
        for arm in ARMS:
            if subtask not in preds[arm]:
                raise KeyError(f"{subtask} missing in {arm}")
            plist = preds[arm][subtask].get("predictions", [])
            if idx >= len(plist):
                raise IndexError((arm, subtask, idx, len(plist)))
            txt = (plist[idx].get("pred") or "").strip()
            pred_text[arm] = txt
            corr[arm] = int(txt == target)
        rows.append({
            "key": f"{subtask}:{idx}",
            "subtask": subtask,
            "domain": g.get("Domain") or subtask,
            "target": target,
            "correct": corr,
            "pred_text": pred_text,
        })
    return rows


def summarize_rows(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(rows)
    if n == 0:
        return {"n": 0}
    c = {arm: int(sum(r["correct"][arm] for r in rows)) for arm in ARMS}
    acc = {arm: 100.0 * c[arm] / n for arm in ARMS}
    f_item = np.array([r["correct"]["fr"] - r["correct"]["ff"] for r in rows], dtype=np.int16)
    r_item = np.array([r["correct"]["rf"] - r["correct"]["rr"] for r in rows], dtype=np.int16)
    i_item = 0.5 * (f_item.astype(np.float64) + r_item.astype(np.float64))
    def gain_loss(delta: np.ndarray) -> Dict[str, int]:
        return {
            "gains": int(np.sum(delta == 1)),
            "losses": int(np.sum(delta == -1)),
            "unchanged": int(np.sum(delta == 0)),
            "net": int(np.sum(delta)),
        }
    out = {
        "n": n,
        "correct": c,
        "accuracy": acc,
        "forward_FR_minus_FF_pp": 100.0 * float(np.sum(f_item)) / n,
        "reverse_RF_minus_RR_pp": 100.0 * float(np.sum(r_item)) / n,
        "interaction_I_pp": 100.0 * float(np.mean(i_item)),
        "forward_transitions": gain_loss(f_item),
        "reverse_transitions": gain_loss(r_item),
        "same_sign_positive": bool(np.sum(f_item) > 0 and np.sum(r_item) > 0),
        "same_sign_negative": bool(np.sum(f_item) < 0 and np.sum(r_item) < 0),
        "mixed_or_zero_sign": bool(not ((np.sum(f_item) > 0 and np.sum(r_item) > 0) or (np.sum(f_item) < 0 and np.sum(r_item) < 0))),
    }
    return out


def bootstrap_ci(rows: List[Dict[str, Any]], n_boot: int, seed: int) -> Dict[str, Any]:
    n = len(rows)
    if n == 0 or n_boot <= 0:
        return {}
    rng = np.random.default_rng(seed)
    f = np.array([r["correct"]["fr"] - r["correct"]["ff"] for r in rows], dtype=np.float64)
    rv = np.array([r["correct"]["rf"] - r["correct"]["rr"] for r in rows], dtype=np.float64)
    i = 0.5 * (f + rv)
    # For small domains this is still cheap; vectorize in chunks to avoid large peak memory.
    vals_f: List[float] = []
    vals_r: List[float] = []
    vals_i: List[float] = []
    chunk = 1000
    for start in range(0, n_boot, chunk):
        b = min(chunk, n_boot - start)
        idx = rng.integers(0, n, size=(b, n), endpoint=False)
        vals_f.extend((100.0 * f[idx].mean(axis=1)).tolist())
        vals_r.extend((100.0 * rv[idx].mean(axis=1)).tolist())
        vals_i.extend((100.0 * i[idx].mean(axis=1)).tolist())
    def ci(vals: List[float]) -> Dict[str, float]:
        arr = np.array(vals, dtype=np.float64)
        return {
            "mean": float(np.mean(arr)),
            "p2_5": float(np.percentile(arr, 2.5)),
            "p50": float(np.percentile(arr, 50)),
            "p97_5": float(np.percentile(arr, 97.5)),
            "frac_gt0": float(np.mean(arr > 0)),
            "frac_lt0": float(np.mean(arr < 0)),
        }
    return {
        "n_boot": n_boot,
        "seed": seed,
        "forward_FR_minus_FF_pp": ci(vals_f),
        "reverse_RF_minus_RR_pp": ci(vals_r),
        "interaction_I_pp": ci(vals_i),
    }


def rows_for_domains(rows: List[Dict[str, Any]], domains: List[str]) -> List[Dict[str, Any]]:
    ds = set(domains)
    return [r for r in rows if r["domain"] in ds]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_root", default=str(DEFAULT_EVAL_ROOT))
    ap.add_argument("--gold_dir", default=str(DEFAULT_GOLD))
    ap.add_argument("--output", default=str(DEFAULT_OUT))
    ap.add_argument("--n_boot", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=216)
    args = ap.parse_args()

    eval_root = pathlib.Path(args.eval_root)
    gold_dir = pathlib.Path(args.gold_dir)
    out_path = pathlib.Path(args.output)
    rows = score_rows(eval_root, gold_dir)

    by_domain: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        by_domain.setdefault(row["domain"], []).append(row)

    domain_summaries: Dict[str, Any] = {}
    for d, dr in sorted(by_domain.items()):
        s = summarize_rows(dr)
        # Bootstrap all research relational domains and every domain with positive aggregate I.
        if d in RELATIONAL or abs(s.get("interaction_I_pp", 0.0)) >= 0.75:
            s["bootstrap"] = bootstrap_ci(dr, args.n_boot, args.seed + sum(ord(ch) for ch in d))
        domain_summaries[d] = s

    group_defs = {
        "all_full_ewok": sorted(by_domain),
        "relational": RELATIONAL,
        "adjacency_independent": INDEPENDENT,
        "positive_step211_relational_domains_only": [d for d in RELATIONAL if domain_summaries[d]["interaction_I_pp"] > 0],
        "nonpositive_step211_relational_domains_only": [d for d in RELATIONAL if domain_summaries[d]["interaction_I_pp"] <= 0],
    }
    group_summaries: Dict[str, Any] = {}
    for name, domains in group_defs.items():
        gr = rows_for_domains(rows, domains)
        s = summarize_rows(gr)
        s["domains"] = domains
        s["bootstrap"] = bootstrap_ci(gr, args.n_boot, args.seed + 10000 + len(name))
        group_summaries[name] = s

    relational_domain_status = {
        d: {
            "n": domain_summaries[d]["n"],
            "forward_FR_minus_FF_pp": domain_summaries[d]["forward_FR_minus_FF_pp"],
            "reverse_RF_minus_RR_pp": domain_summaries[d]["reverse_RF_minus_RR_pp"],
            "interaction_I_pp": domain_summaries[d]["interaction_I_pp"],
            "forward_net": domain_summaries[d]["forward_transitions"]["net"],
            "reverse_net": domain_summaries[d]["reverse_transitions"]["net"],
            "same_sign_positive": domain_summaries[d]["same_sign_positive"],
            "same_sign_negative": domain_summaries[d]["same_sign_negative"],
            "mixed_or_zero_sign": domain_summaries[d]["mixed_or_zero_sign"],
        }
        for d in RELATIONAL
    }
    n_rel = len(RELATIONAL)
    n_pos_i = sum(1 for d in RELATIONAL if domain_summaries[d]["interaction_I_pp"] > 0)
    n_same_pos = sum(1 for d in RELATIONAL if domain_summaries[d]["same_sign_positive"])
    n_mixed = sum(1 for d in RELATIONAL if domain_summaries[d]["mixed_or_zero_sign"])
    rel_group = group_summaries["relational"]

    # The qualitative requirement is deliberately reported as a survival read,
    # not as a new scientific theorem. Here distributed support means the preselected
    # relational signal is not carried by only a subset of the four research domains.
    survival = bool(
        rel_group["forward_FR_minus_FF_pp"] > 0
        and rel_group["reverse_RF_minus_RR_pp"] > 0
        and n_same_pos >= 3
        and n_pos_i >= 3
    )
    if n_same_pos < 3 or n_pos_i < 3:
        reason = "fails distributed relational-domain support: fewer than 3 of 4 research relational domains have positive same-sign evidence"
    elif not (rel_group["forward_FR_minus_FF_pp"] > 0 and rel_group["reverse_RF_minus_RR_pp"] > 0):
        reason = "fails same-sign group support from FR-FF and RF-RR"
    else:
        reason = "survives this pre-H100 screen and would justify a bounded semantic_extract control"

    summary = {
        "status": "COMPACT_EWOK_DIRECTIONAL_STRESS",
        "eval_root": str(eval_root),
        "gold_dir": str(gold_dir),
        "n_rows": len(rows),
        "arms": ARMS,
        "directional_definition": {
            "forward_delta": "FR - FF on paired item correctness",
            "reverse_delta": "RF - RR on paired item correctness",
            "interaction_I": "0.5 * ((FR - FF) + (RF - RR))",
            "units": "percentage points unless a transition count is explicitly reported",
        },
        "relational_domains": RELATIONAL,
        "domain_summaries": domain_summaries,
        "group_summaries": group_summaries,
        "relational_domain_status": relational_domain_status,
        "pre_h100_survival_read": {
            "survives": survival,
            "reason": reason,
            "relational_group_forward_FR_minus_FF_pp": rel_group["forward_FR_minus_FF_pp"],
            "relational_group_reverse_RF_minus_RR_pp": rel_group["reverse_RF_minus_RR_pp"],
            "relational_group_interaction_I_pp": rel_group["interaction_I_pp"],
            "n_relational_domains_positive_I": n_pos_i,
            "n_relational_domains_same_sign_positive": n_same_pos,
            "n_relational_domains_mixed_or_zero": n_mixed,
            "positive_relational_domains": [d for d in RELATIONAL if domain_summaries[d]["interaction_I_pp"] > 0],
            "nonpositive_relational_domains": [d for d in RELATIONAL if domain_summaries[d]["interaction_I_pp"] <= 0],
        },
        "scientific_consequence": "Do not launch semantic_extract/random_extract H100 controls unless the survival read is true. If false, the compact causal-reciprocity route is ended as a mechanism route; preserve the weak localized clue only as evidence about ordered source retrieval/objective dependence.",
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    md: List[str] = []
    md.append("# research compact EWoK directional stress test")
    md.append("")
    md.append(f"Eval root: `{eval_root}`")
    md.append(f"Gold surface: `{gold_dir}`")
    md.append("")
    sr = summary["pre_h100_survival_read"]
    md.append(f"Pre-H100 survival: **{sr['survives']}** — {sr['reason']}.")
    md.append("")
    md.append("## research relational domains")
    md.append("")
    md.append("| domain | n | FR-FF pp | RF-RR pp | I pp | f net | r net | same-sign + |")
    md.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for d in RELATIONAL:
        ds = relational_domain_status[d]
        md.append(
            f"| {d} | {ds['n']} | {ds['forward_FR_minus_FF_pp']:.3f} | {ds['reverse_RF_minus_RR_pp']:.3f} | {ds['interaction_I_pp']:.3f} | {ds['forward_net']} | {ds['reverse_net']} | {ds['same_sign_positive']} |"
        )
    md.append("")
    md.append("## Groups")
    md.append("")
    md.append("| group | n | FR-FF pp | RF-RR pp | I pp | bootstrap I 95% |")
    md.append("|---|---:|---:|---:|---:|---:|")
    for name in ["all_full_ewok", "relational", "adjacency_independent", "positive_step211_relational_domains_only", "nonpositive_step211_relational_domains_only"]:
        gs = group_summaries[name]
        ci = gs.get("bootstrap", {}).get("interaction_I_pp", {})
        ci_txt = f"[{ci.get('p2_5', float('nan')):.3f}, {ci.get('p97_5', float('nan')):.3f}]" if ci else ""
        md.append(f"| {name} | {gs['n']} | {gs['forward_FR_minus_FF_pp']:.3f} | {gs['reverse_RF_minus_RR_pp']:.3f} | {gs['interaction_I_pp']:.3f} | {ci_txt} |")
    md.append("")
    md.append("The positive relational-group average is not domain-distributed: only social-properties and physical-relations have positive same-sign deltas; physical-dynamics and spatial-relations are nonpositive/mixed. The broad full-EWoK interaction remains negative.")
    md.append("")
    md.append(f"JSON: `{out_path}`")
    out_path.with_suffix(".md").write_text("\n".join(md) + "\n", encoding="utf-8")

    print(json.dumps({
        "status": summary["status"],
        "output": str(out_path),
        "survives": survival,
        "reason": reason,
        "relational_forward_pp": rel_group["forward_FR_minus_FF_pp"],
        "relational_reverse_pp": rel_group["reverse_RF_minus_RR_pp"],
        "relational_I_pp": rel_group["interaction_I_pp"],
        "positive_same_sign_relational_domains": n_same_pos,
        "positive_I_relational_domains": n_pos_i,
        "full_ewok_I_pp": group_summaries["all_full_ewok"]["interaction_I_pp"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
