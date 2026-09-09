#!/usr/bin/env python3
"""Analyze paired research topology probe runs.

The script reads completed +1/-1 run directories, computes per-relation state
accuracy and row-paired sign reversal, compares hashes, and writes a compact
research note. It is deliberately separate from the training script so partially
completed cells do not masquerade as results.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, sort_keys=True) + "\n")


def read_json(path: Path):
    return json.loads(path.read_text())


def read_jsonl(path: Path) -> List[dict]:
    rows = []
    with path.open() as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def unique_state_queries(rows: Iterable[dict]) -> List[dict]:
    by = {}
    for r in rows:
        key = r.get("row_id") or r.get("query_key")
        # d_e and query-level correctness are duplicated across the two candidates.
        by[key] = {
            "key": key,
            "suite": r.get("suite"),
            "relation": r.get("relation"),
            "relation_family": r.get("relation_family"),
            "initial_pattern": r.get("initial_pattern"),
            "is_changed": bool(r.get("is_changed")),
            "is_direct_anchor": bool(r.get("is_direct_anchor")),
            "correct": int(r.get("correct", 0)),
            "d_e": float(r.get("d_e", 0.0)),
            "object": r.get("object"),
            "names": tuple(r.get("names", [])),
        }
    return list(by.values())


def mean(xs):
    xs = list(xs)
    return float(sum(xs) / len(xs)) if xs else math.nan


def per_relation_metrics(qs: List[dict]) -> Dict[str, dict]:
    out = {}
    for rel in sorted(set(q["relation"] for q in qs)):
        rel_rows = [q for q in qs if q["relation"] == rel and q["is_changed"]]
        if not rel_rows:
            continue
        for subset_name, filt in {
            "all_changed": lambda q: True,
            "paired_same": lambda q: q.get("suite") == "paired_state_conservation" and q.get("initial_pattern") == "same",
            "cross_template": lambda q: q.get("suite") == "cross_template_state_readout",
        }.items():
            xs = [q for q in rel_rows if filt(q)]
            if xs:
                out.setdefault(rel, {})[subset_name] = {
                    "n": len(xs),
                    "acc": mean(q["correct"] for q in xs),
                    "mean_d_e": mean(q["d_e"] for q in xs),
                    "mean_abs_d_e": mean(abs(q["d_e"]) for q in xs),
                }
    return out


def pair_key(q: dict) -> Tuple:
    key = str(q.get("key", ""))
    # row_id/query_key is stable across bridge signs in this harness.
    return (key, q.get("relation"), q.get("suite"), q.get("initial_pattern"), q.get("object"), tuple(q.get("names", ())))


def sign_pair_metrics(qp: List[dict], qm: List[dict]) -> Dict[str, dict]:
    ip = {pair_key(q): q for q in qp if q["is_changed"]}
    im = {pair_key(q): q for q in qm if q["is_changed"]}
    out = {}
    for rel in sorted(set([q["relation"] for q in qp + qm])):
        keys = [k for k in ip if k in im and k[1] == rel]
        if not keys:
            continue
        dp = np.array([ip[k]["d_e"] for k in keys], dtype=float)
        dm = np.array([im[k]["d_e"] for k in keys], dtype=float)
        same = int(np.sum(dp * dm > 0))
        opp = int(np.sum(dp * dm < 0))
        zeros = int(np.sum(dp * dm == 0))
        acc_p = mean(ip[k]["correct"] for k in keys)
        acc_m = mean(im[k]["correct"] for k in keys)
        entry = {
            "matched": len(keys),
            "same_sign": same,
            "opposite_sign": opp,
            "zero_product": zeros,
            "same_sign_frac": same / len(keys),
            "opposite_sign_frac": opp / len(keys),
            "acc_plus": acc_p,
            "acc_minus": acc_m,
            "mean_d_e_plus": float(np.mean(dp)),
            "mean_d_e_minus": float(np.mean(dm)),
            "mean_abs_d_e_plus": float(np.mean(np.abs(dp))),
            "mean_abs_d_e_minus": float(np.mean(np.abs(dm))),
        }
        if len(keys) > 1 and float(np.std(dp)) > 0 and float(np.std(dm)) > 0:
            entry["corr_plus_minus"] = float(np.corrcoef(dp, dm)[0, 1])
            entry["anti_corr_plus_minus"] = float(np.corrcoef(dp, -dm)[0, 1])
        out[rel] = entry
    return out


def comp_edge_metrics(rows: List[dict]) -> Dict[str, dict]:
    by = defaultdict(list)
    for r in rows:
        by[f"{r.get('relation1')}->{r.get('relation2')}"] .append(r)
    out = {}
    for edge, xs in sorted(by.items()):
        margins = [float(r.get("signed_margin", 0.0)) for r in xs]
        out[edge] = {
            "n": len(xs),
            "acc": mean(int(r.get("correct", 0)) for r in xs),
            "mean_signed_margin": mean(margins),
        }
    return out


def collect_pair(label: str, plus_dir: Path, minus_dir: Path) -> dict:
    rp = read_json(plus_dir / "result.json")
    rm = read_json(minus_dir / "result.json")
    sp = unique_state_queries(read_jsonl(plus_dir / "state_predictions.jsonl"))
    sm = unique_state_queries(read_jsonl(minus_dir / "state_predictions.jsonl"))
    cp = read_jsonl(plus_dir / "comparison_predictions_original_eval.jsonl")
    cm = read_jsonl(minus_dir / "comparison_predictions_original_eval.jsonl")
    return {
        "label": label,
        "plus_dir": str(plus_dir),
        "minus_dir": str(minus_dir),
        "plus_central": rp.get("central_eval", {}),
        "minus_central": rm.get("central_eval", {}),
        "plus_train_last": rp.get("train_info", {}).get("history", [{}])[-1] if rp.get("train_info", {}).get("history") else {},
        "minus_train_last": rm.get("train_info", {}).get("history", [{}])[-1] if rm.get("train_info", {}).get("history") else {},
        "hash_comparison": {
            "init_event_trunk_equal": rp.get("hashes", {}).get("init", {}).get("event_trunk") == rm.get("hashes", {}).get("init", {}).get("event_trunk"),
            "pretrain_event_trunk_equal": rp.get("hashes", {}).get("after_pretrain", {}).get("event_trunk") == rm.get("hashes", {}).get("after_pretrain", {}).get("event_trunk"),
            "final_event_trunk_equal": rp.get("hashes", {}).get("final", {}).get("event_trunk") == rm.get("hashes", {}).get("final", {}).get("event_trunk"),
            "final_event_state_head_equal": rp.get("hashes", {}).get("final", {}).get("event_state_head") == rm.get("hashes", {}).get("final", {}).get("event_state_head"),
            "plus_final_hashes": rp.get("hashes", {}).get("final", {}),
            "minus_final_hashes": rm.get("hashes", {}).get("final", {}),
        },
        "plus_per_relation": per_relation_metrics(sp),
        "minus_per_relation": per_relation_metrics(sm),
        "sign_pair_by_relation": sign_pair_metrics(sp, sm),
        "plus_eval_comparison_by_edge": comp_edge_metrics(cp),
        "minus_eval_comparison_by_edge": comp_edge_metrics(cm),
        "plus_manifest": rp.get("manifest", {}),
        "minus_manifest": rm.get("manifest", {}),
    }


def md_for_pair(pair: dict) -> str:
    lines = [f"## {pair['label']}", ""]
    lines.append("### Central metrics")
    lines.append("")
    lines.append("| bridge sign | train_state | train_cmp_modified | train_cmp_original | direct_same | graph_same | pair_both_graph_same | hh_closure | mixed_acc | unchanged |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for sign, c, t in [
        ("+1", pair["plus_central"], pair["plus_train_last"]),
        ("-1", pair["minus_central"], pair["minus_train_last"]),
    ]:
        lines.append(f"| {sign} | {t.get('train_state', math.nan):.3f} | {t.get('train_cmp_modified', math.nan):.3f} | {t.get('train_cmp_original_labels', math.nan):.3f} | {c.get('direct_same', math.nan):.3f} | {c.get('graph_same', math.nan):.3f} | {c.get('pair_both_graph_same', math.nan):.3f} | {c.get('hh_closure', math.nan):.3f} | {c.get('mixed_acc', math.nan):.3f} | {c.get('unchanged', math.nan):.3f} |")
    lines.append("")
    lines.append("### Hash comparison")
    lines.append("")
    for k, v in pair["hash_comparison"].items():
        if not isinstance(v, dict):
            lines.append(f"- {k}: `{v}`")
    lines.append("")
    lines.append("### Per-relation paired sign analysis")
    lines.append("")
    lines.append("| relation | matched | acc + | acc - | opposite frac | same frac | mean d_e + | mean d_e - | abs d_e + | abs d_e - | anti-corr |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for rel, r in pair["sign_pair_by_relation"].items():
        lines.append(f"| {rel} | {r['matched']} | {r['acc_plus']:.3f} | {r['acc_minus']:.3f} | {r['opposite_sign_frac']:.3f} | {r['same_sign_frac']:.3f} | {r['mean_d_e_plus']:.3f} | {r['mean_d_e_minus']:.3f} | {r['mean_abs_d_e_plus']:.3f} | {r['mean_abs_d_e_minus']:.3f} | {r.get('anti_corr_plus_minus', math.nan):.3f} |")
    lines.append("")
    lines.append("### Per-relation changed-state accuracy by sign")
    lines.append("")
    for sign_name, per in [("+1", pair["plus_per_relation"]), ("-1", pair["minus_per_relation"])]:
        lines.append(f"#### bridge sign {sign_name}")
        lines.append("")
        lines.append("| relation | subset | n | acc | mean d_e | mean abs d_e |")
        lines.append("|---|---|---:|---:|---:|---:|")
        for rel, subsets in per.items():
            for subset, r in subsets.items():
                lines.append(f"| {rel} | {subset} | {r['n']} | {r['acc']:.3f} | {r['mean_d_e']:.3f} | {r['mean_abs_d_e']:.3f} |")
        lines.append("")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", action="append", nargs=3, metavar=("LABEL", "PLUS_DIR", "MINUS_DIR"), required=True,
                    help="Analyze one sign pair: label plus_run_dir minus_run_dir")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    pairs = []
    for label, plus, minus in args.pair:
        pairs.append(collect_pair(label, Path(plus), Path(minus)))

    args.out.mkdir(parents=True, exist_ok=True)
    write_json(args.out / "topology_pair_analysis.json", {"pairs": pairs})
    md = ["# research topology-pair analysis", ""]
    for pair in pairs:
        md.append(md_for_pair(pair))
        md.append("")
    (args.out / "topology_pair_analysis.md").write_text("\n".join(md) + "\n")
    print(json.dumps({
        "status": "TOPOLOGY_PAIR_ANALYSIS_DONE",
        "out_json": str(args.out / "topology_pair_analysis.json"),
        "out_md": str(args.out / "topology_pair_analysis.md"),
        "labels": [p["label"] for p in pairs],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
