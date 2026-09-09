#!/usr/bin/env python3
"""research: error structure for raw-name query-attention saved predictions."""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Sequence


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists(): return []
    with path.open(encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def mean(xs: Sequence[float]):
    return sum(xs)/len(xs) if xs else None


def choice_rows(state_rows: List[Dict[str, Any]], bridge_sign: int) -> List[Dict[str, Any]]:
    by = defaultdict(list)
    for r in state_rows:
        by[r.get("query_key")].append(r)
    out = []
    for k, rs in by.items():
        if len(rs) != 2: continue
        rs = sorted(rs, key=lambda r: r.get("candidate_index", 0))
        r0, r1 = rs
        d = float(r0.get("d_e", 0.0))
        # True canonical target for candidate 0; bridge_sign flips only direct anchor state targets.
        lab0 = bool(r0.get("label_true"))
        if bridge_sign == -1 and bool(r0.get("is_direct_anchor")):
            lab0 = not lab0
        pred0 = d > 0
        out.append({**r0, "choice_correct": pred0 == lab0, "choice_margin": abs(d) * (1 if pred0 == lab0 else -1)})
    return out


def summarize_rows(rows: List[Dict[str, Any]], keys: List[str], value_name: str = "choice_correct") -> Dict[str, Any]:
    out = {}
    for key in keys:
        groups = defaultdict(list)
        for r in rows:
            groups[str(r.get(key))].append(r)
        out[key] = {g: {"n": len(v), "acc": mean([float(x.get(value_name, False)) for x in v]),
                         "margin": mean([float(x.get("choice_margin", 0.0)) for x in v]) if "choice_margin" in v[0] else None}
                    for g, v in sorted(groups.items())}
    return out


def summarize_comps(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    out = {}
    for key in ["suite", "relation1", "relation2"]:
        groups = defaultdict(list)
        for r in rows:
            groups[str(r.get(key))].append(r)
        out[key] = {g: {"n": len(v), "acc": mean([float(x.get("correct", False)) for x in v]),
                         "signed_margin": mean([float(x.get("signed_margin", 0.0)) for x in v])}
                    for g, v in sorted(groups.items())}
    return out


def run_dirs(base: Path):
    return sorted([d for d in base.iterdir() if d.is_dir() and (d/"result.json").exists()]) if base.exists() else []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--attention-dir", type=Path, default=Path("experiments/archive/representation_and_objectives/data/raw_name_attention_primary"))
    ap.add_argument("--out", type=Path, default=Path("experiments/archive/representation_and_objectives/data/attention_error_structure"))
    args = ap.parse_args()

    report = {"runs": {}, "train_fit_table": []}
    for d in run_dirs(args.attention_dir):
        rec = load_json(d/"result.json")
        cond, bs, seed = rec.get("condition"), rec.get("bridge_sign"), rec.get("seed")
        tag = f"{cond}|bs{bs:+d}|seed{seed}"
        states = load_jsonl(d/"state_predictions.jsonl") or load_jsonl(d/"eval_state_predictions.jsonl")
        comps = load_jsonl(d/"comp_predictions.jsonl") or load_jsonl(d/"eval_comparison_predictions.jsonl")
        choices = choice_rows(states, int(bs))
        changed = [r for r in choices if bool(r.get("is_changed"))]
        unchanged = [r for r in choices if not bool(r.get("is_changed"))]
        graph_same = [r for r in changed if r.get("relation_family") == "graph_transfer" and r.get("initial_pattern") == "same"]
        direct_same = [r for r in changed if r.get("relation_family") == "direct_anchor" and r.get("initial_pattern") == "same"]
        # name-pair signatures: useful for seeing memorized/easy held pairs.
        by_pair = defaultdict(list)
        for r in choices:
            by_pair[tuple(r.get("names", []))].append(r)
        pair_stats = {"/".join(k): {"n": len(v), "acc": mean([float(x["choice_correct"]) for x in v])}
                      for k, v in sorted(by_pair.items())}
        report["runs"][tag] = {
            "dir": str(d),
            "central_eval": rec.get("central_eval"),
            "state_choice_counts": {"all": len(choices), "changed": len(changed), "unchanged": len(unchanged),
                                      "graph_same": len(graph_same), "direct_same": len(direct_same)},
            "state_acc_by": summarize_rows(choices, ["suite", "relation_family", "relation", "initial_pattern", "static_slot", "object"]),
            "changed_acc_by": summarize_rows(changed, ["suite", "relation_family", "relation", "initial_pattern", "object"]),
            "unchanged_acc_by": summarize_rows(unchanged, ["suite", "relation_family", "relation", "initial_pattern", "object"]),
            "comparison_acc_by": summarize_comps(comps),
            "name_pair_acc": pair_stats,
        }
        c = rec.get("central_eval", {})
        report["train_fit_table"].append({"tag": tag, "train_state": c.get("train_state_acc"), "train_cmp": c.get("train_cmp_acc"),
                                           "eval_changed_graph_same": c.get("graph_same"), "eval_unchanged": c.get("unchanged"),
                                           "hh_closure": c.get("hh_closure"), "mixed_acc": c.get("mixed_acc")})

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out/"attention_error_structure.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = ["# research raw-name attention error structure\n\n",
             "## Train/eval overview\n\n",
             "| run | train_state | train_cmp | graph_same | unchanged | hh_closure | mixed_acc |\n",
             "|---|---:|---:|---:|---:|---:|---:|\n"]
    for r in report["train_fit_table"]:
        def fmt(x): return f"{x:.3f}" if isinstance(x, (float,int)) else str(x)
        lines.append(f"| {r['tag']} | {fmt(r['train_state'])} | {fmt(r['train_cmp'])} | {fmt(r['eval_changed_graph_same'])} | {fmt(r['eval_unchanged'])} | {fmt(r['hh_closure'])} | {fmt(r['mixed_acc'])} |\n")
    lines.append("\n## Main interpretation\n\n")
    lines.append("Held-name evaluation is the hard part of the raw-name pilot. If train fit is high while changed, unchanged, and held-held eval are far from 1.0, the failure is not only gauge transport; it includes non-variable candidate/name generalization under the query-attention architecture.\n")
    lines.append("\n## Per-run coarse error slices\n\n")
    for tag, d in report["runs"].items():
        lines.append(f"### {tag}\n")
        for slice_name in ["suite", "relation_family", "initial_pattern"]:
            lines.append(f"- state by {slice_name}: {d['state_acc_by'].get(slice_name)}\n")
        lines.append(f"- comparison by suite: {d['comparison_acc_by'].get('suite')}\n\n")
    (args.out/"attention_error_structure.md").write_text("".join(lines), encoding="utf-8")
    print(json.dumps({"status":"ATTENTION_ERROR_STRUCTURE_COMPLETE", "json":str(args.out/"attention_error_structure.json"), "summary":str(args.out/"attention_error_structure.md"), "n_runs":len(report["runs"])}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
