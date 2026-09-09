#!/usr/bin/env python3
"""research detailed decomposition of the pair-binding learned pilot.

CPU-only analysis over saved logits and raw substrate metadata.  It separates
relation-level state behavior, train neutral/informative comparison fitting,
held-held edge closure, exact dyad membership, and row-paired aligned/inverted
margins.  No model loading or training.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

PROJECT = Path("experiments/archive/representation_and_objectives")
DATA_ROOT = PROJECT / "data/pair_binding_neutral_substrate"
DEFAULT_INPUTS = [
    PROJECT / "data/pair_binding_probe_connected_seed28700",
    PROJECT / "data/pair_binding_probe_rewired_seed28700",
]
DEFAULT_OUT = PROJECT / "data/pair_binding_decomposition"
DIRECT = {"h0_dax", "h2_norp"}
GRAPH = {"h1_mep", "h3_ziv"}
CONDITIONS = ["pair_connected", "pair_rewired"]
ARMS = ["aligned_state_bridge", "inverted_state_bridge"]

B_PAIRS = {"Mira::Omar", "Noel::Iris", "Lena::Pavel", "Rina::Tomas", "Nia::Felix", "Ava::Jonas", "Keira::Milo", "Sara::Theo"}
R_PAIRS = {"Mira::Iris", "Noel::Pavel", "Lena::Tomas", "Rina::Felix", "Nia::Jonas", "Ava::Milo", "Keira::Theo", "Sara::Omar"}


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    if not path.exists():
        return rows
    with path.open(encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                rows.append(json.loads(s))
    return rows


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def mean(xs: Sequence[float]) -> float | None:
    return float(sum(xs) / len(xs)) if xs else None


def std(xs: Sequence[float]) -> float | None:
    return float(np.std(np.asarray(xs, dtype=float))) if xs else None


def sem(xs: Sequence[float]) -> float | None:
    return float(np.std(np.asarray(xs, dtype=float)) / math.sqrt(len(xs))) if xs else None


def summarize(vals: Sequence[float]) -> Dict[str, Any]:
    return {"n": len(vals), "mean": mean(vals), "std": std(vals), "sem": sem(vals), "values": [float(v) for v in vals]}


def key_join(*xs: Any) -> str:
    return "|".join("None" if x is None else str(x) for x in xs)


def group(rows: Sequence[Dict[str, Any]], fields: Sequence[str]) -> Dict[str, List[Dict[str, Any]]]:
    d: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        d[key_join(*(r.get(f) for f in fields))].append(r)
    return dict(d)


def exact_pair(row: Dict[str, Any]) -> str | None:
    vals = row.get("arg_order1") if row.get("task") == "relation_comparison" else row.get("arg_order")
    if vals and len(vals) == 2:
        return f"{vals[0]}::{vals[1]}"
    return None


def pair_set(pk: str | None) -> str:
    if pk in B_PAIRS:
        return "B_bridge_pairs"
    if pk in R_PAIRS:
        return "R_rewired_pairs"
    if pk is None:
        return "none"
    return "novel_or_other"


def rel_family(rel: str | None) -> str:
    if rel in DIRECT:
        return "direct_anchor"
    if rel in GRAPH:
        return "graph_transfer"
    if rel is None:
        return "none"
    return "seen_or_other"


def raw_maps(data_root: Path) -> Dict[Tuple[str, str, str], Dict[str, Any]]:
    maps: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    for cond in CONDITIONS:
        cdir = data_root / cond
        common = load_jsonl(cdir / "common_seen_train.jsonl")
        for arm in ARMS + ["heldheld_only"]:
            rows = list(common) + load_jsonl(cdir / "arms" / arm / "train_supervised.jsonl")
            for r in rows:
                rid = r.get("row_id")
                if rid is not None:
                    maps[(cond, arm, str(rid))] = r
    eval_rows = []
    for p in (data_root / "eval").glob("*.jsonl"):
        eval_rows.extend(load_jsonl(p))
    for cond in CONDITIONS:
        for arm in ARMS + ["heldheld_only"]:
            for r in eval_rows:
                rid = r.get("row_id")
                if rid is not None:
                    maps[(cond, arm, str(rid))] = r
    return maps


def enrich(rows: Sequence[Dict[str, Any]], raw: Dict[Tuple[str, str, str], Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for r in rows:
        rr = dict(r)
        meta = raw.get((str(r.get("condition")), str(r.get("arm")), str(r.get("row_id"))), {})
        for k in ["neutral_control", "geom_same_final_owner_under_true", "neutral_flip_bit", "pair_set", "arg_order", "arg_order1", "arg_order2", "surface_order", "surface_order1", "surface_order2", "voice", "voice1", "voice2"]:
            if k in meta and k not in rr:
                rr[k] = meta[k]
        pk = exact_pair(meta if meta else rr)
        rr["exact_pair"] = pk
        rr["exact_pair_set"] = pair_set(pk)
        rel = rr.get("relation") or rr.get("cause_relation")
        rr["relation_family"] = rel_family(rel)
        if rr.get("task") == "relation_comparison":
            if rr.get("suite") in {"mixed_held_seen_orientation", "heldheld_unseen_edge_closure"}:
                rr["comparison_info_kind"] = "eval"
            elif bool(rr.get("neutral_control")):
                rr["comparison_info_kind"] = "neutral_train"
            else:
                rr["comparison_info_kind"] = "informative_train"
        out.append(rr)
    return out


def row_score(r: Dict[str, Any]) -> float:
    return float(r["margin1_minus_0"])


def comparison_records(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for r in rows:
        if r.get("task") != "relation_comparison":
            continue
        margin = row_score(r)
        signed = margin if bool(r.get("label")) else -margin
        out.append({
            "condition": r.get("condition"), "arm": r.get("arm"), "seed": r.get("seed"), "suite": r.get("suite"),
            "row_id": r.get("row_id"), "relation1": r.get("relation1"), "relation2": r.get("relation2"),
            "label": bool(r.get("label")), "correct": bool(r.get("correct")), "pred": bool(r.get("pred")),
            "margin": margin, "signed_margin": signed,
            "exact_pair_set": r.get("exact_pair_set"), "exact_pair": r.get("exact_pair"),
            "neutral_control": r.get("neutral_control"), "geom_same_final_owner_under_true": r.get("geom_same_final_owner_under_true"),
            "comparison_info_kind": r.get("comparison_info_kind"),
        })
    return out


def summarize_comp(rows: Sequence[Dict[str, Any]], fields: Sequence[str]) -> Dict[str, Dict[str, Any]]:
    out = {}
    for k, rr in sorted(group(rows, fields).items()):
        out[k] = {
            "n": len(rr),
            "acc": mean([float(r["correct"]) for r in rr]),
            "pred_true_frac": mean([float(r["pred"]) for r in rr]),
            "signed_margin_mean": mean([float(r["signed_margin"]) for r in rr]),
            "signed_margin_std": std([float(r["signed_margin"]) for r in rr]),
            "true_accept": mean([float(r["pred"]) for r in rr if r["label"]]),
            "false_reject": mean([float(not r["pred"]) for r in rr if not r["label"]]),
        }
    return out


def state_choice(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        if r.get("task") != "state_query":
            continue
        by[(r.get("condition"), r.get("arm"), r.get("seed"), r.get("suite"), r.get("pair_id"), r.get("query_kind"))].append(r)
    out = []
    for _k, rr in by.items():
        tr = [r for r in rr if bool(r.get("label"))]
        fa = [r for r in rr if not bool(r.get("label"))]
        if len(tr) != 1 or len(fa) != 1:
            continue
        t, f = tr[0], fa[0]
        margin = row_score(t) - row_score(f)
        rel = t.get("relation") or t.get("cause_relation")
        out.append({
            "condition": t.get("condition"), "arm": t.get("arm"), "seed": t.get("seed"), "suite": t.get("suite"),
            "pair_id": t.get("pair_id"), "query_kind": t.get("query_kind"),
            "choice_correct": bool(margin > 0), "true_minus_false_margin": margin,
            "initial_pattern": t.get("initial_pattern"), "static_slot": t.get("static_slot"),
            "relation": rel, "relation_family": rel_family(rel), "voice": t.get("voice"),
            "exact_pair_set": t.get("exact_pair_set"), "exact_pair": t.get("exact_pair"),
        })
    return out


def pair_both(choices: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by: Dict[Tuple[Any, ...], Dict[str, Dict[str, Any]]] = defaultdict(dict)
    for r in choices:
        by[(r.get("condition"), r.get("arm"), r.get("seed"), r.get("suite"), r.get("pair_id"))][str(r.get("query_kind"))] = r
    out = []
    for _k, d in by.items():
        if "changed" not in d or "unchanged" not in d:
            continue
        c, u = d["changed"], d["unchanged"]
        out.append({
            "condition": c.get("condition"), "arm": c.get("arm"), "seed": c.get("seed"), "suite": c.get("suite"),
            "pair_id": c.get("pair_id"), "pair_both_correct": bool(c.get("choice_correct") and u.get("choice_correct")),
            "changed_correct": bool(c.get("choice_correct")), "unchanged_correct": bool(u.get("choice_correct")),
            "initial_pattern": c.get("initial_pattern"), "static_slot": c.get("static_slot"),
            "relation": c.get("relation"), "relation_family": c.get("relation_family"), "voice": c.get("voice"),
            "exact_pair_set": c.get("exact_pair_set"),
        })
    return out


def summarize_choice(rows: Sequence[Dict[str, Any]], fields: Sequence[str]) -> Dict[str, Dict[str, Any]]:
    out = {}
    for k, rr in sorted(group(rows, fields).items()):
        out[k] = {"n": len(rr), "acc": mean([float(r["choice_correct"]) for r in rr]), "margin_mean": mean([float(r["true_minus_false_margin"]) for r in rr]), "margin_std": std([float(r["true_minus_false_margin"]) for r in rr])}
    return out


def summarize_both(rows: Sequence[Dict[str, Any]], fields: Sequence[str]) -> Dict[str, Dict[str, Any]]:
    out = {}
    for k, rr in sorted(group(rows, fields).items()):
        out[k] = {"n": len(rr), "pair_both_acc": mean([float(r["pair_both_correct"]) for r in rr]), "changed_acc": mean([float(r["changed_correct"]) for r in rr]), "unchanged_acc": mean([float(r["unchanged_correct"]) for r in rr])}
    return out


def paired_arm_diffs(comp: Sequence[Dict[str, Any]], suite: str) -> Dict[str, Any]:
    rows = [r for r in comp if r.get("suite") == suite]
    by: Dict[Tuple[Any, ...], Dict[str, Dict[str, Any]]] = defaultdict(dict)
    for r in rows:
        by[(r.get("condition"), r.get("seed"), r.get("row_id"))][str(r.get("arm"))] = r
    out_rows = []
    for (cond, seed, row_id), d in by.items():
        if "aligned_state_bridge" in d and "inverted_state_bridge" in d:
            a, inv = d["aligned_state_bridge"], d["inverted_state_bridge"]
            out_rows.append({
                "condition": cond, "seed": seed, "row_id": row_id,
                "relation1": a.get("relation1"), "relation2": a.get("relation2"),
                "aligned_minus_inverted_signed_margin": float(a["signed_margin"]) - float(inv["signed_margin"]),
                "aligned_minus_inverted_accuracy": float(a["correct"]) - float(inv["correct"]),
                "aligned_signed": float(a["signed_margin"]), "inverted_signed": float(inv["signed_margin"]),
            })
    return {
        "by_condition": {k: {"n": len(v), "margin_diff": summarize([r["aligned_minus_inverted_signed_margin"] for r in v]), "acc_diff": summarize([r["aligned_minus_inverted_accuracy"] for r in v])} for k, v in sorted(group(out_rows, ["condition"]).items())},
        "by_condition_relation_pair": {k: {"n": len(v), "margin_diff": summarize([r["aligned_minus_inverted_signed_margin"] for r in v]), "acc_diff": summarize([r["aligned_minus_inverted_accuracy"] for r in v])} for k, v in sorted(group(out_rows, ["condition", "relation1", "relation2"]).items())},
    }


def load_outputs(input_dirs: Sequence[Path], data_root: Path) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    raw = raw_maps(data_root)
    train: List[Dict[str, Any]] = []
    eval_rows: List[Dict[str, Any]] = []
    src = {}
    for d in input_dirs:
        tr = enrich(load_jsonl(Path(d) / "per_row_train_predictions.jsonl"), raw)
        er = enrich(load_jsonl(Path(d) / "per_row_eval_predictions.jsonl"), raw)
        train.extend(tr); eval_rows.extend(er)
        src[str(d)] = {"train_rows": len(tr), "eval_rows": len(er)}
    return train, eval_rows, src


def build_report(input_dirs: Sequence[Path], data_root: Path) -> Dict[str, Any]:
    train, eval_rows, src = load_outputs(input_dirs, data_root)
    tr_comp = comparison_records(train)
    ev_comp = comparison_records(eval_rows)
    ev_choice = state_choice(eval_rows)
    ev_both = pair_both(ev_choice)
    tr_choice = state_choice(train)
    tr_both = pair_both(tr_choice)
    return {
        "status": "PAIR_BINDING_DECOMPOSITION_COMPLETE",
        "sources": src,
        "n_train_rows": len(train),
        "n_eval_rows": len(eval_rows),
        "train_comparison_by_condition_arm_info_pairset": summarize_comp(tr_comp, ["condition", "arm", "comparison_info_kind", "exact_pair_set"]),
        "train_state_by_condition_arm_relation_pattern_pairset": summarize_choice([r for r in tr_choice if r.get("query_kind") == "changed"], ["condition", "arm", "relation", "initial_pattern", "exact_pair_set"]),
        "train_pair_both_by_condition_arm_relation_pattern_pairset": summarize_both(tr_both, ["condition", "arm", "relation", "initial_pattern", "exact_pair_set"]),
        "eval_state_changed_by_condition_arm_suite_relation_pattern": summarize_choice([r for r in ev_choice if r.get("query_kind") == "changed"], ["condition", "arm", "suite", "relation", "initial_pattern"]),
        "eval_state_changed_by_condition_arm_suite_relation_family_pattern_voice": summarize_choice([r for r in ev_choice if r.get("query_kind") == "changed"], ["condition", "arm", "suite", "relation_family", "initial_pattern", "voice"]),
        "eval_pair_both_by_condition_arm_suite_relation_pattern": summarize_both(ev_both, ["condition", "arm", "suite", "relation", "initial_pattern"]),
        "eval_pair_both_by_condition_arm_suite_relation_family_pattern": summarize_both(ev_both, ["condition", "arm", "suite", "relation_family", "initial_pattern"]),
        "eval_heldheld_edge_closure_by_condition_arm_relation_pair": summarize_comp([r for r in ev_comp if r.get("suite") == "heldheld_unseen_edge_closure"], ["condition", "arm", "relation1", "relation2"]),
        "eval_mixed_by_condition_arm_relation_pair": summarize_comp([r for r in ev_comp if r.get("suite") == "mixed_held_seen_orientation"], ["condition", "arm", "relation1", "relation2"]),
        "row_paired_aligned_inverted_mixed": paired_arm_diffs(ev_comp, "mixed_held_seen_orientation"),
        "row_paired_aligned_inverted_heldheld_closure": paired_arm_diffs(ev_comp, "heldheld_unseen_edge_closure"),
    }


def fmt(x: Any, digits: int = 3) -> str:
    if x is None:
        return "n/a"
    try:
        return f"{float(x):.{digits}f}"
    except Exception:
        return str(x)


def write_summary(path: Path, rep: Dict[str, Any]) -> None:
    lines: List[str] = []
    lines.append("# research pair-binding pilot decomposition")
    lines.append("")
    lines.append("This CPU-only decomposition joins saved logits with raw substrate metadata. It separates direct-anchor relations h0/h2 from graph-transfer relations h1/h3, train informative versus neutral held-held rows, held-held edge closure, exact dyad membership, and row-paired aligned/inverted signed margins.")
    lines.append("")
    lines.append(f"- train prediction rows: {rep['n_train_rows']}")
    lines.append(f"- eval prediction rows: {rep['n_eval_rows']}")
    lines.append("")
    lines.append("## Train held-held comparison fit")
    lines.append("")
    lines.append("| condition | arm | info kind | pair set | acc | signed margin | pred true frac |")
    lines.append("|---|---|---|---|---:|---:|---:|")
    for k, r in rep["train_comparison_by_condition_arm_info_pairset"].items():
        cond, arm, info, pset = k.split("|")
        lines.append(f"| {cond} | {arm} | {info} | {pset} | {fmt(r.get('acc'))} | {fmt(r.get('signed_margin_mean'))} | {fmt(r.get('pred_true_frac'))} |")
    lines.append("")
    lines.append("## Eval state: relation-by-relation changed exact choice")
    lines.append("")
    lines.append("| condition | arm | relation | pattern | changed exact | margin | pair-both |")
    lines.append("|---|---|---|---|---:|---:|---:|")
    ch = rep["eval_state_changed_by_condition_arm_suite_relation_pattern"]
    pb = rep["eval_pair_both_by_condition_arm_suite_relation_pattern"]
    for k in sorted(ch):
        cond, arm, suite, rel, pat = k.split("|")
        if suite != "paired_state_conservation":
            continue
        lines.append(f"| {cond} | {arm} | {rel} | {pat} | {fmt(ch[k].get('acc'))} | {fmt(ch[k].get('margin_mean'))} | {fmt(pb.get(k, {}).get('pair_both_acc'))} |")
    lines.append("")
    lines.append("## Eval state: voice split for graph-transfer same-initial")
    lines.append("")
    lines.append("| condition | arm | voice | changed exact | margin |")
    lines.append("|---|---|---|---:|---:|")
    voice_tab = rep["eval_state_changed_by_condition_arm_suite_relation_family_pattern_voice"]
    for k in sorted(voice_tab):
        cond, arm, suite, fam, pat, voice = k.split("|")
        if suite == "paired_state_conservation" and fam == "graph_transfer" and pat == "same":
            r = voice_tab[k]
            lines.append(f"| {cond} | {arm} | {voice} | {fmt(r.get('acc'))} | {fmt(r.get('margin_mean'))} |")
    lines.append("")
    lines.append("## Held-held unseen edge closure")
    lines.append("")
    lines.append("| condition | arm | rel1 | rel2 | acc | signed margin | pred true frac |")
    lines.append("|---|---|---|---|---:|---:|---:|")
    for k, r in rep["eval_heldheld_edge_closure_by_condition_arm_relation_pair"].items():
        cond, arm, r1, r2 = k.split("|")
        lines.append(f"| {cond} | {arm} | {r1} | {r2} | {fmt(r.get('acc'))} | {fmt(r.get('signed_margin_mean'))} | {fmt(r.get('pred_true_frac'))} |")
    lines.append("")
    lines.append("## Mixed held-seen row-paired aligned/inverted differences")
    lines.append("")
    lines.append("| condition | n rows | mean signed-margin diff | std | mean accuracy diff |")
    lines.append("|---|---:|---:|---:|---:|")
    for cond, r in rep["row_paired_aligned_inverted_mixed"]["by_condition"].items():
        md = r["margin_diff"]; ad = r["acc_diff"]
        lines.append(f"| {cond} | {md.get('n')} | {fmt(md.get('mean'))} | {fmt(md.get('std'))} | {fmt(ad.get('mean'))} |")
    lines.append("")
    lines.append("## Scientific reading")
    lines.append("")
    lines.append("The saved-output decomposition should be read together with the main analysis. The strongest pattern is direct-anchor state learning without graph-transfer state updating: directly bridged h0/h2 same-initial rows can be high in aligned arms, but h1/h3 same-initial rows remain low. Held-held edge closure and row-paired mixed margins determine whether the comparison graph itself was learned; if closure is weak or polarity is not mirrored, the failure is earlier than state composition. If closure is strong while graph-transfer state fails, the comparison and state objectives remain functionally separate despite shared text exposure.")
    lines.append("")
    lines.append("## Files")
    lines.append(f"- full JSON: `{path.parent / 'pair_binding_decomposition.json'}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dirs", nargs="+", type=Path, default=DEFAULT_INPUTS)
    ap.add_argument("--data-root", type=Path, default=DATA_ROOT)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()
    rep = build_report(args.input_dirs, args.data_root)
    args.out.mkdir(parents=True, exist_ok=True)
    write_json(args.out / "pair_binding_decomposition.json", rep)
    write_summary(args.out / "pair_binding_decomposition_summary.md", rep)
    print(json.dumps({
        "status": rep["status"],
        "summary": str(args.out / "pair_binding_decomposition_summary.md"),
        "json": str(args.out / "pair_binding_decomposition.json"),
        "n_train_rows": rep["n_train_rows"],
        "n_eval_rows": rep["n_eval_rows"],
        "no_model_loading_training_evaluation_upload_or_leaderboard": True,
    }, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
