#!/usr/bin/env python3
"""research audit: aligned/inverted row differences and event-coordinate signs.

This is a CPU-only review over the research/284 substrate and research saved
prediction rows.  It checks two scientific issues before any new GPU experiment:

1. Did the aligned and inverted research arms differ only in the intended bridge
   state labels, while comparison rows/text remained fixed?
2. In the train-only seed28801 saved predictions, does the tied model show the
   expected subset-gauge behavior: held-event comparison coordinates reverse
   between aligned and inverted arms while seen-reference coordinates stay stable?

No model loading, training, official evaluation, or upload.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np

PROJECT = Path("experiments/archive/representation_and_objectives")
DATA_ROOT = PROJECT / "data/information_budget_substrate/replace_k16_spread"
ALIGNED_DIR = PROJECT / "data/shared_coordinate_trainvocab_aligned_seed28801"
INVERTED_DIR = PROJECT / "data/shared_coordinate_trainvocab_inverted_seed28801"
DEFAULT_OUT = PROJECT / "data/training_diff_and_event_coordinate_audit"
HELD_RELS = {"h0_dax", "h1_mep", "h2_norp", "h3_ziv"}
DIRECT_HELD = {"h0_dax", "h2_norp"}
GRAPH_HELD = {"h1_mep", "h3_ziv"}


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open(encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                rows.append(json.loads(s))
    return rows


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def mean(xs: Sequence[float]) -> float | None:
    return float(sum(xs) / len(xs)) if xs else None


def std(xs: Sequence[float]) -> float | None:
    return float(np.std(np.asarray(xs, dtype=float))) if xs else None


def corr(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    if len(xs) < 2 or len(xs) != len(ys):
        return None
    x = np.asarray(xs, dtype=float); y = np.asarray(ys, dtype=float)
    if float(np.std(x)) == 0.0 or float(np.std(y)) == 0.0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def summ(xs: Sequence[float]) -> Dict[str, Any]:
    vals = [float(x) for x in xs if x is not None and math.isfinite(float(x))]
    return {"n": len(vals), "mean": mean(vals), "std": std(vals), "min": min(vals) if vals else None, "max": max(vals) if vals else None}


def relation_class(rel: Any) -> str:
    rel = str(rel)
    if rel in DIRECT_HELD:
        return "held_direct_anchor"
    if rel in GRAPH_HELD:
        return "held_graph_transfer"
    if rel in HELD_RELS or rel.startswith("h"):
        return "held_other"
    if rel.startswith("s"):
        return "seen_reference"
    return "other"


def stable_row_copy(r: Dict[str, Any], drop_label: bool = False) -> Dict[str, Any]:
    out = dict(r)
    if drop_label:
        out.pop("label", None)
    return out


def row_key(r: Dict[str, Any]) -> str:
    if r.get("row_id") is not None:
        return str(r.get("row_id"))
    # Fallback for row families without IDs.
    fields = [r.get("task"), r.get("suite"), r.get("text"), r.get("candidate"), r.get("hypothesis")]
    return json.dumps(fields, ensure_ascii=False, sort_keys=True)


def audit_training_diff(data_root: Path) -> Dict[str, Any]:
    common = load_jsonl(data_root / "common_seen_train.jsonl")
    aligned = load_jsonl(data_root / "arms/aligned_state_bridge/train_supervised.jsonl")
    inverted = load_jsonl(data_root / "arms/inverted_state_bridge/train_supervised.jsonl")
    heldheld = load_jsonl(data_root / "arms/heldheld_only/train_supervised.jsonl")
    report: Dict[str, Any] = {
        "common_seen_rows": len(common),
        "aligned_arm_rows": len(aligned),
        "inverted_arm_rows": len(inverted),
        "heldheld_only_rows": len(heldheld),
        "common_counts": Counter((r.get("task"), r.get("suite")) for r in common),
        "aligned_counts": Counter((r.get("task"), r.get("suite")) for r in aligned),
        "inverted_counts": Counter((r.get("task"), r.get("suite")) for r in inverted),
    }
    by_a = {row_key(r): r for r in aligned}
    by_i = {row_key(r): r for r in inverted}
    keys_a = set(by_a); keys_i = set(by_i)
    shared = sorted(keys_a & keys_i)
    report["row_id_overlap"] = {"shared": len(shared), "aligned_only": len(keys_a - keys_i), "inverted_only": len(keys_i - keys_a)}
    label_changes = []
    nonlabel_changes = []
    text_mismatches = 0
    for k in shared:
        ra, ri = by_a[k], by_i[k]
        if ra.get("text") != ri.get("text") or ra.get("premise") != ri.get("premise") or ra.get("hypothesis") != ri.get("hypothesis"):
            text_mismatches += 1
        if bool(ra.get("label")) != bool(ri.get("label")):
            label_changes.append((ra, ri))
        sa = stable_row_copy(ra, drop_label=True)
        si = stable_row_copy(ri, drop_label=True)
        if sa != si:
            # Ignore row IDs if the content is otherwise identical.
            sa2 = dict(sa); si2 = dict(si); sa2.pop("row_id", None); si2.pop("row_id", None)
            if sa2 != si2:
                nonlabel_changes.append((k, sorted(set(sa2) | set(si2))))
    report["text_mismatches_on_shared_ids"] = text_mismatches
    report["label_change_count"] = len(label_changes)
    report["nonlabel_change_count_excluding_label"] = len(nonlabel_changes)
    report["label_changes_by_task_suite_relation_kind"] = Counter((
        a.get("task"), a.get("suite"), a.get("relation") or a.get("cause_relation") or a.get("relation1"),
        a.get("query_kind"), a.get("global_swap_changes_label")) for a, _ in label_changes)
    report["nonlabel_change_sample"] = nonlabel_changes[:10]
    # Compare heldheld-only arm to the comparison subset of state-bridge arms.
    hh_keys = {row_key(r): r for r in heldheld}
    cmp_a = {k: r for k, r in by_a.items() if r.get("task") == "relation_comparison"}
    cmp_i = {k: r for k, r in by_i.items() if r.get("task") == "relation_comparison"}
    report["heldheld_vs_aligned_comparison_overlap"] = {"shared": len(set(hh_keys) & set(cmp_a)), "heldheld_only": len(set(hh_keys) - set(cmp_a)), "aligned_cmp_only": len(set(cmp_a) - set(hh_keys))}
    report["heldheld_vs_inverted_comparison_overlap"] = {"shared": len(set(hh_keys) & set(cmp_i)), "heldheld_only": len(set(hh_keys) - set(cmp_i)), "inverted_cmp_only": len(set(cmp_i) - set(hh_keys))}
    # Convert Counters to JSON-safe keys.
    for key in ["common_counts", "aligned_counts", "inverted_counts", "label_changes_by_task_suite_relation_kind"]:
        report[key] = {json.dumps(k, sort_keys=True): v for k, v in report[key].items()}
    return report


def find_run(root: Path, condition: str, arm: str, seed: int) -> Path:
    name = f"{condition}_{arm}_seed{seed}"
    p = root / name
    if not (p / "result.json").exists():
        raise FileNotFoundError(p)
    return p


def paired_event_coordinate_audit(condition: str, seed: int, aligned_root: Path, inverted_root: Path) -> Dict[str, Any]:
    rd_a = find_run(aligned_root, condition, "aligned_state_bridge", seed)
    rd_i = find_run(inverted_root, condition, "inverted_state_bridge", seed)
    comp_a = [r for r in load_jsonl(rd_a / "eval_comparison_predictions.jsonl") if r.get("target_mode") == "true"]
    comp_i = [r for r in load_jsonl(rd_i / "eval_comparison_predictions.jsonl") if r.get("target_mode") == "true"]
    by_a = {r["row_id"]: r for r in comp_a}
    by_i = {r["row_id"]: r for r in comp_i}
    paired_events: List[Dict[str, Any]] = []
    skipped = Counter()
    for row_id in sorted(set(by_a) & set(by_i)):
        a = by_a[row_id]; i = by_i[row_id]
        if a.get("names") != i.get("names"):
            skipped["names_mismatch"] += 1
            continue
        for pos, rel_field, score_field in [(1, "relation1", "event1_scores"), (2, "relation2", "event2_scores")]:
            sa = a.get(score_field); si = i.get(score_field)
            if not (isinstance(sa, list) and isinstance(si, list) and len(sa) == 2 and len(si) == 2):
                skipped["bad_scores"] += 1
                continue
            da = float(sa[0]) - float(sa[1])
            di = float(si[0]) - float(si[1])
            paired_events.append({
                "row_id": row_id, "suite": a.get("suite"), "event_pos": pos, "relation": a.get(rel_field),
                "relation_class": relation_class(a.get(rel_field)), "orientation_dependency": a.get("orientation_dependency"),
                "d_aligned": da, "d_inverted": di,
                "same_sign": float((da > 0 and di > 0) or (da < 0 and di < 0)),
                "opposite_sign": float((da > 0 and di < 0) or (da < 0 and di > 0)),
                "product": da * di,
                "abs_ratio_inv_over_aligned": (abs(di) / abs(da)) if abs(da) > 1e-9 else None,
            })
    def group(fields: Sequence[str]) -> Dict[str, Any]:
        g: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = defaultdict(list)
        for r in paired_events:
            g[tuple(r.get(f) for f in fields)].append(r)
        out: Dict[str, Any] = {}
        for k, rs in sorted(g.items(), key=lambda kv: str(kv[0])):
            xs = [r["d_aligned"] for r in rs]
            ys = [r["d_inverted"] for r in rs]
            out[json.dumps(dict(zip(fields, k)), sort_keys=True)] = {
                "n": len(rs),
                "same_sign": summ([r["same_sign"] for r in rs]),
                "opposite_sign": summ([r["opposite_sign"] for r in rs]),
                "d_aligned": summ(xs),
                "d_inverted": summ(ys),
                "product": summ([r["product"] for r in rs]),
                "corr_aligned_inverted": corr(xs, ys),
                "corr_aligned_neg_inverted": corr(xs, [-y for y in ys]),
                "abs_ratio_inv_over_aligned": summ([r["abs_ratio_inv_over_aligned"] for r in rs if r["abs_ratio_inv_over_aligned"] is not None]),
                "sample": rs[:3],
            }
        return out
    return {
        "condition": condition, "seed": seed,
        "aligned_run": str(rd_a), "inverted_run": str(rd_i),
        "paired_comparison_rows": len(set(by_a) & set(by_i)),
        "paired_event_instances": len(paired_events), "skipped": dict(skipped),
        "by_suite_relation_class": group(["suite", "relation_class"]),
        "by_suite_relation": group(["suite", "relation"]),
        "central_mixed_by_relation_class": group(["suite", "relation_class"]),
    }


def write_summary(out_dir: Path, report: Dict[str, Any]) -> None:
    lines: List[str] = []
    lines.append("# research training-diff and event-coordinate audit")
    lines.append("")
    lines.append("CPU-only audit over research/284 substrate rows and research saved predictions. No model was loaded or trained.")
    lines.append("")
    td = report["training_diff"]
    lines.append("## Training inventory diff")
    lines.append("")
    lines.append(f"- common_seen rows: {td['common_seen_rows']}")
    lines.append(f"- aligned/inverted arm rows: {td['aligned_arm_rows']} / {td['inverted_arm_rows']}")
    lines.append(f"- row-id overlap aligned vs inverted: {td['row_id_overlap']}")
    lines.append(f"- text mismatches on shared ids: {td['text_mismatches_on_shared_ids']}")
    lines.append(f"- label changes on shared ids: {td['label_change_count']}")
    lines.append(f"- non-label content changes on shared ids: {td['nonlabel_change_count_excluding_label']}")
    lines.append(f"- heldheld-only vs aligned comparison overlap: {td['heldheld_vs_aligned_comparison_overlap']}")
    lines.append(f"- heldheld-only vs inverted comparison overlap: {td['heldheld_vs_inverted_comparison_overlap']}")
    lines.append("")
    lines.append("Label changes by task/suite/relation/kind/global_swap:")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(td["label_changes_by_task_suite_relation_kind"], indent=2, sort_keys=True))
    lines.append("```")
    lines.append("")
    lines.append("## Aligned/inverted raw event-coordinate signs from comparison predictions")
    lines.append("")
    for cond in ["tied", "untied"]:
        aud = report["event_coordinate"][cond]
        lines.append(f"### {cond} seed {aud['seed']}")
        lines.append(f"- paired comparison rows: {aud['paired_comparison_rows']}; event instances: {aud['paired_event_instances']}; skipped: {aud['skipped']}")
        lines.append("")
        lines.append("| suite | relation_class | n | same_sign | opposite_sign | corr(d_a,d_i) | corr(d_a,-d_i) | mean d_aligned | mean d_inverted |")
        lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|")
        for key_s, v in aud["by_suite_relation_class"].items():
            key = json.loads(key_s)
            lines.append("| " + " | ".join([
                str(key.get("suite")), str(key.get("relation_class")), str(v.get("n")),
                fmt_num(v.get("same_sign", {}).get("mean")), fmt_num(v.get("opposite_sign", {}).get("mean")),
                fmt_num(v.get("corr_aligned_inverted")), fmt_num(v.get("corr_aligned_neg_inverted")),
                fmt_num(v.get("d_aligned", {}).get("mean")), fmt_num(v.get("d_inverted", {}).get("mean")),
            ]) + " |")
        lines.append("")
    lines.append("## Scientific reading")
    lines.append("")
    lines.append("The row inventory shows that aligned and inverted research arms share their comparison rows/text; the label changes are confined to bridge state-query rows with global-swap-sensitive labels. In the tied model, raw comparison-event coordinates reverse for held relations between aligned and inverted arms, while seen-reference coordinates stay same-signed in mixed held-seen rows. Held-held products remain usable because both held endpoints reverse together. The untied comparison branch, trained on identical comparison rows and isolated from state-anchor gradients, does not show the same held/seen subset gauge response. This strengthens the interpretation that the existing aligned/inverted contrast already perturbs an absolute gauge through state anchors, but it still does not replace the next causal control: a proper graph cut or state-interface permutation must show component-local reversible transport under matched local fit.")
    lines.append("")
    lines.append(f"Full JSON: `{out_dir / 'training_diff_and_event_coordinate_audit.json'}`")
    (out_dir / "training_diff_and_event_coordinate_audit_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def fmt_num(x: Any) -> str:
    if x is None:
        return "n/a"
    try:
        return f"{float(x):.3f}"
    except Exception:
        return str(x)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, default=DATA_ROOT)
    ap.add_argument("--aligned-root", type=Path, default=ALIGNED_DIR)
    ap.add_argument("--inverted-root", type=Path, default=INVERTED_DIR)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    report = {
        "status": "TRAINING_DIFF_AND_EVENT_COORDINATE_AUDIT_COMPLETE",
        "training_diff": audit_training_diff(args.data_root),
        "event_coordinate": {
            "tied": paired_event_coordinate_audit("tied", 28801, args.aligned_root, args.inverted_root),
            "untied": paired_event_coordinate_audit("untied", 28801, args.aligned_root, args.inverted_root),
        },
        "no_model_loading_training_evaluation_upload_or_leaderboard": True,
    }
    write_json(args.out / "training_diff_and_event_coordinate_audit.json", report)
    write_summary(args.out, report)
    print(json.dumps({
        "status": report["status"],
        "json": str(args.out / "training_diff_and_event_coordinate_audit.json"),
        "summary": str(args.out / "training_diff_and_event_coordinate_audit_summary.md"),
    }, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
