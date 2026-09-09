#!/usr/bin/env python3
"""research prediction-transition analysis for the research cheap7 common screen.

The common screen used during functional_learning real-pretraining work is not the same as
full official evaluation: BLiMP/Supplement/EWoK/Entity/GlobalPIQA use fast splits,
COMPS uses full_eval, and Reading is a regression metric.  This script compares
two completed common-screen evaluations at the prediction level and recomputes
macro accuracy from the saved predictions.  It is designed for the scale-confound
question in research:

  coherent86 alpha0.75  -> coherent86 alpha0.5  (readout scale only)
  coherent86 alpha0.5   -> pa87 alpha0.5        (new 1M-word parent-anchored child at same scale)

Item transitions are diagnostic evidence about what changed; the common-screen
score remains a fast research screen, not a complete official result.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse
import collections
import json
import pathlib
from typing import Any, Optional


def find_user_root() -> pathlib.Path:
    return _PUBLIC_ROOT


ROOT = find_user_root()
STRICT = ROOT / "experiments/archive/initial_model_studies/repos/babylm-eval/strict"
FAST = STRICT / "evaluation_data/fast_eval"
FULL = STRICT / "evaluation_data/full_eval"
COLS = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA_parallel", "GlobalPIQA_nonparallel"]
CHEAP_KEYS = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA_mean", "Reading"]
COMPS_FILE = {"base": "comps_base", "wugs_dist_before": "comps_wugs_dist-before", "wugs_dist_in_between": "comps_wugs_dist-in-between", "wugs": "comps_wugs"}
DATA_DIR = {
    "BLiMP": FAST / "blimp_fast",
    "Supplement": FAST / "supplement_fast",
    "EWoK": FAST / "evaluation_data/fast_eval/ewok_fast",
    "Entity": FAST / "entity_tracking_fast",
    "COMPS": FULL / "comps",
    "GlobalPIQA_parallel": FAST / "global_piqa_parallel",
    "GlobalPIQA_nonparallel": FAST / "global_piqa_nonparallel",
}


def rel(path: pathlib.Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except Exception:
        return str(path)


def as_path(x: str | pathlib.Path) -> pathlib.Path:
    p = pathlib.Path(x)
    return p if p.is_absolute() else ROOT / p


def load_json(path: pathlib.Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
    out = []
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line))
    return out


def clean(x: Any) -> Any:
    return x.strip() if isinstance(x, str) else x


def find_pred(out_root: pathlib.Path, tag: str, col: str) -> pathlib.Path:
    root = out_root / "outputs" / tag / col
    hits = sorted(root.rglob("predictions.json"))
    if len(hits) != 1:
        raise FileNotFoundError(f"expected one predictions.json under {root}, found {len(hits)}")
    return hits[0]


def make_item(column: str, subtask: str, idx: int, item_id: Any, pred: Any, target: Any, meta: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    return {
        "column": column,
        "subtask": str(subtask),
        "index": int(idx),
        "id": str(item_id),
        "pred": clean(pred),
        "target": clean(target),
        "correct": clean(pred) == clean(target),
        "metadata": meta or {},
    }


def pred_list(preds: dict[str, Any], key: str) -> list[dict[str, Any]]:
    return preds[key]["predictions"]


def blimp_like(pred_path: pathlib.Path, col: str) -> list[dict[str, Any]]:
    preds = load_json(pred_path)
    out = []
    for subtask in preds.keys():
        data = read_jsonl((DATA_DIR[col] / subtask).with_suffix(".jsonl"))
        for i, (pr, gold) in enumerate(zip(pred_list(preds, subtask), data)):
            out.append(make_item(col, subtask, i, pr.get("id", f"{subtask}_{i}"), pr.get("pred"), gold["sentence_good"],
                                 {k: gold.get(k) for k in ["field", "linguistics_term", "UID", "contrast", "pair_id", "row"] if k in gold}))
    return out


def ewok(pred_path: pathlib.Path) -> list[dict[str, Any]]:
    preds = load_json(pred_path)
    out = []
    for subtask in preds.keys():
        data = read_jsonl((DATA_DIR["EWoK"] / subtask).with_suffix(".jsonl"))
        for i, (pr, gold) in enumerate(zip(pred_list(preds, subtask), data)):
            target = " ".join([gold["Context1"], gold["Target1"]]).strip()
            out.append(make_item("EWoK", subtask, i, pr.get("id", f"{subtask}_{i}"), pr.get("pred"), target,
                                 {k: gold.get(k) for k in ["Domain", "ConceptA", "ConceptB", "ContextType", "ContextDiff", "TargetDiff"] if k in gold}))
    return out


def entity_targets() -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for gold in read_jsonl(DATA_DIR["Entity"] / "regular.jsonl"):
        if any("nothing" in str(opt) for opt in gold["options"]):
            continue
        out[f'regular_{gold["numops"]}_ops'].append(gold)
    return dict(out)


def entity(pred_path: pathlib.Path) -> list[dict[str, Any]]:
    preds = load_json(pred_path)
    targets = entity_targets()
    out = []
    for subtask in preds.keys():
        data = targets[subtask]
        for i, (pr, gold) in enumerate(zip(pred_list(preds, subtask), data)):
            out.append(make_item("Entity", subtask, i, pr.get("id", f"{subtask}_{i}"), pr.get("pred"), gold["options"][0],
                                 {"numops": gold.get("numops"), "example_id": gold.get("example_id"), "sample_id": gold.get("sample_id")}))
    return out


def comps(pred_path: pathlib.Path) -> list[dict[str, Any]]:
    preds = load_json(pred_path)
    out = []
    for subtask in preds.keys():
        data = read_jsonl((DATA_DIR["COMPS"] / COMPS_FILE[subtask]).with_suffix(".jsonl"))
        for i, (pr, gold) in enumerate(zip(pred_list(preds, subtask), data)):
            target = " ".join([gold["prefix_acceptable"], gold["property_phrase"]]).strip()
            out.append(make_item("COMPS", subtask, i, pr.get("id", f"{subtask}_{i}"), pr.get("pred"), target,
                                 {k: gold.get(k) for k in ["condition", "pair_id", "property_phrase"] if k in gold}))
    return out


def globalpiqa(pred_path: pathlib.Path, col: str, task_name: str) -> list[dict[str, Any]]:
    preds = load_json(pred_path)
    data = read_jsonl(DATA_DIR[col] / "eng_latn.jsonl")
    out = []
    for i, gold in enumerate(data):
        eid = gold["example_id"]
        pr = pred_list(preds, eid)[0]
        target = gold[f"solution{gold['label']}"]
        out.append(make_item(col, task_name, i, eid, pr.get("pred"), target,
                             {k: gold.get(k) for k in ["categories", "label", "language", "supplement"] if k in gold}))
    return out


def load_items(out_root: pathlib.Path, tag: str) -> dict[str, list[dict[str, Any]]]:
    return {
        "BLiMP": blimp_like(find_pred(out_root, tag, "BLiMP"), "BLiMP"),
        "Supplement": blimp_like(find_pred(out_root, tag, "Supplement"), "Supplement"),
        "EWoK": ewok(find_pred(out_root, tag, "EWoK")),
        "Entity": entity(find_pred(out_root, tag, "Entity")),
        "COMPS": comps(find_pred(out_root, tag, "COMPS")),
        "GlobalPIQA_parallel": globalpiqa(find_pred(out_root, tag, "GlobalPIQA_parallel"), "GlobalPIQA_parallel", "global_piqa_parallel"),
        "GlobalPIQA_nonparallel": globalpiqa(find_pred(out_root, tag, "GlobalPIQA_nonparallel"), "GlobalPIQA_nonparallel", "global_piqa_nonparallel"),
    }


def key_item(x: dict[str, Any]) -> tuple[str, int, str]:
    return (str(x["subtask"]), int(x["index"]), str(x.get("id", "")))


def compare(a: list[dict[str, Any]], b: list[dict[str, Any]], max_examples: int) -> dict[str, Any]:
    amap = {key_item(x): x for x in a}
    bmap = {key_item(x): x for x in b}
    keys = sorted(set(amap) & set(bmap))
    sub = collections.defaultdict(lambda: {"n": 0, "a": 0, "b": 0, "net": 0, "a_only": 0, "b_only": 0})
    both_correct = a_only = b_only = both_wrong = 0
    gains = []
    losses = []
    for k in keys:
        ar = amap[k]
        br = bmap[k]
        ac = bool(ar["correct"])
        bc = bool(br["correct"])
        if ac and bc:
            both_correct += 1
        elif ac and not bc:
            a_only += 1
            if len(losses) < max_examples:
                losses.append({"subtask": ar["subtask"], "id": ar["id"], "index": ar["index"], "target": ar["target"], "a_pred": ar["pred"], "b_pred": br["pred"], "metadata": ar.get("metadata", {})})
        elif (not ac) and bc:
            b_only += 1
            if len(gains) < max_examples:
                gains.append({"subtask": ar["subtask"], "id": ar["id"], "index": ar["index"], "target": ar["target"], "a_pred": ar["pred"], "b_pred": br["pred"], "metadata": ar.get("metadata", {})})
        else:
            both_wrong += 1
        sr = sub[str(ar["subtask"])]
        sr["n"] += 1
        sr["a"] += int(ac)
        sr["b"] += int(bc)
        sr["net"] += int(bc) - int(ac)
        sr["a_only"] += int(ac and not bc)
        sr["b_only"] += int((not ac) and bc)
    subtasks = []
    for name, sr in sub.items():
        n = sr["n"]
        aa = 100.0 * sr["a"] / n
        ba = 100.0 * sr["b"] / n
        subtasks.append({"subtask": name, "n": n, "a_acc": aa, "b_acc": ba, "delta_b_minus_a": ba - aa, "net_item_delta": sr["net"], "a_only_correct": sr["a_only"], "b_only_correct": sr["b_only"]})
    subtasks.sort(key=lambda x: (abs(float(x["delta_b_minus_a"])), x["n"]), reverse=True)
    n = len(keys)
    a_micro = 100.0 * (both_correct + a_only) / n
    b_micro = 100.0 * (both_correct + b_only) / n
    a_macro = sum(x["a_acc"] for x in subtasks) / len(subtasks)
    b_macro = sum(x["b_acc"] for x in subtasks) / len(subtasks)
    return {
        "n_common": n,
        "a_micro_acc": a_micro,
        "b_micro_acc": b_micro,
        "delta_micro_b_minus_a": b_micro - a_micro,
        "a_macro_acc": a_macro,
        "b_macro_acc": b_macro,
        "delta_macro_b_minus_a": b_macro - a_macro,
        "both_correct": both_correct,
        "a_only_correct": a_only,
        "b_only_correct": b_only,
        "both_wrong": both_wrong,
        "net_item_delta": b_only - a_only,
        "subtasks_by_abs_delta": subtasks,
        "example_gains_b_over_a": gains,
        "example_losses_b_vs_a": losses,
    }


def computed_scores(items: dict[str, list[dict[str, Any]]]) -> dict[str, float]:
    out: dict[str, float] = {}
    for col, rows in items.items():
        groups = collections.defaultdict(list)
        for r in rows:
            groups[str(r["subtask"])].append(r)
        out[col] = sum(100.0 * sum(int(x["correct"]) for x in g) / len(g) for g in groups.values()) / len(groups)
    if "GlobalPIQA_parallel" in out and "GlobalPIQA_nonparallel" in out:
        out["GlobalPIQA_mean"] = (out["GlobalPIQA_parallel"] + out["GlobalPIQA_nonparallel"]) / 2.0
    return out


def eval_scores(eval_json: pathlib.Path) -> dict[str, float]:
    d = load_json(eval_json)
    scores = d.get("scores") or {}
    return {k: float(v) for k, v in scores.items() if isinstance(v, (int, float))}


def add_reading_and_mean(scores: dict[str, float], eval_json: pathlib.Path) -> dict[str, float]:
    out = dict(scores)
    es = eval_scores(eval_json)
    for k in ["Reading", "Reading_eye", "Reading_self_paced"]:
        if k in es:
            out[k] = es[k]
    vals = [out.get(k) for k in CHEAP_KEYS]
    if all(v is not None for v in vals):
        out["equal_valid_mean"] = sum(float(v) for v in vals) / len(vals)
    return out


def write_md(path: pathlib.Path, result: dict[str, Any]) -> None:
    lines = []
    lines.append(f"# research common-screen transition: {result['b_label']} vs {result['a_label']}\n")
    lines.append("## Common-screen score summary\n")
    lines.append("Computed scores are recomputed from saved predictions with the research data coordinate; Reading is copied from the eval JSON. Payload scores are from the saved evaluator JSON.\n")
    lines.append("| column | A payload | B payload | payload B-A | A computed | B computed | computed B-A |\n|---|---:|---:|---:|---:|---:|---:|")
    for k in ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA_parallel", "GlobalPIQA_nonparallel", "GlobalPIQA_mean", "Reading", "equal_valid_mean"]:
        ap = result["a_payload_scores"].get(k)
        bp = result["b_payload_scores"].get(k)
        ac = result["a_computed_scores"].get(k)
        bc = result["b_computed_scores"].get(k)
        def fmt(x): return "" if x is None else f"{float(x):.6f}"
        def delt(x, y): return "" if x is None or y is None else f"{float(y)-float(x):+.6f}"
        lines.append(f"| {k} | {fmt(ap)} | {fmt(bp)} | {delt(ap, bp)} | {fmt(ac)} | {fmt(bc)} | {delt(ac, bc)} |")
    lines.append("\n## Prediction-level transitions\n")
    lines.append("| column | n | A macro | B macro | B-A macro | A micro | B micro | B-A micro | B-only | A-only | net item |\n|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for col, rec in result["comparisons"].items():
        lines.append(f"| {col} | {rec['n_common']} | {rec['a_macro_acc']:.6f} | {rec['b_macro_acc']:.6f} | {rec['delta_macro_b_minus_a']:+.6f} | {rec['a_micro_acc']:.6f} | {rec['b_micro_acc']:.6f} | {rec['delta_micro_b_minus_a']:+.6f} | {rec['b_only_correct']} | {rec['a_only_correct']} | {rec['net_item_delta']:+d} |")
    lines.append("\n## Largest subtask movements\n")
    for col, rec in result["comparisons"].items():
        lines.append(f"\n### {col}\n")
        lines.append("| subtask | n | A acc | B acc | B-A | net items |\n|---|---:|---:|---:|---:|---:|")
        for sr in rec["subtasks_by_abs_delta"][:15]:
            lines.append(f"| {sr['subtask']} | {sr['n']} | {sr['a_acc']:.6f} | {sr['b_acc']:.6f} | {sr['delta_b_minus_a']:+.6f} | {sr['net_item_delta']:+d} |")
    lines.append("\n## Interpretation note\n")
    lines.append(result["interpretation_note"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--a-label", required=True)
    ap.add_argument("--a-out-root", required=True)
    ap.add_argument("--a-tag", required=True)
    ap.add_argument("--a-eval-json", required=True)
    ap.add_argument("--b-label", required=True)
    ap.add_argument("--b-out-root", required=True)
    ap.add_argument("--b-tag", required=True)
    ap.add_argument("--b-eval-json", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out-root", default="experiments/archive/functional_learning/data/common_screen_transitions")
    ap.add_argument("--max-examples", type=int, default=20)
    args = ap.parse_args()

    a_items = load_items(as_path(args.a_out_root), args.a_tag)
    b_items = load_items(as_path(args.b_out_root), args.b_tag)
    comps_out = {col: compare(a_items[col], b_items[col], args.max_examples) for col in COLS}
    a_comp = add_reading_and_mean(computed_scores(a_items), as_path(args.a_eval_json))
    b_comp = add_reading_and_mean(computed_scores(b_items), as_path(args.b_eval_json))
    a_payload = eval_scores(as_path(args.a_eval_json))
    b_payload = eval_scores(as_path(args.b_eval_json))
    result = {
        "status": "COMMON_SCREEN_TRANSITION",
        "a_label": args.a_label,
        "b_label": args.b_label,
        "a_eval_json": rel(as_path(args.a_eval_json)),
        "b_eval_json": rel(as_path(args.b_eval_json)),
        "a_output_root": rel(as_path(args.a_out_root)),
        "b_output_root": rel(as_path(args.b_out_root)),
        "a_payload_scores": a_payload,
        "b_payload_scores": b_payload,
        "a_computed_scores": a_comp,
        "b_computed_scores": b_comp,
        "comparisons": comps_out,
        "interpretation_note": "This is a fast common-screen analysis. The result separates readout-scale effects from same-scale child training effects only on the research screen; complete official-style evaluation and training provenance are still needed before treating any movement as a BabyLM advance or a general learning principle.",
    }
    out_root = as_path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    out_json = out_root / f"{args.tag}_common_transition.json"
    out_md = out_root / f"{args.tag}_common_transition.md"
    out_json.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_md(out_md, result)
    print(json.dumps({"status": result["status"], "tag": args.tag, "out_json": rel(out_json), "out_md": rel(out_md)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
