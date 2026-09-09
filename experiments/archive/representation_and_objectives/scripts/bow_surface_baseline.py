#!/usr/bin/env python3
"""research raw-text BoW/LogReg surface baseline.

A transparent baseline complementary to order-feature checks.  It asks whether a
plain lexical surface learner trained on each arm can transfer to the repaired
mixed/state readouts.  Names and objects are disjoint between train/eval; an
optional normalized view replaces them with placeholders.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

WORKSPACE = _public_path('experiments/archive/representation_and_objectives')
PROJECT_ROOT = _public_path('.')
SUBSTRATE_DEFAULT = _public_path('experiments/archive/representation_and_objectives/data/equivariant_symmetry_substrate')
OUT_DEFAULT = _public_path('experiments/archive/representation_and_objectives/data/bow_surface_baseline')
ARMS = ["exposure_only", "heldheld_only", "aligned_state_bridge", "inverted_state_bridge", "neutral_decoupled", "mixed_event_bridge"]
EVAL_SUITES = ["heldheld_unseen_edge_closure", "mixed_held_seen_orientation", "paired_state_conservation", "cross_template_state_readout", "name_permutation_counterfactual"]
TRAIN_NAMES = ["Mira", "Noel", "Iris", "Omar", "Lena", "Pavel", "Rina", "Tomas", "Nia", "Felix", "Ava", "Jonas", "Keira", "Milo", "Sara", "Theo"]
EVAL_NAMES = ["Zara", "Eli", "Nora", "Caleb", "Vera", "Hugo", "Maya", "Luca", "June", "Arun", "Leah", "Mateo", "Tara", "Simon", "Yara", "Ben"]
TRAIN_CHANGED = ["lantern", "parcel", "compass", "violin", "notebook", "teapot", "camera", "blanket", "basket", "tablet", "wallet", "sketchbook", "helmet", "map", "jacket", "medal"]
EVAL_CHANGED = ["goblet", "key", "telescope", "bracelet", "vase", "satchel", "radio", "painting", "flute", "badge", "ticket", "scarf", "hammer", "shell", "drum", "booklet"]
TRAIN_STATIC = ["mug", "rope", "spoon", "bottle", "pillow", "clock", "broom", "coin", "brush", "kite", "stone", "candle", "plate", "glove", "bell", "saddle"]
EVAL_STATIC = ["ruler", "mirror", "bucket", "comb", "ladder", "needle", "pencil", "towel", "anchor", "fork", "whistle", "basketball", "bowl", "rug", "lamp", "button"]


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    if not path.exists():
        return rows
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def row_text(r: Dict[str, Any]) -> str:
    if r.get("task") == "state_query":
        return r.get("premise", "") + " [SEP] " + r.get("hypothesis", "")
    return r.get("text", "")


def normalize_text(s: str) -> str:
    # Replace known train/eval names and objects with role-neutral placeholders.
    for n in sorted(TRAIN_NAMES + EVAL_NAMES, key=len, reverse=True):
        s = re.sub(rf"\b{re.escape(n)}\b", "NAME", s)
    for o in sorted(TRAIN_CHANGED + EVAL_CHANGED, key=len, reverse=True):
        s = re.sub(rf"\b{re.escape(o)}\b", "CHANGEDOBJ", s)
    for o in sorted(TRAIN_STATIC + EVAL_STATIC, key=len, reverse=True):
        s = re.sub(rf"\b{re.escape(o)}\b", "STATICOBJ", s)
    return s


def fit_eval(train_rows: Sequence[Dict[str, Any]], eval_rows: Sequence[Dict[str, Any]], task: str | None, normalize: bool, ngram_range=(1, 2)) -> Dict[str, Any]:
    labeled_train = [r for r in train_rows if "label" in r and (task is None or r.get("task") == task)]
    labeled_eval = [r for r in eval_rows if "label" in r and (task is None or r.get("task") == task)]
    if not labeled_eval:
        return {"n_eval": 0}
    y_eval = np.array([int(bool(r["label"])) for r in labeled_eval])
    if not labeled_train or len(set(int(bool(r["label"])) for r in labeled_train)) < 2:
        pred = np.zeros_like(y_eval)
        return summarize(labeled_eval, pred, y_eval, note="default_false_no_train")
    y_train = np.array([int(bool(r["label"])) for r in labeled_train])
    x_train = [normalize_text(row_text(r)) if normalize else row_text(r) for r in labeled_train]
    x_eval = [normalize_text(row_text(r)) if normalize else row_text(r) for r in labeled_eval]
    try:
        from sklearn.feature_extraction.text import CountVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        clf = make_pipeline(
            CountVectorizer(lowercase=True, ngram_range=ngram_range, min_df=1),
            LogisticRegression(max_iter=2000, solver="liblinear", C=1.0, random_state=0),
        )
        clf.fit(x_train, y_train)
        pred_train = clf.predict(x_train)
        pred = clf.predict(x_eval)
        out = summarize(labeled_eval, pred, y_eval, note="sklearn_logreg")
        out["train_acc"] = float((pred_train == y_train).mean())
        out["n_train"] = len(y_train)
        return out
    except Exception as e:
        # Fallback: majority by exact normalized text, mostly a sanity path.
        counts: Dict[str, Counter] = defaultdict(Counter)
        for x, y in zip(x_train, y_train):
            counts[x][int(y)] += 1
        default = int(np.mean(y_train) >= 0.5)
        pred = np.array([1 if counts.get(x, Counter())[1] >= counts.get(x, Counter())[0] and x in counts else default for x in x_eval])
        out = summarize(labeled_eval, pred, y_eval, note=f"fallback_exact:{type(e).__name__}")
        out["train_acc"] = None
        out["n_train"] = len(y_train)
        return out


def summarize(rows: Sequence[Dict[str, Any]], pred: Sequence[int], y: Sequence[int], note: str) -> Dict[str, Any]:
    pred = np.array(pred, dtype=int)
    y = np.array(y, dtype=int)
    out: Dict[str, Any] = {
        "n_eval": int(len(y)),
        "accuracy": float((pred == y).mean()) if len(y) else None,
        "pred_true_frac": float(pred.mean()) if len(pred) else None,
        "label_true_frac": float(y.mean()) if len(y) else None,
        "note": note,
    }
    for q in ["changed", "unchanged"]:
        idx = [i for i, r in enumerate(rows) if r.get("query_kind") == q]
        if idx:
            out[f"acc_{q}"] = float((pred[idx] == y[idx]).mean())
            out[f"pred_true_frac_{q}"] = float(pred[idx].mean())
    for dep in sorted(set(r.get("orientation_dependency") for r in rows)):
        if dep is None:
            continue
        idx = [i for i, r in enumerate(rows) if r.get("orientation_dependency") == dep]
        if idx:
            out[f"acc_dep_{dep}"] = float((pred[idx] == y[idx]).mean())
    if any(r.get("task") == "state_query" for r in rows):
        by_pair: Dict[str, List[int]] = defaultdict(list)
        for i, r in enumerate(rows):
            if r.get("task") == "state_query":
                by_pair[str(r.get("pair_id"))].append(i)
        c = Counter()
        for pid, idxs in by_pair.items():
            ch = [i for i in idxs if rows[i].get("query_kind") == "changed"]
            un = [i for i in idxs if rows[i].get("query_kind") == "unchanged"]
            cok = bool(ch) and all(pred[i] == y[i] for i in ch)
            uok = bool(un) and all(pred[i] == y[i] for i in un)
            c["both_correct" if cok and uok else "changed_only" if cok else "unchanged_only" if uok else "neither"] += 1
        denom = max(sum(c.values()), 1)
        out.update({f"pair_{k}": float(c[k] / denom) for k in ["both_correct", "changed_only", "unchanged_only", "neither"]})
    return out


def fmt(x: Any) -> str:
    if x is None:
        return "nan"
    if isinstance(x, float):
        return f"{x:.3f}"
    return str(x)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--substrate", type=Path, default=SUBSTRATE_DEFAULT)
    ap.add_argument("--out", type=Path, default=OUT_DEFAULT)
    ap.add_argument("--include-common-seen", action="store_true")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    common = load_jsonl(args.substrate / "common_seen_train.jsonl") if args.include_common_seen else []
    arms = {a: load_jsonl(args.substrate / "arms" / a / "train_supervised.jsonl") for a in ARMS}
    evals = {s: load_jsonl(args.substrate / "eval" / f"{s}.jsonl") for s in EVAL_SUITES}
    records: List[Dict[str, Any]] = []
    for arm, rows in arms.items():
        train = rows + common
        for normalize in [False, True]:
            for task in ["relation_comparison", "state_query"]:
                for suite, erows in evals.items():
                    rec = {
                        "arm": arm,
                        "normalize": normalize,
                        "task_fit": task,
                        "suite": suite,
                    }
                    rec.update(fit_eval(train, erows, task, normalize))
                    records.append(rec)

    suffix = "with_common" if args.include_common_seen else "no_common"
    jp = args.out / f"bow_surface_{suffix}.json"
    jp.write_text(json.dumps(records, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    md = args.out / f"bow_surface_{suffix}_summary.md"
    lines = [f"# research BoW surface baseline ({suffix})", ""]
    lines.append("## Critical relation-comparison transfer")
    lines.append("")
    lines.append("| arm | normalized | train acc | hh eval | mixed eval |")
    lines.append("|---|---:|---:|---:|---:|")
    for arm in ARMS:
        for norm in [False, True]:
            recs = [r for r in records if r["arm"] == arm and r["normalize"] == norm and r["task_fit"] == "relation_comparison"]
            hh = next((r for r in recs if r["suite"] == "heldheld_unseen_edge_closure"), None)
            mx = next((r for r in recs if r["suite"] == "mixed_held_seen_orientation"), None)
            if hh and mx:
                lines.append(f"| {arm} | {norm} | {fmt(mx.get('train_acc'))} | {fmt(hh.get('accuracy'))} | {fmt(mx.get('accuracy'))} |")
    lines.append("")
    lines.append("## Critical state-query transfer")
    lines.append("")
    lines.append("| arm | normalized | train acc | paired changed | paired unchanged | paired both | cross changed |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for arm in ARMS:
        for norm in [False, True]:
            recs = [r for r in records if r["arm"] == arm and r["normalize"] == norm and r["task_fit"] == "state_query"]
            ps = next((r for r in recs if r["suite"] == "paired_state_conservation"), None)
            ct = next((r for r in recs if r["suite"] == "cross_template_state_readout"), None)
            if ps and ct:
                lines.append(f"| {arm} | {norm} | {fmt(ps.get('train_acc'))} | {fmt(ps.get('acc_changed'))} | {fmt(ps.get('acc_unchanged'))} | {fmt(ps.get('pair_both_correct'))} | {fmt(ct.get('acc_changed'))} |")
    lines.append("")
    lines.append("A transparent BoW model can legitimately solve unchanged rows by textual matching of the static fact. The mechanism-relevant check is whether it can orient held changed-state or mixed held-seen rows without aligned/inverted evidence.")
    lines.append("")
    lines.append(f"- results_json: `{jp.relative_to(PROJECT_ROOT)}`")
    md.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"status": "BOW_SURFACE_BASELINE_COMPLETE", "summary_md": str(md.relative_to(PROJECT_ROOT)), "results_json": str(jp.relative_to(PROJECT_ROOT)), "include_common_seen": args.include_common_seen}, indent=2), flush=True)


if __name__ == "__main__":
    main()
