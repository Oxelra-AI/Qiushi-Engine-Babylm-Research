#!/usr/bin/env python3
"""research: CPU surface linear baselines for balanced k0/k16 probe.

Trains shallow text classifiers on the exact serialized input used by the neural
probe.  This tests whether word/character surface evidence alone can solve the
balanced state readouts or signed mixed-relation transfer.

No GPU, no model checkpoint loading, no official BabyLM evaluation/upload.
"""
from __future__ import annotations

import json
import math
import re
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

PROJECT = Path("experiments/archive/representation_and_objectives")
DATA_ROOT = PROJECT / "data/information_budget_substrate"
OUT = PROJECT / "data/surface_linear_baselines"
CONDITIONS = ["replace_k00_spread", "replace_k16_spread"]
ARMS = ["aligned_state_bridge", "inverted_state_bridge", "heldheld_only"]
SUITES = ["paired_state_conservation", "cross_template_state_readout", "name_permutation_counterfactual", "mixed_held_seen_orientation", "heldheld_unseen_edge_closure"]


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path) as f:
        for line in f:
            s = line.strip()
            if s:
                rows.append(json.loads(s))
    return rows


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def serialize(row: Dict[str, Any]) -> str:
    if "premise" in row and "hypothesis" in row:
        return str(row["premise"]) + " [SEP] " + str(row["hypothesis"])
    return str(row.get("text", ""))


def anonymize_names_and_objects(text: str) -> str:
    # Keep relation words and function words, replace capitalized names and item nouns after articles.
    text = re.sub(r"\b[A-Z][a-z]+\b", "NAME", text)
    text = re.sub(r"\b(the|a|an)\s+[a-z]+\b", lambda m: m.group(1) + " OBJ", text)
    return text


def mean(xs: Sequence[float]) -> float | None:
    return float(sum(xs) / len(xs)) if xs else None


def safe_key(*xs: Any) -> str:
    return "|".join("None" if x is None else str(x) for x in xs)


def group_rows(rows: Sequence[Dict[str, Any]], fields: Sequence[str]) -> Dict[str, List[Dict[str, Any]]]:
    d: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        d[safe_key(*(r.get(f) for f in fields))].append(r)
    return dict(d)


def summarize(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    if not rows:
        return {"n": 0}
    out = {
        "n": len(rows),
        "acc": mean([float(r["correct"]) for r in rows]),
    }
    if all("pred" in r for r in rows):
        out["pred_true_frac"] = mean([float(r["pred"]) for r in rows])
    if all("label" in r for r in rows):
        out["label_true_frac"] = mean([float(r["label"]) for r in rows])
    return out


def pair_both(true_state: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    by_pair: Dict[str, Dict[str, Dict[str, Any]]] = defaultdict(dict)
    for r in true_state:
        by_pair[str(r.get("pair_id"))][str(r.get("query_kind"))] = r
    pairs: List[Dict[str, Any]] = []
    for pid, d in by_pair.items():
        if "changed" in d and "unchanged" in d:
            c, u = d["changed"], d["unchanged"]
            pairs.append({
                "pair_id": pid,
                "correct": bool(c["correct"] and u["correct"]),
                "initial_pattern": c.get("initial_pattern"),
                "static_slot": c.get("static_slot"),
                "relation": c.get("relation"),
                "voice": c.get("voice"),
            })
    return {
        "all": summarize(pairs),
        "by_pattern": {k: summarize(v) for k, v in sorted(group_rows(pairs, ["initial_pattern"]).items())},
        "by_pattern_static": {k: summarize(v) for k, v in sorted(group_rows(pairs, ["initial_pattern", "static_slot"]).items())},
    }


def summarize_pred_rows(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {"overall": summarize(rows)}
    state = [r for r in rows if r.get("task") == "state_query"]
    comp = [r for r in rows if r.get("task") == "relation_comparison"]
    if state:
        true_state = [r for r in state if bool(r.get("label"))]
        changed = [r for r in true_state if r.get("query_kind") == "changed"]
        unchanged = [r for r in true_state if r.get("query_kind") == "unchanged"]
        out["state"] = {
            "true_changed": summarize(changed),
            "true_unchanged": summarize(unchanged),
            "changed_by_pattern": {k: summarize(v) for k, v in sorted(group_rows(changed, ["initial_pattern"]).items())},
            "changed_by_pattern_static": {k: summarize(v) for k, v in sorted(group_rows(changed, ["initial_pattern", "static_slot"]).items())},
            "pair_both": pair_both(true_state),
        }
    if comp:
        true_comp = [r for r in comp if bool(r.get("label"))]
        out["comparison"] = {
            "all": summarize(comp),
            "true_statement": summarize(true_comp),
            "true_by_relation_pair": {k: summarize(v) for k, v in sorted(group_rows(true_comp, ["relation1", "relation2"]).items())},
        }
    return out


def try_import_sklearn():
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
        from sklearn.linear_model import LogisticRegression, SGDClassifier
        from sklearn.pipeline import make_pipeline
        return TfidfVectorizer, CountVectorizer, LogisticRegression, SGDClassifier, make_pipeline
    except Exception as e:
        raise SystemExit(f"sklearn unavailable: {e}")


def fit_predict_baseline(kind: str, train_rows: Sequence[Dict[str, Any]], eval_rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    y = np.array([int(bool(r["label"])) for r in train_rows], dtype=int)
    x_train_raw = [serialize(r) for r in train_rows]
    x_eval_raw = [serialize(r) for r in eval_rows]
    if kind.endswith("anon"):
        x_train_raw = [anonymize_names_and_objects(x) for x in x_train_raw]
        x_eval_raw = [anonymize_names_and_objects(x) for x in x_eval_raw]
    if kind == "majority_true":
        pred_class = int(y.mean() >= 0.5)
        preds = [pred_class for _ in eval_rows]
    else:
        TfidfVectorizer, CountVectorizer, LogisticRegression, SGDClassifier, make_pipeline = try_import_sklearn()
        if kind.startswith("word"):
            vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1, lowercase=True, token_pattern=r"(?u)\b\w+\b")
        elif kind.startswith("char"):
            vec = TfidfVectorizer(analyzer="char", ngram_range=(3, 6), min_df=1, lowercase=False)
        elif kind.startswith("count_word"):
            vec = CountVectorizer(ngram_range=(1, 3), min_df=1, lowercase=True, token_pattern=r"(?u)\b\w+\b")
        else:
            raise ValueError(kind)
        # Logistic regression is deterministic and robust for the small datasets.
        clf = LogisticRegression(max_iter=2000, solver="liblinear", C=1.0, random_state=0)
        model = make_pipeline(vec, clf)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(x_train_raw, y)
        preds = [int(p) for p in model.predict(x_eval_raw)]
    out: List[Dict[str, Any]] = []
    for r, p in zip(eval_rows, preds):
        rr = dict(r)
        rr["pred"] = bool(p)
        rr["correct"] = bool(p) == bool(r["label"])
        rr["baseline_kind"] = kind
        out.append(rr)
    return out


def train_rows(condition: str, arm: str) -> List[Dict[str, Any]]:
    root = DATA_ROOT / condition
    rows = load_jsonl(root / "common_seen_train.jsonl")
    rows += load_jsonl(root / "arms" / arm / "train_supervised.jsonl")
    return rows


def eval_rows(condition: str, suite: str) -> List[Dict[str, Any]]:
    return load_jsonl(DATA_ROOT / condition / "eval" / f"{suite}.jsonl")


def run() -> Dict[str, Any]:
    kinds = ["majority_true", "word_tfidf", "word_tfidf_anon", "char_tfidf", "char_tfidf_anon", "count_word"]
    report: Dict[str, Any] = {"conditions": CONDITIONS, "arms": ARMS, "suites": SUITES, "baselines": kinds, "results": {}}
    OUT.mkdir(parents=True, exist_ok=True)
    per_rows_path = OUT / "surface_baseline_per_row_predictions.jsonl"
    if per_rows_path.exists():
        per_rows_path.unlink()
    with per_rows_path.open("w", encoding="utf-8") as fout:
        for condition in CONDITIONS:
            report["results"].setdefault(condition, {})
            for arm in ARMS:
                tr = train_rows(condition, arm)
                report["results"][condition].setdefault(arm, {})
                for kind in kinds:
                    report["results"][condition][arm].setdefault(kind, {})
                    for suite in SUITES:
                        ev = eval_rows(condition, suite)
                        pr = fit_predict_baseline(kind, tr, ev)
                        for rr in pr:
                            fout.write(json.dumps({
                                "condition": condition,
                                "arm": arm,
                                "suite": suite,
                                "baseline_kind": kind,
                                "row_id": rr.get("row_id"),
                                "pair_id": rr.get("pair_id"),
                                "task": rr.get("task"),
                                "label": bool(rr.get("label")),
                                "pred": bool(rr.get("pred")),
                                "correct": bool(rr.get("correct")),
                                "query_kind": rr.get("query_kind"),
                                "initial_pattern": rr.get("initial_pattern"),
                                "static_slot": rr.get("static_slot"),
                                "relation": rr.get("relation"),
                                "voice": rr.get("voice"),
                                "relation1": rr.get("relation1"),
                                "relation2": rr.get("relation2"),
                            }, sort_keys=True) + "\n")
                        report["results"][condition][arm][kind][suite] = summarize_pred_rows(pr)
    report["per_row_predictions"] = str(per_rows_path)
    return report


def metric_get(d: Dict[str, Any], path: Sequence[str]) -> Any:
    x: Any = d
    for p in path:
        if not isinstance(x, dict) or p not in x:
            return None
        x = x[p]
    return x


def summary_text(report: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append("# research CPU surface linear baselines")
    lines.append("")
    lines.append("Shallow classifiers trained on the exact serialized text of the balanced k0/k16 neural probe.  These baselines are used to interpret whether a learned DeBERTa effect could be a simple surface pattern rather than event-role structure.")
    lines.append("")
    lines.append("## Central metrics")
    lines.append("")
    lines.append("| condition | arm | baseline | psc changed same | psc changed opposite | psc pair-both same | psc pair-both opposite | mixed true |")
    lines.append("|---|---|---|---:|---:|---:|---:|---:|")
    for condition in CONDITIONS:
        for arm in ARMS:
            for kind in report["baselines"]:
                d = report["results"][condition][arm][kind]
                def f(path: Sequence[str]) -> str:
                    x = metric_get(d, path)
                    return "n/a" if x is None else f"{float(x):.3f}"
                lines.append(
                    f"| {condition} | {arm} | {kind} | "
                    f"{f(['paired_state_conservation','state','changed_by_pattern','same','acc'])} | "
                    f"{f(['paired_state_conservation','state','changed_by_pattern','opposite','acc'])} | "
                    f"{f(['paired_state_conservation','state','pair_both','by_pattern','same','acc'])} | "
                    f"{f(['paired_state_conservation','state','pair_both','by_pattern','opposite','acc'])} | "
                    f"{f(['mixed_held_seen_orientation','comparison','true_statement','acc'])} |"
                )
    lines.append("")
    lines.append("## Reading")
    lines.append("")
    lines.append("- A surface baseline that reaches high same-initial changed accuracy and pair-both in k16 would weaken a neural interpretation as role learning, because the text form itself would provide a shallow cue.")
    lines.append("- A surface baseline that remains near the anti-copy/copy-initial floors while DeBERTa succeeds would support that pretrained representations or nonlinear fine-tuning are using more than ordinary lexical n-grams.")
    lines.append("- Mixed true-statement accuracy must be interpreted with aligned versus inverted comparison; high values in both or unstable values are not signed coordinate evidence.")
    lines.append("")
    lines.append("## Files")
    lines.append(f"- full JSON: `{OUT / 'surface_linear_baselines_report.json'}`")
    lines.append(f"- per-row: `{report['per_row_predictions']}`")
    return "\n".join(lines) + "\n"


def main() -> None:
    report = run()
    write_json(OUT / "surface_linear_baselines_report.json", report)
    ((OUT.parents[4] / 'research/documents/representation_and_objectives/data/surface_linear_baselines/surface_linear_baselines_summary.md')).write_text(summary_text(report), encoding="utf-8")
    print(json.dumps({
        "status": "SURFACE_LINEAR_BASELINES_COMPLETE",
        "summary": str((OUT.parents[4] / 'research/documents/representation_and_objectives/data/surface_linear_baselines/surface_linear_baselines_summary.md')),
        "report": str(OUT / "surface_linear_baselines_report.json"),
        "per_row_predictions": report["per_row_predictions"],
        "no_gpu_model_checkpoint_official_eval_upload_or_leaderboard": True,
    }, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
