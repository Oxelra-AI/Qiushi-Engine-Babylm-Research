#!/usr/bin/env python3
"""research: sequence-aware shortcut stress tests and teacher realization verification.

Scientific purpose
------------------
research built source-attested paired-world role-flip families whose grouped
bag-of-words baseline was near chance. This script applies a stronger pressure
before any student training:
  1. sequence-aware text baselines under family, template, sport, and perturbation
     splits;
  2. deterministic adversarial role/order perturbations;
  3. teacher-model verification that the score-ablated realizations express the
     intended event worlds.

It performs no BabyLM training, no student training, no official evaluation, no
upload, and no submission.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import os
import random
import re
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

STUDY = Path("experiments/archive/representation_and_objectives")
ROOT = STUDY
IN_DIR = ROOT / "data/paired_world_pilot"
OUT_DIR = ROOT / "data/paired_world_sequence_teacher"
PROMPT_DIR = ROOT / "training/data/paired_world_teacher_verify"
SEED = 252031

TRAIN_FILE = IN_DIR / "families_train.jsonl"
HELD_FILE = IN_DIR / "families_held.jsonl"

TEACHERS = ["qwen3.5-9b", "llama3.1-8b-instruct"]

# Clean unique template list mirroring research's final realizations. Used only
# for adversarial re-templating; original contexts remain the source artifacts.
TEMPLATES: dict[int, tuple[str, str, str]] = {
    1:  ("On {D}, {W} defeated {L} in the {R} of {T}.", "w", "train"),
    2:  ("{W} beat {L} at {T} during the {R} on {D}.", "w", "train"),
    3:  ("{W} won against {L} in the {R} of {T} on {D}.", "w", "train"),
    4:  ("{W} overcame {L} in the {R} at {T} on {D}.", "w", "train"),
    5:  ("{W} proved too strong for {L} in the {R} at {T} on {D}.", "w", "train"),
    6:  ("{L} lost to {W} in the {R} of {T} on {D}.", "l", "train"),
    7:  ("{L} fell to {W} at {T} in the {R} on {D}.", "l", "train"),
    8:  ("{L} was defeated by {W} in the {R} of {T} on {D}.", "l", "train"),
    9:  ("{L} was beaten by {W} at {T} during the {R} on {D}.", "l", "train"),
    10: ("In the {R} at {T} on {D}, {L} was unable to overcome {W}.", "l", "train"),
    11: ("At {T} on {D}, the {R} saw {W} triumph over {L}.", "m", "train"),
    12: ("{W} emerged victorious over {L} in the {R} at {T} on {D}.", "m", "train"),
    13: ("During the {R} at {T} on {D}, {W} prevailed against {L}.", "m", "train"),
    14: ("It was {W} who came out on top against {L} in the {R} of {T} on {D}.", "m", "train"),
    15: ("The {R} of {T} on {D} ended with {W} victorious over {L}.", "m", "train"),
    16: ("{W} edged out {L} in the {R} of {T} on {D}.", "w", "held"),
    17: ("{L} succumbed to {W} at {T} during the {R} on {D}.", "l", "held"),
    18: ("A {R} contest at {T} on {D} resulted in a victory for {W} over {L}.", "m", "held"),
    19: ("{W} claimed the win against {L} in the {R} of {T} on {D}.", "w", "held"),
    20: ("At {T} on {D}, {L} went down to {W} in the {R}.", "l", "held"),
}
ALT_BY_FIRST = {
    "w": [1, 2, 3, 4, 5, 16, 19],
    "l": [6, 7, 8, 9, 10, 17, 20],
    "m": [11, 12, 13, 14, 15, 18],
}


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_families() -> list[dict[str, Any]]:
    fams = []
    for split, path in [("train", TRAIN_FILE), ("held", HELD_FILE)]:
        for r in read_jsonl(path):
            r = dict(r)
            r["family_split"] = split
            fams.append(r)
    return fams


def replace_names(text: str, a: str, b: str, a_tok: str = " ENTITY_A ", b_tok: str = " ENTITY_B ") -> str:
    # Protect against overlapping substrings by length-descending replacement.
    out = str(text)
    pairs = sorted([(a, "@@A@@"), (b, "@@B@@")], key=lambda x: len(x[0]), reverse=True)
    for name, marker in pairs:
        if name:
            out = out.replace(name, marker)
    out = out.replace("@@A@@", a_tok.strip()).replace("@@B@@", b_tok.strip())
    return re.sub(r"\s+", " ", out).strip()


def swap_names(text: str, a: str, b: str) -> str:
    out = str(text)
    pairs = sorted([(a, "@@A@@"), (b, "@@B@@")], key=lambda x: len(x[0]), reverse=True)
    for name, marker in pairs:
        if name:
            out = out.replace(name, marker)
    out = out.replace("@@A@@", b).replace("@@B@@", a)
    return re.sub(r"\s+", " ", out).strip()


def name_positions(text: str, a: str, b: str) -> dict[str, Any]:
    lower = text.lower()
    aa = a.lower()
    bb = b.lower()
    pa = lower.find(aa)
    pb = lower.find(bb)
    if pa < 0 and pb < 0:
        order = "neither"
    elif pa < 0:
        order = "B_only"
    elif pb < 0:
        order = "A_only"
    elif pa < pb:
        order = "A_before_B"
    else:
        order = "B_before_A"
    return {"pos_a": pa, "pos_b": pb, "order": order, "distance": None if pa < 0 or pb < 0 else abs(pa - pb)}


def event_text(ctx: dict[str, Any], template_id: int) -> str:
    pat, _, _ = TEMPLATES[template_id]
    ev = ctx.get("event_raw", {})
    text = pat.format(
        W=ctx["winner_name"],
        L=ctx["loser_name"],
        R=ctx.get("round_normalized") or "match",
        T=ev.get("tournament") or "an event",
        D=ctx.get("date_formatted") or ev.get("date_raw") or "an unrecorded date",
    )
    return re.sub(r"\s+", " ", text).strip()


def deterministic_label_for_hyp(fam: dict[str, Any], cx: str, hyp_kind: str, hyp_direction: str) -> bool:
    ctx = fam["context1"] if cx == "c1" else fam["context2"]
    winner_label = ctx["winner_label"]
    if hyp_kind == "defeated":
        winner_in_hyp = "A" if hyp_direction == "AB" else "B"
        return winner_label == winner_in_hyp
    if hyp_kind in {"lost_to", "was_defeated_by"}:
        loser_in_hyp = "A" if hyp_direction == "AB" else "B"
        return winner_label != loser_in_hyp
    raise ValueError(hyp_kind)


def make_hyp(fam: dict[str, Any], hyp_kind: str, direction: str) -> str:
    a, b = fam["participant_a"], fam["participant_b"]
    subj, obj = (a, b) if direction == "AB" else (b, a)
    if hyp_kind == "defeated":
        return f"{subj} defeated {obj}."
    if hyp_kind == "lost_to":
        return f"{subj} lost to {obj}."
    if hyp_kind == "was_defeated_by":
        return f"{subj} was defeated by {obj}."
    raise ValueError(hyp_kind)


def build_examples(fams: list[dict[str, Any]], variant: str = "original_ablated") -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    rng = random.Random(SEED + 17)
    for fam in fams:
        a, b = fam["participant_a"], fam["participant_b"]
        for cx in ["c1", "c2"]:
            ctx = fam["context1"] if cx == "c1" else fam["context2"]
            base_text = ctx["text_score_ablated"]
            score_text = ctx["text_score_visible"]
            text = base_text
            text_variant = variant
            gold_transform = lambda x: x
            hyp_kinds = ["defeated"]
            alt_template_id = None
            if variant == "original_score_visible":
                text = score_text
            elif variant == "name_swapped_context":
                text = swap_names(base_text, a, b)
                gold_transform = lambda x: not x
            elif variant == "hyp_lost_to":
                hyp_kinds = ["lost_to"]
            elif variant == "hyp_passive":
                hyp_kinds = ["was_defeated_by"]
            elif variant == "retemplate_opposite_order":
                current_tid = int(ctx.get("template_id"))
                current_first = TEMPLATES.get(current_tid, ("", ctx.get("template_first_mention", "m"), ""))[1]
                target_first = "l" if current_first == "w" else "w" if current_first == "l" else rng.choice(["w", "l"])
                candidates = [t for t in ALT_BY_FIRST[target_first] if t != current_tid]
                alt_template_id = rng.choice(candidates)
                text = event_text(ctx, alt_template_id)
            elif variant != "original_ablated":
                raise ValueError(variant)
            for hk in hyp_kinds:
                for direction in ["AB", "BA"]:
                    raw_gold = deterministic_label_for_hyp(fam, cx, hk, direction)
                    gold_bool = bool(gold_transform(raw_gold))
                    hyp = make_hyp(fam, hk, direction)
                    pos = name_positions(text, a, b)
                    rows.append({
                        "id": f"{fam['family_id']}_{cx}_{variant}_{hk}_{direction}",
                        "family_id": fam["family_id"],
                        "family_split": fam.get("family_split"),
                        "sport": fam["sport"],
                        "source_type": fam.get("source_type"),
                        "cx": cx,
                        "context_template_id": int(ctx.get("template_id")),
                        "context_template_split": ctx.get("template_split"),
                        "context_template_first_mention": ctx.get("template_first_mention"),
                        "alt_template_id": alt_template_id,
                        "variant": text_variant,
                        "hyp_kind": hk,
                        "hyp_direction": direction,
                        "text": text,
                        "hyp": hyp,
                        "gold_label": "ENTAILED" if gold_bool else "NOT_ENTAILED",
                        "y": 1 if gold_bool else 0,
                        "participant_a": a,
                        "participant_b": b,
                        "canonical_text": replace_names(text, a, b),
                        "canonical_hyp": replace_names(hyp, a, b),
                        "ctx_name_order": pos["order"],
                        "ctx_name_distance": pos["distance"],
                    })
    return rows


def label_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    c = collections.Counter(r["y"] for r in rows)
    return {"n": len(rows), "entailed": c.get(1, 0), "not_entailed": c.get(0, 0), "positive_rate": c.get(1, 0) / max(1, len(rows))}


def acc(y_true: list[int], y_pred: list[int]) -> float:
    return sum(int(a == b) for a, b in zip(y_true, y_pred)) / max(1, len(y_true))


def mean_sd(vals: list[float]) -> dict[str, Any]:
    if not vals:
        return {"n": 0}
    return {"n": len(vals), "mean": statistics.fmean(vals), "sd": statistics.pstdev(vals), "values": vals}


def try_sklearn_imports():
    from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
    from sklearn.feature_extraction import DictVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold, StratifiedKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import FunctionTransformer
    return TfidfVectorizer, CountVectorizer, DictVectorizer, LogisticRegression, GroupKFold, StratifiedKFold, make_pipeline, FunctionTransformer


def text_for_view(row: dict[str, Any], view: str) -> str:
    if view == "raw_word12":
        return row["text"] + " [SEP] " + row["hyp"]
    if view == "canonical_word12":
        return row["canonical_text"] + " [SEP] " + row["canonical_hyp"]
    if view == "raw_char35":
        return row["text"] + " [SEP] " + row["hyp"]
    if view == "canonical_char35":
        return row["canonical_text"] + " [SEP] " + row["canonical_hyp"]
    if view == "symbolic_order_text":
        return (
            f"SPORT_{row['sport']} TEMPLATE_{row['context_template_id']} "
            f"SPLIT_{row['context_template_split']} FIRST_{row['context_template_first_mention']} "
            f"ORDER_{row['ctx_name_order']} HYP_{row['hyp_kind']}_{row['hyp_direction']} CX_{row['cx']}"
        )
    raise ValueError(view)


def fit_text_model(train_rows: list[dict[str, Any]], view: str):
    TfidfVectorizer, CountVectorizer, DictVectorizer, LogisticRegression, GroupKFold, StratifiedKFold, make_pipeline, FunctionTransformer = try_sklearn_imports()
    texts = [text_for_view(r, view) for r in train_rows]
    y = [r["y"] for r in train_rows]
    if view.endswith("char35"):
        vec = TfidfVectorizer(analyzer="char", ngram_range=(3, 5), min_df=1, max_features=200000, lowercase=True)
    elif view == "symbolic_order_text":
        vec = CountVectorizer(token_pattern=r"[^ ]+")
    else:
        vec = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=1, max_features=200000, lowercase=True)
    clf = LogisticRegression(max_iter=1000, solver="liblinear", random_state=SEED)
    model = make_pipeline(vec, clf)
    model.fit(texts, y)
    return model


def predict_text_model(model: Any, rows: list[dict[str, Any]], view: str) -> list[int]:
    if not rows:
        return []
    return [int(x) for x in model.predict([text_for_view(r, view) for r in rows])]


def group_cv(rows: list[dict[str, Any]], view: str, n_splits: int = 5) -> dict[str, Any]:
    _, _, _, _, GroupKFold, *_ = try_sklearn_imports()
    groups = [r["family_id"] for r in rows]
    y = [r["y"] for r in rows]
    folds = []
    gkf = GroupKFold(n_splits=n_splits)
    idxs = list(range(len(rows)))
    for tr, te in gkf.split(idxs, y, groups):
        train_rows = [rows[i] for i in tr]
        test_rows = [rows[i] for i in te]
        model = fit_text_model(train_rows, view)
        pred = predict_text_model(model, test_rows, view)
        folds.append(acc([r["y"] for r in test_rows], pred))
    out = mean_sd(folds)
    out["split"] = "GroupKFold_by_family"
    return out


def train_eval(train_rows: list[dict[str, Any]], test_rows: list[dict[str, Any]], view: str) -> dict[str, Any]:
    model = fit_text_model(train_rows, view)
    pred = predict_text_model(model, test_rows, view)
    return {"n_train": len(train_rows), "n_test": len(test_rows), "accuracy": acc([r["y"] for r in test_rows], pred)}


def oracle_template_order_rule(rows: list[dict[str, Any]]) -> dict[str, Any]:
    # A transparent hand parser using research metadata: whether a template places
    # winner or loser before the other participant, plus which participant appears
    # first in the context and which participant is the subject of the hypothesis.
    # This is intentionally not a learning result; it quantifies the residual
    # templated-role route a student might exploit if the corpus is too narrow.
    preds = []
    usable = 0
    for r in rows:
        first_role = r.get("context_template_first_mention")
        order = r.get("ctx_name_order")
        if first_role not in {"w", "l"} or order not in {"A_before_B", "B_before_A"} or r["hyp_kind"] != "defeated":
            preds.append(0)
            continue
        first_entity = "A" if order == "A_before_B" else "B"
        winner = first_entity if first_role == "w" else ("B" if first_entity == "A" else "A")
        hyp_winner = "A" if r["hyp_direction"] == "AB" else "B"
        preds.append(1 if winner == hyp_winner else 0)
        usable += 1
    return {"n": len(rows), "usable_by_simple_rule": usable, "coverage": usable / max(1, len(rows)), "accuracy_treating_mixed_as_negative": acc([r["y"] for r in rows], preds), "accuracy_on_usable": acc([r["y"] for r in rows if r.get("context_template_first_mention") in {"w", "l"}], [p for r, p in zip(rows, preds) if r.get("context_template_first_mention") in {"w", "l"}])}


def run_sequence_stress(_: argparse.Namespace) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fams = load_families()
    original = build_examples(fams, "original_ablated")
    visible = build_examples(fams, "original_score_visible")
    swapped = build_examples(fams, "name_swapped_context")
    hyp_lost = build_examples(fams, "hyp_lost_to")
    hyp_passive = build_examples(fams, "hyp_passive")
    retemplated = build_examples(fams, "retemplate_opposite_order")
    write_jsonl(OUT_DIR / "sequence_examples_original_ablated.jsonl", original)
    write_jsonl(OUT_DIR / "sequence_examples_perturbations.jsonl", swapped + hyp_lost + hyp_passive + retemplated)

    train_orig = [r for r in original if r["family_split"] == "train"]
    held_orig = [r for r in original if r["family_split"] == "held"]
    held_train_templates = [r for r in held_orig if r["context_template_split"] == "train"]
    held_held_templates = [r for r in held_orig if r["context_template_split"] == "held"]
    held_swapped = [r for r in swapped if r["family_split"] == "held"]
    held_lost = [r for r in hyp_lost if r["family_split"] == "held"]
    held_passive = [r for r in hyp_passive if r["family_split"] == "held"]
    held_retemplated = [r for r in retemplated if r["family_split"] == "held"]
    views = ["raw_word12", "raw_char35", "canonical_word12", "canonical_char35", "symbolic_order_text"]
    results: dict[str, Any] = {}
    for view in views:
        view_res: dict[str, Any] = {}
        view_res["group_family_cv_original"] = group_cv(original, view)
        view_res["train_to_all_held_original"] = train_eval(train_orig, held_orig, view)
        view_res["train_to_held_train_templates"] = train_eval(train_orig, held_train_templates, view)
        view_res["train_to_held_held_templates"] = train_eval(train_orig, held_held_templates, view)
        view_res["train_to_held_name_swapped_context"] = train_eval(train_orig, held_swapped, view)
        view_res["train_to_held_hypothesis_lost_to"] = train_eval(train_orig, held_lost, view)
        view_res["train_to_held_hypothesis_passive"] = train_eval(train_orig, held_passive, view)
        view_res["train_to_held_retemplate_opposite_order"] = train_eval(train_orig, held_retemplated, view)
        # Leave-one-sport-out on original ablated examples.
        sport_res = {}
        for sport in sorted({r["sport"] for r in original}):
            tr = [r for r in original if r["sport"] != sport]
            te = [r for r in original if r["sport"] == sport]
            sport_res[sport] = train_eval(tr, te, view)
        view_res["leave_one_sport_out_original"] = sport_res
        results[view] = view_res

    by_variant = {name: label_stats(rows) for name, rows in [
        ("original_ablated", original),
        ("original_score_visible", visible),
        ("name_swapped_context", swapped),
        ("hyp_lost_to", hyp_lost),
        ("hyp_passive", hyp_passive),
        ("retemplate_opposite_order", retemplated),
    ]}
    context_order_counts = collections.Counter(r["ctx_name_order"] for r in original)
    template_counts = collections.Counter(str(r["context_template_id"]) for r in original)
    split_counts = collections.Counter(f"{r['family_split']}::{r['context_template_split']}" for r in original)

    summary = {
        "status": "SEQUENCE_SHORTCUT_STRESS_COMPLETED",
        "created_utc": now(),
        "input_files": {
            "families_train": str(TRAIN_FILE),
            "families_train_sha256": sha256_file(TRAIN_FILE),
            "families_held": str(HELD_FILE),
            "families_held_sha256": sha256_file(HELD_FILE),
        },
        "families": len(fams),
        "examples_by_variant": by_variant,
        "template_counts_original": dict(sorted(template_counts.items(), key=lambda kv: int(kv[0]))),
        "split_counts_original": dict(sorted(split_counts.items())),
        "context_name_order_counts_original": dict(context_order_counts),
        "oracle_template_order_rule_original": oracle_template_order_rule(original),
        "model_results": results,
        "interpretation": {
            "bow_step251": "Grouped linear bag-of-words near chance remains useful but does not exclude sequence/template/order routes.",
            "what_this_tests": "Whether raw or canonicalized sequence classifiers can use template identity, token order, participant position, or context-hypothesis alignment on held families/templates/sports and perturbations.",
            "scientific_boundary": "High accuracy by these baselines would show the current sports-predicate corpus can be solved by narrow templated role parsing. That would not be a data-efficient learning principle and would require stronger cross-template/cross-predicate and event-to-state construction before student training.",
        },
        "no_babylm_training_eval_upload_submission": True,
    }
    (OUT_DIR / "sequence_shortcut_stress_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    md = [
        "# research paired-world sequence shortcut stress", "",
        "## Purpose", "",
        "The research grouped bag-of-words result is not enough to say the corpus forces general role assignment. This stress test asks whether sequence features, participant order, template identity, and context--hypothesis alignment can solve the current score-ablated paired-world task before any student training.", "",
        "## Data", "",
        f"- families: {len(fams)}", 
        f"- original score-ablated NLI rows: {len(original)}", 
        f"- held original rows: {len(held_orig)}; held rows using held templates: {len(held_held_templates)}", "",
        "## Transparent template/order parser", "",
        f"- coverage on winner-first/loser-first templates: {summary['oracle_template_order_rule_original']['coverage']:.3f}",
        f"- accuracy on covered rows: {summary['oracle_template_order_rule_original']['accuracy_on_usable']:.3f}", "",
        "## Learned baselines", "",
    ]
    for view, view_res in results.items():
        md += [f"### {view}", "", "| split | accuracy | n_test |", "|---|---:|---:|"]
        for key in ["train_to_all_held_original", "train_to_held_train_templates", "train_to_held_held_templates", "train_to_held_name_swapped_context", "train_to_held_hypothesis_lost_to", "train_to_held_hypothesis_passive", "train_to_held_retemplate_opposite_order"]:
            r = view_res[key]
            md.append(f"| {key} | {r['accuracy']:.3f} | {r['n_test']} |")
        cv = view_res["group_family_cv_original"]
        md.append(f"| group-family CV original mean | {cv['mean']:.3f} +/- {cv['sd']:.3f} | {len(original)} |")
        md.append("")
    md += [
        "## Scientific meaning", "",
        "A high score for canonicalized sequence or symbolic-order baselines means the current sports outcome task is still a narrow role-parsing substrate. It can remain useful for language-realization verification and for designing harder transfer tests, but it should not trigger student training by itself. The next source object must add cross-predicate transfer and a distinct event-to-state family.", "",
        "## Files", "",
        f"- summary JSON: `{OUT_DIR / 'sequence_shortcut_stress_summary.json'}`",
        f"- original examples: `{OUT_DIR / 'sequence_examples_original_ablated.jsonl'}`",
        f"- perturbation examples: `{OUT_DIR / 'sequence_examples_perturbations.jsonl'}`",
    ]
    ((OUT_DIR.parents[4] / 'research/documents/representation_and_objectives/data/paired_world_sequence_teacher/sequence_shortcut_stress_summary.md')).write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": summary["status"],
        "families": len(fams),
        "original_rows": len(original),
        "held_held_template_rows": len(held_held_templates),
        "raw_word12_held": results["raw_word12"]["train_to_all_held_original"]["accuracy"],
        "canonical_word12_held": results["canonical_word12"]["train_to_all_held_original"]["accuracy"],
        "canonical_word12_held_templates": results["canonical_word12"]["train_to_held_held_templates"]["accuracy"],
        "canonical_word12_lost_to": results["canonical_word12"]["train_to_held_hypothesis_lost_to"]["accuracy"],
        "symbolic_order_held": results["symbolic_order_text"]["train_to_all_held_original"]["accuracy"],
        "summary_json": str(OUT_DIR / "sequence_shortcut_stress_summary.json"),
        "summary_md": str((OUT_DIR.parents[4] / 'research/documents/representation_and_objectives/data/paired_world_sequence_teacher/sequence_shortcut_stress_summary.md')),
    }, indent=2), flush=True)


def build_teacher_prompt(context: str, hyp: str) -> str:
    return (
        "Use only the CONTEXT sentence. Decide whether the HYPOTHESIS is directly supported by that sentence.\n"
        "Track the roles of the two named competitors precisely. Do not use outside knowledge.\n"
        "Answer exactly one label: ENTAILED or NOT_ENTAILED. Do not explain.\n\n"
        f"CONTEXT: {context}\n"
        f"HYPOTHESIS: {hyp}\n"
        "LABEL:"
    )


def select_teacher_families(fams: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rng = random.Random(SEED)
    train = [f for f in fams if f["family_split"] == "train"]
    held = [f for f in fams if f["family_split"] == "held"]
    # Deterministic proportional train sample preserving all three sports.
    by_sport: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for f in train:
        by_sport[f["sport"]].append(f)
    chosen: list[dict[str, Any]] = []
    target_by_sport = {"tennis": 70, "badminton": 20, "football": 10}
    for sport, n in target_by_sport.items():
        xs = list(by_sport.get(sport, []))
        rng.shuffle(xs)
        chosen.extend(xs[: min(n, len(xs))])
    if len(chosen) < 100:
        remaining = [f for f in train if f not in chosen]
        rng.shuffle(remaining)
        chosen.extend(remaining[: 100 - len(chosen)])
    chosen = chosen[:100] + held
    chosen.sort(key=lambda x: x["family_id"])
    return chosen


def teacher_rows_for_family(fam: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    variants = [
        "original_ablated",
        "original_score_visible",
        "name_swapped_context",
        "hyp_lost_to",
    ]
    for variant in variants:
        for ex in build_examples([fam], variant):
            # The teacher set uses exactly two hypotheses per context per variant.
            rows.append({
                "id": f"step252_{ex['id']}",
                "prompt_id": f"step252_{ex['id']}",
                "family_id": ex["family_id"],
                "family_split": ex["family_split"],
                "sport": ex["sport"],
                "cx": ex["cx"],
                "variant": variant,
                "context_template_id": ex["context_template_id"],
                "context_template_split": ex["context_template_split"],
                "hyp_kind": ex["hyp_kind"],
                "hyp_direction": ex["hyp_direction"],
                "context": ex["text"],
                "hypothesis": ex["hyp"],
                "gold_label": ex["gold_label"],
                "prompt": build_teacher_prompt(ex["text"], ex["hyp"]),
            })
    return rows


def prepare_teacher_verify(_: argparse.Namespace) -> None:
    PROMPT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fams = load_families()
    chosen = select_teacher_families(fams)
    rows: list[dict[str, Any]] = []
    for fam in chosen:
        rows.extend(teacher_rows_for_family(fam))
    write_jsonl(PROMPT_DIR / "teacher_prompts_all.jsonl", rows)
    # Same prompt set for both models; separate files ease independent GPU runs.
    write_jsonl(PROMPT_DIR / "teacher_prompts_qwen.jsonl", rows)
    write_jsonl(PROMPT_DIR / "teacher_prompts_llama.jsonl", rows)
    write_jsonl(OUT_DIR / "teacher_selected_families.jsonl", chosen)
    summary = {
        "status": "TEACHER_REALIZATION_PROMPTS_READY",
        "created_utc": now(),
        "selected_families": len(chosen),
        "selected_family_split_counts": dict(collections.Counter(f["family_split"] for f in chosen)),
        "selected_sport_counts": dict(collections.Counter(f["sport"] for f in chosen)),
        "prompts_per_teacher": len(rows),
        "variant_counts": dict(collections.Counter(r["variant"] for r in rows)),
        "gold_counts": dict(collections.Counter(r["gold_label"] for r in rows)),
        "prompt_qwen": str(PROMPT_DIR / "teacher_prompts_qwen.jsonl"),
        "prompt_llama": str(PROMPT_DIR / "teacher_prompts_llama.jsonl"),
        "intended_commands": {
            "qwen": f"CUDA_VISIBLE_DEVICES=0 ${{BABYLM_GENERATOR:?configure-an-external-generator}} --model qwen3.5-9b --prompts-jsonl {PROMPT_DIR / 'teacher_prompts_qwen.jsonl'} --output-jsonl {PROMPT_DIR / 'teacher_outputs_qwen.jsonl'} --batch-size 32 --max-new-tokens 8 --temperature 0.0 --device cuda",
            "llama": f"CUDA_VISIBLE_DEVICES=1 ${{BABYLM_GENERATOR:?configure-an-external-generator}} --model llama3.1-8b-instruct --prompts-jsonl {PROMPT_DIR / 'teacher_prompts_llama.jsonl'} --output-jsonl {PROMPT_DIR / 'teacher_outputs_llama.jsonl'} --batch-size 32 --max-new-tokens 8 --temperature 0.0 --device cuda",
        },
        "scientific_use": "Verify language realization of the paired worlds, including score ablation, score visibility, name-swap perturbations, and lost-to hypothesis paraphrases. This is not evidence that a small student has learned a transferable principle.",
        "no_babylm_training_eval_upload_submission": True,
    }
    (OUT_DIR / "teacher_prompt_manifest.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


def parse_label(output: Any) -> str | None:
    s = str(output or "").strip().upper()
    s = s.replace("ENTAILLED", "ENTAILED")
    s = re.sub(r"[^A-Z_ ]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    if re.search(r"\bNOT[_ ]?ENTAILED\b", s) or re.search(r"\bNOT\s+SUPPORTED\b", s) or re.search(r"\bUNSUPPORTED\b", s) or re.search(r"\bNO\b", s):
        return "NOT_ENTAILED"
    if re.search(r"\bENTAILED\b", s) or re.search(r"\bSUPPORTED\b", s) or re.search(r"\bYES\b", s):
        return "ENTAILED"
    return None


def read_teacher_outputs(prompt_path: Path, output_path: Path, teacher: str) -> list[dict[str, Any]]:
    prompts = read_jsonl(prompt_path)
    outs = read_jsonl(output_path)
    if len(prompts) != len(outs):
        raise RuntimeError(f"{teacher}: prompt/output length mismatch {len(prompts)} vs {len(outs)}")
    rows = []
    for i, (p, o) in enumerate(zip(prompts, outs)):
        raw = str(o.get("output") or o.get("generated_text") or o.get("text") or o.get("completion") or "")
        lab = parse_label(raw)
        rows.append({
            **{k: p[k] for k in ["id", "family_id", "family_split", "sport", "cx", "variant", "context_template_id", "context_template_split", "hyp_kind", "hyp_direction", "context", "hypothesis", "gold_label"]},
            "teacher": teacher,
            "output_raw": raw,
            "parsed_label": lab,
            "correct": bool(lab == p["gold_label"]),
            "output_index": o.get("index", i),
        })
    return rows


def summarize_teacher_rows(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for val in sorted({str(r.get(key)) for r in rows}):
        rs = [r for r in rows if str(r.get(key)) == val]
        valid = [r for r in rs if r.get("parsed_label")]
        out[val] = {
            "n": len(rs),
            "valid": len(valid),
            "invalid": len(rs) - len(valid),
            "accuracy": sum(1 for r in valid if r.get("correct")) / max(1, len(valid)),
            "entailed_gold": sum(1 for r in rs if r.get("gold_label") == "ENTAILED"),
            "not_entailed_gold": sum(1 for r in rs if r.get("gold_label") == "NOT_ENTAILED"),
        }
    return out


def analyze_teacher_verify(args: argparse.Namespace) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    q_rows = read_teacher_outputs(PROMPT_DIR / "teacher_prompts_qwen.jsonl", Path(args.qwen_outputs), "qwen3.5-9b")
    l_rows = read_teacher_outputs(PROMPT_DIR / "teacher_prompts_llama.jsonl", Path(args.llama_outputs), "llama3.1-8b-instruct")
    rows = q_rows + l_rows
    write_jsonl(OUT_DIR / "teacher_labeled_rows.jsonl", rows)
    valid = [r for r in rows if r.get("parsed_label")]

    by_id: dict[str, dict[str, dict[str, Any]]] = collections.defaultdict(dict)
    for r in rows:
        by_id[r["id"]][r["teacher"]] = r
    pairs = []
    for pid, d in sorted(by_id.items()):
        if not all(t in d for t in TEACHERS):
            continue
        q, l = d[TEACHERS[0]], d[TEACHERS[1]]
        same = bool(q.get("parsed_label") and q.get("parsed_label") == l.get("parsed_label"))
        both_correct = bool(same and q.get("parsed_label") == q.get("gold_label"))
        pairs.append({
            "id": pid,
            "family_id": q["family_id"],
            "family_split": q["family_split"],
            "sport": q["sport"],
            "variant": q["variant"],
            "context_template_split": q["context_template_split"],
            "hyp_kind": q["hyp_kind"],
            "hyp_direction": q["hyp_direction"],
            "gold_label": q["gold_label"],
            "qwen_label": q.get("parsed_label"),
            "llama_label": l.get("parsed_label"),
            "same_label": same,
            "both_correct": both_correct,
            "context": q["context"],
            "hypothesis": q["hypothesis"],
            "qwen_raw": q["output_raw"],
            "llama_raw": l["output_raw"],
        })
    write_jsonl(OUT_DIR / "teacher_cross_model_pairs.jsonl", pairs)

    def pair_summary(key: str) -> dict[str, Any]:
        out = {}
        for val in sorted({str(p.get(key)) for p in pairs}):
            ps = [p for p in pairs if str(p.get(key)) == val]
            out[val] = {
                "n": len(ps),
                "agreement": sum(1 for p in ps if p["same_label"]) / max(1, len(ps)),
                "both_correct": sum(1 for p in ps if p["both_correct"]) / max(1, len(ps)),
            }
        return out

    teacher_summaries = {}
    for teacher in TEACHERS:
        trs = [r for r in rows if r["teacher"] == teacher]
        tv = [r for r in trs if r.get("parsed_label")]
        teacher_summaries[teacher] = {
            "n": len(trs),
            "valid": len(tv),
            "accuracy": sum(1 for r in tv if r.get("correct")) / max(1, len(tv)),
            "by_variant": summarize_teacher_rows(trs, "variant"),
            "by_family_split": summarize_teacher_rows(trs, "family_split"),
            "by_template_split": summarize_teacher_rows(trs, "context_template_split"),
            "by_sport": summarize_teacher_rows(trs, "sport"),
            "by_hyp_kind": summarize_teacher_rows(trs, "hyp_kind"),
        }

    ablated_pairs = [p for p in pairs if p["variant"] == "original_ablated"]
    held_template_pairs = [p for p in pairs if p["context_template_split"] == "held"]
    summary = {
        "status": "TEACHER_REALIZATION_VERIFICATION_ANALYZED",
        "created_utc": now(),
        "prompts_per_teacher": len(q_rows),
        "total_rows": len(rows),
        "valid_rows": len(valid),
        "teacher_summaries": teacher_summaries,
        "cross_teacher": {
            "n_pairs": len(pairs),
            "agreement": sum(1 for p in pairs if p["same_label"]) / max(1, len(pairs)),
            "both_correct": sum(1 for p in pairs if p["both_correct"]) / max(1, len(pairs)),
            "ablated_agreement": sum(1 for p in ablated_pairs if p["same_label"]) / max(1, len(ablated_pairs)),
            "ablated_both_correct": sum(1 for p in ablated_pairs if p["both_correct"]) / max(1, len(ablated_pairs)),
            "held_template_agreement": sum(1 for p in held_template_pairs if p["same_label"]) / max(1, len(held_template_pairs)),
            "held_template_both_correct": sum(1 for p in held_template_pairs if p["both_correct"]) / max(1, len(held_template_pairs)),
            "by_variant": pair_summary("variant"),
            "by_family_split": pair_summary("family_split"),
            "by_template_split": pair_summary("context_template_split"),
            "by_sport": pair_summary("sport"),
            "by_hyp_kind": pair_summary("hyp_kind"),
        },
        "interpretation_boundary": "Teacher success verifies that the generated contexts and hypotheses express the intended paired-world event relations. It does not show that a BabyLM-scale or TinyMLM student learns a transferable data-efficient principle.",
        "no_babylm_training_eval_upload_submission": True,
    }
    (OUT_DIR / "teacher_realization_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    failures = [r for r in rows if (not r.get("parsed_label")) or not r.get("correct")]
    pair_fail = [p for p in pairs if not p.get("both_correct")]
    md = [
        "# research teacher realization verification", "",
        "## Purpose", "",
        "Approved teacher models are used only to verify that the paired-world sentences and hypotheses express the intended role assignment. This is not a student-learning result.", "",
        "## Aggregate", "",
        f"- prompts per teacher: {len(q_rows)}", 
        f"- cross-teacher agreement: {summary['cross_teacher']['agreement']:.3f}",
        f"- both-teachers correct: {summary['cross_teacher']['both_correct']:.3f}",
        f"- score-ablated both-teachers correct: {summary['cross_teacher']['ablated_both_correct']:.3f}",
        f"- held-template both-teachers correct: {summary['cross_teacher']['held_template_both_correct']:.3f}", "",
        "## By variant", "", "| variant | agreement | both correct | n |", "|---|---:|---:|---:|",
    ]
    for k, v in summary["cross_teacher"]["by_variant"].items():
        md.append(f"| {k} | {v['agreement']:.3f} | {v['both_correct']:.3f} | {v['n']} |")
    md += ["", "## Failure sample", ""]
    for p in pair_fail[:40]:
        md += [
            f"### {p['id']}",
            f"- variant={p['variant']} sport={p['sport']} split={p['family_split']} template_split={p['context_template_split']} gold={p['gold_label']} qwen={p['qwen_label']} llama={p['llama_label']}",
            f"- context: {p['context']}",
            f"- hypothesis: {p['hypothesis']}",
            "",
        ]
    md += [
        "## Scientific meaning", "",
        "If these values are high, the current realizations are linguistically faithful enough to serve as probes and as inputs to harder source construction. The sequence stress results still decide whether student training is premature: current sports outcomes alone remain too narrow unless cross-template, cross-predicate, and event-to-state transfer are established.", "",
        "## Files", "",
        f"- rows: `{OUT_DIR / 'teacher_labeled_rows.jsonl'}`",
        f"- paired model rows: `{OUT_DIR / 'teacher_cross_model_pairs.jsonl'}`",
        f"- summary JSON: `{OUT_DIR / 'teacher_realization_summary.json'}`",
    ]
    ((OUT_DIR.parents[4] / 'research/documents/representation_and_objectives/data/paired_world_sequence_teacher/teacher_realization_summary.md')).write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": summary["status"],
        "prompts_per_teacher": len(q_rows),
        "cross_teacher_agreement": summary["cross_teacher"]["agreement"],
        "both_correct": summary["cross_teacher"]["both_correct"],
        "ablated_both_correct": summary["cross_teacher"]["ablated_both_correct"],
        "held_template_both_correct": summary["cross_teacher"]["held_template_both_correct"],
        "summary_json": str(OUT_DIR / "teacher_realization_summary.json"),
        "summary_md": str((OUT_DIR.parents[4] / 'research/documents/representation_and_objectives/data/paired_world_sequence_teacher/teacher_realization_summary.md')),
    }, indent=2), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("sequence_stress")
    sub.add_parser("prepare_teacher")
    an = sub.add_parser("analyze_teacher")
    an.add_argument("--qwen-outputs", default=str(PROMPT_DIR / "teacher_outputs_qwen.jsonl"))
    an.add_argument("--llama-outputs", default=str(PROMPT_DIR / "teacher_outputs_llama.jsonl"))
    args = ap.parse_args()
    if args.cmd == "sequence_stress":
        run_sequence_stress(args)
    elif args.cmd == "prepare_teacher":
        prepare_teacher_verify(args)
    elif args.cmd == "analyze_teacher":
        analyze_teacher_verify(args)


if __name__ == "__main__":
    main()
