#!/usr/bin/env python3
"""Summarize research/058 compact learner evidence into a concise table."""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import json
import pathlib

ROOT = _public_path('.')
S = _public_path('experiments/archive/functional_learning')
research = _public_path('experiments/archive/functional_learning/data/concentrated_compact_learning_schedulematched/summary.json')
CONT = _public_path('experiments/archive/functional_learning/data/compact_continuation_control/summary.json')
COMMON = _public_path('experiments/archive/functional_learning/data/common_target_probe/summary.json')
COMMON_CONT = _public_path('experiments/archive/functional_learning/data/common_target_probe_continuation/summary.json')
OUT = _public_path('experiments/archive/functional_learning/data/compact_evidence_synthesis')


def rel(p: pathlib.Path) -> str:
    try:
        return str(p.resolve().relative_to(ROOT))
    except Exception:
        return str(p)


def read_json(p: pathlib.Path):
    return json.loads(p.read_text(encoding="utf-8"))


def by_name(arms):
    return {a["arm"]["name"]: a for a in arms}


def view_delta(arm, view, cond="with_source"):
    return arm["eval_summary"]["by_view_condition"][f"{view}/{cond}"]["mean_delta_nll_vs_parent"]


def source_follow(summary, model, split=None):
    sf = summary["model_summaries"][model]["source_follow"]
    if split is None:
        return {"both": sf["both_source_conditions_correct"], "n": sf["n"], "mean_swing": sf["mean_source_follow_swing"]}
    g = sf["by_split"][split]
    return {"both": g["both_correct"], "n": g["n"], "mean_swing": g["mean_swing"], "mean_orig": g["mean_orig_margin"], "mean_altered": g["mean_altered_margin"], "mean_no_source": g["mean_no_source_original_margin"]}


def delta(summary, model, split=None, condition=None):
    d = summary["deltas_vs_parent"][model]
    if split is not None:
        return d["by_split"][split]["mean_delta"]
    if condition is not None:
        return d["by_condition"][condition]["mean_delta"]
    return d["mean_delta_expected_margin_vs_parent"]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    s57 = read_json(research)
    cont = read_json(CONT)
    common = read_json(COMMON)
    common_cont = read_json(COMMON_CONT)
    arms = by_name(s57["arm_summaries"])

    surface = []
    for name in ["current_equal_epoch", "compact_equal_epoch", "compact_wordmatched"]:
        arm = arms[name]
        surface.append({
            "model": name,
            "trained_view": arm["arm"]["train_view"],
            "epochs": arm["completed_epochs"],
            "charged_row_words": arm["final_train_log"]["charged_row_words_cum"],
            "view_words": arm["final_train_log"]["view_words_cum"],
            "current_with_source_delta_nll": view_delta(arm, "current"),
            "compact_with_source_delta_nll": view_delta(arm, "compact"),
            "current_view_only_delta_nll": view_delta(arm, "current", "view_only"),
            "compact_view_only_delta_nll": view_delta(arm, "compact", "view_only"),
        })
    cks = cont["checkpoint_summaries"]
    continuation_surface = []
    for tag in ["epoch0080", "epoch0092"]:
        ev = cks[tag]["eval_summary"]["by_view_condition"]
        continuation_surface.append({
            "tag": tag,
            "epoch": cks[tag]["epoch"],
            "charged_row_words": cks[tag]["train_log"]["charged_row_words_cum"],
            "view_words": cks[tag]["train_log"]["view_words_cum"],
            "compact_with_source_delta_nll": ev["compact/with_source"]["mean_delta_nll_vs_parent"],
            "current_with_source_delta_nll": ev["current/with_source"]["mean_delta_nll_vs_parent"],
            "compact_view_only_delta_nll": ev["compact/view_only"]["mean_delta_nll_vs_parent"],
            "current_view_only_delta_nll": ev["current/view_only"]["mean_delta_nll_vs_parent"],
        })
    exact_gain = {
        "compact_with_source_extra_delta_nll_epoch92_minus_epoch80": continuation_surface[1]["compact_with_source_delta_nll"] - continuation_surface[0]["compact_with_source_delta_nll"],
        "current_with_source_extra_delta_nll_epoch92_minus_epoch80": continuation_surface[1]["current_with_source_delta_nll"] - continuation_surface[0]["current_with_source_delta_nll"],
        "extra_charged_row_words": continuation_surface[1]["charged_row_words"] - continuation_surface[0]["charged_row_words"],
        "extra_view_words": continuation_surface[1]["view_words"] - continuation_surface[0]["view_words"],
    }

    common_rows = []
    for name in ["parent", "current_equal_epoch", "compact_equal_epoch", "compact_wordmatched"]:
        row = {"model": name, **source_follow(common, name)}
        if name != "parent":
            row.update({
                "delta_all": delta(common, name),
                "delta_trained_content": delta(common, name, split="trained_content"),
                "delta_held_source": delta(common, name, split="held_source"),
                "delta_source_original": delta(common, name, condition="source_original"),
                "delta_source_altered": delta(common, name, condition="source_altered"),
                "delta_no_source": delta(common, name, condition="no_source"),
            })
        row["trained_content"] = source_follow(common, name, "trained_content")
        row["held_source"] = source_follow(common, name, "held_source")
        common_rows.append(row)

    common_cont_rows = []
    for name in ["parent", "compact_exact_epoch80", "compact_exact_epoch92"]:
        row = {"model": name, **source_follow(common_cont, name)}
        if name != "parent":
            row.update({
                "delta_all": delta(common_cont, name),
                "delta_trained_content": delta(common_cont, name, split="trained_content"),
                "delta_held_source": delta(common_cont, name, split="held_source"),
                "delta_source_original": delta(common_cont, name, condition="source_original"),
                "delta_source_altered": delta(common_cont, name, condition="source_altered"),
                "delta_no_source": delta(common_cont, name, condition="no_source"),
            })
        row["trained_content"] = source_follow(common_cont, name, "trained_content")
        row["held_source"] = source_follow(common_cont, name, "held_source")
        common_cont_rows.append(row)
    exact_common_gain = {
        "delta_all_epoch92_minus_epoch80": common_cont_rows[2]["delta_all"] - common_cont_rows[1]["delta_all"],
        "delta_trained_content_epoch92_minus_epoch80": common_cont_rows[2]["delta_trained_content"] - common_cont_rows[1]["delta_trained_content"],
        "delta_held_source_epoch92_minus_epoch80": common_cont_rows[2]["delta_held_source"] - common_cont_rows[1]["delta_held_source"],
        "mean_swing_epoch92_minus_epoch80": common_cont_rows[2]["mean_swing"] - common_cont_rows[1]["mean_swing"],
    }

    final = {
        "status": "COMPACT_EVIDENCE_SYNTHESIS",
        "inputs": {"research": rel(research), "continuation": rel(CONT), "common": rel(COMMON), "common_continuation": rel(COMMON_CONT)},
        "surface_step57": surface,
        "exact_continuation_surface": continuation_surface,
        "exact_extra_recurrence_surface_gain": exact_gain,
        "common_target_step57_models": common_rows,
        "common_target_exact_continuation": common_cont_rows,
        "exact_extra_recurrence_common_gain": exact_common_gain,
        "interpretation": {
            "surface": "Own-view NLL gains are large relative to cross-view gains, especially for compact training, so the previous readout mainly measured adaptation to practiced forms.",
            "common_targets": "Fixed cloze targets with original and altered sources show much smaller mean margin changes. The coherent86 parent already follows most source edits, and compact training adds a small positive margin shift on trained-content and held-source items.",
            "extra_recurrence": "Continuing the exact compact state from epoch 80 to 92 gives a real but small additional gain on both the compact surface and common targets; this is not yet evidence that saved words would be better spent on new support in a legal stream."
        }
    }
    (_public_path('experiments/archive/functional_learning/data/compact_evidence_synthesis/summary.json')).write_text(json.dumps(final, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(final, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
