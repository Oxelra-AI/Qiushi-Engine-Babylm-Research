#!/usr/bin/env python3
"""research: arm-to-arm endpoint signature for packed target-selective 100M arms.

This avoids requiring historical compact-view item predictions.  It reads the two
newly evaluated packed target-selective endpoints, scores their prediction files
against the current official-compatible gold surfaces, and computes paired item
transitions for the primary arm-to-arm contrast.

Orientation:
  drop_abs_minus_drop_copied_word: positive means the model trained with
      source-absent compact-content labels removed is more accurate.
  drop_copied_word_minus_drop_abs: positive means retaining the source-absent
      compact-content labels while deleting matched copied-content labels is more
      accurate.  This is the direction that would support source-absent-label
      mediation if it also matches the historical Supplement/EWoK pattern.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import statistics
from typing import Any

ROOT = pathlib.Path("experiments/archive/representation_and_objectives")
SCRIPT_DIR = ROOT / "scripts"
import sys
if str(SCRIPT_DIR.resolve()) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR.resolve()))

import packed_targetselect_endpoint_signature_v2 as sig  # noqa: E402

DEFAULT_DROP_ABS = ROOT / "data/packed_targetselect_postrun_readout/endpoint_eval/drop_abs/per_target/dropabs100.json"
DEFAULT_DROP_COPIED = ROOT / "data/packed_targetselect_postrun_readout/endpoint_eval/drop_copied_word/per_target/dropcopied_word100.json"
DEFAULT_OUT = ROOT / "data/packed_targetselect_armonly_signature"

ARM_A = "drop_abs"
ARM_B = "drop_copied_word"
SCORE_COLUMNS = sig.SCORE_COLUMNS
ITEM_TASKS = sig.ITEM_TASKS
RELATIONAL = sig.RELATIONAL
ADJ_INDEPENDENT = sig.ADJ_INDEPENDENT
HIST_VECTOR_KEYS = sig.HIST_VECTOR_KEYS


def round_or_none(x: float | None, nd: int = 6) -> float | None:
    return None if x is None else round(float(x), nd)


def make_score_table(arms: dict[str, dict[str, Any]]) -> dict[str, Any]:
    scores: dict[str, dict[str, float | None]] = {}
    for arm, payload in arms.items():
        sc: dict[str, float | None] = {c: sig.task_score(payload, c) for c in SCORE_COLUMNS}
        vals = [sc[c] for c in SCORE_COLUMNS]
        sc["cheap7_equal_mean"] = None if any(v is None for v in vals) else round(sum(float(v) for v in vals) / len(vals), 6)
        scores[arm] = sc
    deltas: dict[str, dict[str, float | None]] = {}
    for a, b in [(ARM_A, ARM_B), (ARM_B, ARM_A)]:
        key = f"{a}_minus_{b}"
        deltas[key] = {}
        for c in SCORE_COLUMNS + ["cheap7_equal_mean"]:
            av = scores[a].get(c)
            bv = scores[b].get(c)
            deltas[key][c] = None if av is None or bv is None else round(float(av) - float(bv), 6)
    return {"scores": scores, "deltas": deltas}


def transition_pair(items_by_arm: dict[str, list[dict[str, Any]]], better: str, base: str) -> dict[str, Any]:
    # sig.transition(a_items, b_items, a_name, b_name) reports b-minus-a.
    return sig.transition(items_by_arm[base], items_by_arm[better], base, better)


def grouped_pair_transitions(items_by_arm: dict[str, list[dict[str, Any]]], group_key: str, values: list[str] | None = None) -> dict[str, Any]:
    out = {}
    base_items = sig.group_items(items_by_arm[ARM_B], group_key, values)
    abs_items = sig.group_items(items_by_arm[ARM_A], group_key, values)
    out[f"{ARM_A}_minus_{ARM_B}"] = sig.transition(base_items, abs_items, ARM_B, ARM_A)
    out[f"{ARM_B}_minus_{ARM_A}"] = sig.transition(abs_items, base_items, ARM_A, ARM_B)
    return out


def by_group(items_by_arm: dict[str, list[dict[str, Any]]], group_key: str) -> dict[str, dict[str, Any]]:
    groups = sorted(set(str(x.get(group_key, "unknown")) for arm in items_by_arm.values() for x in arm))
    return {g: grouped_pair_transitions(items_by_arm, group_key, [g]) for g in groups}


def aggregate_by_group(items_by_arm: dict[str, list[dict[str, Any]]], group_key: str, values: list[str] | None) -> dict[str, Any]:
    return {arm: sig.aggregate_accuracy(rows, group_key, values) for arm, rows in items_by_arm.items()}


def vector_from_transitions(transitions: dict[str, Any], contrast_key: str) -> dict[str, float]:
    vec: dict[str, float] = {}
    supp = transitions.get("Supplement", {}).get(contrast_key)
    if supp:
        vec["Supplement"] = float(supp["accuracy_delta_b_minus_a_pp"])
    doms = transitions.get("EWoK_domain_vectors", {})
    for dom in RELATIONAL + ADJ_INDEPENDENT:
        rec = doms.get(dom, {}).get(contrast_key)
        if rec:
            vec[dom] = float(rec["accuracy_delta_b_minus_a_pp"])
    return vec


def similarity_block(intervention_vectors: dict[str, dict[str, float]]) -> dict[str, Any]:
    hist = {
        "view_vs_adjbreak": sig.load_step211_vector("view_vs_adjbreak"),
        "view_vs_repeat": sig.load_step211_vector("view_vs_repeat"),
    }
    sims: dict[str, Any] = {}
    for ck, vec in intervention_vectors.items():
        sims[ck] = {}
        for hname, hvec in hist.items():
            use = [k for k in HIST_VECTOR_KEYS if k in vec and k in hvec]
            sims[ck][hname] = {
                "common_keys": use,
                "sign_agreement_fraction": None if not use else round(sum(1 for k in use if (vec[k] == 0 and hvec[k] == 0) or ((vec[k] > 0) == (hvec[k] > 0))) / len(use), 6),
                "cosine": None if sig.cosine(vec, hvec, use) is None else round(sig.cosine(vec, hvec, use), 6),
                "pearson": None if sig.corr(vec, hvec, use) is None else round(sig.corr(vec, hvec, use), 6),
            }
    return {"historical_vectors": hist, "vector_similarity_to_historical": sims}


def compact_summary(scores: dict[str, Any], transitions: dict[str, Any]) -> dict[str, Any]:
    rel_key = "relational_domains"
    adj_key = "adjacency_independent_domains"
    return {
        "score_delta_drop_abs_minus_copied": scores["deltas"].get(f"{ARM_A}_minus_{ARM_B}"),
        "score_delta_copied_minus_drop_abs": scores["deltas"].get(f"{ARM_B}_minus_{ARM_A}"),
        "supplement_transition_drop_abs_minus_copied": transitions.get("Supplement", {}).get(f"{ARM_A}_minus_{ARM_B}"),
        "supplement_transition_copied_minus_drop_abs": transitions.get("Supplement", {}).get(f"{ARM_B}_minus_{ARM_A}"),
        "ewok_relational_group_drop_abs_minus_copied": transitions.get("EWoK_groups", {}).get(rel_key, {}).get("contrasts", {}).get(f"{ARM_A}_minus_{ARM_B}"),
        "ewok_relational_group_copied_minus_drop_abs": transitions.get("EWoK_groups", {}).get(rel_key, {}).get("contrasts", {}).get(f"{ARM_B}_minus_{ARM_A}"),
        "ewok_adjind_group_drop_abs_minus_copied": transitions.get("EWoK_groups", {}).get(adj_key, {}).get("contrasts", {}).get(f"{ARM_A}_minus_{ARM_B}"),
        "ewok_adjind_group_copied_minus_drop_abs": transitions.get("EWoK_groups", {}).get(adj_key, {}).get("contrasts", {}).get(f"{ARM_B}_minus_{ARM_A}"),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--drop-abs-json", default=str(DEFAULT_DROP_ABS))
    ap.add_argument("--drop-copied-json", default=str(DEFAULT_DROP_COPIED))
    ap.add_argument("--output_dir", default=str(DEFAULT_OUT))
    args = ap.parse_args()

    out_dir = pathlib.Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    arms = {
        ARM_A: sig.load_json(pathlib.Path(args.drop_abs_json)),
        ARM_B: sig.load_json(pathlib.Path(args.drop_copied_json)),
    }
    scores = make_score_table(arms)
    items = sig.build_items(arms)

    transitions: dict[str, Any] = {}
    for task in ITEM_TASKS:
        transitions[task] = {
            f"{ARM_A}_minus_{ARM_B}": transition_pair(items[task], ARM_A, ARM_B),
            f"{ARM_B}_minus_{ARM_A}": transition_pair(items[task], ARM_B, ARM_A),
        }
        if task in {"Supplement", "COMPS"}:
            transitions[task]["by_subtask"] = by_group(items[task], "subtask")
        if task == "EWoK":
            transitions[task]["by_domain"] = by_group(items[task], "domain")

    ewok_groups: dict[str, Any] = {}
    for gname, domains in [("relational_domains", RELATIONAL), ("adjacency_independent_domains", ADJ_INDEPENDENT)]:
        ewok_groups[gname] = {
            "domains": domains,
            "scores": aggregate_by_group(items["EWoK"], "domain", domains),
            "contrasts": grouped_pair_transitions(items["EWoK"], "domain", domains),
        }
    ewok_domain: dict[str, Any] = {}
    for dom in sorted(set(str(x.get("domain")) for arm in items["EWoK"].values() for x in arm)):
        ewok_domain[dom] = grouped_pair_transitions(items["EWoK"], "domain", [dom])
    transitions["EWoK_groups"] = ewok_groups
    transitions["EWoK_domain_vectors"] = ewok_domain

    intervention_vectors = {
        f"{ARM_A}_minus_{ARM_B}": vector_from_transitions(transitions, f"{ARM_A}_minus_{ARM_B}"),
        f"{ARM_B}_minus_{ARM_A}": vector_from_transitions(transitions, f"{ARM_B}_minus_{ARM_A}"),
    }
    sim = similarity_block(intervention_vectors)

    micro = sig.official_micro_report(arms, items)
    payload: dict[str, Any] = {
        "status": "PACKED_TARGETSELECT_ARMONLY_SIGNATURE",
        "meaning": "Primary official-compatible arm-to-arm transition readout for the two new packed target-selective 100M endpoints, without requiring historical compact-view item prediction surfaces.",
        "inputs": {"drop_abs_json": args.drop_abs_json, "drop_copied_json": args.drop_copied_json},
        "orientation": {
            f"{ARM_A}_minus_{ARM_B}": "positive means deleting source-absent compact-content labels was better than deleting matched copied-content labels",
            f"{ARM_B}_minus_{ARM_A}": "positive means retaining source-absent compact-content labels was better than retaining the matched copied-content labels",
            "mediation_direction": f"{ARM_B}_minus_{ARM_A}",
        },
        "cheap7": scores,
        "official_vs_micro": micro,
        "item_transitions": transitions,
        "intervention_vectors": intervention_vectors,
        **sim,
        "summary": compact_summary(scores, transitions),
    }

    out_json = out_dir / "packed_targetselect_armonly_signature.json"
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    md = [
        "# research packed target-selective arm-to-arm endpoint signature",
        "",
        f"JSON: `{out_json}`",
        "",
        "## Scores",
        "",
        "| arm | " + " | ".join(SCORE_COLUMNS + ["equal7"]) + " |",
        "|---" + "|---:" * (len(SCORE_COLUMNS) + 1) + "|",
    ]
    for arm, sc in scores["scores"].items():
        md.append("| " + arm + " | " + " | ".join("NA" if sc.get(c) is None else f"{float(sc[c]):.3f}" for c in SCORE_COLUMNS + ["cheap7_equal_mean"]) + " |")
    md += ["", "## Main score deltas", "", "| contrast | Supplement | EWoK | COMPS | GlobalPIQA | equal7 |", "|---|---:|---:|---:|---:|---:|"]
    for ck, de in scores["deltas"].items():
        md.append(f"| {ck} | {de.get('Supplement')} | {de.get('EWoK')} | {de.get('COMPS')} | {de.get('GlobalPIQA')} | {de.get('cheap7_equal_mean')} |")
    md += ["", "## EWoK research groups", ""]
    for gname, rec in ewok_groups.items():
        md.append(f"### {gname}")
        md.append("| arm | n | micro accuracy |")
        md.append("|---|---:|---:|")
        for arm, s in rec["scores"].items():
            md.append(f"| {arm} | {s['n']} | {s['micro_accuracy']:.3f} |")
        md.append("")
        md.append("| contrast | delta pp | net items | bootstrap p025 | bootstrap p975 |")
        md.append("|---|---:|---:|---:|---:|")
        for ck, tr in rec["contrasts"].items():
            boot = tr.get("bootstrap_accuracy_delta_b_minus_a_pp") or {}
            md.append(f"| {ck} | {tr['accuracy_delta_b_minus_a_pp']} | {tr['net_gain_b_minus_a_items']} | {boot.get('p025')} | {boot.get('p975')} |")
        md.append("")
    md += ["## Historical-vector comparison", ""]
    for ck, sv in payload["vector_similarity_to_historical"].items():
        md.append(f"- `{ck}`: " + json.dumps(sv, ensure_ascii=False))
    (out_dir / "packed_targetselect_armonly_signature.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "out": str(out_json), "cheap7_deltas": scores["deltas"], "ewok_groups": ewok_groups, "vector_similarity_to_historical": payload["vector_similarity_to_historical"]}, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
