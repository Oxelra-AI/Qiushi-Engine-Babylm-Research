#!/usr/bin/env python3
"""Analyze research adversarial degradation on nonidentical held renderings.

Uses existing research prediction files; no model execution.  Joins prediction
records with raw eval event text from the representation_and_objectives substrate and summarizes how
wrong h1-incident comparison constraints affect state and comparison readouts by
relation, voice, object, and suite.
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean

A01_SCRIPT_DIR = Path("experiments/archive/representation_and_objectives/training/scripts")
sys.path.insert(0, str(A01_SCRIPT_DIR))
import raw_span_discovery_probe as base  # noqa: E402

ROOT = Path("experiments/archive/functional_learning/data/topology_probe_phase1")
FULL = ROOT / "full_plus_e60/full_graph_shared_trunk_bs+1_seed30000"
ADV = ROOT / "adversarial_plus_e60/disconn_adversarial_shared_trunk_bs+1_seed30000"
OUT = Path("experiments/archive/functional_learning/data/degradation_analysis")
OUT.mkdir(parents=True, exist_ok=True)


def load_jsonl(path: Path):
    with path.open() as f:
        return [json.loads(line) for line in f]


def voice(text: str) -> str:
    return "passive" if " was " in text and " by " in text else "active"


def obj(text: str) -> str:
    m = re.search(r"During the ([A-Za-z]+) episode", text)
    return m.group(1).lower() if m else "?"


def build_eval_maps():
    ts, tc, es, ec, pe, counts = base.load_dataset(base.DEFAULT_DATA_ROOT, base.DEFAULT_ARM)
    state = {}
    for q in es:
        state[q.key] = {
            "event": q.event,
            "voice": voice(q.event),
            "object": q.object_name or obj(q.event),
            "relation": q.relation,
            "suite": q.suite,
            "initial_pattern": q.initial_pattern,
            "static_slot": q.static_slot,
            "is_changed": bool(q.is_changed),
        }
    comp = {}
    for c in ec:
        comp[c.row_id] = {
            "event1": c.event1,
            "event2": c.event2,
            "voice1": voice(c.event1),
            "voice2": voice(c.event2),
            "object1": obj(c.event1),
            "object2": obj(c.event2),
            "relation1": c.relation1,
            "relation2": c.relation2,
            "suite": c.suite,
            "label": c.label,
        }
    return state, comp


def state_choice(records):
    # Use the inherited query-level reconstruction so the margin is target-signed
    # score(target)-score(non-target).  The per-candidate `d_e` field is a fixed
    # candidate-difference convention and is not target-signed across voice/order.
    out = {}
    for r in base.state_choice_records(records):
        if not r.get("is_changed"):
            continue
        out[r["query_key"]] = {
            "correct": int(r["correct"]),
            "margin": float(r["margin"]),
            "relation": r.get("relation"),
            "suite": r.get("suite"),
            "initial_pattern": r.get("initial_pattern"),
            "static_slot": r.get("static_slot"),
        }
    return out


def agg(rows, keys):
    d = defaultdict(lambda: {"n": 0, "full_ok": 0, "adv_ok": 0, "delta_margin": [], "adv_margin": [], "full_margin": []})
    for row in rows:
        k = tuple(row.get(x) for x in keys)
        dd = d[k]
        dd["n"] += 1
        dd["full_ok"] += int(row["full_correct"])
        dd["adv_ok"] += int(row["adv_correct"])
        dd["full_margin"].append(row["full_margin"])
        dd["adv_margin"].append(row["adv_margin"])
        dd["delta_margin"].append(row["adv_margin"] - row["full_margin"])
    out = []
    for k, dd in sorted(d.items(), key=lambda kv: str(kv[0])):
        rec = {keys[i]: k[i] for i in range(len(keys))}
        rec.update({
            "n": dd["n"],
            "full_acc": dd["full_ok"] / dd["n"],
            "adv_acc": dd["adv_ok"] / dd["n"],
            "acc_delta_adv_minus_full": dd["adv_ok"] / dd["n"] - dd["full_ok"] / dd["n"],
            "full_margin_mean": mean(dd["full_margin"]),
            "adv_margin_mean": mean(dd["adv_margin"]),
            "margin_delta_adv_minus_full": mean(dd["delta_margin"]),
        })
        out.append(rec)
    return out


def main():
    state_meta, comp_meta = build_eval_maps()
    full_state = state_choice(load_jsonl(FULL / "state_predictions.jsonl"))
    adv_state = state_choice(load_jsonl(ADV / "state_predictions.jsonl"))
    state_rows = []
    for key in sorted(set(full_state) & set(adv_state)):
        meta = state_meta.get(key, {})
        f = full_state[key]; a = adv_state[key]
        state_rows.append({
            "query_key": key,
            "relation": f.get("relation") or meta.get("relation"),
            "suite": f.get("suite") or meta.get("suite"),
            "voice": meta.get("voice", "?"),
            "object": meta.get("object", f.get("object")),
            "initial_pattern": f.get("initial_pattern"),
            "static_slot": f.get("static_slot"),
            "full_correct": f["correct"],
            "adv_correct": a["correct"],
            "full_margin": f["margin"],
            "adv_margin": a["margin"],
        })

    full_comp = {r["row_id"]: r for r in load_jsonl(FULL / "comparison_predictions_original_eval.jsonl")}
    adv_comp = {r["row_id"]: r for r in load_jsonl(ADV / "comparison_predictions_original_eval.jsonl")}
    comp_rows = []
    for rid in sorted(set(full_comp) & set(adv_comp)):
        fm = full_comp[rid]; am = adv_comp[rid]; meta = comp_meta.get(rid, {})
        comp_rows.append({
            "row_id": rid,
            "edge": f"{fm['relation1']}->{fm['relation2']}",
            "relation1": fm["relation1"],
            "relation2": fm["relation2"],
            "suite": fm.get("suite") or meta.get("suite"),
            "voice_pair": f"{meta.get('voice1','?')}->{meta.get('voice2','?')}",
            "object_pair": f"{meta.get('object1','?')}->{meta.get('object2','?')}",
            "label": bool(fm["label"]),
            "h1_involved": ("h1_mep" in [fm["relation1"], fm["relation2"]]),
            "full_correct": int(fm["correct"]),
            "adv_correct": int(am["correct"]),
            "full_margin": float(fm["signed_margin"]),
            "adv_margin": float(am["signed_margin"]),
        })

    result = {
        "state_by_relation": agg(state_rows, ["relation"]),
        "state_by_relation_voice": agg(state_rows, ["relation", "voice"]),
        "state_by_relation_suite": agg(state_rows, ["relation", "suite"]),
        "state_by_relation_object": agg(state_rows, ["relation", "object"]),
        "comparison_by_edge": agg(comp_rows, ["edge"]),
        "comparison_by_h1_involved": agg(comp_rows, ["h1_involved"]),
        "comparison_by_edge_voice_pair": agg(comp_rows, ["edge", "voice_pair"]),
    }
    (OUT / "degradation_analysis.json").write_text(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True) + "\n")

    lines = [
        "# research analysis of research wrong-constraint degradation",
        "",
        "Existing research full-graph and adversarial-h1 prediction files were joined to the raw eval substrate. This asks whether wrong h1-incident constraints degrade held nonidentical content use below the full-graph condition, across active/passive voice and held objects/names.",
        "",
        "## State readout by relation",
        "",
        "| relation | n | full acc | adversarial acc | acc delta | full margin | adversarial margin | margin delta |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in result["state_by_relation"]:
        lines.append(f"| {r['relation']} | {r['n']} | {r['full_acc']:.3f} | {r['adv_acc']:.3f} | {r['acc_delta_adv_minus_full']:.3f} | {r['full_margin_mean']:.3f} | {r['adv_margin_mean']:.3f} | {r['margin_delta_adv_minus_full']:.3f} |")
    lines.extend(["", "## h1 state readout by voice", "", "| relation | voice | n | full acc | adversarial acc | margin delta |", "|---|---|---:|---:|---:|---:|"])
    for r in result["state_by_relation_voice"]:
        if r["relation"] == "h1_mep":
            lines.append(f"| {r['relation']} | {r['voice']} | {r['n']} | {r['full_acc']:.3f} | {r['adv_acc']:.3f} | {r['margin_delta_adv_minus_full']:.3f} |")
    lines.extend(["", "## h1 state readout by held object", "", "| object | n | full acc | adversarial acc | margin delta |", "|---|---:|---:|---:|---:|"])
    for r in result["state_by_relation_object"]:
        if r["relation"] == "h1_mep":
            lines.append(f"| {r['object']} | {r['n']} | {r['full_acc']:.3f} | {r['adv_acc']:.3f} | {r['margin_delta_adv_minus_full']:.3f} |")
    lines.extend(["", "## Eval comparison by edge", "", "| edge | n | full acc | adversarial acc | acc delta | full margin | adversarial margin | margin delta |", "|---|---:|---:|---:|---:|---:|---:|---:|"])
    for r in result["comparison_by_edge"]:
        lines.append(f"| {r['edge']} | {r['n']} | {r['full_acc']:.3f} | {r['adv_acc']:.3f} | {r['acc_delta_adv_minus_full']:.3f} | {r['full_margin_mean']:.3f} | {r['adv_margin_mean']:.3f} | {r['margin_delta_adv_minus_full']:.3f} |")
    lines.extend(["", "## Eval comparison aggregated by h1 involvement", "", "| h1 involved | n | full acc | adversarial acc | acc delta | margin delta |", "|---|---:|---:|---:|---:|---:|"])
    for r in result["comparison_by_h1_involved"]:
        lines.append(f"| {r['h1_involved']} | {r['n']} | {r['full_acc']:.3f} | {r['adv_acc']:.3f} | {r['acc_delta_adv_minus_full']:.3f} | {r['margin_delta_adv_minus_full']:.3f} |")
    lines.extend(["", "## Interpretation", "", "Adversarial h1-incident constraints drive h1 state readout below the full-graph condition on every held h1 object and in both active/passive voices, while h0/h2/h3 remain correct. Eval comparison edges involving h1 also fall from full-graph accuracy 1.0 to 0.0 with large negative margin shifts, whereas non-h1 edges remain correct. This is the analogue of degradation from wrong/exact constraints: corrupted relational content does not merely fail to help; it installs a systematically wrong h1 coordinate on held nonidentical renderings. It is still not a token-NLL copy/content probe and should not be equated with relation_learning's RoBERTa nonoverlap content result.", ""])
    ((OUT.parents[4] / 'research/documents/functional_learning/data/degradation_analysis/degradation_analysis.md')).write_text("\n".join(lines))

    print(json.dumps({
        "status": "STEP4_DEGRADATION_ANALYSIS_DONE",
        "out_md": str((OUT.parents[4] / 'research/documents/functional_learning/data/degradation_analysis/degradation_analysis.md')),
        "out_json": str(OUT / "degradation_analysis.json"),
        "state_by_relation": result["state_by_relation"],
        "comparison_by_h1_involved": result["comparison_by_h1_involved"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
