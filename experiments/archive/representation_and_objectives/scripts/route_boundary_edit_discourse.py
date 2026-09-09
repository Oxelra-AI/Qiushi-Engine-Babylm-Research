#!/usr/bin/env python3
"""Route-boundary analysis for edit/discourse discriminators.

This script reads zero-training artifacts and extracts route-relevant
numbers: whether directed edit-state/discourse-state evidence supports
an explicit dense equivariant transformation source, or whether it is mostly
copy/topic/token-value structure that should not be turned into another training
fork without stronger evidence.
"""
from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median

ROOT = Path("experiments/archive/representation_and_objectives")
OUT_DIR = ROOT / "data/route_boundary_edit_discourse"
OUT_DIR.mkdir(parents=True, exist_ok=True)

A02 = Path("experiments/archive/frontier_consolidation/data")
EDIT_MD = (A02.parents[3] / 'research/documents/frontier_consolidation/data/edit_state_token_value/edit_token_value_private_readouts.md')
EDIT_JSON = A02 / "edit_state_token_value/edit_token_value_private_readouts.json"
EDIT_CSV = A02 / "edit_state_token_value/edit_token_value_examples.csv"
EDIT_PILOT_V2_MD = (A02.parents[3] / 'research/documents/frontier_consolidation/data/pilot_edit_private_readout_v2/edit_token_value_private_readouts.md')
EDIT_PILOT_V2_JSON = A02 / "pilot_edit_private_readout_v2/edit_token_value_private_readouts.json"
EDIT_PILOT_V2_CSV = A02 / "pilot_edit_private_readout_v2/edit_token_value_examples.csv"
DISCOURSE_MD = (A02.parents[3] / 'research/documents/frontier_consolidation/data/discourse_token_value/discourse_token_value_private_readouts.md')
DISCOURSE_JSON = A02 / "discourse_token_value/discourse_token_value_private_readouts.json"
DISCOURSE_CSV = A02 / "discourse_token_value/discourse_token_value_examples.csv"
PLAN = Path("research/plans/frontier_consolidation/next_route_discriminator_plan.md")
INV = (A02.parents[3] / 'research/documents/frontier_consolidation/data/candidate_signal_inventory/candidate_signal_inventory.md')


def load_json(p: Path):
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def f(row, key, default=math.nan):
    try:
        return float(row.get(key, default))
    except Exception:
        return default


def summarize(xs):
    xs = [float(x) for x in xs if x is not None and not math.isnan(float(x))]
    if not xs:
        return {"n": 0}
    xs2 = sorted(xs)
    def q(p):
        if len(xs2) == 1:
            return xs2[0]
        idx = p * (len(xs2) - 1)
        lo = int(math.floor(idx)); hi = int(math.ceil(idx))
        if lo == hi:
            return xs2[lo]
        return xs2[lo] * (hi - idx) + xs2[hi] * (idx - lo)
    return {
        "n": len(xs2), "mean": mean(xs2), "median": median(xs2),
        "p05": q(0.05), "p25": q(0.25), "p75": q(0.75), "p95": q(0.95),
        "min": xs2[0], "max": xs2[-1],
    }


def read_csv(path: Path):
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def edit_csv_summary(path: Path):
    rows = read_csv(path)
    out = {"path": str(path), "n": len(rows)}
    true_adv = [f(r, "nll_decoy") - f(r, "nll_true") for r in rows]
    true_vs_base = [f(r, "nll_base") - f(r, "nll_true") for r in rows]
    out["true_advantage_vs_decoy"] = summarize(true_adv)
    out["true_reduction_vs_base"] = summarize(true_vs_base)
    out["target_kind_counts"] = dict(Counter(r.get("target_kind", "") for r in rows))
    out["top1"] = {
        "base": sum(int(r.get("top1_base", 0) or 0) for r in rows) / max(1, len(rows)),
        "true": sum(int(r.get("top1_true", 0) or 0) for r in rows) / max(1, len(rows)),
        "decoy": sum(int(r.get("top1_decoy", 0) or 0) for r in rows) / max(1, len(rows)),
    }
    # Target text surface categories. Single-piece items may still be morphology fragments,
    # so quantify the share that looks like full lexical words vs suffix/prefix fragments.
    surface = Counter()
    for r in rows:
        t = r.get("target_text", "")
        if not t:
            surface["empty"] += 1
        elif t[0].isupper() and t.isalpha() and len(t) >= 3:
            surface["capitalized_wordlike"] += 1
        elif t.isalpha() and len(t) >= 4:
            surface["lower_wordlike_len_ge4"] += 1
        elif t.isalpha():
            surface["short_alpha_fragment"] += 1
        elif any(ch.isdigit() for ch in t):
            surface["number_or_digit"] += 1
        elif "-" in t or "'" in t:
            surface["hyphen_or_apostrophe_piece"] += 1
        else:
            surface["other_piece"] += 1
    out["surface_counts"] = dict(surface)
    out["surface_frac_wordlike_ge4_or_capitalized"] = (surface["capitalized_wordlike"] + surface["lower_wordlike_len_ge4"]) / max(1, len(rows))
    # The current CSV may or may not contain target_id_in_source. If absent, record absence explicitly.
    bool_fields = [k for k in ["target_id_in_source", "target_id_in_decoy_source"] if k in rows[0]] if rows else []
    out["boolean_meta_fields_present"] = bool_fields
    for key in bool_fields:
        groups = defaultdict(list)
        for i, r in enumerate(rows):
            val = str(r.get(key, "")).lower() in {"true", "1", "yes"}
            groups[str(val)].append(true_adv[i])
        out[f"true_advantage_by_{key}"] = {g: summarize(vals) for g, vals in groups.items()}
    for gf in ["target_kind", "overlap_bin", "changed_frac_bin"]:
        groups = defaultdict(list)
        for i, r in enumerate(rows):
            groups[r.get(gf, "NA")].append(true_adv[i])
        out[f"true_advantage_by_{gf}"] = {g: summarize(vals) for g, vals in sorted(groups.items())}
    # High advantages often reveal copy-like source support. Store a small set of examples for manual reading.
    examples = []
    for i, (r, adv, red) in enumerate(sorted(zip(rows, true_adv, true_vs_base), key=lambda x: -x[1])[:30]):
        examples.append({
            "target_text": r.get("target_text"),
            "target_kind": r.get("target_kind"),
            "adv_true_vs_decoy": adv,
            "reduction_vs_base": red,
            "rank_base": r.get("rank_base"),
            "rank_true": r.get("rank_true"),
            "rank_decoy": r.get("rank_decoy"),
            "overlap_bin": r.get("overlap_bin"),
            "changed_frac_bin": r.get("changed_frac_bin"),
            "content_overlap": r.get("content_overlap"),
            "pair_id": r.get("pair_id"),
            "row_index": r.get("row_index"),
        })
    out["top_true_advantage_examples"] = examples
    return out


def private_flags(j):
    rr = j.get("route_readout", {})
    pc = j.get("private_readout", j.get("private_readout_comparison", {}))
    false_keys = []
    if j.get("mode") == "edit":
        false_keys = ["decoy"]
    elif j.get("mode") == "discourse":
        false_keys = ["reversed", "shuffled"]
    vals = {}
    for fk in false_keys:
        vals[f"private_true_advantage_vs_{fk}_test"] = pc.get(f"private_true_advantage_vs_{fk}_test_positive_good", {})
        vals[f"private_true_advantage_vs_{fk}_high_error"] = pc.get(f"private_true_advantage_vs_{fk}_high_error_positive_good", {})
    return {
        "route_readout": rr,
        "private_comparisons": vals,
        "raw_true_vs_false": {k: v for k, v in j.get("raw_frozen_token_value", {}).items() if k.startswith("true_vs_")},
        "sample": j.get("sample", {}),
        "build_info": j.get("build_info", {}),
    }


def discourse_summary():
    j = load_json(DISCOURSE_JSON)
    rows = read_csv(DISCOURSE_CSV)
    out = private_flags(j)
    true_vs_rev = [f(r, "nll_reversed") - f(r, "nll_true") for r in rows]
    true_vs_shuf = [f(r, "nll_shuffled") - f(r, "nll_true") for r in rows]
    out["csv_true_advantage_vs_reversed"] = summarize(true_vs_rev)
    out["csv_true_advantage_vs_shuffled"] = summarize(true_vs_shuf)
    out["source_counts"] = dict(Counter(r.get("source", "") for r in rows))
    by_source = defaultdict(list)
    by_source_shuf = defaultdict(list)
    for i, r in enumerate(rows):
        by_source[r.get("source", "")].append(true_vs_rev[i])
        by_source_shuf[r.get("source", "")].append(true_vs_shuf[i])
    out["by_source_true_vs_reversed"] = {k: summarize(v) for k, v in sorted(by_source.items())}
    out["by_source_true_vs_shuffled"] = {k: summarize(v) for k, v in sorted(by_source_shuf.items())}
    return out


def extract_mean(d):
    if isinstance(d, dict):
        return d.get("mean")
    return None


def main():
    edit = load_json(EDIT_JSON)
    edit_v2 = load_json(EDIT_PILOT_V2_JSON)
    disc = load_json(DISCOURSE_JSON)
    summary = {
        "status": "A02_EDIT_DISCOURSE_ROUTE_BOUNDARY_DONE",
        "inputs": {
            "a02_plan": str(PLAN),
            "a02_inventory": str(INV),
            "edit_main": str(EDIT_JSON),
            "edit_pilot_v2": str(EDIT_PILOT_V2_JSON),
            "discourse_main": str(DISCOURSE_JSON),
            "acs_synthesis": "research/notes/representation_and_objectives/acs_screen_synthesis.md",
        },
        "a02_edit_main_private": private_flags(edit),
        "a02_edit_pilot_v2_private": private_flags(edit_v2),
        "a02_discourse_private": private_flags(disc),
        "edit_main_csv": edit_csv_summary(EDIT_CSV),
        "edit_pilot_v2_csv": edit_csv_summary(EDIT_PILOT_V2_CSV),
        "discourse_csv": discourse_summary(),
    }
    # Route interpretation as direct data, not prose only.
    em = summary["a02_edit_main_private"]["private_comparisons"]["private_true_advantage_vs_decoy_test"].get("mean")
    ev2 = summary["a02_edit_pilot_v2_private"]["private_comparisons"]["private_true_advantage_vs_decoy_test"].get("mean")
    dv_rev = summary["a02_discourse_private"]["private_comparisons"]["private_true_advantage_vs_reversed_test"].get("mean")
    dv_shuf = summary["a02_discourse_private"]["private_comparisons"]["private_true_advantage_vs_shuffled_test"].get("mean")
    disc_raw_rev = summary["a02_discourse_private"]["raw_true_vs_false"].get("true_vs_reversed", {}).get("mean")
    disc_raw_shuf = summary["a02_discourse_private"]["raw_true_vs_false"].get("true_vs_shuffled", {}).get("mean")
    edit_surface_wordlike = summary["edit_main_csv"]["surface_frac_wordlike_ge4_or_capitalized"]
    summary["route_judgment"] = {
        "edit_main_private_advantage_mean": em,
        "edit_pilot_v2_private_advantage_mean": ev2,
        "discourse_private_true_vs_reversed_mean": dv_rev,
        "discourse_private_true_vs_shuffled_mean": dv_shuf,
        "discourse_raw_true_vs_reversed_mean": disc_raw_rev,
        "discourse_raw_true_vs_shuffled_mean": disc_raw_shuf,
        "edit_main_wordlike_fraction": edit_surface_wordlike,
        "interpretation": [
            "Main edit-state discriminator has a large token-value/private signal, but the object is changed-token prediction with the source prepended; many targets are surface subword pieces and the source-absent/low-capacity pilot did not preserve private residual value.",
            "Intra-row discourse has abundant legal substrate but its true-vs-reversed separation is near zero while true-vs-shuffled is moderate, so it is mostly topical/neighbor state rather than a directional transformation.",
            "Neither object yet satisfies the stricter research requirement of dense natural equivariant context-candidate coupling with a known permutation/directional change among the same plausible alternatives.",
            "A02 should continue owning edit/discourse discriminator repair; A01 should not launch a training fork from these results, and should instead design an independent representation-forming mechanism or a tiny CPU/GPU-free transformation-inventory test that looks for explicit source-absent lexical/semantic rewrite operators before any training."
        ],
    }
    out_json = OUT_DIR / "route_boundary_edit_discourse.json"
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    note = (OUT_DIR.parents[4] / 'research/documents/representation_and_objectives/data/route_boundary_edit_discourse/route_boundary_edit_discourse.md')
    lines = []
    lines.append("# research route boundary from A02 edit/discourse evidence")
    lines.append("")
    lines.append("## What A02 evidence says")
    lines.append(f"- Edit main private true-vs-decoy held-out NLL advantage: `{em}`; pilot-v2 small run advantage: `{ev2}`.")
    lines.append(f"- Edit main raw true-vs-decoy advantage: `{summary['a02_edit_main_private']['raw_true_vs_false'].get('true_vs_decoy', {}).get('mean')}`; pilot-v2 raw advantage: `{summary['a02_edit_pilot_v2_private']['raw_true_vs_false'].get('true_vs_decoy', {}).get('mean')}`.")
    lines.append(f"- Edit targets in main CSV are wordlike length>=4 or capitalized only `{edit_surface_wordlike:.4f}` of items; many high-advantage targets are subword/morphological pieces.")
    lines.append(f"- Discourse raw true-vs-reversed `{disc_raw_rev}` versus true-vs-shuffled `{disc_raw_shuf}`; private true-vs-reversed `{dv_rev}`, private true-vs-shuffled `{dv_shuf}`.")
    lines.append("")
    lines.append("## Interpretation for A01")
    lines.append("ACS closed single-context hard-negative sharpening; A02 edit/discourse evidence is useful but not yet a training route for A01. The edit object contains real legal source-conditioned changed-token value, but current evidence does not yet show an explicit dense equivariant transformation in which a context change induces a known permutation or direction among the same plausible alternatives. The discourse object is abundant but its order direction is nearly absent: true and reversed neighbors behave almost the same while shuffled neighbors lose value, so this is mostly topic/entity/register state.")
    lines.append("")
    lines.append("## Next useful work")
    lines.append("Do not duplicate A02's edit-state or intra-row discriminator work and do not launch GPU training from these artifacts. A valuable next A01 construction should either (a) build a small transformation-inventory test that filters source→rewrite pairs for source-absent, full-word, semantic/operator edits and tests whether such operators are dense enough to matter, or (b) invent a different representation-forming mechanism not based on correlated paired MLM/discriminator examples. Any later training screen must have an explicit legal false-structure control and fixed natural GlobalPIQA hard52/EWoK readouts.")
    lines.append("")
    lines.append(f"JSON: `{out_json}`")
    note.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": summary["status"], "json": str(out_json), "note": str(note)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
