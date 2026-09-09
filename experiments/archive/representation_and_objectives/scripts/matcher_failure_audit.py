#!/usr/bin/env python3
"""Audit why research no-name-vocab learned span discovery failed.

The central question is whether the CharGRU matcher failed at equality itself,
failed to route enough probability away from the neither channel, or failed only
in the relational comparison path. The script reads saved predictions only.
"""
from __future__ import annotations
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean

PROJECT = Path("experiments/archive/representation_and_objectives")
OUT = PROJECT / "data/matcher_failure_audit"
RUNS = {
    "lexical_shared_plus": PROJECT / "data/learned_shared/learned_shared_trunk_bs+1_seed29600",
    "lexical_untied_plus": PROJECT / "data/learned_untied/learned_untied_bs+1_seed29600",
    "noname_shared_plus": PROJECT / "data/noname_shared_bsplus_r2/learned_shared_trunk_bs+1_seed29600",
    "noname_shared_minus": PROJECT / "data/noname_shared_bsminus_r2/learned_shared_trunk_bs-1_seed29600",
}


def load_jsonl(p: Path):
    rows = []
    if not p.exists():
        return rows
    with p.open() as f:
        for line in f:
            s = line.strip()
            if s:
                rows.append(json.loads(s))
    return rows


def avg(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else None


def summarize_match_recs(rows):
    recs = []
    for r in rows:
        if r.get("candidate_index") != 0 or not r.get("is_changed"):
            continue
        m = r.get("matching") or {}
        if not m:
            continue
        rec = {
            "suite": r.get("suite"),
            "relation": r.get("relation"),
            "relation_family": r.get("relation_family"),
            "candidate": m.get("cand_name"),
            "other": m.get("other_name"),
            "cand_len": len(str(m.get("cand_name", ""))),
            "other_len": len(str(m.get("other_name", ""))),
            "cand_p_cand": float(m.get("cand_p_cand", 0.0)),
            "cand_p_other": float(m.get("cand_p_other", 0.0)),
            "cand_p_neither": float(m.get("cand_p_neither", 0.0)),
            "other_p_cand": float(m.get("other_p_cand", 0.0)),
            "other_p_other": float(m.get("other_p_other", 0.0)),
            "other_p_neither": float(m.get("other_p_neither", 0.0)),
        }
        rec["cand_rel_ok"] = rec["cand_p_cand"] > rec["cand_p_other"]
        rec["other_rel_ok"] = rec["other_p_other"] > rec["other_p_cand"]
        rec["cand_wins_gate"] = rec["cand_p_cand"] > max(rec["cand_p_other"], rec["cand_p_neither"])
        rec["other_wins_gate"] = rec["other_p_other"] > max(rec["other_p_cand"], rec["other_p_neither"])
        rec["cand_above_half"] = rec["cand_p_cand"] > 0.5
        rec["other_above_half"] = rec["other_p_other"] > 0.5
        rec["both_rel_ok"] = rec["cand_rel_ok"] and rec["other_rel_ok"]
        rec["both_win_gate"] = rec["cand_wins_gate"] and rec["other_wins_gate"]
        rec["both_above_half"] = rec["cand_above_half"] and rec["other_above_half"]
        recs.append(rec)
    return recs


def summarize_group(recs):
    if not recs:
        return {"n": 0}
    keys = ["cand_p_cand", "cand_p_other", "cand_p_neither",
            "other_p_cand", "other_p_other", "other_p_neither"]
    bools = ["cand_rel_ok", "other_rel_ok", "cand_wins_gate", "other_wins_gate",
             "cand_above_half", "other_above_half", "both_rel_ok", "both_win_gate", "both_above_half"]
    out = {"n": len(recs)}
    out.update({"mean_" + k: avg(r[k] for r in recs) for k in keys})
    out.update({k + "_frac": avg(1.0 if r[k] else 0.0 for r in recs) for k in bools})
    return out


def by_field(recs, field):
    groups = defaultdict(list)
    for r in recs:
        groups[str(r.get(field))].append(r)
    return {k: summarize_group(v) for k, v in sorted(groups.items())}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    result = {}
    for label, rd in RUNS.items():
        state = load_jsonl(rd / "state_predictions.jsonl")
        recs = summarize_match_recs(state)
        result[label] = {
            "path": str(rd),
            "exists": rd.exists(),
            "n_state_rows": len(state),
            "overall": summarize_group(recs),
            "by_suite": by_field(recs, "suite"),
            "by_relation_family": by_field(recs, "relation_family"),
            "by_candidate": by_field(recs, "candidate"),
            "by_other": by_field(recs, "other"),
        }
    (OUT / "matcher_failure_audit.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")

    lines = ["# research matcher failure audit", "",
             "Rows are changed-event eval rows with candidate_index=0. `rel_ok` asks whether the true name channel beats the opposite name channel; `wins_gate` asks whether it also beats the neither channel. The latter is the actual routing condition for replacing word embeddings by candidate/other embeddings.", "",
             "## Overall", "",
             "| run | n | cand rel ok | other rel ok | cand wins gate | other wins gate | both wins gate | cand >0.5 | other >0.5 | mean cand_neither | mean other_neither |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for label, d in result.items():
        o = d["overall"]
        lines.append("| {} | {} | {:.3f} | {:.3f} | {:.3f} | {:.3f} | {:.3f} | {:.3f} | {:.3f} | {:.3f} | {:.3f} |".format(
            label, o.get("n",0), o.get("cand_rel_ok_frac",0.0) or 0.0, o.get("other_rel_ok_frac",0.0) or 0.0,
            o.get("cand_wins_gate_frac",0.0) or 0.0, o.get("other_wins_gate_frac",0.0) or 0.0,
            o.get("both_win_gate_frac",0.0) or 0.0, o.get("cand_above_half_frac",0.0) or 0.0,
            o.get("other_above_half_frac",0.0) or 0.0, o.get("mean_cand_p_neither",0.0) or 0.0,
            o.get("mean_other_p_neither",0.0) or 0.0))
    lines += ["", "## By suite", ""]
    for label, d in result.items():
        lines.append(f"### {label}")
        lines.append("| suite | n | both rel ok | both wins gate | cand wins gate | other wins gate |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        for suite, o in d["by_suite"].items():
            lines.append("| {} | {} | {:.3f} | {:.3f} | {:.3f} | {:.3f} |".format(
                suite, o.get("n",0), o.get("both_rel_ok_frac",0.0) or 0.0,
                o.get("both_win_gate_frac",0.0) or 0.0, o.get("cand_wins_gate_frac",0.0) or 0.0,
                o.get("other_wins_gate_frac",0.0) or 0.0))
        lines.append("")
    ((OUT.parents[4] / 'research/documents/representation_and_objectives/data/matcher_failure_audit/matcher_failure_audit.md')).write_text("\n".join(lines) + "\n")
    print(json.dumps({"status": "MATCHER_FAILURE_AUDIT_COMPLETE",
                      "md": str((OUT.parents[4] / 'research/documents/representation_and_objectives/data/matcher_failure_audit/matcher_failure_audit.md')),
                      "json": str(OUT / "matcher_failure_audit.json")}, indent=2))

if __name__ == "__main__":
    main()
