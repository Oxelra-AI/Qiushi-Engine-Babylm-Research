#!/usr/bin/env python3
"""Sharper analysis for Step019b embedding-role decomposition.

Reads the Step019b results JSON and produces a compact scientific note/JSON that
focuses on invariants, per-seed endpoint transplants, and the mechanism contrast:
held input-row rewriting versus broader contextual/readout specialization.
"""
import argparse, json, math
from pathlib import Path
from statistics import mean

METRIC_PATHS = {
    "train_top4": ("qf", "std", "ctx_top1"),
    "train_b": ("qf", "bswap", "mean"),
    "train_b_frac": ("qf", "bswap", "frac_pos"),
    "train_q": ("qf", "qswap", "margin"),
    "train_q_frac": ("qf", "qswap", "frac_pos"),
    "train_q_both": ("qf", "qswap", "both"),
    "train_sel": ("qf", "corrupt", "novel_selectivity"),
    "held_top4": ("qf", "held", "ctx_top1"),
    "held_b": ("qf", "held_bswap", "mean"),
    "held_b_frac": ("qf", "held_bswap", "frac_pos"),
    "held_q": ("qf", "held_qswap", "margin"),
    "held_q_frac": ("qf", "held_qswap", "frac_pos"),
    "held_q_both": ("qf", "held_qswap", "both"),
    "held_sel": ("qf", "held_corrupt", "novel_selectivity"),
    "blocked_train_top4": ("qf_blocked", "std", "ctx_top1"),
    "blocked_train_b": ("qf_blocked", "bswap", "mean"),
    "held_input_l2": ("drift", "input", "held_ent", "l2_mean"),
    "held_input_cos": ("drift", "input", "held_ent", "cos_mean"),
    "held_output_l2": ("drift", "output", "held_ent", "l2_mean"),
    "held_pair_disp_cos": ("drift", "input", "held_pair_displacement_cos"),
    "train_input_l2": ("drift", "input", "train_ent", "l2_mean"),
    "rwt_input_l2": ("drift", "input", "rwt", "l2_mean"),
    "rwt_output_l2": ("drift", "output", "rwt", "l2_mean"),
}
CORE = ["train_top4", "train_b", "train_q", "train_q_both", "train_sel", "held_top4", "held_b", "held_sel"]
HYBRID_NAMES = [
    "final_FF",
    "final_PF_inputprep_outputfinal",
    "final_FP_inputfinal_outputprep",
    "final_PP_inputprep_outputprep",
    "prep_PP_baseline",
    "prep_FP_inputfinal_outputprep",
]


def get(obj, path, default=None):
    cur = obj
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def num(x):
    return None if x is None else float(x)


def metrics(rec):
    return {k: num(get(rec, p)) for k, p in METRIC_PATHS.items()}


def recs_for(data, seed=None, branch=None):
    rs = data.get("branches", [])
    if seed is not None:
        rs = [r for r in rs if int(r.get("seed")) == int(seed)]
    if branch is not None:
        rs = [r for r in rs if r.get("branch") == branch]
    return sorted(rs, key=lambda r: (int(r.get("seed", 0)), r.get("branch", ""), int(r.get("epoch", 0))))


def nearest(rs, branch_epoch):
    if not rs:
        return None
    return min(rs, key=lambda r: abs(int(r.get("branch_epoch", 10**9)) - int(branch_epoch)))


def max_abs_diff(a, b, keys=CORE):
    vals = []
    for k in keys:
        x = metrics(a).get(k); y = metrics(b).get(k)
        if x is None or y is None:
            continue
        vals.append(abs(x - y))
    return None if not vals else max(vals)


def train_behavior(m):
    return (m["train_top4"] is not None and m["train_top4"] >= 0.95 and
            m["train_b"] is not None and m["train_b"] >= 5.0 and
            m["train_b_frac"] is not None and m["train_b_frac"] >= 0.95 and
            m["train_q"] is not None and m["train_q"] >= 5.0 and
            m["train_q_frac"] is not None and m["train_q_frac"] >= 0.95 and
            m["train_q_both"] is not None and m["train_q_both"] >= 0.90 and
            m["train_sel"] is not None and m["train_sel"] >= 0.80)


def high_held(m):
    return (m["held_top4"] is not None and m["held_top4"] >= 0.75 and
            m["held_b"] is not None and m["held_b"] >= 5.0 and
            m["held_sel"] is not None and m["held_sel"] >= 0.50)


def gap_closed(prep, final, test, key):
    den = prep.get(key) - final.get(key)
    if den is None or test.get(key) is None or abs(den) < 1e-9:
        return None
    return (test[key] - final[key]) / den


def summarize_branch(data, branch):
    seeds = [int(s) for s in data["config"]["seeds"]]
    finals = []
    be500 = []
    for sd in seeds:
        rs = recs_for(data, sd, branch)
        if not rs:
            continue
        finals.append(metrics(rs[-1]))
        r500 = nearest(rs, 500)
        if r500 is not None:
            be500.append(metrics(r500))
    def avg(key, arr=finals):
        vals = [m[key] for m in arr if m.get(key) is not None]
        return None if not vals else round(mean(vals), 6)
    return {
        "n": len(finals),
        "train_strong": sum(1 for m in finals if train_behavior(m)),
        "held_high": sum(1 for m in finals if high_held(m)),
        "final_train_top4": avg("train_top4"),
        "final_held_top4": avg("held_top4"),
        "final_held_b": avg("held_b"),
        "final_held_sel": avg("held_sel"),
        "final_blocked_train_top4": avg("blocked_train_top4"),
        "final_held_input_l2": avg("held_input_l2"),
        "be500_train_top4": avg("train_top4", be500),
        "be500_held_top4": avg("held_top4", be500),
        "be500_held_b": avg("held_b", be500),
        "be500_held_sel": avg("held_sel", be500),
    }


def compare_step018(data, path):
    if not path or not Path(path).exists():
        return None
    s18 = json.load(open(path))
    out = []
    for sd in data["config"]["seeds"]:
        rs19 = recs_for(data, sd, "tied_carry_full")
        rs18 = [r for r in s18.get("records", []) if int(r.get("seed")) == int(sd) and r.get("arm") == "prep_bound_ans_then_full"]
        # Match Step019b nested qf records to research flat qf records by epoch.
        by18 = {int(r["epoch"]): r for r in rs18}
        diffs = []
        for r19 in rs19:
            ep = int(r19["epoch"])
            if ep not in by18:
                continue
            # Wrap research-like flat record into the same shape expected by metrics.
            r18 = {"qf": {k: by18[ep][k] for k in ["std", "held", "bswap", "qswap", "corrupt", "held_bswap", "held_qswap", "held_corrupt"]},
                   "qf_blocked": r19.get("qf_blocked", {}), "drift": r19.get("drift", {})}
            diffs.append(max_abs_diff(r19, r18, keys=["train_top4", "train_b", "train_q", "train_q_both", "train_sel", "held_top4", "held_b", "held_sel"]))
        out.append({"seed": int(sd), "matched_epochs": len(diffs), "max_core_metric_abs_diff": None if not diffs else max(diffs)})
    return out


def analyze(data, path=None):
    seeds = [int(s) for s in data["config"]["seeds"]]
    branches = [b["name"] for b in data["branches_spec"]]
    prep_last = {}
    for sd in seeds:
        P = int(data["config"]["prep_epochs"][str(sd)])
        prs = [r for r in data.get("prep", []) if int(r.get("seed")) == sd]
        prs = sorted(prs, key=lambda r: int(r["epoch"]))
        prep_last[sd] = prs[-1] if prs else None
    A = {"config": data["config"], "branch_summary": {}, "by_seed": {}, "invariants": {}, "match": compare_step018(data, path)}
    for br in branches:
        A["branch_summary"][br] = summarize_branch(data, br)
    inv = {"prep_reproduction": [], "frozen_rows": [], "hybrid_FF": [], "hybrid_row_role": []}
    for sd in seeds:
        A["by_seed"][str(sd)] = {}
        p0 = prep_last[sd]
        p0m = metrics({"qf": p0["qf"], "qf_blocked": p0["qf_blocked"], "drift": p0["drift"]}) if p0 and "qf" in p0 else None
        for br in branches:
            rs = recs_for(data, sd, br)
            if not rs:
                continue
            start, final = rs[0], rs[-1]
            sm, fm = metrics(start), metrics(final)
            inv["prep_reproduction"].append({"seed": sd, "branch": br, "max_core_metric_abs_diff": None if p0 is None else max_abs_diff(start, {"qf": p0["qf"], "qf_blocked": p0["qf_blocked"], "drift": p0["drift"]})})
            if "freeze" in br:
                inv["frozen_rows"].append({"seed": sd, "branch": br, "final_held_input_l2": fm.get("held_input_l2")})
            item = {
                "prep": sm,
                "be500": metrics(nearest(rs, 500)) if nearest(rs, 500) is not None else None,
                "final": fm,
                "delta_final_minus_prep": {k: None if fm.get(k) is None or sm.get(k) is None else fm[k] - sm[k]
                                           for k in ["train_top4", "train_b", "train_sel", "held_top4", "held_b", "held_sel"]},
                "train_strong_final": train_behavior(fm),
                "held_high_final": high_held(fm),
            }
            Hraw = data.get("hybrids", {}).get(str(sd), {}).get(br)
            if Hraw:
                Hm = {name: metrics(pack) for name, pack in Hraw.items() if name in HYBRID_NAMES}
                item["hybrid"] = Hm
                ff = Hraw.get("final_FF")
                inv["hybrid_FF"].append({"seed": sd, "branch": br, "max_core_metric_abs_diff_vs_final": max_abs_diff(final, ff) if ff else None})
                if all(n in Hm for n in ["final_FF", "final_PF_inputprep_outputfinal", "final_FP_inputfinal_outputprep", "final_PP_inputprep_outputprep", "prep_PP_baseline", "prep_FP_inputfinal_outputprep"]):
                    item["input_restore_gap_closed"] = {k: gap_closed(sm, Hm["final_FF"], Hm["final_PF_inputprep_outputfinal"], k) for k in ["held_top4", "held_b", "held_sel"]}
                    item["output_restore_gap_closed"] = {k: gap_closed(sm, Hm["final_FF"], Hm["final_FP_inputfinal_outputprep"], k) for k in ["held_top4", "held_b", "held_sel"]}
                    item["both_restore_gap_closed"] = {k: gap_closed(sm, Hm["final_FF"], Hm["final_PP_inputprep_outputprep"], k) for k in ["held_top4", "held_b", "held_sel"]}
                    item["reverse_final_input_delta"] = {k: Hm["prep_FP_inputfinal_outputprep"].get(k) - Hm["prep_PP_baseline"].get(k) for k in ["held_top4", "held_b", "held_sel"]}
                    inv["hybrid_row_role"].append({
                        "seed": sd, "branch": br,
                        "PF_input_l2": Hm["final_PF_inputprep_outputfinal"].get("held_input_l2"),
                        "PF_output_l2": Hm["final_PF_inputprep_outputfinal"].get("held_output_l2"),
                        "FP_input_l2": Hm["final_FP_inputfinal_outputprep"].get("held_input_l2"),
                        "FP_output_l2": Hm["final_FP_inputfinal_outputprep"].get("held_output_l2"),
                        "PP_input_l2": Hm["final_PP_inputprep_outputprep"].get("held_input_l2"),
                        "PP_output_l2": Hm["final_PP_inputprep_outputprep"].get("held_output_l2"),
                    })
            A["by_seed"][str(sd)][br] = item
    A["invariants"] = inv
    return A


def f3(x):
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return ""
    return f"{x:.3f}" if isinstance(x, float) else str(x)


def write_note(A, note_path, result_path):
    lines = []
    lines.append("# Step019b analysis: separating held-row drift from distributed specialization")
    lines.append("")
    lines.append(f"Read result JSON: `{result_path}`.")
    lines.append("")
    lines.append("## Branch means")
    lines.append("")
    lines.append("| branch | n | train strong | held high | final train4 | final held4 | final heldB | final heldSel | blocked train4 | held-input L2 | be500 held4 |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for br, sm in A["branch_summary"].items():
        lines.append(f"| {br} | {sm['n']} | {sm['train_strong']} | {sm['held_high']} | {f3(sm['final_train_top4'])} | {f3(sm['final_held_top4'])} | {f3(sm['final_held_b'])} | {f3(sm['final_held_sel'])} | {f3(sm['final_blocked_train_top4'])} | {f3(sm['final_held_input_l2'])} | {f3(sm['be500_held_top4'])} |")
    lines.append("")
    lines.append("## Invariants checked")
    lines.append("")
    pr = A["invariants"].get("prep_reproduction", [])
    mx_pr = max([x["max_core_metric_abs_diff"] for x in pr if x["max_core_metric_abs_diff"] is not None], default=None)
    ff = A["invariants"].get("hybrid_FF", [])
    mx_ff = max([x["max_core_metric_abs_diff_vs_final"] for x in ff if x["max_core_metric_abs_diff_vs_final"] is not None], default=None)
    fr = A["invariants"].get("frozen_rows", [])
    mx_fr = max([abs(x["final_held_input_l2"]) for x in fr if x["final_held_input_l2"] is not None], default=None)
    lines.append(f"- Branch epoch-0 records reproduce preparation metrics with max core absolute difference `{f3(mx_pr)}`.")
    lines.append(f"- Untied `final_FF` hybrids reproduce final tied behavior with max core absolute difference `{f3(mx_ff)}`.")
    lines.append(f"- Frozen held-input/shared rows have maximum final held-input L2 `{f3(mx_fr)}` across frozen branches.")
    if A.get("match") is not None:
        lines.append("- research comparison for `tied_carry_full`:")
        for x in A["match"]:
            lines.append(f"  - seed {x['seed']}: matched epochs {x['matched_epochs']}, max core absolute difference `{f3(x['max_core_metric_abs_diff'])}`.")
    lines.append("")
    lines.append("## Per-seed endpoint deltas")
    lines.append("")
    lines.append("| seed | branch | prep h4 | final train4 | final h4 | Δh4 | final hB | ΔhB | final hSel | ΔhSel | final hL2 | blocked train4 |")
    lines.append("|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for sd, byb in A["by_seed"].items():
        for br, item in byb.items():
            p, f, d = item["prep"], item["final"], item["delta_final_minus_prep"]
            lines.append(f"| {sd} | {br} | {f3(p['held_top4'])} | {f3(f['train_top4'])} | {f3(f['held_top4'])} | {f3(d['held_top4'])} | {f3(f['held_b'])} | {f3(d['held_b'])} | {f3(f['held_sel'])} | {f3(d['held_sel'])} | {f3(f['held_input_l2'])} | {f3(f['blocked_train_top4'])} |")
    lines.append("")
    lines.append("## Tied endpoint hybrids")
    lines.append("")
    lines.append("PF restores only held input rows to preparation values in an untied final copy; FP restores only held output rows; PP restores both. The reverse row transplant inserts final held input rows into the preparation network.")
    lines.append("")
    lines.append("| seed | branch | prep h4 | FF h4 | PF h4 | FP h4 | PP h4 | prep+final-input h4 | PF gap h4 | PF gap hB | PF gap hSel | reverse Δh4 | reverse ΔhB |")
    lines.append("|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for sd, byb in A["by_seed"].items():
        for br, item in byb.items():
            if "hybrid" not in item:
                continue
            H = item["hybrid"]; p = item["prep"]
            gc = item.get("input_restore_gap_closed", {})
            rev = item.get("reverse_final_input_delta", {})
            lines.append(f"| {sd} | {br} | {f3(p['held_top4'])} | {f3(H['final_FF']['held_top4'])} | {f3(H['final_PF_inputprep_outputfinal']['held_top4'])} | {f3(H['final_FP_inputfinal_outputprep']['held_top4'])} | {f3(H['final_PP_inputprep_outputprep']['held_top4'])} | {f3(H['prep_FP_inputfinal_outputprep']['held_top4'])} | {f3(gc.get('held_top4'))} | {f3(gc.get('held_b'))} | {f3(gc.get('held_sel'))} | {f3(rev.get('held_top4'))} | {f3(rev.get('held_b'))} |")
    lines.append("")
    lines.append("## Scientific reading rule")
    lines.append("")
    lines.append("A selective PF rescue would show that the final contextual network and output readout still know how to use the preparation-era held-symbol input code. A reverse transplant that damages the preparation network would show that the final held input rows are sufficient to break a network that still implements held binding. If neither occurs while trained binding remains strong, then held-transfer loss is not explained by held input rows alone and the next work should measure distributed contextual/readout specialization, attribute/RWT geometry, or training-state effects.")
    note = Path(note_path); note.parent.mkdir(parents=True, exist_ok=True)
    note.write_text("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--result", default="experiments/archive/functional_learning/data/revision_019b_embedding_role_decomposition/results.json")
    ap.add_argument("--out", default="experiments/archive/functional_learning/data/revision_019b_embedding_role_decomposition/analysis.json")
    ap.add_argument("--note", default="research/notes/functional_learning/019b_embedding_role_decomposition_analysis.md")
    ap.add_argument("--research", default="experiments/archive/functional_learning/data/budget_matched_full_objective/results.json")
    A = ap.parse_args()
    data = json.load(open(A.result))
    analysis = analyze(data, A.research)
    outp = Path(A.out); outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(analysis, indent=2))
    write_note(analysis, A.note, A.result)
    print(json.dumps({"status": "ok", "analysis": str(outp), "note": A.note,
                      "branches": list(analysis["branch_summary"].keys())}, indent=2))


if __name__ == "__main__":
    main()
