#!/usr/bin/env python3
"""research: third-seed CLEAN baseline decomposition for held-out probes.

Merges research seed43222 VIEW/REPEAT probe rows with research parallel CLEAN rows.
Computes R-C active cost and V-C/V-R content-conditioning at seed43222, plus copy
and Entity cue-ablation contrasts. This is CPU-only after the forward-pass scores.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import csv
import json
import math
import pathlib
import statistics
from collections import defaultdict
from typing import Any

ROOT = _public_path('experiments/archive/relation_learning/scripts/seed43222_clean_probe_decomposition.py')
ROOT = _PUBLIC_ROOT
WS = _public_path('experiments/archive/relation_learning')
VR_DIR = _public_path('experiments/archive/relation_learning/data/seed43222_probes')
C_DIR = _public_path('experiments/archive/relation_learning/data/seed43222_parallel_clean_probes')
OUT = _public_path('experiments/archive/relation_learning/data/seed43222_threearm_probe_decomposition')
NOTE = _public_path('research/notes/relation_learning/seed43222_threearm_probe_decomposition.md')
CKS = ["chck_80M", "chck_90M", "chck_100M"]


def rel(p: pathlib.Path) -> str:
    try: return str(p.relative_to(ROOT))
    except Exception: return str(p)


def read_csv(p: pathlib.Path) -> list[dict[str, str]]:
    with p.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(p: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        p.write_text("\n", encoding="utf-8"); return
    fields = list(rows[0].keys())
    with p.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)


def mean(xs):
    vals = [float(x) for x in xs if x is not None and math.isfinite(float(x))]
    return statistics.mean(vals) if vals else float("nan")


def sd(xs):
    vals = [float(x) for x in xs if x is not None and math.isfinite(float(x))]
    return statistics.stdev(vals) if len(vals) > 1 else float("nan")


def read_rows(kind: str) -> list[dict[str, Any]]:
    rows = []
    files = []
    prefix = f"{kind}_pair_rows" if kind in {"rewrite", "copy"} else "entity_ablation_rows"
    for arm in ["D_V_43222", "D_R_43222"]:
        for ck in CKS:
            files.append(VR_DIR / f"{prefix}_{arm}_{ck}.csv")
    for ck in CKS:
        files.append(C_DIR / f"{prefix}_D_C_43222_{ck}.csv")
    for p in files:
        if not p.exists():
            raise FileNotFoundError(p)
        for r in read_csv(p):
            rr: dict[str, Any] = dict(r)
            if "gain" in rr and rr["gain"] != "":
                rr["gain"] = float(rr["gain"])
            for k in ["true_source_nll", "unrelated_source_nll", "repeated_nll", "unrepeated_nll", "margin_full", "effect_no_initial_qbox", "effect_no_last_relevant_update", "effect_no_all_relevant_updates"]:
                if k in rr and rr[k] != "":
                    rr[k] = float(rr[k])
            rows.append(rr)
    return rows


def role_of(arm: str) -> str:
    return arm.split("_")[1]


def rewrite_summary(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by = defaultdict(list)
    for r in rows:
        if r["checkpoint"] not in CKS: continue
        for g in ["ALL", "token_" + r["token_class"]]:
            by[(role_of(r["arm"]), r["checkpoint"], g)].append(r)
    arm_ck_group = []
    for (role, ck, g), xs in sorted(by.items()):
        arm_ck_group.append({
            "role": role, "checkpoint": ck, "group": g, "n": len(xs),
            "mean_gain": mean([x["gain"] for x in xs]),
            "mean_true_source_nll": mean([x["true_source_nll"] for x in xs]),
            "mean_unrelated_source_nll": mean([x["unrelated_source_nll"] for x in xs]),
        })
    late = defaultdict(list)
    for r in arm_ck_group:
        late[(r["role"], r["group"])].append(r)
    late_means = {}
    for k, vals in late.items():
        late_means[k] = {
            "n_min": min(v["n"] for v in vals),
            "gain": mean([v["mean_gain"] for v in vals]),
            "T": mean([v["mean_true_source_nll"] for v in vals]),
            "U": mean([v["mean_unrelated_source_nll"] for v in vals]),
        }
    contrasts = []
    for g in sorted({k[1] for k in late_means}):
        for a,b in [("R","C"),("V","C"),("V","R")]:
            A = late_means.get((a,g)); B = late_means.get((b,g))
            if not A or not B: continue
            dG = A["gain"] - B["gain"]
            dT = A["T"] - B["T"]
            dU = A["U"] - B["U"]
            contrasts.append({
                "family": "rewrite", "seed": 43222, "group": g, "contrast": f"{a}minus{b}",
                "n_min": min(A["n_min"], B["n_min"]),
                "gain_delta": dG, "true_source_delta": dT, "unrelated_source_delta": dU,
                "excess_true_cost": dT - dU,
            })
    return arm_ck_group, contrasts


def copy_summary(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by = defaultdict(list)
    for r in rows:
        for g in ["ALL", "span_" + str(r.get("span_len"))]:
            by[(role_of(r["arm"]), r["checkpoint"], g)].append(r)
    arm_ck_group = []
    for (role, ck, g), xs in sorted(by.items()):
        arm_ck_group.append({
            "role": role, "checkpoint": ck, "group": g, "n": len(xs),
            "mean_gain": mean([x["gain"] for x in xs]),
            "mean_repeated_nll": mean([x["repeated_nll"] for x in xs]),
            "mean_unrepeated_nll": mean([x["unrepeated_nll"] for x in xs]),
        })
    late = defaultdict(list)
    for r in arm_ck_group:
        late[(r["role"], r["group"])].append(r)
    late_means = {k: {"n_min": min(v["n"] for v in vals), "gain": mean([v["mean_gain"] for v in vals]), "rep": mean([v["mean_repeated_nll"] for v in vals]), "unrep": mean([v["mean_unrepeated_nll"] for v in vals])} for k, vals in late.items()}
    contrasts = []
    for g in sorted({k[1] for k in late_means}):
        for a,b in [("R","C"),("V","C"),("R","V")]:
            A = late_means.get((a,g)); B = late_means.get((b,g))
            if not A or not B: continue
            contrasts.append({
                "family": "copy", "seed": 43222, "group": g, "contrast": f"{a}minus{b}",
                "n_min": min(A["n_min"], B["n_min"]), "gain_delta": A["gain"] - B["gain"],
                "repeated_nll_delta": A["rep"] - B["rep"], "unrepeated_nll_delta": A["unrep"] - B["unrep"],
            })
    return arm_ck_group, contrasts


def entity_summary(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by = defaultdict(list)
    for r in rows:
        relu = int(float(r["relevant_updates"]))
        groups = ["ALL", f"rel{relu}"]
        if relu >= 2: groups.append("rel_ge2")
        if relu >= 3: groups.append("rel_ge3")
        for g in groups:
            by[(role_of(r["arm"]), r["checkpoint"], g)].append(r)
    arm_ck_group = []
    for (role, ck, g), xs in sorted(by.items()):
        arm_ck_group.append({
            "role": role, "checkpoint": ck, "group": g, "n": len(xs),
            "margin_full": mean([x["margin_full"] for x in xs]),
            "effect_no_initial": mean([x["effect_no_initial_qbox"] for x in xs]),
            "effect_no_last": mean([x["effect_no_last_relevant_update"] for x in xs]),
            "effect_no_all": mean([x["effect_no_all_relevant_updates"] for x in xs]),
        })
    late = defaultdict(list)
    for r in arm_ck_group:
        late[(r["role"], r["group"])].append(r)
    late_means = {k: {kk: mean([v[kk] for v in vals]) for kk in ["margin_full", "effect_no_initial", "effect_no_last", "effect_no_all"]} | {"n_min": min(v["n"] for v in vals)} for k, vals in late.items()}
    contrasts = []
    for g in sorted({k[1] for k in late_means}):
        for a,b in [("V","C"),("R","C"),("V","R")]:
            A = late_means.get((a,g)); B = late_means.get((b,g))
            if not A or not B: continue
            rec = {"family": "entity_ablation", "seed": 43222, "group": g, "contrast": f"{a}minus{b}", "n_min": min(A["n_min"], B["n_min"])}
            for kk in ["margin_full", "effect_no_initial", "effect_no_last", "effect_no_all"]:
                rec[kk + "_delta"] = A[kk] - B[kk]
            contrasts.append(rec)
    return arm_ck_group, contrasts


def load_prior_r_costs():
    p = _public_path('experiments/archive/relation_learning/data/relation_decomposition/rewrite_gain_terms_late_summary.csv')
    vals = []
    if p.exists():
        for r in read_csv(p):
            if r.get("contrast") == "RminusC" and r.get("group") in {"token_nonoverlap", "nonoverlap", "tokenizer_nonoverlap"}:
                vals.append(r)
    return vals


def write_note(rewrite_con, copy_con, entity_con):
    def pick(rows, fam, group, contrast, field):
        for r in rows:
            if r.get("family") == fam and r.get("group") == group and r.get("contrast") == contrast:
                return r.get(field)
        return None
    def fmt(x):
        return "NA" if x is None else f"{float(x):+.4f}"
    lines = []
    lines.append("# research seed43222 CLEAN baseline probe decomposition")
    lines.append("")
    lines.append("This analysis merges seed43222 VIEW/REPEAT probe rows from research with the completed research parallel CLEAN run scored in research. It is the first seed43222 test of whether REPEAT falls below CLEAN on nonidentical true-source use, the research active-cost center.")
    lines.append("")
    lines.append("## Rewrite content-conditioning")
    lines.append("")
    lines.append("| group | contrast | gain Δ | true-source Δ | unrelated-source Δ | excess true-source cost | n |")
    lines.append("|---|---|---:|---:|---:|---:|---:|")
    for r in rewrite_con:
        if r["group"] in ["token_nonoverlap", "ALL", "token_overlap"] and r["contrast"] in ["RminusC", "VminusC", "VminusR"]:
            lines.append(f"| {r['group']} | {r['contrast']} | {fmt(r['gain_delta'])} | {fmt(r['true_source_delta'])} | {fmt(r['unrelated_source_delta'])} | {fmt(r['excess_true_cost'])} | {r['n_min']} |")
    lines.append("")
    lines.append("For token-nonoverlap targets, seed43222 reproduces the active cost of exact recurrence: R-C gain is " + fmt(pick(rewrite_con, "rewrite", "token_nonoverlap", "RminusC", "gain_delta")) + ", because REPEAT is worse than CLEAN on the true-source term by " + fmt(pick(rewrite_con, "rewrite", "token_nonoverlap", "RminusC", "true_source_delta")) + " while the unrelated-source term differs by only " + fmt(pick(rewrite_con, "rewrite", "token_nonoverlap", "RminusC", "unrelated_source_delta")) + ". This gives a third DeBERTa seed for the below-baseline recurrence cost.")
    lines.append("")
    lines.append("VIEW also remains above CLEAN on the same targets: V-C gain is " + fmt(pick(rewrite_con, "rewrite", "token_nonoverlap", "VminusC", "gain_delta")) + ", with true-source improvement larger than unrelated-source improvement. Thus the positive VIEW-side content-conditioning result is now three-seed DeBERTa evidence, while RoBERTa remains weak on V-C after register fit is subtracted.")
    lines.append("")
    lines.append("## Held-out natural copy gain")
    lines.append("")
    lines.append("| group | contrast | gain Δ | repeated-term Δ | unrepeated-term Δ | n |")
    lines.append("|---|---|---:|---:|---:|---:|")
    for r in copy_con:
        if r["group"] in ["ALL", "span_1", "span_4"] and r["contrast"] in ["RminusC", "VminusC", "RminusV"]:
            lines.append(f"| {r['group']} | {r['contrast']} | {fmt(r['gain_delta'])} | {fmt(r['repeated_nll_delta'])} | {fmt(r['unrepeated_nll_delta'])} | {r['n_min']} |")
    lines.append("")
    lines.append("The R-C copy advantage at seed43222 is smaller than prior seeds but positive on the aggregate: " + fmt(pick(copy_con, "copy", "ALL", "RminusC", "gain_delta")) + ". This supports the copy side directionally but reinforces the research caution that installed probe quantities are more stable than Entity magnitude, and copy conversion varies more than rewrite conditioning.")
    lines.append("")
    lines.append("## Entity cue-ablation with CLEAN baseline")
    lines.append("")
    lines.append("| group | contrast | full-margin Δ | no-initial effect Δ | no-last effect Δ | no-all-updates effect Δ | n |")
    lines.append("|---|---|---:|---:|---:|---:|---:|")
    for r in entity_con:
        if r["group"] in ["ALL", "rel_ge2", "rel_ge3"] and r["contrast"] in ["VminusC", "RminusC", "VminusR"]:
            lines.append(f"| {r['group']} | {r['contrast']} | {fmt(r['margin_full_delta'])} | {fmt(r['effect_no_initial_delta'])} | {fmt(r['effect_no_last_delta'])} | {fmt(r['effect_no_all_delta'])} | {r['n_min']} |")
    lines.append("")
    lines.append("The full gold-over-stale margins at seed43222 do not preserve a simple V>C>R ladder against CLEAN: V-R remains positive, but C is not a passive midpoint in this cue-ablation subset. This is another reason to keep Entity as a downstream correlate with direction-robust V-R structure, not the primary installed quantity.")
    lines.append("")
    lines.append("## Files")
    lines.append("")
    lines.append(f"- Output directory: `{rel(OUT)}`")
    lines.append(f"- VIEW/REPEAT source rows: `{rel(VR_DIR)}`")
    lines.append(f"- CLEAN source rows: `{rel(C_DIR)}`")
    _public_path('research/notes/relation_learning').mkdir(parents=True, exist_ok=True)
    NOTE.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rew_rows = read_rows("rewrite")
    cop_rows = read_rows("copy")
    ent_rows = read_rows("entity_ablation")
    rew_sum, rew_con = rewrite_summary(rew_rows)
    cop_sum, cop_con = copy_summary(cop_rows)
    ent_sum, ent_con = entity_summary(ent_rows)
    write_csv(_public_path('experiments/archive/relation_learning/data/seed43222_threearm_probe_decomposition/rewrite_arm_checkpoint_summary.csv'), rew_sum)
    write_csv(_public_path('experiments/archive/relation_learning/data/seed43222_threearm_probe_decomposition/rewrite_late_contrasts.csv'), rew_con)
    write_csv(_public_path('experiments/archive/relation_learning/data/seed43222_threearm_probe_decomposition/copy_arm_checkpoint_summary.csv'), cop_sum)
    write_csv(_public_path('experiments/archive/relation_learning/data/seed43222_threearm_probe_decomposition/copy_late_contrasts.csv'), cop_con)
    write_csv(_public_path('experiments/archive/relation_learning/data/seed43222_threearm_probe_decomposition/entity_ablation_arm_checkpoint_summary.csv'), ent_sum)
    write_csv(_public_path('experiments/archive/relation_learning/data/seed43222_threearm_probe_decomposition/entity_ablation_late_contrasts.csv'), ent_con)
    write_note(rew_con, cop_con, ent_con)
    result = {"status": "SEED43222_THREEARM_PROBE_DECOMPOSITION_DONE", "note": rel(NOTE), "outputs": rel(OUT)}
    (_public_path('experiments/archive/relation_learning/data/seed43222_threearm_probe_decomposition/seed43222_threearm_probe_decomposition_result.json')).write_text(json.dumps(result, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)

if __name__ == "__main__":
    main()
