#!/usr/bin/env python3
"""research: quantify score-vector tradeoffs across recent legal-coordinate interventions.

This script intentionally uses only existing official-compatible cheap-column summaries.
It does not run model inference.  It extracts deltas versus the matched research legal
reference for interventions that changed optimizer/objective/data/architecture, then
computes simple vector geometry: cheap7, positive/negative domain counts, correlations,
and a low-dimensional SVD of standardized deltas.  The scientific goal is to test
whether the repeated mature failures look like one common competence-redistribution
axis or several distinct failure modes.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from statistics import mean
from typing import Dict, Any, List

import numpy as np

ROOT = Path("experiments/archive/frontier_consolidation")
OUT_DIR = ROOT / "data/score_vector_tradeoff"
OUT_JSON = OUT_DIR / "score_vector_tradeoff.json"
OUT_MD = (OUT_DIR.parents[4] / 'research/documents/frontier_consolidation/data/score_vector_tradeoff/score_vector_tradeoff.md')

COLS = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA", "Reading"]
REF20 = {"BLiMP":59.69,"Supplement":55.45,"EWoK":50.73,"Entity":18.65,"COMPS":50.26,"GlobalPIQA":34.195,"Reading":8.67}
# research legal mature references: research per-target at 70/80, research endpoint at 100.
REF70 = {"BLiMP":65.39,"Supplement":59.31,"EWoK":50.47,"Entity":26.98,"COMPS":51.82,"GlobalPIQA":35.55,"Reading":8.74}
REF80 = {"BLiMP":66.11,"Supplement":60.66,"EWoK":51.01,"Entity":27.06,"COMPS":51.93,"GlobalPIQA":35.58,"Reading":8.29}
REF100 = {"BLiMP":65.8707181799453,"Supplement":61.16566092036889,"EWoK":50.39323748109589,"Entity":27.400833994026197,"COMPS":52.00834536316919,"GlobalPIQA":36.0631067961165,"Reading":8.13816768550987}


def load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text())


def cheap7(scores: Dict[str, float]) -> float:
    return float(sum(float(scores[c]) for c in COLS) / len(COLS))


def row(label: str, family: str, exposure: str, scores: Dict[str, float], ref: Dict[str, float], source: str, note: str="") -> Dict[str, Any]:
    deltas = {c: float(scores[c]) - float(ref[c]) for c in COLS}
    out = {
        "label": label,
        "family": family,
        "exposure": exposure,
        "scores": {c: float(scores[c]) for c in COLS},
        "reference": {c: float(ref[c]) for c in COLS},
        "deltas": deltas,
        "cheap7": cheap7(scores),
        "ref_cheap7": cheap7(ref),
        "cheap7_delta": cheap7(scores) - cheap7(ref),
        "positive_columns": sum(1 for v in deltas.values() if v > 0),
        "negative_columns": sum(1 for v in deltas.values() if v < 0),
        "large_gains_ge_1": {c:v for c,v in deltas.items() if v >= 1.0},
        "large_losses_le_m1": {c:v for c,v in deltas.items() if v <= -1.0},
        "source": source,
        "note": note,
    }
    # axis summaries with signs chosen from the recurring observations.
    syntax_entity = mean([deltas["BLiMP"], deltas["Supplement"], deltas["Entity"], deltas["COMPS"]])
    relation_reading = mean([deltas["EWoK"], deltas["GlobalPIQA"], deltas["Reading"]])
    entity_gp = mean([deltas["Entity"], deltas["GlobalPIQA"]])
    ewok_reading = mean([deltas["EWoK"], deltas["Reading"]])
    out["axis_scores"] = {
        "syntax_entity_minus_relation_reading": syntax_entity - relation_reading,
        "entity_gp_minus_ewok_reading": entity_gp - ewok_reading,
        "broad_positive_min_delta": min(deltas.values()),
        "damage_sum_negative": sum(min(0.0, v) for v in deltas.values()),
        "gain_sum_positive": sum(max(0.0, v) for v in deltas.values()),
    }
    return out

records: List[Dict[str, Any]] = []

# 20M early interventions.
rtd = load_json(ROOT/"data/mlm_rtd_20M_summary/mlm_rtd_20M_summary.json")
records.append(row("RTD/GDES lambda1", "objective", "20M", rtd["candidate_scores"], REF20, str(ROOT/"data/mlm_rtd_20M_summary/mlm_rtd_20M_summary.json"), "weak early movement; route closed after mechanism probe"))

mu20 = load_json(ROOT/"data/muon_wdmatched_20M_eval/muon_wdmatched_20M_comparison.json")
records.append(row("Muon wdmatched lr0.008", "optimizer", "20M", mu20["rows"]["muon_lr008_wd00125"]["scores"], REF20, str(ROOT/"data/muon_wdmatched_20M_eval/muon_wdmatched_20M_comparison.json"), "large early gain that reversed at 80M"))

lamb = load_json(ROOT/"data/lamb_20M_eval_merged/lamb_20M_eval_merged.json")
# Structure can vary; search rows recursively for score dictionaries.
def find_score_rows(obj: Any, prefix="") -> List[tuple[str, Dict[str, float]]]:
    found=[]
    if isinstance(obj, dict):
        if all(k in obj for k in COLS):
            try:
                found.append((prefix.strip("/"), {c: float(obj[c]) for c in COLS}))
            except Exception:
                pass
        for k,v in obj.items():
            found.extend(find_score_rows(v, prefix+"/"+str(k)))
    elif isinstance(obj, list):
        for i,v in enumerate(obj):
            found.extend(find_score_rows(v, prefix+f"/{i}"))
    return found
for name, scores in find_score_rows(lamb):
    lname=name.lower()
    if "baseline" in lname or "research" in lname:
        continue
    if "lr007" in lname or "0.007" in lname or "lr0p007" in lname:
        records.append(row("LAMB lr0.007", "optimizer", "20M", scores, REF20, str(ROOT/"data/lamb_20M_eval_merged/lamb_20M_eval_merged.json"), "trust-ratio clamp/rank collapse failure"))
        break
# If exact label not found, choose worst non-baseline score row whose cheap7 is in expected LAMB range.
if not any(r["label"] == "LAMB lr0.007" for r in records):
    candidates=[]
    for name,scores in find_score_rows(lamb):
        if "baseline" not in name.lower() and "research" not in name.lower():
            candidates.append((cheap7(scores), name, scores))
    if candidates:
        c = sorted(candidates)[0]
        records.append(row("LAMB lr0.007", "optimizer", "20M", c[2], REF20, str(ROOT/"data/lamb_20M_eval_merged/lamb_20M_eval_merged.json"), f"auto-selected LAMB score row {c[1]}"))

adapt104 = load_json(ROOT/"data/scaled_train_20M_eval_merged/scaled_train_20M_merged.json")
records.append(row("Adapter scale1.75 train", "architecture", "20M", adapt104["rows"]["train_scale1p75"]["scores"], REF20, str(ROOT/"data/scaled_train_20M_eval_merged/scaled_train_20M_merged.json"), "current residual-capacity route, early point"))
records.append(row("Adapter scale2.00 train", "architecture", "20M", adapt104["rows"]["train_scale2p00"]["scores"], REF20, str(ROOT/"data/scaled_train_20M_eval_merged/scaled_train_20M_merged.json"), "amplitude interpolation showed weaker signal"))

# 70/80/100 mature-ish interventions.
minf = load_json(ROOT/"data/minfreq50_decision/minfreq50_decision.json")
records.append(row("Minfreq50 support-floor", "tokenizer", "80M", minf["minfreq50_scores"]["80M"], REF80, str(ROOT/"data/minfreq50_decision/minfreq50_decision.json"), "legal tokenizer support-floor repair closed"))

word = load_json(ROOT/"data/wordmean_70_80M_eval/per_target/wordmean_mlm_seed43022_80M.json")
records.append(row("Word-mean MLM", "objective", "80M", word["official_overall"]["scores"], REF80, str(ROOT/"data/wordmean_70_80M_eval/per_target/wordmean_mlm_seed43022_80M.json"), "global word-mean group credit closed"))

strict = load_json(ROOT/"data/strict_innovation_70_80M_eval/per_target/strict_content_innovation_seed43022_80M.json")
records.append(row("Strict content-innovation WWM", "objective", "80M", strict["official_overall"]["scores"], REF80, str(ROOT/"data/strict_innovation_70_80M_eval/per_target/strict_content_innovation_seed43022_80M.json"), "targeted innovation masking closed"))

mu_m = load_json(ROOT/"data/muon_wdmatched_mature_eval/muon_wdmatched_mature_comparison.json")
records.append(row("Muon wdmatched lr0.008", "optimizer", "80M", mu_m["rows"]["80M"]["muon_scores"], REF80, str(ROOT/"data/muon_wdmatched_mature_eval/muon_wdmatched_mature_comparison.json"), "mature reversal despite early gain"))

sw = load_json(ROOT/"data/muon_switch_actual_merged/actual_switch_eval_merged.json")
records.append(row("Muon20->AdamW", "optimizer_switch", "80M", sw["rows"]["muon20_80M"]["scores"], REF80, str(ROOT/"data/muon_switch_actual_merged/actual_switch_eval_merged.json"), "early Muon then AdamW switch"))
records.append(row("Muon40->AdamW", "optimizer_switch", "80M", sw["rows"]["muon40_80M"]["scores"], REF80, str(ROOT/"data/muon_switch_actual_merged/actual_switch_eval_merged.json"), "longer Muon then AdamW switch"))

fwabs = load_json(ROOT/"data/fw_absolute_decision/fw_absolute_decision.json")
for comp in fwabs["absolute_comparisons"]:
    if comp["checkpoint"] == "100M" and comp["arm"] == "compact_view":
        scores = {c: fwabs["legal_reference"]["100M"][c] + comp["column_deltas_vs_legal_ref"][c] for c in COLS}
        records.append(row("Expanded FineWeb compact-view", "data", "100M", scores, REF100, str(ROOT/"data/fw_absolute_decision/fw_absolute_decision.json"), "absolute 100M FW-family endpoint, not enough for target"))
    if comp["checkpoint"] == "100M" and comp["arm"] == "source_breadth":
        scores = {c: fwabs["legal_reference"]["100M"][c] + comp["column_deltas_vs_legal_ref"][c] for c in COLS}
        records.append(row("Expanded FineWeb source-breadth", "data", "100M", scores, REF100, str(ROOT/"data/fw_absolute_decision/fw_absolute_decision.json"), "source-breadth comparator, Entity cost"))

traj = load_json(ROOT/"data/scale1p75_trajectory/scale1p75_trajectory.json")
records.append(row("Adapter scale1.75 train", "architecture", "50M", traj["rows"]["scale1p75"]["50M"]["scores"], traj["rows"]["research"]["50M"]["scores"], str(ROOT/"data/scale1p75_trajectory/scale1p75_trajectory.json"), "current residual-capacity route, 50M positive but mixed point"))

# Deduplicate accidental duplicates by label+exposure+family preserving first.
unique=[]; seen=set()
for r in records:
    key=(r["label"],r["family"],r["exposure"])
    if key not in seen:
        seen.add(key); unique.append(r)
records=unique

# Matrix analysis.
X = np.array([[r["deltas"][c] for c in COLS] for r in records], dtype=float)
# Correlation across columns over records; protect tiny std.
col_std = X.std(axis=0, ddof=1)
col_corr = np.full((len(COLS), len(COLS)), np.nan)
for i in range(len(COLS)):
    for j in range(len(COLS)):
        if col_std[i] > 1e-9 and col_std[j] > 1e-9:
            col_corr[i,j] = float(np.corrcoef(X[:,i], X[:,j])[0,1])
# Standardize rows? For PCA across interventions, center columns but do not scale scores, because units are same pct points.
Xc = X - X.mean(axis=0, keepdims=True)
U,S,Vt = np.linalg.svd(Xc, full_matrices=False)
var = (S*S) / max(1, (len(records)-1))
varfrac = var / var.sum() if var.sum() > 0 else np.zeros_like(var)
pcs=[]
for k in range(min(3, len(S))):
    vec = {COLS[i]: float(Vt[k,i]) for i in range(len(COLS))}
    # orient so GlobalPIQA loading is positive when possible for easier reading.
    if vec.get("GlobalPIQA",0.0) < 0:
        vec = {c:-v for c,v in vec.items()}
    pcs.append({"component": k+1, "singular_value": float(S[k]), "variance_fraction": float(varfrac[k]), "loadings": vec})

# Projections onto PC1/PC2 and predefined axes.
for r, urow in zip(records, U):
    r["pca_projection"] = {f"PC{k+1}": float(urow[k] * S[k]) for k in range(min(3, len(S)))}

# Group summaries.
by_exp: Dict[str, Any] = {}
for exp in sorted(set(r["exposure"] for r in records)):
    rs=[r for r in records if r["exposure"]==exp]
    by_exp[exp]={
        "n": len(rs),
        "mean_delta": {c: float(mean([r["deltas"][c] for r in rs])) for c in COLS},
        "mean_cheap7_delta": float(mean([r["cheap7_delta"] for r in rs])),
        "best_cheap7_delta": max(r["cheap7_delta"] for r in rs),
        "worst_cheap7_delta": min(r["cheap7_delta"] for r in rs),
    }

# Focus on mature >=70M/80M/100M excluding current 50M.
mature = [r for r in records if r["exposure"] in {"70M","80M","100M"}]
if mature:
    mX = np.array([[r["deltas"][c] for c in COLS] for r in mature], dtype=float)
    mature_summary = {
        "n": len(mature),
        "mean_delta": {c: float(mean([r["deltas"][c] for r in mature])) for c in COLS},
        "mean_cheap7_delta": float(mean([r["cheap7_delta"] for r in mature])),
        "count_with_globalpiqa_gain_ge_1": sum(1 for r in mature if r["deltas"]["GlobalPIQA"] >= 1),
        "count_with_entity_loss_le_m1": sum(1 for r in mature if r["deltas"]["Entity"] <= -1),
        "count_with_supp_or_ewok_loss_le_m1": sum(1 for r in mature if r["deltas"]["Supplement"] <= -1 or r["deltas"]["EWoK"] <= -1),
        "count_all_columns_positive": sum(1 for r in mature if all(r["deltas"][c] > 0 for c in COLS)),
        "best_mature_record": max(mature, key=lambda r: r["cheap7_delta"])["label"] + " " + max(mature, key=lambda r: r["cheap7_delta"])["exposure"],
        "best_mature_delta": max(r["cheap7_delta"] for r in mature),
    }
else:
    mature_summary = {}

interpretation=[]
if mature_summary:
    interpretation.append(f"Across {mature_summary['n']} mature legal-coordinate records, the mean cheap7 delta is {mature_summary['mean_cheap7_delta']:+.4f}; none has all seven columns positive ({mature_summary['count_all_columns_positive']}).")
    interpretation.append(f"GlobalPIQA gains >= +1 occur in {mature_summary['count_with_globalpiqa_gain_ge_1']} mature records, but Entity losses <= -1 occur in {mature_summary['count_with_entity_loss_le_m1']} and Supplement/EWoK losses <= -1 occur in {mature_summary['count_with_supp_or_ewok_loss_le_m1']}; the pattern is not broad expansion.")
    interpretation.append(f"Best mature cheap7 movement in this set is {mature_summary['best_mature_record']} at {mature_summary['best_mature_delta']:+.4f}, still far below the +0.697 cheap7 equivalent needed if SuperGLUE/AoA stay flat.")
if pcs:
    load = pcs[0]["loadings"]
    strongest_pos = sorted(load.items(), key=lambda kv: kv[1], reverse=True)[:3]
    strongest_neg = sorted(load.items(), key=lambda kv: kv[1])[:3]
    interpretation.append(f"PC1 explains {pcs[0]['variance_fraction']:.3f} of delta-vector variance; oriented positive with GlobalPIQA, strongest positive loadings {strongest_pos}, strongest negative loadings {strongest_neg}.")
# Compare current adapter 50M to mature tradeoff.
ad50 = next((r for r in records if r["label"]=="Adapter scale1.75 train" and r["exposure"]=="50M"), None)
if ad50:
    interpretation.append(f"Current scale1.75 at 50M has cheap7 delta {ad50['cheap7_delta']:+.4f}, with gains {ad50['large_gains_ge_1']} and losses {ad50['large_losses_le_m1']}; it is a mixed positive point, not an already broad mature repair.")

out={
    "status":"SCORE_VECTOR_TRADEOFF_ANALYSIS",
    "columns":COLS,
    "records":records,
    "by_exposure":by_exp,
    "column_correlation": {COLS[i]: {COLS[j]: (None if math.isnan(col_corr[i,j]) else float(col_corr[i,j])) for j in range(len(COLS))} for i in range(len(COLS))},
    "pca": pcs,
    "mature_summary": mature_summary,
    "interpretation": interpretation,
}
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_JSON.write_text(json.dumps(out, indent=2))

# Markdown report.
lines=[]
lines.append("# research score-vector tradeoff analysis")
lines.append("")
lines.append("CPU-only analysis of existing official-compatible cheap-column summaries. Deltas are against matched research legal references at the same exposure when available.")
lines.append("")
lines.append("## Records")
lines.append("")
lines.append("| label | family | exposure | Δcheap7 | ΔBLiMP | ΔSupp | ΔEWoK | ΔEntity | ΔCOMPS | ΔGP | ΔRead | strong gains | strong losses |")
lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|")
for r in records:
    d=r["deltas"]
    gains=", ".join(f"{k} {v:+.2f}" for k,v in r["large_gains_ge_1"].items()) or "-"
    losses=", ".join(f"{k} {v:+.2f}" for k,v in r["large_losses_le_m1"].items()) or "-"
    lines.append(f"| {r['label']} | {r['family']} | {r['exposure']} | {r['cheap7_delta']:+.4f} | {d['BLiMP']:+.2f} | {d['Supplement']:+.2f} | {d['EWoK']:+.2f} | {d['Entity']:+.2f} | {d['COMPS']:+.2f} | {d['GlobalPIQA']:+.2f} | {d['Reading']:+.2f} | {gains} | {losses} |")
lines.append("")
lines.append("## Mature summary")
lines.append("")
for k,v in mature_summary.items():
    lines.append(f"- {k}: {v}")
lines.append("")
lines.append("## Principal components")
lines.append("")
for pc in pcs:
    load=", ".join(f"{c}:{v:+.3f}" for c,v in pc["loadings"].items())
    lines.append(f"- PC{pc['component']} variance_fraction={pc['variance_fraction']:.3f}, loadings: {load}")
lines.append("")
lines.append("## Column correlation")
lines.append("")
lines.append("| col | " + " | ".join(COLS) + " |")
lines.append("|---" + "|---:"*len(COLS) + "|")
for c in COLS:
    vals=[]
    for c2 in COLS:
        val=out["column_correlation"][c][c2]
        vals.append("" if val is None else f"{val:+.2f}")
    lines.append(f"| {c} | " + " | ".join(vals) + " |")
lines.append("")
lines.append("## Interpretation")
lines.append("")
for x in interpretation:
    lines.append(f"- {x}")
lines.append("")
lines.append("## Sources")
for r in records:
    lines.append(f"- {r['label']} {r['exposure']}: `{r['source']}`")
OUT_MD.write_text("\n".join(lines)+"\n")
print(json.dumps({"status":out["status"],"out_json":str(OUT_JSON),"out_md":str(OUT_MD),"n_records":len(records),"mature_n":len(mature),"best_mature_delta":mature_summary.get("best_mature_delta")}, indent=2))
