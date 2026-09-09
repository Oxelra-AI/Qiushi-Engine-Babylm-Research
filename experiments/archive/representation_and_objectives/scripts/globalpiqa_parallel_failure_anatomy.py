#!/usr/bin/env python3
"""research: analyze GlobalPIQA parallel/nonparallel failure anatomy across endpoints.

This is CPU-only and does not rerun evaluation. It reads official eval data and existing
prediction.json files to expose whether the large GlobalPIQA deficit is concentrated in
answer-position bias, category-specific physical/temporal/spatial items, or stable cross-model
wrong rows. The scientific purpose is to convert the striking low GlobalPIQA_parallel score
(~22-23 for legal endpoints) into a concrete mechanism hypothesis for future data/model work.
"""
from __future__ import annotations

import csv
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

ROOT = Path("experiments/archive/representation_and_objectives")
OUT = ROOT / "data" / "globalpiqa_parallel_anatomy"
NOTE = (ROOT / 'notes'.parents[3] / 'research/notes/representation_and_objectives/globalpiqa_parallel_failure_anatomy.md')
DATA_ROOT = ROOT / "data" / "globalpiqa_official_lineage" / "official_dl_scratch" / "generated_by_current_official_dl" / "evaluation_data" / "full_eval"
PAR_DATA = DATA_ROOT / "global_piqa_parallel" / "eng_latn.jsonl"
NONPAR_DATA = DATA_ROOT / "global_piqa_nonparallel" / "eng_latn.jsonl"

# Manually curate the endpoints that matter for the current research decision.  These are
# official-compatible prediction artifacts already produced in prior steps; no scorer is rerun here.
ENDPOINTS = {
    "legal40k8x480_43022": {
        "label": "legal40k 8x480 seed43022 (best compliant endpoint)",
        "parallel": ROOT / "data/legal40k_accum_seed43022_full_eval/official_outputs/legal40k_reinvest_seed43022/GlobalPIQA_parallel/chck_100M/full_legal40k_reinvest_seed43022_GlobalPIQA_parallel/zero_shot/mlm/global_piqa_parallel/global_piqa_parallel/predictions.json",
        "nonparallel": ROOT / "data/legal40k_accum_seed43022_full_eval/official_outputs/legal40k_reinvest_seed43022/GlobalPIQA_nonparallel/chck_100M/full_legal40k_reinvest_seed43022_GlobalPIQA_nonparallel/zero_shot/mlm/global_piqa_nonparallel/global_piqa_nonparallel/predictions.json",
        "overall": 41.140577774478444,
        "gpiqa": 34.66504854368932,
    },
    "legal40k8x480_43122": {
        "label": "legal40k 8x480 seed43122",
        "parallel": ROOT / "data/legal40k_accum_seed43122_full_eval/official_outputs/legal40k_reinvest_seed43122/GlobalPIQA_parallel/chck_100M/full_legal40k_reinvest_seed43122_GlobalPIQA_parallel/zero_shot/mlm/global_piqa_parallel/global_piqa_parallel/predictions.json",
        "nonparallel": ROOT / "data/legal40k_accum_seed43122_full_eval/official_outputs/legal40k_reinvest_seed43122/GlobalPIQA_nonparallel/chck_100M/full_legal40k_reinvest_seed43122_GlobalPIQA_nonparallel/zero_shot/mlm/global_piqa_nonparallel/global_piqa_nonparallel/predictions.json",
        "overall": 40.42013171935382,
        "gpiqa": 32.16504854368932,
    },
    "legal16k8x480_43022": {
        "label": "legal16k 8x480 seed43022",
        "parallel": ROOT / "data/strictsmalltok_seed43022_full_eval/official_outputs/strictsmalltok_reinvest_seed43022/GlobalPIQA_parallel/chck_100M/full_strictsmalltok_reinvest_seed43022_GlobalPIQA_parallel/zero_shot/mlm/global_piqa_parallel/global_piqa_parallel/predictions.json",
        "nonparallel": ROOT / "data/strictsmalltok_seed43022_full_eval/official_outputs/strictsmalltok_reinvest_seed43022/GlobalPIQA_nonparallel/chck_100M/full_strictsmalltok_reinvest_seed43022_GlobalPIQA_nonparallel/zero_shot/mlm/global_piqa_nonparallel/global_piqa_nonparallel/predictions.json",
        "overall": 40.703956400082454,
        "gpiqa": 36.10679611650485,
    },
    "legal16k8x480_43122": {
        "label": "legal16k 8x480 seed43122",
        "parallel": ROOT / "data/strictsmalltok_seed43122_full_eval/official_outputs/strictsmalltok_reinvest_seed43122/GlobalPIQA_parallel/chck_100M/full_strictsmalltok_reinvest_seed43122_GlobalPIQA_parallel/zero_shot/mlm/global_piqa_parallel/global_piqa_parallel/predictions.json",
        "nonparallel": ROOT / "data/strictsmalltok_seed43122_full_eval/official_outputs/strictsmalltok_reinvest_seed43122/GlobalPIQA_nonparallel/chck_100M/full_strictsmalltok_reinvest_seed43122_GlobalPIQA_nonparallel/zero_shot/mlm/global_piqa_nonparallel/global_piqa_nonparallel/predictions.json",
        "overall": 41.023994024744404,
        "gpiqa": 38.605000000000004,
    },
    "depth12x384_43022": {
        "label": "legal40k 12x384 depth seed43022",
        "parallel": ROOT / "data/legal40k_12x384_depth_seed43022_full_eval/official_outputs/legal40k_12x384_depth_seed43022/GlobalPIQA_parallel/chck_100M/full_legal40k_12x384_depth_seed43022_GlobalPIQA_parallel/zero_shot/mlm/global_piqa_parallel/global_piqa_parallel/predictions.json",
        "nonparallel": ROOT / "data/legal40k_12x384_depth_seed43022_full_eval/official_outputs/legal40k_12x384_depth_seed43022/GlobalPIQA_nonparallel/chck_100M/full_legal40k_12x384_depth_seed43022_GlobalPIQA_nonparallel/zero_shot/mlm/global_piqa_nonparallel/global_piqa_nonparallel/predictions.json",
        "overall": 41.02759583135309,
        "gpiqa": 35.63592233009709,
    },
    "sgcr12x384_43022": {
        "label": "exact-prefix SGCR 12x384 seed43022",
        "parallel": ROOT / "data/legal40k_12x384_sgcrK50d64_seed43022_full_eval/official_outputs/legal40k_12x384_sgcrK50d64_seed43022/GlobalPIQA_parallel/chck_100M/full_legal40k_12x384_sgcrK50d64_seed43022_GlobalPIQA_parallel/zero_shot/mlm/global_piqa_parallel/global_piqa_parallel/predictions.json",
        "nonparallel": ROOT / "data/legal40k_12x384_sgcrK50d64_seed43022_full_eval/official_outputs/legal40k_12x384_sgcrK50d64_seed43022/GlobalPIQA_nonparallel/chck_100M/full_legal40k_12x384_sgcrK50d64_seed43022_GlobalPIQA_nonparallel/zero_shot/mlm/global_piqa_nonparallel/global_piqa_nonparallel/predictions.json",
        "overall": 40.331709413463635,
        "gpiqa": 34.150485436893206,
    },
    "inherited16k_noncompliant_43022": {
        "label": "inherited-tokenizer compact reinvest seed43022 (non-submission evidence)",
        "parallel": ROOT / "data/compact_reinvest_full_eval/official_outputs/compact_view_reinvest/GlobalPIQA_parallel/chck_100M/full_compact_view_reinvest_GlobalPIQA_parallel/zero_shot/mlm/global_piqa_parallel/global_piqa_parallel/predictions.json",
        "nonparallel": ROOT / "data/compact_reinvest_full_eval/official_outputs/compact_view_reinvest/GlobalPIQA_nonparallel/chck_100M/full_compact_view_reinvest_GlobalPIQA_nonparallel/zero_shot/mlm/global_piqa_nonparallel/global_piqa_nonparallel/predictions.json",
        "overall": 42.0331347900748,
        "gpiqa": 35.62135922330097,
    },
    "inherited16k_noncompliant_43122": {
        "label": "inherited-tokenizer compact reinvest seed43122 (non-submission evidence)",
        "parallel": ROOT / "data/seed43122_full_eval/official_outputs/compact_view_reinvest_seed43122/GlobalPIQA_parallel/chck_100M/full_compact_view_reinvest_seed43122_GlobalPIQA_parallel/zero_shot/mlm/global_piqa_parallel/global_piqa_parallel/predictions.json",
        "nonparallel": ROOT / "data/seed43122_full_eval/official_outputs/compact_view_reinvest_seed43122/GlobalPIQA_nonparallel/chck_100M/full_compact_view_reinvest_seed43122_GlobalPIQA_nonparallel/zero_shot/mlm/global_piqa_nonparallel/global_piqa_nonparallel/predictions.json",
        "overall": 41.24823958912208,
        "gpiqa": None,
    },
    "semantic_view_treatment_100M": {
        "label": "SimpleWiki semantic-view treatment chck_100M",
        "parallel": ROOT / "data/semantic_view_noaoa_eval/official_outputs/semantic_view_treatment/chck_100M/GlobalPIQA_parallel/chck_100M/semantic_view_treatment_chck_100M_GlobalPIQA_parallel/zero_shot/mlm/global_piqa_parallel/global_piqa_parallel/predictions.json",
        "nonparallel": ROOT / "data/semantic_view_noaoa_eval/official_outputs/semantic_view_treatment/chck_100M/GlobalPIQA_nonparallel/chck_100M/semantic_view_treatment_chck_100M_GlobalPIQA_nonparallel/zero_shot/mlm/global_piqa_nonparallel/global_piqa_nonparallel/predictions.json",
        "overall": 40.1399,
        "gpiqa": None,
    },
    "semantic_view_control_100M": {
        "label": "SimpleWiki packet-local control chck_100M",
        "parallel": ROOT / "data/semantic_view_noaoa_eval/official_outputs/original_packet_local/chck_100M/GlobalPIQA_parallel/chck_100M/original_packet_local_chck_100M_GlobalPIQA_parallel/zero_shot/mlm/global_piqa_parallel/global_piqa_parallel/predictions.json",
        "nonparallel": ROOT / "data/semantic_view_noaoa_eval/official_outputs/original_packet_local/chck_100M/GlobalPIQA_nonparallel/chck_100M/original_packet_local_chck_100M_GlobalPIQA_nonparallel/zero_shot/mlm/global_piqa_nonparallel/global_piqa_nonparallel/predictions.json",
        "overall": 38.5540,
        "gpiqa": None,
    },
}

CATEGORY_KEYWORDS = {
    "temporal_arithmetic_or_order": [r"\bmonth\b", r"\bday\b", r"\byear\b", r"\bAM\b", r"\bPM\b", r"morning", r"afternoon", r"\bdate\b", r"every \d+"],
    "direction_spatial": [r"\bnorth\b", r"\bsouth\b", r"\beast\b", r"\bwest\b", r"left", r"right", r"direction", r"facing", r"turn"],
    "physical_object_interaction": [r"ball", r"window", r"water", r"air", r"bag", r"fabric", r"door", r"glass", r"object", r"push", r"pull", r"break", r"falls", r"roll", r"float", r"sink"],
    "tool_affordance": [r"best to use", r"use .* to", r"utensil", r"tool", r"write", r"cut", r"clean", r"serve"],
}


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())


def load_items(path: Path) -> List[Dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def pred_choice(item: Dict[str, Any], pred_val: Any) -> Tuple[Optional[int], str, Optional[str]]:
    # Official predictions contain {"predictions": [{"pred": " text"}]}; accept variants.
    pred_text = None
    if isinstance(pred_val, dict):
        preds = pred_val.get("predictions")
        if isinstance(preds, list) and preds:
            first = preds[0]
            if isinstance(first, dict):
                pred_text = first.get("pred")
            elif isinstance(first, str):
                pred_text = first
        if pred_text is None:
            pred_text = pred_val.get("pred") or pred_val.get("prediction")
    elif isinstance(pred_val, str):
        pred_text = pred_val
    if pred_text is None:
        return None, "", "missing_pred_text"
    pnorm = norm(str(pred_text))
    opts=[]
    for j in range(10):
        key=f"solution{j}"
        if key in item:
            opts.append((j, norm(str(item[key]))))
    exact=[j for j, o in opts if o == pnorm]
    if len(exact)==1:
        return exact[0], str(pred_text).strip(), None
    contains=[j for j,o in opts if pnorm and (pnorm in o or o in pnorm)]
    if len(contains)==1:
        return contains[0], str(pred_text).strip(), "substring_match"
    return None, str(pred_text).strip(), f"unmatched_pred choices={contains}"


def classify_item(item: Dict[str, Any]) -> List[str]:
    text = " ".join(str(item.get(k,"")) for k in ["prompt","solution0","solution1","solution2","solution3","categories","supplement"])
    cats = []
    for name, pats in CATEGORY_KEYWORDS.items():
        if any(re.search(pat, text, re.I) for pat in pats):
            cats.append(name)
    if item.get("categories"):
        cats.append("officialcat:" + str(item["categories"]))
    if not cats:
        cats.append("other")
    return cats


def score_endpoint(items: List[Dict[str, Any]], pred_path: Path, mode: str) -> Dict[str, Any]:
    if not pred_path.exists():
        return {"exists": False, "path": str(pred_path)}
    preds = json.loads(pred_path.read_text())
    rows=[]
    label_counts=Counter()
    choice_counts=Counter()
    pair_counts=Counter()
    category_total=Counter()
    category_ok=Counter()
    errors=[]
    for item in items:
        eid=item["example_id"]
        lab=int(item["label"])
        label_counts[lab]+=1
        pv=preds.get(eid)
        ch, ptxt, err=pred_choice(item,pv)
        if err and err != "substring_match":
            errors.append({"example_id": eid, "error": err, "pred_text": ptxt})
        ok=(ch==lab)
        if ch is not None:
            choice_counts[ch]+=1
            pair_counts[(lab,ch)]+=1
        for cat in classify_item(item):
            category_total[cat]+=1
            if ok:
                category_ok[cat]+=1
        rows.append({
            "mode": mode,
            "example_id": eid,
            "prompt": item.get("prompt"),
            "label": lab,
            "choice": ch,
            "ok": bool(ok),
            "prediction_text": ptxt,
            "categories": classify_item(item),
            **{f"solution{j}": item.get(f"solution{j}") for j in range(4) if f"solution{j}" in item},
        })
    n=len(rows)
    acc=100*sum(r["ok"] for r in rows)/n if n else math.nan
    inv_acc=None
    if mode=="parallel":
        # Four-way inversion: if the model consistently selects the most generic wrong option,
        # how good would an anti-score be?  This is NOT a usable official score, only a bias signal.
        inv_ok=0
        inv_n=0
        for r in rows:
            if r["choice"] is not None:
                inv_n+=1
                # count as anti-correct if chosen option is not label in a 4-choice item
                inv_ok += int(r["choice"] != r["label"])
        inv_acc = 100*inv_ok/inv_n if inv_n else None
    cat_table=[]
    for cat,ncat in category_total.most_common():
        cat_table.append({"category": cat, "n": ncat, "accuracy": 100*category_ok[cat]/ncat})
    return {
        "exists": True,
        "path": str(pred_path),
        "n": n,
        "accuracy": acc,
        "anti_choice_rate_if_fourway": inv_acc,
        "label_counts": {str(k): v for k,v in sorted(label_counts.items())},
        "choice_counts": {str(k): v for k,v in sorted(choice_counts.items())},
        "label_choice_matrix": {f"{a}->{b}": v for (a,b),v in sorted(pair_counts.items())},
        "category_accuracy": cat_table,
        "unmatched_prediction_errors": errors[:20],
        "rows": rows,
    }


def agreement_analysis(endpoint_results: Dict[str, Any], mode: str) -> Dict[str, Any]:
    # Cross-model stable wrong/right rows for endpoints with available rows.
    avail={k:v[mode] for k,v in endpoint_results.items() if v.get(mode,{}).get("exists")}
    if not avail:
        return {}
    ids=[r["example_id"] for r in next(iter(avail.values()))["rows"]]
    per_id=[]
    for eid in ids:
        rec={"example_id": eid}
        oks=[]; choices=[]
        for k,res in avail.items():
            rr=next(r for r in res["rows"] if r["example_id"]==eid)
            rec[k+"_ok"]=rr["ok"]
            rec[k+"_choice"]=rr["choice"]
            oks.append(rr["ok"])
            choices.append(rr["choice"])
            if "prompt" not in rec:
                rec.update({kk: rr.get(kk) for kk in ["prompt","label","solution0","solution1","solution2","solution3","categories"] if kk in rr})
        rec["n_ok"] = sum(oks)
        rec["n_wrong"] = len(oks)-sum(oks)
        rec["choice_consensus"] = Counter(choices).most_common(1)[0][0] if choices else None
        rec["choice_consensus_count"] = Counter(choices).most_common(1)[0][1] if choices else 0
        per_id.append(rec)
    return {
        "n_endpoints": len(avail),
        "endpoint_keys": list(avail.keys()),
        "n_items": len(ids),
        "all_available_wrong_count": sum(1 for r in per_id if r["n_ok"]==0),
        "all_available_right_count": sum(1 for r in per_id if r["n_wrong"]==0),
        "wrong_by_at_least_half_count": sum(1 for r in per_id if r["n_wrong"]>=math.ceil(len(avail)/2)),
        "rows": per_id,
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    par_items=load_items(PAR_DATA)
    non_items=load_items(NONPAR_DATA)
    endpoint_results={}
    summary_rows=[]
    for key,meta in ENDPOINTS.items():
        endpoint_results[key]={"meta": {k:str(v) for k,v in meta.items() if k not in ("parallel","nonparallel")}}
        for mode,items in [("parallel",par_items),("nonparallel",non_items)]:
            res=score_endpoint(items, Path(meta[mode]), mode)
            endpoint_results[key][mode]=res
        par_acc=endpoint_results[key]["parallel"].get("accuracy")
        non_acc=endpoint_results[key]["nonparallel"].get("accuracy")
        agg=(par_acc+non_acc)/2 if par_acc is not None and non_acc is not None else None
        summary_rows.append({
            "endpoint": key,
            "label": meta["label"],
            "parallel_acc": par_acc,
            "nonparallel_acc": non_acc,
            "computed_globalpiqa": agg,
            "known_globalpiqa": meta.get("gpiqa"),
        "aggregate_known_minus_computed": (float(meta["gpiqa"]) - agg) if (meta.get("gpiqa") is not None and agg is not None) else None,
            "overall": meta.get("overall"),
            "parallel_choice_counts": json.dumps(endpoint_results[key]["parallel"].get("choice_counts")),
            "parallel_label_counts": json.dumps(endpoint_results[key]["parallel"].get("label_counts")),
        })
    # Write detailed JSON (drop full rows from endpoint summary? keep them; file remains modest)
    agreement={
        "parallel": agreement_analysis(endpoint_results,"parallel"),
        "nonparallel": agreement_analysis(endpoint_results,"nonparallel"),
    }
    payload={
        "status": "GLOBALPIQA_PARALLEL_ANATOMY_DONE",
        "data_paths": {"parallel": str(PAR_DATA), "nonparallel": str(NONPAR_DATA)},
        "n_items": {"parallel": len(par_items), "nonparallel": len(non_items)},
        "summary": summary_rows,
        "endpoint_results": endpoint_results,
        "agreement": agreement,
        "interpretation": {
            "load_bearing_observation": "GlobalPIQA_parallel is a 103-item four-choice English physical/temporal/spatial set; several strong endpoints score only about 22-29%, while nonparallel two-choice PIQA is about 41-49%.",
            "not_a_simple_tokenizer_issue": "Both legal and inherited-tokenizer endpoints share low parallel accuracy; inherited 42.033 has only 26.21% parallel. This column is a data/knowledge/reasoning failure mode, not merely compliant-tokenizer damage.",
            "position_bias_hypothesis": "If chosen-option counts are strongly concentrated away from the label distribution, the model may be selecting generic plausible continuations rather than evaluating physical/temporal consequences; answer-position calibration or contrastive consequence exposure may matter.",
            "sota_relevance": "Best compliant legal40k is ahead of the visible leader on BLiMP, Supplement, and Reading but loses about 5 column-points on GlobalPIQA; raising GlobalPIQA_parallel is one of the few single levers large enough to close much of the 41.80 gap.",
        },
    }
    (OUT/"globalpiqa_parallel_anatomy.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    with (OUT/"globalpiqa_endpoint_summary.csv").open("w", newline="") as f:
        writer=csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader(); writer.writerows(summary_rows)
    # Stable wrong rows for compact note / later route.
    par_ag=agreement["parallel"]
    stable_rows=sorted(par_ag.get("rows", []), key=lambda r: (r["n_ok"], -r["choice_consensus_count"], r["example_id"]))
    with (OUT/"globalpiqa_parallel_cross_endpoint_rows.csv").open("w", newline="") as f:
        fields=["example_id","label","n_ok","n_wrong","choice_consensus","choice_consensus_count","prompt","categories","solution0","solution1","solution2","solution3"]
        writer=csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader(); writer.writerows(stable_rows)
    # Write a compact research note.
    best = next(r for r in summary_rows if r["endpoint"]=="legal40k8x480_43022")
    lines=[]
    lines.append("# research — GlobalPIQA parallel failure anatomy")
    lines.append("")
    lines.append("## Why this matters")
    lines.append("")
    lines.append("The best compliant endpoint (`legal40k8x480_43022`) is already ahead of the visible leader on BLiMP, Supplement, and Reading, but remains ~0.659 Overall below 41.80. Since Overall is the mean of 9 columns, this is only ~5.93 summed column-points. GlobalPIQA alone is ~5.00 column-points behind the leader, so understanding its failure is directly SOTA-relevant.")
    lines.append("")
    lines.append("## Core measurement")
    lines.append("")
    lines.append(f"The official full-eval GlobalPIQA data used here has {len(par_items)} `parallel` English four-choice items and {len(non_items)} `nonparallel` English two-choice items. Existing prediction artifacts were read directly; no scorer or model evaluation was rerun.")
    lines.append("")
    lines.append("| endpoint | parallel | nonparallel | aggregate | notes |")
    lines.append("|---|---:|---:|---:|---|")
    for r in summary_rows:
        pa=r['parallel_acc']; na=r['nonparallel_acc']; ag=r['computed_globalpiqa']
        lines.append(f"| `{r['endpoint']}` | {pa:.2f} | {na:.2f} | {ag:.2f} | {r['label']} |")
    lines.append("")
    lines.append("## What the pattern says")
    lines.append("")
    lines.append("1. The low score is concentrated in `global_piqa_parallel`, not in the nonparallel PIQA-like set. The best compliant model gets only %.2f on parallel but %.2f on nonparallel." % (best['parallel_acc'], best['nonparallel_acc']))
    lines.append("2. This is not only a compliant-tokenizer artifact: the inherited-tokenizer 42.033 model also remains weak on parallel, so GlobalPIQA is a real knowledge/reasoning/data-substrate weakness of the lineage.")
    lines.append("3. The parallel set is small but high-leverage: 103 four-choice English items cover physical consequences, time arithmetic/order, spatial directions, object interactions, and tool affordances. A 10–20 point movement in the parallel subset changes the aggregate GlobalPIQA by 5–10 points and Overall by 0.56–1.11.")
    lines.append("4. Since official predictions contain only the winning option, this analysis cannot inspect probability margins. The next stronger cheap check, if needed, is a pseudo-log-likelihood/margin reader over these 103 rows for completed candidate checkpoints, not another 100M training run.")
    lines.append("")
    lines.append("## Files")
    lines.append("")
    lines.append(f"- JSON: `{OUT/'globalpiqa_parallel_anatomy.json'}`")
    lines.append(f"- endpoint summary CSV: `{OUT/'globalpiqa_endpoint_summary.csv'}`")
    lines.append(f"- cross-endpoint row table: `{OUT/'globalpiqa_parallel_cross_endpoint_rows.csv'}`")
    NOTE.write_text("\n".join(lines)+"\n")
    print(json.dumps({
        "status": payload["status"],
        "out_json": str(OUT/"globalpiqa_parallel_anatomy.json"),
        "summary_csv": str(OUT/"globalpiqa_endpoint_summary.csv"),
        "row_csv": str(OUT/"globalpiqa_parallel_cross_endpoint_rows.csv"),
        "note": str(NOTE),
        "best_compliant_parallel": best["parallel_acc"],
        "best_compliant_nonparallel": best["nonparallel_acc"],
    }, indent=2))

if __name__ == "__main__":
    main()
