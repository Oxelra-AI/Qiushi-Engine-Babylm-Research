#!/usr/bin/env python3
"""research: align legal-tokenizer deficit subtasks with corpus-only static-prior pressure.

This is a CPU-only analysis. It does NOT design or tune the prior from evaluation data.
The prior already exists (`relation_only_v1`, `relation_info_v1`) and was built only from
the legal 10M training pool. Here we ask a narrower scientific question before any GPU
intervention: do the subtasks that the legal research endpoint lost vs the old reference
actually receive high pressure from the existing relation_only prior?

If legal losses concentrate in subtasks with low prior pressure (e.g. QA-congruence or
property-style domains), relation-weighted masking would be poorly matched even though the
static-prior trainer is mechanically ready. If losses concentrate in high-prior dynamic /
relation subtasks, it remains a plausible single-variable screen once mature clean-vs-
reinvest deltas specifically implicate those contents.
"""
import json
import math
import pathlib
import statistics
from typing import Iterable

from transformers import AutoTokenizer

ROOT = pathlib.Path("experiments/archive/frontier_consolidation")
PRIOR_JSON = ROOT / "data/static_token_mask_prior/static_token_mask_prior.json"
DEFICIT_JSON = ROOT / "data/legal_deficit_subtask_decomposition/legal_deficit_subtask_decomposition.json"
TOKENIZER_DIR = ROOT / "data/compliant_tokenizer"
EVAL_ROOT = pathlib.Path("experiments/archive/representation_and_objectives/data/pristine_official_coordinate/babylm-eval/strict/evaluation_data/full_eval")

OUT_DIR = ROOT / "data/static_prior_eval_alignment"

SPECIAL_IDS = {0, 1, 2, 3, 4}
SCHEMES = ["relation_only_v1", "relation_info_v1"]


def iter_blimp_or_supp_texts(uid: str, subdir: str) -> Iterable[str]:
    p = EVAL_ROOT / subdir / f"{uid}.jsonl"
    with p.open() as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            yield d["sentence_good"]
            yield d["sentence_bad"]


def iter_ewok_texts(uid: str) -> Iterable[str]:
    p = EVAL_ROOT / "ewok_filtered" / f"{uid}.jsonl"
    with p.open() as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            yield " ".join([d["Context1"], d["Target1"]])
            yield " ".join([d["Context2"], d["Target2"]])


def safe_mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else None


def pearson(xs, ys):
    pairs = [(x, y) for x, y in zip(xs, ys) if x is not None and y is not None]
    if len(pairs) < 3:
        return None
    xs, ys = zip(*pairs)
    mx, my = statistics.mean(xs), statistics.mean(ys)
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx <= 0 or vy <= 0:
        return None
    return sum((x - mx) * (y - my) for x, y in pairs) / math.sqrt(vx * vy)


def ranks(vals):
    # Average ranks for ties not needed; deterministic dense-ish rank is enough for Spearman sign.
    order = sorted(range(len(vals)), key=lambda i: vals[i])
    r = [0.0] * len(vals)
    for rank, i in enumerate(order):
        r[i] = rank + 1
    return r


def spearman(xs, ys):
    pairs = [(x, y) for x, y in zip(xs, ys) if x is not None and y is not None]
    if len(pairs) < 3:
        return None
    rx = ranks([x for x, _ in pairs])
    ry = ranks([y for _, y in pairs])
    return pearson(rx, ry)


def tokenize_weights(tokenizer, weights, text: str):
    ids = tokenizer.encode(text, add_special_tokens=False)
    ids = [i for i in ids if i not in SPECIAL_IDS and i < len(weights)]
    ws = [float(weights[i]) for i in ids]
    return ids, ws


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    prior = json.loads(PRIOR_JSON.read_text())
    deficit = json.loads(DEFICIT_JSON.read_text())
    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER_DIR), use_fast=True)
    weights = {s: prior["schemes"][s]["weights"] for s in SCHEMES}

    per_uid = []
    for col, rows in deficit["columns"].items():
        for row in rows:
            uid = row["uid"]
            delta = row.get("delta_legal_step35")
            if delta is None:
                continue
            if col == "BLiMP":
                texts = list(iter_blimp_or_supp_texts(uid, "blimp_filtered"))
            elif col == "Supplement":
                texts = list(iter_blimp_or_supp_texts(uid, "supplement_filtered"))
            elif col == "EWoK":
                texts = list(iter_ewok_texts(uid))
            else:
                continue
            item = {
                "column": col,
                "uid": uid,
                "delta_legal_step35_minus_old_ref": delta,
                "n_eval_strings": len(texts),
            }
            for scheme in SCHEMES:
                all_ws = []
                all_ids = []
                for t in texts:
                    ids, ws = tokenize_weights(tokenizer, weights[scheme], t)
                    all_ids.extend(ids)
                    all_ws.extend(ws)
                if all_ws:
                    item[f"{scheme}_mean_weight"] = sum(all_ws) / len(all_ws)
                    item[f"{scheme}_mean_excess"] = item[f"{scheme}_mean_weight"] - 1.0
                    item[f"{scheme}_frac_gt_1p10"] = sum(w > 1.10 for w in all_ws) / len(all_ws)
                    item[f"{scheme}_frac_gt_1p25"] = sum(w > 1.25 for w in all_ws) / len(all_ws)
                    item[f"{scheme}_frac_gt_1p50"] = sum(w > 1.50 for w in all_ws) / len(all_ws)
                    item[f"{scheme}_tokens"] = len(all_ws)
                    item[f"{scheme}_distinct_ids"] = len(set(all_ids))
                else:
                    item[f"{scheme}_mean_weight"] = None
                    item[f"{scheme}_mean_excess"] = None
                    item[f"{scheme}_frac_gt_1p10"] = None
                    item[f"{scheme}_frac_gt_1p25"] = None
                    item[f"{scheme}_frac_gt_1p50"] = None
                    item[f"{scheme}_tokens"] = 0
                    item[f"{scheme}_distinct_ids"] = 0
            per_uid.append(item)

    # Correlate: positive corr(delta, pressure) means higher-prior tasks gained more/lost less.
    # Negative corr means higher-prior tasks lost more.
    summaries = {}
    for col in ["BLiMP", "Supplement", "EWoK", "ALL"]:
        rows = per_uid if col == "ALL" else [r for r in per_uid if r["column"] == col]
        summaries[col] = {"n_uid": len(rows)}
        for scheme in SCHEMES:
            pressure = [r[f"{scheme}_mean_weight"] for r in rows]
            excess = [r[f"{scheme}_mean_excess"] for r in rows]
            deltas = [r["delta_legal_step35_minus_old_ref"] for r in rows]
            summaries[col][scheme] = {
                "mean_eval_weight": round(safe_mean([p for p in pressure if p is not None]), 6) if pressure else None,
                "pearson_delta_vs_mean_weight": round(pearson(deltas, pressure), 6) if pearson(deltas, pressure) is not None else None,
                "spearman_delta_vs_mean_weight": round(spearman(deltas, pressure), 6) if spearman(deltas, pressure) is not None else None,
                "mean_delta_highest_quartile_pressure": None,
                "mean_delta_lowest_quartile_pressure": None,
            }
            valid = sorted([(r[f"{scheme}_mean_weight"], r["delta_legal_step35_minus_old_ref"], r["uid"])
                            for r in rows if r[f"{scheme}_mean_weight"] is not None])
            if len(valid) >= 4:
                q = max(1, len(valid)//4)
                lo = valid[:q]
                hi = valid[-q:]
                summaries[col][scheme]["mean_delta_lowest_quartile_pressure"] = round(safe_mean(d for _, d, _ in lo), 4)
                summaries[col][scheme]["mean_delta_highest_quartile_pressure"] = round(safe_mean(d for _, d, _ in hi), 4)
                summaries[col][scheme]["lowest_quartile_uids"] = [(u, round(w,4), d) for w,d,u in lo]
                summaries[col][scheme]["highest_quartile_uids"] = [(u, round(w,4), d) for w,d,u in hi]

    # Focus rows requested by recent science: QA congruence and EWoK dynamic/property losses.
    focus_uids = {
        "qa_congruence_easy", "qa_congruence_tricky", "turn_taking", "subject_aux_inversion", "hypernym",
        "physical-dynamics", "material-dynamics", "spatial-relations", "physical-relations",
        "social-properties", "material-properties", "quantitative-properties", "social-interactions",
    }
    focus = [r for r in per_uid if r["uid"] in focus_uids]

    result = {
        "status": "STATIC_PRIOR_EVAL_ALIGNMENT",
        "purpose": "Post-hoc diagnostic: whether already-built corpus-only static prior targets the legal-tokenizer deficit; not used to tune the prior.",
        "inputs": {
            "prior_json": str(PRIOR_JSON),
            "deficit_json": str(DEFICIT_JSON),
            "tokenizer_dir": str(TOKENIZER_DIR),
            "eval_root": str(EVAL_ROOT),
        },
        "summaries": summaries,
        "focus_rows": sorted(focus, key=lambda r: (r["column"], r["delta_legal_step35_minus_old_ref"])),
        "per_uid": per_uid,
    }
    out_json = OUT_DIR / "static_prior_eval_alignment.json"
    out_json.write_text(json.dumps(result, indent=2, ensure_ascii=False))

    md = []
    md.append("# research static-prior/evaluation-loss alignment\n")
    md.append("This is a post-hoc diagnostic of the already-built corpus-only priors. It does **not** tune the prior from evaluation data.\n")
    md.append("## Correlation of legal_step35 deficit with evaluation-token prior pressure\n")
    md.append("Positive Pearson/Spearman means higher-prior subtasks gained more or lost less vs the old non-submittable reference; negative means higher-prior subtasks lost more.\n")
    md.append("| column | n_uid | scheme | mean eval weight | Pearson(delta,weight) | Spearman | low-quartile Δ | high-quartile Δ |")
    md.append("|---|---:|---|---:|---:|---:|---:|---:|")
    for col in ["BLiMP", "Supplement", "EWoK", "ALL"]:
        for scheme in SCHEMES:
            s = summaries[col][scheme]
            def f(x):
                return "—" if x is None else (f"{x:.4f}" if isinstance(x, float) else str(x))
            md.append(f"| {col} | {summaries[col]['n_uid']} | {scheme} | {f(s['mean_eval_weight'])} | "
                      f"{f(s['pearson_delta_vs_mean_weight'])} | {f(s['spearman_delta_vs_mean_weight'])} | "
                      f"{f(s['mean_delta_lowest_quartile_pressure'])} | {f(s['mean_delta_highest_quartile_pressure'])} |")
    md.append("")
    md.append("## Focus subtasks\n")
    md.append("| column | uid | Δ legal_step35-old | relation_only mean weight | frac >1.25 | relation_info mean weight |")
    md.append("|---|---|---:|---:|---:|---:|")
    for r in sorted(focus, key=lambda r: (r["column"], r["delta_legal_step35_minus_old_ref"])):
        md.append(f"| {r['column']} | {r['uid']} | {r['delta_legal_step35_minus_old_ref']:.2f} | "
                  f"{r['relation_only_v1_mean_weight']:.4f} | {r['relation_only_v1_frac_gt_1p25']:.4f} | "
                  f"{r['relation_info_v1_mean_weight']:.4f} |")
    md.append("")
    md.append("## Interpretation for route selection\n")
    md.append("- The static-prior branch is mechanically valid, but this diagnostic asks whether it is scientifically matched to the legal deficit.")
    md.append("- If high `relation_only_v1` pressure correlates weakly or positively with Δ, the existing relation prior is not concentrated on the losing subtasks; choose it only if mature clean-vs-reinvest deltas independently show the amplified relation content is where reinvestment weakens.")
    md.append("- If high pressure correlates negatively and the mature deltas show the same relation/dynamic loss, then a single-variable `relation_only_v1` screen is better justified.")
    md.append(f"\nFull JSON: `{out_json}`")
    out_md = (OUT_DIR.parents[4] / 'research/documents/frontier_consolidation/data/static_prior_eval_alignment/static_prior_eval_alignment.md')
    out_md.write_text("\n".join(md))

    print(json.dumps({
        "status": result["status"],
        "out_json": str(out_json),
        "out_md": str(out_md),
        "summaries": summaries,
        "focus_count": len(focus),
    }, indent=2))


if __name__ == "__main__":
    main()
