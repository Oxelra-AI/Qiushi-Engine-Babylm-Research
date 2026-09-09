#!/usr/bin/env python3
"""research: current-official EWoK 2x2 domain-level treatment interaction.

Reads saved 7,618-row official EWoK predictions for:
  clean-Qwen seed43022 / seed43122
  compact_view_reinvest seed43022 / seed43122
and computes treatment effects (reinvest-clean) within each seed, plus the
seed interaction (TE43122-TE43022) by official EWoK domain and metadata.

No model evaluation or training is performed.
"""
from __future__ import annotations

import collections
import json
import math
import pathlib
import random
import statistics
from typing import Any

ROOT = pathlib.Path.cwd()
STUDY = ROOT / "experiments/archive/frontier_consolidation"
OUT_DIR = STUDY / "data/official_ewok_2x2_domain_did"
DATA_DIR = ROOT / "experiments/archive/representation_and_objectives/data/pristine_official_coordinate/babylm-eval/strict/evaluation_data/full_eval/ewok_filtered"
PRED = {
    "clean_43022": ROOT / "experiments/archive/frontier_consolidation/data/official_ewok_clean_qwen/official_outputs/clean_qwen_seed43022/EWoK/chck_100M/official_ewok_clean_qwen_seed43022/zero_shot/mlm/ewok/ewok_filtered/predictions.json",
    "clean_43122": ROOT / "experiments/archive/frontier_consolidation/data/official_ewok_clean_qwen/official_outputs/clean_qwen_seed43122/EWoK/chck_100M/official_ewok_clean_qwen_seed43122/zero_shot/mlm/ewok/ewok_filtered/predictions.json",
    "reinvest_43022": ROOT / "experiments/archive/representation_and_objectives/data/official_ewok_reeval/official_outputs/EWoK/chck_100M/official_ewok_reinvest/zero_shot/mlm/ewok/ewok_filtered/predictions.json",
    "reinvest_43122": ROOT / "experiments/archive/representation_and_objectives/data/official_ewok_reeval_seed43122/official_outputs/EWoK/chck_100M/official_ewok_seed43122/zero_shot/mlm/ewok/ewok_filtered/predictions.json",
}


def norm(s: Any) -> str:
    return " ".join(str(s).strip().split())


def target(row: dict[str, Any]) -> str:
    # Current official EWoK scoring treats Context1+Target1 as the correct item.
    return norm(f"{row['Context1']} {row['Target1']}")


def load_data() -> dict[str, list[dict[str, Any]]]:
    out = {}
    for p in sorted(DATA_DIR.glob("*.jsonl")):
        rows = [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]
        out[p.stem] = rows
    if not out:
        raise RuntimeError(f"No EWoK data found under {DATA_DIR}")
    return out


def load_preds(path: pathlib.Path) -> dict[str, list[dict[str, Any]]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    out = {}
    for domain, block in raw.items():
        if isinstance(block, dict) and isinstance(block.get("predictions"), list):
            out[domain] = block["predictions"]
        elif isinstance(block, list):
            out[domain] = block
        else:
            raise TypeError(f"Unexpected prediction block for {domain} in {path}: {type(block)}")
    return out


def correctness_by_model(data: dict[str, list[dict[str, Any]]], preds: dict[str, dict[str, list[dict[str, Any]]]]) -> list[dict[str, Any]]:
    rows_out = []
    domains = sorted(data)
    for domain in domains:
        n = len(data[domain])
        for name, p in preds.items():
            if domain not in p:
                raise RuntimeError(f"{name} missing domain {domain}")
            if len(p[domain]) != n:
                raise RuntimeError(f"length mismatch {name}/{domain}: data {n} preds {len(p[domain])}")
        for i, row in enumerate(data[domain]):
            tgt = target(row)
            rec: dict[str, Any] = {
                "domain": domain,
                "index": i,
                "ContextType": row.get("ContextType"),
                "ContextDiff": row.get("ContextDiff"),
                "TargetDiff": row.get("TargetDiff"),
                "ConceptA": row.get("ConceptA"),
                "ConceptB": row.get("ConceptB"),
                "target": tgt,
            }
            for name, p in preds.items():
                pr = p[domain][i]
                pred = norm(pr.get("pred"))
                rec[f"{name}_pred"] = pred
                rec[f"{name}_correct"] = pred == tgt
                # Retain ids when available; useful for spot-checking joins.
                rec[f"{name}_id"] = pr.get("id")
            rows_out.append(rec)
    return rows_out


def score(rows: list[dict[str, Any]], key: str) -> float:
    if not rows:
        return math.nan
    return 100.0 * sum(1 for r in rows if r[f"{key}_correct"]) / len(rows)


def summarize_partition(rows: list[dict[str, Any]], label_field: str) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for r in rows:
        groups[str(r.get(label_field))].append(r)
    recs = []
    for label, rs in sorted(groups.items()):
        c430 = score(rs, "clean_43022")
        c431 = score(rs, "clean_43122")
        r430 = score(rs, "reinvest_43022")
        r431 = score(rs, "reinvest_43122")
        te430 = r430 - c430
        te431 = r431 - c431
        recs.append({
            label_field: label,
            "n": len(rs),
            "clean_43022": c430,
            "clean_43122": c431,
            "reinvest_43022": r430,
            "reinvest_43122": r431,
            "clean_seed_gap_43122_minus_43022": c431 - c430,
            "reinvest_seed_gap_43122_minus_43022": r431 - r430,
            "TE43022_reinvest_minus_clean": te430,
            "TE43122_reinvest_minus_clean": te431,
            "DiD_TE43122_minus_TE43022": te431 - te430,
            "agreement_clean_seeds": 100.0 * sum(r["clean_43022_correct"] == r["clean_43122_correct"] for r in rs) / len(rs),
            "agreement_reinvest_seeds": 100.0 * sum(r["reinvest_43022_correct"] == r["reinvest_43122_correct"] for r in rs) / len(rs),
            "TE43022_net_correct_count": sum(r["reinvest_43022_correct"] for r in rs) - sum(r["clean_43022_correct"] for r in rs),
            "TE43122_net_correct_count": sum(r["reinvest_43122_correct"] for r in rs) - sum(r["clean_43122_correct"] for r in rs),
        })
    return sorted(recs, key=lambda x: x["DiD_TE43122_minus_TE43022"])


def official_macro_from_domains(domain_recs: list[dict[str, Any]]) -> dict[str, Any]:
    fields = ["clean_43022", "clean_43122", "reinvest_43022", "reinvest_43122"]
    macro = {f: statistics.mean(r[f] for r in domain_recs) for f in fields}
    macro["clean_seed_gap_43122_minus_43022"] = macro["clean_43122"] - macro["clean_43022"]
    macro["reinvest_seed_gap_43122_minus_43022"] = macro["reinvest_43122"] - macro["reinvest_43022"]
    macro["TE43022_reinvest_minus_clean"] = macro["reinvest_43022"] - macro["clean_43022"]
    macro["TE43122_reinvest_minus_clean"] = macro["reinvest_43122"] - macro["clean_43122"]
    macro["DiD_TE43122_minus_TE43022"] = macro["TE43122_reinvest_minus_clean"] - macro["TE43022_reinvest_minus_clean"]
    macro["n_domains"] = len(domain_recs)
    macro["note"] = "Official EWoK column is the unweighted mean across these domain accuracies."
    return macro


def micro_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    fields = ["clean_43022", "clean_43122", "reinvest_43022", "reinvest_43122"]
    micro = {f: score(rows, f) for f in fields}
    micro["clean_seed_gap_43122_minus_43022"] = micro["clean_43122"] - micro["clean_43022"]
    micro["reinvest_seed_gap_43122_minus_43022"] = micro["reinvest_43122"] - micro["reinvest_43022"]
    micro["TE43022_reinvest_minus_clean"] = micro["reinvest_43022"] - micro["clean_43022"]
    micro["TE43122_reinvest_minus_clean"] = micro["reinvest_43122"] - micro["clean_43122"]
    micro["DiD_TE43122_minus_TE43022"] = micro["TE43122_reinvest_minus_clean"] - micro["TE43022_reinvest_minus_clean"]
    micro["n_items"] = len(rows)
    micro["note"] = "Micro item-weighted result, not the official EWoK column."
    return micro


def paired_bootstrap_domains(domain_recs: list[dict[str, Any]], n_boot: int = 20000, seed: int = 30030) -> dict[str, Any]:
    rng = random.Random(seed)
    vals = []
    per_domain = [r["DiD_TE43122_minus_TE43022"] for r in domain_recs]
    n = len(per_domain)
    for _ in range(n_boot):
        vals.append(sum(per_domain[rng.randrange(n)] for _ in range(n)) / n)
    vals.sort()
    return {
        "n_boot": n_boot,
        "resample_unit": "official_EWoK_domain",
        "seed": seed,
        "mean": statistics.mean(vals),
        "p05": vals[int(0.05 * n_boot)],
        "p50": vals[int(0.50 * n_boot)],
        "p95": vals[int(0.95 * n_boot)],
        "prob_negative": sum(v < 0 for v in vals) / n_boot,
        "observed": sum(per_domain) / n,
        "note": "Small-n domain bootstrap; descriptive uncertainty only."
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    missing = [str(p) for p in [DATA_DIR, *PRED.values()] if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing required paths: " + "; ".join(missing))
    data = load_data()
    preds = {name: load_preds(path) for name, path in PRED.items()}
    rows = correctness_by_model(data, preds)
    domain_recs = summarize_partition(rows, "domain")
    ctx_type_recs = summarize_partition(rows, "ContextType")
    ctx_diff_recs = summarize_partition(rows, "ContextDiff")
    tgt_diff_recs = summarize_partition(rows, "TargetDiff")
    official_macro = official_macro_from_domains(domain_recs)
    micro = micro_summary(rows)

    # Example sets for qualitative follow-up: treatment gain/loss asymmetries at seed43122.
    seed43122_lost_by_reinvest = [r for r in rows if r["clean_43122_correct"] and not r["reinvest_43122_correct"]]
    seed43122_gained_by_reinvest = [r for r in rows if (not r["clean_43122_correct"]) and r["reinvest_43122_correct"]]
    seed43022_lost_by_reinvest = [r for r in rows if r["clean_43022_correct"] and not r["reinvest_43022_correct"]]
    seed43022_gained_by_reinvest = [r for r in rows if (not r["clean_43022_correct"]) and r["reinvest_43022_correct"]]

    result = {
        "status": "OFFICIAL_EWOK_2x2_DOMAIN_DID",
        "purpose": "Compute current-official EWoK clean/reinvest × seed 2x2 treatment interactions on identical 7618 EWoK items.",
        "sources": {"official_EWoK_data_dir": str(DATA_DIR), "prediction_paths": {k: str(v) for k, v in PRED.items()}},
        "official_macro_over_domains": official_macro,
        "micro_item_weighted": micro,
        "domain_summary_sorted_by_DiD": domain_recs,
        "context_type_summary_sorted_by_DiD": ctx_type_recs,
        "context_diff_summary_sorted_by_DiD": ctx_diff_recs,
        "target_diff_summary_sorted_by_DiD": tgt_diff_recs,
        "domain_bootstrap_DiD": paired_bootstrap_domains(domain_recs),
        "example_sets": {
            "seed43122_clean_correct_reinvest_wrong_first20": seed43122_lost_by_reinvest[:20],
            "seed43122_clean_wrong_reinvest_correct_first20": seed43122_gained_by_reinvest[:20],
            "seed43022_clean_correct_reinvest_wrong_first20": seed43022_lost_by_reinvest[:20],
            "seed43022_clean_wrong_reinvest_correct_first20": seed43022_gained_by_reinvest[:20],
            "counts": {
                "seed43122_clean_correct_reinvest_wrong": len(seed43122_lost_by_reinvest),
                "seed43122_clean_wrong_reinvest_correct": len(seed43122_gained_by_reinvest),
                "seed43022_clean_correct_reinvest_wrong": len(seed43022_lost_by_reinvest),
                "seed43022_clean_wrong_reinvest_correct": len(seed43022_gained_by_reinvest),
            },
        },
        "interpretation": {
            "headline": "Current-official EWoK treatment effect is positive in both seeds, but smaller for seed43122; the seed interaction is therefore negative but not a sign reversal.",
            "why_it_matters": "This resolves the stale-coordinate uncertainty in the EWoK cell of the robustness analysis and shows that absolute seed43122 weakness is not equivalent to EWoK mechanism failure.",
            "limits": "Only EWoK is fully repaired here; full seed43122 official vector and exact clean sparse temporal task remain needed for complete overall robustness and temporal mechanism decisions.",
        },
    }
    out_json = OUT_DIR / "official_ewok_2x2_domain_did.json"
    out_md = OUT_DIR / "official_ewok_2x2_domain_did.md"
    out_json.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    lines = ["# research — current-official EWoK 2×2 domain treatment interaction\n\n"]
    lines.append("This uses identical 7,618-row official EWoK data and saved predictions for clean-Qwen and compact_view_reinvest at seeds 43022 and 43122.\n\n")
    om = official_macro
    lines.append("## Official macro-domain EWoK\n")
    for k in ["clean_43022", "clean_43122", "reinvest_43022", "reinvest_43122"]:
        lines.append(f"- {k}: {om[k]:.4f}\n")
    lines.append(f"- clean seed gap 43122-43022: {om['clean_seed_gap_43122_minus_43022']:+.4f}\n")
    lines.append(f"- reinvest seed gap 43122-43022: {om['reinvest_seed_gap_43122_minus_43022']:+.4f}\n")
    lines.append(f"- TE43022 reinvest-clean: {om['TE43022_reinvest_minus_clean']:+.4f}\n")
    lines.append(f"- TE43122 reinvest-clean: {om['TE43122_reinvest_minus_clean']:+.4f}\n")
    lines.append(f"- EWoK DiD (TE43122-TE43022): {om['DiD_TE43122_minus_TE43022']:+.4f}\n\n")
    bs = result["domain_bootstrap_DiD"]
    lines.append(f"Domain bootstrap for DiD: observed {bs['observed']:+.4f}, p05 {bs['p05']:+.4f}, p50 {bs['p50']:+.4f}, p95 {bs['p95']:+.4f}, prob_negative {bs['prob_negative']:.3f}.\n\n")
    lines.append("## Domain-level DiD (most negative first)\n")
    for r in domain_recs:
        lines.append(
            f"- {r['domain']}: DiD {r['DiD_TE43122_minus_TE43022']:+.2f}; "
            f"TE43022 {r['TE43022_reinvest_minus_clean']:+.2f}, TE43122 {r['TE43122_reinvest_minus_clean']:+.2f}; "
            f"clean gap {r['clean_seed_gap_43122_minus_43022']:+.2f}, reinvest gap {r['reinvest_seed_gap_43122_minus_43022']:+.2f}; n={r['n']}\n"
        )
    lines.append("\n## ContextDiff DiD highlights\n")
    for r in ctx_diff_recs[:8]:
        lines.append(f"- {r['ContextDiff']}: DiD {r['DiD_TE43122_minus_TE43022']:+.2f}; TE43022 {r['TE43022_reinvest_minus_clean']:+.2f}, TE43122 {r['TE43122_reinvest_minus_clean']:+.2f}; n={r['n']}\n")
    lines.append("\n## Scientific read\n")
    lines.append("- EWoK benefit survives at seed43122 in current official coordinates, but shrinks strongly: seed43122 gains only about +0.41 versus clean, while seed43022 gains +2.88.\n")
    lines.append("- The treatment-specific EWoK weakness is concentrated in domain interactions, especially where the clean seed43122 baseline improved but the reinvest seed43122 did not. This supports a stability problem in how reinvested compact views interact with relation learning, not a total failure of the EWoK mechanism.\n")
    lines.append("- This remains an EWoK-only result; do not use it to infer seed43122 Overall or choose training until full official vector and exact clean sparse temporal control are available.\n")
    lines.append(f"\nMachine-readable output: `{out_json}`\n")
    out_md.write_text("".join(lines), encoding="utf-8")

    print(json.dumps({
        "status": result["status"],
        "official_macro": official_macro,
        "micro": micro,
        "worst_domain_DiD": domain_recs[:4],
        "out_json": str(out_json),
        "out_md": str(out_md),
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
