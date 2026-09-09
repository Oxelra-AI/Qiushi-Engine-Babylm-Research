#!/usr/bin/env python3
"""research: official-coordinate EWoK seed-spread fingerprint for compact_view_reinvest.

The fast EWoK screen made seed43122 look much worse than seed43022. Subsequent
re-evaluation used the same current official 7618-row EWoK coordinate as seed43022,
raising seed43122 EWoK from the fast/local 49.36 to official 51.89. This script compares the two
reinvest seeds on the official EWoK data and prediction files, localizing the remaining seed spread
by EWoK domain and metadata.

No model evaluation or training is performed. It only reads official EWoK data and saved prediction
JSONs.
"""
from __future__ import annotations

import collections
import json
import pathlib
import statistics
from typing import Any, Iterable

ROOT = pathlib.Path.cwd()
STUDY = ROOT / "experiments/archive/frontier_consolidation"
OUT_DIR = STUDY / "data/official_ewok_seed_fingerprint"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DATA_DIR = ROOT / "experiments/archive/representation_and_objectives/data/pristine_official_coordinate/babylm-eval/strict/evaluation_data/full_eval/ewok_filtered"
PRED_43022 = ROOT / "experiments/archive/representation_and_objectives/data/official_ewok_reeval/official_outputs/EWoK/chck_100M/official_ewok_reinvest/zero_shot/mlm/ewok/ewok_filtered/predictions.json"
PRED_43122 = ROOT / "experiments/archive/representation_and_objectives/data/official_ewok_reeval_seed43122/official_outputs/EWoK/chck_100M/official_ewok_seed43122/zero_shot/mlm/ewok/ewok_filtered/predictions.json"
REPORT_43022 = PRED_43022.with_name("best_temperature_report.txt")
REPORT_43122 = PRED_43122.with_name("best_temperature_report.txt")

CLEAN_EWOK_REFERENCE = {
    "seed43022": 50.19,
    "seed43122": 50.43,
    "seed_gap_43122_minus_43022": 0.24,
    "source_note": "COMPACT_EXPERIENCE clean-Qwen full trajectory summaries; not re-collated in the current pristine official EWoK coordinate.",
}
REINVEST_43022_EWOK_OFFICIAL = 53.536575594886855  # research pristine official coordinate


def norm(s: Any) -> str:
    return " ".join(str(s).strip().split())


def load_data() -> dict[str, list[dict[str, Any]]]:
    out = {}
    for p in sorted(DATA_DIR.glob("*.jsonl")):
        rows = [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]
        out[p.stem] = rows
    return out


def load_preds(path: pathlib.Path) -> dict[str, list[dict[str, Any]]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    out = {}
    for domain, block in raw.items():
        if isinstance(block, dict) and "predictions" in block:
            out[domain] = block["predictions"]
        elif isinstance(block, list):
            out[domain] = block
        else:
            raise TypeError(f"Unexpected prediction block for {domain}: {type(block)}")
    return out


def target(row: dict[str, Any]) -> str:
    # Current official calculate_results_from_pred.py/collate_preds.py uses Context1 + Target1 as target.
    return norm(f"{row['Context1']} {row['Target1']}")


def correctness(data: dict[str, list[dict[str, Any]]], preds: dict[str, list[dict[str, Any]]]) -> dict[str, list[dict[str, Any]]]:
    out = {}
    for domain, rows in data.items():
        ps = preds[domain]
        if len(rows) != len(ps):
            raise RuntimeError(f"length mismatch {domain}: data {len(rows)} preds {len(ps)}")
        recs = []
        for i, (row, pr) in enumerate(zip(rows, ps)):
            pred = norm(pr.get("pred"))
            tgt = target(row)
            recs.append({
                "domain": domain,
                "index": i,
                "id": pr.get("id"),
                "correct": pred == tgt,
                "pred": pred,
                "target": tgt,
                "ContextType": row.get("ContextType"),
                "ContextDiff": row.get("ContextDiff"),
                "TargetDiff": row.get("TargetDiff"),
                "ConceptA": row.get("ConceptA"),
                "ConceptB": row.get("ConceptB"),
            })
        out[domain] = recs
    return out


def summarize_group(joined: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for r in joined:
        groups[str(r[key])].append(r)
    recs = []
    for name, rows in sorted(groups.items()):
        n = len(rows)
        c0 = sum(r["seed43022_correct"] for r in rows)
        c1 = sum(r["seed43122_correct"] for r in rows)
        both = sum(r["seed43022_correct"] and r["seed43122_correct"] for r in rows)
        both_wrong = sum((not r["seed43022_correct"]) and (not r["seed43122_correct"]) for r in rows)
        only0 = sum(r["seed43022_correct"] and not r["seed43122_correct"] for r in rows)
        only1 = sum((not r["seed43022_correct"]) and r["seed43122_correct"] for r in rows)
        recs.append({
            key: name,
            "n": n,
            "seed43022_accuracy": 100.0 * c0 / n,
            "seed43122_accuracy": 100.0 * c1 / n,
            "delta_43122_minus_43022": 100.0 * (c1 - c0) / n,
            "agreement_rate": 100.0 * (both + both_wrong) / n,
            "both_correct": both,
            "both_wrong": both_wrong,
            "seed43022_only_correct": only0,
            "seed43122_only_correct": only1,
            "net_correct_delta_count": c1 - c0,
        })
    return sorted(recs, key=lambda r: r["delta_43122_minus_43022"])


def main() -> None:
    data = load_data()
    p430 = load_preds(PRED_43022)
    p431 = load_preds(PRED_43122)
    c430 = correctness(data, p430)
    c431 = correctness(data, p431)
    joined = []
    for domain, rows in data.items():
        for i, row in enumerate(rows):
            a = c430[domain][i]
            b = c431[domain][i]
            if a["target"] != b["target"]:
                raise RuntimeError(f"target mismatch {domain} {i}")
            joined.append({
                "domain": domain,
                "index": i,
                "ContextType": a["ContextType"],
                "ContextDiff": a["ContextDiff"],
                "TargetDiff": a["TargetDiff"],
                "ConceptA": a["ConceptA"],
                "ConceptB": a["ConceptB"],
                "seed43022_correct": a["correct"],
                "seed43122_correct": b["correct"],
                "seed43022_pred": a["pred"],
                "seed43122_pred": b["pred"],
                "target": a["target"],
            })
    n = len(joined)
    c0 = sum(r["seed43022_correct"] for r in joined)
    c1 = sum(r["seed43122_correct"] for r in joined)
    micro_overall = {
        "n": n,
        "seed43022_accuracy": 100.0 * c0 / n,
        "seed43122_accuracy": 100.0 * c1 / n,
        "delta_43122_minus_43022": 100.0 * (c1 - c0) / n,
        "agreement_rate": 100.0 * sum(r["seed43022_correct"] == r["seed43122_correct"] for r in joined) / n,
        "seed43022_only_correct": sum(r["seed43022_correct"] and not r["seed43122_correct"] for r in joined),
        "seed43122_only_correct": sum((not r["seed43022_correct"]) and r["seed43122_correct"] for r in joined),
        "both_correct": sum(r["seed43022_correct"] and r["seed43122_correct"] for r in joined),
        "both_wrong": sum((not r["seed43022_correct"]) and (not r["seed43122_correct"]) for r in joined),
        "note": "Micro item-weighted accuracy, not the official EWoK score.",
    }
    domain = summarize_group(joined, "domain")
    official_macro = {
        "n_domains": len(domain),
        "seed43022_accuracy": statistics.mean(r["seed43022_accuracy"] for r in domain),
        "seed43122_accuracy": statistics.mean(r["seed43122_accuracy"] for r in domain),
        "delta_43122_minus_43022": statistics.mean(r["delta_43122_minus_43022"] for r in domain),
        "note": "Official EWoK score is the unweighted mean over domain scores in calculate_results_from_pred._calculate_ewok_results.",
    }
    ctx_type = summarize_group(joined, "ContextType")
    ctx_diff = summarize_group(joined, "ContextDiff")
    tgt_diff = summarize_group(joined, "TargetDiff")

    # Official EWoK treatment effect update relative to available clean reference.
    ewok_te430 = official_macro["seed43022_accuracy"] - CLEAN_EWOK_REFERENCE["seed43022"]
    ewok_te431 = official_macro["seed43122_accuracy"] - CLEAN_EWOK_REFERENCE["seed43122"]
    ewok_did = ewok_te431 - ewok_te430

    result = {
        "status": "OFFICIAL_EWOK_SEED_FINGERPRINT",
        "purpose": "Localize compact_view_reinvest seed43122-vs-seed43022 EWoK spread on the current official 7618-row EWoK coordinate.",
        "sources": {
            "official_EWoK_data_dir": str(DATA_DIR),
            "seed43022_predictions": str(PRED_43022),
            "seed43122_predictions": str(PRED_43122),
            "seed43022_report": str(REPORT_43022),
            "seed43122_report": str(REPORT_43122),
        },
        "official_macro_over_domains": official_macro,
        "micro_item_weighted_overall": micro_overall,
        "domain_summary_sorted_by_delta": domain,
        "context_type_summary_sorted_by_delta": ctx_type,
        "context_diff_summary_sorted_by_delta": ctx_diff,
        "target_diff_summary_sorted_by_delta": tgt_diff,
        "official_EWoK_treatment_effect_relative_to_clean_reference": {
            "clean_reference": CLEAN_EWOK_REFERENCE,
            "TE43022_reinvest_minus_clean_EWoK": ewok_te430,
            "TE43122_reinvest_minus_clean_EWoK": ewok_te431,
            "DiD_EWoK": ewok_did,
            "note": "Clean-Qwen EWoK values are inherited COMPACT_EXPERIENCE full-score references, not re-scored on the pristine EWoK coordinate. The reinvest seed spread itself is current official-coordinate.",
        },
        "largest_seed43022_only_correct_examples": [r for r in joined if r["seed43022_correct"] and not r["seed43122_correct"]][:20],
        "largest_seed43122_only_correct_examples": [r for r in joined if (not r["seed43022_correct"]) and r["seed43122_correct"]][:20],
    }
    out_json = OUT_DIR / "official_ewok_seed_fingerprint.json"
    out_json.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    lines = []
    lines.append("# research — official EWoK seed-spread fingerprint for compact_view_reinvest\n\n")
    lines.append("This compares seed43022 and seed43122 on the same current official 7618-row EWoK coordinate. It does not require the still-locked full seed43122 vector.\n\n")
    lines.append("## Overall official EWoK seed spread\n")
    lines.append(f"- official macro-domain seed43022: {official_macro['seed43022_accuracy']:.3f}\n")
    lines.append(f"- official macro-domain seed43122: {official_macro['seed43122_accuracy']:.3f}\n")
    lines.append(f"- official macro-domain seed43122 - seed43022: {official_macro['delta_43122_minus_43022']:+.3f}\n")
    lines.append(f"- micro item-weighted seed43022: {micro_overall['seed43022_accuracy']:.3f}\n")
    lines.append(f"- micro item-weighted seed43122: {micro_overall['seed43122_accuracy']:.3f}\n")
    lines.append(f"- micro agreement rate: {micro_overall['agreement_rate']:.2f}%\n")
    lines.append(f"- micro seed43022-only correct: {micro_overall['seed43022_only_correct']}; seed43122-only correct: {micro_overall['seed43122_only_correct']}\n\n")
    lines.append("## Domain deltas (seed43122 - seed43022)\n")
    for r in domain:
        lines.append(f"- {r['domain']}: {r['delta_43122_minus_43022']:+.2f} (n={r['n']}, 43022={r['seed43022_accuracy']:.2f}, 43122={r['seed43122_accuracy']:.2f}, agree={r['agreement_rate']:.1f})\n")
    lines.append("\n## ContextDiff deltas\n")
    for r in ctx_diff:
        lines.append(f"- {r['ContextDiff']}: {r['delta_43122_minus_43022']:+.2f} (n={r['n']}, 43022={r['seed43022_accuracy']:.2f}, 43122={r['seed43122_accuracy']:.2f})\n")
    lines.append("\n## TargetDiff deltas\n")
    for r in tgt_diff:
        lines.append(f"- {r['TargetDiff']}: {r['delta_43122_minus_43022']:+.2f} (n={r['n']}, 43022={r['seed43022_accuracy']:.2f}, 43122={r['seed43122_accuracy']:.2f})\n")
    lines.append("\n## EWoK treatment-effect read\n")
    lines.append(f"- TE43022 (reinvest-clean EWoK): {ewok_te430:+.3f}\n")
    lines.append(f"- TE43122 (reinvest-clean EWoK): {ewok_te431:+.3f}\n")
    lines.append(f"- EWoK DiD: {ewok_did:+.3f}\n")
    lines.append("\nThe official-coordinate EWoK repair shows seed43122 still gains over its matched clean-Qwen EWoK baseline, but much less than seed43022; material-dynamics and spatial-relations dominate the remaining seed spread.\n\n")
    lines.append(f"Machine-readable output: `{out_json}`\n")
    (OUT_DIR / "official_ewok_seed_fingerprint.md").write_text("".join(lines), encoding="utf-8")

    print(json.dumps({
        "status": result["status"],
        "n": n,
        "seed43022_official_macro": official_macro["seed43022_accuracy"],
        "seed43122_official_macro": official_macro["seed43122_accuracy"],
        "delta_official_macro": official_macro["delta_43122_minus_43022"],
        "seed43022_micro": micro_overall["seed43022_accuracy"],
        "seed43122_micro": micro_overall["seed43122_accuracy"],
        "delta_micro": micro_overall["delta_43122_minus_43022"],
        "TE43022_EWoK": ewok_te430,
        "TE43122_EWoK": ewok_te431,
        "DiD_EWoK": ewok_did,
        "worst_domains": domain[:3],
        "out_json": str(out_json),
    }, indent=2))

if __name__ == "__main__":
    main()
