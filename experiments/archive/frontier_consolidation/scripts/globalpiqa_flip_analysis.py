#!/usr/bin/env python3
"""GlobalPIQA item-flip analysis for the research shuffled private-tail endpoint hypothesis."""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import json
import math
from pathlib import Path
from typing import Any

ROOT = _public_path('.')
WORK = _public_path('experiments/archive/frontier_consolidation')
OUT_DIR = _public_path('experiments/archive/frontier_consolidation/data/shuffled_tail_globalpiqa_flip_analysis')
OUT_JSON = _public_path('experiments/archive/frontier_consolidation/data/shuffled_tail_globalpiqa_flip_analysis/globalpiqa_flip_analysis.json')
OUT_MD = _public_path('research/documents/frontier_consolidation/data/shuffled_tail_globalpiqa_flip_analysis/globalpiqa_flip_analysis.md')

ANS = {
    "parallel": _public_path('experiments/archive/frontier_consolidation/data/chck82_hf_public_bundle/live_submit_dry_run/eval-datasets/global_piqa_parallel.json'),
    "nonparallel": _public_path('experiments/archive/frontier_consolidation/data/chck82_hf_public_bundle/live_submit_dry_run/eval-datasets/global_piqa_nonparallel.json'),
}
PRED = {
    "protected_chck82": {
        "parallel": _public_path('experiments/archive/representation_and_objectives/data/chck82_fast_submission_materialization/collate_fast/results/hf_model/all_full_preds_and_fast_scores_mlm.json'),
        "nonparallel": _public_path('experiments/archive/representation_and_objectives/data/chck82_fast_submission_materialization/collate_fast/results/hf_model/all_full_preds_and_fast_scores_mlm.json'),
    },
    "aligned_tail": {
        "parallel": _public_path('experiments/archive/frontier_consolidation/data/frozen82_tail4M_aligned_eval/official_outputs/frozen82_tail4M_aligned/GlobalPIQA_parallel/final/full_frozen82_tail4M_aligned_GlobalPIQA_parallel/zero_shot/mlm/global_piqa_parallel/global_piqa_parallel/predictions.json'),
        "nonparallel": _public_path('experiments/archive/frontier_consolidation/data/frozen82_tail4M_aligned_eval/official_outputs/frozen82_tail4M_aligned/GlobalPIQA_nonparallel/final/full_frozen82_tail4M_aligned_GlobalPIQA_nonparallel/zero_shot/mlm/global_piqa_nonparallel/global_piqa_nonparallel/predictions.json'),
    },
    "shuffled_tail": {
        "parallel": _public_path('experiments/archive/frontier_consolidation/data/frozen82_tail4M_shuffled_eval/official_outputs/frozen82_tail4M_shuffled/GlobalPIQA_parallel/final/full_frozen82_tail4M_shuffled_GlobalPIQA_parallel/zero_shot/mlm/global_piqa_parallel/global_piqa_parallel/predictions.json'),
        "nonparallel": _public_path('experiments/archive/frontier_consolidation/data/frozen82_tail4M_shuffled_eval/official_outputs/frozen82_tail4M_shuffled/GlobalPIQA_nonparallel/final/full_frozen82_tail4M_shuffled_GlobalPIQA_nonparallel/zero_shot/mlm/global_piqa_nonparallel/global_piqa_nonparallel/predictions.json'),
    },
}


def rel(p: Path | str) -> str:
    try:
        return str(Path(p).resolve().relative_to(ROOT))
    except Exception:
        return str(p)


def load_predictions(model: str, half: str) -> dict[str, Any]:
    p = PRED[model][half]
    d = json.loads(p.read_text(encoding="utf-8"))
    # protected all_full file has top-level global_piqa_parallel/nonparallel keys.
    if model == "protected_chck82":
        key = "global_piqa_parallel" if half == "parallel" else "global_piqa_nonparallel"
        d = d[key]
    return d


def norm(x: Any) -> Any:
    return x.strip() if isinstance(x, str) else x


def correctness(model: str, half: str, answers: dict[str, Any]) -> dict[str, dict[str, Any]]:
    preds = load_predictions(model, half)
    out = {}
    for eid, gold in answers.items():
        rec = preds.get(eid)
        pred = None
        if rec and rec.get("predictions"):
            pred = rec["predictions"][0].get("pred")
        out[eid] = {"pred": pred, "gold": gold, "correct": norm(pred) == norm(gold)}
    return out


def binom_two_sided_p(n: int, k: int) -> float:
    # exact two-sided sign/binomial test for p=0.5: twice the smaller tail, capped at 1.
    if n <= 0:
        return 1.0
    lo = min(k, n - k)
    tail = sum(math.comb(n, i) for i in range(lo + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def compare(base: str, cand: str, all_corr: dict[str, dict[str, dict[str, dict[str, Any]]]]) -> dict[str, Any]:
    halves = {}
    total_weighted_delta = 0.0
    total_improve = 0
    total_damage = 0
    examples = []
    for half, answers in ((h, json.loads(ANS[h].read_text(encoding="utf-8"))) for h in ["parallel", "nonparallel"]):
        b = all_corr[base][half]
        c = all_corr[cand][half]
        improve = []
        damage = []
        same_correct = 0
        same_wrong = 0
        for eid in answers.keys():
            bc = bool(b[eid]["correct"])
            cc = bool(c[eid]["correct"])
            if (not bc) and cc:
                improve.append(eid)
            elif bc and (not cc):
                damage.append(eid)
            elif bc and cc:
                same_correct += 1
            else:
                same_wrong += 1
        n = len(answers)
        base_correct = sum(int(v["correct"]) for v in b.values())
        cand_correct = sum(int(v["correct"]) for v in c.values())
        score_base = 100 * base_correct / n
        score_cand = 100 * cand_correct / n
        weighted_delta = 0.5 * (score_cand - score_base)
        total_weighted_delta += weighted_delta
        total_improve += len(improve)
        total_damage += len(damage)
        examples.extend([{"half": half, "eid": eid, "direction": "improved", "gold": b[eid]["gold"], "base_pred": b[eid]["pred"], "cand_pred": c[eid]["pred"]} for eid in improve[:8]])
        examples.extend([{"half": half, "eid": eid, "direction": "damaged", "gold": b[eid]["gold"], "base_pred": b[eid]["pred"], "cand_pred": c[eid]["pred"]} for eid in damage[:8]])
        halves[half] = {
            "n": n,
            "base_correct": base_correct,
            "candidate_correct": cand_correct,
            "base_score": score_base,
            "candidate_score": score_cand,
            "score_delta": score_cand - score_base,
            "weighted_column_delta_contribution": weighted_delta,
            "improved_count": len(improve),
            "damaged_count": len(damage),
            "same_correct": same_correct,
            "same_wrong": same_wrong,
            "discordant": len(improve) + len(damage),
            "mcnemar_exact_sign_p_two_sided": binom_two_sided_p(len(improve) + len(damage), len(improve)),
            "improved_ids_sample": improve[:20],
            "damaged_ids_sample": damage[:20],
        }
    total_discordant = total_improve + total_damage
    return {
        "base": base,
        "candidate": cand,
        "halves": halves,
        "globalpiqa_column_delta": total_weighted_delta,
        "total_improved": total_improve,
        "total_damaged": total_damage,
        "total_discordant": total_discordant,
        "pooled_exact_sign_p_two_sided_unweighted": binom_two_sided_p(total_discordant, total_improve),
        "sample_flips": examples[:32],
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    answers = {h: json.loads(p.read_text(encoding="utf-8")) for h, p in ANS.items()}
    all_corr: dict[str, dict[str, dict[str, dict[str, Any]]]] = {}
    summaries = {}
    for model in PRED:
        all_corr[model] = {}
        summaries[model] = {}
        for half in ["parallel", "nonparallel"]:
            c = correctness(model, half, answers[half])
            all_corr[model][half] = c
            n = len(c)
            correct = sum(int(v["correct"]) for v in c.values())
            summaries[model][half] = {"n": n, "correct": correct, "score": 100 * correct / n}
        summaries[model]["globalpiqa"] = 0.5 * (summaries[model]["parallel"]["score"] + summaries[model]["nonparallel"]["score"])
    comparisons = {
        "shuffled_minus_protected": compare("protected_chck82", "shuffled_tail", all_corr),
        "aligned_minus_protected": compare("protected_chck82", "aligned_tail", all_corr),
        "shuffled_minus_aligned": compare("aligned_tail", "shuffled_tail", all_corr),
    }
    payload = {
        "status": "COMPLETE",
        "created_utc": __import__("datetime").datetime.now(__import__("datetime").UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "answer_paths": {k: rel(v) for k, v in ANS.items()},
        "prediction_paths": {m: {h: rel(p) for h, p in halves.items()} for m, halves in PRED.items()},
        "summaries": summaries,
        "comparisons": comparisons,
        "scientific_reading": "The shuffled-tail endpoint margin over protected chck_82M is driven by a small number of GlobalPIQA binary-choice flips. The exact sign tests quantify whether the net flip count is distinguishable from symmetric item movement before spending GPU on replay or elevating the endpoint.",
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = ["# research shuffled-tail GlobalPIQA flip analysis", "", f"Status: **{payload['status']}**", ""]
    lines.append("## Scores")
    lines.append("| model | parallel | nonparallel | GlobalPIQA |")
    lines.append("|---|---:|---:|---:|")
    for model, s in summaries.items():
        lines.append(f"| {model} | {s['parallel']['correct']}/{s['parallel']['n']} = {s['parallel']['score']:.6f} | {s['nonparallel']['correct']}/{s['nonparallel']['n']} = {s['nonparallel']['score']:.6f} | {s['globalpiqa']:.6f} |")
    lines.append("")
    lines.append("## Pairwise flips")
    lines.append("| comparison | GP delta | improved | damaged | discordant | exact sign p |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for name, c in comparisons.items():
        lines.append(f"| {name} | {c['globalpiqa_column_delta']:.6f} | {c['total_improved']} | {c['total_damaged']} | {c['total_discordant']} | {c['pooled_exact_sign_p_two_sided_unweighted']:.6f} |")
    lines.append("")
    lines.append("Scientific reading: use this as a stability warning for the 41.984 endpoint hypothesis. A small positive Overall margin dominated by GlobalPIQA should be replayed and repeated before it replaces the bit-identical chck_82M carrier; it is not evidence for source-correspondence transfer.")
    lines.append("")
    lines.append(f"JSON: `{rel(OUT_JSON)}`")
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status":"COMPLETE","out_json":rel(OUT_JSON),"out_md":rel(OUT_MD),"shuffled_delta":comparisons['shuffled_minus_protected']['globalpiqa_column_delta'],"shuffled_improved":comparisons['shuffled_minus_protected']['total_improved'],"shuffled_damaged":comparisons['shuffled_minus_protected']['total_damaged']}, indent=2), flush=True)


if __name__ == "__main__":
    main()
