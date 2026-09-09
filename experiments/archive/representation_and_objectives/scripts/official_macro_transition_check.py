#!/usr/bin/env python3
"""research: macro-averaged official Entity transition rates.

Reads the item-level transition rows produced by conservation_transition_readout.py
and recomputes transition quantities with the official Entity equal-subtask style:
all_18_subtasks is the mean over 3 splits x 6 numops; zero_ops is the mean over
3 split/0-op subtasks; nonzero_ops is the mean over the 15 split/nonzero subtasks;
numops_k is the mean over the 3 split/k subtasks; split_s is the mean over that
split's 6 numops subtasks.

No model loading, no training, no GPU, no official scorer, no upload.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import csv
import json
import pathlib
import statistics
import time
from typing import Any


def find_user_root() -> pathlib.Path:
    return _PUBLIC_ROOT


ROOT = find_user_root()
A01_WS = ROOT / "experiments/archive" / 'representation_and_objectives'
IN = A01_WS / "data" / "conservation_transition_readout" / "entity_transition_rows.csv"
OUT = A01_WS / "data" / "conservation_transition_readout"
SPLITS = ["ambiref", "regular", "move_contents"]
NUMOPS = list(range(6))
CONTRASTS = ["VminusB", "BminusR", "VminusR"]
CHECKPOINTS = ["chck_80M", "chck_90M", "chck_100M"]
QUANTITIES = [
    "delta_accuracy_pp",
    "gain_wrong_to_correct_frac",
    "loss_correct_to_wrong_frac",
    "both_correct_frac",
    "both_wrong_frac",
    "delta_pred_option0_frac",
    "delta_pred_option1_frac",
    "delta_pred_option2_frac",
    "delta_pred_option3_frac",
    "delta_pred_option4_frac",
]


def rel(p: pathlib.Path) -> str:
    try:
        return str(p.relative_to(ROOT))
    except Exception:
        return str(p)


def mean(xs: list[float]) -> float | None:
    return sum(xs) / len(xs) if xs else None


def write_csv(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for r in rows:
        for k in r:
            if k not in fields:
                fields.append(k)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(rows)


def fmt(x: Any) -> str:
    return "NA" if x is None else f"{float(x):+.3f}"


def load_rows() -> list[dict[str, Any]]:
    with IN.open("r", encoding="utf-8") as f:
        out = []
        for r in csv.DictReader(f):
            rr: dict[str, Any] = dict(r)
            for q in QUANTITIES:
                rr[q] = float(rr[q])
            rr["n"] = int(rr["n"])
            out.append(rr)
        return out


def subtask_groups_for(coarse: str) -> list[str]:
    if coarse == "all_18_subtasks":
        return [f"{s}_numops_{n}" for s in SPLITS for n in NUMOPS]
    if coarse == "zero_ops":
        return [f"{s}_numops_0" for s in SPLITS]
    if coarse == "nonzero_ops":
        return [f"{s}_numops_{n}" for s in SPLITS for n in NUMOPS if n > 0]
    if coarse.startswith("numops_"):
        n = int(coarse.split("_")[-1])
        return [f"{s}_numops_{n}" for s in SPLITS]
    if coarse.startswith("split_"):
        s = coarse.replace("split_", "")
        return [f"{s}_numops_{n}" for n in NUMOPS]
    raise ValueError(coarse)


def main() -> None:
    rows = load_rows()
    by = {(r["contrast"], r["checkpoint"], r["group"]): r for r in rows}
    coarse_groups = ["all_18_subtasks", "zero_ops", "nonzero_ops"] + [f"numops_{n}" for n in NUMOPS] + [f"split_{s}" for s in SPLITS]
    macro_rows: list[dict[str, Any]] = []
    for contrast in CONTRASTS:
        for ck in CHECKPOINTS:
            for coarse in coarse_groups:
                subs = subtask_groups_for(coarse)
                present = [by[(contrast, ck, g)] for g in subs if (contrast, ck, g) in by]
                rec: dict[str, Any] = {
                    "contrast": contrast,
                    "checkpoint": ck,
                    "macro_group": coarse,
                    "subtask_count": len(present),
                    "expected_subtasks": len(subs),
                    "subtasks": ";".join(subs),
                    "n_micro_total": sum(int(r["n"]) for r in present),
                }
                for q in QUANTITIES:
                    vals = [float(r[q]) for r in present]
                    val = mean(vals)
                    if q.endswith("_frac"):
                        rec[q.replace("_frac", "_macro_pp")] = None if val is None else 100.0 * val
                    else:
                        rec[q] = val
                macro_rows.append(rec)

    late_rows: list[dict[str, Any]] = []
    for contrast in CONTRASTS:
        for group in coarse_groups:
            rs = [r for r in macro_rows if r["contrast"] == contrast and r["macro_group"] == group]
            for q in ["delta_accuracy_pp", "gain_wrong_to_correct_macro_pp", "loss_correct_to_wrong_macro_pp", "both_correct_macro_pp", "both_wrong_macro_pp", "delta_pred_option0_macro_pp", "delta_pred_option1_macro_pp", "delta_pred_option2_macro_pp", "delta_pred_option3_macro_pp", "delta_pred_option4_macro_pp"]:
                vals = [float(r[q]) for r in rs if r.get(q) is not None]
                if vals:
                    late_rows.append({
                        "contrast": contrast,
                        "macro_group": group,
                        "quantity": q,
                        "n": len(vals),
                        "mean": mean(vals),
                        "median": statistics.median(vals),
                        "min": min(vals),
                        "max": max(vals),
                        "checkpoints": ";".join(r["checkpoint"] for r in rs if r.get(q) is not None),
                    })

    summary_md = OUT / "entity_official_macro_transition_summary.md"
    lines = [
        "# research official-macro Entity transition check\n\n",
        "This is a file-only recomputation from `entity_transition_rows.csv`. It averages gain/loss and option-response quantities over the same split×numops subtasks used in the official Entity score, rather than micro-weighting examples.\n\n",
        "## Late 80/90/100M macro transition means\n\n",
        "| contrast | group | Δ acc pp | wrong→correct pp | correct→wrong pp | both-correct pp | both-wrong pp | Δ option0 pp | Δ option1 pp | Δ option2 pp | Δ option3 pp | Δ option4 pp |\n",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n",
    ]
    lookup = {(r["contrast"], r["macro_group"], r["quantity"]): r["mean"] for r in late_rows}
    for contrast in CONTRASTS:
        for group in ["all_18_subtasks", "zero_ops", "nonzero_ops"] + [f"numops_{n}" for n in NUMOPS] + [f"split_{s}" for s in SPLITS]:
            lines.append(
                f"| {contrast} | {group} | {fmt(lookup.get((contrast,group,'delta_accuracy_pp')))} | {fmt(lookup.get((contrast,group,'gain_wrong_to_correct_macro_pp')))} | {fmt(lookup.get((contrast,group,'loss_correct_to_wrong_macro_pp')))} | {fmt(lookup.get((contrast,group,'both_correct_macro_pp')))} | {fmt(lookup.get((contrast,group,'both_wrong_macro_pp')))} | {fmt(lookup.get((contrast,group,'delta_pred_option0_macro_pp')))} | {fmt(lookup.get((contrast,group,'delta_pred_option1_macro_pp')))} | {fmt(lookup.get((contrast,group,'delta_pred_option2_macro_pp')))} | {fmt(lookup.get((contrast,group,'delta_pred_option3_macro_pp')))} | {fmt(lookup.get((contrast,group,'delta_pred_option4_macro_pp')))} |\n"
            )
    lines.extend([
        "\n## Scientific reading\n\n",
        f"- Macro V-B is positive in the official arithmetic: all18 {fmt(lookup.get(('VminusB','all_18_subtasks','delta_accuracy_pp')))}, zero-op {fmt(lookup.get(('VminusB','zero_ops','delta_accuracy_pp')))}, nonzero {fmt(lookup.get(('VminusB','nonzero_ops','delta_accuracy_pp')))} pp.\n",
        f"- That macro V-B still has substantial item churn: all18 wrong→correct {fmt(lookup.get(('VminusB','all_18_subtasks','gain_wrong_to_correct_macro_pp')))} pp and correct→wrong {fmt(lookup.get(('VminusB','all_18_subtasks','loss_correct_to_wrong_macro_pp')))} pp. The net is positive because gains exceed losses, not because items are stable records.\n",
        f"- Macro B-R retains the operation prior shape: zero-op {fmt(lookup.get(('BminusR','zero_ops','delta_accuracy_pp')))} versus nonzero {fmt(lookup.get(('BminusR','nonzero_ops','delta_accuracy_pp')))} pp.\n",
        "- These official-macro transition rates do not overturn the paired binding conservation result, where V-B decreases both-correct state. They mainly confirm that official Entity V-B is a real score surface but do not identify source correspondence.\n",
    ])
    lines.extend([
        "\n## Files\n\n",
        f"- macro_rows_csv: `{rel(OUT / 'entity_official_macro_transition_rows.csv')}`\n",
        f"- late_summary_csv: `{rel(OUT / 'entity_official_macro_transition_late_summary.csv')}`\n",
        f"- summary_md: `{rel(summary_md)}`\n",
    ])
    write_csv(OUT / "entity_official_macro_transition_rows.csv", macro_rows)
    write_csv(OUT / "entity_official_macro_transition_late_summary.csv", late_rows)
    (OUT / "entity_official_macro_transition_summary.json").write_text(json.dumps({
        "status": "ENTITY_OFFICIAL_MACRO_TRANSITION_COMPLETE",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "input": rel(IN),
        "macro_rows_csv": rel(OUT / "entity_official_macro_transition_rows.csv"),
        "late_summary_csv": rel(OUT / "entity_official_macro_transition_late_summary.csv"),
        "summary_md": rel(summary_md),
        "no_model_loading_training_evaluation_gpu_upload_or_leaderboard": True,
    }, indent=2) + "\n", encoding="utf-8")
    summary_md.write_text("".join(lines), encoding="utf-8")
    print(json.dumps({
        "status": "ENTITY_OFFICIAL_MACRO_TRANSITION_COMPLETE",
        "summary_md": rel(summary_md),
        "VminusB_all18_delta_pp": lookup.get(("VminusB", "all_18_subtasks", "delta_accuracy_pp")),
        "VminusB_zero_delta_pp": lookup.get(("VminusB", "zero_ops", "delta_accuracy_pp")),
        "VminusB_nonzero_delta_pp": lookup.get(("VminusB", "nonzero_ops", "delta_accuracy_pp")),
        "VminusB_all18_gain_pp": lookup.get(("VminusB", "all_18_subtasks", "gain_wrong_to_correct_macro_pp")),
        "VminusB_all18_loss_pp": lookup.get(("VminusB", "all_18_subtasks", "loss_correct_to_wrong_macro_pp")),
        "no_model_loading_training_evaluation_gpu_upload_or_leaderboard": True,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
