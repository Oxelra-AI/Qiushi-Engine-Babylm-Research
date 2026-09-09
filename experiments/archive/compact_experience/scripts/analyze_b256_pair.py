#!/usr/bin/env python3
"""Analyze research b256/fixed-seq fixed-WWM vs WWM->token pair."""
from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path("experiments/archive/compact_experience")
OUT = ROOT / "data/b256_pair_eval"
NOTES = (ROOT.parents[2] / 'research/notes/compact_experience')
SUMMARY = OUT / "b256_pair_100M_eval_summary.json"
TRAJ = OUT / "b256_pair_trajectory_eval_summary.json"
SEQ_SUMMARY = ROOT / "data/curriculum_100M_eval/curriculum_100M_eval_summary.json"
INITIAL_MODEL_STUDIES_STEP322 = Path("experiments/archive/initial_model_studies/data/current_best_direct_checkpoint_recheck.json")
CSV_PATH = OUT / "b256_pair_compact_table.csv"
JSON_PATH = OUT / "b256_pair_mechanism_summary.json"
NOTE_PATH = (NOTES.parents[3] / 'research/notes/compact_experience/b256_fixedseq_pair_mechanism.md')

KEYS = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA_mean", "Reading"]


def load(p: Path):
    return json.loads(p.read_text(encoding="utf-8"))


def fmt(x):
    return "NA" if x is None else f"{float(x):.3f}"


def row_for(table, label):
    return table[label]


def delta(a, b):
    return {k: (a.get(k) - b.get(k) if a.get(k) is not None and b.get(k) is not None else None) for k in KEYS}


def proxy(row):
    vals = [row.get(k) for k in ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA_mean"]]
    if any(v is None for v in vals):
        return None
    return (3.0 / 28.0) * sum(vals) + (1.0 / 8.0) * (row.get("Reading") or 0.0)


def main():
    summary = load(SUMMARY)
    traj = load(TRAJ)
    seq_summary = load(SEQ_SUMMARY)
    initial_model_studies = load(INITIAL_MODEL_STUDIES_STEP322)

    table = traj["table"]
    rows = []
    for ck in ["chck_70M", "chck_80M", "chck_100M"]:
        for arm in ["wwm_fixed", "wwm_to_token"]:
            label = f"{arm}__{ck}"
            r = dict(table[label])
            r["target"] = label
            r["weighted_fast_proxy"] = proxy(r)
            rows.append(r)
        a = table[f"wwm_to_token__{ck}"]
        b = table[f"wwm_fixed__{ck}"]
        d = delta(a, b)
        d["target"] = f"wwm_to_token_minus_wwm_fixed__{ck}"
        d["weighted_fast_proxy"] = proxy(d)
        rows.append(d)

    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    with CSV_PATH.open("w", newline="", encoding="utf-8") as f:
        fields = ["target"] + KEYS + ["weighted_fast_proxy"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in fields})

    b256_final_delta = summary["weighted_fast_proxy_delta"].get("wwm_to_token_minus_wwm_fixed")
    seq_delta = seq_summary["weighted_fast_proxy_delta"].get("wwm_to_token_minus_wwm_fixed")
    seq_d = seq_summary["deltas"].get("wwm_to_token_minus_wwm_fixed")
    b256_d = summary["deltas"].get("wwm_to_token_minus_wwm_fixed")
    initial_model_studies_scores = initial_model_studies["profiles"]["wwm43_chck_100M"]["scores"]

    payload = {
        "status": "B256_FIXEDSEQ_PAIR_ANALYZED",
        "pair_runs": {
            "wwm_fixed": "experiments/archive/compact_experience/training/runs/wwm_fixed_100M_b256_seq256_seed43",
            "wwm_to_token": "experiments/archive/compact_experience/training/runs/wwm_to_token_100M_b256_seq256_seed43",
        },
        "evidence_files": {
            "final_summary": str(SUMMARY),
            "trajectory_summary": str(TRAJ),
            "compact_csv": str(CSV_PATH),
            "note": str(NOTE_PATH),
        },
        "b256_final_delta": b256_d,
        "b256_final_weighted_fast_proxy_delta": b256_final_delta,
        "seq_scheduled_final_delta": seq_d,
        "seq_scheduled_final_weighted_fast_proxy_delta": seq_delta,
        "initial_model_studies_step322_wwm100_scores_subset": initial_model_studies_scores,
    }
    JSON_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    lines = []
    lines += [
        "# research — research-regime fixed WWM vs WWM→token pair",
        "",
        f"Final pair summary: `{SUMMARY}`",
        f"Trajectory summary: `{TRAJ}`",
        f"Compact CSV: `{CSV_PATH}`",
        f"Machine-readable mechanism summary: `{JSON_PATH}`",
        "",
        "## What was recovered from the timed-out background task",
        "",
        "Wave 2 was incomplete, but wave 1 completed the matched paired comparison. Both wave-1 runs have `scientific_metrics.json`, `training_log.jsonl`, ten 10M-spaced checkpoints, and valid `hf_model/chck_100M` directories. They use the research-like regime: official corpus, DeBERTa-v2 8×480, baseline16k tokenizer, fixed seq256, batch256, seed43, 100M word exposure, and WWM 0.15 until the WWM→token arm switches at 70M.",
        "",
        "The pair is internally clean because the arms are identical through 70M except for insignificant evaluation noise; it is not a replacement for the INITIAL_MODEL_STUDIES 40.7028 coordinate because it was trained with the COMPACT_EXPERIENCE curriculum trainer rather than the original research trainer and the fixed-WWM Supplement score differs strongly from the old direct recheck.",
        "",
        "## Fast task trajectory",
        "",
        "Weighted fast proxy = (3/28) × (BLiMP + Supplement + EWoK + Entity + COMPS + GlobalPIQA mean) + (1/8) × Reading. It is a research statistic, not official Overall.",
        "",
        "| target | BLiMP | Supplement | EWoK | Entity | COMPS | GPIQA mean | Reading | weighted fast proxy |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append("| " + r["target"] + " | " + " | ".join(fmt(r.get(k)) for k in KEYS + ["weighted_fast_proxy"]) + " |")
    lines += [
        "",
        "## Mechanistic reading",
        "",
        "At 70M, before the switch, fixed WWM and WWM→token match almost exactly: proxy delta -0.001 and no meaningful column movement. The causal movement begins after token-level masking starts.",
        "",
        f"At 80M, WWM→token is higher on the fast proxy by {fmt(traj['trajectory_contrasts']['wwm_to_token_minus_wwm_fixed__chck_80M']['weighted_fast_proxy_delta'])}: BLiMP +0.62 and Supplement +3.20, but EWoK -1.27, COMPS -0.27, GlobalPIQA mean -0.47, and Reading -0.23. Entity is temporarily +0.81.",
        f"At 100M, WWM→token remains higher by {fmt(b256_final_delta)}: BLiMP +0.27 and Supplement +4.00 carry the gain, while EWoK -0.46, Entity -1.17, COMPS -0.54, and Reading -0.345 move down; GlobalPIQA mean is unchanged.",
        "",
        "This differs from the earlier seq-length-scheduled b64 run, where WWM→token at 100M had proxy delta " + fmt(seq_delta) + ", Supplement -5.20, Entity -1.90, Reading -0.46, and GlobalPIQA mean +3.41. The sign of the Supplement and GlobalPIQA effects is therefore not stable across these two training regimes. The stable part is that the late token switch changes task allocation after 70M and tends to reduce Reading and often Entity/relational continuity, even when it improves BLiMP or Supplement.",
        "",
        "The route implication is sharper than either early interpretation. Unconditional WWM→token should not be discarded solely because of the seq-scheduled screen, but it also should not become the main SOTA route from this fast pair alone. The useful signal is late token-level pressure interacting with the training regime; the unresolved problem is how to keep the lexical/sentence gains while preserving entity tracking, reading dynamics, and full-column transfer.",
        "",
        "## Alignment with INITIAL_MODEL_STUDIES research coordinate",
        "",
        "INITIAL_MODEL_STUDIES research direct recheck for the original research fixed-WWM `chck_100M` reported BLiMP 67.34, Supplement 65.2, EWoK 49.64, Entity 21.24, COMPS 53.11, Reading 7.33. The COMPACT_EXPERIENCE fixed-WWM rerun in this pair reports BLiMP 67.36, Supplement 57.2, EWoK 50.91, Entity 24.95, COMPS 52.72, Reading 8.23. The pair reproduces much of the research regime but not the exact old score surface, especially Supplement; conclusions should use within-pair deltas and then be checked by full official evaluation before any scale decision.",
        "",
        "## Next scientific work",
        "",
        "1. Run a full official-compatible evaluation for the two b256/fixed-seq `chck_100M` checkpoints, including (Super)GLUE, full GlobalPIQA, Reading, and AoA where the current evaluator supports it, so the +0.182 fast proxy is tested on the actual Overall surface.",
        "2. In parallel, inspect why the COMPACT_EXPERIENCE fixed-WWM Supplement score is far below the INITIAL_MODEL_STUDIES research direct recheck despite similar BLiMP and recipe; compare data order, checkpoint schedule, trainer implementation, and evaluation output paths rather than treating this rerun as identical to research.",
        "3. If full evaluation preserves the same split, build the next mechanism as conditional late granularity rather than a hard global switch: retain WWM on entity/long/clause-rich examples or a fixed fraction of late batches, while adding token-level pressure only where it improves lexical/sentence judgments. Do not mix this with paired-rewrite data until the granularity mechanism is isolated.",
    ]
    NOTE_PATH.parent.mkdir(parents=True, exist_ok=True)
    NOTE_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"note": str(NOTE_PATH), "json": str(JSON_PATH), "csv": str(CSV_PATH)}, indent=2))


if __name__ == "__main__":
    main()
