#!/usr/bin/env python3
from __future__ import annotations

import json
import pathlib

ROOT = pathlib.Path("experiments/archive/initial_model_studies")
BASE = ROOT / "data/debertav2_b256_coordinate.json"
EWOK = ROOT / "data/debertav2_b256_full_ewok_word_tokenize_score.json"
EWOK_FILTER = ROOT / "data/local_ewok_word_tokenize_filter_summary.json"
AOA = ROOT / "data/debertav2_b256_aoa_local_ckpts_result.json"
CKPT_PROBE = ROOT / "data/checkpoint_resolution_probe.json"
LEADERBOARD = ROOT / "data/leaderboard_probe/leaderboard_config_parsed.json"
OUT = ROOT / "data/debertav2_b256_true_9of9_coordinate.json"
NOTE = (ROOT.parents[2] / 'research/notes/initial_model_studies/debertav2_b256_true_9of9_coordinate.md')


def read_json(p: pathlib.Path):
    return json.loads(p.read_text(encoding="utf-8"))


def mean(xs):
    return sum(float(x) for x in xs) / len(xs)


def get_rows(obj, key="strict-small"):
    if isinstance(obj, dict):
        if key in obj and isinstance(obj[key], list):
            return obj[key]
        for v in obj.values():
            got = get_rows(v, key)
            if got:
                return got
    elif isinstance(obj, list):
        for v in obj:
            got = get_rows(v, key)
            if got:
                return got
    return []


def num(x):
    try:
        return float(x)
    except Exception:
        return None


def main():
    base = read_json(BASE)
    ewok = read_json(EWOK)
    ewok_filter = read_json(EWOK_FILTER)
    aoa = read_json(AOA)
    ckpt_probe = read_json(CKPT_PROBE)
    b = base["scores_official_columns"]
    scores = {
        "BLiMP": float(b["blimp"]),
        "BLiMP Supplement": float(b["supplement"]),
        "EWoK": float(ewok["ewok_full_score"]),
        "Entity Tracking": float(b["entity_tracking"]),
        "COMPS": float(b["comps"]),
        "(Super)GLUE": float(b["superglue"]),
        "GlobalPIQA": float(b["global_piqa"]),
        "Reading": float(b["reading"]),
        "AoA": float(aoa["aoa"]),
    }
    nlp_keys = ["BLiMP", "BLiMP Supplement", "EWoK", "Entity Tracking", "COMPS", "(Super)GLUE", "GlobalPIQA"]
    human_keys = ["Reading", "AoA"]
    nlp_avg = mean([scores[k] for k in nlp_keys])
    human_avg = mean([scores[k] for k in human_keys])
    overall = mean([scores[k] for k in nlp_keys + human_keys])

    lb = read_json(LEADERBOARD)
    rows = get_rows(lb, "strict-small")
    valid_rows = []
    for r in rows:
        ov = num(r.get("Overall Average")) if isinstance(r, dict) else None
        if ov is not None:
            valid_rows.append(r)
    valid_rows = sorted(valid_rows, key=lambda r: float(r["Overall Average"]), reverse=True)
    better = [r for r in valid_rows if float(r["Overall Average"]) > overall]
    equal_or_below = [r for r in valid_rows if float(r["Overall Average"]) <= overall]
    insertion_rank = len(better) + 1
    top10 = valid_rows[:10]
    visible_reference = None
    for r in valid_rows:
        if r.get("Model_plain") == "wwm_curriculum_simplification_40k":
            visible_reference = r
            break
    gap_to_ref = None
    if visible_reference:
        gap_to_ref = {k: scores[k] - float(visible_reference[k]) for k in scores if visible_reference.get(k) is not None}
        gap_to_ref["Overall Average"] = overall - float(visible_reference["Overall Average"])
        gap_to_ref["NLP Average"] = nlp_avg - float(visible_reference["NLP Average"])
        gap_to_ref["Human-like Average"] = human_avg - float(visible_reference["Human-like Average"])

    payload = {
        "status": "BASELINE16K_DEBERTA_B256_FIRST_LOCAL_9OF9_COORDINATE",
        "model": "baseline16k DeBERTa-v2 8x480 WWM, 100M exposure, chck_100M",
        "model_path": base["model_path"],
        "scores_official_columns": scores,
        "subcolumns": base.get("subcolumns", {}),
        "NLP Average": nlp_avg,
        "Human-like Average": human_avg,
        "Overall Average": overall,
        "evidence_files": {
            "base_coordinate_without_full_ewok_repaired_aoa": str(BASE),
            "full_ewok_word_tokenize": str(EWOK),
            "ewok_filter_summary": str(EWOK_FILTER),
            "aoa_local_checkpoint": str(AOA),
            "checkpoint_resolution_probe": str(CKPT_PROBE),
            "leaderboard_parsed": str(LEADERBOARD),
        },
        "measurement_notes": {
            "ewok": "Local EWoK parquet; filtered with official vocab and nltk.word_tokenize using local data/nltk_data resources. Count 4374 raw -> 3809 kept -> 7618 swapped lines, matching official collate EWOK_SIZES total.",
            "aoa": "AoA run uses official StepConfig, target words, compute_surprisal_mlm, and AoAEvaluator scoring, but overrides local checkpoint resolution to load hf_model/chck_*M directly because transformers revision=... does not select local subdirectories.",
            "checkpoint_resolution": "Direct hashes and fixed MLM log-probs differ across checkpoints, showing saved checkpoints are distinct and old flat AoA was a local-revision loading artifact.",
            "strict_small_leaderboard": "Comparison uses parsed local leaderboard snapshot from the earlier retrieval; re-fetch before any submission claim.",
        },
        "ewok_filter_counts": {"raw_rows": ewok_filter["raw_rows"], "filtered_items": ewok_filter["filtered_items"], "jsonl_lines_with_swaps": ewok_filter["jsonl_lines_with_swaps"]},
        "aoa_step_mean_surprisal": aoa["step_mean_surprisal"],
        "leaderboard_strict_small": {
            "num_valid_rows": len(valid_rows),
            "insertion_rank_by_overall": insertion_rank,
            "num_rows_above": len(better),
            "num_rows_equal_or_below": len(equal_or_below),
            "top10_snapshot": top10,
            "visible_reference_wwm_curriculum_simplification_40k": visible_reference,
            "gap_to_visible_reference": gap_to_ref,
        },
        "interpretation": "This is the first complete local 9/9 coordinate for the protected model after repairing full EWoK access and local-checkpoint AoA. It is not SOTA: Overall is below the visible strict-small reference and far below the current parsed strict-small leaders in the snapshot.",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    ref_name = visible_reference.get("Model_plain") if visible_reference else "N/A"
    ref_ov = float(visible_reference["Overall Average"]) if visible_reference else float("nan")
    lines = [
        "# research — first complete local 9/9 coordinate for baseline16k DeBERTa-v2", "",
        f"Evidence JSON: `{OUT}`", "",
        "## 9 columns", "", "| column | score |", "|---|---:|",
    ]
    for k in nlp_keys + human_keys:
        lines.append(f"| {k} | {scores[k]:.4f} |")
    lines += ["", "## Aggregates", "", f"- NLP Average: **{nlp_avg:.4f}**", f"- Human-like Average: **{human_avg:.4f}**", f"- Overall Average: **{overall:.4f}**", "", "## Leaderboard snapshot comparison", "", f"- Insertion rank among parsed strict-small rows with non-null Overall: **{insertion_rank}** / {len(valid_rows)+1}", f"- Visible reference `{ref_name}` Overall: {ref_ov:.4f}; our gap: {overall-ref_ov:+.4f}" if visible_reference else "- Visible reference not found.", "", "## Measurement notes", "", "- EWoK uses local gated parquet with official vocab and `nltk.word_tokenize`; 4374 raw items -> 3809 retained items -> 7618 swapped lines.", "- AoA uses direct local checkpoint loading (`hf_model/chck_*M`) with official AoA target words, surprisal extraction and curve-fitness scoring; repaired AoA is negative (-0.1745), not the old flat 0.0 artifact.", "- Re-fetch leaderboard before any external claim; this coordinate is a research measurement, not a submission result.", "", "## Main gaps vs `wwm_curriculum_simplification_40k` reference", ""]
    if gap_to_ref:
        for k in ["BLiMP", "BLiMP Supplement", "EWoK", "Entity Tracking", "COMPS", "(Super)GLUE", "GlobalPIQA", "Reading", "AoA", "Overall Average"]:
            lines.append(f"- {k}: {gap_to_ref[k]:+.4f}")
    NOTE.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "out": str(OUT), "overall": overall, "nlp": nlp_avg, "human": human_avg, "rank": insertion_rank, "gap_to_wwm_curriculum": overall-ref_ov if visible_reference else None}, indent=2))

if __name__ == "__main__":
    main()
