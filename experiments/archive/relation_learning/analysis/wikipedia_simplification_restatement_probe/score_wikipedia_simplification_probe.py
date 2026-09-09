#!/usr/bin/env python3
"""Score the WikiLarge T/U/N restatement probe on original DeBERTa arms.

This is intentionally not run by the CPU-only construction loop. Positive
gain_T_vs_U = NLL(U)-NLL(T) means the true Wikipedia source helps relative to an
unrelated same-register source. Positive gain_T_vs_N = NLL(N)-NLL(T) means it
helps relative to ordinary BabyLM text. gain_U_vs_N = NLL(N)-NLL(U) diagnoses
register/target-fit effects not specific to the true source.
"""

from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse
import csv
import json
import math
import statistics
import time
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer


HERE = _public_path('experiments/archive/relation_learning/analysis/wikipedia_simplification_restatement_probe')
ROOT = _public_path('.')
OLD_RUNS = _public_path('experiments/archive/frontier_consolidation/training/runs')
NEW_RUNS = _public_path('experiments/archive/relation_learning/training/runs')
RECORDS = _public_path('experiments/archive/relation_learning/analysis/wikipedia_simplification_restatement_probe/wikipedia_simplification_probe_records.jsonl')
PROBE_STATS = _public_path('experiments/archive/relation_learning/analysis/wikipedia_simplification_restatement_probe/probe_stats.json')
VALIDATION = _public_path('experiments/archive/relation_learning/analysis/wikipedia_simplification_restatement_probe/validation_results.json')
DEFAULT_OUT = _public_path('experiments/archive/relation_learning/analysis/wikipedia_simplification_restatement_probe/score_output')
CHECKPOINTS = ["chck_80M", "chck_90M", "chck_100M"]
ARM_CONFIGS = {
    "D_C_43022": _public_path('experiments/archive/frontier_consolidation/training/runs/full_p2c_c2p_abs_clean_dose2p64x_matched_rowholdout_deberta100M_seed43022'),
    "D_R_43022": _public_path('experiments/archive/frontier_consolidation/training/runs/full_p2c_c2p_abs_repeat_dose2p64x_matched_rowholdout_deberta100M_seed43022'),
    "D_V_43022": _public_path('experiments/archive/frontier_consolidation/training/runs/full_p2c_c2p_abs_view_dose2p64x_matched_rowholdout_deberta100M_seed43022'),
    "D_C_43122": _public_path('experiments/archive/frontier_consolidation/training/runs/full_p2c_c2p_abs_clean_dose2p64x_matched_rowholdout_deberta100M_seed43122'),
    "D_R_43122": _public_path('experiments/archive/frontier_consolidation/training/runs/full_p2c_c2p_abs_repeat_dose2p64x_matched_rowholdout_deberta100M_seed43122'),
    "D_V_43122": _public_path('experiments/archive/frontier_consolidation/training/runs/full_p2c_c2p_abs_view_dose2p64x_matched_rowholdout_deberta100M_seed43122'),
    "D_C_43222": _public_path('experiments/archive/relation_learning/training/runs/full_p2c_c2p_abs_clean_dose2p64x_matched_rowholdout_deberta100M_seed43222'),
    "D_R_43222": _public_path('experiments/archive/relation_learning/training/runs/full_p2c_c2p_abs_repeat_dose2p64x_matched_rowholdout_deberta100M_seed43222'),
    "D_V_43222": _public_path('experiments/archive/relation_learning/training/runs/full_p2c_c2p_abs_view_dose2p64x_matched_rowholdout_deberta100M_seed43222'),
}


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT))


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8"); return
    keys = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys); writer.writeheader(); writer.writerows(rows)


def se(values) -> float:
    values = [float(x) for x in values if math.isfinite(float(x))]
    return statistics.pstdev(values) / math.sqrt(len(values)) if len(values) > 1 else float("nan")


def parse_arm(name: str) -> tuple[str, int]:
    _, role, seed = name.split("_")
    return role, int(seed)


@torch.no_grad()
def score(model, records: list[dict], device: torch.device, pad_id: int, batch_size: int) -> list[dict]:
    output = []
    for start in range(0, len(records), batch_size):
        batch = records[start:start + batch_size]
        max_len = max(len(row["input_ids"]) for row in batch)
        ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
        attention = torch.zeros((len(batch), max_len), dtype=torch.long)
        for i, row in enumerate(batch):
            length = len(row["input_ids"])
            ids[i, :length] = torch.tensor(row["input_ids"], dtype=torch.long)
            attention[i, :length] = 1
        logp = model(input_ids=ids.to(device), attention_mask=attention.to(device)).logits.float().log_softmax(-1)
        for i, row in enumerate(batch):
            value = float(logp[i, int(row["mask_position"]), int(row["target_token_id"])].cpu())
            output.append({
                "record_id": row["record_id"], "pair_id": row["pair_id"], "target_key": row["target_key"],
                "condition_code": row["condition_code"], "overlap_bin": row["overlap_bin"],
                "token_class": row["token_class"], "surface_word_class": row["surface_word_class"],
                "target_word": row["target_word"],
                "target_token_id": row["target_token_id"], "nll": -value, "logp": value,
            })
    return output


def aggregate(scored: list[dict], out: Path) -> None:
    frame = pd.DataFrame(scored)
    index = ["arm", "role", "seed", "checkpoint", "pair_id", "target_key", "overlap_bin", "token_class", "surface_word_class", "target_word", "target_token_id"]
    target = frame.pivot(index=index, columns="condition_code", values="nll").reset_index()
    if {"T", "U", "N"} - set(target.columns):
        raise RuntimeError("T/U/N condition missing after score pivot")
    target["gain_T_vs_U"] = target["U"] - target["T"]
    target["gain_T_vs_N"] = target["N"] - target["T"]
    target["gain_U_vs_N"] = target["N"] - target["U"]
    target.to_csv(out / "wikipedia_target_TUN_rows.csv", index=False)

    gains = ["gain_T_vs_U", "gain_T_vs_N", "gain_U_vs_N"]
    by_checkpoint = target.groupby(["role", "seed", "checkpoint", "overlap_bin", "token_class"], dropna=False).agg(
        n_targets=("target_key", "nunique"), n_pairs=("pair_id", "nunique"),
        **{f"mean_{g}": (g, "mean") for g in gains},
        **{f"se_{g}": (g, lambda x: se(x)) for g in gains},
    ).reset_index()
    by_checkpoint.to_csv(out / "wikipedia_by_checkpoint_summary.csv", index=False)

    pair_checkpoint = target.groupby(["role", "seed", "checkpoint", "pair_id", "overlap_bin", "token_class"], dropna=False).agg(
        n_targets=("target_key", "nunique"), **{g: (g, "mean") for g in gains},
    ).reset_index()
    pair_late = pair_checkpoint.groupby(["role", "seed", "pair_id", "overlap_bin", "token_class"], dropna=False).agg(
        n_checkpoints=("checkpoint", "nunique"), n_targets=("n_targets", "mean"), **{g: (g, "mean") for g in gains},
    ).reset_index()
    pair_late = pair_late[pair_late["n_checkpoints"] == 3].copy()
    pair_late.to_csv(out / "wikipedia_pair_late_rows.csv", index=False)

    # Add all-bin and all-class summaries from pair-level values, never by
    # treating correlated masked targets as independent replicates.
    frames = [pair_late]
    all_class = pair_late.groupby(["role", "seed", "pair_id", "overlap_bin"], dropna=False).agg(
        n_checkpoints=("n_checkpoints", "first"), n_targets=("n_targets", "sum"), **{g: (g, "mean") for g in gains},
    ).reset_index(); all_class["token_class"] = "ALL"; frames.append(all_class)
    all_bin = pair_late.groupby(["role", "seed", "pair_id", "token_class"], dropna=False).agg(
        n_checkpoints=("n_checkpoints", "first"), n_targets=("n_targets", "sum"), **{g: (g, "mean") for g in gains},
    ).reset_index(); all_bin["overlap_bin"] = "ALL"; frames.append(all_bin)
    all_both = pair_late.groupby(["role", "seed", "pair_id"], dropna=False).agg(
        n_checkpoints=("n_checkpoints", "first"), n_targets=("n_targets", "sum"), **{g: (g, "mean") for g in gains},
    ).reset_index(); all_both["overlap_bin"] = "ALL"; all_both["token_class"] = "ALL"; frames.append(all_both)
    extended = pd.concat(frames, ignore_index=True)
    late = extended.groupby(["role", "seed", "overlap_bin", "token_class"], dropna=False).agg(
        n_pairs=("pair_id", "nunique"),
        **{f"mean_{g}": (g, "mean") for g in gains},
        **{f"se_pair_{g}": (g, lambda x: se(x)) for g in gains},
    ).reset_index()
    late.to_csv(out / "wikipedia_late_role_summary.csv", index=False)

    contrasts = []
    for (seed, overlap_bin, token_class), group in extended.groupby(["seed", "overlap_bin", "token_class"], dropna=False):
        for role_a, role_b in (("V", "C"), ("R", "C"), ("V", "R")):
            aa = group[group.role == role_a][["pair_id"] + gains]
            bb = group[group.role == role_b][["pair_id"] + gains]
            joined = aa.merge(bb, on="pair_id", suffixes=("_a", "_b"))
            for gain in gains:
                diff = joined[f"{gain}_a"] - joined[f"{gain}_b"]
                contrasts.append({
                    "seed": seed, "overlap_bin": overlap_bin, "token_class": token_class,
                    "contrast": f"{role_a}minus{role_b}", "estimand": gain, "n_pairs": len(diff),
                    "mean_difference": float(diff.mean()), "se_pair_difference": se(diff),
                    "fraction_positive": float((diff > 0).mean()),
                })
    contrast_frame = pd.DataFrame(contrasts)
    contrast_frame.to_csv(out / "wikipedia_late_arm_contrasts.csv", index=False)
    across_seed = contrast_frame.groupby(["overlap_bin", "token_class", "contrast", "estimand"], dropna=False).agg(
        n_seeds=("seed", "nunique"), mean_difference_across_seeds=("mean_difference", "mean"),
        seed_sd=("mean_difference", lambda x: statistics.stdev(x) if len(x) > 1 else float("nan")),
        mean_pairs=("n_pairs", "mean"),
    ).reset_index()
    across_seed.to_csv(out / "wikipedia_across_seed_contrasts.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arms", nargs="+", default=list(ARM_CONFIGS))
    parser.add_argument("--checkpoints", nargs="+", default=CHECKPOINTS)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    validation = json.loads(VALIDATION.read_text(encoding="utf-8"))
    stats = json.loads(PROBE_STATS.read_text(encoding="utf-8"))
    if validation.get("status") != "PASS":
        raise RuntimeError("probe validation_results.json is not PASS")
    records = read_jsonl(RECORDS)
    checkpoints = []
    for arm in args.arms:
        if arm not in ARM_CONFIGS: raise KeyError(arm)
        for checkpoint in args.checkpoints:
            model_path = ARM_CONFIGS[arm] / "hf_model" / checkpoint
            if not model_path.exists(): raise FileNotFoundError(model_path)
            checkpoints.append({"arm": arm, "checkpoint": checkpoint, "model_path": rel(model_path)})
    plan = {
        "status": "WIKIPEDIA_TUN_SCORE_PLAN", "created_utc": now(),
        "records_per_model": len(records), "pairs": stats["selected_pairs"],
        "paired_targets": validation["paired_targets"], "models": len(checkpoints),
        "device": args.device, "batch_size": args.batch_size,
        "probe_records": rel(RECORDS), "output_dir": rel(args.output_dir.resolve()),
        "checkpoints": checkpoints,
    }
    (args.output_dir / "score_plan.json").write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    if args.plan_only:
        print(json.dumps(plan, indent=2)); return
    tokenizer = AutoTokenizer.from_pretrained(str(ARM_CONFIGS[args.arms[0]] / "hf_model"), use_fast=True)
    probe_tokenizer = AutoTokenizer.from_pretrained(str(ROOT / stats["tokenizer"]), local_files_only=True, use_fast=True)
    if (len(tokenizer) != int(stats["tokenizer_vocab_size"]) or
            tokenizer.get_vocab() != probe_tokenizer.get_vocab() or
            tokenizer.all_special_ids != probe_tokenizer.all_special_ids):
        raise RuntimeError("checkpoint tokenizer does not match probe tokenizer")
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    pad_id = int(tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 3)
    scored = []; meta = []
    for item in checkpoints:
        arm = item["arm"]; checkpoint = item["checkpoint"]; role, seed = parse_arm(arm); start = time.time()
        print(f"[LOAD] {arm} {checkpoint}", flush=True)
        model = AutoModelForMaskedLM.from_pretrained(str(ROOT / item["model_path"]), torch_dtype=torch.float32).eval().to(device)
        rows = score(model, records, device, pad_id, args.batch_size)
        for row in rows: row.update({"arm": arm, "role": role, "seed": seed, "checkpoint": checkpoint})
        scored.extend(rows); elapsed = time.time() - start
        meta.append({"arm": arm, "role": role, "seed": seed, "checkpoint": checkpoint, "records": len(rows), "elapsed_seconds": elapsed, "device": str(device)})
        del model
        if device.type == "cuda": torch.cuda.empty_cache()
        print(f"[DONE] {arm} {checkpoint}: {len(rows)} records in {elapsed:.1f}s", flush=True)
    write_csv(args.output_dir / "wikipedia_scored_rows.csv", scored)
    write_csv(args.output_dir / "score_meta.csv", meta)
    aggregate(scored, args.output_dir)
    result = {**plan, "status": "WIKIPEDIA_TUN_SCORE_COMPLETE", "scored_rows": len(scored)}
    (args.output_dir / "score_summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
