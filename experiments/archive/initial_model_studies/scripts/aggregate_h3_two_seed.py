#!/usr/bin/env python3
"""Aggregate the two pre-registered research WWM forgetting diagnoses."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def linear_fit_predict(train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray) -> np.ndarray:
    mean = train_x.mean(axis=0)
    scale = train_x.std(axis=0)
    scale[scale < 1e-12] = 1.0
    train_z = (train_x - mean) / scale
    test_z = (test_x - mean) / scale
    design = np.column_stack([np.ones(len(train_z)), train_z])
    coefficients = np.linalg.lstsq(design, train_y, rcond=None)[0]
    return np.column_stack([np.ones(len(test_z)), test_z]) @ coefficients


def blocked_predictive_test(rows: list[dict]) -> dict:
    blocks = sorted(set(row["block"] for row in rows))
    base_errors: list[float] = []
    conflict_errors: list[float] = []
    for held_out in blocks:
        train = [row for row in rows if row["block"] != held_out]
        test = [row for row in rows if row["block"] == held_out]
        train_y = np.asarray([row["future_loss_delta"] for row in train], dtype=np.float64)
        test_y = np.asarray([row["future_loss_delta"] for row in test], dtype=np.float64)
        base_train = np.asarray([[row["current_loss"]] for row in train], dtype=np.float64)
        base_test = np.asarray([[row["current_loss"]] for row in test], dtype=np.float64)
        full_train = np.asarray(
            [[row["current_loss"], row["interference_score"]] for row in train], dtype=np.float64
        )
        full_test = np.asarray(
            [[row["current_loss"], row["interference_score"]] for row in test], dtype=np.float64
        )
        base_prediction = linear_fit_predict(base_train, train_y, base_test)
        full_prediction = linear_fit_predict(full_train, train_y, full_test)
        base_errors.extend((test_y - base_prediction).tolist())
        conflict_errors.extend((test_y - full_prediction).tolist())
    base_mse = float(np.mean(np.square(base_errors)))
    conflict_mse = float(np.mean(np.square(conflict_errors)))
    return {
        "leave_one_seed_transition_out_base_mse": base_mse,
        "leave_one_seed_transition_out_conflict_mse": conflict_mse,
        "relative_mse_change": (conflict_mse / base_mse - 1.0) if base_mse > 0 else None,
    }


def centered_test(rows: list[dict], permutations: int, seed: int) -> dict:
    centered_x = np.empty(len(rows), dtype=np.float64)
    centered_y = np.empty(len(rows), dtype=np.float64)
    blocks: dict[str, list[int]] = {}
    for index, row in enumerate(rows):
        blocks.setdefault(row["block"], []).append(index)
    for indices in blocks.values():
        x = np.asarray([rows[index]["interference_score"] for index in indices])
        y = np.asarray([rows[index]["future_loss_delta"] for index in indices])
        centered_x[indices] = x - x.mean()
        centered_y[indices] = y - y.mean()
    observed = float(spearmanr(centered_x, centered_y).statistic)
    rng = np.random.default_rng(seed)
    null = np.empty(permutations, dtype=np.float64)
    for permutation in range(permutations):
        permuted = centered_y.copy()
        for indices in blocks.values():
            permuted[indices] = rng.permutation(permuted[indices])
        null[permutation] = spearmanr(centered_x, permuted).statistic
    return {
        "n": len(rows),
        "blocks": len(blocks),
        "within_block_centered_spearman_rho": observed,
        "one_sided_blocked_permutation_p": float(
            (1 + np.sum(null >= observed)) / (permutations + 1)
        ),
        "null_rho_mean": float(np.nanmean(null)),
        "null_rho_q95": float(np.nanquantile(null, 0.95)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed42-json", required=True)
    parser.add_argument("--seed43-json", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--out-note", required=True)
    parser.add_argument("--permutations", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=301999)
    args = parser.parse_args()

    inputs = [Path(args.seed42_json).resolve(), Path(args.seed43_json).resolve()]
    payloads = [json.loads(path.read_text(encoding="utf-8")) for path in inputs]
    for path, payload in zip(inputs, payloads):
        if payload.get("status") != "GRADIENT_FORGETTING_DIAGNOSIS_COMPLETE":
            raise RuntimeError(f"diagnosis is not complete: {path} status={payload.get('status')}")

    rows: list[dict] = []
    block_results: dict[str, dict] = {}
    seed_rows: dict[str, list[dict]] = {}
    for payload in payloads:
        run_name = payload["run_name"]
        seed_rows[run_name] = []
        for transition_name, transition in payload["transitions"].items():
            block = f"{run_name}::{transition_name}"
            block_rows = []
            for source in transition["rows"]:
                row = {
                    "run_name": run_name,
                    "transition": transition_name,
                    "block": block,
                    "current_loss": float(source["current_loss"]),
                    "future_loss_delta": float(source["future_loss_delta"]),
                    "interference_score": float(source["interference_score"]),
                }
                rows.append(row)
                block_rows.append(row)
                seed_rows[run_name].append(row)
            rho = spearmanr(
                [row["interference_score"] for row in block_rows],
                [row["future_loss_delta"] for row in block_rows],
            )
            block_results[block] = {
                "n": len(block_rows),
                "spearman_rho": float(rho.statistic),
                "p": float(rho.pvalue),
            }

    combined = centered_test(rows, args.permutations, args.seed)
    per_seed = {
        run_name: centered_test(values, args.permutations // 2, args.seed + index + 1)
        for index, (run_name, values) in enumerate(seed_rows.items())
    }
    predictive = blocked_predictive_test(rows)
    seed_rhos = [result["within_block_centered_spearman_rho"] for result in per_seed.values()]
    positive_blocks = sum(result["spearman_rho"] > 0 for result in block_results.values())
    predictive_gain = (
        predictive["relative_mse_change"] is not None
        and predictive["relative_mse_change"] <= -0.10
    )
    promote = (
        all(rho >= 0.20 for rho in seed_rhos)
        and positive_blocks >= 5
        and combined["one_sided_blocked_permutation_p"] <= 0.01
        and predictive_gain
    )
    reject = (
        combined["one_sided_blocked_permutation_p"] >= 0.20
        and (abs(combined["within_block_centered_spearman_rho"]) < 0.10 or np.prod(seed_rhos) <= 0)
        and not predictive_gain
    )
    if promote:
        decision = "SUPPORTED_FOR_MINIMAL_INTERVENTION"
        next_action = "Run an exposure-matched 2x2 conflict intervention with two seeds; do not add data or architecture changes."
    elif reject:
        decision = "REJECT_H3_AS_PRIMARY_BOTTLENECK"
        next_action = "Do not build conflict-aware training. Move the main route to H1 coverage or a sharper H2 causal-use test."
    else:
        decision = "INCONCLUSIVE_ONE_FOLLOWUP_ONLY"
        next_action = "Permit one exact-update replication with full-gradient scope; kill H3 if it does not resolve the ambiguity."

    output = {
        "status": "H3_TWO_SEED_AGGREGATE_COMPLETE",
        "inputs": [str(path) for path in inputs],
        "rows": len(rows),
        "block_results": block_results,
        "per_seed": per_seed,
        "combined": combined,
        "predictive_test": predictive,
        "pre_registered_thresholds": {
            "per_seed_centered_rho_min": 0.20,
            "positive_blocks_min": 5,
            "combined_permutation_p_max": 0.01,
            "relative_mse_improvement_min": 0.10,
        },
        "decision": decision,
        "next_action": next_action,
    }
    out_json = Path(args.out_json).resolve()
    out_note = Path(args.out_note).resolve()
    write_json(out_json, output)
    lines = [
        "# research H3 two-seed decision",
        "",
        f"Decision: **{decision}**",
        "",
        f"Combined within-block Spearman rho: {combined['within_block_centered_spearman_rho']:+.4f}",
        f"Blocked permutation p: {combined['one_sided_blocked_permutation_p']:.6f}",
        f"Positive transition blocks: {positive_blocks}/{len(block_results)}",
        f"Conflict feature relative MSE change: {predictive['relative_mse_change']:+.4f}",
        "",
        "| seed | centered rho | permutation p |",
        "|---|---:|---:|",
    ]
    for run_name, result in per_seed.items():
        lines.append(
            f"| {run_name} | {result['within_block_centered_spearman_rho']:+.4f} | "
            f"{result['one_sided_blocked_permutation_p']:.6f} |"
        )
    lines.extend(["", f"Next action: {next_action}", "", f"Evidence JSON: `{out_json}`"])
    out_note.parent.mkdir(parents=True, exist_ok=True)
    out_note.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
