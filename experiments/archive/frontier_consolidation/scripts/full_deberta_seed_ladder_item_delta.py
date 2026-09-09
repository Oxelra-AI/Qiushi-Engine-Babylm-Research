#!/usr/bin/env python3
"""research: item-level compact-minus-repeat seed-noise analysis over the full-DeBERTa ladder.

Consumes the stable-family selected ladder per-target payloads produced by
`full_deberta_seed_ladder_stable_eval.py` plus reused existing payloads.
It measures whether the *items* helped/hurt by compact views recur across two
full-DeBERTa seeds at each 10M checkpoint. This is CPU/file-only.

The aggregate ladder tells how large the selected family movement is. This item
analysis tells whether that movement has a reproducible item signature or is a
basin-specific threshold realization over shared item difficulty.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse
import csv
import importlib.util
import json
import math
import pathlib
import statistics
import sys
import time
from collections import defaultdict
from typing import Any


def find_user_root() -> pathlib.Path:
    return _PUBLIC_ROOT


USER_ROOT = find_user_root()
WS = USER_ROOT / "experiments/archive/frontier_consolidation"
MOVEMENT_READER = WS / "scripts/selected_prediction_movement_reader.py"
DEFAULT_LADDER_OUT = WS / "data/full_deberta_seed_ladder_stable_eval"
DEFAULT_OUT = WS / "data/full_deberta_seed_ladder_item_delta"
CHECKPOINTS = [f"chck_{i}M" for i in range(10, 101, 10)]
STABLE_FIVE = {"BLiMP", "Supplement", "EWoK", "Entity", "COMPS"}
STABLE_SIX = {"BLiMP", "Supplement", "EWoK", "Entity", "COMPS"}  # Reading has no item records here.

EXISTING_PER_TARGETS: dict[tuple[str, str], pathlib.Path] = {
    ("seed43022_compact", "chck_20M"): WS / "data/legal20m_treatment_effect_eval/per_target/complianttok_reinvest_seed43022_20M.json",
    ("seed43022_compact", "chck_70M"): WS / "data/legal_mature_treatment_effect_eval/per_target/complianttok_reinvest_seed43022_70M.json",
    ("seed43022_compact", "chck_80M"): WS / "data/legal_mature_treatment_effect_eval/per_target/complianttok_reinvest_seed43022_80M.json",
    ("seed43022_compact", "chck_100M"): WS / "data/compliant_full_eval/per_target/complianttok_reinvest_seed43022.json",
    ("seed43022_repeat", "chck_80M"): WS / "data/architecture_interaction_selected_panel_final/full_repeat/chck_80M/eval/per_target/arch_full_repeat_chck_80M.json",
    ("seed43022_repeat", "chck_100M"): WS / "data/architecture_interaction_selected_panel_final/full_repeat/chck_100M/eval/per_target/arch_full_repeat_chck_100M.json",
}


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def resolve(path: str | pathlib.Path) -> pathlib.Path:
    p = pathlib.Path(path)
    return p if p.is_absolute() else USER_ROOT / p


def read_json(path: pathlib.Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_reader():
    spec = importlib.util.spec_from_file_location("selected_prediction_movement_reader", MOVEMENT_READER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {MOVEMENT_READER}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def per_target_path(ladder_out: pathlib.Path, arm: str, ck: str) -> pathlib.Path:
    existing = EXISTING_PER_TARGETS.get((arm, ck))
    if existing is not None and existing.exists():
        return existing
    return ladder_out / "eval" / "per_target" / f"ladder_{arm}_{ck}.json"


def wait_for_file(path: pathlib.Path, timeout_sec: int, poll_sec: int, log_path: pathlib.Path) -> dict[str, Any]:
    start = time.time()
    while True:
        exists = path.exists()
        rec = {"event": "file_wait_sample", "utc": now(), "path": str(path), "exists": exists}
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(json.dumps(rec), flush=True)
        if exists:
            return {"status": "file_ready", "waited_sec": round(time.time() - start, 1), "path": str(path)}
        if time.time() - start > timeout_sec:
            return {"status": "file_wait_timeout", "waited_sec": round(time.time() - start, 1), "path": str(path)}
        time.sleep(poll_sec)


def pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 3:
        return None
    mx = statistics.mean(xs)
    my = statistics.mean(ys)
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx <= 0 or vy <= 0:
        return None
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / math.sqrt(vx * vy)


def summarize_binary_delta(delta: dict[str, int], items: dict[str, dict[str, Any]]) -> dict[str, Any]:
    ids = list(delta)
    vals = [delta[i] for i in ids]
    n = len(vals)
    if not n:
        return {"n": 0}
    return {
        "n": n,
        "gain_count": sum(v > 0 for v in vals),
        "loss_count": sum(v < 0 for v in vals),
        "same_count": sum(v == 0 for v in vals),
        "net_correct_delta_count": sum(vals),
        "delta_item_acc_pp": 100.0 * sum(vals) / n,
        "flip_fraction_pct": 100.0 * sum(v != 0 for v in vals) / n,
        "by_column": by_column_delta(delta, items),
    }


def by_column_delta(delta: dict[str, int], items: dict[str, dict[str, Any]]) -> dict[str, Any]:
    groups: dict[str, list[int]] = defaultdict(list)
    for iid, d in delta.items():
        groups[str(items[iid]["column"])].append(int(d))
    out: dict[str, Any] = {}
    for col, vals in sorted(groups.items()):
        n = len(vals)
        out[col] = {
            "n": n,
            "gain_count": sum(v > 0 for v in vals),
            "loss_count": sum(v < 0 for v in vals),
            "net_correct_delta_count": sum(vals),
            "delta_item_acc_pp": 100.0 * sum(vals) / n if n else None,
            "flip_fraction_pct": 100.0 * sum(v != 0 for v in vals) / n if n else None,
        }
    return out


def directional_overlap(d0: dict[str, int], d1: dict[str, int], ids: list[str]) -> dict[str, Any]:
    both_nonzero = [(d0[i], d1[i]) for i in ids if d0[i] != 0 and d1[i] != 0]
    n = len(ids)
    if not both_nonzero:
        return {"n_common": n, "n_both_nonzero": 0}
    same = sum(1 for x, y in both_nonzero if (x > 0) == (y > 0))
    both_pos = sum(1 for x, y in both_nonzero if x > 0 and y > 0)
    both_neg = sum(1 for x, y in both_nonzero if x < 0 and y < 0)
    pos_neg = sum(1 for x, y in both_nonzero if x > 0 and y < 0)
    neg_pos = sum(1 for x, y in both_nonzero if x < 0 and y > 0)
    p0p = sum(1 for i in ids if d0[i] > 0) / n
    p0n = sum(1 for i in ids if d0[i] < 0) / n
    p1p = sum(1 for i in ids if d1[i] > 0) / n
    p1n = sum(1 for i in ids if d1[i] < 0) / n
    expected_same = (p0p * p1p + p0n * p1n) * n
    expected_both_nonzero = (p0p * p1p + p0n * p1n + p0p * p1n + p0n * p1p) * n
    return {
        "n_common": n,
        "n_both_nonzero": len(both_nonzero),
        "same_sign_count": same,
        "opposite_sign_count": len(both_nonzero) - same,
        "same_sign_rate_given_both_nonzero": same / len(both_nonzero),
        "both_pos": both_pos,
        "both_neg": both_neg,
        "pos_neg": pos_neg,
        "neg_pos": neg_pos,
        "expected_same_sign_count_independent_marginals": expected_same,
        "same_sign_enrichment_over_independent_marginals": same / expected_same if expected_same else None,
        "both_nonzero_enrichment_over_independent_marginals": len(both_nonzero) / expected_both_nonzero if expected_both_nonzero else None,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ladder-out", default=str(DEFAULT_LADDER_OUT))
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT))
    ap.add_argument("--checkpoints", nargs="*", default=CHECKPOINTS)
    ap.add_argument("--wait-for-summary", action="store_true")
    ap.add_argument("--wait-timeout-sec", type=int, default=64_800)
    ap.add_argument("--poll-sec", type=int, default=180)
    ap.add_argument("--plan-only", action="store_true")
    args = ap.parse_args()

    ladder_out = resolve(args.ladder_out)
    out_dir = resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "item_delta_wait_log.jsonl"
    summary_path = ladder_out / "full_deberta_seed_ladder_stable_summary.json"

    arms = ["seed43022_compact", "seed43022_repeat", "seed43122_compact", "seed43122_repeat"]
    paths = {f"{arm}:{ck}": str(per_target_path(ladder_out, arm, ck)) for arm in arms for ck in args.checkpoints}
    plan = {
        "status": "FULL_DEBERTA_SEED_LADDER_ITEM_DELTA_PLAN",
        "created_utc": now(),
        "decision_role": "Measure whether compact-minus-repeat binary item deltas recur across full-DeBERTa seeds along the common 10M ladder after stable-family selected evaluation exists.",
        "ladder_out": str(ladder_out),
        "summary_path": str(summary_path),
        "summary_exists": summary_path.exists(),
        "checkpoints": args.checkpoints,
        "per_target_paths": {k: {"path": v, "exists": pathlib.Path(v).exists()} for k, v in paths.items()},
        "boundary": "CPU/file-only item analysis; no model loading, evaluation, training, SuperGLUE, AoA, upload, or leaderboard action.",
    }
    (out_dir / "full_deberta_seed_ladder_item_delta_plan.json").write_text(json.dumps(plan, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": plan["status"], "out_dir": str(out_dir), "summary_exists": summary_path.exists(), "plan_json": str(out_dir / "full_deberta_seed_ladder_item_delta_plan.json")}, indent=2), flush=True)
    if args.plan_only:
        return
    if args.wait_for_summary:
        wr = wait_for_file(summary_path, args.wait_timeout_sec, args.poll_sec, log_path)
        (out_dir / "wait_for_ladder_summary_record.json").write_text(json.dumps(wr, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        if wr.get("status") != "file_ready":
            raise SystemExit(2)

    reader = load_reader()
    cell_items: dict[str, dict[str, dict[str, Any]]] = {}
    load_problems: list[str] = []
    for arm in arms:
        for ck in args.checkpoints:
            key = f"{arm}:{ck}"
            path = per_target_path(ladder_out, arm, ck)
            if not path.exists():
                load_problems.append(f"missing per-target {key}: {path}")
                continue
            try:
                record = reader.load_eval_record(path)
                items, meta = reader.load_items_from_record(record)
                warnings = meta.get("warnings") or []
                if warnings:
                    load_problems.append(f"warnings {key}: {warnings[:3]}")
                # Stable selected item records only; Reading has no item-level records in this reader.
                cell_items[key] = {r["item_id"]: r for r in items if r.get("column") in STABLE_FIVE}
            except Exception as exc:
                load_problems.append(f"load failed {key}: {exc!r}")
    if load_problems:
        (out_dir / "load_problems.json").write_text(json.dumps(load_problems, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(json.dumps({"event": "load_problems", "count": len(load_problems), "first": load_problems[:8]}, indent=2), flush=True)
        raise SystemExit(3)

    analysis_rows: list[dict[str, Any]] = []
    examples: dict[str, list[dict[str, Any]]] = {}
    for ck in args.checkpoints:
        keys = [f"{arm}:{ck}" for arm in arms]
        common = sorted(set.intersection(*(set(cell_items[k]) for k in keys)))
        d43022 = {iid: int(cell_items[f"seed43022_compact:{ck}"][iid]["correct"]) - int(cell_items[f"seed43022_repeat:{ck}"][iid]["correct"]) for iid in common}
        d43122 = {iid: int(cell_items[f"seed43122_compact:{ck}"][iid]["correct"]) - int(cell_items[f"seed43122_repeat:{ck}"][iid]["correct"]) for iid in common}
        corr = pearson([float(d43022[i]) for i in common], [float(d43122[i]) for i in common])
        correct_corrs: dict[str, Any] = {}
        for data_arm in ["compact", "repeat"]:
            k0 = f"seed43022_{data_arm}:{ck}"
            k1 = f"seed43122_{data_arm}:{ck}"
            correct_corrs[f"{data_arm}_correctness_cross_seed_r"] = pearson(
                [1.0 if cell_items[k0][i]["correct"] else 0.0 for i in common],
                [1.0 if cell_items[k1][i]["correct"] else 0.0 for i in common],
            )
        row = {
            "checkpoint": ck,
            "words": int(ck[5:-1]) * 1_000_000,
            "n_common_stable_items": len(common),
            "compact_minus_repeat_delta_cross_seed_pearson_r": corr,
            **correct_corrs,
            "seed43022_delta_summary": summarize_binary_delta(d43022, cell_items[f"seed43022_compact:{ck}"]),
            "seed43122_delta_summary": summarize_binary_delta(d43122, cell_items[f"seed43122_compact:{ck}"]),
            "directional_overlap": directional_overlap(d43022, d43122, common),
        }
        analysis_rows.append(row)
        # Preserve a few same-sign and opposite-sign examples for later scientific reading.
        ex: list[dict[str, Any]] = []
        for iid in common:
            if len(ex) >= 20:
                break
            if d43022[iid] != 0 and d43122[iid] != 0:
                item = cell_items[f"seed43022_compact:{ck}"][iid]
                ex.append({"item_id": iid, "column": item.get("column"), "subtask": item.get("subtask"), "seed43022_delta": d43022[iid], "seed43122_delta": d43122[iid], "target": item.get("target"), "meta": item.get("meta")})
        examples[ck] = ex

    csv_path = out_dir / "full_deberta_seed_ladder_item_delta_rows.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        fields = ["checkpoint", "words", "n_common_stable_items", "compact_minus_repeat_delta_cross_seed_pearson_r", "compact_correctness_cross_seed_r", "repeat_correctness_cross_seed_r", "seed43022_delta_item_acc_pp", "seed43122_delta_item_acc_pp", "seed43022_flip_fraction_pct", "seed43122_flip_fraction_pct", "same_sign_enrichment"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in analysis_rows:
            w.writerow({
                "checkpoint": r["checkpoint"],
                "words": r["words"],
                "n_common_stable_items": r["n_common_stable_items"],
                "compact_minus_repeat_delta_cross_seed_pearson_r": r["compact_minus_repeat_delta_cross_seed_pearson_r"],
                "compact_correctness_cross_seed_r": r["compact_correctness_cross_seed_r"],
                "repeat_correctness_cross_seed_r": r["repeat_correctness_cross_seed_r"],
                "seed43022_delta_item_acc_pp": r["seed43022_delta_summary"].get("delta_item_acc_pp"),
                "seed43122_delta_item_acc_pp": r["seed43122_delta_summary"].get("delta_item_acc_pp"),
                "seed43022_flip_fraction_pct": r["seed43022_delta_summary"].get("flip_fraction_pct"),
                "seed43122_flip_fraction_pct": r["seed43122_delta_summary"].get("flip_fraction_pct"),
                "same_sign_enrichment": r["directional_overlap"].get("same_sign_enrichment_over_independent_marginals"),
            })

    result = {
        "status": "FULL_DEBERTA_SEED_LADDER_ITEM_DELTA_DONE",
        "created_utc": now(),
        "ladder_out": str(ladder_out),
        "rows_csv": str(csv_path),
        "analysis_rows": analysis_rows,
        "examples_both_nonzero_first20_by_checkpoint": examples,
        "interpretation_boundary": "Binary item deltas are threshold crossings on stable selected items; use with aggregate ladder and EWoK continuous margins before making a learning-principle claim.",
        "no_model_loading_training_evaluation_superglue_aoa_upload_or_leaderboard": True,
    }
    out_json = out_dir / "full_deberta_seed_ladder_item_delta.json"
    out_json.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = ["# research full-DeBERTa seed ladder item-delta analysis", "", result["interpretation_boundary"], "", "## Per-checkpoint summary"]
    for r in analysis_rows:
        lines.append(f"- {r['checkpoint']}: delta_cross_seed_r={r['compact_minus_repeat_delta_cross_seed_pearson_r']}, compact_correct_r={r['compact_correctness_cross_seed_r']}, repeat_correct_r={r['repeat_correctness_cross_seed_r']}, seed43022_item_pp={r['seed43022_delta_summary'].get('delta_item_acc_pp')}, seed43122_item_pp={r['seed43122_delta_summary'].get('delta_item_acc_pp')}, same_sign_enrichment={r['directional_overlap'].get('same_sign_enrichment_over_independent_marginals')}")
    lines += ["", f"Rows CSV: `{csv_path}`", f"JSON: `{out_json}`"]
    (out_dir / "full_deberta_seed_ladder_item_delta.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "out_json": str(out_json), "rows_csv": str(csv_path)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
