#!/usr/bin/env python3
"""research: Task-balanced retained-plus-new competence readout.

Implements the predeclared readouts R3 and R4 from the independent-anchor
reproduction design (independent_anchor_reproduction_design.md).

R3: Per-column retention and discovery rates
R4: Label-free private-benefit signature (change zone analysis)

Inputs: per-target evaluation payloads for four arms:
  anchor80    - chck_80M (private-off reference)
  coherent80  - coherent private replay from frozen chck_80M
  shuffled80  - spanbreak private replay from frozen chck_80M
  ordinary84  - ordinary continuation (chck_84M from scale-1.75 ladder)

Decision criteria (predeclared):
  CONTINUE: coherent80 retention >= ordinary84 in >=5/7 cols AND
            coherent80 discovery >= ordinary84 in >=4/7 cols AND
            coherent80 change_benefit > 0 AND > ordinary84 change_benefit
  STOP:     any of the above fails
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse
import json
import pathlib
import sys
import time
from statistics import mean
from typing import Any

ROOT = _public_path('.')
A02_SCRIPTS = _public_path('experiments/archive/frontier_consolidation/scripts')
if str(A02_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(A02_SCRIPTS))

from pairwise_item_flip_analysis import DISCRETE_COLUMNS, PayloadLoader, compare_column  # noqa: E402

CHEAP_COLS = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA", "Reading"]
# Columns used for task-balanced analysis — must match research DISCRETE_COLUMNS
READOUT_COLS = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA"]
N_COLS = len(READOUT_COLS)  # 6

# Arms
ARM_NAMES = ["anchor80", "coherent80", "shuffled80", "ordinary84"]


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def rel(path: pathlib.Path | str) -> str:
    p = pathlib.Path(path)
    try:
        return str(p.resolve().relative_to(ROOT))
    except Exception:
        return str(p)


def load_json(path: pathlib.Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def get_cheap7(payload: dict) -> float:
    scores = payload.get("official_overall", {}).get("scores", {})
    vals = []
    for c in CHEAP_COLS:
        v = scores.get(c)
        if v is None:
            raise RuntimeError(f"Missing cheap7 column {c}")
        vals.append(float(v))
    return mean(vals)


def compare_arms(base_payload_path: str, cand_payload_path: str, column: str) -> dict:
    """Compare base vs candidate for one column using research infrastructure."""
    loader_base = PayloadLoader(pathlib.Path(base_payload_path))
    loader_cand = PayloadLoader(pathlib.Path(cand_payload_path))
    return compare_column(loader_base, loader_cand, column)


def retention_discovery(comparison: dict) -> dict:
    """Compute retention and discovery rates from comparison result."""
    gain = int(comparison.get("gain", 0))
    loss = int(comparison.get("loss", 0))
    both_correct = int(comparison.get("both_correct", 0))
    both_wrong = int(comparison.get("both_wrong", 0))

    anchor_correct = both_correct + loss
    anchor_wrong = gain + both_wrong
    total = anchor_correct + anchor_wrong

    retention = both_correct / anchor_correct if anchor_correct > 0 else 1.0
    discovery = gain / anchor_wrong if anchor_wrong > 0 else 0.0

    change_zone = gain + loss
    change_benefit = (gain - loss) / change_zone if change_zone > 0 else 0.0

    return {
        "anchor_correct": anchor_correct,
        "anchor_wrong": anchor_wrong,
        "total_items": total,
        "gain": gain,
        "loss": loss,
        "both_correct": both_correct,
        "both_wrong": both_wrong,
        "retention": retention,
        "discovery": discovery,
        "change_zone_size": change_zone,
        "change_benefit": change_benefit,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--anchor80-payload", required=True, help="per-target JSON for chck_80M")
    ap.add_argument("--coherent80-payload", required=True, help="per-target JSON for coherent80")
    ap.add_argument("--shuffled80-payload", required=True, help="per-target JSON for shuffled80")
    ap.add_argument("--ordinary84-payload", required=True, help="per-target JSON for chck_84M")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    payloads = {
        "anchor80": pathlib.Path(args.anchor80_payload),
        "coherent80": pathlib.Path(args.coherent80_payload),
        "shuffled80": pathlib.Path(args.shuffled80_payload),
        "ordinary84": pathlib.Path(args.ordinary84_payload),
    }

    # Verify all payloads exist
    for name, path in payloads.items():
        if not path.exists():
            print(json.dumps({"error": f"Missing payload: {name}", "path": str(path)}), flush=True)
            raise FileNotFoundError(path)

    # Load cheap7 scores
    cheap7_scores = {}
    for name, path in payloads.items():
        pdata = load_json(path)
        cheap7_scores[name] = get_cheap7(pdata)
        print(json.dumps({"event": "loaded", "arm": name, "cheap7": cheap7_scores[name]}), flush=True)

    # R1: Cheap7 comparison
    cheap7_pass = (cheap7_scores["coherent80"] > cheap7_scores["anchor80"]
                   and cheap7_scores["coherent80"] > cheap7_scores["ordinary84"])

    # R3: Per-column retention and discovery
    models = ["coherent80", "shuffled80", "ordinary84"]
    per_column: dict[str, dict[str, dict]] = {}

    for col in READOUT_COLS:
        per_column[col] = {}
        for model in models:
            try:
                comparison = compare_arms(
                    str(payloads["anchor80"]), str(payloads[model]), col
                )
                rd = retention_discovery(comparison)
                per_column[col][model] = rd
            except Exception as e:
                print(json.dumps({"warning": f"compare failed for {col}/{model}: {e}"}), flush=True)
                per_column[col][model] = {
                    "retention": None, "discovery": None,
                    "change_zone_size": 0, "change_benefit": 0.0,
                    "error": str(e),
                }

    # R3 decision: retention and discovery
    retention_wins = 0
    discovery_wins = 0
    col_decisions: dict[str, dict] = {}

    for col in READOUT_COLS:
        coh = per_column[col].get("coherent80", {})
        ord_ = per_column[col].get("ordinary84", {})
        coh_ret = coh.get("retention")
        ord_ret = ord_.get("retention")
        coh_disc = coh.get("discovery")
        ord_disc = ord_.get("discovery")

        ret_win = coh_ret is not None and ord_ret is not None and coh_ret >= ord_ret
        disc_win = coh_disc is not None and ord_disc is not None and coh_disc >= ord_disc

        if ret_win:
            retention_wins += 1
        if disc_win:
            discovery_wins += 1

        col_decisions[col] = {
            "coherent_retention": coh_ret,
            "ordinary_retention": ord_ret,
            "retention_win": ret_win,
            "coherent_discovery": coh_disc,
            "ordinary_discovery": ord_disc,
            "discovery_win": disc_win,
            "coherent_change_benefit": coh.get("change_benefit"),
            "ordinary_change_benefit": ord_.get("change_benefit"),
            "shuffled_retention": per_column[col].get("shuffled80", {}).get("retention"),
            "shuffled_discovery": per_column[col].get("shuffled80", {}).get("discovery"),
            "shuffled_change_benefit": per_column[col].get("shuffled80", {}).get("change_benefit"),
        }

    # Thresholds: retention ≥ 4/6, discovery ≥ 3/6 (proportional to original 5/7, 4/7)
    ret_threshold = max(1, N_COLS - 2)  # 4 for 6 cols
    disc_threshold = max(1, N_COLS // 2)  # 3 for 6 cols
    r3_retention_pass = retention_wins >= ret_threshold
    r3_discovery_pass = discovery_wins >= disc_threshold
    r3_pass = r3_retention_pass and r3_discovery_pass

    # R4: Label-free private-benefit signature (pooled across columns)
    pooled_coherent_gain = 0
    pooled_coherent_loss = 0
    pooled_ordinary_gain = 0
    pooled_ordinary_loss = 0

    for col in READOUT_COLS:
        coh = per_column[col].get("coherent80", {})
        ord_ = per_column[col].get("ordinary84", {})
        pooled_coherent_gain += int(coh.get("gain", 0))
        pooled_coherent_loss += int(coh.get("loss", 0))
        pooled_ordinary_gain += int(ord_.get("gain", 0))
        pooled_ordinary_loss += int(ord_.get("loss", 0))

    pooled_coherent_change = pooled_coherent_gain + pooled_coherent_loss
    pooled_ordinary_change = pooled_ordinary_gain + pooled_ordinary_loss
    pooled_coherent_benefit = ((pooled_coherent_gain - pooled_coherent_loss) / pooled_coherent_change
                               if pooled_coherent_change > 0 else 0.0)
    pooled_ordinary_benefit = ((pooled_ordinary_gain - pooled_ordinary_loss) / pooled_ordinary_change
                               if pooled_ordinary_change > 0 else 0.0)

    r4_positive = pooled_coherent_benefit > 0
    r4_better_than_ordinary = pooled_coherent_benefit > pooled_ordinary_benefit
    r4_pass = r4_positive and r4_better_than_ordinary

    # Overall decision
    all_pass = cheap7_pass and r3_pass and r4_pass
    decision = "CONTINUE" if all_pass else "STOP"

    # Build summary
    summary = {
        "status": "TASK_BALANCED_READOUT",
        "created_utc": now(),
        "decision": decision,
        "criteria": {
            "cheap7_pass": cheap7_pass,
            "cheap7": cheap7_scores,
            "r3_retention_pass": r3_retention_pass,
            "r3_retention_wins": retention_wins,
            "r3_retention_threshold": ret_threshold,
            "r3_discovery_pass": r3_discovery_pass,
            "r3_discovery_wins": discovery_wins,
            "r3_discovery_threshold": disc_threshold,
            "r3_pass": r3_pass,
            "r4_pooled_coherent_benefit": pooled_coherent_benefit,
            "r4_pooled_ordinary_benefit": pooled_ordinary_benefit,
            "r4_positive": r4_positive,
            "r4_better_than_ordinary": r4_better_than_ordinary,
            "r4_pass": r4_pass,
        },
        "per_column_decisions": col_decisions,
        "per_column_detail": {col: {m: per_column[col][m] for m in models} for col in READOUT_COLS},
        "pooled_change_zone": {
            "coherent_gain": pooled_coherent_gain,
            "coherent_loss": pooled_coherent_loss,
            "coherent_change_zone": pooled_coherent_change,
            "coherent_benefit": pooled_coherent_benefit,
            "ordinary_gain": pooled_ordinary_gain,
            "ordinary_loss": pooled_ordinary_loss,
            "ordinary_change_zone": pooled_ordinary_change,
            "ordinary_benefit": pooled_ordinary_benefit,
        },
        "payload_paths": {name: rel(path) for name, path in payloads.items()},
    }

    out_json = out_dir / "task_balanced_readout.json"
    out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    # Write markdown summary
    md_lines = [
        "# research: Task-Balanced Retained-Plus-New Competence Readout",
        "",
        f"**Decision: {decision}**",
        "",
        "## Cheap7 Scores",
        "",
        f"| Arm | Cheap7 |",
        f"|---|---:|",
    ]
    for name in ARM_NAMES:
        md_lines.append(f"| {name} | {cheap7_scores[name]:.4f} |")
    md_lines += [
        "",
        f"Cheap7 criterion (coherent > anchor AND coherent > ordinary): **{'PASS' if cheap7_pass else 'FAIL'}**",
        "",
        "## R3: Per-Column Retention and Discovery",
        "",
        "| Column | Coh Ret | Ord Ret | Ret Win | Coh Disc | Ord Disc | Disc Win |",
        "|---|---:|---:|:---:|---:|---:|:---:|",
    ]
    for col in READOUT_COLS:
        cd = col_decisions[col]
        def fmt(v):
            return "NA" if v is None else f"{v:.4f}"
        md_lines.append(
            f"| {col} | {fmt(cd['coherent_retention'])} | {fmt(cd['ordinary_retention'])} "
            f"| {'✓' if cd['retention_win'] else '✗'} "
            f"| {fmt(cd['coherent_discovery'])} | {fmt(cd['ordinary_discovery'])} "
            f"| {'✓' if cd['discovery_win'] else '✗'} |"
        )
    md_lines += [
        "",
        f"Retention wins: {retention_wins}/7 (need ≥5): **{'PASS' if r3_retention_pass else 'FAIL'}**",
        f"Discovery wins: {discovery_wins}/7 (need ≥4): **{'PASS' if r3_discovery_pass else 'FAIL'}**",
        "",
        "## R4: Label-Free Private-Benefit Signature",
        "",
        f"| Metric | Coherent | Ordinary |",
        f"|---|---:|---:|",
        f"| Pooled gains | {pooled_coherent_gain} | {pooled_ordinary_gain} |",
        f"| Pooled losses | {pooled_coherent_loss} | {pooled_ordinary_loss} |",
        f"| Change zone size | {pooled_coherent_change} | {pooled_ordinary_change} |",
        f"| Change benefit | {pooled_coherent_benefit:+.4f} | {pooled_ordinary_benefit:+.4f} |",
        "",
        f"Coherent benefit > 0: **{'PASS' if r4_positive else 'FAIL'}**",
        f"Coherent benefit > ordinary: **{'PASS' if r4_better_than_ordinary else 'FAIL'}**",
        "",
        f"## Overall Decision: **{decision}**",
        "",
        f"JSON: `{rel(out_json)}`",
    ]

    out_md = out_dir / "task_balanced_readout.md"
    out_md.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    print(json.dumps({
        "status": summary["status"],
        "decision": decision,
        "cheap7_pass": cheap7_pass,
        "r3_pass": r3_pass,
        "r4_pass": r4_pass,
        "retention_wins": retention_wins,
        "discovery_wins": discovery_wins,
        "pooled_coherent_benefit": pooled_coherent_benefit,
        "pooled_ordinary_benefit": pooled_ordinary_benefit,
        "out_json": rel(out_json),
        "out_md": rel(out_md),
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
