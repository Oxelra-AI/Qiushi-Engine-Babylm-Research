#!/usr/bin/env python3
"""Summarize research RoBERTa relation-structure probes.

Scientific purpose: the same held-out copy, compact-rewrite conditioning, and
Entity cue-ablation instruments from research are applied to research RoBERTa
VIEW/REPEAT/CLEAN arms.  This script extracts the ordering, checkpoint
consistency, and decomposed NLL terms so the result can be judged as a test of
architecture generality rather than as another aggregate benchmark score.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import csv
import json
import math
import pathlib
import statistics
import time
from collections import defaultdict
from typing import Any


def find_root() -> pathlib.Path:
    return _PUBLIC_ROOT


ROOT = find_root()
WS = ROOT / "experiments/archive" / 'relation_learning'
IN = WS / "data" / "roberta_probe_dynamic"
OUT = WS / "data" / "roberta_probe_summary"
NOTE = WS / "notes" / "roberta_probe_result.md"


def rel(p: pathlib.Path) -> str:
    try:
        return str(p.relative_to(ROOT))
    except Exception:
        return str(p)


def read_csv(path: pathlib.Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def f(x: Any) -> float:
    try:
        y = float(x)
        return y if math.isfinite(y) else float("nan")
    except Exception:
        return float("nan")


def mean(xs: list[float]) -> float:
    xs = [x for x in xs if math.isfinite(x)]
    return statistics.mean(xs) if xs else float("nan")


def sd(xs: list[float]) -> float:
    xs = [x for x in xs if math.isfinite(x)]
    return statistics.pstdev(xs) if len(xs) > 1 else 0.0


def rows_filter(rows: list[dict[str, str]], **conds: str) -> list[dict[str, str]]:
    return [r for r in rows if all(str(r.get(k)) == str(v) for k, v in conds.items())]


def sign_consistency(rows: list[dict[str, str]], key: str, positive: bool) -> dict[str, Any]:
    vals = [(r.get("checkpoint"), f(r.get(key))) for r in rows]
    ok = [(ck, v) for ck, v in vals if math.isfinite(v) and ((v > 0) if positive else (v < 0))]
    return {
        "n_checkpoints": len([v for _, v in vals if math.isfinite(v)]),
        "n_consistent": len(ok),
        "values": [{"checkpoint": ck, "value": v} for ck, v in vals],
        "mean": mean([v for _, v in vals]),
        "sd": sd([v for _, v in vals]),
    }


def summarize_gain(summary_rows: list[dict[str, str]], late_rows: list[dict[str, str]], family: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for group in sorted({r.get("group", "") for r in summary_rows if r.get("group") in {"ALL", "span_1", "span_4", "token_nonoverlap", "token_overlap"}}):
        out[group] = {}
        per_arm = rows_filter(summary_rows, group=group)
        for role in ["R", "V", "C"]:
            arm_rows = [r for r in per_arm if r.get("role") == role]
            out[group][f"role_{role}"] = {
                "mean_gain_60_100": mean([f(r.get("mean_gain")) for r in arm_rows]),
                "mean_a_nll_60_100": mean([f(r.get("mean_a_nll")) for r in arm_rows]),
                "mean_b_nll_60_100": mean([f(r.get("mean_b_nll")) for r in arm_rows]),
                "n_rows": int(arm_rows[0].get("n", 0)) if arm_rows else 0,
            }
        contrast_roles = {"RminusC": ("R", "C"), "RminusV": ("R", "V"), "VminusC": ("V", "C"), "VminusR": ("V", "R"), "CminusR": ("C", "R")}
        for contrast, (ra, rb) in contrast_roles.items():
            con_rows = rows_filter(read_csv(IN / f"{family}_contrasts.csv"), group=group, contrast=contrast)
            late = rows_filter(late_rows, group=group, contrast=contrast)
            gain_a_vals = [f(r.get("mean_gain_a")) for r in con_rows]
            gain_b_vals = [f(r.get("mean_gain_b")) for r in con_rows]
            role_a = out[group].get(f"role_{ra}", {})
            role_b = out[group].get(f"role_{rb}", {})
            cue_a = f(role_a.get("mean_a_nll_60_100"))
            cue_b = f(role_b.get("mean_a_nll_60_100"))
            ctrl_a = f(role_a.get("mean_b_nll_60_100"))
            ctrl_b = f(role_b.get("mean_b_nll_60_100"))
            out[group][contrast] = {
                "late_mean_delta_gain": f(late[0].get("late_mean_delta_mean_gain_a_minus_b")) if late else float("nan"),
                "late_mean_gain_a": mean(gain_a_vals),
                "late_mean_gain_b": mean(gain_b_vals),
                "late_mean_cued_or_intact_nll_a": cue_a,
                "late_mean_cued_or_intact_nll_b": cue_b,
                "late_delta_cued_or_intact_nll_a_minus_b": cue_a - cue_b if math.isfinite(cue_a) and math.isfinite(cue_b) else float("nan"),
                "late_mean_control_nll_a": ctrl_a,
                "late_mean_control_nll_b": ctrl_b,
                "late_delta_control_nll_a_minus_b": ctrl_a - ctrl_b if math.isfinite(ctrl_a) and math.isfinite(ctrl_b) else float("nan"),
                "checkpoint_consistency_positive": sign_consistency(con_rows, "delta_mean_gain_a_minus_b", True),
                "checkpoint_consistency_negative": sign_consistency(con_rows, "delta_mean_gain_a_minus_b", False),
            }
    return out


def summarize_entity() -> dict[str, Any]:
    summary_rows = read_csv(IN / "entity_ablation_summary.csv")
    con_rows = read_csv(IN / "entity_ablation_contrasts.csv")
    late_rows = read_csv(IN / "entity_ablation_late_contrasts.csv")
    out: dict[str, Any] = {}
    for group in ["ALL", "rel_ge2", "rel_ge3", "rel_updates_2", "rel_updates_3", "rel_updates_4", "rel_updates_5"]:
        out[group] = {}
        per_arm = rows_filter(summary_rows, group=group)
        for role in ["V", "C", "R"]:
            arm_rows = [r for r in per_arm if r.get("role") == role]
            out[group][f"role_{role}"] = {
                "mean_margin_full_60_100": mean([f(r.get("mean_margin_full")) for r in arm_rows]),
                "mean_effect_no_initial_60_100": mean([f(r.get("mean_effect_no_initial_qbox")) for r in arm_rows]),
                "mean_effect_no_last_60_100": mean([f(r.get("mean_effect_no_last_relevant_update")) for r in arm_rows]),
                "n_items": int(arm_rows[0].get("n", 0)) if arm_rows else 0,
            }
        for contrast in ["VminusC", "VminusR", "RminusC", "RminusV"]:
            cks = rows_filter(con_rows, group=group, contrast=contrast)
            late = rows_filter(late_rows, group=group, contrast=contrast)
            out[group][contrast] = {
                "late_delta_margin_full": f(late[0].get("late_mean_delta_mean_margin_full_a_minus_b")) if late else float("nan"),
                "late_delta_no_initial_effect": f(late[0].get("late_mean_delta_mean_effect_no_initial_qbox_a_minus_b")) if late else float("nan"),
                "late_delta_no_last_effect": f(late[0].get("late_mean_delta_mean_effect_no_last_relevant_update_a_minus_b")) if late else float("nan"),
                "margin_consistency_positive": sign_consistency(cks, "delta_mean_margin_full_a_minus_b", True),
                "no_last_consistency_negative": sign_consistency(cks, "delta_mean_effect_no_last_relevant_update_a_minus_b", False),
            }
    return out


def fmt(x: Any, nd: int = 4) -> str:
    y = f(x)
    if not math.isfinite(y):
        return "NA"
    return f"{y:+.{nd}f}"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    copy_summary = read_csv(IN / "copy_summary.csv")
    rewrite_summary = read_csv(IN / "rewrite_summary.csv")
    copy_late = read_csv(IN / "copy_late_contrasts.csv")
    rewrite_late = read_csv(IN / "rewrite_late_contrasts.csv")
    result = {
        "status": "ROBERTA_PROBE_SUMMARY",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "input_dir": rel(IN),
        "copy": summarize_gain(copy_summary, copy_late, "copy"),
        "rewrite": summarize_gain(rewrite_summary, rewrite_late, "rewrite"),
        "entity_ablation": summarize_entity(),
        "interpretation_short": "RoBERTa seed43022 does not replicate the DeBERTa exact-copy ordering: held-out copy gain is slightly R<V/C over 60M-100M. It does replicate the content-conditioning ordering V>C>R on compact rewrites, including nonoverlap tokens, though V-C is small. Entity cue-ablation tendencies are weak and do not reproduce the DeBERTa stale/update pattern as a strong behavioral mechanism.",
    }
    out_json = OUT / "roberta_probe_summary.json"
    out_json.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    lines: list[str] = []
    lines.append("# research RoBERTa relation-structure probe result")
    lines.append("")
    lines.append(f"Source probe outputs: `{rel(IN)}`. Summary JSON: `{rel(out_json)}`.")
    lines.append("")
    lines.append("## Scientific reading")
    lines.append("")
    lines.append("The same checkpoint probes used for DeBERTa seeds43022/43122 were run on the research RoBERTa seed43022 VIEW/REPEAT/CLEAN arms at 60M, 70M, 80M, 90M, and 100M. This is an architecture/learner-coordinate test, not a new leaderboard run.")
    lines.append("")
    lines.append("RoBERTa does **not** reproduce the DeBERTa held-out natural copy ordering. Across 60M-100M, copy-gain contrasts on 2,000 held-out natural source-repeat pairs are RminusC " + fmt(result["copy"]["ALL"]["RminusC"]["late_mean_delta_gain"]) + ", RminusV " + fmt(result["copy"]["ALL"]["RminusV"]["late_mean_delta_gain"]) + ", and VminusC " + fmt(result["copy"]["ALL"]["VminusC"]["late_mean_delta_gain"]) + ". Thus the RoBERTa learner has no visible exact-recurrence advantage on this probe; if anything, REPEAT is slightly below the other arms. This means the exact-copy tendency found in DeBERTa is not architecture-independent at this checkpoint/loss regime.")
    lines.append("")
    non = result["rewrite"]["token_nonoverlap"]
    lines.append("RoBERTa **does** reproduce the content-conditioning ordering on compact held-out rewrites. On all rewrite content tokens, VminusC is " + fmt(result["rewrite"]["ALL"]["VminusC"]["late_mean_delta_gain"]) + " and VminusR is " + fmt(result["rewrite"]["ALL"]["VminusR"]["late_mean_delta_gain"]) + "; on tokenizer-nonoverlap tokens, VminusC is " + fmt(non["VminusC"]["late_mean_delta_gain"]) + " and VminusR is " + fmt(non["VminusR"]["late_mean_delta_gain"]) + ". CminusR is also positive, especially nonoverlap " + fmt(non["CminusR"]["late_mean_delta_gain"]) + ". For nonoverlap tokens, REPEAT has worse true-source NLL than CLEAN by " + fmt(non["RminusC"]["late_delta_cued_or_intact_nll_a_minus_b"]) + " nats/token and worse unrelated-source NLL by " + fmt(non["RminusC"]["late_delta_control_nll_a_minus_b"]) + "; the lower gain is not created by an unusually easy unrelated-source denominator. VIEW has better true-source NLL than CLEAN by " + fmt(non["VminusC"]["late_delta_cued_or_intact_nll_a_minus_b"]) + " while also having better unrelated-source NLL by " + fmt(non["VminusC"]["late_delta_control_nll_a_minus_b"]) + ", so the small V-C gain is the residual benefit after a broad rewrite-register fit advantage.")
    lines.append("")
    lines.append("The Entity stale/update cue-ablation signals are small in RoBERTa and should not be read as the same behavioral mechanism as DeBERTa. On the stale-foil update subset, full gold-over-stale margin VminusR is " + fmt(result["entity_ablation"]["ALL"]["VminusR"]["late_delta_margin_full"]) + " while VminusC is " + fmt(result["entity_ablation"]["ALL"]["VminusC"]["late_delta_margin_full"]) + ". The no-last-update effect does not reproduce the DeBERTa sign: VminusR is " + fmt(result["entity_ablation"]["ALL"]["VminusR"]["late_delta_no_last_effect"]) + " rather than negative. This weakens any claim that RoBERTa already expresses the Entity state-update behavior, even though the content-conditioning probe separates the training arms.")
    lines.append("")
    lines.append("## Consequence for the data-efficient learning principle")
    lines.append("")
    lines.append("The result improves the principle by forcing the learner coordinate back into the account. Fixed-budget relation structure is not converted into identical computations by every architecture/training phase. Nonidentical restatement produces a content-conditioning tendency in both DeBERTa and RoBERTa, but exact recurrence becoming a transferable natural-copy advantage appears stronger in DeBERTa than in this RoBERTa run. The principle should therefore be stated as an interaction between experience relation and learner coordinate/training phase, not as an architecture-free law. The next behavioral evidence must decide whether the DeBERTa seed43222 Entity crossover survives and whether a natural re-mention probe can show the copy/content split outside compact rewrites and boxes.")
    NOTE.parent.mkdir(parents=True, exist_ok=True)
    NOTE.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "summary_json": rel(out_json), "note": rel(NOTE)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
