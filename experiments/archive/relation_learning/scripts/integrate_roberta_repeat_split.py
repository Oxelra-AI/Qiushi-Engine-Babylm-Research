#!/usr/bin/env python3
"""research: integrate RoBERTa REPEAT_SPLIT probe rows against prior RoBERTa C/R/V.

The research scorer evaluated only the delivered RBT_RS_43022 arm, so its native
contrast files are empty.  This script combines those raw rows with research
RoBERTa C/R/V rows and recomputes late 60M..100M contrasts for copy, compact
rewrite, and Entity cue-ablation probes.
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

ROOT = _public_path('experiments/archive/relation_learning/scripts/integrate_roberta_repeat_split.py')
ROOT = _PUBLIC_ROOT

WS = _public_path('experiments/archive/relation_learning')
OLD = _public_path('experiments/archive/relation_learning/data/roberta_probe_dynamic')
RS = _public_path('experiments/archive/relation_learning/data/roberta_repeat_split_probe_dynamic')
OUT = _public_path('experiments/archive/relation_learning/data/roberta_repeat_split_integration')
NOTE = _public_path('research/notes/relation_learning/roberta_repeat_split_integration.md')
CKS = ["chck_60M", "chck_70M", "chck_80M", "chck_90M", "chck_100M"]
ROLE = {"RBT_C_43022": "C", "RBT_R_43022": "R", "RBT_V_43022": "V", "RBT_RS_43022": "RS"}


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def rel(p: pathlib.Path) -> str:
    try:
        return str(p.relative_to(ROOT))
    except Exception:
        return str(p)


def read_csv(path: pathlib.Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("\n", encoding="utf-8")
        return
    keys: list[str] = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader(); w.writerows(rows)


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


def se(xs: list[float]) -> float:
    xs = [x for x in xs if math.isfinite(x)]
    return (statistics.pstdev(xs) / math.sqrt(len(xs))) if len(xs) > 1 else 0.0


def attach_role(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    out = []
    for r in rows:
        rr: dict[str, Any] = dict(r)
        rr["role"] = ROLE.get(str(r.get("arm")), str(r.get("role") or r.get("arm")))
        out.append(rr)
    return out


def load_rows(name: str) -> list[dict[str, Any]]:
    old = attach_role(read_csv(OLD / name))
    rs = attach_role(read_csv(RS / name))
    return old + rs


def summarize_copy(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    d: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        if str(r.get("checkpoint")) not in CKS:
            continue
        role = str(r.get("role")); ck = str(r.get("checkpoint")); group = str(r.get("span_group") or r.get("group") or "ALL")
        d[(role, ck, "ALL")].append(r)
        d[(role, ck, group)].append(r)
    out = []
    for (role, ck, group), xs in sorted(d.items()):
        out.append({
            "family": "copy", "role": role, "checkpoint": ck, "group": group, "n": len(xs),
            "mean_gain": mean([f(x.get("gain")) for x in xs]),
            "mean_intact_nll": mean([f(x.get("repeated_nll") or x.get("a_nll")) for x in xs]),
            "mean_control_nll": mean([f(x.get("unrepeated_nll") or x.get("b_nll")) for x in xs]),
        })
    return out


def summarize_rewrite(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    d: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        if str(r.get("checkpoint")) not in CKS:
            continue
        role = str(r.get("role")); ck = str(r.get("checkpoint")); tc = str(r.get("token_class") or "ALL")
        d[(role, ck, "ALL")].append(r)
        d[(role, ck, f"token_{tc}")].append(r)
    out = []
    for (role, ck, group), xs in sorted(d.items()):
        out.append({
            "family": "rewrite", "role": role, "checkpoint": ck, "group": group, "n": len(xs),
            "mean_gain": mean([f(x.get("gain")) for x in xs]),
            "mean_true_source_nll": mean([f(x.get("true_source_nll")) for x in xs]),
            "mean_unrelated_source_nll": mean([f(x.get("unrelated_source_nll")) for x in xs]),
        })
    return out


def summarize_entity(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    def groups_for(r: dict[str, Any]) -> list[str]:
        rel = int(float(r.get("relevant_updates") or r.get("num_relevant_updates") or 0))
        gs = ["ALL"]
        gs.append(f"rel_updates_{rel}")
        if rel >= 2: gs.append("rel_ge2")
        if rel >= 3: gs.append("rel_ge3")
        return gs
    d: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        if str(r.get("checkpoint")) not in CKS:
            continue
        role = str(r.get("role")); ck = str(r.get("checkpoint"))
        for g in groups_for(r):
            d[(role, ck, g)].append(r)
    out = []
    for (role, ck, group), xs in sorted(d.items()):
        out.append({
            "family": "entity", "role": role, "checkpoint": ck, "group": group, "n": len(xs),
            "mean_margin_full": mean([f(x.get("margin_full")) for x in xs]),
            "mean_effect_no_initial_qbox": mean([f(x.get("effect_no_initial_qbox")) for x in xs]),
            "mean_effect_no_last_relevant_update": mean([f(x.get("effect_no_last_relevant_update")) for x in xs]),
            "mean_effect_no_all_relevant_updates": mean([f(x.get("effect_no_all_relevant_updates")) for x in xs]),
        })
    return out


def by_key(rows: list[dict[str, Any]], family: str) -> dict[tuple[str, str, str], dict[str, Any]]:
    return {(str(r["role"]), str(r["checkpoint"]), str(r["group"])): r for r in rows if r.get("family") == family}


def contrast_summary(summary: list[dict[str, Any]], family: str, keys: list[str], contrasts: list[tuple[str, str, str]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    idx = by_key(summary, family)
    groups = sorted({g for (_, _, g) in idx})
    checkpoint_rows: list[dict[str, Any]] = []
    late_rows: list[dict[str, Any]] = []
    for group in groups:
        for cname, ra, rb in contrasts:
            per_ck: list[dict[str, Any]] = []
            for ck in CKS:
                a = idx.get((ra, ck, group)); b = idx.get((rb, ck, group))
                if not a or not b:
                    continue
                row: dict[str, Any] = {"family": family, "group": group, "contrast": cname, "checkpoint": ck, "role_a": ra, "role_b": rb, "n_a": a.get("n"), "n_b": b.get("n")}
                for k in keys:
                    row[f"mean_{k}_a"] = a.get(k); row[f"mean_{k}_b"] = b.get(k)
                    row[f"delta_{k}_a_minus_b"] = f(a.get(k)) - f(b.get(k))
                checkpoint_rows.append(row); per_ck.append(row)
            if per_ck:
                lrow: dict[str, Any] = {"family": family, "group": group, "contrast": cname, "role_a": ra, "role_b": rb, "n_checkpoints": len(per_ck), "n_a": per_ck[-1].get("n_a"), "n_b": per_ck[-1].get("n_b")}
                for k in keys:
                    vals = [f(r.get(f"delta_{k}_a_minus_b")) for r in per_ck]
                    lrow[f"late_mean_delta_{k}_a_minus_b"] = mean(vals)
                    lrow[f"late_sd_delta_{k}_a_minus_b"] = sd(vals)
                    lrow[f"positive_checkpoints_{k}"] = sum(1 for v in vals if v > 0)
                    lrow[f"negative_checkpoints_{k}"] = sum(1 for v in vals if v < 0)
                late_rows.append(lrow)
    return checkpoint_rows, late_rows


def fmt(x: Any, nd: int = 4, sign: bool = True) -> str:
    y = f(x)
    if not math.isfinite(y):
        return "NA"
    return f"{y:{'+' if sign else ''}.{nd}f}"


def get(late: list[dict[str, Any]], family: str, group: str, contrast: str, metric: str) -> float:
    key = f"late_mean_delta_{metric}_a_minus_b"
    for r in late:
        if r.get("family") == family and r.get("group") == group and r.get("contrast") == contrast:
            return f(r.get(key))
    return float("nan")


def getrow(late: list[dict[str, Any]], family: str, group: str, contrast: str) -> dict[str, Any]:
    for r in late:
        if r.get("family") == family and r.get("group") == group and r.get("contrast") == contrast:
            return r
    return {}


def main() -> None:
    for p in [_public_path('experiments/archive/relation_learning/data/roberta_probe_dynamic/copy_pair_rows.csv'), _public_path('experiments/archive/relation_learning/data/roberta_probe_dynamic/rewrite_pair_rows.csv'), _public_path('experiments/archive/relation_learning/data/roberta_probe_dynamic/entity_ablation_rows.csv'), _public_path('experiments/archive/relation_learning/data/roberta_repeat_split_probe_dynamic/copy_pair_rows.csv'), _public_path('experiments/archive/relation_learning/data/roberta_repeat_split_probe_dynamic/rewrite_pair_rows.csv'), _public_path('experiments/archive/relation_learning/data/roberta_repeat_split_probe_dynamic/entity_ablation_rows.csv')]:
        if not p.exists():
            raise FileNotFoundError(p)
    OUT.mkdir(parents=True, exist_ok=True)
    copy_rows = load_rows("copy_pair_rows.csv")
    rewrite_rows = load_rows("rewrite_pair_rows.csv")
    entity_rows = load_rows("entity_ablation_rows.csv")
    copy_summary = summarize_copy(copy_rows)
    rewrite_summary = summarize_rewrite(rewrite_rows)
    entity_summary = summarize_entity(entity_rows)
    all_summary = copy_summary + rewrite_summary + entity_summary
    copy_ck, copy_late = contrast_summary(copy_summary, "copy", ["mean_gain", "mean_intact_nll", "mean_control_nll"], [
        ("RminusC", "R", "C"), ("RSminusC", "RS", "C"), ("RminusRS", "R", "RS"), ("RSminusR", "RS", "R"), ("VminusC", "V", "C"), ("RSminusV", "RS", "V")])
    rewrite_ck, rewrite_late = contrast_summary(rewrite_summary, "rewrite", ["mean_gain", "mean_true_source_nll", "mean_unrelated_source_nll"], [
        ("RminusC", "R", "C"), ("RSminusC", "RS", "C"), ("RminusRS", "R", "RS"), ("RSminusR", "RS", "R"), ("VminusC", "V", "C"), ("VminusR", "V", "R"), ("CminusR", "C", "R"), ("RSminusV", "RS", "V")])
    ent_ck, ent_late = contrast_summary(entity_summary, "entity", ["mean_margin_full", "mean_effect_no_initial_qbox", "mean_effect_no_last_relevant_update", "mean_effect_no_all_relevant_updates"], [
        ("RminusC", "R", "C"), ("RSminusC", "RS", "C"), ("RminusRS", "R", "RS"), ("RSminusR", "RS", "R"), ("VminusC", "V", "C"), ("RSminusV", "RS", "V")])
    late = copy_late + rewrite_late + ent_late
    write_csv(_public_path('experiments/archive/relation_learning/data/roberta_repeat_split_integration/roberta_roles_summary.csv'), all_summary)
    write_csv(_public_path('experiments/archive/relation_learning/data/roberta_repeat_split_integration/roberta_repeat_split_checkpoint_contrasts.csv'), copy_ck + rewrite_ck + ent_ck)
    write_csv(_public_path('experiments/archive/relation_learning/data/roberta_repeat_split_integration/roberta_repeat_split_late_contrasts.csv'), late)

    # Compact source-use polarity compatible with research: A_T = U? no neutral N
    # was scored for RoBERTa. We therefore report gain U-T and decompose into
    # true-source and unrelated-source NLL deltas.
    non_r_c = getrow(late, "rewrite", "token_nonoverlap", "RminusC")
    non_rs_c = getrow(late, "rewrite", "token_nonoverlap", "RSminusC")
    non_r_rs = getrow(late, "rewrite", "token_nonoverlap", "RminusRS")
    copy_r_c = getrow(late, "copy", "ALL", "RminusC")
    copy_rs_c = getrow(late, "copy", "ALL", "RSminusC")
    copy_r_rs = getrow(late, "copy", "ALL", "RminusRS")
    ent_r_c = getrow(late, "entity", "rel_ge3", "RminusC")
    ent_rs_c = getrow(late, "entity", "rel_ge3", "RSminusC")

    result = {
        "status": "ROBERTA_REPEAT_SPLIT_INTEGRATION_DONE",
        "created_utc": now(),
        "input_old_roberta": rel(OLD),
        "input_repeat_split": rel(RS),
        "outputs": {"late_contrasts": rel(_public_path('experiments/archive/relation_learning/data/roberta_repeat_split_integration/roberta_repeat_split_late_contrasts.csv')), "summary": rel(_public_path('experiments/archive/relation_learning/data/roberta_repeat_split_integration/roberta_roles_summary.csv')), "note": rel(NOTE)},
        "key_numbers": {
            "copy_RminusC_gain": get(copy_late, "copy", "ALL", "RminusC", "mean_gain"),
            "copy_RSminusC_gain": get(copy_late, "copy", "ALL", "RSminusC", "mean_gain"),
            "copy_RminusRS_gain": get(copy_late, "copy", "ALL", "RminusRS", "mean_gain"),
            "rewrite_nonoverlap_RminusC_gain": get(rewrite_late, "rewrite", "token_nonoverlap", "RminusC", "mean_gain"),
            "rewrite_nonoverlap_RSminusC_gain": get(rewrite_late, "rewrite", "token_nonoverlap", "RSminusC", "mean_gain"),
            "rewrite_nonoverlap_RminusRS_gain": get(rewrite_late, "rewrite", "token_nonoverlap", "RminusRS", "mean_gain"),
            "rewrite_nonoverlap_RminusC_true_source_nll": get(rewrite_late, "rewrite", "token_nonoverlap", "RminusC", "mean_true_source_nll"),
            "rewrite_nonoverlap_RSminusC_true_source_nll": get(rewrite_late, "rewrite", "token_nonoverlap", "RSminusC", "mean_true_source_nll"),
            "rewrite_nonoverlap_RminusC_unrelated_source_nll": get(rewrite_late, "rewrite", "token_nonoverlap", "RminusC", "mean_unrelated_source_nll"),
            "rewrite_nonoverlap_RSminusC_unrelated_source_nll": get(rewrite_late, "rewrite", "token_nonoverlap", "RSminusC", "mean_unrelated_source_nll"),
        },
    }
    (_public_path('experiments/archive/relation_learning/data/roberta_repeat_split_integration/roberta_repeat_split_integration_result.json')).write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    lines: list[str] = []
    lines.append("# research RoBERTa REPEAT_SPLIT integration")
    lines.append("")
    lines.append("This note integrates the delivered RoBERTa `REPEAT_SPLIT` seed43022 run with the prior RoBERTa CLEAN/REPEAT/VIEW research probe rows. It is an architecture/learner-coordinate extent test, not a new BabyLM leaderboard run.")
    lines.append("")
    lines.append("## Main numbers")
    lines.append("")
    lines.append("Late means average checkpoints 60M,70M,80M,90M,100M to match the prior RoBERTa readout. Rewrite gain is `unrelated-source NLL - true-source NLL`, so a positive arm contrast means stronger true-source use relative to its unrelated-source denominator.")
    lines.append("")
    lines.append("| Readout | R−C | RS−C | R−RS | Interpretation |")
    lines.append("|---|---:|---:|---:|---|")
    lines.append(f"| Natural-copy gain | {fmt(copy_r_c.get('late_mean_delta_mean_gain_a_minus_b'))} | {fmt(copy_rs_c.get('late_mean_delta_mean_gain_a_minus_b'))} | {fmt(copy_r_rs.get('late_mean_delta_mean_gain_a_minus_b'))} | RoBERTa had no positive original exact-copy advantage, and splitting does not reveal a DeBERTa-like copy routine. |")
    lines.append(f"| Compact rewrite nonoverlap gain | {fmt(non_r_c.get('late_mean_delta_mean_gain_a_minus_b'))} | {fmt(non_rs_c.get('late_mean_delta_mean_gain_a_minus_b'))} | {fmt(non_r_rs.get('late_mean_delta_mean_gain_a_minus_b'))} | REPEAT remains below CLEAN, but RS is also below CLEAN; unlike DeBERTa, row splitting does not collapse the gain all the way to CLEAN. |")
    lines.append(f"| Nonoverlap true-source NLL (lower better; arm contrast A−B) | {fmt(non_r_c.get('late_mean_delta_mean_true_source_nll_a_minus_b'))} | {fmt(non_rs_c.get('late_mean_delta_mean_true_source_nll_a_minus_b'))} | {fmt(non_r_rs.get('late_mean_delta_mean_true_source_nll_a_minus_b'))} | The RoBERTa true-source deficit is not uniquely local-repeat-specific at this checkpoint; split exposure still carries a deficit. |")
    lines.append(f"| Nonoverlap unrelated-source NLL (lower better; arm contrast A−B) | {fmt(non_r_c.get('late_mean_delta_mean_unrelated_source_nll_a_minus_b'))} | {fmt(non_rs_c.get('late_mean_delta_mean_unrelated_source_nll_a_minus_b'))} | {fmt(non_r_rs.get('late_mean_delta_mean_unrelated_source_nll_a_minus_b'))} | Decomposition does not reproduce DeBERTa's clean true-worse/unrelated-better locality collapse. |")
    lines.append(f"| Entity rel≥3 full margin | {fmt(ent_r_c.get('late_mean_delta_mean_margin_full_a_minus_b'))} | {fmt(ent_rs_c.get('late_mean_delta_mean_margin_full_a_minus_b'))} | {fmt(getrow(late,'entity','rel_ge3','RminusRS').get('late_mean_delta_mean_margin_full_a_minus_b'))} | RoBERTa Entity remains weak and should not be used as the behavioral face of the DeBERTa mechanism. |")
    lines.append("")
    lines.append("## Scientific reading")
    lines.append("")
    lines.append("The delivered RoBERTa split arm constrains architecture generality rather than strengthening it. In DeBERTa, exact local recurrence produced a large compact nonoverlap cost relative to CLEAN and the matched REPEAT_SPLIT arm strongly attenuated that source-specific residual. In this single RoBERTa seed, the original R−C nonoverlap gain was already weaker, and RS−C remains negative rather than landing near CLEAN. RoBERTa also lacks the DeBERTa natural-copy advantage. Therefore the safest conclusion is that relation type is a training variable whose conversion depends on learner coordinate/objective phase; the DeBERTa MLM locality result should not be stated as an architecture-free law.")
    lines.append("")
    lines.append("The useful positive evidence from RoBERTa remains narrower: RoBERTa separates the VIEW/restatement arm from REPEAT on compact rewrite source use, but the exact-recurrence locality mechanism is not reproduced in the same form. This is not a failure of the DeBERTa result; it is a boundary measurement that protects the general principle from overextension.")
    lines.append("")
    lines.append("## Files")
    lines.append("")
    lines.append(f"- Raw RS rows: `{rel(RS)}`")
    lines.append(f"- Integrated late contrasts: `{rel(_public_path('experiments/archive/relation_learning/data/roberta_repeat_split_integration/roberta_repeat_split_late_contrasts.csv'))}`")
    lines.append(f"- Role summaries: `{rel(_public_path('experiments/archive/relation_learning/data/roberta_repeat_split_integration/roberta_roles_summary.csv'))}`")
    lines.append(f"- Result JSON: `{rel(_public_path('experiments/archive/relation_learning/data/roberta_repeat_split_integration/roberta_repeat_split_integration_result.json'))}`")
    _public_path('research/notes/relation_learning').mkdir(parents=True, exist_ok=True)
    NOTE.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "note": rel(NOTE), "key_numbers": result["key_numbers"]}, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
