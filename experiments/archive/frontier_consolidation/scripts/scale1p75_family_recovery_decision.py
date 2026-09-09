#!/usr/bin/env python3
"""research family-recovery decision for fixed scale1.75 residual adapter.

This is the continuation rule after the research item-flip result. It is not a
simple aggregate-sign check. The fixed-scale adapter should continue to a 100M
endpoint only if the 50M losing families recover while the winning families
persist. If the same families remain traded, fixed scale1.75 is localized
redistribution and should close even if cheap7 is modestly positive.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from statistics import mean
from typing import Any

ROOT = _public_path('.')
W = _public_path('experiments/archive/frontier_consolidation')
DATA = _public_path('experiments/archive/frontier_consolidation/data')
COMPARATOR = _public_path('experiments/archive/frontier_consolidation/scripts/pairwise_item_flip_analysis.py')
OUT = _public_path('experiments/archive/frontier_consolidation/data/scale1p75_family_recovery_decision')
CHEAP = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA", "Reading"]
DISCRETE = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA"]

BASE_PAYLOADS = {
    "50M": _public_path('experiments/archive/frontier_consolidation/data/50M_eval/eval/per_target/legal_chck50M.json'),
    "70M": _public_path('experiments/archive/frontier_consolidation/data/legal_mature_treatment_effect_eval/per_target/complianttok_reinvest_seed43022_70M.json'),
    "80M": _public_path('experiments/archive/frontier_consolidation/data/legal_mature_treatment_effect_eval/per_target/complianttok_reinvest_seed43022_80M.json'),
}
CAND_PAYLOADS = {
    "50M": _public_path('experiments/archive/frontier_consolidation/data/scale1p75_50M_eval/eval/per_target/adapter128_scale1p75_h100M50M_seed43022.json'),
    "70M": _public_path('experiments/archive/frontier_consolidation/data/scale1p75_mature_merged/adapter128_scale1p75_chck_70M.json'),
    "80M": _public_path('experiments/archive/frontier_consolidation/data/scale1p75_mature_merged/adapter128_scale1p75_chck_80M.json'),
}

# Families selected by the mechanism readout.
WIN_GROUPS = {
    "BLiMP": ["superlative_quantifiers_1", "superlative_quantifiers_2", "wh_questions_object_gap", "regular_plural_subject_verb_agreement_1"],
    "Entity": ["move_contents_5_ops", "regular_4_ops", "regular_2_ops", "move_contents_0_ops"],
}
LOSS_GROUPS = {
    "BLiMP": ["npi_present_2", "left_branch_island_echo_question", "adjunct_island", "wh_vs_that_with_gap", "complex_NP_island", "npi_present_1", "wh_island"],
    "EWoK": ["material-dynamics", "material-properties", "quantitative-properties", "social-interactions"],
}
WATCH_COLUMNS = ["EWoK", "GlobalPIQA", "Reading"]


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def load(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def score_dict(payload: dict[str, Any]) -> dict[str, float | None]:
    oo = payload.get("official_overall", {}) if isinstance(payload, dict) else {}
    scores = oo.get("scores", {}) if isinstance(oo, dict) else {}
    if not scores:
        scores = payload.get("cheap7_scores", {}) if isinstance(payload, dict) else {}
    return {c: (None if scores.get(c) is None else float(scores.get(c))) for c in CHEAP}


def cheap7(scores: dict[str, float | None]) -> float | None:
    vals = [scores.get(c) for c in CHEAP]
    if any(v is None for v in vals):
        return None
    return float(mean(float(v) for v in vals))


def comp_path(exp: str) -> Path:
    return OUT / f"scale1p75_vs_step35_{exp}.json"


def ensure_comparator(exp: str, force: bool = False) -> dict[str, Any]:
    OUT.mkdir(parents=True, exist_ok=True)
    cp = comp_path(exp)
    if cp.exists() and not force:
        return {"status": "existing", "path": str(cp.relative_to(ROOT))}
    base = BASE_PAYLOADS[exp]
    cand = CAND_PAYLOADS[exp]
    if not base.exists() or not cand.exists():
        return {"status": "missing_payload", "base_exists": base.exists(), "candidate_exists": cand.exists(), "base": str(base.relative_to(ROOT)), "candidate": str(cand.relative_to(ROOT))}
    cmd = [sys.executable, str(COMPARATOR), "--base", str(base.relative_to(ROOT)), "--candidate", str(cand.relative_to(ROOT)), "--out-dir", str(OUT.relative_to(ROOT)), "--label", f"scale1p75_vs_step35_{exp}"]
    proc = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True, timeout=1200)
    rec = {"status": "ran", "returncode": proc.returncode, "stdout_tail": proc.stdout[-2000:], "stderr_tail": proc.stderr[-2000:], "path": str(cp.relative_to(ROOT))}
    if proc.returncode != 0:
        rec["status"] = "failed"
    return rec


def group_table(comp: dict[str, Any], column: str) -> dict[str, dict[str, Any]]:
    c = comp.get("columns", {}).get(column, {})
    direct = c.get("groups_by_name")
    if isinstance(direct, dict):
        return {k: v for k, v in direct.items() if isinstance(v, dict)}
    rows = []
    rows.extend(c.get("groups_all_by_item_net", []))
    rows.extend(c.get("best_groups_by_item_net", []))
    rows.extend(c.get("worst_groups_by_item_net", []))
    return {r.get("group"): r for r in rows if isinstance(r, dict) and r.get("group")}


def named_aggregate(comp: dict[str, Any], spec: dict[str, list[str]]) -> dict[str, Any]:
    by_col = {}
    total_n = total_net = total_gain = total_loss = 0
    missing = []
    for col, names in spec.items():
        table = group_table(comp, col)
        col_n = col_net = col_gain = col_loss = 0
        col_rows = []
        for name in names:
            row = table.get(name)
            if row is None:
                missing.append({"column": col, "group": name})
                continue
            n = int(row.get("n", 0)); net = int(row.get("net_gain_minus_loss", 0)); gain = int(row.get("gain", 0)); loss = int(row.get("loss", 0))
            col_n += n; col_net += net; col_gain += gain; col_loss += loss
            col_rows.append(row)
        by_col[col] = {"n": col_n, "net": col_net, "gain": col_gain, "loss": col_loss, "net_pct": (100.0 * col_net / col_n if col_n else None), "rows": col_rows}
        total_n += col_n; total_net += col_net; total_gain += col_gain; total_loss += col_loss
    return {"n": total_n, "net": total_net, "gain": total_gain, "loss": total_loss, "net_pct": (100.0 * total_net / total_n if total_n else None), "by_column": by_col, "missing_groups": missing}


def exposure_readout(exp: str, run_comparator: bool, force: bool) -> dict[str, Any]:
    base = load(BASE_PAYLOADS[exp]); cand = load(CAND_PAYLOADS[exp])
    score_rec: dict[str, Any] = {"base_payload": str(BASE_PAYLOADS[exp].relative_to(ROOT)), "candidate_payload": str(CAND_PAYLOADS[exp].relative_to(ROOT)), "base_exists": base is not None, "candidate_exists": cand is not None}
    if base and cand:
        bs = score_dict(base); cs = score_dict(cand)
        b7 = cheap7(bs); c7 = cheap7(cs)
        score_rec.update({"base_scores": bs, "candidate_scores": cs, "base_cheap7": b7, "candidate_cheap7": c7, "cheap7_delta": (None if b7 is None or c7 is None else c7 - b7), "column_deltas": {c: (None if bs.get(c) is None or cs.get(c) is None else cs[c] - bs[c]) for c in CHEAP}})
    comp_status = ensure_comparator(exp, force=force) if run_comparator else {"status": "not_requested", "path": str(comp_path(exp).relative_to(ROOT))}
    rec = {"exposure": exp, "scores": score_rec, "comparator_status": comp_status}
    comp = load(comp_path(exp))
    if comp:
        rec["win_groups"] = named_aggregate(comp, WIN_GROUPS)
        rec["loss_groups"] = named_aggregate(comp, LOSS_GROUPS)
        rec["discrete_column_deltas"] = {c: comp.get("columns", {}).get(c, {}).get("delta_score_payload") for c in DISCRETE}
    return rec


def interpret(readouts: dict[str, Any]) -> tuple[str, list[str]]:
    lines = []
    route = "pending_mature_family_readout"
    for exp in ["50M", "70M", "80M"]:
        r = readouts.get(exp, {})
        sc = r.get("scores", {})
        if sc.get("cheap7_delta") is not None:
            lines.append(f"{exp}: cheap7 Δ {sc['cheap7_delta']:+.4f}; watch-column deltas " + str({c: sc.get("column_deltas", {}).get(c) for c in WATCH_COLUMNS}))
        if r.get("win_groups") and r.get("loss_groups"):
            lines.append(f"{exp}: named win net {r['win_groups']['net']} ({r['win_groups']['net_pct']:+.2f}%), named loss net {r['loss_groups']['net']} ({r['loss_groups']['net_pct']:+.2f}%).")
    r80 = readouts.get("80M", {})
    if not (r80.get("scores", {}).get("cheap7_delta") is not None and r80.get("win_groups") and r80.get("loss_groups")):
        return route, lines + ["80M scores and item-family readout are not complete; do not launch 100M."]
    d80 = r80["scores"].get("column_deltas", {})
    cheap_delta = r80["scores"]["cheap7_delta"]
    win_net = r80["win_groups"]["net"]
    loss_net = r80["loss_groups"]["net"]
    watch_bad = {c: v for c, v in d80.items() if c in WATCH_COLUMNS and v is not None and v < -0.35}
    # Conservative implementation of the stated decision rule.
    if cheap_delta >= 0.35 and win_net > 0 and loss_net >= 0 and not watch_bad:
        route = "finish_exact_100M_only_after_family_recovery"
        lines.append("80M passes the family-recovery pattern: named winning groups remain positive, named 50M losing groups are nonnegative, and EWoK/GlobalPIQA/Reading are not materially below research. Exact 100M full-trajectory run may be scientifically justified.")
    else:
        route = "close_fixed_scale1p75_as_localized_redistribution"
        lines.append(f"80M does not pass the family-recovery pattern (cheap7 Δ={cheap_delta:+.4f}, win_net={win_net}, loss_net={loss_net}, watch_bad={watch_bad}). A positive aggregate alone is not enough if the same families stay traded.")
    return route, lines


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run-comparators", action="store_true")
    p.add_argument("--force-comparators", action="store_true")
    args = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    readouts = {exp: exposure_readout(exp, args.run_comparators, args.force_comparators) for exp in ["50M", "70M", "80M"]}
    route, interpretation = interpret(readouts)
    out = {"status": "SCALE1P75_FAMILY_RECOVERY_DECISION", "created_utc": now(), "route_signal": route, "readouts": readouts, "interpretation": interpretation, "rule": "Only launch exact 100M if losing families (BLiMP island/NPI and EWoK material/quantitative/social-interaction groups) recover while winning superlative/wh/high-operation Entity families persist; otherwise close fixed scale1.75 even if cheap7 remains modestly positive."}
    out_json = _public_path('experiments/archive/frontier_consolidation/data/scale1p75_family_recovery_decision/scale1p75_family_recovery_decision.json')
    out_md = _public_path('research/documents/frontier_consolidation/data/scale1p75_family_recovery_decision/scale1p75_family_recovery_decision.md')
    out_json.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = ["# research scale1.75 family-recovery decision", "", f"Route signal: `{route}`", "", "## Rule", "", out["rule"], "", "## Exposure readouts", "", "| exposure | cheap7 Δ | EWoK Δ | GP Δ | Reading Δ | win net | win net % | loss net | loss net % |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for exp in ["50M", "70M", "80M"]:
        r = readouts[exp]; sc = r.get("scores", {}); cd = sc.get("column_deltas", {}) or {}
        wg = r.get("win_groups", {}); lg = r.get("loss_groups", {})
        def fmt(x, nd=3):
            return "NA" if x is None else f"{float(x):+.{nd}f}"
        lines.append(f"| {exp} | {fmt(sc.get('cheap7_delta'),4)} | {fmt(cd.get('EWoK'))} | {fmt(cd.get('GlobalPIQA'))} | {fmt(cd.get('Reading'))} | {wg.get('net', 'NA')} | {fmt(wg.get('net_pct'),2)} | {lg.get('net', 'NA')} | {fmt(lg.get('net_pct'),2)} |")
    lines += ["", "## Interpretation", ""]
    for s in interpretation:
        lines.append(f"- {s}")
    lines += ["", "## Comparator status", ""]
    for exp in ["50M", "70M", "80M"]:
        lines.append(f"- {exp}: {readouts[exp].get('comparator_status')}")
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": out["status"], "out_json": str(out_json.relative_to(ROOT)), "out_md": str(out_md.relative_to(ROOT)), "route_signal": route}, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
