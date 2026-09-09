#!/usr/bin/env python3
"""research: UID/domain interpreter for a future init-matched minfreq50 screen.

Safe to run before the minfreq50 evaluation exists: it records missing payloads and
makes no scientific inference from absent scores.  Once the research minfreq50
70M/80M cheap official-compatible eval exists, this script compares it to the
research legal16k compact-view reinvest trajectory at the BLiMP/Supplement/EWoK UID
layer, with context from the support-floor substrate alignment.
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import re
import statistics
import time
from typing import Any

STUDY = pathlib.Path("experiments/archive/frontier_consolidation")
DEFAULT_OUT = STUDY / "data/minfreq50_subtask_interpreter"
MINFREQ_DIR = STUDY / "data/minfreq50_initmatched_70_80M_eval/per_target"
DIR = STUDY / "data/legal_mature_treatment_effect_eval/per_target"
CLEAN_DIR = STUDY / "data/legal_mature_clean_control_eval/per_target"
SUBSTRATE_JSON = STUDY / "data/supportfloor_substrate_alignment/supportfloor_substrate_alignment.json"
FACTOR_NOTE = (STUDY.parents[2] / 'research/documents/frontier_consolidation/data/supportfloor_factor_audit/supportfloor_factor_audit.md')
COLUMNS = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA", "Reading"]
UID_COLUMNS = ["BLiMP", "Supplement", "EWoK"]
EWOK_RELATION = {"spatial-relations", "physical-relations", "social-relations", "physical-dynamics", "material-dynamics", "physical-interactions", "social-interactions"}
EWOK_PROPERTY = {"agent-properties", "material-properties", "social-properties", "quantitative-properties"}


def now_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def load_payload(path: pathlib.Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def task_score(payload: dict[str, Any] | None, column: str) -> float | None:
    if payload is None:
        return None
    tasks = payload.get("tasks", {})
    if column == "GlobalPIQA":
        vals = []
        for c in ["GlobalPIQA_parallel", "GlobalPIQA_nonparallel"]:
            rec = tasks.get(c)
            if not isinstance(rec, dict) or rec.get("score") is None:
                return None
            vals.append(float(rec["score"]))
        return sum(vals) / len(vals)
    if column == "Reading":
        rec = tasks.get("Reading")
        if not isinstance(rec, dict):
            return None
        scores = rec.get("scores")
        if isinstance(scores, dict) and scores.get("Reading") is not None:
            return float(scores["Reading"])
        return float(rec["score"]) if rec.get("score") is not None else None
    rec = tasks.get(column)
    if not isinstance(rec, dict):
        return None
    return float(rec["score"]) if rec.get("score") is not None else None


def finite_mean(vals: list[float | None]) -> float | None:
    if any(v is None or not math.isfinite(float(v)) for v in vals):
        return None
    return sum(float(v) for v in vals) / len(vals) if vals else None


def parse_uid_accuracy(report_path: str | None) -> dict[str, float]:
    if not report_path:
        return {}
    path = pathlib.Path(report_path)
    if not path.exists():
        return {}
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    out: dict[str, float] = {}
    in_uid = False
    for ln in lines:
        s = ln.strip()
        if s.startswith("###"):
            in_uid = s.upper().startswith("### UID ACCURACY")
            continue
        if in_uid and s:
            m = re.match(r"^(.+?):\s*([-+]?\d+(?:\.\d+)?)\s*$", s)
            if m:
                out[m.group(1).strip()] = float(m.group(2))
    return out


def report_for(payload: dict[str, Any] | None, column: str) -> str | None:
    if payload is None:
        return None
    rec = payload.get("tasks", {}).get(column)
    if isinstance(rec, dict):
        return rec.get("report")
    return None


def pearson(xs: list[float | None], ys: list[float | None]) -> float | None:
    pairs = [(float(x), float(y)) for x, y in zip(xs, ys) if x is not None and y is not None and math.isfinite(float(x)) and math.isfinite(float(y))]
    if len(pairs) < 3:
        return None
    xv, yv = zip(*pairs)
    mx, my = statistics.mean(xv), statistics.mean(yv)
    vx = sum((x - mx) ** 2 for x in xv)
    vy = sum((y - my) ** 2 for y in yv)
    if vx <= 0.0 or vy <= 0.0:
        return None
    return sum((x - mx) * (y - my) for x, y in pairs) / math.sqrt(vx * vy)


def ranks(vals: list[float]) -> list[float]:
    order = sorted(range(len(vals)), key=lambda i: vals[i])
    out = [0.0] * len(vals)
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and vals[order[j]] == vals[order[i]]:
            j += 1
        avg = (i + 1 + j) / 2.0
        for k in range(i, j):
            out[order[k]] = avg
        i = j
    return out


def spearman(xs: list[float | None], ys: list[float | None]) -> float | None:
    pairs = [(float(x), float(y)) for x, y in zip(xs, ys) if x is not None and y is not None and math.isfinite(float(x)) and math.isfinite(float(y))]
    if len(pairs) < 3:
        return None
    return pearson(ranks([x for x, _ in pairs]), ranks([y for _, y in pairs]))


def mean_of(rows: list[dict[str, Any]], key: str) -> float | None:
    vals = [float(r[key]) for r in rows if r.get(key) is not None and math.isfinite(float(r[key]))]
    return sum(vals) / len(vals) if vals else None


def round_or_none(x: Any, nd: int = 6) -> Any:
    return round(float(x), nd) if isinstance(x, (int, float)) and math.isfinite(float(x)) else None


def load_substrate() -> dict[tuple[str, str], dict[str, Any]]:
    if not SUBSTRATE_JSON.exists():
        return {}
    data = json.loads(SUBSTRATE_JSON.read_text(encoding="utf-8"))
    out: dict[tuple[str, str], dict[str, Any]] = {}
    for r in data.get("per_uid", []):
        out[(r.get("column"), r.get("uid"))] = r
    return out


def payload_paths(m: int) -> dict[str, pathlib.Path]:
    return {
        "minfreq50": MINFREQ_DIR / f"minfreq50_initmatched_seed43022_{m}M.json",
        "reinvest": DIR / f"complianttok_reinvest_seed43022_{m}M.json",
        "clean_control": CLEAN_DIR / f"complianttok_cleanqwen_seed43022_{m}M.json",
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT))
    ap.add_argument("--exposures", type=int, nargs="*", default=[70, 80])
    args = ap.parse_args()

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    substrate = load_substrate()

    paths = {f"{m}M_{k}": v for m in args.exposures for k, v in payload_paths(int(m)).items()}
    payloads = {k: load_payload(v) for k, v in paths.items()}
    missing = [k for k, v in payloads.items() if v is None]

    exposure_rows: list[dict[str, Any]] = []
    subtask_results: dict[str, Any] = {}
    for m in args.exposures:
        mf = payloads.get(f"{m}M_minfreq50")
        st = payloads.get(f"{m}M_step35_reinvest")
        cl = payloads.get(f"{m}M_clean_control")
        row: dict[str, Any] = {
            "exposure_m": int(m),
            "complete_pair": mf is not None and st is not None,
            "complete_with_clean": mf is not None and st is not None and cl is not None,
            "minfreq50_path": str(paths[f"{m}M_minfreq50"]),
            "path": str(paths[f"{m}M_step35_reinvest"]),
            "clean_path": str(paths[f"{m}M_clean_control"]),
        }
        for col in COLUMNS:
            a = task_score(mf, col)
            b = task_score(st, col)
            c = task_score(cl, col)
            row[f"minfreq50_{col}"] = a
            row[f"step35_{col}"] = b
            row[f"clean_{col}"] = c
            row[f"delta_minfreq50_minus_step35_{col}"] = (a - b) if a is not None and b is not None else None
            row[f"delta_minfreq50_minus_clean_{col}"] = (a - c) if a is not None and c is not None else None
            row[f"delta_step35_minus_clean_{col}"] = (b - c) if b is not None and c is not None else None
        row["minfreq50_mean7"] = finite_mean([row[f"minfreq50_{c}"] for c in COLUMNS])
        row["mean7"] = finite_mean([row[f"step35_{c}"] for c in COLUMNS])
        row["clean_mean7"] = finite_mean([row[f"clean_{c}"] for c in COLUMNS])
        row["delta_minfreq50_minus_step35_mean7"] = (row["minfreq50_mean7"] - row["mean7"]) if row["minfreq50_mean7"] is not None and row["mean7"] is not None else None
        row["delta_minfreq50_minus_clean_mean7"] = (row["minfreq50_mean7"] - row["clean_mean7"]) if row["minfreq50_mean7"] is not None and row["clean_mean7"] is not None else None
        exposure_rows.append(row)

        exp_key = f"{m}M"
        subtask_results[exp_key] = {"complete_pair": row["complete_pair"], "columns": {}}
        if not row["complete_pair"]:
            continue
        for col in UID_COLUMNS:
            mf_uid = parse_uid_accuracy(report_for(mf, col))
            st_uid = parse_uid_accuracy(report_for(st, col))
            uid_rows: list[dict[str, Any]] = []
            for uid in sorted(set(mf_uid) & set(st_uid)):
                item: dict[str, Any] = {
                    "column": col,
                    "uid": uid,
                    "minfreq50": mf_uid[uid],
                    "research": st_uid[uid],
                    "delta_minfreq50_minus_step35": round(mf_uid[uid] - st_uid[uid], 4),
                }
                sub = substrate.get((col, uid), {})
                for k in [
                    "legal_delta_step35_minus_old", "legal_deficit_magnitude",
                    "minfreq50_minus_step35_tpw", "minfreq50_token_reduction_pct_vs_step35",
                    "minfreq50_minus_step35_support_log_gain", "minfreq50_minus_step35_frac_lt50_delta",
                    "minfreq50_minus_step35_vs_old_token_multiset_jaccard", "minfreq50_minus_step35_vs_old_b_not_in_a_frac",
                ]:
                    if k in sub:
                        item[k] = sub[k]
                if col == "EWoK":
                    if uid in EWOK_RELATION:
                        item["ewok_family"] = "relation_or_dynamics"
                    elif uid in EWOK_PROPERTY:
                        item["ewok_family"] = "property"
                    else:
                        item["ewok_family"] = "other"
                uid_rows.append(item)
            summary: dict[str, Any] = {
                "n_uid": len(uid_rows),
                "mean_delta": round_or_none(mean_of(uid_rows, "delta_minfreq50_minus_step35")),
                "biggest_minfreq50_losses": sorted(uid_rows, key=lambda x: x["delta_minfreq50_minus_step35"])[:8],
                "biggest_minfreq50_gains": sorted(uid_rows, key=lambda x: x["delta_minfreq50_minus_step35"], reverse=True)[:8],
            }
            if col == "EWoK":
                rel = [u for u in uid_rows if u.get("ewok_family") == "relation_or_dynamics"]
                prop = [u for u in uid_rows if u.get("ewok_family") == "property"]
                summary["mean_relation_delta"] = round_or_none(mean_of(rel, "delta_minfreq50_minus_step35"))
                summary["mean_property_delta"] = round_or_none(mean_of(prop, "delta_minfreq50_minus_step35"))
                summary["relation_domain_deltas"] = {u["uid"]: u["delta_minfreq50_minus_step35"] for u in rel}
                summary["property_domain_deltas"] = {u["uid"]: u["delta_minfreq50_minus_step35"] for u in prop}
            for metric in [
                "legal_deficit_magnitude", "minfreq50_minus_step35_tpw",
                "minfreq50_token_reduction_pct_vs_step35", "minfreq50_minus_step35_support_log_gain",
                "minfreq50_minus_step35_frac_lt50_delta", "minfreq50_minus_step35_vs_old_token_multiset_jaccard",
            ]:
                xs = [u.get("delta_minfreq50_minus_step35") for u in uid_rows]
                ys = [u.get(metric) for u in uid_rows]
                summary[f"pearson_delta_vs_{metric}"] = round_or_none(pearson(xs, ys))
                summary[f"spearman_delta_vs_{metric}"] = round_or_none(spearman(xs, ys))
            subtask_results[exp_key]["columns"][col] = {"summary": summary, "uid_rows": uid_rows}

    mature_complete = all(subtask_results[f"{m}M"]["complete_pair"] for m in args.exposures)
    result = {
        "status": "MINFREQ50_SUBTASK_INTERPRETER",
        "created_utc": now_utc(),
        "interpretation_scope": "minfreq50 init-matched support-floor vs research legal16k compact-view reinvest at cheap official-compatible columns and BLiMP/Supplement/EWoK UID layer; no SuperGLUE or AoA.",
        "substrate_context": f"Read with {SUBSTRATE_JSON}: support-floor is a broad representation/support/segmentation package, not a targeted UID repair.",
        "factor_note": str(FACTOR_NOTE),
        "paths": {k: str(v) for k, v in paths.items()},
        "missing": missing,
        "mature_complete": mature_complete,
        "exposure_rows": exposure_rows,
        "subtask_results": subtask_results,
    }
    out_json = out_dir / "minfreq50_subtask_interpreter.json"
    out_json.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    def f(x: Any) -> str:
        return "NA" if x is None else f"{float(x):.4f}"

    lines = [
        "# research minfreq50 subtask interpreter",
        "",
        result["interpretation_scope"],
        "",
        result["substrate_context"],
        "",
        "## Payload status",
    ]
    for k, path in paths.items():
        lines.append(f"- `{k}` exists={path.exists()}: `{path}`")
    lines += [
        "",
        "## Column deltas",
        "| exposure | complete | minfreq50 mean7 | research mean7 | clean mean7 | Δ minfreq-research | Δ minfreq-clean | BLiMP | Supp | EWoK | Entity | COMPS | GPIQA | Reading |",
        "|---:|:---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in exposure_rows:
        lines.append(f"| {row['exposure_m']} | {'yes' if row['complete_pair'] else 'no'} | {f(row['minfreq50_mean7'])} | {f(row['mean7'])} | {f(row['clean_mean7'])} | {f(row['delta_minfreq50_minus_step35_mean7'])} | {f(row['delta_minfreq50_minus_clean_mean7'])} | {f(row['delta_minfreq50_minus_step35_BLiMP'])} | {f(row['delta_minfreq50_minus_step35_Supplement'])} | {f(row['delta_minfreq50_minus_step35_EWoK'])} | {f(row['delta_minfreq50_minus_step35_Entity'])} | {f(row['delta_minfreq50_minus_step35_COMPS'])} | {f(row['delta_minfreq50_minus_step35_GlobalPIQA'])} | {f(row['delta_minfreq50_minus_step35_Reading'])} |")
    lines += ["", "## UID summaries"]
    if not mature_complete:
        lines.append("The minfreq50 70M/80M payloads are not both present; do not choose from this file yet.")
    for exp_key in [f"{m}M" for m in args.exposures]:
        exp = subtask_results[exp_key]
        lines.append(f"\n### {exp_key}")
        if not exp["complete_pair"]:
            lines.append("Pair missing.")
            continue
        for col in UID_COLUMNS:
            summary = exp["columns"][col]["summary"]
            lines.append(f"#### {col}")
            lines.append(f"- n_uid: {summary['n_uid']}; mean Δ minfreq50-research: {summary['mean_delta']}")
            if col == "EWoK":
                lines.append(f"- EWoK relation/dynamics mean Δ: {summary['mean_relation_delta']}; property mean Δ: {summary['mean_property_delta']}")
            lines.append(f"- Pearson Δ vs prior legal deficit magnitude: {summary.get('pearson_delta_vs_legal_deficit_magnitude')}; Spearman: {summary.get('spearman_delta_vs_legal_deficit_magnitude')}")
            lines.append(f"- Pearson Δ vs minfreq50 token reduction: {summary.get('pearson_delta_vs_minfreq50_token_reduction_pct_vs_step35')}; Spearman: {summary.get('spearman_delta_vs_minfreq50_token_reduction_pct_vs_step35')}")
            lines.append("- largest minfreq50 losses: " + ", ".join(f"{u['uid']} ({u['delta_minfreq50_minus_step35']})" for u in summary["biggest_minfreq50_losses"]))
            lines.append("- largest minfreq50 gains: " + ", ".join(f"{u['uid']} ({u['delta_minfreq50_minus_step35']})" for u in summary["biggest_minfreq50_gains"]))
    lines += ["", f"Full JSON: `{out_json}`"]
    out_md = out_dir / "minfreq50_subtask_interpreter.md"
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "out_json": str(out_json), "out_md": str(out_md), "missing": missing, "mature_complete": mature_complete}, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
