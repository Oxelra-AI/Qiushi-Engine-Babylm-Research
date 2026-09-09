#!/usr/bin/env python3
"""research: subtask-level interpreter for the pending legal-tokenizer treatment trajectory.

The pending evaluations are expected to produce matched research-tokenizer reinvest and
clean-Qwen cheap-column evaluations at 20M/70M/80M.  The existing research summarizer reads
column averages.  This script adds the subtask layer needed for scientific route choice:
BLiMP UID, Supplement UID, and EWoK-domain reinvest-minus-clean differences, plus their
alignment with the already-built corpus-only static-prior pressure map.

It is safe to run before the required results are complete: it writes a not-ready file with the
missing payloads and no inferred scores.  It reads only existing per-target JSON files and
best_temperature_report.txt files; it does not poll tasks, evaluate checkpoints, train a
model, or change data.
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
DEFAULT_OUT = STUDY / "data/mature_treatment_subtask_interpreter"
DEFAULT_FILES = {
    "reinvest_20M": STUDY / "data/legal20m_treatment_effect_eval/per_target/complianttok_reinvest_seed43022_20M.json",
    "clean_20M": STUDY / "data/legal_mature_clean_control_eval/per_target/complianttok_cleanqwen_seed43022_20M.json",
    "reinvest_70M": STUDY / "data/legal_mature_treatment_effect_eval/per_target/complianttok_reinvest_seed43022_70M.json",
    "clean_70M": STUDY / "data/legal_mature_clean_control_eval/per_target/complianttok_cleanqwen_seed43022_70M.json",
    "reinvest_80M": STUDY / "data/legal_mature_treatment_effect_eval/per_target/complianttok_reinvest_seed43022_80M.json",
    "clean_80M": STUDY / "data/legal_mature_clean_control_eval/per_target/complianttok_cleanqwen_seed43022_80M.json",
}
PRIOR_ALIGNMENT = STUDY / "data/static_prior_eval_alignment/static_prior_eval_alignment.json"
COLUMNS = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA", "Reading"]
UID_COLUMNS = ["BLiMP", "Supplement", "EWoK"]
EWOK_RELATION_DOMAINS = {
    "spatial-relations", "physical-relations", "social-relations",
    "physical-dynamics", "material-dynamics", "physical-interactions",
    "social-interactions",
}
EWOK_PROPERTY_DOMAINS = {
    "agent-properties", "material-properties", "social-properties", "quantitative-properties",
}


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
    pairs = [(x, y) for x, y in zip(xs, ys) if x is not None and y is not None and math.isfinite(x) and math.isfinite(y)]
    if len(pairs) < 3:
        return None
    xs2, ys2 = zip(*pairs)
    mx, my = statistics.mean(xs2), statistics.mean(ys2)
    vx = sum((x - mx) ** 2 for x in xs2)
    vy = sum((y - my) ** 2 for y in ys2)
    if vx <= 0 or vy <= 0:
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
    pairs = [(x, y) for x, y in zip(xs, ys) if x is not None and y is not None and math.isfinite(x) and math.isfinite(y)]
    if len(pairs) < 3:
        return None
    return pearson(ranks([x for x, _ in pairs]), ranks([y for _, y in pairs]))


def mean_of(rows: list[dict[str, Any]], key: str) -> float | None:
    vals = [r[key] for r in rows if r.get(key) is not None]
    return sum(vals) / len(vals) if vals else None


def load_prior_pressure() -> dict[tuple[str, str], dict[str, Any]]:
    if not PRIOR_ALIGNMENT.exists():
        return {}
    data = json.loads(PRIOR_ALIGNMENT.read_text(encoding="utf-8"))
    out = {}
    for r in data.get("per_uid", []):
        out[(r.get("column"), r.get("uid"))] = r
    return out


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT))
    for key, path in DEFAULT_FILES.items():
        ap.add_argument(f"--{key.replace('_', '-')}", default=str(path))
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {key: pathlib.Path(getattr(args, key)) for key in DEFAULT_FILES}
    payloads = {key: load_payload(path) for key, path in paths.items()}
    missing = [key for key, payload in payloads.items() if payload is None]
    prior_pressure = load_prior_pressure()

    exposure_rows = []
    subtask_results: dict[str, Any] = {}
    for exposure in [20, 70, 80]:
        r = payloads.get(f"reinvest_{exposure}M")
        c = payloads.get(f"clean_{exposure}M")
        row: dict[str, Any] = {
            "exposure_m": exposure,
            "complete_pair": r is not None and c is not None,
            "reinvest_path": str(paths[f"reinvest_{exposure}M"]),
            "clean_path": str(paths[f"clean_{exposure}M"]),
        }
        for col in COLUMNS:
            rv = task_score(r, col)
            cv = task_score(c, col)
            row[f"reinvest_{col}"] = rv
            row[f"clean_{col}"] = cv
            row[f"delta_{col}"] = (rv - cv) if rv is not None and cv is not None else None
        row["reinvest_mean7"] = finite_mean([row[f"reinvest_{col}"] for col in COLUMNS])
        row["clean_mean7"] = finite_mean([row[f"clean_{col}"] for col in COLUMNS])
        row["delta_mean7"] = (row["reinvest_mean7"] - row["clean_mean7"]) if row["reinvest_mean7"] is not None and row["clean_mean7"] is not None else None
        exposure_rows.append(row)

        exp_key = f"{exposure}M"
        subtask_results[exp_key] = {"complete_pair": row["complete_pair"], "columns": {}}
        if not row["complete_pair"]:
            continue
        for col in UID_COLUMNS:
            r_uid = parse_uid_accuracy(report_for(r, col))
            c_uid = parse_uid_accuracy(report_for(c, col))
            uid_rows = []
            for uid in sorted(set(r_uid) & set(c_uid)):
                item: dict[str, Any] = {
                    "uid": uid,
                    "reinvest": r_uid[uid],
                    "clean": c_uid[uid],
                    "delta_reinvest_minus_clean": round(r_uid[uid] - c_uid[uid], 4),
                }
                pp = prior_pressure.get((col, uid), {})
                for k in ["relation_only_v1_mean_weight", "relation_info_v1_mean_weight", "delta_legal_step35_minus_old_ref"]:
                    if k in pp:
                        item[k] = pp[k]
                uid_rows.append(item)
            summary: dict[str, Any] = {
                "n_uid": len(uid_rows),
                "mean_delta": round(mean_of(uid_rows, "delta_reinvest_minus_clean"), 4) if uid_rows else None,
                "biggest_reinvest_losses": sorted(uid_rows, key=lambda x: x["delta_reinvest_minus_clean"])[:8],
                "biggest_reinvest_gains": sorted(uid_rows, key=lambda x: x["delta_reinvest_minus_clean"], reverse=True)[:8],
            }
            if col == "EWoK":
                rel = [u for u in uid_rows if u["uid"] in EWOK_RELATION_DOMAINS]
                prop = [u for u in uid_rows if u["uid"] in EWOK_PROPERTY_DOMAINS]
                summary["mean_relation_delta"] = round(mean_of(rel, "delta_reinvest_minus_clean"), 4) if rel else None
                summary["mean_property_delta"] = round(mean_of(prop, "delta_reinvest_minus_clean"), 4) if prop else None
                summary["relation_domain_deltas"] = {u["uid"]: u["delta_reinvest_minus_clean"] for u in rel}
                summary["property_domain_deltas"] = {u["uid"]: u["delta_reinvest_minus_clean"] for u in prop}
            # Does the treatment loss/gain line up with the existing static prior?
            for scheme_key in ["relation_only_v1_mean_weight", "relation_info_v1_mean_weight"]:
                xs = [u["delta_reinvest_minus_clean"] for u in uid_rows if scheme_key in u]
                ys = [u.get(scheme_key) for u in uid_rows if scheme_key in u]
                summary[f"pearson_treatment_delta_vs_{scheme_key}"] = round(pearson(xs, ys), 6) if pearson(xs, ys) is not None else None
                summary[f"spearman_treatment_delta_vs_{scheme_key}"] = round(spearman(xs, ys), 6) if spearman(xs, ys) is not None else None
            subtask_results[exp_key]["columns"][col] = {"summary": summary, "uid_rows": uid_rows}

    mature_complete = all(subtask_results[f"{m}M"]["complete_pair"] for m in [70, 80])
    route_reading = {
        "mature_complete": mature_complete,
        "text": (
            "Mature 70M/80M subtask pairs are present; read column deltas together with UID deltas and the static-prior alignment."
            if mature_complete else
            "Mature 70M/80M pairs are not both present. Do not choose a route from this file yet."
        ),
    }
    if mature_complete:
        mature = [row for row in exposure_rows if row["exposure_m"] in (70, 80)]
        route_reading["mature_delta_mean7_values"] = [row["delta_mean7"] for row in mature]
        route_reading["mature_column_delta_means"] = {
            col: round(mean_of(mature, f"delta_{col}"), 4) if mean_of(mature, f"delta_{col}") is not None else None
            for col in COLUMNS
        }
        # Not a route decision, only salient features for the next scientific reading.
        route_reading["salient_features"] = {
            "mean7_positive_both_mature": all((row["delta_mean7"] is not None and row["delta_mean7"] > 0) for row in mature),
            "ewok_relation_vs_property_70M": {
                "relation": subtask_results["70M"].get("columns", {}).get("EWoK", {}).get("summary", {}).get("mean_relation_delta"),
                "property": subtask_results["70M"].get("columns", {}).get("EWoK", {}).get("summary", {}).get("mean_property_delta"),
            },
            "ewok_relation_vs_property_80M": {
                "relation": subtask_results["80M"].get("columns", {}).get("EWoK", {}).get("summary", {}).get("mean_relation_delta"),
                "property": subtask_results["80M"].get("columns", {}).get("EWoK", {}).get("summary", {}).get("mean_property_delta"),
            },
        }

    result = {
        "status": "MATURE_TREATMENT_SUBTASK_INTERPRETER",
        "created_utc": now_utc(),
        "interpretation_scope": "cheap official-compatible zero-shot/reading columns plus BLiMP/Supplement/EWoK UID reports; no SuperGLUE or AoA; clean-Qwen uses fixed reinvest tokenizer and is a scientific control only",
        "paths": {k: str(v) for k, v in paths.items()},
        "missing": missing,
        "exposure_rows": exposure_rows,
        "subtask_results": subtask_results,
        "route_reading": route_reading,
        "prior_alignment_path": str(PRIOR_ALIGNMENT),
    }
    out_json = out_dir / "mature_treatment_subtask_interpreter.json"
    out_json.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = ["# research mature legal-tokenizer treatment subtask interpreter", "", result["interpretation_scope"], ""]
    lines.append("## Payload status")
    for key, path in paths.items():
        lines.append(f"- `{key}` exists={path.exists()}: `{path}`")
    lines.append("")
    lines.append("## Column deltas")
    lines.append("| exposure | complete | reinvest mean7 | clean mean7 | Δ mean7 | BLiMP | Supp | EWoK | Entity | COMPS | GPIQA | Reading |")
    lines.append("|---:|:---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    def f(x: Any) -> str:
        return "NA" if x is None else f"{float(x):.4f}"
    for row in exposure_rows:
        lines.append(f"| {row['exposure_m']} | {'yes' if row['complete_pair'] else 'no'} | {f(row['reinvest_mean7'])} | {f(row['clean_mean7'])} | {f(row['delta_mean7'])} | {f(row['delta_BLiMP'])} | {f(row['delta_Supplement'])} | {f(row['delta_EWoK'])} | {f(row['delta_Entity'])} | {f(row['delta_COMPS'])} | {f(row['delta_GlobalPIQA'])} | {f(row['delta_Reading'])} |")
    lines.append("")
    lines.append("## Mature route-reading status")
    lines.append(route_reading["text"])
    lines.append("")
    for exp_key in ["20M", "70M", "80M"]:
        exp = subtask_results[exp_key]
        lines.append(f"## {exp_key} UID summaries")
        if not exp["complete_pair"]:
            lines.append("Pair missing; no UID deltas computed.")
            lines.append("")
            continue
        for col in UID_COLUMNS:
            summary = exp["columns"][col]["summary"]
            lines.append(f"### {col}")
            lines.append(f"- n_uid: {summary['n_uid']}; mean Δ reinvest-clean: {summary['mean_delta']}")
            if col == "EWoK":
                lines.append(f"- EWoK relation mean Δ: {summary['mean_relation_delta']}; property mean Δ: {summary['mean_property_delta']}")
            lines.append(f"- Pearson Δ vs relation_only pressure: {summary.get('pearson_treatment_delta_vs_relation_only_v1_mean_weight')}; Spearman: {summary.get('spearman_treatment_delta_vs_relation_only_v1_mean_weight')}")
            lines.append("- largest reinvest losses: " + ", ".join(f"{u['uid']} ({u['delta_reinvest_minus_clean']})" for u in summary["biggest_reinvest_losses"]))
            lines.append("- largest reinvest gains: " + ", ".join(f"{u['uid']} ({u['delta_reinvest_minus_clean']})" for u in summary["biggest_reinvest_gains"]))
        lines.append("")
    lines.append(f"Full JSON: `{out_json}`")
    out_md = out_dir / "mature_treatment_subtask_interpreter.md"
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(json.dumps({
        "status": result["status"],
        "out_json": str(out_json),
        "out_md": str(out_md),
        "missing": missing,
        "mature_complete": mature_complete,
    }, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
