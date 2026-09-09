#!/usr/bin/env python3
"""research: subtask reader for word-mean MLM vs token-mean legal reinvest.

Safe to run before evaluation lands: it records missing payloads and makes no
inference.  Once research word-mean 70M/80M cheap-column eval exists, this script
adds BLiMP/Supplement/EWoK UID/domain differences so the objective can be judged
as broad late-benefit strengthening versus ability redistribution.
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
DEFAULT_OUT = STUDY / "data/wordmean_subtask_interpreter"
WORDMEAN_DIR = STUDY / "data/wordmean_70_80M_eval/per_target"
TOKENMEAN_DIR = STUDY / "data/legal_mature_treatment_effect_eval/per_target"
SCALE_NOTE = (STUDY.parents[2] / 'research/documents/frontier_consolidation/data/wordmean_credit_scale_analysis/wordmean_credit_scale_analysis.md')
PRIOR_ALIGNMENT = STUDY / "data/static_prior_eval_alignment/static_prior_eval_alignment.json"
COLUMNS = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA", "Reading"]
UID_COLUMNS = ["BLiMP", "Supplement", "EWoK"]
EWOK_RELATION_DOMAINS = {"spatial-relations", "physical-relations", "social-relations", "physical-dynamics", "material-dynamics", "physical-interactions", "social-interactions"}
EWOK_PROPERTY_DOMAINS = {"agent-properties", "material-properties", "social-properties", "quantitative-properties"}


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
    pairs = [(x, y) for x, y in zip(xs, ys) if x is not None and y is not None and math.isfinite(float(x)) and math.isfinite(float(y))]
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
    pairs = [(x, y) for x, y in zip(xs, ys) if x is not None and y is not None and math.isfinite(float(x)) and math.isfinite(float(y))]
    if len(pairs) < 3:
        return None
    return pearson(ranks([float(x) for x, _ in pairs]), ranks([float(y) for _, y in pairs]))


def mean_of(rows: list[dict[str, Any]], key: str) -> float | None:
    vals = [float(r[key]) for r in rows if r.get(key) is not None]
    return sum(vals) / len(vals) if vals else None


def load_prior_pressure() -> dict[tuple[str, str], dict[str, Any]]:
    if not PRIOR_ALIGNMENT.exists():
        return {}
    data = json.loads(PRIOR_ALIGNMENT.read_text(encoding="utf-8"))
    out = {}
    for r in data.get("per_uid", []):
        out[(r.get("column"), r.get("uid"))] = r
    return out


def payload_paths(m: int) -> dict[str, pathlib.Path]:
    return {
        "wordmean": WORDMEAN_DIR / f"wordmean_mlm_seed43022_{m}M.json",
        "tokenmean": TOKENMEAN_DIR / f"complianttok_reinvest_seed43022_{m}M.json",
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT))
    ap.add_argument("--exposures", type=int, nargs="*", default=[70, 80])
    args = ap.parse_args()
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    prior = load_prior_pressure()

    paths = {f"{m}M_{k}": v for m in args.exposures for k, v in payload_paths(int(m)).items()}
    payloads = {k: load_payload(v) for k, v in paths.items()}
    missing = [k for k, v in payloads.items() if v is None]
    exposure_rows = []
    subtask_results: dict[str, Any] = {}
    for m in args.exposures:
        w = payloads.get(f"{m}M_wordmean")
        t = payloads.get(f"{m}M_tokenmean")
        row: dict[str, Any] = {"exposure_m": int(m), "complete_pair": w is not None and t is not None, "wordmean_path": str(paths[f"{m}M_wordmean"]), "tokenmean_path": str(paths[f"{m}M_tokenmean"])}
        for col in COLUMNS:
            wv = task_score(w, col)
            tv = task_score(t, col)
            row[f"wordmean_{col}"] = wv
            row[f"tokenmean_{col}"] = tv
            row[f"delta_wordmean_minus_tokenmean_{col}"] = (wv - tv) if wv is not None and tv is not None else None
        row["wordmean_mean7"] = finite_mean([row[f"wordmean_{c}"] for c in COLUMNS])
        row["tokenmean_mean7"] = finite_mean([row[f"tokenmean_{c}"] for c in COLUMNS])
        row["delta_wordmean_minus_tokenmean_mean7"] = (row["wordmean_mean7"] - row["tokenmean_mean7"]) if row["wordmean_mean7"] is not None and row["tokenmean_mean7"] is not None else None
        exposure_rows.append(row)

        exp_key = f"{m}M"
        subtask_results[exp_key] = {"complete_pair": row["complete_pair"], "columns": {}}
        if not row["complete_pair"]:
            continue
        for col in UID_COLUMNS:
            w_uid = parse_uid_accuracy(report_for(w, col))
            t_uid = parse_uid_accuracy(report_for(t, col))
            uid_rows = []
            for uid in sorted(set(w_uid) & set(t_uid)):
                item: dict[str, Any] = {"uid": uid, "wordmean": w_uid[uid], "tokenmean": t_uid[uid], "delta_wordmean_minus_tokenmean": round(w_uid[uid] - t_uid[uid], 4)}
                pp = prior.get((col, uid), {})
                for k in ["relation_only_v1_mean_weight", "relation_info_v1_mean_weight", "delta_legal_step35_minus_old_ref"]:
                    if k in pp:
                        item[k] = pp[k]
                uid_rows.append(item)
            summary: dict[str, Any] = {
                "n_uid": len(uid_rows),
                "mean_delta": round(mean_of(uid_rows, "delta_wordmean_minus_tokenmean"), 4) if uid_rows else None,
                "biggest_wordmean_losses": sorted(uid_rows, key=lambda x: x["delta_wordmean_minus_tokenmean"])[:8],
                "biggest_wordmean_gains": sorted(uid_rows, key=lambda x: x["delta_wordmean_minus_tokenmean"], reverse=True)[:8],
            }
            if col == "EWoK":
                rel = [u for u in uid_rows if u["uid"] in EWOK_RELATION_DOMAINS]
                prop = [u for u in uid_rows if u["uid"] in EWOK_PROPERTY_DOMAINS]
                summary["mean_relation_delta"] = round(mean_of(rel, "delta_wordmean_minus_tokenmean"), 4) if rel else None
                summary["mean_property_delta"] = round(mean_of(prop, "delta_wordmean_minus_tokenmean"), 4) if prop else None
                summary["relation_domain_deltas"] = {u["uid"]: u["delta_wordmean_minus_tokenmean"] for u in rel}
                summary["property_domain_deltas"] = {u["uid"]: u["delta_wordmean_minus_tokenmean"] for u in prop}
            for scheme in ["relation_only_v1_mean_weight", "relation_info_v1_mean_weight"]:
                xs = [u["delta_wordmean_minus_tokenmean"] for u in uid_rows if scheme in u]
                ys = [u.get(scheme) for u in uid_rows if scheme in u]
                pr = pearson(xs, ys)
                sr = spearman(xs, ys)
                summary[f"pearson_delta_vs_{scheme}"] = round(pr, 6) if pr is not None else None
                summary[f"spearman_delta_vs_{scheme}"] = round(sr, 6) if sr is not None else None
            subtask_results[exp_key]["columns"][col] = {"summary": summary, "uid_rows": uid_rows}

    mature_complete = all(subtask_results[f"{m}M"]["complete_pair"] for m in args.exposures)
    result = {
        "status": "WORDMEAN_SUBTASK_INTERPRETER",
        "created_utc": now_utc(),
        "interpretation_scope": "word-mean vs token-mean legal compact-view reinvest at cheap official-compatible columns and BLiMP/Supplement/EWoK UID layer; no SuperGLUE or AoA",
        "scale_caution": f"Read with {SCALE_NOTE}: word-mean changes relative word-length credit and gradient-scale geometry.",
        "paths": {k: str(v) for k, v in paths.items()},
        "missing": missing,
        "mature_complete": mature_complete,
        "exposure_rows": exposure_rows,
        "subtask_results": subtask_results,
    }
    out_json = out_dir / "wordmean_subtask_interpreter.json"
    out_json.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    def f(x: Any) -> str:
        return "NA" if x is None else f"{float(x):.4f}"
    lines = ["# research word-mean MLM subtask interpreter", "", result["interpretation_scope"], "", result["scale_caution"], "", "## Payload status"]
    for k, path in paths.items():
        lines.append(f"- `{k}` exists={path.exists()}: `{path}`")
    lines += ["", "## Column deltas", "| exposure | complete | wordmean mean7 | tokenmean mean7 | Δ mean7 | BLiMP | Supp | EWoK | Entity | COMPS | GPIQA | Reading |", "|---:|:---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in exposure_rows:
        lines.append(f"| {row['exposure_m']} | {'yes' if row['complete_pair'] else 'no'} | {f(row['wordmean_mean7'])} | {f(row['tokenmean_mean7'])} | {f(row['delta_wordmean_minus_tokenmean_mean7'])} | {f(row['delta_wordmean_minus_tokenmean_BLiMP'])} | {f(row['delta_wordmean_minus_tokenmean_Supplement'])} | {f(row['delta_wordmean_minus_tokenmean_EWoK'])} | {f(row['delta_wordmean_minus_tokenmean_Entity'])} | {f(row['delta_wordmean_minus_tokenmean_COMPS'])} | {f(row['delta_wordmean_minus_tokenmean_GlobalPIQA'])} | {f(row['delta_wordmean_minus_tokenmean_Reading'])} |")
    lines += ["", "## UID summaries"]
    if not mature_complete:
        lines.append("Word-mean 70M/80M payloads are not both present yet; do not choose from this file.")
    for exp_key in [f"{m}M" for m in args.exposures]:
        exp = subtask_results[exp_key]
        lines.append(f"\n### {exp_key}")
        if not exp["complete_pair"]:
            lines.append("Pair missing.")
            continue
        for col in UID_COLUMNS:
            summary = exp["columns"][col]["summary"]
            lines.append(f"#### {col}")
            lines.append(f"- n_uid: {summary['n_uid']}; mean Δ wordmean-tokenmean: {summary['mean_delta']}")
            if col == "EWoK":
                lines.append(f"- EWoK relation mean Δ: {summary['mean_relation_delta']}; property mean Δ: {summary['mean_property_delta']}")
            lines.append(f"- Pearson Δ vs relation_only pressure: {summary.get('pearson_delta_vs_relation_only_v1_mean_weight')}; Spearman: {summary.get('spearman_delta_vs_relation_only_v1_mean_weight')}")
            lines.append("- largest wordmean losses: " + ", ".join(f"{u['uid']} ({u['delta_wordmean_minus_tokenmean']})" for u in summary["biggest_wordmean_losses"]))
            lines.append("- largest wordmean gains: " + ", ".join(f"{u['uid']} ({u['delta_wordmean_minus_tokenmean']})" for u in summary["biggest_wordmean_gains"]))
    lines += ["", f"Full JSON: `{out_json}`"]
    out_md = out_dir / "wordmean_subtask_interpreter.md"
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "out_json": str(out_json), "out_md": str(out_md), "missing": missing, "mature_complete": mature_complete}, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
