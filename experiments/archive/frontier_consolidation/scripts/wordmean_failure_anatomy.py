#!/usr/bin/env python3
"""research: word-mean MLM failure anatomy from recovered cheap official-compatible reports.

CPU-only analysis.  Reads already-produced research word-mean 70M/80M reports and
research token-mean/clean references.  It does not train or evaluate a model.
"""
from __future__ import annotations

import csv
import json
import math
import pathlib
import re
import statistics
import time
from typing import Any

STUDY = pathlib.Path("experiments/archive/frontier_consolidation")
WORDMEAN_DIR = STUDY / "data/wordmean_70_80M_eval/per_target"
TOKENMEAN_DIR = STUDY / "data/legal_mature_treatment_effect_eval/per_target"
CLEAN_DIR = STUDY / "data/legal_mature_clean_control_eval/per_target"
SUPPORT_CSV = STUDY / "data/supportfloor_substrate_alignment/supportfloor_substrate_alignment_per_uid.csv"
OUT_DIR = STUDY / "data/wordmean_failure_anatomy"
NOTE = (STUDY.parents[2] / 'research/notes/frontier_consolidation/wordmean_failure_anatomy.md')

CHEAP7 = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA", "Reading"]
RAW_TASKS = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA_parallel", "GlobalPIQA_nonparallel", "Reading"]
REPORT_COLUMNS = ["BLiMP", "Supplement", "EWoK"]
EWOK_RELATION = {"spatial-relations", "physical-relations", "social-relations", "physical-dynamics", "material-dynamics", "physical-interactions", "social-interactions"}
EWOK_PROPERTY = {"agent-properties", "material-properties", "social-properties", "quantitative-properties"}


def now_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def load_json(path: pathlib.Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def payload_path(kind: str, m: int) -> pathlib.Path:
    if kind == "wordmean":
        return WORDMEAN_DIR / f"wordmean_mlm_seed43022_{m}M.json"
    if kind == "tokenmean":
        return TOKENMEAN_DIR / f"complianttok_reinvest_seed43022_{m}M.json"
    if kind == "clean":
        return CLEAN_DIR / f"complianttok_cleanqwen_seed43022_{m}M.json"
    raise KeyError(kind)


def task_score(payload: dict[str, Any], col: str) -> float:
    tasks = payload["tasks"]
    if col == "GlobalPIQA":
        return (float(tasks["GlobalPIQA_parallel"]["score"]) + float(tasks["GlobalPIQA_nonparallel"]["score"])) / 2.0
    if col == "Reading":
        rec = tasks["Reading"]
        if isinstance(rec.get("scores"), dict) and rec["scores"].get("Reading") is not None:
            return float(rec["scores"]["Reading"])
        return float(rec["score"])
    return float(tasks[col]["score"])


def raw_task_score(payload: dict[str, Any], task: str) -> float:
    rec = payload["tasks"][task]
    if task == "Reading":
        if isinstance(rec.get("scores"), dict) and rec["scores"].get("Reading") is not None:
            return float(rec["scores"]["Reading"])
    return float(rec["score"])


def mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else float("nan")


def median(xs: list[float]) -> float | None:
    return float(statistics.median(xs)) if xs else None


def pearson(xs: list[float], ys: list[float]) -> float | None:
    pairs = [(float(x), float(y)) for x, y in zip(xs, ys) if math.isfinite(float(x)) and math.isfinite(float(y))]
    if len(pairs) < 3:
        return None
    x2, y2 = zip(*pairs)
    mx = sum(x2) / len(x2)
    my = sum(y2) / len(y2)
    vx = sum((x - mx) ** 2 for x in x2)
    vy = sum((y - my) ** 2 for y in y2)
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


def spearman(xs: list[float], ys: list[float]) -> float | None:
    pairs = [(float(x), float(y)) for x, y in zip(xs, ys) if math.isfinite(float(x)) and math.isfinite(float(y))]
    if len(pairs) < 3:
        return None
    return pearson(ranks([x for x, _ in pairs]), ranks([y for _, y in pairs]))


def parse_report_sections(path: str | None) -> dict[str, dict[str, float]]:
    if not path:
        return {}
    p = pathlib.Path(path)
    if not p.exists():
        return {}
    sections: dict[str, dict[str, float]] = {}
    current: str | None = None
    for ln in p.read_text(encoding="utf-8", errors="replace").splitlines():
        s = ln.strip()
        if s.startswith("###"):
            current = s.lstrip("#").strip().upper()
            sections.setdefault(current, {})
            continue
        if current and s:
            m = re.match(r"^(.+?):\s*([-+]?\d+(?:\.\d+)?)\s*$", s)
            if m:
                sections[current][m.group(1).strip()] = float(m.group(2))
    return sections


def report_path(payload: dict[str, Any], col: str) -> str | None:
    rec = payload.get("tasks", {}).get(col)
    return rec.get("report") if isinstance(rec, dict) else None


def load_support_rows() -> dict[tuple[str, str], dict[str, float]]:
    out: dict[tuple[str, str], dict[str, float]] = {}
    if not SUPPORT_CSV.exists():
        return out
    with SUPPORT_CSV.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rec: dict[str, float] = {}
            for k, v in row.items():
                if k in {"column", "uid"}:
                    continue
                try:
                    rec[k] = float(v)
                except (TypeError, ValueError):
                    pass
            out[(row["column"], row["uid"])] = rec
    return out


def summarize_deltas(rows: list[dict[str, Any]], delta_key: str = "delta_wordmean_minus_tokenmean") -> dict[str, Any]:
    vals = [float(r[delta_key]) for r in rows]
    return {
        "n": len(vals),
        "mean": mean(vals) if vals else None,
        "median": median(vals),
        "min": min(vals) if vals else None,
        "max": max(vals) if vals else None,
        "num_negative": sum(v < 0 for v in vals),
        "num_positive": sum(v > 0 for v in vals),
        "num_leq_minus2": sum(v <= -2 for v in vals),
        "num_geq_plus2": sum(v >= 2 for v in vals),
    }


def fmt(x: Any, nd: int = 4) -> str:
    if x is None:
        return "NA"
    try:
        xf = float(x)
    except Exception:
        return str(x)
    if math.isnan(xf):
        return "NA"
    return f"{xf:.{nd}f}"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    NOTE.parent.mkdir(parents=True, exist_ok=True)
    support = load_support_rows()

    payloads: dict[str, dict[str, Any]] = {}
    missing = []
    for m in [70, 80]:
        for kind in ["wordmean", "tokenmean", "clean"]:
            p = payload_path(kind, m)
            if not p.exists():
                missing.append(str(p))
            else:
                payloads[f"{kind}_{m}M"] = load_json(p)

    column_rows: list[dict[str, Any]] = []
    raw_task_rows: list[dict[str, Any]] = []
    report_results: dict[str, Any] = {}
    if not missing:
        for m in [70, 80]:
            w = payloads[f"wordmean_{m}M"]
            t = payloads[f"tokenmean_{m}M"]
            c = payloads[f"clean_{m}M"]
            crow: dict[str, Any] = {"exposure_m": m}
            for col in CHEAP7:
                wv, tv, cv = task_score(w, col), task_score(t, col), task_score(c, col)
                crow[f"wordmean_{col}"] = wv
                crow[f"tokenmean_{col}"] = tv
                crow[f"clean_{col}"] = cv
                crow[f"delta_wordmean_minus_tokenmean_{col}"] = wv - tv
                crow[f"delta_wordmean_minus_clean_{col}"] = wv - cv
                crow[f"delta_tokenmean_minus_clean_{col}"] = tv - cv
            crow["wordmean_mean7"] = mean([crow[f"wordmean_{c}"] for c in CHEAP7])
            crow["tokenmean_mean7"] = mean([crow[f"tokenmean_{c}"] for c in CHEAP7])
            crow["clean_mean7"] = mean([crow[f"clean_{c}"] for c in CHEAP7])
            crow["delta_wordmean_minus_tokenmean_mean7"] = crow["wordmean_mean7"] - crow["tokenmean_mean7"]
            crow["delta_wordmean_minus_clean_mean7"] = crow["wordmean_mean7"] - crow["clean_mean7"]
            crow["delta_tokenmean_minus_clean_mean7"] = crow["tokenmean_mean7"] - crow["clean_mean7"]
            column_rows.append(crow)

            for task in RAW_TASKS:
                wv, tv, cv = raw_task_score(w, task), raw_task_score(t, task), raw_task_score(c, task)
                raw_task_rows.append({
                    "exposure_m": m,
                    "task": task,
                    "wordmean": wv,
                    "tokenmean": tv,
                    "clean": cv,
                    "delta_wordmean_minus_tokenmean": wv - tv,
                    "delta_wordmean_minus_clean": wv - cv,
                    "delta_tokenmean_minus_clean": tv - cv,
                })

            exp_key = f"{m}M"
            report_results[exp_key] = {}
            for col in REPORT_COLUMNS:
                w_sections = parse_report_sections(report_path(w, col))
                t_sections = parse_report_sections(report_path(t, col))
                c_sections = parse_report_sections(report_path(c, col))
                col_result: dict[str, Any] = {}
                for section in sorted(set(w_sections) | set(t_sections) | set(c_sections)):
                    keys = sorted(set(w_sections.get(section, {})) & set(t_sections.get(section, {})))
                    rows = []
                    for uid in keys:
                        rec: dict[str, Any] = {
                            "name": uid,
                            "wordmean": w_sections[section][uid],
                            "tokenmean": t_sections[section][uid],
                            "delta_wordmean_minus_tokenmean": round(w_sections[section][uid] - t_sections[section][uid], 4),
                        }
                        if uid in c_sections.get(section, {}):
                            rec["clean"] = c_sections[section][uid]
                            rec["delta_wordmean_minus_clean"] = round(w_sections[section][uid] - c_sections[section][uid], 4)
                            rec["delta_tokenmean_minus_clean"] = round(t_sections[section][uid] - c_sections[section][uid], 4)
                        srec = support.get((col, uid), {}) if section == "UID ACCURACY" else {}
                        for k in [
                            "legal_delta_step35_minus_old",
                            "legal_deficit_magnitude",
                            "minfreq50_token_reduction_pct_vs_step35",
                            "minfreq50_minus_step35_tpw",
                            "minfreq50_minus_step35_support_log_gain",
                            "minfreq50_minus_step35_frac_lt50_delta",
                        ]:
                            if k in srec:
                                rec[k] = srec[k]
                        rows.append(rec)
                    summary = summarize_deltas(rows) if rows else {"n": 0}
                    if section == "UID ACCURACY" and rows:
                        deficit_rows = [r for r in rows if "legal_deficit_magnitude" in r]
                        if len(deficit_rows) >= 3:
                            xs = [r["delta_wordmean_minus_tokenmean"] for r in deficit_rows]
                            ys = [r["legal_deficit_magnitude"] for r in deficit_rows]
                            summary["pearson_delta_vs_legal_deficit_magnitude"] = pearson(xs, ys)
                            summary["spearman_delta_vs_legal_deficit_magnitude"] = spearman(xs, ys)
                            for metric in ["minfreq50_token_reduction_pct_vs_step35", "minfreq50_minus_step35_support_log_gain"]:
                                metric_rows = [r for r in deficit_rows if metric in r]
                                if len(metric_rows) >= 3:
                                    summary[f"pearson_delta_vs_{metric}"] = pearson(
                                        [r["delta_wordmean_minus_tokenmean"] for r in metric_rows],
                                        [r[metric] for r in metric_rows],
                                    )
                        if col == "EWoK":
                            rel = [r for r in rows if r["name"] in EWOK_RELATION]
                            prop = [r for r in rows if r["name"] in EWOK_PROPERTY]
                            summary["relation_mean_delta"] = mean([r["delta_wordmean_minus_tokenmean"] for r in rel]) if rel else None
                            summary["property_mean_delta"] = mean([r["delta_wordmean_minus_tokenmean"] for r in prop]) if prop else None
                            summary["relation_deltas"] = {r["name"]: r["delta_wordmean_minus_tokenmean"] for r in rel}
                            summary["property_deltas"] = {r["name"]: r["delta_wordmean_minus_tokenmean"] for r in prop}
                    col_result[section] = {
                        "summary": summary,
                        "largest_wordmean_losses": sorted(rows, key=lambda r: r["delta_wordmean_minus_tokenmean"])[:12],
                        "largest_wordmean_gains": sorted(rows, key=lambda r: r["delta_wordmean_minus_tokenmean"], reverse=True)[:12],
                        "rows": rows,
                    }
                report_results[exp_key][col] = col_result

    result = {
        "status": "WORDMEAN_FAILURE_ANATOMY",
        "created_utc": now_utc(),
        "scope": "CPU-only analysis of already completed word-mean/token-mean/clean 70M/80M official-compatible cheap reports; no model training or evaluation.",
        "paths": {
            f"{kind}_{m}M": str(payload_path(kind, m)) for m in [70, 80] for kind in ["wordmean", "tokenmean", "clean"]
        },
        "missing": missing,
        "column_rows": column_rows,
        "raw_task_rows": raw_task_rows,
        "report_results": report_results,
        "scientific_reading": [
            "Word-mean remains a positive compact-view treatment relative to the clean fixed-tokenizer control, but it is weaker than token-mean reinvestment on the broad language surface by 80M.",
            "Its advantage is concentrated in GlobalPIQA, especially nonparallel examples, and modestly COMPS; BLiMP, Supplement, EWoK, Entity, and Reading move down together.",
            "Because the 41.8 gap requires roughly +0.70 cheap7 mean gain if SuperGLUE/AoA stay flat, the observed 80M -0.25 mean7 and damaged core columns make global word-mean unsuitable for 100M continuation.",
            "The failure is not evidence against compact semantic second views: word-mean still beats clean at 70M/80M, while the ordinary token-mean objective carries the stronger late-emerging treatment effect.",
            "If the GlobalPIQA gain is reused later, it should be isolated as a small scheduled or mixed credit component, not as a wholesale replacement for token-mean MLM, and only after current minfreq50 evidence is read.",
        ],
    }
    out_json = OUT_DIR / "wordmean_failure_anatomy.json"
    out_json.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    lines: list[str] = []
    lines.append("# research word-mean failure anatomy")
    lines.append("")
    lines.append(result["scope"])
    lines.append("")
    if missing:
        lines.append("Missing required payloads; no interpretation should be drawn.")
        for p in missing:
            lines.append(f"- `{p}`")
    else:
        lines.append("## Column movement")
        lines.append("| exposure | wm mean7 | tm mean7 | clean mean7 | wm-tm | wm-clean | BLiMP | Supp | EWoK | Entity | COMPS | GPIQA | Reading |")
        lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for r in column_rows:
            lines.append(
                f"| {r['exposure_m']} | {fmt(r['wordmean_mean7'])} | {fmt(r['tokenmean_mean7'])} | {fmt(r['clean_mean7'])} | "
                f"{fmt(r['delta_wordmean_minus_tokenmean_mean7'])} | {fmt(r['delta_wordmean_minus_clean_mean7'])} | "
                f"{fmt(r['delta_wordmean_minus_tokenmean_BLiMP'])} | {fmt(r['delta_wordmean_minus_tokenmean_Supplement'])} | "
                f"{fmt(r['delta_wordmean_minus_tokenmean_EWoK'])} | {fmt(r['delta_wordmean_minus_tokenmean_Entity'])} | "
                f"{fmt(r['delta_wordmean_minus_tokenmean_COMPS'])} | {fmt(r['delta_wordmean_minus_tokenmean_GlobalPIQA'])} | "
                f"{fmt(r['delta_wordmean_minus_tokenmean_Reading'])} |"
            )
        lines.append("")
        lines.append("## Raw GlobalPIQA split")
        lines.append("| exposure | task | wordmean | tokenmean | clean | wm-tm | wm-clean |")
        lines.append("|---:|---|---:|---:|---:|---:|---:|")
        for rr in raw_task_rows:
            if rr["task"].startswith("GlobalPIQA") or rr["task"] == "COMPS":
                lines.append(f"| {rr['exposure_m']} | {rr['task']} | {fmt(rr['wordmean'])} | {fmt(rr['tokenmean'])} | {fmt(rr['clean'])} | {fmt(rr['delta_wordmean_minus_tokenmean'])} | {fmt(rr['delta_wordmean_minus_clean'])} |")
        lines.append("")
        lines.append("## 80M UID and domain movement")
        exp80 = report_results.get("80M", {})
        for col in REPORT_COLUMNS:
            lines.append(f"### {col}")
            uid = exp80.get(col, {}).get("UID ACCURACY", {})
            summ = uid.get("summary", {})
            lines.append(f"- UID count {summ.get('n')}; mean wm-tm {fmt(summ.get('mean'))}; median {fmt(summ.get('median'))}; negative {summ.get('num_negative')}; positive {summ.get('num_positive')}; <=-2 {summ.get('num_leq_minus2')}; >=+2 {summ.get('num_geq_plus2')}.")
            if "pearson_delta_vs_legal_deficit_magnitude" in summ:
                lines.append(f"- Pearson wm-tm UID delta vs research legal-loss magnitude: {fmt(summ.get('pearson_delta_vs_legal_deficit_magnitude'), 4)}; Spearman {fmt(summ.get('spearman_delta_vs_legal_deficit_magnitude'), 4)}.")
            if col == "EWoK":
                lines.append(f"- EWoK relation mean wm-tm {fmt(summ.get('relation_mean_delta'))}; property mean {fmt(summ.get('property_mean_delta'))}.")
                rd = summ.get("relation_deltas", {})
                pd = summ.get("property_deltas", {})
                lines.append("- relation deltas: " + ", ".join(f"{k} {fmt(v)}" for k, v in sorted(rd.items())))
                lines.append("- property deltas: " + ", ".join(f"{k} {fmt(v)}" for k, v in sorted(pd.items())))
            losses = uid.get("largest_wordmean_losses", [])[:8]
            gains = uid.get("largest_wordmean_gains", [])[:8]
            lines.append("- largest UID losses: " + ", ".join(f"{x['name']} ({fmt(x['delta_wordmean_minus_tokenmean'])})" for x in losses))
            lines.append("- largest UID gains: " + ", ".join(f"{x['name']} ({fmt(x['delta_wordmean_minus_tokenmean'])})" for x in gains))
            # Include coarse report terms when available.
            for section in ["FIELD ACCURACY", "LINGUISTICS_TERM ACCURACY"]:
                sec = exp80.get(col, {}).get(section)
                if not sec:
                    continue
                ss = sec.get("summary", {})
                lines.append(f"- {section.lower()} mean wm-tm {fmt(ss.get('mean'))}; largest losses: " + ", ".join(f"{x['name']} ({fmt(x['delta_wordmean_minus_tokenmean'])})" for x in sec.get("largest_wordmean_losses", [])[:5]))
            lines.append("")
        lines.append("## Scientific reading")
        for s in result["scientific_reading"]:
            lines.append(f"- {s}")
    lines.append("")
    lines.append(f"JSON: `{out_json}`")
    NOTE.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(json.dumps({
        "status": result["status"],
        "out_json": str(out_json),
        "out_note": str(NOTE),
        "missing": missing,
        "n_column_rows": len(column_rows),
        "n_raw_task_rows": len(raw_task_rows),
    }, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
