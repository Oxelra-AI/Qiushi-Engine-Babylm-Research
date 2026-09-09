#!/usr/bin/env python3
"""research: parse fast-eval reports into subtask-level compact-density deltas.

This CPU-only script reads already-finished frontier_consolidation fast official-compatible
no-AoA outputs for compact_repeat_core, compact_view_core, and
compact_view_reinvest.  It does not touch the running research full/seed jobs.

Purpose: expose which benchmark subfamilies carry the compact-view and reinvest
movement so that a future endpoint result can be interpreted by mechanism and
not only by a scalar Overall projection.
"""
from __future__ import annotations

import json
import math
import pathlib
import re
from typing import Any

ROOT = pathlib.Path(".").resolve()
A01_WORKSPACE = ROOT / "experiments/archive" / 'representation_and_objectives'
OUT_DIR = A01_WORKSPACE / "data" / "compact_density_subtask_delta"
OUT_JSON = OUT_DIR / "compact_density_subtask_delta.json"
OUT_NOTE = (ROOT / 'research/notes/representation_and_objectives/compact_density_subtask_delta.md')

PAYLOADS = {
    "compact_repeat_core": ROOT / "experiments/archive" / 'frontier_consolidation' / "data" / "density_noaoa_eval_compact_core" / "per_target" / "compact_repeat_core.json",
    "compact_view_core": ROOT / "experiments/archive" / 'frontier_consolidation' / "data" / "density_noaoa_eval_compact_core" / "per_target" / "compact_view_core.json",
    "compact_view_reinvest": ROOT / "experiments/archive" / 'frontier_consolidation' / "data" / "density_noaoa_eval_reinvest" / "per_target" / "compact_view_reinvest.json",
}

PRIMARY_COLUMNS = [
    "BLiMP", "Supplement", "EWoK", "Entity", "Entity_full", "COMPS",
    "GlobalPIQA_parallel", "GlobalPIQA_nonparallel", "GlobalPIQA_mean",
    "Reading", "Reading_eye", "Reading_self_paced", "equal7_mean", "equal7_full_entity",
]
TASK_OVERALL_KEYS = [
    "BLiMP", "Supplement", "EWoK", "Entity", "Entity_full", "COMPS",
    "GlobalPIQA_parallel", "GlobalPIQA_nonparallel", "Reading",
]
REPORT_SECTIONS = [
    "FIELD ACCURACY", "UID ACCURACY", "LINGUISTICS_TERM ACCURACY",
    "CONTEXT_TYPE ACCURACY", "CONTEXT_CONTRAST ACCURACY", "TARGET_CONTRAST ACCURACY",
]

LEADER_FAST = {
    "BLiMP": 67.20, "Supplement": 56.01, "EWoK": 56.07, "Entity": 28.45,
    "COMPS": 53.57, "GlobalPIQA_mean": 39.67, "Reading": 5.42,
}
COMPACT_EXPERIENCE_CLEAN_FAST = {
    "BLiMP": 66.84, "Supplement": 62.84, "EWoK": 50.19, "Entity": 25.76,
    "COMPS": 51.78, "GlobalPIQA_mean": 36.62, "Reading": 7.76,
}


def read_json(path: pathlib.Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def latest_file(root: pathlib.Path, pattern: str) -> pathlib.Path | None:
    hits = sorted(root.rglob(pattern), key=lambda p: (p.stat().st_mtime, str(p)))
    return hits[-1] if hits else None


def fnum(text: str) -> float | None:
    try:
        v = float(text)
    except Exception:
        return None
    return v if math.isfinite(v) else None


def parse_report(path: pathlib.Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {"exists": False, "path": str(path) if path else None}
    sections: dict[str, dict[str, float]] = {}
    current: str | None = None
    average: float | None = None
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("### "):
            current = line[4:].strip()
            sections.setdefault(current, {})
            continue
        if current == "AVERAGE ACCURACY":
            v = fnum(line)
            if v is not None:
                average = v
            continue
        if current and ":" in line:
            key, val = line.split(":", 1)
            m = re.search(r"[+-]?[0-9]+(?:\.[0-9]+)?", val)
            if m:
                v = fnum(m.group(0))
                if v is not None:
                    sections.setdefault(current, {})[key.strip()] = v
    return {"exists": True, "path": str(path), "average_accuracy": average, "sections": sections}


def parse_reading_report(path: pathlib.Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {"exists": False, "path": str(path) if path else None}
    text = path.read_text(encoding="utf-8", errors="replace")
    scores: dict[str, float] = {}
    for label, key in [("EYE TRACKING SCORE", "Reading_eye"), ("SELF-PACED READING SCORE", "Reading_self_paced")]:
        m = re.search(re.escape(label) + r":\s*([+-]?[0-9]+(?:\.[0-9]+)?)", text)
        if m:
            val = fnum(m.group(1))
            if val is not None:
                scores[key] = val
    if "Reading_eye" in scores and "Reading_self_paced" in scores:
        scores["Reading"] = (scores["Reading_eye"] + scores["Reading_self_paced"]) / 2.0
    return {"exists": True, "path": str(path), "scores": scores}


def read_prediction_counts(path: pathlib.Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {"exists": False, "path": str(path) if path else None, "counts": {}, "total": 0}
    data = read_json(path)
    counts: dict[str, int] = {}
    if isinstance(data, dict):
        for key, val in data.items():
            if isinstance(val, dict) and isinstance(val.get("predictions"), list):
                counts[str(key)] = len(val["predictions"])
            elif isinstance(val, list):
                counts[str(key)] = len(val)
    total = sum(counts.values())
    return {"exists": True, "path": str(path), "counts": counts, "total": total}


def extract_scores(payload: dict[str, Any]) -> dict[str, float | None]:
    tasks = payload.get("tasks", {})
    out: dict[str, float | None] = {k: None for k in PRIMARY_COLUMNS}
    for col in ["BLiMP", "Supplement", "EWoK", "Entity", "Entity_full", "COMPS", "GlobalPIQA_parallel", "GlobalPIQA_nonparallel"]:
        rec = tasks.get(col)
        if isinstance(rec, dict) and rec.get("score") is not None:
            out[col] = float(rec["score"])
    r = tasks.get("Reading")
    if isinstance(r, dict):
        for k, v in (r.get("scores") or {}).items():
            if k in out and v is not None:
                out[k] = float(v)
    if out["GlobalPIQA_parallel"] is not None and out["GlobalPIQA_nonparallel"] is not None:
        out["GlobalPIQA_mean"] = (float(out["GlobalPIQA_parallel"]) + float(out["GlobalPIQA_nonparallel"])) / 2.0
    eq = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA_mean", "Reading"]
    if all(out.get(k) is not None for k in eq):
        out["equal7_mean"] = sum(float(out[k]) for k in eq) / len(eq)
    eqf = ["BLiMP", "Supplement", "EWoK", "Entity_full", "COMPS", "GlobalPIQA_mean", "Reading"]
    if all(out.get(k) is not None for k in eqf):
        out["equal7_full_entity"] = sum(float(out[k]) for k in eqf) / len(eqf)
    return out


def load_target(name: str, path: pathlib.Path) -> dict[str, Any]:
    payload = read_json(path)
    tasks = payload.get("tasks", {})
    parsed_tasks: dict[str, Any] = {}
    for col, rec in tasks.items():
        if not isinstance(rec, dict):
            continue
        out_dir = pathlib.Path(rec.get("output_dir", ""))
        report = None
        pred = None
        if out_dir.exists():
            if col == "Reading":
                report = latest_file(out_dir, "report.txt")
                pred = latest_file(out_dir, "predictions.json")
                parsed = parse_reading_report(report)
            else:
                report = latest_file(out_dir, "best_temperature_report.txt") or latest_file(out_dir, "*.txt")
                pred = latest_file(out_dir, "predictions.json")
                parsed = parse_report(report)
        else:
            parsed = {"exists": False, "path": None}
        parsed_tasks[col] = {
            "score": rec.get("score"),
            "scores": rec.get("scores"),
            "returncode": rec.get("returncode"),
            "elapsed_sec": rec.get("elapsed_sec"),
            "output_dir": str(out_dir),
            "report": parsed,
            "prediction_counts": read_prediction_counts(pred),
        }
    return {
        "payload_path": str(path),
        "model_path": payload.get("model_path"),
        "scores": extract_scores(payload),
        "tasks": parsed_tasks,
        "run_summary": payload.get("run_summary"),
    }


def safe_round(v: Any, ndigits: int = 6) -> Any:
    if isinstance(v, (int, float)) and math.isfinite(float(v)):
        return round(float(v), ndigits)
    return v


def diff_scores(a: dict[str, Any], b: dict[str, Any]) -> dict[str, float]:
    out: dict[str, float] = {}
    for k in PRIMARY_COLUMNS:
        av, bv = a.get(k), b.get(k)
        if av is not None and bv is not None:
            out[k] = round(float(av) - float(bv), 6)
    return out


def section_deltas(targets: dict[str, Any], a: str, b: str, col: str, section: str) -> dict[str, Any]:
    ar = targets[a]["tasks"].get(col, {}).get("report", {}).get("sections", {}).get(section, {})
    br = targets[b]["tasks"].get(col, {}).get("report", {}).get("sections", {}).get(section, {})
    counts = targets[a]["tasks"].get(col, {}).get("prediction_counts", {}).get("counts", {})
    total = targets[a]["tasks"].get(col, {}).get("prediction_counts", {}).get("total", 0)
    items = []
    for key in sorted(set(ar) & set(br)):
        delta = float(ar[key]) - float(br[key])
        n = counts.get(key)
        weighted = (delta * n / total) if (n is not None and total) else None
        items.append({
            "key": key,
            "target_value": ar[key],
            "baseline_value": br[key],
            "delta": round(delta, 6),
            "n_predictions": n,
            "weighted_column_point_delta_if_example_weighted": None if weighted is None else round(weighted, 6),
        })
    pos = sorted(items, key=lambda x: (x["delta"], x.get("weighted_column_point_delta_if_example_weighted") or 0.0), reverse=True)
    neg = sorted(items, key=lambda x: (x["delta"], x.get("weighted_column_point_delta_if_example_weighted") or 0.0))
    return {
        "section": section,
        "n_items": len(items),
        "top_positive": pos[:15],
        "top_negative": neg[:15],
        "all_items": items,
    }


def task_deltas(targets: dict[str, Any], a: str, b: str) -> dict[str, Any]:
    out: dict[str, Any] = {"score_delta": diff_scores(targets[a]["scores"], targets[b]["scores"]), "sections": {}}
    for col in TASK_OVERALL_KEYS:
        if col == "Reading":
            # Reading has only two explicit sub-scores in reports.
            ascores = targets[a]["tasks"].get(col, {}).get("report", {}).get("scores", {})
            bscores = targets[b]["tasks"].get(col, {}).get("report", {}).get("scores", {})
            out["sections"][col] = {
                "READING_SUBSCORES": [
                    {"key": k, "target_value": ascores.get(k), "baseline_value": bscores.get(k), "delta": None if ascores.get(k) is None or bscores.get(k) is None else round(float(ascores[k]) - float(bscores[k]), 6)}
                    for k in ["Reading_eye", "Reading_self_paced", "Reading"]
                ]
            }
            continue
        common_sections: dict[str, Any] = {}
        for section in REPORT_SECTIONS:
            d = section_deltas(targets, a, b, col, section)
            if d["n_items"]:
                common_sections[section] = d
        if common_sections:
            out["sections"][col] = common_sections
    return out


def compact_rows_for_note(comp: dict[str, Any], col: str, section: str, limit: int = 5) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    sec = comp.get("sections", {}).get(col, {}).get(section)
    if not sec:
        return [], []
    keep = ["key", "target_value", "baseline_value", "delta", "n_predictions", "weighted_column_point_delta_if_example_weighted"]
    pos = [{k: x.get(k) for k in keep} for x in sec.get("top_positive", [])[:limit]]
    neg = [{k: x.get(k) for k in keep} for x in sec.get("top_negative", [])[:limit]]
    return pos, neg


def count_flips(sec: dict[str, Any]) -> dict[str, int]:
    items = sec.get("all_items", [])
    return {
        "count_items": len(items),
        "positive_items": sum(1 for x in items if x.get("delta", 0) > 0),
        "negative_items": sum(1 for x in items if x.get("delta", 0) < 0),
        "unchanged_items": sum(1 for x in items if x.get("delta", 0) == 0),
        "zero_to_hundred": sum(1 for x in items if x.get("baseline_value") == 0.0 and x.get("target_value") == 100.0),
        "hundred_to_zero": sum(1 for x in items if x.get("baseline_value") == 100.0 and x.get("target_value") == 0.0),
    }


def build_interpretation(targets: dict[str, Any], comparisons: dict[str, Any]) -> dict[str, Any]:
    reinvest = targets["compact_view_reinvest"]["scores"]
    need_sg_aoa_418 = 9 * 41.8 - sum(float(reinvest[k]) for k in ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA_mean", "Reading"])
    need_sg_aoa_420 = 9 * 42.0 - sum(float(reinvest[k]) for k in ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA_mean", "Reading"])
    vcr = comparisons["compact_view_core_minus_repeat_core"]["score_delta"]
    rvc = comparisons["compact_view_reinvest_minus_view_core"]["score_delta"]
    # Compact textual reading of the largest load-bearing movements.
    return {
        "sota_arithmetic_from_fast_reinvest": {
            "fast_seven_sum": safe_round(sum(float(reinvest[k]) for k in ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA_mean", "Reading"])),
            "superglue_plus_aoa_needed_for_41p8": safe_round(need_sg_aoa_418),
            "superglue_plus_aoa_needed_for_42p0": safe_round(need_sg_aoa_420),
        },
        "core_view_effect_reading": {
            "component_delta": vcr,
            "short_text": "Core compact views improve the same-source repeat endpoint mainly through Supplement, EWoK, Entity, COMPS, GlobalPIQA_mean, and Reading while slightly reducing BLiMP; the fast evidence supports a real source-own-compact-view learning effect, not only endpoint noise, but it does not separate semantic correspondence from lexical-token geometry.",
        },
        "reinvestment_effect_reading": {
            "component_delta": rvc,
            "short_text": "Reinvestment turns the saved compact-view words into more source-view pairs and raises EWoK, Entity, Supplement, and GlobalPIQA_mean over compact_view_core while slightly reducing BLiMP, COMPS, and Reading; the future full result should be read as a trade between stronger lexical/world-relation exposure and possible syntactic/procedural dilution.",
        },
    }


def write_note(payload: dict[str, Any]) -> None:
    targets = payload["targets"]
    comparisons = payload["comparisons"]
    lines: list[str] = []
    lines.append("# research compact-density subtask deltas")
    lines.append("")
    lines.append("This CPU-only pass parses finished A02 fast no-AoA reports for `compact_repeat_core`, `compact_view_core`, and `compact_view_reinvest`. It does not read or modify the running research full-evaluation or seed43122 outputs.")
    lines.append("")
    lines.append("## Fast component surface")
    lines.append("")
    cols = ["BLiMP", "Supplement", "EWoK", "Entity", "Entity_full", "COMPS", "GlobalPIQA_mean", "Reading", "equal7_mean", "equal7_full_entity"]
    lines.append("| target | " + " | ".join(cols) + " |")
    lines.append("|---|" + "---:|" * len(cols))
    for name in ["compact_repeat_core", "compact_view_core", "compact_view_reinvest"]:
        s = targets[name]["scores"]
        vals = []
        for c in cols:
            v = s.get(c)
            vals.append("" if v is None else f"{float(v):.3f}")
        lines.append(f"| {name} | " + " | ".join(vals) + " |")
    lines.append("")
    lines.append("## Component deltas")
    lines.append("")
    for cname in ["compact_view_core_minus_repeat_core", "compact_view_reinvest_minus_view_core", "compact_view_reinvest_minus_repeat_core", "compact_view_reinvest_minus_visible_leader_fast", "compact_view_reinvest_minus_compact_experience_clean_fast"]:
        lines.append(f"- `{cname}`: " + json.dumps(comparisons[cname]["score_delta"] if cname in comparisons and "score_delta" in comparisons[cname] else comparisons[cname], ensure_ascii=False))
    lines.append("")
    interp = payload["interpretation"]
    lines.append("## SOTA arithmetic from the fast reinvest surface")
    lines.append("")
    lines.append(json.dumps(interp["sota_arithmetic_from_fast_reinvest"], indent=2, ensure_ascii=False))
    lines.append("")
    lines.append("## Largest subtask movements")
    lines.append("")
    for cname, title in [("compact_view_core_minus_repeat_core", "Core compact view minus same-source repeat"), ("compact_view_reinvest_minus_view_core", "Reinvest minus core view")]:
        comp = comparisons[cname]
        lines.append(f"### {title}")
        lines.append("")
        for col, section in [
            ("BLiMP", "LINGUISTICS_TERM ACCURACY"),
            ("BLiMP", "UID ACCURACY"),
            ("Supplement", "UID ACCURACY"),
            ("EWoK", "UID ACCURACY"),
            ("Entity_full", "UID ACCURACY"),
            ("COMPS", "UID ACCURACY"),
        ]:
            pos, neg = compact_rows_for_note(comp, col, section, limit=5)
            if not pos and not neg:
                continue
            lines.append(f"**{col} / {section}**")
            lines.append("")
            lines.append("Top positive:")
            for x in pos:
                lines.append(f"- {x['key']}: {x['baseline_value']:.2f} -> {x['target_value']:.2f} (delta {x['delta']:+.2f}, n={x.get('n_predictions')})")
            lines.append("Top negative:")
            for x in neg:
                lines.append(f"- {x['key']}: {x['baseline_value']:.2f} -> {x['target_value']:.2f} (delta {x['delta']:+.2f}, n={x.get('n_predictions')})")
            lines.append("")
        # GlobalPIQA has one-example UIDs; summarize flips rather than listing a long table.
        for col in ["GlobalPIQA_parallel", "GlobalPIQA_nonparallel"]:
            sec = comp.get("sections", {}).get(col, {}).get("UID ACCURACY")
            if sec:
                lines.append(f"**{col} UID movement counts**: " + json.dumps(count_flips(sec), ensure_ascii=False))
        rsec = comp.get("sections", {}).get("Reading", {}).get("READING_SUBSCORES", [])
        if rsec:
            lines.append("**Reading subscores**: " + json.dumps(rsec, ensure_ascii=False))
        lines.append("")
    lines.append("## Mechanistic reading")
    lines.append("")
    lines.append("- " + interp["core_view_effect_reading"]["short_text"])
    lines.append("- " + interp["reinvestment_effect_reading"]["short_text"])
    lines.append("- The weak area remains GlobalPIQA: reinvest improves the fast mean over core but is still about four points under the public 41.8 leader on that column. Any future data repair should increase practical affordance and causal-procedural coverage without sacrificing the high Supplement/Entity/EWoK surface or SuperGLUE/AoA.")
    lines.append("")
    lines.append(f"JSON: `{OUT_JSON}`")
    OUT_NOTE.parent.mkdir(parents=True, exist_ok=True)
    OUT_NOTE.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    targets = {name: load_target(name, path) for name, path in PAYLOADS.items()}
    comparisons: dict[str, Any] = {
        "compact_view_core_minus_repeat_core": task_deltas(targets, "compact_view_core", "compact_repeat_core"),
        "compact_view_reinvest_minus_view_core": task_deltas(targets, "compact_view_reinvest", "compact_view_core"),
        "compact_view_reinvest_minus_repeat_core": task_deltas(targets, "compact_view_reinvest", "compact_repeat_core"),
        "compact_view_reinvest_minus_visible_leader_fast": diff_scores(targets["compact_view_reinvest"]["scores"], LEADER_FAST),
        "compact_view_reinvest_minus_compact_experience_clean_fast": diff_scores(targets["compact_view_reinvest"]["scores"], COMPACT_EXPERIENCE_CLEAN_FAST),
    }
    payload = {
        "status": "COMPACT_DENSITY_SUBTASK_DELTA",
        "inputs": {k: str(v) for k, v in PAYLOADS.items()},
        "targets": targets,
        "comparisons": comparisons,
        "interpretation": build_interpretation(targets, comparisons),
        "note": str(OUT_NOTE),
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_note(payload)
    print(json.dumps({
        "status": payload["status"],
        "json": str(OUT_JSON),
        "note": str(OUT_NOTE),
        "reinvest_equal7": targets["compact_view_reinvest"]["scores"].get("equal7_mean"),
        "need_sg_aoa_41p8": payload["interpretation"]["sota_arithmetic_from_fast_reinvest"]["superglue_plus_aoa_needed_for_41p8"],
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
