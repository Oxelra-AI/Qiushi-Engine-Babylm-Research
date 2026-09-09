#!/usr/bin/env python3
"""research: localize seed43122 official-coordinate failure relative to seed43022.

Reads the two staged official-coordinate collated JSON files and official data, then
computes per-subtask accuracy/delta for all zero-shot families plus SuperGLUE and
Reading scalar details. This is CPU-only and uses no partial managed outputs.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import collections
import csv
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any, Iterable

import pandas as pd
import statsmodels.formula.api as smf


def find_user_root() -> Path:
    return _PUBLIC_ROOT


ROOT = find_user_root()
WORKSPACE = ROOT / "experiments/archive/representation_and_objectives"
PRISTINE_STRICT = WORKSPACE / "data/pristine_official_coordinate/babylm-eval/strict"
FULL_DATA = PRISTINE_STRICT / "evaluation_data/full_eval"
LOCAL_STRICT_WITH_GLOBALPIQA = ROOT / "experiments/archive/initial_model_studies/repos/babylm-eval/strict"
SEED430_COLLATED = WORKSPACE / "data/pristine_collate_seed43022/results/hf_model/all_full_preds_and_fast_scores_mlm.json"
SEED431_COLLATED = WORKSPACE / "data/pristine_collate_seed43122/results/hf_model/all_full_preds_and_fast_scores_mlm.json"
SEED430_SUMMARY = WORKSPACE / "data/pristine_collate_seed43022/pristine_collate_seed43022_summary.json"
SEED431_SUMMARY = WORKSPACE / "data/pristine_collate_seed43122/pristine_collate_seed43122_summary.json"
COMPARE_JSON = WORKSPACE / "data/seed43122_official_comparison_after_delivery/seed43122_official_comparison_after_delivery.json"
OUT_DIR = WORKSPACE / "data/seed43122_official_failure_localization"
NOTE = WORKSPACE / "notes/seed43122_official_failure_localization.md"

SUPERGLUE = ["boolq", "mnli", "mrpc", "multirc", "qqp", "rte", "wsc"]
FINETUNE_METRIC = {"boolq": "accuracy", "mnli": "accuracy", "mrpc": "f1", "multirc": "accuracy", "qqp": "f1", "rte": "accuracy", "wsc": "accuracy"}


def now_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sha256_file(path: Path) -> str | None:
    if not path.exists():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    return {"path": str(path), "exists": path.exists(), "size_bytes": path.stat().st_size if path.exists() else None, "sha256": sha256_file(path)}


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def safe_get(obj: Any, dotted: str, default: Any = None) -> Any:
    cur = obj
    for p in dotted.split("."):
        if isinstance(cur, dict) and p in cur:
            cur = cur[p]
        else:
            return default
    return cur


def normalize_blimp_name(name: str) -> str:
    return name.replace("_", "_")


def score_blimp_like(block430: dict[str, Any], block431: dict[str, Any], data_dir: Path, family: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for subtask in sorted(block430.keys()):
        p = (data_dir / subtask).with_suffix(".jsonl")
        preds430 = block430[subtask]["predictions"]
        preds431 = block431[subtask]["predictions"]
        data_lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
        if len(preds430) != len(data_lines) or len(preds431) != len(data_lines):
            raise RuntimeError(f"{family}/{subtask} length mismatch: {len(preds430)} {len(preds431)} data {len(data_lines)}")
        c430 = c431 = both = regressions = recoveries = 0
        for r430, r431, line in zip(preds430, preds431, data_lines):
            ex = json.loads(line)
            good = ex["sentence_good"].strip()
            ok430 = r430["pred"].strip() == good
            ok431 = r431["pred"].strip() == good
            c430 += ok430
            c431 += ok431
            both += ok430 and ok431
            regressions += ok430 and not ok431
            recoveries += (not ok430) and ok431
        s430 = 100.0 * c430 / len(data_lines)
        s431 = 100.0 * c431 / len(data_lines)
        rows.append({
            "family": family, "subtask": subtask, "total": len(data_lines),
            "score430": s430, "score431": s431, "delta431_minus_430": s431 - s430,
            "correct430": c430, "correct431": c431, "both_correct": both,
            "regressions_430_correct_431_wrong": regressions,
            "recoveries_430_wrong_431_correct": recoveries,
            "net_correct_delta": c431 - c430,
        })
    return rows


def score_ewok(block430: dict[str, Any], block431: dict[str, Any], data_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for subtask in sorted(block430.keys()):
        p = (data_dir / subtask).with_suffix(".jsonl")
        preds430 = block430[subtask]["predictions"]
        preds431 = block431[subtask]["predictions"]
        data_lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
        if len(preds430) != len(data_lines) or len(preds431) != len(data_lines):
            raise RuntimeError(f"EWoK/{subtask} length mismatch")
        c430 = c431 = both = regressions = recoveries = 0
        for r430, r431, line in zip(preds430, preds431, data_lines):
            ex = json.loads(line)
            good = " ".join([ex["Context1"], ex["Target1"]]).strip()
            ok430 = r430["pred"].strip() == good
            ok431 = r431["pred"].strip() == good
            c430 += ok430
            c431 += ok431
            both += ok430 and ok431
            regressions += ok430 and not ok431
            recoveries += (not ok430) and ok431
        s430 = 100.0 * c430 / len(data_lines)
        s431 = 100.0 * c431 / len(data_lines)
        rows.append({
            "family": "EWoK", "subtask": subtask, "total": len(data_lines),
            "score430": s430, "score431": s431, "delta431_minus_430": s431 - s430,
            "correct430": c430, "correct431": c431, "both_correct": both,
            "regressions_430_correct_431_wrong": regressions,
            "recoveries_430_wrong_431_correct": recoveries,
            "net_correct_delta": c431 - c430,
        })
    return rows


def load_entity_targets(data_dir: Path) -> dict[str, list[dict[str, Any]]]:
    subtask_to_targets: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for data_path in sorted(data_dir.glob("*.jsonl")):
        with data_path.open("r", encoding="utf-8", errors="replace") as f:
            for line in f:
                if not line.strip():
                    continue
                ex = json.loads(line)
                # Matches research/research official-coordinate scorer repair: skip `nothing` examples.
                if any("nothing" in option for option in ex["options"]):
                    continue
                subtask_to_targets[f'{data_path.stem}_{ex["numops"]}_ops'].append(ex)
    return subtask_to_targets


def score_entity(block430: dict[str, Any], block431: dict[str, Any], data_dir: Path) -> list[dict[str, Any]]:
    targets = load_entity_targets(data_dir)
    rows: list[dict[str, Any]] = []
    for subtask in sorted(block430.keys()):
        preds430 = block430[subtask]["predictions"]
        preds431 = block431[subtask]["predictions"]
        t = targets[subtask]
        if len(preds430) != len(t) or len(preds431) != len(t):
            raise RuntimeError(f"Entity/{subtask} length mismatch: {len(preds430)} {len(preds431)} targets {len(t)}")
        c430 = c431 = both = regressions = recoveries = 0
        for r430, r431, ex in zip(preds430, preds431, t):
            good = ex["options"][0].strip()
            ok430 = r430["pred"].strip() == good
            ok431 = r431["pred"].strip() == good
            c430 += ok430
            c431 += ok431
            both += ok430 and ok431
            regressions += ok430 and not ok431
            recoveries += (not ok430) and ok431
        s430 = 100.0 * c430 / len(t)
        s431 = 100.0 * c431 / len(t)
        rows.append({
            "family": "Entity", "subtask": subtask, "total": len(t),
            "score430": s430, "score431": s431, "delta431_minus_430": s431 - s430,
            "correct430": c430, "correct431": c431, "both_correct": both,
            "regressions_430_correct_431_wrong": regressions,
            "recoveries_430_wrong_431_correct": recoveries,
            "net_correct_delta": c431 - c430,
        })
    return rows


def score_comps(block430: dict[str, Any], block431: dict[str, Any], data_dir: Path) -> list[dict[str, Any]]:
    subtask_to_file = {"base": "comps_base", "wugs_dist_before": "comps_wugs_dist-before", "wugs_dist_in_between": "comps_wugs_dist-in-between", "wugs": "comps_wugs"}
    rows: list[dict[str, Any]] = []
    for subtask in sorted(block430.keys()):
        p = (data_dir / subtask_to_file[subtask]).with_suffix(".jsonl")
        preds430 = block430[subtask]["predictions"]
        preds431 = block431[subtask]["predictions"]
        lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
        if len(preds430) != len(lines) or len(preds431) != len(lines):
            raise RuntimeError(f"COMPS/{subtask} length mismatch")
        c430 = c431 = both = regressions = recoveries = 0
        for r430, r431, line in zip(preds430, preds431, lines):
            ex = json.loads(line)
            good = " ".join([ex["prefix_acceptable"], ex["property_phrase"]]).strip()
            ok430 = r430["pred"].strip() == good
            ok431 = r431["pred"].strip() == good
            c430 += ok430
            c431 += ok431
            both += ok430 and ok431
            regressions += ok430 and not ok431
            recoveries += (not ok430) and ok431
        s430 = 100.0 * c430 / len(lines)
        s431 = 100.0 * c431 / len(lines)
        rows.append({
            "family": "COMPS", "subtask": subtask, "total": len(lines),
            "score430": s430, "score431": s431, "delta431_minus_430": s431 - s430,
            "correct430": c430, "correct431": c431, "both_correct": both,
            "regressions_430_correct_431_wrong": regressions,
            "recoveries_430_wrong_431_correct": recoveries,
            "net_correct_delta": c431 - c430,
        })
    return rows


def score_globalpiqa(block430: dict[str, Any], block431: dict[str, Any], data_file: Path, family: str) -> dict[str, Any]:
    # Predictions are keyed by example id in the collated output. The official collate scorer returns one fraction.
    # Here use target/answer fields conservatively by detecting which sentence is correct from the generated jsonl.
    # To avoid making assumptions about exact field names, import the official collator helper for scalar score and
    # also compute item-wise agreement by checking whether the two seeds chose the same pred string.
    sys.path.insert(0, str((PRISTINE_STRICT / "evaluation_pipeline").resolve()))
    import collate_preds as coll  # type: ignore
    frac430 = float(coll._calculate_global_piqa_results(block430, data_file, family)[family])
    frac431 = float(coll._calculate_global_piqa_results(block431, data_file, family)[family])
    keys = sorted(set(block430.keys()) | set(block431.keys()))
    same_pred = 0
    changed = 0
    for k in keys:
        p430 = block430.get(k, {}).get("predictions", [{}])[0].get("pred") if block430.get(k, {}).get("predictions") else None
        p431 = block431.get(k, {}).get("predictions", [{}])[0].get("pred") if block431.get(k, {}).get("predictions") else None
        same_pred += (p430 == p431)
        changed += (p430 != p431)
    return {
        "family": family, "subtask": family, "total": len(keys),
        "score430": 100.0 * frac430, "score431": 100.0 * frac431,
        "delta431_minus_430": 100.0 * (frac431 - frac430),
        "same_pred_count": same_pred, "changed_pred_count": changed,
        "net_correct_delta": round(len(keys) * (frac431 - frac430)),
    }


def parse_results_txt(path: Path, metric: str) -> float:
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        k, _, v = line.partition(":")
        if k.strip() == metric:
            return float(v.strip()) * 100.0
    raise ValueError(f"metric {metric} not found in {path}")


def superglue_rows(summary430: dict[str, Any], summary431: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    d430 = safe_get(summary430, "score_summary.official_overall.details.SuperGLUE") or safe_get(summary430, "score_summary.details.SuperGLUE")
    d431 = safe_get(summary431, "score_summary.details.SuperGLUE")
    for task in SUPERGLUE:
        metric = FINETUNE_METRIC[task]
        score430 = float(d430[task]["score"])
        score431 = float(d431[task]["score"])
        rows.append({
            "family": "SuperGLUE", "subtask": task, "metric": metric,
            "score430": score430, "score431": score431,
            "delta431_minus_430": score431 - score430,
            "results430": d430[task]["results_txt"], "results431": d431[task]["results_txt"],
        })
    return rows


def reading_details(collated: dict[str, Any], data_path: Path) -> dict[str, float]:
    data = pd.read_csv(data_path, dtype={"item": str})
    preds = [item["pred"] for item in collated["reading"]["reading"]["predictions"]]
    prev_preds = [item["prev_pred"] for item in collated["reading"]["reading"]["predictions"]]
    data = data.copy()
    data["pred"] = preds
    data["prev_pred"] = prev_preds
    eye_tracking_vars = ["RTfirstfix", "RTfirstpass", "RTgopast", "RTrightbound"]
    detail: dict[str, float] = {}
    vals = []
    for dv in eye_tracking_vars:
        temp = data[[dv, "Subtlex_log10", "length", "context_length"]].dropna()
        b = smf.ols(formula=dv + " ~ Subtlex_log10 + length + context_length + Subtlex_log10:length + Subtlex_log10:context_length + length:context_length", data=temp).fit().rsquared
        temp = data[[dv, "Subtlex_log10", "length", "context_length", "pred"]].dropna()
        m = smf.ols(formula=dv + " ~ Subtlex_log10 + length + context_length + Subtlex_log10:length + Subtlex_log10:context_length + length:context_length + pred", data=temp).fit().rsquared
        val = ((float(m) - float(b)) / (1 - float(b))) * 100.0
        detail[dv] = val
        vals.append(val)
    detail["eye_tracking_mean"] = sum(vals) / len(vals)
    temp = data[["self_paced_reading_time", "Subtlex_log10", "length", "context_length", "prev_length", "prev_pred"]].dropna()
    b = smf.ols(formula="self_paced_reading_time ~ Subtlex_log10 + length + context_length + prev_length + prev_pred + Subtlex_log10:length + Subtlex_log10:context_length + Subtlex_log10:prev_length + Subtlex_log10:prev_pred + length:context_length + length:prev_length + length:prev_pred + context_length:prev_length + context_length:prev_pred + prev_length:prev_pred", data=temp).fit().rsquared
    temp = data[["self_paced_reading_time", "Subtlex_log10", "length", "context_length", "prev_length", "prev_pred", "pred"]].dropna()
    m = smf.ols(formula="self_paced_reading_time ~ Subtlex_log10 + length + context_length + prev_length + prev_pred + Subtlex_log10:length + Subtlex_log10:context_length + Subtlex_log10:prev_length + Subtlex_log10:prev_pred + length:context_length + length:prev_length + length:prev_pred + context_length:prev_length + context_length:prev_pred + prev_length:prev_pred + pred", data=temp).fit().rsquared
    detail["self_paced_reading"] = ((float(m) - float(b)) / (1 - float(b))) * 100.0
    detail["leaderboard_reading"] = (detail["eye_tracking_mean"] + detail["self_paced_reading"]) / 2.0
    return detail


def write_rows(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    rows = list(rows)
    keys: list[str] = []
    for r in rows:
        for k in r.keys():
            if k not in keys:
                keys.append(k)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    NOTE.parent.mkdir(parents=True, exist_ok=True)
    c430 = load_json(SEED430_COLLATED)
    c431 = load_json(SEED431_COLLATED)
    s430 = load_json(SEED430_SUMMARY)
    s431 = load_json(SEED431_SUMMARY)
    compare = load_json(COMPARE_JSON)

    rows: list[dict[str, Any]] = []
    rows += score_blimp_like(c430["blimp"], c431["blimp"], FULL_DATA / "blimp_filtered", "BLiMP")
    rows += score_blimp_like(c430["blimp_supplement"], c431["blimp_supplement"], FULL_DATA / "supplement_filtered", "Supplement")
    rows += score_ewok(c430["ewok"], c431["ewok"], FULL_DATA / "ewok_filtered")
    rows += score_entity(c430["entity_tracking_filtered"], c431["entity_tracking_filtered"], FULL_DATA / "entity_tracking")
    rows += score_comps(c430["comps"], c431["comps"], FULL_DATA / "comps")
    rows.append(score_globalpiqa(c430["global_piqa_parallel"], c431["global_piqa_parallel"], LOCAL_STRICT_WITH_GLOBALPIQA / "evaluation_data/full_eval/global_piqa_parallel/eng_latn.jsonl", "global_piqa_parallel"))
    rows.append(score_globalpiqa(c430["global_piqa_nonparallel"], c431["global_piqa_nonparallel"], LOCAL_STRICT_WITH_GLOBALPIQA / "evaluation_data/full_eval/global_piqa_nonparallel/eng_latn.jsonl", "global_piqa_nonparallel"))
    rows += superglue_rows(s430, s431)

    # For weighted/scalar inspection, sort by raw delta and by contribution to the official mean within each family.
    family_counts = collections.Counter(r["family"] for r in rows)
    for r in rows:
        if "delta431_minus_430" in r:
            r["within_family_equal_subtask_contribution"] = float(r["delta431_minus_430"]) / max(1, family_counts[r["family"]])
            # Official Overall averages columns, then family subtasks where applicable. Column contribution is /9.
            if r["family"] in {"BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "SuperGLUE"}:
                r["approx_overall_contribution"] = float(r["delta431_minus_430"]) / max(1, family_counts[r["family"]]) / 9.0
            elif r["family"].startswith("global_piqa"):
                r["approx_overall_contribution"] = float(r["delta431_minus_430"]) / 2.0 / 9.0

    reading430 = reading_details(c430, FULL_DATA / "reading/reading_data.csv")
    reading431 = reading_details(c431, FULL_DATA / "reading/reading_data.csv")
    reading_delta = {k: reading431[k] - reading430[k] for k in sorted(reading430)}

    worst_by_delta = sorted(rows, key=lambda r: float(r.get("delta431_minus_430", 0.0)))[:35]
    worst_by_overall_contrib = sorted(rows, key=lambda r: float(r.get("approx_overall_contribution", 0.0)))[:35]
    best_by_delta = sorted(rows, key=lambda r: float(r.get("delta431_minus_430", 0.0)), reverse=True)[:20]

    family_summary = []
    for family in sorted(set(r["family"] for r in rows)):
        fr = [r for r in rows if r["family"] == family]
        family_summary.append({
            "family": family,
            "num_subtasks": len(fr),
            "mean_delta": sum(float(r.get("delta431_minus_430", 0.0)) for r in fr) / len(fr),
            "sum_net_correct_delta_where_available": sum(int(r.get("net_correct_delta", 0) or 0) for r in fr),
            "worst_subtasks": [(r["subtask"], r.get("delta431_minus_430")) for r in sorted(fr, key=lambda x: float(x.get("delta431_minus_430", 0.0)))[:5]],
            "best_subtasks": [(r["subtask"], r.get("delta431_minus_430")) for r in sorted(fr, key=lambda x: float(x.get("delta431_minus_430", 0.0)), reverse=True)[:5]],
        })

    out_csv = OUT_DIR / "seed43122_minus_seed43022_official_subtask_deltas.csv"
    write_rows(out_csv, rows)

    payload = {
        "status": "SEED43122_OFFICIAL_FAILURE_LOCALIZATION",
        "created_utc": now_utc(),
        "scientific_purpose": "Localize why independent seed43122 falls below seed43022 and the visible 41.8 leader on the same official coordinate, before any new expensive branch.",
        "inputs": {
            "seed430_collated": file_record(SEED430_COLLATED),
            "seed431_collated": file_record(SEED431_COLLATED),
            "seed430_summary": file_record(SEED430_SUMMARY),
            "seed431_summary": file_record(SEED431_SUMMARY),
            "comparison": file_record(COMPARE_JSON),
        },
        "official_comparison": safe_get(compare, "comparison", {}),
        "reading_details": {"seed43022": reading430, "seed43122": reading431, "delta431_minus_430": reading_delta},
        "family_summary": family_summary,
        "worst_by_delta": worst_by_delta,
        "worst_by_approx_overall_contribution": worst_by_overall_contrib,
        "best_by_delta": best_by_delta,
        "csv": str(out_csv),
        "interpretation": {
            "seed43122_is_not_submission_robust_sota": True,
            "primary_pattern": "Seed43122 is lower across nearly all capability columns except Reading; AoA remains 0.0. This is not an EWoK/AoA coordinate artifact after research staging.",
            "next_research_need": "Separate pretraining-RNG sensitivity from checkpoint/fine-tuning randomness using cheaper evidence before launching corpus/recipe variants; any repair should target broad representation stability rather than a single benchmark patch.",
        },
    }
    out_json = OUT_DIR / "seed43122_official_failure_localization.json"
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    lines = [
        "# research — seed43122 official-coordinate failure localization",
        "",
        f"Seed43122 staged official Overall: `{safe_get(compare, 'comparison.seed43122_official_overall')}`; margin over 41.8: `{safe_get(compare, 'comparison.seed43122_margin_over_visible_leader_41p8')}`.",
        f"Seed43122 minus seed43022 Overall: `{safe_get(compare, 'comparison.seed43122_minus_seed43022_overall')}`; NLP average delta: `{safe_get(compare, 'comparison.seed43122_minus_seed43022_nlp_average')}`.",
        "",
        "Column deltas (seed43122 - seed43022) show broad weakening except Reading; AoA remains equal at 0.0.",
        "",
        "Worst family summaries:",
    ]
    for fs in sorted(family_summary, key=lambda x: x["mean_delta"])[:8]:
        lines.append(f"- {fs['family']}: mean_delta `{fs['mean_delta']}`; worst {fs['worst_subtasks'][:3]}")
    lines += ["", "Largest raw subtask losses:"]
    for r in worst_by_delta[:12]:
        lines.append(f"- {r['family']}/{r['subtask']}: delta `{r.get('delta431_minus_430')}`, net_correct_delta `{r.get('net_correct_delta')}`")
    lines += [
        "",
        "Reading details (positive for seed43122) are preserved separately because they do not offset the broad NLP-column loss enough to clear the visible leader.",
        f"CSV: `{out_csv}`",
        f"JSON: `{out_json}`",
    ]
    NOTE.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": payload["status"],
        "out_json": str(out_json),
        "csv": str(out_csv),
        "note": str(NOTE),
        "overall431": safe_get(compare, "comparison.seed43122_official_overall"),
        "margin431": safe_get(compare, "comparison.seed43122_margin_over_visible_leader_41p8"),
        "worst_families": sorted([(fs["family"], fs["mean_delta"]) for fs in family_summary], key=lambda x: x[1])[:6],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
