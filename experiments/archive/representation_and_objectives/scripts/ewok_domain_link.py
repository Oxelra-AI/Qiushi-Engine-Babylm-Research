#!/usr/bin/env python3
"""research: official-surface domain link for interference ladder.

Aggregates already-scored full-EWoK interaction records for representative models,
compares material-dynamics against aggregate EWoK, and relates those official
sub-surfaces to the research action-update ladder (`last_event`) and explicit-
override gap.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import csv
import json
import math
import os
import statistics
import time
from pathlib import Path
from typing import Any

USER_ROOT = _public_path('.')
os.chdir(USER_ROOT)
A01_WS = _public_path('experiments/archive/representation_and_objectives')
OUT_DIR = _public_path('experiments/archive/representation_and_objectives/data/ewok_domain_link')
OUT_JSON = _public_path('experiments/archive/representation_and_objectives/data/ewok_domain_link/ewok_domain_link.json')
OUT_MD = _public_path('research/notes/representation_and_objectives/ewok_domain_link.md')

RECORDS = {
    "legal16k_base_100M_seed43022": _public_path('experiments/archive/representation_and_objectives/data/scale1p75_matched_ewok_interaction/legal16k_100M/ewok_interaction_records.csv'),
    "scale1p75_100M": _public_path('experiments/archive/representation_and_objectives/data/scale1p75_matched_ewok_interaction/scale1p75_100M/ewok_interaction_records.csv'),
    "legal40k_8x480_100M_seed43022": _public_path('experiments/archive/representation_and_objectives/data/full_ewok_interaction_specificity/ewok_full_interaction_specificity_records_fullcpu.csv'),
    "legal40k_depth12_100M_seed43022": _public_path('experiments/archive/representation_and_objectives/data/full_ewok_interaction_specificity/ewok_full_interaction_specificity_records_fullcpu.csv'),
    "fw_compact_100M_seed43022": _public_path('experiments/archive/representation_and_objectives/data/fw_ewok_interaction_reader/fw_compact_fullbatch_seed43022/ewok_interaction_records.csv'),
    "fw_rowblock_100M_seed43022": _public_path('experiments/archive/representation_and_objectives/data/fw_ewok_interaction_reader/fw_breadth_rowblock_fullbatch_seed43022/ewok_interaction_records.csv'),
    "mlm_only_20M": _public_path('experiments/archive/representation_and_objectives/data/full_ewok_coupled_turnover/targets/mlm_only_20M/ewok_interaction_records.csv'),
    "coupled_aligned_20M": _public_path('experiments/archive/representation_and_objectives/data/full_ewok_coupled_turnover/targets/coupled_aligned_20M/ewok_interaction_records.csv'),
    "coupled_shuffled_20M": _public_path('experiments/archive/representation_and_objectives/data/full_ewok_coupled_turnover/targets/coupled_shuffled_20M/ewok_interaction_records.csv'),
}
MODEL_FIELD = {
    "legal40k_8x480_100M_seed43022": "legal40_8x480_43022",
    "legal40k_depth12_100M_seed43022": "legal40_depth_12x384_43022",
}

# From research panel table (crossed success)
LADDER = {
    "legal16k_base_100M_seed43022": {"last_event": 0.9875, "explicit_override": 1.0, "contradict_bare": 0.0},
    "scale1p75_100M": {"last_event": 0.6125, "explicit_override": 1.0, "contradict_bare": 0.0},
    "legal40k_8x480_100M_seed43022": {"last_event": 0.3625, "explicit_override": 1.0, "contradict_bare": 0.0},
    "legal40k_depth12_100M_seed43022": {"last_event": 0.0375, "explicit_override": 0.5, "contradict_bare": 0.0},
    "fw_compact_100M_seed43022": {"last_event": 0.0625, "explicit_override": 0.875, "contradict_bare": 0.0},
    "fw_rowblock_100M_seed43022": {"last_event": 0.1625, "explicit_override": 0.875, "contradict_bare": 0.0},
    "mlm_only_20M": {"last_event": 0.1125, "explicit_override": 0.125, "contradict_bare": 0.0},
    "coupled_aligned_20M": {"last_event": 0.0, "explicit_override": 0.0, "contradict_bare": 0.0},
    "coupled_shuffled_20M": {"last_event": 0.0, "explicit_override": 0.0, "contradict_bare": 0.0},
}


def f(x: Any) -> float:
    try:
        v = float(x)
        return v if math.isfinite(v) else float("nan")
    except Exception:
        return float("nan")


def pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 3:
        return None
    mx = statistics.fmean(xs)
    my = statistics.fmean(ys)
    sx = math.sqrt(sum((x - mx) ** 2 for x in xs) / (len(xs) - 1))
    sy = math.sqrt(sum((y - my) ** 2 for y in ys) / (len(ys) - 1))
    if sx < 1e-12 or sy < 1e-12:
        return None
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / ((len(xs) - 1) * sx * sy)


def read_records(path: Path, target_model: str | None = None) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            if target_model is not None and row.get("target") != target_model:
                continue
            rows.append(row)
    return rows


def summarize(rows: list[dict[str, str]]) -> dict[str, Any]:
    n = len(rows)
    if not n:
        return {"n": 0}
    acc = sum(1 for r in rows if r.get("saved_model_correct_flag") == "True") / n
    stable = sum(1 for r in rows if r.get("conditional_reversal_failure_stable") == "True")
    stable_frac = stable / n
    interactions = [f(r.get("interaction_sum")) for r in rows]
    interactions = [x for x in interactions if math.isfinite(x)]
    return {
        "n": n,
        "accuracy": acc,
        "stable_failures": stable,
        "stable_failure_frac": stable_frac,
        "interaction_mean": statistics.fmean(interactions) if interactions else None,
        "interaction_median": statistics.median(interactions) if interactions else None,
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out: dict[str, Any] = {"created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "targets": {}}
    corr_points = []
    for name, path in RECORDS.items():
        if not path.exists():
            raise FileNotFoundError(f"missing {path}")
        rows = read_records(path, MODEL_FIELD.get(name))
        mat = [r for r in rows if r.get("domain") == "material-dynamics"]
        agent = [r for r in rows if r.get("domain") == "agent-properties"]
        allsum = summarize(rows)
        matsum = summarize(mat)
        agentsum = summarize(agent)
        ladder = LADDER[name]
        entry = {
            "records_path": str(path.relative_to(USER_ROOT)),
            "n_rows": len(rows),
            "all_ewok": allsum,
            "material_dynamics": matsum,
            "agent_properties": agentsum,
            "ladder_last_event": ladder["last_event"],
            "ladder_explicit_override": ladder["explicit_override"],
            "ladder_contradict_bare": ladder["contradict_bare"],
            "action_override_gap": ladder["explicit_override"] - ladder["contradict_bare"],
        }
        out["targets"][name] = entry
        corr_points.append({
            "target": name,
            "ladder_last_event": ladder["last_event"],
            "action_override_gap": entry["action_override_gap"],
            "ewok_accuracy": allsum.get("accuracy"),
            "ewok_stable_frac": allsum.get("stable_failure_frac"),
            "material_accuracy": matsum.get("accuracy"),
            "material_stable_frac": matsum.get("stable_failure_frac"),
            "agent_accuracy": agentsum.get("accuracy"),
        })

    metrics = ["ewok_accuracy", "ewok_stable_frac", "material_accuracy", "material_stable_frac", "agent_accuracy"]
    out["correlations"] = {}
    for metric in metrics:
        valid = [(p["ladder_last_event"], p["action_override_gap"], p[metric])
                 for p in corr_points if p[metric] is not None and math.isfinite(p[metric])]
        if len(valid) < 3:
            out["correlations"][metric] = {"n": len(valid), "last_event_pearson": None, "action_override_gap_pearson": None}
            continue
        xs1 = [v[0] for v in valid]
        xs2 = [v[1] for v in valid]
        ys = [v[2] for v in valid]
        out["correlations"][metric] = {
            "n": len(valid),
            "last_event_pearson": pearson(xs1, ys),
            "action_override_gap_pearson": pearson(xs2, ys),
        }

    OUT_JSON.write_text(json.dumps(out, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "# research EWoK domain link",
        "",
        "Status: **AGGREGATED** from already-scored EWoK records; no new inference.",
        "",
        "| target | ladder last_event | override gap | EWoK acc | EWoK stable frac | material acc | material stable frac | agent acc |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    def fmt_cell(v):
        return "n/a" if v is None or (isinstance(v, float) and not math.isfinite(v)) else f"{v:.4f}"
    for p in corr_points:
        lines.append(
            f"| {p['target']} | {fmt_cell(p['ladder_last_event'])} | {fmt_cell(p['action_override_gap'])} | "
            f"{fmt_cell(p['ewok_accuracy'])} | {fmt_cell(p['ewok_stable_frac'])} | "
            f"{fmt_cell(p['material_accuracy'])} | {fmt_cell(p['material_stable_frac'])} | {fmt_cell(p['agent_accuracy'])} |"
        )
    lines.extend(["", "## Correlations", "", "| metric | n | Pearson(last_event) | Pearson(action override gap) |", "|---|---:|---:|---:|"])
    for m in metrics:
        c = out["correlations"][m]
        def fmt(v):
            return "n/a" if v is None else f"{v:.4f}"
        lines.append(f"| {m} | {c['n']} | {fmt(c['last_event_pearson'])} | {fmt(c['action_override_gap_pearson'])} |")
    lines.extend(["", f"JSON: `{OUT_JSON.relative_to(USER_ROOT)}`"])
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": "EWOK_DOMAIN_LINK_DONE", "out_json": str(OUT_JSON.relative_to(USER_ROOT)), "out_md": str(OUT_MD.relative_to(USER_ROOT))}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
