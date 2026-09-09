#!/usr/bin/env python3
"""research helper: read research strict innovation scores after managed evaluation completes.

This script is intentionally read-only with respect to training/evaluation outputs. It
turns the 70M/80M cheap-column trajectory plus training mask statistics into a
research-facing interpretation that separates: (i) continuing the launched
p_strict/p_copy probability route, (ii) closing that route while keeping the
exact-swap variant alive, and (iii) weakening the whole innovation-masking idea.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import json
import math
import pathlib
import time
from typing import Any


def find_user_root() -> pathlib.Path:
    return _PUBLIC_ROOT


ROOT = find_user_root()
WS = ROOT / "experiments/archive/frontier_consolidation"
TRAJECTORY = WS / "data/strict_innovation_trajectory/strict_innovation_vs_trajectory.json"
TRAIN_METRICS = WS / "training/runs/strict_content_innovation_wwm_reinvest_seed43022_80M/scientific_metrics.json"
WAIT_SUMMARY = WS / "data/strict_innovation_70_80M_eval/strict_innovation_wait_eval_70_80M_summary.json"
INTERVENTION_COMPARISON = WS / "data/innovation_intervention_comparison/innovation_intervention_comparison.json"
OUT_DIR = WS / "data/strict_innovation_result_reading"
OUT_JSON = OUT_DIR / "strict_innovation_result_reading.json"
OUT_MD = OUT_DIR / "strict_innovation_result_reading.md"

CORE_COLUMNS = ["BLiMP", "Supplement", "EWoK"]
ALL_COLUMNS = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA", "Reading"]


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def load_json(path: pathlib.Path) -> Any | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def f(x: Any) -> float | None:
    if x is None:
        return None
    try:
        y = float(x)
    except Exception:
        return None
    return y if math.isfinite(y) else None


def fmt(x: Any) -> str:
    y = f(x)
    return "NA" if y is None else f"{y:.4f}"


def summarize_row(row: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {"exposure_m": row.get("exposure_m"), "complete": bool(row.get("complete"))}
    out["delta_mean7"] = f(row.get("delta_strict_minus_step35_mean7"))
    out["delta_core_mean"] = None
    core_vals = [f(row.get(f"delta_strict_minus_step35_{c}")) for c in CORE_COLUMNS]
    if all(v is not None for v in core_vals):
        out["delta_core_mean"] = sum(core_vals) / len(core_vals)  # type: ignore[arg-type]
    out["deltas"] = {c: f(row.get(f"delta_strict_minus_step35_{c}")) for c in ALL_COLUMNS}
    out["n_positive_all7"] = sum(1 for v in out["deltas"].values() if v is not None and v > 0)
    out["n_positive_core3"] = sum(1 for c in CORE_COLUMNS if out["deltas"].get(c) is not None and out["deltas"][c] > 0)
    return out


def classify(rows: list[dict[str, Any]], raw_rows: list[dict[str, Any]], metrics: dict[str, Any] | None, comparison: dict[str, Any] | None) -> dict[str, Any]:
    if not rows or any(not r.get("complete") for r in rows):
        return {
            "state": "waiting_for_real_scores",
            "reading": "The research cheap trajectory is incomplete; no scientific score interpretation is made.",
            "next_action": "After managed evaluation writes the 70M/80M per-target JSONs and trajectory file, rerun this reader and inspect the actual deltas.",
        }
    by_m = {int(r["exposure_m"]): summarize_row(r) for r in rows}
    r80 = by_m.get(80) or by_m.get(max(by_m))
    delta_mean7 = f(r80.get("delta_mean7"))
    delta_core = f(r80.get("delta_core_mean"))
    d = r80.get("deltas", {})
    gpiqa = f(d.get("GlobalPIQA"))
    comps = f(d.get("COMPS"))
    blimp = f(d.get("BLiMP"))
    supp = f(d.get("Supplement"))
    ewok = f(d.get("EWoK"))
    positive_core = int(r80.get("n_positive_core3", 0))
    positive_all = int(r80.get("n_positive_all7", 0))

    raw80 = next((x for x in raw_rows if int(x.get("exposure_m", -1)) == int(r80["exposure_m"])), {})
    gp_non = f(raw80.get("delta_strict_minus_step35_GlobalPIQA_nonparallel"))
    gp_par = f(raw80.get("delta_strict_minus_step35_GlobalPIQA_parallel"))

    mask_stats = (metrics or {}).get("cumulative_mask_stats", {}) if isinstance(metrics, dict) else {}
    realized = {
        "strict_selected_groups": mask_stats.get("strict_selected_groups"),
        "strict_available_groups": mask_stats.get("strict_available_groups"),
        "copy_selected_groups": mask_stats.get("copy_selected_groups"),
        "copy_available_groups": mask_stats.get("copy_available_groups"),
        "source_selected_groups": mask_stats.get("source_selected_groups"),
        "source_available_groups": mask_stats.get("source_available_groups"),
        "total_words": (metrics or {}).get("total_words") if isinstance(metrics, dict) else None,
        "checkpoint_count": len((metrics or {}).get("checkpoints", [])) if isinstance(metrics, dict) else None,
    }
    if realized["strict_selected_groups"] is not None and realized["strict_available_groups"]:
        realized["strict_rate"] = float(realized["strict_selected_groups"]) / float(realized["strict_available_groups"])
    if realized["copy_selected_groups"] is not None and realized["copy_available_groups"]:
        realized["copy_rate"] = float(realized["copy_selected_groups"]) / float(realized["copy_available_groups"])

    # The thresholds below are intentionally coarse: they express research meaning,
    # not a formal stopping theorem. They prevent a tiny or redistributed movement
    # from being treated as evidence for an 80M->100M extension.
    if delta_mean7 is not None and delta_core is not None and delta_mean7 >= 0.25 and delta_core > 0 and positive_core >= 2 and positive_all >= 5:
        route_state = "continue_launched_probability_route"
        reading = "The 80M surface is broadly positive versus research, including the BLiMP/Supplement/EWoK core; continue the launched strict probability route toward 100M/full official evaluation before starting another innovation variant."
        next_action = "Launch 100M continuation/full official evaluation for research only after checking training metrics and per-column files; keep exact-swap dormant."
    elif delta_mean7 is not None and delta_mean7 > 0 and positive_all >= 4 and positive_core >= 1:
        route_state = "needs_close_reading_before_continuation"
        reading = "The 80M movement is positive but not decisively broad. Read individual task/subtask changes and, if possible, an innovation conditional probe before spending a 100M/full run."
        next_action = "Inspect per-target outputs and run a CPU conditional-innovation probe on research 70M/80M if the cheap surface suggests a real but unclear mechanism."
    elif (gp_non is not None and gp_non > 4 and (delta_mean7 is None or delta_mean7 <= 0.15)) or (gpiqa is not None and gpiqa > 2.0 and positive_core <= 1):
        route_state = "close_probability_route_as_redistribution"
        reading = "The result resembles the word-mean/minfreq50 redistribution pattern: GlobalPIQA_nonparallel or GlobalPIQA/COMPS moves without broad BLiMP/Supplement/EWoK repair. Close research without 100M continuation."
        next_action = "If innovation masking remains attractive, port the exact-swap variant as a lower-perturbation single-variable test; otherwise return to route selection."
    elif delta_mean7 is not None and (delta_mean7 <= 0 or positive_core == 0):
        route_state = "close_probability_route"
        reading = "The launched p_strict/p_copy intervention does not improve the mature cheap surface. Close this probability route without 100M continuation, second seed, or retune."
        next_action = "Use the pattern of damage to decide whether exact-swap is still scientifically distinct enough to test; if damage is not tied to copyable suppression/pressure, reopen route search."
    else:
        route_state = "ambiguous_requires_mechanism_reading"
        reading = "The cheap surface does not clearly separate broad improvement, redistribution, or failure. Do not launch 100M/full before reading per-task anatomy and the innovation conditional object."
        next_action = "Run the targeted CPU source-conditioned innovation probe and inspect UID/subtask deltas before spending more H100 time."

    return {
        "state": route_state,
        "reading": reading,
        "next_action": next_action,
        "summary_by_exposure": by_m,
        "raw80_globalpiqa_parallel_delta": gp_par,
        "raw80_globalpiqa_nonparallel_delta": gp_non,
        "realized_training_mask_stats": realized,
        "codex_exact_swap_context": (comparison or {}).get("codex_exact_swap", {}).get("projected_per_10m_from_broad_smoke") if isinstance(comparison, dict) else None,
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    traj = load_json(TRAJECTORY)
    metrics = load_json(TRAIN_METRICS)
    wait = load_json(WAIT_SUMMARY)
    comp = load_json(INTERVENTION_COMPARISON)
    missing = [str(p) for p, obj in [(TRAJECTORY, traj), (TRAIN_METRICS, metrics), (WAIT_SUMMARY, wait), (INTERVENTION_COMPARISON, comp)] if obj is None]
    rows = traj.get("rows", []) if isinstance(traj, dict) else []
    raw_rows = traj.get("raw_rows", []) if isinstance(traj, dict) else []
    interpretation = classify(rows, raw_rows, metrics if isinstance(metrics, dict) else None, comp if isinstance(comp, dict) else None)
    result = {
        "status": "STRICT_INNOVATION_RESULT_READING",
        "created_utc": now(),
        "purpose": "Read the research 70M/80M cheap surface after managed evaluation and separate continuation, closure, and exact-swap follow-up interpretations.",
        "inputs": {
            "trajectory": str(TRAJECTORY),
            "training_metrics": str(TRAIN_METRICS),
            "wait_summary": str(WAIT_SUMMARY),
            "intervention_comparison": str(INTERVENTION_COMPARISON),
        },
        "missing_inputs": missing,
        "interpretation": interpretation,
    }
    OUT_JSON.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = [
        "# research strict-innovation result reading",
        "",
        result["purpose"],
        "",
        f"State: `{interpretation['state']}`",
        "",
        interpretation["reading"],
        "",
        f"Next action: {interpretation['next_action']}",
        "",
        "## Missing inputs",
    ]
    lines.extend([f"- `{m}`" for m in missing] if missing else ["- none"])
    if isinstance(interpretation.get("summary_by_exposure"), dict):
        lines += ["", "## Exposure summary", "", "| exposure | Δ mean7 | Δ core(BLiMP/Supp/EWoK) | positive all7 | positive core3 | BLiMP | Supplement | EWoK | GlobalPIQA | COMPS |", "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for m, row in sorted(interpretation["summary_by_exposure"].items(), key=lambda kv: int(kv[0])):
            d = row.get("deltas", {})
            lines.append(f"| {m} | {fmt(row.get('delta_mean7'))} | {fmt(row.get('delta_core_mean'))} | {row.get('n_positive_all7')} | {row.get('n_positive_core3')} | {fmt(d.get('BLiMP'))} | {fmt(d.get('Supplement'))} | {fmt(d.get('EWoK'))} | {fmt(d.get('GlobalPIQA'))} | {fmt(d.get('COMPS'))} |")
    lines += ["", f"Full JSON: `{OUT_JSON}`"]
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "state": interpretation["state"], "missing_inputs": missing, "out_json": str(OUT_JSON), "out_md": str(OUT_MD)}, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
