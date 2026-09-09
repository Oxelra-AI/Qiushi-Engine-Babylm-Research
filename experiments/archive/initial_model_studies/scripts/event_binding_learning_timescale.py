#!/usr/bin/env python3
"""research: audit when event-binding causal use emerges during pretraining.

This is a read-only checkpoint analysis.  It loads exact child checkpoint
directories for two matched WWM seeds, applies the Step302b training-free
diagnostic, and aligns the effect with already-recorded zero-shot trajectories.
The purpose is to determine whether 10M-exposure mechanism screens were too
early and whether the diagnostic tracks BabyLM-facing capability at all.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr


STUDY = _public_path('experiments/archive/initial_model_studies')
PROBE_PATH = _public_path('experiments/archive/initial_model_studies/scripts/revision_302b_event_binding_positive_control.py')
SCORES_PATH = _public_path('experiments/archive/initial_model_studies/data/existing_runs_exposure_trajectory.json')
OUT_DIR = _public_path('experiments/archive/initial_model_studies/data/event_binding_timescale')
OUT = _public_path('experiments/archive/initial_model_studies/data/event_binding_timescale.json')
NOTE = _public_path('research/notes/initial_model_studies/event_binding_timescale.md')

RUNS = {
    "wwm_seed42": _public_path('experiments/archive/initial_model_studies/training/runs/babylm_fullcycle_debertav2_8x480_wwm_seed42_100M_b256/hf_model'),
    "wwm_seed43": _public_path('experiments/archive/initial_model_studies/training/runs/fullcycle_official_wwm_debertav2_8x480_seed43_100M_b256/hf_model'),
}
MILESTONES = [5, 10, 20, 40, 60, 80, 100]


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def effect_vector(rows: list[dict]) -> np.ndarray:
    ordered = sorted(rows, key=lambda row: row["case_index"])
    return np.asarray(
        [row["normalized_extra_entity_effect"] for row in ordered], dtype=np.float64
    )


def safe_spearman(left: list[float], right: list[float]) -> float:
    result = spearmanr(left, right).statistic
    return float(result) if np.isfinite(result) else float("nan")


def main() -> None:
    started = time.time()
    probe = load_module(PROBE_PATH, "step302b_timescale")
    scorer = probe.load_step302_module()
    scorer.setup_environment()
    device = scorer.torch.device("cuda:0" if scorer.torch.cuda.is_available() else "cpu")
    cases = probe.build_cases(420, 3021)
    score_trajectory = json.loads(SCORES_PATH.read_text(encoding="utf-8"))["runs"]
    rng = np.random.default_rng(307042)

    missing = []
    for run_dir in RUNS.values():
        for exposure in MILESTONES:
            path = run_dir / f"chck_{exposure}M"
            if not path.is_dir():
                missing.append(str(path))
    if missing:
        raise RuntimeError(f"missing exact checkpoints: {missing}")

    statistics = {}
    model_identities = {}
    for run_name, run_dir in RUNS.items():
        statistics[run_name] = {}
        model_identities[run_name] = {}
        for exposure in MILESTONES:
            checkpoint = f"chck_{exposure}M"
            path = (run_dir / checkpoint).resolve()
            shard = OUT_DIR / f"{run_name}_{checkpoint}.json"
            print(
                json.dumps(
                    {
                        "event": "score_start",
                        "run": run_name,
                        "checkpoint": checkpoint,
                        "path": str(path),
                    }
                ),
                flush=True,
            )
            scored = scorer.score_model(
                f"{run_name}_{checkpoint}", path, cases, 160, 24, device
            )
            values = effect_vector(scored["rows"])
            stat = scorer.bootstrap_mean(values, rng, 20000)
            statistics[run_name][checkpoint] = stat
            model_identities[run_name][checkpoint] = scored["identity"]
            atomic_json(
                shard,
                {
                    "status": "CHECKPOINT_SHARD_COMPLETE",
                    "run": run_name,
                    "checkpoint": checkpoint,
                    "cases": len(values),
                    "identity": scored["identity"],
                    "statistics": stat,
                    "rows": scored["rows"],
                },
            )
            print(
                json.dumps(
                    {
                        "event": "score_complete",
                        "run": run_name,
                        "checkpoint": checkpoint,
                        "mean": stat["mean"],
                        "ci95": stat["bootstrap_ci95"],
                    }
                ),
                flush=True,
            )

    onset = {}
    for run_name in RUNS:
        positive = [
            exposure
            for exposure in MILESTONES
            if statistics[run_name][f"chck_{exposure}M"]["bootstrap_ci95"][0] > 0
        ]
        onset[run_name] = positive[0] if positive else None

    aligned_rows = []
    for run_name in RUNS:
        for exposure in MILESTONES:
            checkpoint = f"chck_{exposure}M"
            official = score_trajectory[run_name][checkpoint]["scores"]
            aligned_rows.append(
                {
                    "run": run_name,
                    "seed": 42 if run_name.endswith("42") else 43,
                    "exposure_m": exposure,
                    "binding_effect": statistics[run_name][checkpoint]["mean"],
                    "binding_ci95": statistics[run_name][checkpoint]["bootstrap_ci95"],
                    "entity_tracking": official["entity_tracking"],
                    "ewok": official["ewok"],
                    "global_piqa": official["GlobalPIQA_mean"],
                    "available7_mean": official["NLP_mean_no_superglue_aoa"],
                }
            )

    metrics = ["entity_tracking", "ewok", "global_piqa", "available7_mean"]
    correlations = {"pooled_levels": {}, "within_seed_levels": {}, "within_seed_deltas": {}}
    effects = [row["binding_effect"] for row in aligned_rows]
    for metric in metrics:
        correlations["pooled_levels"][metric] = safe_spearman(
            effects, [row[metric] for row in aligned_rows]
        )
    for run_name in RUNS:
        rows = [row for row in aligned_rows if row["run"] == run_name]
        correlations["within_seed_levels"][run_name] = {
            metric: safe_spearman(
                [row["binding_effect"] for row in rows], [row[metric] for row in rows]
            )
            for metric in metrics
        }
        correlations["within_seed_deltas"][run_name] = {}
        effect_delta = np.diff([row["binding_effect"] for row in rows]).tolist()
        for metric in metrics:
            metric_delta = np.diff([row[metric] for row in rows]).tolist()
            correlations["within_seed_deltas"][run_name][metric] = safe_spearman(
                effect_delta, metric_delta
            )

    ten_million_premature = all(
        value is not None and value > 10 for value in onset.values()
    )
    entity_level_support = all(
        correlations["within_seed_levels"][run]["entity_tracking"] >= 0.70
        for run in RUNS
    )
    entity_delta_support = all(
        correlations["within_seed_deltas"][run]["entity_tracking"] > 0
        for run in RUNS
    )
    if ten_million_premature and entity_level_support and entity_delta_support:
        decision = "H2_EMERGES_LATE_AND_TRACKS_ENTITY_TRAJECTORY"
        next_action = (
            "Require future H2/data screens to reach the observed onset before route decisions; "
            "use 10M only for runtime validation."
        )
    elif ten_million_premature:
        decision = "TEN_MILLION_H2_SCREEN_PREMATURE_BUT_PROBE_NOT_DECISIVE_FOR_CAPABILITY"
        next_action = (
            "Retract 10M-based mechanism closure, but do not optimize this probe directly; "
            "design future data screens around transfer-per-word and verify at the observed onset."
        )
    else:
        decision = "TEN_MILLION_H2_SCREEN_NOT_SHOWN_PREMATURE"
        next_action = "Keep early-screen conclusions; reduce priority of the event-binding route."

    payload = {
        "status": "EVENT_BINDING_LEARNING_TIMESCALE_COMPLETE",
        "design": {
            "new_training": False,
            "exact_child_checkpoints": True,
            "runs": {name: str(path.resolve()) for name, path in RUNS.items()},
            "milestones_m": MILESTONES,
            "cases": len(cases),
            "case_seed": 3021,
            "device": str(device),
            "no_babylm_eval_data": True,
        },
        "provenance": {
            "script": str(_public_path('experiments/archive/initial_model_studies/scripts/event_binding_learning_timescale.py')),
            "script_sha256": sha256_file(_public_path('experiments/archive/initial_model_studies/scripts/event_binding_learning_timescale.py')),
            "probe": str(_public_path('experiments/archive/initial_model_studies/scripts/revision_302b_event_binding_positive_control.py')),
            "probe_sha256": sha256_file(PROBE_PATH),
            "official_trajectory": str(_public_path('experiments/archive/initial_model_studies/data/existing_runs_exposure_trajectory.json')),
            "official_trajectory_sha256": sha256_file(SCORES_PATH),
        },
        "model_identities": model_identities,
        "statistics": statistics,
        "positive_onset_exposure_m": onset,
        "aligned_rows": aligned_rows,
        "correlations": correlations,
        "decision": decision,
        "next_action": next_action,
        "elapsed_sec": time.time() - started,
    }
    atomic_json(OUT, payload)

    lines = [
        "# research event-binding learning timescale",
        "",
        f"Decision: **{decision}**",
        "",
        "| exposure | seed 42 effect (95% CI) | seed 43 effect (95% CI) |",
        "|---:|---:|---:|",
    ]
    for exposure in MILESTONES:
        checkpoint = f"chck_{exposure}M"
        left = statistics["wwm_seed42"][checkpoint]
        right = statistics["wwm_seed43"][checkpoint]
        lines.append(
            f"| {exposure}M | {left['mean']:+.6f} "
            f"[{left['bootstrap_ci95'][0]:+.6f}, {left['bootstrap_ci95'][1]:+.6f}] | "
            f"{right['mean']:+.6f} "
            f"[{right['bootstrap_ci95'][0]:+.6f}, {right['bootstrap_ci95'][1]:+.6f}] |"
        )
    lines.extend(
        [
            "",
            f"First positive-CI milestone by seed: {onset}",
            "",
            "Within-seed Spearman correlations (effect levels / first differences):",
            "",
        ]
    )
    for run_name in RUNS:
        lines.append(
            f"- {run_name} Entity: "
            f"{correlations['within_seed_levels'][run_name]['entity_tracking']:+.3f} / "
            f"{correlations['within_seed_deltas'][run_name]['entity_tracking']:+.3f}; "
            f"available7: {correlations['within_seed_levels'][run_name]['available7_mean']:+.3f} / "
            f"{correlations['within_seed_deltas'][run_name]['available7_mean']:+.3f}"
        )
    lines.extend(
        [
            "",
            f"Next action: {next_action}",
            "",
            f"Evidence JSON: `{OUT}`",
        ]
    )
    _public_path('research/notes/initial_model_studies').mkdir(parents=True, exist_ok=True)
    NOTE.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": payload["status"],
                "decision": decision,
                "positive_onset_exposure_m": onset,
                "correlations": correlations,
                "out": str(OUT),
                "elapsed_sec": payload["elapsed_sec"],
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
