#!/usr/bin/env python3
"""Attribute the Step 302b event-binding signal across existing checkpoints.

This is a zero-training panel. It applies the exact Step 302b cases and scorer
to already completed architecture, tokenizer, curriculum, optimizer, seed, and
public-leader checkpoints. Pairwise contrasts are only made between conditions
that isolate a documented factor at a matched exposure where available.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np


STUDY = _public_path('experiments/archive/initial_model_studies')
STEP302B_PATH = _public_path('experiments/archive/initial_model_studies/scripts/revision_302b_event_binding_positive_control.py')
OUT = _public_path('experiments/archive/initial_model_studies/data/event_binding_attribution_panel.json')
NOTE = _public_path('research/notes/initial_model_studies/event_binding_attribution_panel.md')


def checkpoint(run: str, exposure: str) -> Path:
    return STUDY / f"training/runs/{run}/hf_model/chck_{exposure}"


MODELS = {
    "protected8x480_10M": checkpoint(
        "babylm_fullcycle_debertav2_8x480_wwm_seed42_100M_b256", "10M"
    ),
    "protected8x480_100M": checkpoint(
        "babylm_fullcycle_debertav2_8x480_wwm_seed42_100M_b256", "100M"
    ),
    "wwm43_100M": checkpoint(
        "fullcycle_official_wwm_debertav2_8x480_seed43_100M_b256", "100M"
    ),
    "s1_shape12x384_10M": checkpoint(
        "babylm_leadershape_s1_10M_aligned_micro128", "10M"
    ),
    "s1_shape12x384_100M": checkpoint(
        "babylm_leadershape_s1_100M_aligned_micro128", "100M"
    ),
    "s2_curriculum12x384_100M": checkpoint(
        "babylm_true_s2_100M_wordclock_wwm70_len64256", "100M"
    ),
    "official40k8x480_100M": checkpoint(
        "babylm_fullcycle_debertav2_8x480_official40k_wwm_seed42_100M_b128_acc2",
        "100M",
    ),
    "s3_shape40k_10M": checkpoint(
        "babylm_s3_12x384_official40k_flatwwm_10M", "10M"
    ),
    "lamb_s1_10M": checkpoint("babylm_lamb_refdefault_s1_10M", "10M"),
    "public_leader": _public_path('experiments/archive/initial_model_studies/data/leader_package_revision_124/model_side/local'),
}

CONTRASTS = {
    "seed43_minus_seed42_at_100M": ("wwm43_100M", "protected8x480_100M"),
    "shape12x384_minus_8x480_at_10M": ("s1_shape12x384_10M", "protected8x480_10M"),
    "shape12x384_minus_8x480_at_100M": ("s1_shape12x384_100M", "protected8x480_100M"),
    "curriculum_minus_flat_at_100M": (
        "s2_curriculum12x384_100M",
        "s1_shape12x384_100M",
    ),
    "40k_minus_16k_under_8x480_at_100M": (
        "official40k8x480_100M",
        "protected8x480_100M",
    ),
    "40k_minus_16k_under_12x384_at_10M": (
        "s3_shape40k_10M",
        "s1_shape12x384_10M",
    ),
    "lamb_minus_adamw_under_12x384_at_10M": ("lamb_s1_10M", "s1_shape12x384_10M"),
    "public_leader_minus_protected": ("public_leader", "protected8x480_100M"),
}

# Seed sensitivity is evidence about measurement variance, not an actionable
# model/data/training factor. Never promote a random seed as a research route.
ACTIONABLE_FACTOR_CONTRASTS = {
    "shape12x384_minus_8x480_at_10M",
    "shape12x384_minus_8x480_at_100M",
    "curriculum_minus_flat_at_100M",
    "40k_minus_16k_under_8x480_at_100M",
    "40k_minus_16k_under_12x384_at_10M",
    "lamb_minus_adamw_under_12x384_at_10M",
}


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def values(rows: dict[int, dict], common: list[int]) -> np.ndarray:
    return np.asarray(
        [rows[index]["normalized_extra_entity_effect"] for index in common],
        dtype=np.float64,
    )


def main() -> None:
    started = time.time()
    probe = load_module(STEP302B_PATH, "step302b_panel_source")
    scorer = probe.load_step302_module()
    scorer.setup_environment()
    device = scorer.torch.device("cuda:0" if scorer.torch.cuda.is_available() else "cpu")
    cases = probe.build_cases(420, 3021)

    missing = [str(path) for path in MODELS.values() if not path.is_dir()]
    if missing:
        raise RuntimeError(f"missing model directories: {missing}")

    scored = {}
    for name, path in MODELS.items():
        print(json.dumps({"event": "score_start", "model": name}), flush=True)
        scored[name] = scorer.score_model(name, path, cases, 160, 24, device)
        print(
            json.dumps({"event": "score_complete", "model": name, "rows": len(scored[name]["rows"])}),
            flush=True,
        )

    row_maps = {
        name: {row["case_index"]: row for row in result["rows"]}
        for name, result in scored.items()
    }
    rng = np.random.default_rng(303042)
    model_statistics = {}
    for name, rows in row_maps.items():
        common = sorted(rows)
        model_statistics[name] = scorer.bootstrap_mean(values(rows, common), rng, 20000)
        model_statistics[name]["by_family"] = probe.summarize_by_source(rows, cases)

    contrasts = {}
    for label, (left, right) in CONTRASTS.items():
        common = sorted(set(row_maps[left]) & set(row_maps[right]))
        delta = values(row_maps[left], common) - values(row_maps[right], common)
        contrasts[label] = {
            "left": left,
            "right": right,
            **scorer.bootstrap_mean(delta, rng, 20000),
        }

    isolated = [
        label
        for label, result in contrasts.items()
        if label in ACTIONABLE_FACTOR_CONTRASTS and result["bootstrap_ci95"][0] > 0
    ]
    leader_gap = contrasts["public_leader_minus_protected"]
    if isolated:
        decision = "ONE_OR_MORE_EXISTING_ISOLATED_FACTORS_INCREASE_EVENT_BINDING"
        next_action = (
            "Rank positive isolated factors by effect size and official-score tradeoff; test only the strongest "
            "factor on a matched two-seed endpoint before changing data."
        )
    elif leader_gap["bootstrap_ci95"][0] > 0:
        decision = "LEADER_SIGNAL_NOT_REPRODUCED_BY_TESTED_NONDATA_FACTORS"
        next_action = (
            "Treat leader-grade experience structure as the main unresolved factor and build a coverage-preserving, "
            "event-binding semantic compression arm with an ordinary-simplification control."
        )
    else:
        decision = "ATTRIBUTION_PANEL_INCONCLUSIVE"
        next_action = "Do not launch training; inspect the probe and factor alignment."

    payload = {
        "status": "EVENT_BINDING_ATTRIBUTION_PANEL_COMPLETE",
        "design": {
            "cases": len(cases),
            "case_seed": 3021,
            "probe": str(STEP302B_PATH),
            "probe_sha256": scorer.sha256_file(STEP302B_PATH),
            "scorer": str(probe.PATH),
            "scorer_sha256": scorer.sha256_file(probe.PATH),
            "device": str(device),
            "new_training": False,
        },
        "models": {name: result["identity"] for name, result in scored.items()},
        "model_statistics": model_statistics,
        "contrasts": contrasts,
        "isolated_positive_contrasts": isolated,
        "decision": decision,
        "next_action": next_action,
        "elapsed_sec": time.time() - started,
    }
    scorer.write_json(OUT, payload)

    lines = [
        "# research event-binding attribution panel",
        "",
        f"Decision: **{decision}**",
        "",
        "No new model was trained. All rows use the exact Step 302b diagnostic.",
        "",
        "## Checkpoints",
        "",
        "| checkpoint | effect mean | 95% CI | fraction > 0 |",
        "|---|---:|---:|---:|",
    ]
    for name, result in model_statistics.items():
        lines.append(
            f"| {name} | {result['mean']:+.6f} | "
            f"[{result['bootstrap_ci95'][0]:+.6f}, {result['bootstrap_ci95'][1]:+.6f}] | "
            f"{result['fraction_positive']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Matched contrasts",
            "",
            "| contrast | delta mean | 95% CI |",
            "|---|---:|---:|",
        ]
    )
    for label, result in contrasts.items():
        lines.append(
            f"| {label} | {result['mean']:+.6f} | "
            f"[{result['bootstrap_ci95'][0]:+.6f}, {result['bootstrap_ci95'][1]:+.6f}] |"
        )
    lines.extend(["", f"Next action: {next_action}", "", f"Evidence JSON: `{OUT}`"])
    _public_path('research/notes/initial_model_studies').mkdir(parents=True, exist_ok=True)
    NOTE.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": payload["status"],
                "decision": decision,
                "isolated_positive_contrasts": isolated,
                "contrasts": contrasts,
                "out": str(OUT),
                "elapsed_sec": payload["elapsed_sec"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
