#!/usr/bin/env python3
"""Test whether prior matched data interventions moved event binding at 10M.

All models use the S1 12x384 shape, baseline16k tokenizer, AdamW, flat WWM,
and 10M exposure. The primary contrasts preserve the intended data controls:
aligned versus shuffled WikiAuto and high-structure versus matched-low/uniform.
No new model is trained and no BabyLM evaluation data is read.
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
PROBE_PATH = _public_path('experiments/archive/initial_model_studies/scripts/revision_302b_event_binding_positive_control.py')
OUT = _public_path('experiments/archive/initial_model_studies/data/prior_data_event_binding_panel.json')
NOTE = _public_path('research/notes/initial_model_studies/prior_data_event_binding_panel.md')


def ckpt(run: str, child: str = "chck_10M") -> Path:
    return STUDY / f"training/runs/{run}/hf_model/{child}"


MODELS = {
    "official_s1": ckpt("babylm_leadershape_s1_10M_aligned_micro128"),
    "wikiauto_aligned": ckpt("babylm_aligned_s1_10M", "chck_9999996w"),
    "wikiauto_shuffled": ckpt("babylm_shuffled_s1_10M", "chck_9999996w"),
    "high_entity_state": ckpt("babylm_high_entity_state_s1_10M", "chck_9999947w"),
    "matched_low_structure": ckpt("babylm_matched_low_s1_10M", "chck_9999960w"),
    "uniform_structure": ckpt("babylm_uniform_s1_10M", "chck_9999997w"),
}

CONTRASTS = {
    "wikiauto_aligned_minus_shuffled": ("wikiauto_aligned", "wikiauto_shuffled"),
    "high_entity_state_minus_matched_low": ("high_entity_state", "matched_low_structure"),
    "high_entity_state_minus_uniform": ("high_entity_state", "uniform_structure"),
}


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def vector(rows: dict[int, dict], common: list[int]) -> np.ndarray:
    return np.asarray(
        [rows[index]["normalized_extra_entity_effect"] for index in common],
        dtype=np.float64,
    )


def main() -> None:
    started = time.time()
    probe = load_module(PROBE_PATH, "step302b_data_panel")
    scorer = probe.load_step302_module()
    scorer.setup_environment()
    device = scorer.torch.device("cuda:0" if scorer.torch.cuda.is_available() else "cpu")
    cases = probe.build_cases(420, 3021)
    missing = [str(path) for path in MODELS.values() if not path.is_dir()]
    if missing:
        raise RuntimeError(f"missing checkpoints: {missing}")

    scored = {}
    for name, path in MODELS.items():
        print(json.dumps({"event": "score_start", "model": name}), flush=True)
        scored[name] = scorer.score_model(name, path, cases, 160, 24, device)
        print(json.dumps({"event": "score_complete", "model": name}), flush=True)
    maps = {
        name: {row["case_index"]: row for row in result["rows"]}
        for name, result in scored.items()
    }
    rng = np.random.default_rng(304042)
    model_stats = {}
    for name, rows in maps.items():
        common = sorted(rows)
        model_stats[name] = scorer.bootstrap_mean(vector(rows, common), rng, 20000)

    contrasts = {}
    for label, (left, right) in CONTRASTS.items():
        common = sorted(set(maps[left]) & set(maps[right]))
        delta = vector(maps[left], common) - vector(maps[right], common)
        contrasts[label] = scorer.bootstrap_mean(delta, rng, 20000)

    positive = [
        label for label, result in contrasts.items() if result["bootstrap_ci95"][0] > 0
    ]
    if positive:
        decision = "PRIOR_DATA_ROUTE_MOVED_EVENT_BINDING_AT_10M"
        next_action = "Only the positive matched data contrast may be considered for a 100M two-seed test."
    else:
        decision = "NO_PRIOR_DATA_ROUTE_MOVED_EVENT_BINDING_AT_10M"
        next_action = (
            "Do not reopen WikiAuto adjacency or static structure-density selection. Test a symmetric "
            "crossed-logit objective that directly requires ordinary MLM logits to reverse with bindings."
        )

    payload = {
        "status": "PRIOR_DATA_EVENT_BINDING_PANEL_COMPLETE",
        "design": {
            "new_training": False,
            "cases": len(cases),
            "case_seed": 3021,
            "matched_geometry": "S1 12x384, baseline16k, AdamW, flat WWM, 10M exposure",
            "device": str(device),
            "probe_sha256": scorer.sha256_file(PROBE_PATH),
        },
        "models": {name: result["identity"] for name, result in scored.items()},
        "model_statistics": model_stats,
        "contrasts": contrasts,
        "positive_matched_contrasts": positive,
        "decision": decision,
        "next_action": next_action,
        "elapsed_sec": time.time() - started,
    }
    scorer.write_json(OUT, payload)
    lines = [
        "# research prior-data event-binding panel",
        "",
        f"Decision: **{decision}**",
        "",
        "| model | effect mean | 95% CI |",
        "|---|---:|---:|",
    ]
    for name, result in model_stats.items():
        lines.append(
            f"| {name} | {result['mean']:+.6f} | "
            f"[{result['bootstrap_ci95'][0]:+.6f}, {result['bootstrap_ci95'][1]:+.6f}] |"
        )
    lines.extend(["", "| matched contrast | delta mean | 95% CI |", "|---|---:|---:|"])
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
                "positive_matched_contrasts": positive,
                "contrasts": contrasts,
                "out": str(OUT),
                "elapsed_sec": payload["elapsed_sec"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
