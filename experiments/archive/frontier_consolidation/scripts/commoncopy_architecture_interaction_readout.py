#!/usr/bin/env python3
"""research: common-copy no-disentangle readout harness.

This is a thin, parameterized wrapper around the hardened research selected-panel
script.  It emits the same canonical four arm names expected by the research
four-cell bootstrap analyzer:

    full_compact, full_repeat, nodis_compact, nodis_repeat

but points nodis_compact/nodis_repeat at the research common-copied
no-disentangle runs.  Therefore the existing interval and bootstrap scripts can
be reused unchanged by passing --panel-dir to this script's output directory.

The readout question is:

    I_commoncopy = (compact-repeat)_commoncopy_no_disentangle
                   - (compact-repeat)_full_DeBERTa

Common-copy removes the construction-RNG drift identified in research by copying
all same-shaped common tensors from the same full DeBERTa initialization before
training the no-disentangle target; it still omits the p2c/c2p positional-score
projection tensors and therefore still has the smaller no-disentangle parameter
count.  Collapse after common-copy strengthens the evidence that the removed
positional-score pathway is load-bearing; recovery means the research plain
no-disentangle collapse was largely an initialization/optimization-basin effect;
intermediate behavior splits responsibility.

This script does not train, run SuperGLUE/AoA, upload, or submit.  In
--plan-only/--summarize-only modes it does not run selected evaluation either.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any


def find_user_root() -> Path:
    return _PUBLIC_ROOT


ROOT = find_user_root()
WS = ROOT / "experiments/archive/frontier_consolidation"
BASE_PANEL_SCRIPT = WS / "scripts/architecture_interaction_selected_panel.py"
DEFAULT_OUT = WS / "data/commoncopy_architecture_interaction_selected_panel"

FULL_COMPACT = WS / "training/runs/complianttok_reinvest_seed43022_r2"
FULL_REPEAT = WS / "training/runs/full_p2c_c2p_abs_repeat_deberta100M_seed43022"
COMMONCOPY_COMPACT = WS / "training/runs/commoncopy_nodis_compact_deberta100M_seed43022"
COMMONCOPY_REPEAT = WS / "training/runs/commoncopy_nodis_repeat_deberta100M_seed43022"


def load_base_module() -> Any:
    spec = importlib.util.spec_from_file_location("panel_for_commoncopy", BASE_PANEL_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {BASE_PANEL_SCRIPT}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def patch_commoncopy_arms(mod: Any) -> None:
    """Patch the research panel module in memory only."""
    mod.DEFAULT_OUT = DEFAULT_OUT
    mod.ARMS = {
        "full_compact": {
            "run_dir": FULL_COMPACT,
            "variant": "full_p2c_c2p_abs",
            "data_arm": "compact",
            "role": "existing legal full-DeBERTa compact cell",
        },
        "full_repeat": {
            "run_dir": FULL_REPEAT,
            "variant": "full_p2c_c2p_abs",
            "data_arm": "repeat",
            "role": "legal full-DeBERTa repeat cell from research",
        },
        "nodis_compact": {
            "run_dir": COMMONCOPY_COMPACT,
            "variant": "commoncopy_no_disentangle_abs",
            "data_arm": "compact",
            "role": "common-copied compact cell with c2p/p2c score terms removed",
        },
        "nodis_repeat": {
            "run_dir": COMMONCOPY_REPEAT,
            "variant": "commoncopy_no_disentangle_abs",
            "data_arm": "repeat",
            "role": "common-copied repeat cell with c2p/p2c score terms removed",
        },
    }
    # Preserve validated reuse for the full compact reference.  Full repeat already
    # has selected payloads in the research panel; reuse them to avoid rescoring.
    mod.KNOWN_EXISTING_PER_TARGETS = {
        ("full_compact", "chck_80M"): WS / "data/legal_mature_treatment_effect_eval/per_target/complianttok_reinvest_seed43022_80M.json",
        ("full_compact", "chck_100M"): WS / "data/compliant_full_eval/per_target/complianttok_reinvest_seed43022.json",
        ("full_repeat", "chck_80M"): WS / "data/architecture_interaction_selected_panel_final/full_repeat/chck_80M/eval/per_target/arch_full_repeat_chck_80M.json",
        ("full_repeat", "chck_100M"): WS / "data/architecture_interaction_selected_panel_final/full_repeat/chck_100M/eval/per_target/arch_full_repeat_chck_100M.json",
    }


def add_commoncopy_metadata(out_dir: Path, payload: dict[str, Any] | None) -> None:
    meta = {
        "status": "COMMONCOPY_READOUT_METADATA",
        "out_dir": str(out_dir),
        "base_panel_script": str(BASE_PANEL_SCRIPT),
        "canonical_arm_names": ["full_compact", "full_repeat", "nodis_compact", "nodis_repeat"],
        "patched_nodis_meaning": "nodis_* rows point to common-copied no-disentangle runs, not the research plain no-disentangle runs",
        "commoncopy_run_dirs": {
            "nodis_compact": str(COMMONCOPY_COMPACT),
            "nodis_repeat": str(COMMONCOPY_REPEAT),
        },
        "selected_panel_summary": str(out_dir / "architecture_interaction_selected_panel_summary.json"),
        "next_cpu_readouts_after_panel": [
            "python3 -B experiments/archive/frontier_consolidation/scripts/architecture_interaction_interval_driver.py --panel-dir " + str(out_dir) + " --out-dir experiments/archive/frontier_consolidation/data/commoncopy_architecture_interaction_pair_intervals --checkpoints chck_80M chck_100M",
            "python3 -B experiments/archive/frontier_consolidation/scripts/architecture_interaction_bootstrap_analyzer.py --panel-dir " + str(out_dir) + " --out-dir experiments/archive/frontier_consolidation/data/commoncopy_architecture_interaction_four_cell_bootstrap --checkpoints chck_80M chck_100M",
        ],
        "decision_reading": {
            "attenuation_persists": "common-copy compact-minus-repeat remains far below full on stable families; removed p2c/c2p positional-score pathway is strongly implicated, with residual parameter-count caveat",
            "effect_recovers": "common-copy compact-minus-repeat becomes full-like on stable families; plain no-disentangle collapse was mostly initialization/optimization-basin dependence",
            "intermediate": "both score-path removal and initialization/capacity/basin contribute",
        },
        "boundary": "No training, SuperGLUE, AoA, upload, or leaderboard submission is performed by this harness.",
    }
    if payload is not None:
        meta["row_count"] = payload.get("row_count")
        meta["problem_count"] = len(payload.get("problems", [])) if isinstance(payload.get("problems"), list) else None
        meta["mean_over_common_checkpoints"] = payload.get("mean_over_common_checkpoints")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "commoncopy_readout_metadata.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--arms", nargs="+", default=["full_compact", "full_repeat", "nodis_compact", "nodis_repeat"])
    ap.add_argument("--checkpoints", nargs="+", default=["chck_80M", "chck_100M"])
    ap.add_argument("--plan-only", action="store_true")
    ap.add_argument("--summarize-only", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    mod = load_base_module()
    patch_commoncopy_arms(mod)
    unknown = [a for a in args.arms if a not in mod.ARMS]
    if unknown:
        raise ValueError(f"Unknown arms: {unknown}; allowed {sorted(mod.ARMS)}")

    mod.make_plan(args.out_dir, args.arms, args.checkpoints)
    payload = None
    if args.plan_only:
        add_commoncopy_metadata(args.out_dir, payload)
        print(json.dumps({
            "status": "COMMONCOPY_ARCHITECTURE_INTERACTION_SELECTED_PANEL_PLAN_WRITTEN",
            "out": str(args.out_dir / "selected_panel_plan.json"),
            "metadata": str(args.out_dir / "commoncopy_readout_metadata.json"),
            "arms": args.arms,
            "checkpoints": args.checkpoints,
        }, indent=2), flush=True)
        return

    if not args.summarize_only:
        mod.run_panel(args.out_dir, args.arms, args.checkpoints, force=args.force)
    payload = mod.summarize(args.out_dir, args.arms, args.checkpoints)
    add_commoncopy_metadata(args.out_dir, payload)


if __name__ == "__main__":
    main()
