#!/usr/bin/env python3
"""research guarded dynamics reader for the compact-view triangle.

This extends the research dynamics summary with an explicit implementation guard:
the repeat and adjbreak arms use the activation-checkpointed trainer, so endpoint
or dynamics gaps should not be read causally against the historical compact-view
reference unless the same-data compact-view GC reference matches the historical
trajectory at the checked early horizon.
"""
from __future__ import annotations

import importlib.util
import json
import pathlib
import sys
from typing import Any

USER_ROOT = pathlib.Path('.')
DYNAMICS = USER_ROOT / 'experiments/archive/representation_and_objectives/scripts/triangle_dynamics_compare.py'
OUT_DIR = USER_ROOT / 'experiments/archive/representation_and_objectives/data/compact_triangle_dynamics_guarded'
EQUIV_SUMMARY = USER_ROOT / 'experiments/archive/representation_and_objectives/data/gc_reference_equivalence/gc_reference_equivalence_summary.json'
GC_VIEW_RUN = USER_ROOT / 'experiments/archive/representation_and_objectives/training/runs/gc_compact_view_reinvest_1M_seed43022'
HIST_VIEW_RUN = USER_ROOT / 'experiments/archive/frontier_consolidation/training/runs/cleanqwen_fineweb_compact_view_reinvest_16k_seed43022'


def load_step202():
    spec = importlib.util.spec_from_file_location('dyn', DYNAMICS)
    if spec is None or spec.loader is None:
        raise RuntimeError(f'could not import {DYNAMICS}')
    mod = importlib.util.module_from_spec(spec)
    sys.modules['dyn'] = mod
    spec.loader.exec_module(mod)
    return mod


def read_json(path: pathlib.Path) -> Any:
    return json.loads(path.read_text(encoding='utf-8'))


def implementation_guard() -> dict[str, Any]:
    out: dict[str, Any] = {
        'summary_path': str(EQUIV_SUMMARY),
        'exists': EQUIV_SUMMARY.exists(),
        'allows_historical_reference_for_causal_triangle': False,
        'required_before_causal_interpretation': True,
    }
    if not EQUIV_SUMMARY.exists():
        out['interpretation'] = 'missing_gc_reference_equivalence__do_not_read_historical_view_vs_gc_controls_causally'
        return out
    try:
        payload = read_json(EQUIV_SUMMARY)
    except Exception as exc:
        out['interpretation'] = 'unreadable_gc_reference_equivalence'
        out['read_error'] = repr(exc)
        return out
    comp = payload.get('log_comparison', {}) if isinstance(payload, dict) else {}
    interp = payload.get('interpretation') if isinstance(payload, dict) else None
    out.update({
        'status': payload.get('status') if isinstance(payload, dict) else None,
        'interpretation': interp,
        'allows_historical_reference_for_causal_triangle': interp == 'trajectory_equivalent_to_checked_horizon',
        'n_compared_steps': comp.get('n_compared_steps') if isinstance(comp, dict) else None,
        'max_abs_delta': comp.get('max_abs_delta') if isinstance(comp, dict) else None,
        'selected_step_records': comp.get('selected_step_records') if isinstance(comp, dict) else None,
        'triangle_use_rule': payload.get('triangle_use_rule') if isinstance(payload, dict) else None,
    })
    return out


def main() -> None:
    mod = load_step202()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    runs = dict(mod.RUNS)
    runs['compact_view_reinvest_gc_1M_reference'] = GC_VIEW_RUN
    summaries = {name: mod.summarize_run(name, run_dir) for name, run_dir in runs.items()}
    payload = {
        'status': 'COMPACT_TRIANGLE_DYNAMICS_WITH_IMPLEMENTATION_GUARD',
        'note': 'Use this guarded dynamics file with the no-AoA triangle readout. The GC reference equivalence must be true before interpreting historical compact_view vs GC-trained controls as a causal data mechanism.',
        'implementation_equivalence': implementation_guard(),
        'runs': summaries,
        'contrasts_historical_triangle': mod.build_contrasts({k: v for k, v in summaries.items() if k in mod.RUNS}),
        'historical_view_run': str(HIST_VIEW_RUN),
        'gc_view_reference_run': str(GC_VIEW_RUN),
    }
    out_json = OUT_DIR / 'triangle_dynamics_guarded_summary.json'
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(json.dumps({
        'status': payload['status'],
        'out_json': str(out_json),
        'implementation_guard': payload['implementation_equivalence'],
        'logged_steps': {k: v.get('n_logged_steps') for k, v in summaries.items()},
        'metrics_exists': {k: v.get('metrics_exists') for k, v in summaries.items()},
    }, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
