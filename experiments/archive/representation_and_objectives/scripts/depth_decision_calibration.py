#!/usr/bin/env python3
"""Depth decision calibration on the legal40k coordinate.

CPU-only. No training, no evaluation, no inferred scores.

Purpose: make the depth-completion decision immediate and rigorous. When the
legal40k 12x384 depth seed43022 vector arrives, the key questions are:
  1. How far is the matched legal40k 8x480 baseline from 41.80, column by column?
  2. What column movement pattern would depth need to cross 41.80?
  3. Given the legal16k calibration (41.8 needs about +0.697 cheap7 mean at 100M if
     SuperGLUE/AoA flat), what is the analogous legal40k requirement?

This quantifies the target so a depth vector can be judged against a concrete
scientific threshold, not a vague "near the frontier" heuristic. It also records
the two-preserve-column deficits (GlobalPIQA, Entity) that depth is hypothesized
to repair.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

ROOT = Path('.').resolve()
WS = ROOT / 'experiments/archive/representation_and_objectives'
OUT_DIR = WS / 'data/depth_decision_calibration'
OUT_JSON = OUT_DIR / 'depth_decision_calibration.json'
OUT_MD = (ROOT / 'research/notes/representation_and_objectives/depth_decision_calibration.md')

# The nine official columns and the Overall aggregation.
# Overall = mean of the nine columns as used by the official pipeline.
TASKS = ['BLiMP', 'Supplement', 'EWoK', 'Entity', 'COMPS', 'SuperGLUE', 'GlobalPIQA', 'Reading', 'AoA']

LEADER = {
    'BLiMP': 67.20, 'Supplement': 56.01, 'EWoK': 56.07, 'Entity': 28.45,
    'COMPS': 53.57, 'SuperGLUE': 69.79, 'GlobalPIQA': 39.67, 'Reading': 5.42,
    'AoA': 0.0, 'Overall': 41.80,
}

SUMMARIES = {
    'legal40k8x480_43022': WS / 'data/legal40k_accum_seed43022_pristine_collate/pristine_collate_legal40k_seed43022_summary.json',
    'legal40k8x480_43122': WS / 'data/legal40k_accum_seed43122_pristine_collate/pristine_collate_legal40k_seed43122_summary.json',
    'legal16k8x480_43022': WS / 'data/strictsmalltok_seed43022_pristine_collate/pristine_collate_strictsmalltok_seed43022_summary.json',
    'inherited16k_43022': WS / 'data/pristine_collate_seed43022/pristine_collate_seed43022_summary.json',
}


def finite(v: Any) -> float | None:
    try:
        x = float(v)
        return x if math.isfinite(x) else None
    except Exception:
        return None


def extract_vector(path: Path) -> dict[str, float]:
    rec = json.loads(path.read_text(encoding='utf-8'))
    ss = rec.get('score_summary', {})
    scores = ss.get('scores', {})
    vec: dict[str, float] = {}
    for t in TASKS:
        x = finite(scores.get(t))
        if x is not None:
            vec[t] = x
    ov = finite(ss.get('Overall'))
    if ov is not None:
        vec['Overall'] = ov
    # detail: GlobalPIQA split
    vec['_GlobalPIQA_parallel'] = finite(ss.get('GlobalPIQA_parallel'))
    vec['_GlobalPIQA_nonparallel'] = finite(ss.get('GlobalPIQA_nonparallel'))
    return vec


def overall_from_columns(vec: dict[str, float]) -> float:
    return sum(vec[t] for t in TASKS) / len(TASKS)


def rel(p: Path) -> str:
    try:
        return str(p.relative_to(ROOT))
    except Exception:
        return str(p)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    vectors = {}
    for label, path in SUMMARIES.items():
        if path.exists():
            vectors[label] = extract_vector(path)

    base = vectors.get('legal40k8x480_43022')
    payload: dict[str, Any] = {
        'status': 'DEPTH_DECISION_CALIBRATION',
        'tasks': TASKS,
        'leader': LEADER,
        'vectors': vectors,
        'available_summaries': {k: rel(v) for k, v in SUMMARIES.items() if v.exists()},
        'missing_summaries': {k: rel(v) for k, v in SUMMARIES.items() if not v.exists()},
    }

    if base is not None:
        # Verify our Overall recomputation matches the stored Overall (aggregation check)
        recomputed = overall_from_columns(base)
        payload['aggregation_check'] = {
            'stored_overall': base.get('Overall'),
            'recomputed_mean9': recomputed,
            'match_within_1e-6': abs(recomputed - base['Overall']) < 1e-6 if base.get('Overall') is not None else None,
        }

        # Per-column gap to the leader on the matched legal40k baseline
        col_gap = {t: LEADER[t] - base[t] for t in TASKS if t in base}
        payload['legal40k8x480_43022_gap_to_leader'] = col_gap
        payload['legal40k8x480_43022_overall_gap'] = LEADER['Overall'] - base['Overall']

        # The Overall gap decomposes as the mean of column gaps.
        # To cross 41.80, sum of column gains must be >= 9 * (41.80 - base_overall).
        needed_total_column_gain = 9.0 * (LEADER['Overall'] - base['Overall'])
        payload['needed_total_column_gain_to_reach_41p8'] = needed_total_column_gain
        payload['needed_mean_column_gain_to_reach_41p8'] = needed_total_column_gain / 9.0

        # The two hypothesized-repairable columns and how much they alone could contribute
        gp = base.get('GlobalPIQA')
        ent = base.get('Entity')
        # If depth restored GlobalPIQA to leader (39.67) and Entity to leader (28.45),
        # how much Overall would that add, holding others fixed?
        gp_max_gain = (LEADER['GlobalPIQA'] - gp) if gp is not None else None
        ent_max_gain = (LEADER['Entity'] - ent) if ent is not None else None
        two_col_ceiling_overall_gain = None
        if gp_max_gain is not None and ent_max_gain is not None:
            two_col_ceiling_overall_gain = (gp_max_gain + ent_max_gain) / 9.0
        payload['two_preserve_column_analysis'] = {
            'GlobalPIQA_base': gp,
            'GlobalPIQA_parallel_base': base.get('_GlobalPIQA_parallel'),
            'GlobalPIQA_nonparallel_base': base.get('_GlobalPIQA_nonparallel'),
            'Entity_base': ent,
            'GlobalPIQA_gain_if_restored_to_leader': gp_max_gain,
            'Entity_gain_if_restored_to_leader': ent_max_gain,
            'overall_gain_if_both_restored_to_leader': two_col_ceiling_overall_gain,
            'overall_after_both_restored': (base['Overall'] + two_col_ceiling_overall_gain) if two_col_ceiling_overall_gain is not None else None,
            'reaches_41p8_from_two_columns_alone': ((base['Overall'] + two_col_ceiling_overall_gain) >= LEADER['Overall']) if two_col_ceiling_overall_gain is not None else None,
        }

        # Cheap7 threshold analogue on the legal40k coordinate.
        # cheap7 = the 7 zero-shot + reading columns (exclude SuperGLUE, AoA).
        cheap7_cols = ['BLiMP', 'Supplement', 'EWoK', 'Entity', 'COMPS', 'GlobalPIQA', 'Reading']
        cheap7_mean = sum(base[t] for t in cheap7_cols) / len(cheap7_cols)
        sg = base.get('SuperGLUE')
        aoa = base.get('AoA', 0.0)
        # Overall = (7*cheap7_mean + SuperGLUE + AoA)/9
        # To reach 41.8 with SuperGLUE/AoA fixed:
        # need cheap7_mean' such that (7*cheap7_mean' + sg + aoa)/9 = 41.8
        needed_cheap7_mean = (9.0 * LEADER['Overall'] - sg - aoa) / 7.0
        payload['cheap7_calibration'] = {
            'cheap7_columns': cheap7_cols,
            'cheap7_mean_base': cheap7_mean,
            'SuperGLUE_base': sg,
            'AoA_base': aoa,
            'needed_cheap7_mean_if_SuperGLUE_AoA_flat': needed_cheap7_mean,
            'needed_cheap7_mean_gain': needed_cheap7_mean - cheap7_mean,
        }

        # Decision thresholds for the depth vector (Overall-level)
        payload['depth_decision_thresholds'] = {
            'crosses_frontier': 'depth_overall >= 41.80: protect endpoint, reproduce seed43122',
            'strong_progress': 'depth_overall >= 41.55 OR matched_delta >= +0.35: seed43122 or focused combination',
            'flat_or_worse': 'matched_delta <= -0.10: do not spend second seed on depth alone',
            'ambiguous_middle': 'otherwise: use column pattern + A02 evidence, consider experience-utilization U256 next',
            'matched_baseline_overall': base['Overall'],
        }

    OUT_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')

    # Markdown
    lines = ['# research — A01 depth decision calibration on the legal40k coordinate\n\n']
    lines.append('CPU-only. No training or evaluation; no inferred scores. This quantifies the concrete threshold the depth vector must meet.\n\n')
    if base is not None:
        lines.append(f"## Matched baseline: legal40k 8x480 seed43022 Overall {base['Overall']:.6f} (gap to 41.80 = {LEADER['Overall']-base['Overall']:+.6f})\n\n")
        lines.append('### Per-column gap to the visible leader\n')
        for t in TASKS:
            if t in base:
                lines.append(f"- {t}: base {base[t]:.4f}, leader {LEADER[t]:.4f}, gap {LEADER[t]-base[t]:+.4f}\n")
        lines.append('\n')
        tpa = payload['two_preserve_column_analysis']
        lines.append('### Two-preserve-column ceiling (GlobalPIQA + Entity restored to leader)\n')
        lines.append(f"- GlobalPIQA base {tpa['GlobalPIQA_base']:.4f} (parallel {tpa['GlobalPIQA_parallel_base']:.2f}, nonparallel {tpa['GlobalPIQA_nonparallel_base']:.2f}), max gain {tpa['GlobalPIQA_gain_if_restored_to_leader']:+.4f}\n")
        lines.append(f"- Entity base {tpa['Entity_base']:.4f}, max gain {tpa['Entity_gain_if_restored_to_leader']:+.4f}\n")
        lines.append(f"- Overall gain if BOTH restored to leader: {tpa['overall_gain_if_both_restored_to_leader']:+.4f} -> Overall {tpa['overall_after_both_restored']:.4f}\n")
        lines.append(f"- Reaches 41.8 from these two columns alone: **{tpa['reaches_41p8_from_two_columns_alone']}**\n\n")
        c7 = payload['cheap7_calibration']
        lines.append('### Cheap7 calibration (A02-style, A01 legal40k coordinate)\n')
        lines.append(f"- cheap7 mean base: {c7['cheap7_mean_base']:.4f}\n")
        lines.append(f"- SuperGLUE base: {c7['SuperGLUE_base']:.4f}, AoA base: {c7['AoA_base']:.4f}\n")
        lines.append(f"- needed cheap7 mean if SuperGLUE/AoA flat: {c7['needed_cheap7_mean_if_SuperGLUE_AoA_flat']:.4f} (gain {c7['needed_cheap7_mean_gain']:+.4f})\n\n")
        lines.append('### Depth decision thresholds\n')
        for k, v in payload['depth_decision_thresholds'].items():
            lines.append(f"- {k}: {v}\n")
    lines.append(f"\nJSON: `{rel(OUT_JSON)}`\n")
    OUT_MD.write_text(''.join(lines), encoding='utf-8')

    print(json.dumps({
        'status': payload['status'],
        'legal40k8x480_43022_overall_gap': payload.get('legal40k8x480_43022_overall_gap'),
        'needed_mean_column_gain': payload.get('needed_mean_column_gain_to_reach_41p8'),
        'two_col_reaches_41p8': payload.get('two_preserve_column_analysis', {}).get('reaches_41p8_from_two_columns_alone'),
        'aggregation_match': payload.get('aggregation_check', {}).get('match_within_1e-6'),
        'out_json': rel(OUT_JSON),
        'note': rel(OUT_MD),
    }, indent=2), flush=True)


if __name__ == '__main__':
    main()
