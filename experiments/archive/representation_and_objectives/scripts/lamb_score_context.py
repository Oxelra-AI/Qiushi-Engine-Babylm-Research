#!/usr/bin/env python3
"""research: build the correct score context for interpreting research LAMB endpoints.

The active research runs should be compared by cheap7 vectors against matched cheap7
vectors, not against full nine-column Overall values.  This script records those
reference vectors and the full-eval trigger thresholds before any LAMB endpoint is
read.
"""
from __future__ import annotations

import json
from pathlib import Path
from statistics import mean

ROOT = Path('.')
OUT = ROOT / 'experiments/archive/representation_and_objectives/data/lamb_score_context'
NOTE = ROOT / 'research/notes/representation_and_objectives/lamb_score_context.md'

BASELINES = {
    'legal40k_8x480_adamw_fixed256_seed43022': {
        'role': 'matched legal40k compact-view AdamW 8×480 baseline, fixed 256-token row truncation, full official coordinate',
        'summary_path': ROOT / 'experiments/archive/representation_and_objectives/data/legal40k_accum_seed43022_pristine_collate/pristine_collate_legal40k_seed43022_summary.json',
    },
    'legal40k_12x384_adamw_fixed256_seed43022': {
        'role': 'matched legal40k compact-view AdamW 12×384 depth baseline, fixed 256-token row truncation, full official coordinate',
        'summary_path': ROOT / 'experiments/archive/representation_and_objectives/data/legal40k_12x384_depth_seed43022_pristine_collate/pristine_collate_legal40k_12x384_depth_seed43022_summary.json',
    },
}

CHEAP7 = ['BLiMP','Supplement','EWoK','Entity','COMPS','GlobalPIQA','Reading']
LEADER = {
    'role': 'visible public leader go76dof/wwm_curriculum_simplification_40k, displayed Overall 41.80; used here only as public frontier context',
    'scores': {
        'BLiMP': 67.20,
        'Supplement': 56.01,
        'EWoK': 56.07,
        'Entity': 28.45,
        'COMPS': 53.57,
        'GlobalPIQA': 39.67,
        'Reading': 5.42,
        'SuperGLUE': 69.79,
        'AoA': 0.0,
    },
    'Overall': 41.80,
}


def load_summary(path: Path) -> dict:
    obj = json.loads(path.read_text(encoding='utf-8'))
    ss = obj['score_summary']
    scores = {k: float(ss['scores'][k]) for k in CHEAP7}
    scores['SuperGLUE'] = float(ss['scores']['SuperGLUE'])
    scores['AoA'] = float(ss['scores'].get('AoA', 0.0))
    return {
        'scores': scores,
        'cheap7': mean(scores[k] for k in CHEAP7),
        'Overall': float(ss['Overall']),
        'NLP_average': float(ss.get('NLP_average', 0.0)),
        'GlobalPIQA_parallel': float(ss['GlobalPIQA_parallel']),
        'GlobalPIQA_nonparallel': float(ss['GlobalPIQA_nonparallel']),
        'summary_path': str(path),
    }


def required_cheap7_for_overall(target_overall: float, superglue: float, aoa: float = 0.0) -> float:
    return (9.0 * target_overall - superglue - aoa) / 7.0


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    NOTE.parent.mkdir(parents=True, exist_ok=True)
    refs = {}
    for key, spec in BASELINES.items():
        rec = load_summary(spec['summary_path'])
        rec['role'] = spec['role']
        refs[key] = rec
    leader = dict(LEADER)
    leader['cheap7'] = mean(leader['scores'][k] for k in CHEAP7)

    best_baseline_key = max(refs, key=lambda k: refs[k]['cheap7'])
    best_baseline = refs[best_baseline_key]
    thresholds = {
        'leader_cheap7': leader['cheap7'],
        'best_matched_baseline_cheap7': best_baseline['cheap7'],
        'cheap7_needed_for_overall_41p80_if_superglue_equals_8x480_baseline': required_cheap7_for_overall(41.80, refs['legal40k_8x480_adamw_fixed256_seed43022']['scores']['SuperGLUE']),
        'cheap7_needed_for_overall_41p80_if_superglue_equals_12x384_baseline': required_cheap7_for_overall(41.80, refs['legal40k_12x384_adamw_fixed256_seed43022']['scores']['SuperGLUE']),
        'cheap7_needed_for_overall_41p80_if_superglue_equals_visible_leader': required_cheap7_for_overall(41.80, leader['scores']['SuperGLUE']),
        'full_eval_priority_rule': (
            'Run full official evaluation immediately if a research endpoint has cheap7 above the visible leader cheap7 or has a vector shape that could plausibly cross 41.80 with its expected SuperGLUE; otherwise analyze mechanism before spending full-eval compute.'
        ),
    }
    # Baseline deltas vs leader.
    for rec in refs.values():
        rec['delta_vs_visible_leader'] = {k: rec['scores'][k] - leader['scores'][k] for k in CHEAP7 + ['SuperGLUE','AoA']}
        rec['cheap7_delta_vs_visible_leader'] = rec['cheap7'] - leader['cheap7']
    summary = {
        'status': 'LAMB_SCORE_CONTEXT',
        'purpose': 'Correct comparison frame for research LAMB×curriculum endpoints: cheap7 vectors against matched cheap7 vectors.',
        'cheap7_columns': CHEAP7,
        'references': refs,
        'visible_leader': leader,
        'best_matched_baseline_key': best_baseline_key,
        'thresholds': thresholds,
        'interpretation': {
            'do_not_compare_to_full_overall_only': True,
            'why': 'Full Overall includes SuperGLUE and AoA; cheap7 is the immediate cheap endpoint readout and must be compared to cheap7 references.',
            'coordinate': 'joint LAMB×sequence-curriculum×chunking×masking intervention until matched LAMB-only and curriculum-only controls exist.',
        },
    }
    sp = OUT / 'lamb_score_context.json'
    sp.write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')

    lines = ['# research — score context for research LAMB endpoints', '']
    lines.append('research endpoints must be interpreted by cheap7 vectors before any full official conclusion.')
    lines.append('')
    lines.append('| reference | BLiMP | Supplement | EWoK | Entity | COMPS | GlobalPIQA | Reading | cheap7 | SuperGLUE | Overall |')
    lines.append('|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|')
    for key, rec in refs.items():
        s = rec['scores']
        lines.append(f"| {key} | {s['BLiMP']:.3f} | {s['Supplement']:.3f} | {s['EWoK']:.3f} | {s['Entity']:.3f} | {s['COMPS']:.3f} | {s['GlobalPIQA']:.3f} | {s['Reading']:.3f} | {rec['cheap7']:.3f} | {s['SuperGLUE']:.3f} | {rec['Overall']:.3f} |")
    s = leader['scores']
    lines.append(f"| visible_leader_41p80 | {s['BLiMP']:.3f} | {s['Supplement']:.3f} | {s['EWoK']:.3f} | {s['Entity']:.3f} | {s['COMPS']:.3f} | {s['GlobalPIQA']:.3f} | {s['Reading']:.3f} | {leader['cheap7']:.3f} | {s['SuperGLUE']:.3f} | {leader['Overall']:.3f} |")
    lines.append('')
    lines.append(f"Best matched baseline cheap7: **{best_baseline_key} = {best_baseline['cheap7']:.6f}**.")
    lines.append(f"Visible leader cheap7: **{leader['cheap7']:.6f}** (gap from best matched baseline {leader['cheap7'] - best_baseline['cheap7']:.6f}).")
    lines.append('')
    lines.append('Cheap7 needed to reach Overall 41.80 with AoA=0 depends on SuperGLUE:')
    lines.append(f"- if SuperGLUE equals 8×480 baseline ({refs['legal40k_8x480_adamw_fixed256_seed43022']['scores']['SuperGLUE']:.3f}): cheap7 > {thresholds['cheap7_needed_for_overall_41p80_if_superglue_equals_8x480_baseline']:.3f}")
    lines.append(f"- if SuperGLUE equals 12×384 baseline ({refs['legal40k_12x384_adamw_fixed256_seed43022']['scores']['SuperGLUE']:.3f}): cheap7 > {thresholds['cheap7_needed_for_overall_41p80_if_superglue_equals_12x384_baseline']:.3f}")
    lines.append(f"- if SuperGLUE equals visible leader ({leader['scores']['SuperGLUE']:.3f}): cheap7 > {thresholds['cheap7_needed_for_overall_41p80_if_superglue_equals_visible_leader']:.3f}")
    lines.append('')
    lines.append('Full official evaluation is scientifically justified only if cheap7 is near or above this crossing region, or if the per-column vector shows a plausible SuperGLUE-backed crossing. Otherwise first analyze the mechanism and do matched cheaper controls.')
    lines.append('')
    lines.append(f"Summary JSON: `{sp}`")
    NOTE.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(json.dumps({'status': summary['status'], 'summary': str(sp), 'note': str(NOTE), 'best_baseline_cheap7': best_baseline['cheap7'], 'leader_cheap7': leader['cheap7']}, indent=2), flush=True)


if __name__ == '__main__':
    main()
