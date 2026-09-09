#!/usr/bin/env python3
"""research: scale1.75 item-family trajectory readout from saved predictions.

CPU/filesystem only.  Reads existing item-flip JSONs at 20/50/70/80/100M and
summarizes how selected groups move relative to the matched research checkpoints.
No model is loaded and no evaluation is run.
"""
from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Any

ROOT = Path('.').resolve()
OUT_DIR = ROOT / 'experiments/archive/frontier_consolidation/data/scale1p75_family_trajectory_readout'
OUT_JSON = OUT_DIR / 'scale1p75_family_trajectory_readout.json'
OUT_MD = (ROOT / 'research/documents/frontier_consolidation/data/scale1p75_family_trajectory_readout/scale1p75_family_trajectory_readout.md')
OUT_CSV = OUT_DIR / 'selected_group_trajectory.csv'

INPUTS = {
    '20M': ROOT / 'experiments/archive/frontier_consolidation/data/scale1p75_20m_item_flips/scale1p75_vs_20M.json',
    '50M': ROOT / 'experiments/archive/frontier_consolidation/data/scale1p75_family_recovery_decision/scale1p75_vs_50M.json',
    '70M': ROOT / 'experiments/archive/frontier_consolidation/data/scale1p75_family_recovery_decision/scale1p75_vs_70M.json',
    '80M': ROOT / 'experiments/archive/frontier_consolidation/data/scale1p75_family_recovery_decision/scale1p75_vs_80M.json',
    '100M': ROOT / 'experiments/archive/frontier_consolidation/data/scale1p75_100m_item_flips_completed_columns/scale1p75_100M_completed_columns_vs_100M.json',
}

SELECTED = {
    'BLiMP': [
        'existential_there_quantifiers_2',
        'wh_questions_object_gap',
        'principle_A_reconstruction',
        'matrix_question_npi_licensor_present',
        'only_npi_licensor_present',
        'npi_present_1',
        'npi_present_2',
        'left_branch_island_echo_question',
        'adjunct_island',
        'wh_vs_that_with_gap',
        'existential_there_object_raising',
    ],
    'Supplement': ['subject_aux_inversion', 'qa_congruence_tricky', 'qa_congruence_easy', 'hypernym', 'turn_taking'],
    'EWoK': ['material-properties', 'material-dynamics', 'social-interactions', 'social-relations', 'spatial-relations', 'quantitative-properties', 'physical-dynamics', 'physical-relations', 'social-properties'],
    'Entity': ['regular_0_ops', 'regular_1_ops', 'regular_2_ops', 'regular_3_ops', 'regular_4_ops', 'regular_5_ops', 'move_contents_0_ops', 'move_contents_5_ops', 'ambiref_3_ops', 'ambiref_4_ops', 'ambiref_5_ops'],
    'COMPS': ['base', 'wugs', 'wugs_dist_before', 'wugs_dist_in_between'],
    'GlobalPIQA': ['GlobalPIQA_parallel', 'GlobalPIQA_nonparallel'],
}

def now() -> str:
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())

def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except Exception:
        return str(path)

def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding='utf-8'))

def group_lookup(data: dict[str, Any], column: str, group: str) -> dict[str, Any] | None:
    col = data.get('columns', {}).get(column, {})
    row = col.get('groups_by_name', {}).get(group)
    if isinstance(row, dict):
        return row
    for row in col.get('groups_all_by_item_net', []) or []:
        if row.get('group') == group:
            return row
    return None

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    loaded: dict[str, dict[str, Any]] = {}
    for exposure, path in INPUTS.items():
        if not path.exists():
            raise FileNotFoundError({'missing_input': rel(path), 'exposure': exposure})
        loaded[exposure] = read_json(path)

    exposure_summary = []
    for exposure, data in loaded.items():
        cheap = data.get('cheap_payload', {})
        aggregate = data.get('aggregate', {})
        exposure_summary.append({
            'exposure': exposure,
            'candidate_cheap7': cheap.get('candidate_cheap7'),
            'base_cheap7': cheap.get('base_cheap7'),
            'cheap7_delta': cheap.get('cheap7_delta'),
            'deltas': cheap.get('deltas'),
            'discrete_reconstructed_mean_delta': aggregate.get('discrete_reconstructed_mean_delta'),
            'total_gain_items': aggregate.get('total_gain_items'),
            'total_loss_items': aggregate.get('total_loss_items'),
            'total_gain_minus_loss': aggregate.get('total_gain_minus_loss'),
        })

    rows = []
    for column, groups in SELECTED.items():
        for group in groups:
            for exposure, data in loaded.items():
                row = group_lookup(data, column, group)
                if row is None:
                    rows.append({'column': column, 'group': group, 'exposure': exposure, 'missing': True})
                    continue
                rows.append({
                    'column': column,
                    'group': group,
                    'exposure': exposure,
                    'missing': False,
                    'n': row.get('n'),
                    'gain': row.get('gain'),
                    'loss': row.get('loss'),
                    'net_gain_minus_loss': row.get('net_gain_minus_loss'),
                    'net_pct': row.get('net_pct'),
                    'base_item_pct': row.get('base_item_pct'),
                    'cand_item_pct': row.get('cand_item_pct'),
                })

    def series(column: str, group: str, field: str = 'net_pct') -> list[Any]:
        return [next((r.get(field) for r in rows if r['column'] == column and r['group'] == group and r['exposure'] == e), None) for e in INPUTS]

    interpretation = {
        'cheap7_path': [s['cheap7_delta'] for s in exposure_summary],
        'cheap7_exposures': list(INPUTS),
        'scale1p75_100M_surface': exposure_summary[-1],
        'persistent_100M_positive_examples': {
            'BLiMP existential_there_quantifiers_2 net_pct': series('BLiMP', 'existential_there_quantifiers_2'),
            'BLiMP wh_questions_object_gap net_pct': series('BLiMP', 'wh_questions_object_gap'),
            'BLiMP only_npi_licensor_present net_pct': series('BLiMP', 'only_npi_licensor_present'),
            'COMPS wugs_dist_in_between net_pct': series('COMPS', 'wugs_dist_in_between'),
        },
        'persistent_or_recovered_negatives': {
            'EWoK material-properties net_pct': series('EWoK', 'material-properties'),
            'EWoK social-interactions net_pct': series('EWoK', 'social-interactions'),
            'EWoK spatial-relations net_pct': series('EWoK', 'spatial-relations'),
            'Supplement subject_aux_inversion net_pct': series('Supplement', 'subject_aux_inversion'),
            'COMPS wugs_dist_before net_pct': series('COMPS', 'wugs_dist_before'),
            'Entity regular_5_ops net_pct': series('Entity', 'regular_5_ops'),
        },
    }

    out = {
        'status': 'SCALE1P75_FAMILY_TRAJECTORY_READOUT',
        'created_utc': now(),
        'inputs': {k: rel(v) for k, v in INPUTS.items()},
        'exposure_summary': exposure_summary,
        'selected_rows': rows,
        'interpretation': interpretation,
    }
    OUT_JSON.write_text(json.dumps(out, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    with OUT_CSV.open('w', newline='', encoding='utf-8') as f:
        cols = ['column', 'group', 'exposure', 'missing', 'n', 'gain', 'loss', 'net_gain_minus_loss', 'net_pct', 'base_item_pct', 'cand_item_pct']
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c) for c in cols})

    lines = ['# research — scale1.75 family trajectory readout', '', 'Readout from saved item-flip JSONs only; no model inference.', '', '## cheap7 path', '', '| exposure | base cheap7 | scale1.75 cheap7 | delta | discrete mean delta | net items |', '|---|---:|---:|---:|---:|---:|']
    for s in exposure_summary:
        lines.append(f"| {s['exposure']} | {float(s['base_cheap7']):.6f} | {float(s['candidate_cheap7']):.6f} | {float(s['cheap7_delta']):+.6f} | {float(s['discrete_reconstructed_mean_delta']):+.6f} | {int(s['total_gain_minus_loss']):+d} |")
    lines += ['', '## Selected group net percentages (20M, 50M, 70M, 80M, 100M)', '', '| column | group | path |', '|---|---|---:|']
    for column, groups in SELECTED.items():
        for group in groups:
            vals = series(column, group)
            txt = ', '.join('NA' if v is None else f'{float(v):+.2f}' for v in vals)
            lines.append(f'| {column} | {group} | {txt} |')
    lines += ['', '## Scientific reading', '', '- The 100M endpoint keeps a large BLiMP rotation and small gains in several cheap columns, but the cheap surface is below the 80M high point and needs SuperGLUE+AoA ≈ 71.405 to reach Overall 41.8.', '- The EWoK material/social/spatial relation losses persist at 100M even though material-dynamics itself is no longer the dominant loss; GlobalPIQA is neutral by net item count.', '- Entity high-operation gains seen at 50M are not preserved as a strong 100M advantage; regular_5_ops is negative.', '', f'JSON: `{rel(OUT_JSON)}`', f'CSV: `{rel(OUT_CSV)}`']
    OUT_MD.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(json.dumps({'status': out['status'], 'out_json': rel(OUT_JSON), 'out_md': rel(OUT_MD), 'out_csv': rel(OUT_CSV), 'cheap7_path': interpretation['cheap7_path']}, indent=2), flush=True)

if __name__ == '__main__':
    main()
