#!/usr/bin/env python3
"""Build the research BabyLM Strict-Small scoreboard snapshot.

Purpose: keep the official 9/9 Overall target visible while mechanism work proceeds.
The script uses locally verified files where available and marks missing/unverified
columns explicitly rather than treating partial coordinates as SOTA progress.
"""
from __future__ import annotations
import json
import pathlib
from typing import Any

ROOT = pathlib.Path('experiments/archive/initial_model_studies')
DATA = ROOT / 'data'
OUT_JSON = DATA / 'babylm_scoreboard.json'
OUT_MD = (ROOT.parents[2] / 'research/notes/initial_model_studies/babylm_scoreboard.md')

COLUMNS = ['BLiMP','BLiMP Supplement','EWoK','Entity Tracking','COMPS','(Super)GLUE','GlobalPIQA','Reading','AoA']


def load(path: pathlib.Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def overall(scores: dict[str, float]) -> float | None:
    if not all(c in scores for c in COLUMNS):
        return None
    return sum(scores[c] for c in COLUMNS) / 9.0


def known_sum(scores: dict[str, float]) -> float:
    return sum(scores.values())


def row(name: str, kind: str, scores: dict[str, float], provenance: dict[str, str], notes: str) -> dict[str, Any]:
    missing = [c for c in COLUMNS if c not in scores]
    return {
        'name': name,
        'kind': kind,
        'scores': scores,
        'missing_columns': missing,
        'known_sum': known_sum(scores),
        'overall_9of9': overall(scores),
        'provenance': provenance,
        'notes': notes,
    }


def main():
    protected = load(DATA / 'debertav2_b256_true_9of9_coordinate.json')
    prot_scores = protected['scores_official_columns']

    leader_avail = load(DATA / 'public_leader_available_coordinate.json')
    leader_ewok = load(DATA / 'public_leader_full_ewok_word_tokenize_score.json')
    # SuperGLUE and AoA from parsed official leaderboard snapshot / public row, not rescored in research available JSON.
    leader_scores = {
        'BLiMP': leader_avail['scores']['blimp'],
        'BLiMP Supplement': leader_avail['scores']['supplement'],
        'EWoK': leader_ewok['ewok_full_score'],
        'Entity Tracking': leader_avail['scores']['entity_tracking'],
        'COMPS': leader_avail['scores']['comps'],
        '(Super)GLUE': 69.79,
        'GlobalPIQA': leader_avail['derived_columns']['GlobalPIQA_mean_parallel_nonparallel'],
        'Reading': leader_avail['derived_columns']['Reading_mean_eye_selfpaced'],
        'AoA': 0.0,
    }

    s1_avail = load(DATA / 's1_100m_available_coordinate.json')
    s1_ewok = load(DATA / 's1_100m_full_ewok_word_tokenize_score.json')
    s1_scores = {
        'BLiMP': s1_avail['scores']['blimp'],
        'BLiMP Supplement': s1_avail['scores']['supplement'],
        'EWoK': s1_ewok['ewok_full_score'],
        'Entity Tracking': s1_avail['scores']['entity_tracking'],
        'COMPS': s1_avail['scores']['comps'],
        'GlobalPIQA': s1_avail['derived_columns']['GlobalPIQA_mean_parallel_nonparallel'],
        'Reading': s1_avail['derived_columns']['Reading_mean_eye_selfpaced'],
    }

    s2_avail = load(DATA / 'true_s2_100m_available_coordinate.json')
    s2_ewok = load(DATA / 'true_s2_100m_full_ewok_word_tokenize_score.json')
    s2_scores = {
        'BLiMP': s2_avail['scores']['blimp'],
        'BLiMP Supplement': s2_avail['scores']['supplement'],
        'EWoK': s2_ewok['ewok_full_score'],
        'Entity Tracking': s2_avail['scores']['entity_tracking'],
        'COMPS': s2_avail['scores']['comps'],
        'GlobalPIQA': s2_avail['derived_columns']['GlobalPIQA_mean_parallel_nonparallel'],
        'Reading': s2_avail['derived_columns']['Reading_mean_eye_selfpaced'],
    }

    rows = [
        row('public_leader_go76dof_wwm_curriculum_simplification_40k','external_public_leader', leader_scores,
            {
                'BLiMP/Supp/Entity/COMPS/GlobalPIQA/Reading': 'research public checkpoint local rescoring',
                'EWoK': 'research full local EWoK word_tokenize',
                '(Super)GLUE/AoA': 'official leaderboard/public row values carried from earlier verified snapshot',
            },
            'Visible Strict-Small leader. Training data gated; not an internal model.'),
        row('protected_internal_debertav2_8x480_wwm_100M','protected_internal_complete_9of9', prot_scores,
            protected.get('evidence_files', {}),
            'Current protected best internally trained complete 9/9 coordinate.'),
        row('S1_leader_shape_12x384_baseline16k_100M','internal_partial_7of9', s1_scores,
            {'available_coordinate': str(DATA/'s1_100m_available_coordinate.json'), 'full_EWoK': str(DATA/'s1_100m_full_ewok_word_tokenize_score.json')},
            'Only 7/9 columns complete. Known-seven sum is below protected; would need unrealistic SuperGLUE+AoA to exceed leader.'),
        row('S2_true_curriculum_12x384_baseline16k_100M','internal_partial_7of9', s2_scores,
            {'available_coordinate': str(DATA/'true_s2_100m_available_coordinate.json'), 'full_EWoK': str(DATA/'true_s2_100m_full_ewok_word_tokenize_score.json')},
            'Only 7/9 columns complete. Curriculum harms most target columns; not SOTA direction.'),
    ]

    leader_total = sum(leader_scores.values())
    protected_total = sum(prot_scores.values())
    protected_gap = {
        'sum_gap_to_leader': leader_total - protected_total,
        'overall_gap_to_leader': leader_total/9.0 - protected_total/9.0,
        'protected_minus_leader_by_column': {c: prot_scores[c] - leader_scores[c] for c in COLUMNS},
    }
    for r in rows:
        if r['overall_9of9'] is None:
            needed = leader_total - r['known_sum']
            r['needed_missing_sum_to_exceed_leader'] = needed
            if set(r['missing_columns']) == {'(Super)GLUE','AoA'}:
                r['needed_superglue_plus_aoa_to_exceed_leader'] = needed
                r['needed_jump_over_protected_superglue_plus_aoa'] = needed - (prot_scores['(Super)GLUE'] + prot_scores['AoA'])

    payload = {
        'status': 'BABYLM_STRICT_SMALL_SCOREBOARD',
        'columns': COLUMNS,
        'current_public_leader': rows[0],
        'current_protected_best_internal': rows[1],
        'protected_gap_to_public_leader': protected_gap,
        'rows': rows,
        'scoreboard_rule': 'Only rows with all nine columns have Overall. Partial coordinates cannot be treated as SOTA progress unless their missing-column requirement is numerically plausible and then verified.',
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + '\n')

    def fmt(v):
        return '' if v is None else f'{v:.3f}'
    lines = []
    lines.append('# research BabyLM Strict-Small 9/9 scoreboard\n')
    lines.append('Purpose: keep official Overall progress visible while mechanism work proceeds. Partial coordinates are recorded but not treated as SOTA.\n')
    lines.append('## Current complete coordinates\n')
    lines.append('| model | type | BLiMP | Supp | EWoK | Entity | COMPS | SG | GPIQA | Reading | AoA | Overall |')
    lines.append('|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|')
    for r in rows:
        s=r['scores']
        vals=[fmt(s.get(c)) for c in COLUMNS]
        lines.append(f"| {r['name']} | {r['kind']} | " + ' | '.join(vals) + f" | {fmt(r['overall_9of9'])} |")
    lines.append('\n## Protected internal best\n')
    lines.append(f"- Protected internal complete 9/9: **{rows[1]['name']}**, Overall **{rows[1]['overall_9of9']:.4f}**.\n")
    lines.append(f"- Public leader Overall: **{rows[0]['overall_9of9']:.4f}**. Gap: **{protected_gap['overall_gap_to_leader']:.4f} Overall** ({protected_gap['sum_gap_to_leader']:.3f} score-sum).\n")
    lines.append('- Column gaps protected-minus-leader:\n')
    for c,d in protected_gap['protected_minus_leader_by_column'].items():
        lines.append(f"  - {c}: {d:+.3f}")
    lines.append('\n## Partial-coordinate warning\n')
    for r in rows[2:]:
        lines.append(f"- {r['name']}: known sum {r['known_sum']:.3f}, missing {r['missing_columns']}; needs missing-column sum > {r['needed_missing_sum_to_exceed_leader']:.3f} to exceed leader.")
        if 'needed_superglue_plus_aoa_to_exceed_leader' in r:
            lines.append(f"  - Equivalent SG+AoA need: > {r['needed_superglue_plus_aoa_to_exceed_leader']:.3f}, which is {r['needed_jump_over_protected_superglue_plus_aoa']:+.3f} above protected SG+AoA.")
    lines.append('\n## Operational rule\n')
    lines.append('Every BabyLM-scale candidate must be added to this scoreboard with all nine columns, or with an explicit missing-column requirement. Mechanism experiments are valuable only if they lead to a candidate whose official 9/9 Overall is evaluated.\n')
    OUT_MD.write_text('\n'.join(lines) + '\n')
    print(json.dumps({'wrote_json': str(OUT_JSON), 'wrote_md': str(OUT_MD), 'protected_overall': rows[1]['overall_9of9'], 'leader_overall': rows[0]['overall_9of9'], 'gap': protected_gap['overall_gap_to_leader']}, indent=2))

if __name__ == '__main__':
    main()
