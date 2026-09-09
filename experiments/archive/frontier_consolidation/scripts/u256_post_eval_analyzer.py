#!/usr/bin/env python3
"""research: post-evaluation analyzer for U256 100M full official surface.

CPU/filesystem only.  Run after the U256 hardened full evaluator has completed.
It verifies the nine-column arithmetic, backfills official_overall into the staged
payload if needed, and runs the saved-prediction item-flip comparator against
research 100M.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

USER_ROOT = Path('.').resolve()
DEFAULT_BASE_OUT = USER_ROOT / 'experiments/archive/frontier_consolidation/data/u256_100M_full_eval_hardened'
DEFAULT_SUMMARY = DEFAULT_BASE_OUT / 'summary/u256_100M_full_eval_hardened_summary.json'
DEFAULT_TARGET = 'U256_100M_seed43022'
DEFAULT_CAND_PAYLOAD = DEFAULT_BASE_OUT / 'staged_full_eval/per_target' / f'{DEFAULT_TARGET}.json'
DEFAULT_BASE_PAYLOAD = USER_ROOT / 'experiments/archive/frontier_consolidation/data/compliant_full_eval/per_target/complianttok_reinvest_seed43022.json'
DEFAULT_COLLATE = DEFAULT_BASE_OUT / 'collate' / DEFAULT_TARGET / f'pristine_collate_{DEFAULT_TARGET}_summary.json'
COMPARATOR = USER_ROOT / 'experiments/archive/frontier_consolidation/scripts/pairwise_item_flip_analysis.py'
DEFAULT_OUT = USER_ROOT / 'experiments/archive/frontier_consolidation/data/u256_100m_post_eval_analysis'
REF = {
    'BLiMP': 65.8707181799453,
    'Supplement': 61.16566092036889,
    'EWoK': 50.39323748109589,
    'Entity': 27.400833994026197,
    'COMPS': 52.00834536316919,
    'SuperGLUE': 70.27986764740969,
    'GlobalPIQA': 36.0631067961165,
    'Reading': 8.13816768550987,
    'AoA': 0.0,
    'Overall': 41.257770896404615,
    'cheap7': 43.00572463231884,
}
CHEAP_COLS = ['BLiMP', 'Supplement', 'EWoK', 'Entity', 'COMPS', 'GlobalPIQA', 'Reading']
ALL_COLS = ['BLiMP', 'Supplement', 'EWoK', 'Entity', 'COMPS', 'SuperGLUE', 'GlobalPIQA', 'Reading', 'AoA']


def now() -> str:
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(USER_ROOT))
    except ValueError:
        return str(path)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding='utf-8'))


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


def mean(xs: list[float]) -> float:
    return sum(float(x) for x in xs) / len(xs)


def close(a: float, b: float, tol: float = 1e-6) -> bool:
    return math.isfinite(a) and math.isfinite(b) and abs(a - b) <= tol


def require_path(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError({'missing': label, 'path': rel(path)})


def extract_scores(summary: dict[str, Any], collate: dict[str, Any] | None) -> tuple[dict[str, float], float]:
    scores = summary.get('scores')
    overall = summary.get('Overall')
    if isinstance(scores, dict) and overall is not None:
        return {k: float(v) for k, v in scores.items()}, float(overall)
    if collate:
        score_summary = collate.get('score_summary', {})
        cscores = score_summary.get('scores')
        coverall = score_summary.get('Overall')
        if isinstance(cscores, dict) and coverall is not None:
            return {k: float(v) for k, v in cscores.items()}, float(coverall)
    raise RuntimeError('Could not extract official score vector from summary/collate JSON')


def summarize_validation(summary: dict[str, Any], collate: dict[str, Any] | None, scores: dict[str, float], overall: float) -> dict[str, Any]:
    errors: list[str] = []
    missing_cols = [c for c in ALL_COLS if c not in scores]
    if missing_cols:
        errors.append(f'missing score columns {missing_cols}')
    computed_overall = mean([scores[c] for c in ALL_COLS]) if not missing_cols else float('nan')
    computed_cheap7 = mean([scores[c] for c in CHEAP_COLS]) if not missing_cols else float('nan')
    if not close(computed_overall, overall, 2e-6):
        errors.append(f'overall arithmetic mismatch computed {computed_overall} vs reported {overall}')
    if 'cheap7' in summary and not close(computed_cheap7, float(summary['cheap7']), 2e-6):
        errors.append(f'cheap7 arithmetic mismatch computed {computed_cheap7} vs reported {summary["cheap7"]}')
    if 'overall_margin_vs_41p8' in summary and not close(overall - 41.8, float(summary['overall_margin_vs_41p8']), 2e-6):
        errors.append('margin_vs_41p8 mismatch')
    endpoint_ready = summary.get('endpoint_ready', {})
    if endpoint_ready:
        if int(endpoint_ready.get('word_exposure', -1)) != 100_000_000:
            errors.append(f'endpoint word_exposure not 100M: {endpoint_ready.get("word_exposure")}')
        # U256 trainer can report 2530 nominal total steps for the ten exact 10M epochs.
        if int(endpoint_ready.get('actual_training_steps', endpoint_ready.get('total_steps_executed', -1))) not in (2529, 2530):
            errors.append(f'unexpected endpoint steps: {endpoint_ready.get("actual_training_steps", endpoint_ready.get("total_steps_executed"))}')
        lf = endpoint_ready.get('loss_first')
        if lf is not None and abs(float(lf) - 9.826857208144903) > 2e-6:
            errors.append(f'U256 first loss changed: {lf}')
        ladder = endpoint_ready.get('aoa_ladder') or endpoint_ready.get('checkpoint_ladder') or endpoint_ready.get('aoa_ladder_present') or []
        if ladder and len(ladder) != 19:
            errors.append(f'AoA/checkpoint ladder length {len(ladder)} != 19')
        missing_ladder = endpoint_ready.get('aoa_ladder_missing') or []
        if missing_ladder:
            errors.append(f'AoA/checkpoint ladder missing entries: {missing_ladder}')
    if collate:
        c_errors = collate.get('validation_errors') or collate.get('errors') or []
        if c_errors:
            errors.append(f'collate validation_errors present: {c_errors}')
        score_summary = collate.get('score_summary', {})
        if score_summary:
            if score_summary.get('aoa_surprisal_num_steps') not in (None, 19):
                errors.append(f'collate aoa_surprisal_num_steps {score_summary.get("aoa_surprisal_num_steps")} != 19')
            row_values = score_summary.get('aoa_surprisal_row_count_values')
            if row_values not in (None, [8005]):
                errors.append(f'collate AoA row count values {row_values} != [8005]')
    deltas = {c: scores[c] - REF[c] for c in ALL_COLS if c in scores}
    return {
        'validation_errors': errors,
        'computed_overall': computed_overall,
        'computed_cheap7': computed_cheap7,
        'reported_overall': overall,
        'margin_vs_41p8': overall - 41.8,
        'score_signal': 'above_41p8' if overall >= 41.8 else 'below_41p8',
        'deltas_vs_step35': deltas,
        'overall_delta_vs_step35': overall - REF['Overall'],
        'cheap7_delta_vs_step35': computed_cheap7 - REF['cheap7'],
        'columns_sorted_by_delta': sorted(deltas.items(), key=lambda kv: kv[1]),
    }


def patch_candidate_payload(path: Path, scores: dict[str, float], overall: float) -> dict[str, Any]:
    require_path(path, 'candidate staged payload')
    payload = read_json(path)
    before = payload.get('official_overall')
    needs_patch = not (isinstance(before, dict) and isinstance(before.get('scores'), dict) and before.get('Overall') is not None)
    if needs_patch:
        payload['official_overall'] = {
            'scores': scores,
            'complete_for_provisional_overall': True,
            'Overall': overall,
            'NLP_average': mean([scores[k] for k in ['BLiMP', 'Supplement', 'EWoK', 'Entity', 'COMPS', 'GlobalPIQA', 'SuperGLUE']]),
            'Human_like_average': mean([scores[k] for k in ['Reading', 'AoA']]),
            'official_like_arithmetic': 'mean(BLiMP, Supplement, EWoK, Entity, COMPS, SuperGLUE, GlobalPIQA, Reading, AoA_leaderboard_score)',
            'aoa_leaderboard_score': scores.get('AoA'),
            'patched_by': 'u256_post_eval_analyzer',
            'patched_utc': now(),
        }
        payload['finished_utc'] = now()
        write_json(path, payload)
    else:
        old_scores = {k: float(v) for k, v in before.get('scores', {}).items() if k in scores}
        score_max_abs_diff = max([abs(old_scores[k] - scores[k]) for k in old_scores] or [0.0])
        if score_max_abs_diff > 1e-6 or abs(float(before.get('Overall')) - overall) > 1e-6:
            raise RuntimeError({'candidate_payload_official_overall_disagrees_with_summary': True, 'payload': rel(path), 'max_score_abs_diff': score_max_abs_diff, 'payload_overall': before.get('Overall'), 'summary_overall': overall})
    return {'payload': rel(path), 'patched': needs_patch, 'had_official_overall_before': before is not None}


def run_item_flips(base_payload: Path, cand_payload: Path, out_dir: Path, label: str) -> dict[str, Any]:
    require_path(COMPARATOR, 'item flip comparator')
    cmd = [sys.executable, '-B', str(COMPARATOR), '--base', str(base_payload), '--candidate', str(cand_payload), '--out-dir', str(out_dir / 'item_flips'), '--label', label]
    proc = subprocess.run(cmd, cwd=str(USER_ROOT), env={**os.environ, 'TOKENIZERS_PARALLELISM': 'false'}, text=True, capture_output=True, timeout=1800)
    rec = {'cmd': cmd, 'returncode': proc.returncode, 'stdout_tail': proc.stdout[-4000:], 'stderr_tail': proc.stderr[-4000:], 'out_json': rel(out_dir / 'item_flips' / f'{label}.json'), 'out_md': rel(out_dir / 'item_flips' / f'{label}.md')}
    if proc.returncode != 0:
        raise RuntimeError(rec)
    if (out_dir / 'item_flips' / f'{label}.json').exists():
        j = read_json(out_dir / 'item_flips' / f'{label}.json')
        rec['aggregate'] = j.get('aggregate')
        rec['cheap_payload'] = j.get('cheap_payload')
    return rec


def write_note(out_dir: Path, validation: dict[str, Any], scores: dict[str, float], patch: dict[str, Any], flips: dict[str, Any] | None) -> None:
    lines: list[str] = []
    lines.append('# research — U256 100M post-evaluation analysis')
    lines.append('')
    lines.append(f"Overall: **{validation['reported_overall']:.6f}**; margin vs 41.8: **{validation['margin_vs_41p8']:+.6f}**; score signal: `{validation['score_signal']}`.")
    lines.append(f"cheap7: **{validation['computed_cheap7']:.6f}**; cheap7 delta vs research: **{validation['cheap7_delta_vs_step35']:+.6f}**; Overall delta vs research: **{validation['overall_delta_vs_step35']:+.6f}**.")
    lines.append('')
    lines.append('| Column | U256 100M | research 100M | Delta |')
    lines.append('|---|---:|---:|---:|')
    for c in ALL_COLS:
        lines.append(f"| {c} | {scores[c]:.6f} | {REF[c]:.6f} | {validation['deltas_vs_step35'][c]:+.6f} |")
    lines.append('')
    if validation['validation_errors']:
        lines.append('Validation errors:')
        for e in validation['validation_errors']:
            lines.append(f'- {e}')
    else:
        lines.append('Validation arithmetic and endpoint/collation checks reported no errors.')
    lines.append('')
    lines.append(f"Candidate payload backfill: patched={patch['patched']}, path=`{patch['payload']}`.")
    if flips:
        lines.append(f"Item-flip analysis JSON: `{flips['out_json']}`")
        if flips.get('aggregate'):
            lines.append(f"Item-flip aggregate: {flips['aggregate']}")
    lines.append('')
    lines.append('Interpret U256 together with `notes/u256_visibility_profile.md`: the mechanism is small, source-tail-concentrated visibility recovery, not new text or a new objective.')
    (out_dir / 'u256_100m_post_eval_analysis.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


def main() -> None:
    p = argparse.ArgumentParser(description='Analyze completed U256 100M full eval artifacts')
    p.add_argument('--summary-json', type=Path, default=DEFAULT_SUMMARY)
    p.add_argument('--collate-summary-json', type=Path, default=DEFAULT_COLLATE)
    p.add_argument('--candidate-payload', type=Path, default=DEFAULT_CAND_PAYLOAD)
    p.add_argument('--base-payload', type=Path, default=DEFAULT_BASE_PAYLOAD)
    p.add_argument('--out-dir', type=Path, default=DEFAULT_OUT)
    p.add_argument('--label', default='u256_100M_vs_step35_100M')
    p.add_argument('--skip-item-flips', action='store_true')
    args = p.parse_args()
    for attr in ['summary_json', 'collate_summary_json', 'candidate_payload', 'base_payload', 'out_dir']:
        val = getattr(args, attr)
        if isinstance(val, Path) and not val.is_absolute():
            setattr(args, attr, USER_ROOT / val)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    require_path(args.summary_json, 'U256 hardened summary')
    require_path(args.base_payload, 'research full payload')
    require_path(args.candidate_payload, 'candidate staged payload')
    summary = read_json(args.summary_json)
    collate = read_json(args.collate_summary_json) if args.collate_summary_json.exists() else None
    scores, overall = extract_scores(summary, collate)
    validation = summarize_validation(summary, collate, scores, overall)
    patch = patch_candidate_payload(args.candidate_payload, scores, overall)
    flips = None if args.skip_item_flips else run_item_flips(args.base_payload, args.candidate_payload, args.out_dir, args.label)
    result = {
        'status': 'U256_100M_POST_EVAL_ANALYSIS',
        'created_utc': now(),
        'inputs': {'summary_json': rel(args.summary_json), 'collate_summary_json': rel(args.collate_summary_json), 'candidate_payload': rel(args.candidate_payload), 'base_payload': rel(args.base_payload)},
        'scores': scores,
        'validation': validation,
        'candidate_payload_patch': patch,
        'item_flip_analysis': flips,
        'mechanism_profile_note': 'research/notes/frontier_consolidation/u256_visibility_profile.md',
        'outputs': {'json': rel(args.out_dir / 'u256_100m_post_eval_analysis.json'), 'md': rel(args.out_dir / 'u256_100m_post_eval_analysis.md')},
    }
    write_json(args.out_dir / 'u256_100m_post_eval_analysis.json', result)
    write_note(args.out_dir, validation, scores, patch, flips)
    if validation['validation_errors']:
        print(json.dumps({'status': result['status'], 'score_signal': validation['score_signal'], 'Overall': overall, 'validation_errors': validation['validation_errors'], 'out_json': result['outputs']['json']}, indent=2), flush=True)
        raise SystemExit(2)
    print(json.dumps({'status': result['status'], 'score_signal': validation['score_signal'], 'Overall': overall, 'margin_vs_41p8': validation['margin_vs_41p8'], 'cheap7': validation['computed_cheap7'], 'patched_payload': patch['patched'], 'out_json': result['outputs']['json']}, indent=2), flush=True)


if __name__ == '__main__':
    main()
