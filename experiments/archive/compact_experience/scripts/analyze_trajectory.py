#!/usr/bin/env python3
"""Analyze research 100M masking-curriculum trajectories and write compact evidence files."""
from __future__ import annotations

import json
import pathlib
from typing import Any, Dict, Optional

SUMMARY = pathlib.Path('experiments/archive/compact_experience/data/curriculum_100M_eval/curriculum_trajectory_eval_summary.json')
FINAL = pathlib.Path('experiments/archive/compact_experience/data/curriculum_100M_eval/curriculum_100M_eval_summary.json')
OUT_MD = pathlib.Path('research/notes/compact_experience/masking_curriculum_trajectory_mechanism.md')
OUT_CSV = pathlib.Path('experiments/archive/compact_experience/data/curriculum_100M_eval/trajectory_compact_table.csv')

CKS = ['chck_60M', 'chck_70M', 'chck_80M', 'chck_100M']
ARMS = ['wwm_fixed', 'wwm_to_token', 'amlm_hard_switch']
COLS = ['BLiMP', 'Supplement', 'EWoK', 'Entity', 'COMPS', 'GlobalPIQA_mean', 'Reading']


def fmt(x: Optional[float]) -> str:
    return 'NA' if x is None else f'{x:.3f}'


def csv_cell(x: Any) -> str:
    if x is None:
        return ''
    s = str(x)
    if any(ch in s for ch in ',"\n'):
        return '"' + s.replace('"', '""') + '"'
    return s


def main() -> None:
    data = json.loads(SUMMARY.read_text(encoding='utf-8'))
    final_data = json.loads(FINAL.read_text(encoding='utf-8'))
    table: Dict[str, Dict[str, Optional[float]]] = data['table']
    absproxy: Dict[str, Optional[float]] = data['absolute_weighted_fast_proxy']
    contrasts: Dict[str, Dict[str, Any]] = data['trajectory_contrasts']
    runs = data['run_summaries']

    dyn: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for arm, rec in runs.items():
        dyn[arm] = {d['checkpoint']: d for d in rec.get('dynamics_by_checkpoint', [])}

    rows = [['arm', 'checkpoint', 'mask_mode', 'mask_prob', 'effective_mask_rate', 'loss_last', 'weighted_fast_proxy'] + COLS]
    for arm in ARMS:
        for ck in CKS:
            label = f'{arm}__{ck}'
            t = table.get(label, {})
            d = dyn.get(arm, {}).get(ck, {})
            rows.append([
                arm,
                ck,
                d.get('mask_mode_at_checkpoint'),
                d.get('mask_prob_at_checkpoint'),
                d.get('effective_mask_rate_mean'),
                runs.get(arm, {}).get('loss_last'),
                absproxy.get(label),
            ] + [t.get(c) for c in COLS])

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    OUT_CSV.write_text('\n'.join(','.join(csv_cell(x) for x in row) for row in rows) + '\n', encoding='utf-8')

    lines = []
    lines.append('# research — 100M masking-curriculum trajectory mechanism note')
    lines.append('')
    lines.append(f'Trajectory summary JSON: `{SUMMARY}`')
    lines.append(f'Final-checkpoint summary JSON: `{FINAL}`')
    lines.append(f'Compact CSV: `{OUT_CSV}`')
    lines.append('')
    lines.append('## Absolute trajectory scores')
    lines.append('')
    lines.append('| arm | checkpoint | mode | eff mask | BLiMP | Supp | EWoK | Entity | COMPS | GPIQA | Reading | proxy |')
    lines.append('|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|')
    for arm in ARMS:
        for ck in CKS:
            label = f'{arm}__{ck}'
            t = table.get(label, {})
            d = dyn.get(arm, {}).get(ck, {})
            lines.append(
                f"| {arm} | {ck} | {d.get('mask_mode_at_checkpoint', '?')} | {fmt(d.get('effective_mask_rate_mean'))} | "
                f"{fmt(t.get('BLiMP'))} | {fmt(t.get('Supplement'))} | {fmt(t.get('EWoK'))} | {fmt(t.get('Entity'))} | "
                f"{fmt(t.get('COMPS'))} | {fmt(t.get('GlobalPIQA_mean'))} | {fmt(t.get('Reading'))} | {fmt(absproxy.get(label))} |"
            )
    lines.append('')
    lines.append('## Contrasts versus fixed WWM by checkpoint')
    lines.append('')
    lines.append('| checkpoint | wwm→token proxy Δ | wwm→token Supp Δ | wwm→token Entity Δ | wwm→token GPIQA Δ | AMLM+switch proxy Δ | AMLM+switch Supp Δ | AMLM+switch Entity Δ | AMLM+switch GPIQA Δ |')
    lines.append('|---|---:|---:|---:|---:|---:|---:|---:|---:|')
    for ck in CKS:
        wt = contrasts.get(f'wwm_to_token_minus_wwm_fixed__{ck}', {})
        ah = contrasts.get(f'amlm_hard_switch_minus_wwm_fixed__{ck}', {})
        wd = wt.get('delta', {})
        ad = ah.get('delta', {})
        lines.append(
            f"| {ck} | {fmt(wt.get('weighted_fast_proxy_delta'))} | {fmt(wd.get('Supplement'))} | {fmt(wd.get('Entity'))} | {fmt(wd.get('GlobalPIQA_mean'))} | "
            f"{fmt(ah.get('weighted_fast_proxy_delta'))} | {fmt(ad.get('Supplement'))} | {fmt(ad.get('Entity'))} | {fmt(ad.get('GlobalPIQA_mean'))} |"
        )
    lines.append('')
    lines.append('## Final 100M contrast context')
    lines.append('')
    for name, val in final_data.get('weighted_fast_proxy_delta', {}).items():
        lines.append(f'- `{name}`: weighted fast proxy Δ {fmt(val)}')
    lines.append('')
    lines.append('## Mechanistic reading')
    lines.append('')
    for ck in CKS:
        wt = contrasts.get(f'wwm_to_token_minus_wwm_fixed__{ck}', {})
        wd = wt.get('delta', {})
        ah = contrasts.get(f'amlm_hard_switch_minus_wwm_fixed__{ck}', {})
        ad = ah.get('delta', {})
        lines.append(
            f"- `{ck}`: WWM→token proxy Δ {fmt(wt.get('weighted_fast_proxy_delta'))}, Supplement Δ {fmt(wd.get('Supplement'))}, "
            f"Entity Δ {fmt(wd.get('Entity'))}, GPIQA Δ {fmt(wd.get('GlobalPIQA_mean'))}; AMLM+switch proxy Δ {fmt(ah.get('weighted_fast_proxy_delta'))}, "
            f"Supplement Δ {fmt(ad.get('Supplement'))}, Entity Δ {fmt(ad.get('Entity'))}, GPIQA Δ {fmt(ad.get('GlobalPIQA_mean'))}."
        )
    lines.append('')
    lines.append('The WWM→token arm is scheduled like fixed WWM through `chck_70M`; differences before `chck_80M` therefore estimate stochastic/training-run variation under nominally identical masking rather than a token-switch effect. The true post-switch comparison is `chck_80M` and `chck_100M`: after switching to token-level masking, WWM→token gains BLiMP and GlobalPIQA but loses Supplement, Entity, and Reading enough that the weighted fast proxy stays below fixed WWM. AMLM+switch shows the same broad pattern, with difficulty weighting partly protecting Entity at 100M but not repairing the Supplement/Reading penalty.')
    lines.append('')
    lines.append('This supports a trade-off mechanism rather than a uniform curriculum improvement: token-level prediction and hard-token emphasis strengthen some local lexical/syntactic or plausibility signals, but they appear to reduce the whole-word/phrase-level structural pressure that benefits Supplement and reading-style measures under the current baseline16k AdamW DeBERTa recipe. The next useful construction should not blindly copy the leaderboard switch. It should either replicate fixed versus switch with additional seeds to quantify the effect, or build a gentler/conditional granularity transition that preserves WWM pressure on structure-sensitive examples while adding token-level pressure where BLiMP/GPIQA benefit.')
    lines.append('')
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text('\n'.join(lines), encoding='utf-8')
    print(json.dumps({'note': str(OUT_MD), 'csv': str(OUT_CSV), 'rows': len(rows) - 1}, indent=2))


if __name__ == '__main__':
    main()
