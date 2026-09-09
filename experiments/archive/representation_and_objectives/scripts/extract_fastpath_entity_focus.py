#!/usr/bin/env python3
"""Extract focused research fast-path item evidence for research route judgment."""
from __future__ import annotations

import json
import pathlib
import time
from statistics import mean
from typing import Any

ROOT = pathlib.Path('.').resolve()
A01 = ROOT / 'experiments/archive/representation_and_objectives'
IN_JSON = A01 / 'data/fastpath_pair_item_family_review/fastpath_pair_item_family_review.json'
OUT_DIR = A01 / 'data/fastpath_entity_focus'
OUT_JSON = OUT_DIR / 'fastpath_entity_focus.json'
OUT_MD = (ROOT / 'research/notes/representation_and_objectives/fastpath_entity_focus.md')
ENTITY_PREFIXES = ('move_contents_5', 'move_contents_4', 'move_contents_3', 'ambiref_4', 'ambiref_3', 'regular_5', 'regular_4')
EWOK_REL = {'material-dynamics','physical-dynamics','physical-interactions','physical-relations','spatial-relations','social-relations','social-interactions','agent-properties'}
DISCRETE_COLUMNS = ['BLiMP','Supplement','EWoK','Entity','COMPS','GlobalPIQA']


def now() -> str:
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())


def rel(p: pathlib.Path | str) -> str:
    try:
        return str(pathlib.Path(p).resolve().relative_to(ROOT))
    except Exception:
        return str(p)


def group_total(groups: list[dict[str, Any]]) -> dict[str, int | float | None]:
    n = sum(int(g.get('n', 0)) for g in groups)
    gain = sum(int(g.get('gain', 0)) for g in groups)
    loss = sum(int(g.get('loss', 0)) for g in groups)
    net = sum(int(g.get('net_gain_minus_loss', 0)) for g in groups)
    return {'n': n, 'gain': gain, 'loss': loss, 'net_gain_minus_loss': net, 'net_pct': None if n == 0 else 100.0 * net / n}


def top_groups(groups: list[dict[str, Any]], k: int = 10, reverse: bool = True) -> list[dict[str, Any]]:
    return sorted(groups, key=lambda g: int(g.get('net_gain_minus_loss', 0)), reverse=reverse)[:k]


def extract_comparison(label: str, comp: dict[str, Any]) -> dict[str, Any]:
    columns = comp.get('columns', {})
    col_table = {}
    for col in DISCRETE_COLUMNS:
        c = columns.get(col, {})
        col_table[col] = {
            'delta_score_payload': c.get('delta_score_payload'),
            'gain': c.get('flip_counts', {}).get('gain'),
            'loss': c.get('flip_counts', {}).get('loss'),
            'gain_minus_loss_items': c.get('gain_minus_loss_items'),
        }
    ent_groups = columns.get('Entity', {}).get('groups_all_by_item_net', [])
    high_ent = [g for g in ent_groups if str(g.get('group','')).startswith(ENTITY_PREFIXES)]
    ew_groups = columns.get('EWoK', {}).get('groups_all_by_item_net', [])
    ew_rel = [g for g in ew_groups if str(g.get('group','')) in EWOK_REL]
    out = {
        'label': label,
        'base': comp.get('base'),
        'candidate': comp.get('candidate'),
        'aggregate': comp.get('aggregate'),
        'columns': col_table,
        'entity_high_operation_total': group_total(high_ent),
        'entity_high_operation_groups': top_groups(high_ent, k=20, reverse=True),
        'entity_worst_groups': columns.get('Entity', {}).get('worst_groups_by_item_net', [])[:12],
        'entity_best_groups': columns.get('Entity', {}).get('best_groups_by_item_net', [])[:12],
        'ewok_relation_total': group_total(ew_rel),
        'ewok_relation_groups': ew_rel,
        'ewok_worst_groups': columns.get('EWoK', {}).get('worst_groups_by_item_net', [])[:12],
        'ewok_best_groups': columns.get('EWoK', {}).get('best_groups_by_item_net', [])[:12],
        'globalpiqa_groups': columns.get('GlobalPIQA', {}).get('groups_all_by_item_net', []),
    }
    return out


def main() -> None:
    data = json.loads(IN_JSON.read_text(encoding='utf-8'))
    comps = data.get('comparisons', {})
    labels = ['coherent_minus_spanbreak', 'coherent_minus_chck82', 'coherent_minus_shuffled86', 'shuffled86_minus_chck82']
    extracted = {label: extract_comparison(label, comps[label]) for label in labels if label in comps}
    out = {
        'status': 'FASTPATH_ENTITY_FOCUS_EXTRACTED',
        'created_utc': now(),
        'source_json': rel(IN_JSON),
        'score_table': data.get('scores'),
        'cheap7': data.get('cheap7'),
        'comparisons': extracted,
        'ordinary86_note': 'truthful_shuffled86 is a frozen-82M shuffled private-tail endpoint, not ordinary scale1.75 chck_86M; ordinary86 prediction payload is pending research evaluation if coherent passes SuperGLUE.',
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    lines = ['# research fast-path Entity focus from research item evidence', '', f"Status: **{out['status']}**", '', 'This note compresses the existing research item/family review before the ordinary scale1.75 chck_86M comparator exists. `shuffled86` here is the frozen-82M shuffled private-tail baseline, not ordinary continuation.', '']
    cheap = data.get('cheap7', {})
    lines += ['## Cheap7 reference', '', '| arm | cheap7 |', '|---|---:|']
    for arm in ['chck82','shuffled86','coherent4M','spanbreak4M']:
        lines.append(f"| {arm} | {cheap.get(arm)} |")
    for label, ex in extracted.items():
        lines += ['', f"## {label}", '', f"Base `{ex['base']}` -> candidate `{ex['candidate']}`", '', f"Aggregate: `{ex['aggregate']}`", '', '| column | score Δ | gains | losses | net items |', '|---|---:|---:|---:|---:|']
        for col in DISCRETE_COLUMNS:
            c = ex['columns'][col]
            lines.append(f"| {col} | {c.get('delta_score_payload')} | {c.get('gain')} | {c.get('loss')} | {c.get('gain_minus_loss_items')} |")
        lines += ['', f"Entity high-operation total: `{ex['entity_high_operation_total']}`", '', 'Entity high-operation groups:', '', '```json', json.dumps(ex['entity_high_operation_groups'], indent=2, ensure_ascii=False)[:8000], '```', '', f"EWoK relation-group total: `{ex['ewok_relation_total']}`", '', 'GlobalPIQA groups:', '', '```json', json.dumps(ex['globalpiqa_groups'], indent=2, ensure_ascii=False)[:4000], '```']
    lines += ['', '## Interpretation for next comparator', '', 'If coherent SuperGLUE keeps the fast-path endpoint alive, ordinary scale1.75 chck_86M must be evaluated and compared at the same item level. The mechanistic question is whether coherent adds high-operation Entity / multi-step tracking beyond ordinary extra exposure without losing EWoK, GlobalPIQA, COMPS, or Supplement items. research alone shows coherent beats the destructive spanbreak arm strongly, but coherent versus protected chck82 has a negative total item balance and no EWoK repair.', '', f"JSON: `{rel(OUT_JSON)}`"]
    OUT_MD.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(json.dumps({'status': out['status'], 'out_json': rel(OUT_JSON), 'out_md': rel(OUT_MD)}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
