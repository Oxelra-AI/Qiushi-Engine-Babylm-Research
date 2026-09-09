#!/usr/bin/env python3
"""research: use already-trained compact_repeat_core as the missing contrast.

This script compares clean-Qwen, compact_repeat_core, and compact_view_core on
full Supplement and any already-finished compact_repeat_core SuperGLUE subtasks.
It performs CPU parsing only and can be rerun after the managed SuperGLUE resume
finishes.
"""
from __future__ import annotations

import json
import pathlib
import re
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Tuple

ROOT = pathlib.Path('.')
EVAL = ROOT / 'experiments/archive/initial_model_studies/repos/babylm-eval/strict/evaluation_data/full_eval'
OUT_DIR = ROOT / 'experiments/archive/frontier_consolidation/data/repeat_core_full_surface_contrast'
NOTE = ROOT / 'research/notes/frontier_consolidation/repeat_core_full_surface_contrast.md'

MODELS = {
    'clean_qwen': {
        'label': 'inherited clean-Qwen endpoint',
        'per_target': ROOT / 'experiments/archive/compact_experience/data/full_eval/per_target/qwen_clean_aligned.json',
    },
    'compact_repeat_core': {
        'label': 'FineWeb compact-core source-repetition control',
        'per_target': ROOT / 'experiments/archive/frontier_consolidation/data/density_full_eval/per_target/compact_repeat_core.json',
    },
    'compact_view_core': {
        'label': 'FineWeb compact-core compact semantic-view model',
        'per_target': ROOT / 'experiments/archive/frontier_consolidation/data/density_full_eval/per_target/compact_view_core.json',
    },
}
SUPPLEMENT_FILES = ['qa_congruence_easy', 'qa_congruence_tricky', 'turn_taking', 'subject_aux_inversion', 'hypernym']
SUPERGLUE_TASKS = ['boolq', 'multirc', 'rte', 'wsc', 'mrpc', 'qqp', 'mnli']


def read_json(path: pathlib.Path) -> Any:
    with path.open('r', encoding='utf-8') as f:
        return json.load(f)


def read_jsonl(path: pathlib.Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def norm(x: Any) -> str:
    return re.sub(r'\s+', ' ', str(x).strip())


def short(x: Any, n: int = 220) -> str:
    s = norm(x)
    return s if len(s) <= n else s[: n - 1] + '…'


def pct(num: float, den: float) -> Optional[float]:
    if den == 0:
        return None
    return round(100.0 * num / den, 6)


def parse_uid_report(path: pathlib.Path) -> Tuple[Dict[str, float], Optional[float]]:
    uid_scores: Dict[str, float] = {}
    average: Optional[float] = None
    if not path.exists():
        return uid_scores, average
    lines = path.read_text(encoding='utf-8', errors='replace').splitlines()
    in_uid = False
    for i, raw in enumerate(lines):
        line = raw.strip()
        if line.startswith('### UID ACCURACY'):
            in_uid = True
            continue
        if in_uid and line.startswith('###'):
            in_uid = False
        if in_uid and ':' in line:
            k, v = line.split(':', 1)
            try:
                uid_scores[k.strip()] = float(v.strip())
            except ValueError:
                pass
        if line.startswith('### AVERAGE ACCURACY') and i + 1 < len(lines):
            try:
                average = float(lines[i + 1].strip())
            except ValueError:
                average = None
    return uid_scores, average


def supplement_bundle(model_key: str) -> Dict[str, Any]:
    pt = read_json(MODELS[model_key]['per_target'])
    task = pt.get('tasks', {}).get('Supplement')
    if not isinstance(task, dict) or 'predictions' not in task:
        raise RuntimeError(f'{model_key} has no full Supplement predictions')
    preds_raw = read_json(pathlib.Path(task['predictions']))
    flat: Dict[str, str] = {}
    for uid in SUPPLEMENT_FILES:
        for rec in preds_raw[uid]['predictions']:
            flat[rec['id']] = norm(rec['pred'])
    uid_scores, avg = parse_uid_report(pathlib.Path(task['report']))
    return {'preds': flat, 'uid_scores': uid_scores, 'average_score': avg, 'task_record': task}


def load_supplement_gold() -> Dict[str, Dict[str, Any]]:
    gold: Dict[str, Dict[str, Any]] = {}
    for uid in SUPPLEMENT_FILES:
        for i, row in enumerate(read_jsonl(EVAL / f'supplement_filtered/{uid}.jsonl')):
            rid = f'{uid}_{i}'
            gold[rid] = {**row, 'uid': uid, 'id': rid}
    return gold


def summarize_pair(rows: Iterable[Dict[str, Any]], a: str, b: str) -> Dict[str, Any]:
    rows = list(rows)
    n = len(rows)
    a_correct = sum(int(r[f'{a}_correct']) for r in rows)
    b_correct = sum(int(r[f'{b}_correct']) for r in rows)
    lost = [r for r in rows if r[f'{a}_correct'] and not r[f'{b}_correct']]
    gained = [r for r in rows if (not r[f'{a}_correct']) and r[f'{b}_correct']]
    return {
        'n': n,
        f'{a}_correct': a_correct,
        f'{b}_correct': b_correct,
        f'{a}_acc': pct(a_correct, n),
        f'{b}_acc': pct(b_correct, n),
        'delta_acc_b_minus_a': round((pct(b_correct, n) or 0.0) - (pct(a_correct, n) or 0.0), 6),
        'lost_from_a_to_b': len(lost),
        'gained_from_a_to_b': len(gained),
        'lost_examples': [{k: r[k] for k in ['id', 'uid', 'contrast', 'fragment', 'template', 'ambiguous', 'good', 'bad'] if k in r} for r in lost[:12]],
        'gained_examples': [{k: r[k] for k in ['id', 'uid', 'contrast', 'fragment', 'template', 'ambiguous', 'good', 'bad'] if k in r} for r in gained[:12]],
    }


def analyze_supplement() -> Dict[str, Any]:
    gold = load_supplement_gold()
    bundles = {m: supplement_bundle(m) for m in MODELS}
    rows: List[Dict[str, Any]] = []
    by_uid: Dict[str, Dict[str, Any]] = {}
    by_field = defaultdict(lambda: {'n': 0, 'clean_qwen_correct': 0, 'compact_repeat_core_correct': 0, 'compact_view_core_correct': 0})
    mismatches: List[Dict[str, Any]] = []

    for rid, row in gold.items():
        good = norm(row['sentence_good'])
        bad = norm(row['sentence_bad'])
        rec: Dict[str, Any] = {
            'id': rid,
            'uid': row['uid'],
            'row': row.get('row'),
            'contrast': row.get('contrast'),
            'fragment': row.get('fragment'),
            'template': row.get('template'),
            'ambiguous': row.get('ambiguous'),
            'good': short(row['sentence_good']),
            'bad': short(row['sentence_bad']),
        }
        for m, bundle in bundles.items():
            pred = norm(bundle['preds'][rid])
            rec[f'{m}_pred'] = short(pred)
            rec[f'{m}_correct'] = pred == good
            if pred not in (good, bad) and len(mismatches) < 20:
                mismatches.append({'id': rid, 'model': m, 'pred': short(pred), 'good': short(good), 'bad': short(bad)})
        rows.append(rec)
        for field in ['contrast', 'fragment', 'template', 'ambiguous']:
            if field in row:
                key = f'{row["uid"]}|{field}={row[field]}'
                s = by_field[key]
                s['n'] += 1
                for m in MODELS:
                    s[f'{m}_correct'] += int(rec[f'{m}_correct'])

    for uid in SUPPLEMENT_FILES:
        subset = [r for r in rows if r['uid'] == uid]
        n = len(subset)
        entry: Dict[str, Any] = {'n': n}
        for m, bundle in bundles.items():
            c = sum(int(r[f'{m}_correct']) for r in subset)
            entry[f'{m}_correct'] = c
            entry[f'{m}_acc'] = pct(c, n)
            entry[f'{m}_report_uid_score'] = bundle['uid_scores'].get(uid)
        entry['repeat_minus_clean'] = round(entry['compact_repeat_core_acc'] - entry['clean_qwen_acc'], 6)
        entry['view_minus_repeat'] = round(entry['compact_view_core_acc'] - entry['compact_repeat_core_acc'], 6)
        entry['view_minus_clean'] = round(entry['compact_view_core_acc'] - entry['clean_qwen_acc'], 6)
        entry['clean_to_repeat'] = summarize_pair(subset, 'clean_qwen', 'compact_repeat_core')
        entry['repeat_to_view'] = summarize_pair(subset, 'compact_repeat_core', 'compact_view_core')
        by_uid[uid] = entry

    overall: Dict[str, Any] = {'n': len(rows)}
    for m, bundle in bundles.items():
        c = sum(int(r[f'{m}_correct']) for r in rows)
        overall[f'{m}_correct'] = c
        overall[f'{m}_all_item_acc'] = pct(c, len(rows))
        overall[f'{m}_official_equal_uid_average'] = bundle['average_score']
    overall['repeat_minus_clean_equal_uid'] = round(overall['compact_repeat_core_official_equal_uid_average'] - overall['clean_qwen_official_equal_uid_average'], 6)
    overall['view_minus_repeat_equal_uid'] = round(overall['compact_view_core_official_equal_uid_average'] - overall['compact_repeat_core_official_equal_uid_average'], 6)
    overall['view_minus_clean_equal_uid'] = round(overall['compact_view_core_official_equal_uid_average'] - overall['clean_qwen_official_equal_uid_average'], 6)

    field_rows: List[Dict[str, Any]] = []
    for key, s in by_field.items():
        rec = {'field': key, **s}
        for m in MODELS:
            rec[f'{m}_acc'] = pct(s[f'{m}_correct'], s['n'])
        rec['repeat_minus_clean'] = round(rec['compact_repeat_core_acc'] - rec['clean_qwen_acc'], 6)
        rec['view_minus_repeat'] = round(rec['compact_view_core_acc'] - rec['compact_repeat_core_acc'], 6)
        rec['view_minus_clean'] = round(rec['compact_view_core_acc'] - rec['clean_qwen_acc'], 6)
        field_rows.append(rec)
    field_rows.sort(key=lambda r: (r['view_minus_clean'], r['repeat_minus_clean'], -r['n']))

    fast_summary = read_json(ROOT / 'experiments/archive/frontier_consolidation/data/density_noaoa_eval_compact_core/density_noaoa_eval_summary.json')
    fast = fast_summary['table']
    fast_vs_full = {
        'compact_repeat_core_fast_supplement': fast['compact_repeat_core']['Supplement'],
        'compact_repeat_core_full_supplement': overall['compact_repeat_core_official_equal_uid_average'],
        'compact_view_core_fast_supplement': fast['compact_view_core']['Supplement'],
        'compact_view_core_full_supplement': overall['compact_view_core_official_equal_uid_average'],
        'fast_view_minus_repeat': fast['compact_view_core']['Supplement'] - fast['compact_repeat_core']['Supplement'],
        'full_view_minus_repeat': overall['view_minus_repeat_equal_uid'],
    }

    return {
        'overall': overall,
        'by_uid': by_uid,
        'worst_fields_by_view_minus_clean': field_rows[:25],
        'best_fields_by_view_minus_clean': field_rows[-25:][::-1],
        'clean_to_repeat_overall': summarize_pair(rows, 'clean_qwen', 'compact_repeat_core'),
        'repeat_to_view_overall': summarize_pair(rows, 'compact_repeat_core', 'compact_view_core'),
        'clean_to_view_overall': summarize_pair(rows, 'clean_qwen', 'compact_view_core'),
        'fast_vs_full': fast_vs_full,
        'prediction_candidate_mismatches_examples': mismatches,
    }


def coerce_label(x: Any) -> Any:
    if isinstance(x, bool):
        return int(x)
    if isinstance(x, (int, float)) and not isinstance(x, bool):
        return int(x)
    if isinstance(x, str):
        y = x.strip()
        if y.isdigit() or (y.startswith('-') and y[1:].isdigit()):
            return int(y)
        return {'false': 0, 'true': 1, 'entailment': 0, 'not_entailment': 1, 'neutral': 1, 'contradiction': 2}.get(y.lower(), y)
    return x


def superglue_task_records(model_key: str) -> Dict[str, Dict[str, Any]]:
    pt = read_json(MODELS[model_key]['per_target'])
    sg = pt.get('tasks', {}).get('SuperGLUE')
    out: Dict[str, Dict[str, Any]] = {}
    if not isinstance(sg, dict):
        return out
    for rec in sg.get('tasks', []) or []:
        if isinstance(rec, dict) and rec.get('returncode') == 0 and rec.get('accuracy') is not None:
            out[rec['task']] = rec
    return out


def load_super_preds(rec: Dict[str, Any], task: str) -> List[Any]:
    data = read_json(pathlib.Path(rec['predictions']))
    block = data.get(task, data)
    preds = block['predictions'] if isinstance(block, dict) and 'predictions' in block else block
    return [coerce_label(p['pred'] if isinstance(p, dict) and 'pred' in p else p) for p in preds]


def analyze_superglue_available() -> Dict[str, Any]:
    recs = {m: superglue_task_records(m) for m in MODELS}
    available_repeat = sorted(recs['compact_repeat_core'].keys(), key=SUPERGLUE_TASKS.index)
    task_rows: Dict[str, Any] = {}
    for task in available_repeat:
        if task not in recs['clean_qwen'] or task not in recs['compact_view_core']:
            continue
        gold_rows = read_jsonl(EVAL / f'glue_filtered/{task}.valid.jsonl')
        labels = [coerce_label(r['label']) for r in gold_rows]
        preds = {m: load_super_preds(recs[m][task], task) for m in MODELS}
        n = min([len(labels)] + [len(preds[m]) for m in MODELS])
        examples: List[Dict[str, Any]] = []
        for i in range(n):
            ex = {'i': i, 'gold': labels[i]}
            for m in MODELS:
                ex[f'{m}_pred'] = preds[m][i]
                ex[f'{m}_correct'] = preds[m][i] == labels[i]
            examples.append(ex)
        entry = {'n': n}
        for m in MODELS:
            c = sum(int(e[f'{m}_correct']) for e in examples)
            entry[f'{m}_correct'] = c
            entry[f'{m}_acc'] = pct(c, n)
            entry[f'{m}_record_accuracy'] = recs[m][task]['accuracy']
            entry[f'{m}_pred_counts'] = recs[m][task].get('pred_counts')
        entry['repeat_minus_clean'] = round(entry['compact_repeat_core_acc'] - entry['clean_qwen_acc'], 6)
        entry['view_minus_repeat'] = round(entry['compact_view_core_acc'] - entry['compact_repeat_core_acc'], 6)
        entry['view_minus_clean'] = round(entry['compact_view_core_acc'] - entry['clean_qwen_acc'], 6)
        entry['clean_to_repeat'] = summarize_pair(examples, 'clean_qwen', 'compact_repeat_core')
        entry['repeat_to_view'] = summarize_pair(examples, 'compact_repeat_core', 'compact_view_core')
        task_rows[task] = entry
    return {
        'available_repeat_tasks': available_repeat,
        'missing_repeat_tasks': [t for t in SUPERGLUE_TASKS if t not in available_repeat],
        'tasks': task_rows,
        'clean_full_tasks': {k: v.get('accuracy') for k, v in recs['clean_qwen'].items()},
        'compact_view_core_full_tasks': {k: v.get('accuracy') for k, v in recs['compact_view_core'].items()},
        'compact_repeat_core_tasks_present': {k: v.get('accuracy') for k, v in recs['compact_repeat_core'].items()},
    }


def make_note(result: Dict[str, Any]) -> str:
    supp = result['supplement']
    sg = result['superglue_available']
    ov = supp['overall']
    lines: List[str] = []
    lines.append('# research compact-repeat-core full-surface contrast')
    lines.append('')
    lines.append('Purpose: use the already-trained compact_repeat_core model as the missing contrast between the FineWeb/repetition substrate and compact semantic transformation. The script only parses existing prediction files; the only new GPU work was the separate official-compatible Supplement+SuperGLUE run for compact_repeat_core, whose Supplement and BoolQ outputs are already present and whose remaining SuperGLUE subtasks are pending.')
    lines.append('')
    lines.append('## Full Supplement triad')
    lines.append(f"- Official equal-UID Supplement: clean-Qwen {ov['clean_qwen_official_equal_uid_average']:.2f}, compact_repeat_core {ov['compact_repeat_core_official_equal_uid_average']:.2f}, compact_view_core {ov['compact_view_core_official_equal_uid_average']:.2f}.")
    lines.append(f"- Decomposition on this column: repeat minus clean {ov['repeat_minus_clean_equal_uid']:+.2f}; view minus repeat {ov['view_minus_repeat_equal_uid']:+.2f}; view minus clean {ov['view_minus_clean_equal_uid']:+.2f}.")
    lines.append(f"- Fast-screen Supplement overstated the compact-view advantage: fast view-repeat {supp['fast_vs_full']['fast_view_minus_repeat']:+.2f}, full view-repeat {supp['fast_vs_full']['full_view_minus_repeat']:+.2f}.")
    lines.append('')
    lines.append('| UID | clean | repeat | view | repeat-clean | view-repeat | view-clean |')
    lines.append('|---|---:|---:|---:|---:|---:|---:|')
    for uid, s in supp['by_uid'].items():
        lines.append(f"| {uid} | {s['clean_qwen_acc']:.2f} | {s['compact_repeat_core_acc']:.2f} | {s['compact_view_core_acc']:.2f} | {s['repeat_minus_clean']:+.2f} | {s['view_minus_repeat']:+.2f} | {s['view_minus_clean']:+.2f} |")
    lines.append('')
    lines.append('Supplement reading: the large full Supplement shortfall is not introduced only by compact semantic views. The repetition substrate is already far below clean-Qwen, while compact views recover part of that loss rather than worsen it on the column aggregate. However compact views still remain below clean-Qwen, and the fast mini-set was too favorable to the view model, so Supplement alone supports neither immediate abandonment of compact views nor immediate Qwen-internal compaction.')
    lines.append('')
    lines.append('## SuperGLUE currently available for compact_repeat_core')
    lines.append(f"- Available compact_repeat_core SuperGLUE tasks: {', '.join(sg['available_repeat_tasks']) if sg['available_repeat_tasks'] else '(none)'}.")
    lines.append(f"- Missing compact_repeat_core SuperGLUE tasks: {', '.join(sg['missing_repeat_tasks']) if sg['missing_repeat_tasks'] else '(none)'}.")
    if sg['tasks']:
        lines.append('')
        lines.append('| task | clean | repeat | view | repeat-clean | view-repeat | view-clean |')
        lines.append('|---|---:|---:|---:|---:|---:|---:|')
        for task, s in sg['tasks'].items():
            lines.append(f"| {task} | {s['clean_qwen_acc']:.3f} | {s['compact_repeat_core_acc']:.3f} | {s['compact_view_core_acc']:.3f} | {s['repeat_minus_clean']:+.3f} | {s['view_minus_repeat']:+.3f} | {s['view_minus_clean']:+.3f} |")
    lines.append('')
    lines.append('SuperGLUE reading so far: only finished compact_repeat_core subtasks should be interpreted. The decisive RTE and MRPC comparisons are not present until the managed resume finishes; do not infer them from compact_view_core alone.')
    lines.append('')
    lines.append('## Consequence for the next route')
    lines.append('- The new Supplement contrast corrects the research explanation: official-source displacement remains possible, but the observed QA/RTE/MRPC structure is also compatible with role/modality/relation losses in compact views, and Supplement now shows a mixed pattern where the FineWeb repetition substrate loses more than the compact view model.')
    lines.append('- Qwen-internal density should remain conditional. If full compact_repeat_core SuperGLUE already has the RTE/MRPC weakness, then the FineWeb/repetition substrate and displaced official mixture are implicated; if repeat preserves RTE/MRPC while view loses them, compact semantic transformation is implicated and internal compaction could transfer the same weakness into the strongest inherited Qwen block.')
    lines.append('- The pending AoA ladder still matters because the scalar AoA discontinuity alone is not a mechanism; compare stable word-level trajectories across repeat/view/reinvest after the AoA results become available.')
    lines.append('')
    lines.append(f"Machine-readable JSON: `{OUT_DIR / 'repeat_core_full_surface_contrast.json'}`")
    return '\n'.join(lines) + '\n'


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    NOTE.parent.mkdir(parents=True, exist_ok=True)
    result = {
        'status': 'REPEAT_CORE_FULL_SURFACE_CONTRAST',
        'method': 'CPU parse of existing full Supplement and available SuperGLUE predictions for clean_qwen, compact_repeat_core, and compact_view_core.',
        'inputs': {m: str(md['per_target']) for m, md in MODELS.items()},
        'supplement': analyze_supplement(),
        'superglue_available': analyze_superglue_available(),
    }
    out_json = OUT_DIR / 'repeat_core_full_surface_contrast.json'
    out_json.write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    NOTE.write_text(make_note(result), encoding='utf-8')
    print(json.dumps({'status': result['status'], 'json': str(out_json), 'note': str(NOTE)}, indent=2))


if __name__ == '__main__':
    main()
